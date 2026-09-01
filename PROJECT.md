# Recoverly — Receivables Leverage Agent

**Razorpay AI Buildathon 2026 · Track 03: AI Revenue Recovery**

Source problem: *"Why can't SMEs negotiate favorable payment terms with large buyers?"* —
Fix My Itch score 82.8, the highest-scored B2B Services problem on Razorpay's board.

---

## One-line pitch

Large buyers pay SMEs late because they can, and SMEs have no leverage and no time to
chase. Recoverly runs a bounded, autonomous collections loop over every open invoice, and
turns what that loop learns into a **negotiating position** the SME can take into a room:
"Buyer X honored Net 60 twice and broke it four times — here is what we propose now, and
here is the evidence."

The collections loop is not the product. It is the **evidence engine**. The product is the
leverage.

---

## What it does (features)

### 1. Watches and scores every open invoice
- Slip probability `P(pays beyond terms)` and expected days late, per invoice.
- Logistic regression (`sklearn`), trained on the first 4 months, held out on the last
  6 weeks. Precision / recall / AUC reported.
- Because the data is synthetic, the model partly recovers its own generator — this is
  stated on camera. The claim is the pipeline, not the AUC.
- Output is a segmentation: `on_track` / `at_risk` / `slipping` / `overdue` / `chronic`.

### 2. Acts autonomously, within a deterministic ladder
- A day-stepping loop evaluates every open invoice, every simulated day.
- **Days-beyond-terms (DBT) picks the ladder rung** — a pure function of DBT and invoice
  state. The LLM does not choose the action.
- The LLM runs **only on a stage transition**, to diagnose *why* the invoice is slipping
  and to draft the message. It never invents an action.

| Stage | Trigger | Action |
|---|---|---|
| 0 | 5 days before due | Pre-due courtesy reminder + payment link |
| 1 | Due date passed | Polite follow-up, restate terms |
| 2 | DBT 7 | Firm reminder, escalate to AP contact |
| 3 | DBT 15 | Early-settlement offer, discount ≤ 2% |
| 4 | DBT 30 | Payment plan / partial settlement |
| 5 | DBT 45, or dispute, or opt-out | **Stop** — hand to human with a full brief |

### 3. Diagnoses the cause (local LLM)
For each at-risk invoice on a stage transition, a **local model (Ollama, `llama3.1:8b`)**
reasons over the buyer profile, invoice context, prior touches, and free-text notes to
produce one of:
- Cash-flow issue on the buyer's side
- Process issue (invoice never entered their AP system)
- Genuine dispute
- Deliberate stretching

The diagnosis is recorded on the audit row and shapes the drafted message.

**No external API.** The model runs on the machine via Ollama — no API key, no network
egress, no per-call cost. The Python side talks to `http://localhost:11434`. Called with
`temperature: 0` and a fixed `seed`, so output is reproducible across runs. If Ollama is
not running, `diagnose()` falls back to a rule-based classifier over the buyer profile and
`draft()` falls back to a static per-stage template — the pipeline never blocks on the model.

### 4. Refuses, and escalates
**Five hard bounds**, defined once as a `BOUNDS` predicate list consumed by the engine,
the tests, and the audit rows:

| Name | Rule |
|---|---|
| `max_touches` | ≤ 6 touches per invoice, ever |
| `min_gap_72h` | ≥ 72 hours between touches |
| `business_hours` | Weekdays only; every action stamped 09:00–19:00 IST (deterministic, seeded from `invoice_id`) |
| `discount_cap_2pc` | Discount authority ≤ 2%; anything beyond → human |
| `maker_checker` | No autonomous action on invoices above an amount threshold without sign-off |

**Four instant stop conditions:**
- Dispute raised → **terminal**: fires the stage-5 escalation, writes the human-handoff audit row
- Buyer opt-out → **terminal**: same
- Promise-to-pay recorded → **silent hold**: agent goes quiet until the promised date
- Payment received → closes the invoice

Ask it for a 5% discount and it refuses, escalates, and sets `human_gate_required = true`.

### 5. Learns that promises are cheap
`promise_kept_rate` is read by the engine, not just printed.
- Buyer promises → agent goes silent until the promised date.
- Promise kept → invoice closes.
- Promise broken → escalate, using the break as leverage.
- **Second broken promise (buyer-level)** → the agent stops honoring new promises from that
  buyer. New promises are still recorded in the audit trail but no longer suppress touches;
  the invoice proceeds to its DBT-appropriate rung.

This is the feedback loop the shipped Gnani.ai × Razorpay collections platform does not
publicly claim.

### 6. Moves real money — once
- **Batch run:** a simulated link adapter. Zero network calls.
- **Demo invoice:** one real Razorpay **test-mode** payment link, created live on camera,
  exact amount, visible on screen.
- Adapter carries a `live` flag so the same code path serves both.

### 7. Leaves an audit trail
One append-only row per agent action (append-only *by convention* + a run hash and
generation timestamp — a DuckDB table can be overwritten, so the claim is **traceability**,
not tamper resistance):

```
timestamp | invoice_id | buyer_id | risk_score | diagnosis | ladder_stage |
action_taken | message_sent | razorpay_object_id | bounds_checked[] |
human_gate_required | outcome | outcome_timestamp
```

Pick any row and trace exactly why it happened.

### 8. The leverage brief — the differentiator
For a featured buyer (sample size n ≥ 20, enforced), one page:
- Honored vs broken terms history (`terms_honored_count` / `terms_broken_count` — one count
  per invoice; honored = cleared on or before due date; partials count as broken until
  settled)
- Mean DBT and variance
- `promise_kept_rate` (**NULL**, not 0.0, for a buyer who never promised)
- Dispute rate
- **Recommended terms** — deterministic rule over honored-rate and mean DBT
- **The justification prose** — LLM-drafted, with the evidence line attached
- **The message to send**

Same split as the ladder: deterministic recommendation, generative language.

### 9. Dashboard (React SPA + read-only API)
- **`api.py`** — FastAPI, ~40 lines. Read-only `GET` endpoints over `results.duckdb`:
  `/invoices`, `/metrics`, `/buyers/{id}/brief`, `/audit`, `/exceptions`. No writes, no
  auth, no webhook. `uvicorn api:app`.
- **`web/`** — Vite + React SPA, three views: **Queue** (risk heatmap, invoice list),
  **Results** (treatment vs control table, exception list), **Leverage** (the brief +
  recommended terms + message). Fetches from `api.py`.
- The pipeline writes `results.duckdb`; the API only reads it; the SPA only renders. Two
  local processes on the demo path, both read-only. Nothing recomputes on request.

### 10. Reports honestly (treatment vs control)
- ~300 invoices split 70% treatment / 30% control (holdout, no agent contact), matched on
  buyer profile and invoice size.
- Reported: ₹ recovered, % of at-risk value recovered, DSO reduction (days), cash pulled
  forward (₹-days), discount cost, **net benefit** = cash-acceleration value − discount cost,
  promise-kept rate.
- **The response model is shown on screen and named as an assumption.** The generator
  decides who pays when; a treatment effect only exists because an authored rule
  (`RESPONSE_LIFT`, benchmark-anchored) maps agent action → earlier payment with some
  probability. The control group proves the **measurement machinery** is correct — it does
  not prove the agent works.
- **Exception list:** every invoice the agent could not resolve, honestly labelled.

---

## Architecture

### System overview

```
                         ┌───────────────────────────────────────────────┐
                         │                 ledger.py                      │
                         │  ~300 invoices · 20 buyers (8 heavy / 12 light)│
                         │  6 months · SEED=42 · generated ONCE → parquet │
                         │                                               │
                         │  ┌─────────────┐  ┌──────────────────────────┐ │
                         │  │ buyer        │  │ dated event stream        │ │
                         │  │ profiles     │  │ promises · disputes ·     │ │
                         │  │ (dbt dist,   │  │ opt-outs · payments       │ │
                         │  │ seasonality, │  │ (discovered on their date,│ │
                         │  │ responsive-  │  │  NOT prebuilt — no future  │ │
                         │  │ ness)        │  │  leakage)                 │ │
                         │  └─────────────┘  └──────────────────────────┘ │
                         │  ┌──────────────────────────────────────────┐  │
                         │  │ RESPONSE_LIFT  (STATED ASSUMPTION)        │  │
                         │  │ {stage: (days_earlier, probability)}      │  │
                         │  │ benchmark-anchored · treatment ONLY      │  │
                         │  └──────────────────────────────────────────┘  │
                         └───────────────────────┬───────────────────────┘
                                                 │ committed parquet
                                                 ▼
     ┌───────────────────────────────────────────────────────────────────────────┐
     │                              engine.py                                     │
     │                    the simulated clock — the agent                         │
     │                                                                           │
     │   for day in date_range(start, end):                                       │
     │       apply_events(day)          ← promises/disputes/opt-outs/payments      │
     │       for inv in open_invoices(day):                                       │
     │           stop = stop_condition(inv, day)                                  │
     │           if stop.terminal:      → escalate_to_human() → AUDIT (stage 5)    │
     │           if stop.silent_hold:   → skip (active promise window)             │
     │           stage = ladder_stage(inv, day)   ← pure fn of DBT + state         │
     │           if stage > last_stage(inv) and bounds_ok(inv, day):              │
     │               act(inv, stage, day) → AUDIT                                  │
     │       settle_payments(day)       ← buyer profile + RESPONSE_LIFT decide     │
     │                                                                           │
     │   ┌──────────────┐   ┌────────────────────────┐   ┌────────────────────┐   │
     │   │ BOUNDS list  │   │ promise feedback loop  │   │ business-hour       │   │
     │   │ (5 named     │   │ 2nd broken promise →   │   │ timestamp seeded    │   │
     │   │  predicates) │   │ stop honoring promises │   │ from invoice_id     │   │
     │   └──────────────┘   └────────────────────────┘   └────────────────────┘   │
     └───────┬───────────────────────┬───────────────────────────┬───────────────┘
             │ on stage transition   │ every action              │ Stage 0 + Stage 3
             ▼                       ▼                           ▼
   ┌──────────────────┐   ┌────────────────────┐      ┌──────────────────────────┐
   │     agent.py      │   │   audit (in        │      │       razorpay.py        │
   │                  │   │   results.duckdb)  │      │                          │
   │ score()          │   │                    │      │ create_link(inv, live=?) │
   │  logreg · 4mo    │   │ append-only by     │      │  live=False → simulated  │
   │  train / 6wk     │   │ convention +       │      │  live=True  → ONE real    │
   │  holdout · AUC   │   │ run hash           │      │  test-mode call (demo)   │
   │                  │   │                    │      │                          │
   │ diagnose()       │   │ 1 row / action     │      │ cached fallback on       │
   │  local LLM       │   │                    │      │ network failure          │
   │  (Ollama         │   └─────────┬──────────┘      └──────────────────────────┘
   │  llama3.1:8b) →  │             │
   │  1 of 4 labels   │             │
   │  temp 0 · cached │             │
   └────────┬─────────┘             │
            │                       ▼
            │            ┌────────────────────────┐
            │            │     results.duckdb      │
            │            │  audit rows · metrics · │
            │            │  per-buyer rollups ·    │
            │            │  run hash + timestamp   │
            │            └───────────┬────────────┘
            │                        │ READ-ONLY
            ▼                        ▼
   ┌──────────────────┐   ┌────────────────────────────┐
   │     brief.py      │   │        api.py (FastAPI)     │
   │                  │   │   GET /invoices  /metrics   │
   │ per-buyer        │   │   /buyers/{id}/brief        │
   │ leverage brief   │   │   /audit  /exceptions       │
   │  honored/broken  │   │   read-only · no auth ·     │
   │  promise rate    │   │   no webhook · uvicorn      │
   │  recommended     │   └─────────────┬──────────────┘
   │  terms (det.)    │                 │ HTTP (localhost)
   │  justification   │                 ▼
   │  (LLM) · message │   ┌────────────────────────────────────────┐
   └──────────────────┘   │        web/  (Vite + React SPA)         │
                          │                                        │
                          │   Queue    │ Results     │ Leverage     │
                          │   risk     │ treatment   │ the brief +  │
                          │   heatmap  │ vs control  │ recommended  │
                          │   invoice  │ + exception │ terms +      │
                          │   list     │ list        │ message      │
                          │                                        │
                          │   ~4 components · no Redux · no router  │
                          └────────────────────────────────────────┘
```

### Modules (6 Python + a React app, plus tests)

| File | Responsibility |
|---|---|
| `ledger.py` | Synthetic generator · buyer behavioral profiles · dated event stream · `RESPONSE_LIFT` · seeded, generated once, committed as parquet |
| `engine.py` | Day loop · event application · deterministic ladder · `BOUNDS` · split stop conditions · promise feedback loop · audit writes |
| `brief.py` | Per-buyer leverage brief — the differentiator; built third, before the model and the LLM |
| `agent.py` | Logistic risk `score()` · local-LLM `diagnose()` and message drafting via Ollama (`llama3.1:8b`, `temp 0`, cached to disk, invoked only on stage transitions, rule-based fallback if Ollama is down) |
| `razorpay.py` | Link adapter with a `live` flag — simulated batch, one real test-mode call, cached fallback |
| `api.py` | FastAPI, ~40 lines — read-only `GET` endpoints over `results.duckdb`. No writes, no auth, no webhook. |
| `web/` | Vite + React SPA — three views (Queue / Results / Leverage), fetches from `api.py`. ~4 components, no Redux, no router. |
| `test_engine.py` | ~15 pytest tests — the three silent-failure gaps, every `BOUNDS` predicate via `parametrize`, the four demo beats as smoke tests |

### Key design rules

- **Deterministic ladder, generative language.** DBT picks the rung; the LLM diagnoses and
  writes.
- **The clock is the only stateful piece.** Everything else is a pure function called from
  the loop. This is what makes it an agent, not a batch report.
- **Events are discovered, not prebuilt.** A `promise` row with `kept` / `resolved_date`
  visible to the engine would hand the agent the future.
- **`BOUNDS` defined once.** Engine gates on it, tests parametrize over it, audit rows
  record which passed. Change a limit, change all three.
- **No external API.** The LLM is local (Ollama). The only network call in the whole
  project is the single Razorpay test-mode payment link for the demo invoice. Everything
  else runs offline.
- **Determinism contract.** `SEED = 42`, ledger committed as parquet, local LLM at
  `temperature 0` with a fixed seed, responses cached.
  Numbers said on camera stay true across re-recordings.
- **The response model is an assumption, not a measurement.** Named, benchmark-anchored,
  shown on screen, applied to treatment only.

---

## Workflow

### Build order (weekend budget)

Steps 1–4 alone are a submittable demo. Everything after is upside.

| # | Step | Notes |
|---|---|---|
| 1 | `ledger.py` — generator, profiles, event stream, `RESPONSE_LIFT` | **Time-boxed to end of day 1.** If 20 buyers aren't convincingly messy, drop to 10. Everything depends on this. |
| 2 | `engine.py` + `test_engine.py` | Day loop, ladder, `BOUNDS`, stop conditions, promise loop, audit. No LLM yet. Assert every `BOUNDS` predicate holds — that assertion block is the on-camera autonomy proof. Tests written alongside, not after. |
| 3 | `brief.py` | The differentiator, built before the model and the LLM so it can't be the thing cut on Sunday night. |
| 4 | `razorpay.py` | One live test-mode link end to end. |
| 5 | `agent.py` — `score()` | Logistic regression, 6-week holdout. Assert train dates precede test dates. |
| 6 | `agent.py` — `diagnose()` | Local LLM via Ollama (`ollama pull llama3.1:8b`), `temp 0`, diagnosis + drafting, cached to disk. Rule-based fallback path first so the pipeline runs even before Ollama is set up. |
| 7 | Control / treatment split + metrics | Boolean column + `GROUP BY`. `RESPONSE_LIFT` treatment-only. |
| 8 | `api.py` + `web/` | FastAPI read-only endpoints over `results.duckdb`, then a Vite + React SPA with three views. Scaffold with `npm create vite@latest`, ~4 components, no state library, no router (three tabs = local state). `uvicorn api:app` + `vite dev` on the demo path. |
| 9 | Failure scenarios | The three bounds demos, scripted and repeatable. |
| 10 | Video | Reserve the final half-day. Steps 5–9 are the cut line. |

### Runtime flow (one simulated day)

```
1. apply_events(day)
   └─ any promise / dispute / opt-out / payment dated today is applied now

2. for each open invoice:
   ├─ terminal stop (dispute / opt-out)?
   │    └─ escalate_to_human() → audit row (stage 5) → done with this invoice
   ├─ silent hold (active promise window)?
   │    └─ skip
   ├─ compute ladder_stage from DBT + state
   └─ stage advanced AND bounds_ok?
        ├─ on stage transition: agent.diagnose() (local LLM via Ollama, temp 0, cached) → draft message
        ├─ stage 0 or 3: razorpay.create_link(live=False for batch)
        ├─ act() → append audit row (business-hour timestamp seeded from invoice_id)
        └─ if 2nd broken promise for this buyer: stop honoring future promises

3. settle_payments(day)
   └─ buyer profile + RESPONSE_LIFT (treatment only) decide who pays today
```

### Demo flow (5-minute video)

| Time | Segment |
|---|---|
| 0:00–0:35 | The problem, concretely. One SME, ₹42L trapped, 6 buyers, avg 34 DBT. Name the Fix My Itch score (82.8). |
| 0:35–1:15 | Architecture — one diagram, 40 seconds. Deterministic ladder, generative language, hard bounds. |
| 1:15–2:30 | Live single-invoice trace: risk score rises → diagnosis → rung chosen → **real Razorpay test-mode link on screen** → drafted message → payment lands → audit row. |
| 2:30–3:15 | Failure & bounds demo: (1) dispute → agent stops, escalation audit row appears; (2) promise-to-pay → agent goes quiet, resumes after; (3) 5% discount ask → agent refuses, escalates. |
| 3:15–4:30 | Batch results: treatment vs control table with `RESPONSE_LIFT` shown and named as an assumption. Then the **exception list**. |
| 4:30–5:00 | **The leverage brief** for a chronically-late buyer — honored/broken history, recommended terms, the message. Then: what's next with real data. |

---

## Tech stack

| Component | Choice |
|---|---|
| Data / ledger | Python + pandas + DuckDB (in-process) |
| ML | `sklearn` LogisticRegression |
| Agent | Local LLM via **Ollama** (`llama3.1:8b`, `temp 0`) — diagnosis + drafting only, on stage transitions, cached, rule-based fallback |
| Payments | `razorpay` PyPI SDK, test-mode keys — the one and only external network call |
| Clock | `pandas.date_range` in a `for` loop |
| API | FastAPI — read-only `GET` over `results.duckdb`, run with `uvicorn` |
| UI | Vite + React SPA — three views, fetches from `api.py`, `vite dev` on the demo path |
| Tests | `pytest` + `parametrize` |

Cut: PySpark, dbt, Airflow, APScheduler, Razorpay MCP server, Smart Collect, `create_invoice`,
FastAPI **webhook** server (the read-only API is not that), XGBoost / LightGBM,
Redux / React Router (three tabs = local state), **any hosted/paid LLM API** (the model
runs locally via Ollama).

---

## Explicitly not in scope

- Tamper-proof audit (append-only by convention + run hash only)
- Real payment settlement / webhook reconciliation (test-mode link proves connectivity, not settlement)
- Hourly simulation clock (day granularity + assigned timestamps covers the bound)
- Full test coverage (~15 tests targeting silent-failure paths; ledger stats, brief formatting, React rendering untested by choice)
- Auth, write endpoints, or a webhook on `api.py` — it is read-only `GET` over `results.duckdb`, nothing more
- A hosted LLM (Claude / OpenAI / etc.) — the model is local via Ollama; the only external call is one Razorpay test-mode link
- LLM fine-tuning — `llama3.1:8b` is used as-shipped; behavior is prompt + `temp 0` only
- CI/CD, packaging, deployment — deliverable is a repo + a video
- A live buyer-negotiation counterparty — the brief is the negotiating position, not a simulated negotiation

## Open decision (unresolved)

**Integration / debug time is unbudgeted.** The plan reserves the final half-day for
recording but nothing for the hours where six modules first talk to each other. Options:
(a) freeze the pipeline Sunday noon with steps 5–9 as the cut line; (b) build a thin
end-to-end skeleton on day 1 so integration risk surfaces Saturday morning; (c) drop
the React app + API and screenshot static output. Decide Friday, not Sunday.

---

*Full design record and 14 implementation tasks:
`~/.gstack/projects/Recoverly/shyamsundar-unknown-design-20260827-231916.md`*
