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

## Status — built and running

| | |
|---|---|
| Pipeline | `python run.py --no-llm` completes end to end and writes `results.duckdb` |
| Tests | **38 pytest tests**, green (`python -m pytest -q`) |
| Dashboard | `uvicorn api:app` + `web/` (Vite + React) on the demo path |
| Live Razorpay | one real test-mode payment link proven (`plink_…`), wired as `run.py --live-link INV-2032` |
| Companion docs | `README.md` · `DATA_MODEL.md` · `MODEL_TRAINING.md` · `RAZORPAY_API.md` |

**Latest run (`SEED=42`, `--no-llm`):**
253 invoices · 20 buyers · paid 213 · escalated 33 · unresolved 7 · exceptions 29 ·
DSO −3.1 days · Model 1 AUC 0.869 · Model 2 MAE 5.6d vs 7.0d baseline ·
net benefit ≈ −₹5k (see feature 10).

---

## What it does (features)

### 1. Watches and scores every open invoice — a two-model cascade
- **Model 1 — late-payment classifier** (`sklearn` `LogisticRegression`): `P(pays beyond
  terms)`. Screening layer. Precision / recall / **AUC** reported on a 6-week holdout.
- **Model 2 — expected-delay regressor** (`GradientBoostingRegressor`): predicts *days
  beyond terms*. Trained **only on invoices that were actually late**, and **only run when
  Model 1 clears `config.RISK_THRESHOLD` (0.5)** — a screened invoice never touches Model 2.
  Reported as **MAE vs a mean-prediction baseline**.
- Both share one feature builder (`agent._features`); split is strictly by `issue_date`
  (last 6 weeks = test) so no test row predates a training row.
- Because the data is synthetic, both models partly recover their own generator — stated on
  camera. The claim is the two-stage pipeline, not the scores.
- Output per invoice: `p_late`, `expected_delay_days`, a `severity` label
  (`mild` / `moderate` / `severe`), and a `segment` (`on_track` / `at_risk` / `slipping` /
  `overdue`). **Informational only** — the ladder does not read it.

### 2. Acts autonomously, within a deterministic ladder
- A day-stepping loop evaluates every open invoice, every simulated day.
- **Days-beyond-terms (DBT) picks the ladder rung** — a pure function of DBT and invoice
  state. Neither the models nor the LLM choose the action.
- The LLM runs **only on a stage transition**, to diagnose *why* the invoice is slipping
  and to draft the message. It never invents an action.

| Stage | Trigger | Action |
|---|---|---|
| 0 | 5 days before due | Pre-due courtesy reminder + payment link |
| 1 | Due date passed | Polite follow-up, restate terms |
| 2 | DBT 7 | Firm reminder, escalate to AP contact |
| 3 | DBT 15 | Early-settlement offer — **0.5% discount** (the 2% bound is the ceiling, not the default) |
| 4 | DBT 30 | Payment plan / partial settlement |
| 5 | DBT 45, or dispute, or opt-out | **Stop** — hand to human with a full brief |

### 3. Diagnoses the cause (local LLM, optional)
On a stage transition, a **local model (Ollama, `llama3.1:8b`)** reasons over the buyer
profile, invoice context, prior touches, Model 2's expected delay, and free-text notes to
produce one of: **cash-flow · process · dispute · stretch**. Recorded on the audit row;
shapes the drafted message.

**No external API.** Ollama runs on the machine — no API key, no network egress, no
per-call cost, `temperature: 0` + fixed `seed`. If Ollama is not running (the default
`run.py --no-llm` path), `diagnose()` falls back to a rule-based classifier over the buyer
profile and `draft()` to a per-stage template. The pipeline never blocks on the model.

### 4. Refuses, and escalates
**Five hard bounds**, defined once as `engine.BOUNDS` — a predicate list consumed by the
engine, the tests, and the audit rows:

| Name | Rule |
|---|---|
| `max_touches` | ≤ 6 touches per invoice, ever |
| `min_gap_72h` | ≥ 72 hours between touches |
| `business_hours` | Weekdays only; every action stamped 09:00–19:00 IST (deterministic, seeded from `invoice_id`) |
| `discount_cap_2pc` | Discount authority ≤ 2%; anything beyond → human |
| `maker_checker` | No autonomous action on invoices above `MAKER_CHECKER_THRESHOLD` (₹10L) |

**Stop conditions** — terminal vs silent hold:
- Dispute raised → **terminal**: fires the stage-5 escalation, writes the human-handoff audit row
- Buyer opt-out → **terminal**: same
- Promise-to-pay recorded → **silent hold**: agent goes quiet until the promised date
- Payment received → closes the invoice

Ask it for a 5% discount (`run.py` / `engine.run(force_discount=…)`) and it refuses,
escalates, and sets `human_gate_required = true`.

### 5. Learns that promises are cheap
- Buyer promises → agent goes silent until the promised date.
- Promise kept → invoice settles on that date.
- Promise broken → escalate, using the break as leverage.
- **Second broken promise (buyer-level)** → the agent stops honoring new promises from that
  buyer. New promises are still written to the audit trail (`promise_noted_ignored`) but no
  longer suppress touches; the invoice proceeds to its DBT-appropriate rung.

### 6. Moves real money — once
- **Batch run:** a simulated link adapter (`razorpay_link.create_link(..., live=False)`),
  zero network calls, deterministic ids.
- **Demo invoice:** `run.py --live-link INV-2032` creates **exactly one** real Razorpay
  **test-mode** payment link during the run (stage 0), simulates every other. Needs
  `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` in the env; the invoice amount must be under
  Razorpay's ₹5,00,000 test-mode cap (`INV-2032` is ₹4,97,000). On any failure the adapter
  falls back to a simulated id and says so — the demo never breaks.
- The link carries `customer` (the buyer's AP contact), `notify: {sms: true, email: false}`
  and `reminder_enable: true` — Razorpay texts the link and auto-nudges it; the
  personalized email is `notify.py`'s job.
- The adapter caches on success, so a link requested at both stage 0 and stage 3 is **one**
  API call.

### 6b. Writes the buyer email
Every rung 0–4 composes a full personalized email to the buyer's AP contact
(`agent.compose_email` — LLM from a facts dict when Ollama is up, deterministic slot
template on the `--no-llm` path, cached by `(invoice_id, stage)` at `temp 0`). Tone scales
with the rung; the ask is conditioned on the diagnosis (a `stretch` buyer at stage 3 gets a
firm notice, not the discount). The internal diagnosis label and the leverage brief are
**never** named to the buyer. `notify.py` mirrors `razorpay_link.py`: dry-run by default
(records `email_to` / `email_body` / `email_message_id` on the audit row and shows them in
the dashboard), `live=True` sends for real over SMTP (stdlib `smtplib`, credentials from
`SMTP_USER`/`SMTP_PASSWORD` env vars only). `run.py --live-link INV-2032 --demo-email
you@gmail.com` sends **exactly one** real email — that invoice's first touch carrying the
real Razorpay link — to the given address; every other email in the run stays dry-run to
the buyer's synthetic `.example` address. Any SMTP failure falls back to dry-run so the
demo never breaks. Terminal stops (dispute, opt-out,
stage 5) send **nothing** to the buyer — the escalation is for a human at the SME.

### 7. Leaves an audit trail
One row per agent action (append-only *by convention* + a generation timestamp — a DuckDB
table can be overwritten, so the claim is **traceability**, not tamper resistance):

```
timestamp | invoice_id | buyer_id | risk_score | expected_delay_days | risk_severity |
risk_segment | diagnosis | ladder_stage | action_taken | message_sent |
razorpay_object_id | bounds_checked | human_gate_required |
email_to | email_body | email_message_id | outcome | outcome_timestamp
```

Pick any row and trace exactly why it happened.

### 8. The leverage brief — the differentiator
For a featured buyer (`brief.MIN_SAMPLE`, n ≥ 20, enforced), one page:
- Honored vs broken count (`honored` / `broken` — one per invoice; honored = cleared on or
  before the due date and never disputed)
- Mean DBT and standard deviation (`mean_dbt`, `sd_dbt`)
- `promise_kept_rate` — **`None`**, not 0.0, for a buyer who never promised
- Dispute rate; peer mean DBT on the same terms
- **`recommended_terms`** — deterministic rule over honored-rate and mean DBT
- **`justification`** — LLM-drafted (template fallback), with the evidence line attached
- **`message`** — the message to send

Same split as the ladder: deterministic recommendation, generative language.

### 9. Dashboard (React SPA + read-only API)
- **`api.py`** — FastAPI, read-only `GET` over `results.duckdb`:
  `/invoices` · `/metrics` · `/buyers` · `/buyers/{id}/brief` · `/audit` · `/exceptions`.
  No writes, no auth, no webhook. `/metrics` returns `model1` + `model2` metrics.
- **`web/`** — Vite + React SPA (`web/src/App.jsx`), three views: **Queue** (invoice list +
  live audit-trail detail panel, shows Model 1 P(late) and Model 2 predicted delay),
  **Recovery results** (treatment vs control, the RESPONSE_LIFT honesty note, exception
  list), **Leverage brief** (buyer switcher + the brief). Razorpay Blade palette (azure
  `#1364F1`, navy `#021331`), Inter, ~1 component file.
- The pipeline writes `results.duckdb`; the API only reads it; the SPA only renders.

### 10. Reports honestly (treatment vs control)
- **253 invoices** split 70% treatment / 30% control (holdout, no agent contact),
  stratified by buyer.
- Reported: ₹ recovered, % of at-risk value recovered, DSO reduction (days), cash pulled
  forward (₹-days), discount cost, **net benefit** = cash-acceleration value − discount
  cost (priced at an 18% SME cost of capital), promise-kept rate.
- **The response model is shown on screen and named as an assumption.** The generator
  decides who pays when; a treatment effect only exists because an authored rule
  (`config.RESPONSE_LIFT`, benchmark-anchored) maps agent action → earlier payment with
  some probability, applied to **treatment only**. The control group proves the
  **measurement machinery** is correct — it does not prove the agent works.
- **Honest finding:** the free reminder rungs (0–2) pull DSO down ~3 days (~₹66k of
  acceleration value); the stage-3 discount rung roughly gives it back, so overall net
  benefit is ≈ break-even (−₹5k). This is surfaced in `/metrics.interpretation` and framed
  as a tuning lever an operator can tighten or disable — not hidden.
- **Exception list:** every treatment invoice the agent could not resolve, honestly
  labelled (dispute / opt-out / partial / maker-checker).

---

## Architecture

### System overview

```
                         ┌───────────────────────────────────────────────┐
                         │                 ledger.py                      │
                         │  253 invoices · 20 buyers (8 heavy / 12 light) │
                         │  6 months · SEED=42 · generated ONCE → parquet │
                         │                                               │
                         │  ┌─────────────┐  ┌──────────────────────────┐ │
                         │  │ buyer        │  │ dated event stream        │ │
                         │  │ profiles     │  │ promises · disputes ·     │ │
                         │  │ (dbt dist,   │  │ opt-outs                  │ │
                         │  │ seasonality, │  │ (discovered on their date,│ │
                         │  │ responsive-  │  │  NOT prebuilt — no future │ │
                         │  │ ness)        │  │  leakage)                 │ │
                         │  └─────────────┘  └──────────────────────────┘ │
                         └───────────────────────┬───────────────────────┘
              config.py: SEED · LADDER · BOUNDS  │ committed parquet
              limits · RISK_THRESHOLD ·          ▼
              RESPONSE_LIFT (STATED ASSUMPTION,  │
              treatment-only)         ┌──────────┴──────────────────────────────────┐
                                      │                  engine.py                  │
                                      │           the simulated clock — agent       │
                                      │                                             │
                                      │  for day in date_range(start, end):         │
                                      │    apply_events(day)   ← promises/disputes/  │
                                      │                          opt-outs arrive HERE│
                                      │    for inv in open_invoices(day):            │
                                      │      terminal stop?  → escalate → AUDIT (5)  │
                                      │      silent hold?     → skip                 │
                                      │      stage = ladder_stage(DBT)  ← pure fn    │
                                      │      if stage advanced and bounds_ok:        │
                                      │        risk = risk_fn(inv)   ← cascade dict  │
                                      │        act(inv, stage) → AUDIT               │
                                      │    settle_payments(day)  ← natural_pay_date  │
                                      │                            + RESPONSE_LIFT   │
                                      │                                             │
                                      │  BOUNDS list · promise feedback loop ·       │
                                      │  business-hour timestamp seeded from id      │
                                      └───┬─────────────────┬───────────────┬────────┘
                        on stage          │  every action   │  stage 0 / 3  │
                        transition ───────┘                 │               │
                                  ▼                         ▼               ▼
   ┌────────────────────────────────────┐  ┌────────────────────┐  ┌──────────────────────┐
   │             agent.py               │  │  audit  (in         │  │   razorpay_link.py   │
   │                                    │  │  results.duckdb)   │  │                      │
   │ Model 1  train_risk_model          │  │                    │  │ create_link(inv,     │
   │   LogisticRegression · P(late)     │  │ 1 row / action     │  │             live=?)  │
   │   6-week holdout · AUC             │  │ append-only by      │  │  live=False → sim    │
   │ Model 2  train_delay_model         │  │ convention         │  │  live=True  → ONE     │
   │   GBR · days late · late-only ·    │  │                    │  │   real test-mode call│
   │   run iff Model 1 ≥ RISK_THRESHOLD │  │                    │  │  cached on success · │
   │ make_risk_fn → {p_late,            │  │                    │  │  falls back on fail  │
   │   expected_delay_days, severity,   │  └─────────┬──────────┘  └──────────────────────┘
   │   segment}                         │            │
   │ diagnose()  local LLM (Ollama      │            ▼
   │   llama3.1:8b, temp 0) or rules    │  ┌────────────────────────┐
   └────────────────┬───────────────────┘  │     results.duckdb      │
                    │                      │  invoices · audit ·     │
   ┌────────────────┴───────┐              │  buyers · events ·      │
   │        brief.py         │              │  briefs · exceptions ·  │
   │ per-buyer leverage brief│              │  meta (metrics + both   │
   │  honored/broken ·       │              │  models' metrics)       │
   │  promise rate (None-safe)│             └───────────┬────────────┘
   │  recommended terms (det.)│                         │ READ-ONLY
   │  justification (LLM) ·   │      ┌──────────────────┴─────────────┐
   │  message                 │      │        api.py (FastAPI)        │
   └─────────────────────────┘      │  GET /invoices /metrics        │
                                    │  /buyers /buyers/{id}/brief    │
   run.py  orchestrates: ledger →   │  /audit /exceptions           │
   train both models → engine.run   │  read-only · no auth · no      │
   → metrics.py → results.duckdb    │  webhook · uvicorn            │
                                    └───────────────┬───────────────┘
                                                    │ HTTP (localhost)
                                                    ▼
                              ┌────────────────────────────────────────┐
                              │        web/  (Vite + React SPA)         │
                              │   Queue  │ Recovery results │ Leverage  │
                              │   list + │ treatment vs     │ brief +   │
                              │   audit  │ control +        │ recommend │
                              │   detail │ exception list   │ + message │
                              │   Razorpay Blade palette · ~1 component │
                              └────────────────────────────────────────┘
```

### Modules

| File | Responsibility |
|---|---|
| `config.py` | Every constant: `SEED`, `LADDER`, `BOUNDS` limits, `RISK_THRESHOLD`, `RESPONSE_LIFT`, `EARLY_SETTLEMENT_DISCOUNT`, thresholds, paths |
| `ledger.py` | Synthetic generator · buyer behavioral profiles · dated event stream · seeded, generated once, committed as parquet under `data/ledger/` |
| `engine.py` | Day loop · event application · deterministic ladder · `BOUNDS` · split stop conditions · promise feedback loop · audit writes · `live_link_invoice` hook |
| `agent.py` | **Model 1** `train_risk_model` · **Model 2** `train_delay_model` (late-only) · `load_or_train_models` (loads `models/*.joblib`, re-fits on miss) · `make_risk_fn` cascade closure · `diagnose()` rule-based + optional Ollama · `draft_justification()` · `compose_email()` |
| `models/` | Committed artifacts `model1_logistic_regression.joblib` + `model2.joblib` and the notebooks (`model1.ipynb` / `Model2.ipynb`) that trained them. `run.py` loads them; `--retrain` regenerates |
| `brief.py` | Per-buyer leverage brief — deterministic recommended terms, generative justification, `None`-safe `promise_kept_rate` |
| `razorpay_link.py` | Link adapter with a `live` flag — simulated batch, one real test-mode call, `customer`/`notify`/`reminders`, cache + graceful fallback (named `_link` so it does not shadow the `razorpay` SDK) |
| `notify.py` | Mailer (mirrors `razorpay_link.py`) — dry-run by default; `live=True` sends one real SMTP email for the `--live-link` / `--demo-email` invoice, fallback on failure |
| `metrics.py` | Treatment vs control · cash pulled forward · net benefit · `interpretation` string · exception list |
| `run.py` | Orchestrator → `results.duckdb`. Flags: `--no-llm`, `--fresh`, `--live-link INV-XXXX` |
| `api.py` | FastAPI — read-only `GET` over `results.duckdb`. No writes, no auth, no webhook |
| `web/` | Vite + React SPA — three views, fetches from `api.py`. Razorpay Blade palette, ~1 component file, no Redux, no router |
| `test_engine.py` | **38 pytest tests** — the three silent-failure gaps, every `BOUNDS` predicate via `parametrize`, the ladder boundaries, the risk cascade gate, the buyer-email rules, the demo beats, determinism |

### Key design rules

- **Deterministic ladder, generative language.** DBT picks the rung; the models score and
  the LLM diagnoses/writes. The prediction is informational.
- **The clock is the only stateful piece.** Everything else is a pure function called from
  the loop. This is what makes it an agent, not a batch report.
- **Events are discovered, not prebuilt.** A `promise` row with `kept` visible to the
  engine would hand the agent the future.
- **`BOUNDS` defined once** in `engine.py`. Engine gates on it, tests parametrize over it,
  audit rows record which passed.
- **No holdout leakage.** `agent.train_test_split_by_date` splits strictly by `issue_date`;
  Model 2 additionally trains on the late subset only.
- **No external API** except the one Razorpay test-mode link. The LLM is local (Ollama).
- **Determinism contract.** `SEED = 42`, ledger committed as parquet, local LLM at
  `temperature 0` + fixed seed, LLM responses cached. Numbers said on camera stay true.
- **The response model is an assumption, not a measurement.** Named, benchmark-anchored,
  shown on screen, applied to treatment only.

---

## Workflow

### As built (order the modules were written)

| # | Step | Notes |
|---|---|---|
| 1 | `config.py` + `ledger.py` | Constants, generator, buyer profiles, dated event stream. Committed parquet. Everything depends on this. |
| 2 | `engine.py` + `test_engine.py` | Day loop, ladder, `BOUNDS`, split stop conditions, promise loop, audit. Tests alongside. |
| 3 | `brief.py` | The differentiator, built before the models and the LLM so it can't be the thing cut. |
| 4 | `razorpay_link.py` | The `live` flag; one real test-mode link end to end. |
| 5 | `agent.py` — Model 1 + Model 2 | `train_risk_model` (logistic, 6-week holdout), then `train_delay_model` (GBR `lr=0.03/depth=2`, late-only, gated). Recipes match `models/*.ipynb`; `load_or_train_models` loads the committed `.joblib`s. `make_risk_fn` wires the cascade. |
| 6 | `agent.py` — `diagnose()` | Rule-based path first, then optional Ollama (`llama3.1:8b`, `temp 0`, cached). |
| 7 | `metrics.py` | Treatment/control split, cash pulled forward, net benefit, exception list. |
| 8 | `run.py` + `api.py` + `web/` | Orchestrator → `results.duckdb`; FastAPI read-only endpoints; Vite + React SPA, three views, Razorpay Blade palette. |
| 9 | Failure scenarios | Dispute / promise / 5%-discount, as `test_engine.py` beats and `--live-link` for the on-camera link. |
| 10 | Video | Reserve the final half-day. |

### Runtime flow (one simulated day)

```
1. apply_events(day)
   └─ any promise / dispute / opt-out dated today is applied now

2. for each open invoice:
   ├─ terminal stop (dispute / opt-out)?  → escalate_to_human() → audit row (stage 5)
   ├─ silent hold (active promise window)? → skip
   ├─ compute ladder_stage from DBT + state
   └─ stage advanced AND bounds_ok?
        ├─ risk = risk_fn(inv)   ← Model 1, then Model 2 iff p_late ≥ RISK_THRESHOLD
        ├─ on stage transition: agent.diagnose() (rules or Ollama) → draft message
        ├─ stage 0 or 3: razorpay_link.create_link(live = inv is the --live-link invoice)
        ├─ act() → append audit row (business-hour timestamp seeded from invoice_id)
        └─ if 2nd broken promise for this buyer: stop honoring future promises

3. settle_payments(day)
   └─ natural_pay_date, then RESPONSE_LIFT (treatment only), then kept promises
```

### Demo flow (5-minute video)

| Time | Segment |
|---|---|
| 0:00–0:35 | The problem, concretely. One SME, receivables trapped, chronic-late buyers. Name the Fix My Itch score (82.8). |
| 0:35–1:15 | Architecture — one diagram, 40 seconds. Deterministic ladder, generative language, hard bounds, two-model cascade. |
| 1:15–2:30 | Live single-invoice trace: Model 1 P(late) → Model 2 expected delay → diagnosis → rung → **real Razorpay test-mode link on screen** (`run.py --live-link INV-2032`) → drafted message → audit row. |
| 2:30–3:15 | Failure & bounds demo: (1) dispute → agent stops, escalation audit row; (2) promise-to-pay → agent goes quiet, resumes; (3) 5% discount ask → agent refuses, escalates. |
| 3:15–4:30 | Batch results: treatment vs control with `RESPONSE_LIFT` named as an assumption, the honest net-benefit framing, then the **exception list**. |
| 4:30–5:00 | **The leverage brief** for a chronically-late buyer — honored/broken history, recommended terms, the message. Then: what's next with real Razorpay data. |

---

## Tech stack

| Component | Choice |
|---|---|
| Data / ledger | Python + pandas + DuckDB (in-process); parquet |
| ML — Model 1 | `sklearn` `LogisticRegression` (late / not-late) |
| ML — Model 2 | `sklearn` `GradientBoostingRegressor(n_estimators=100, learning_rate=0.03, max_depth=2)` (expected days late, late-only) — tuned config from `models/Model2.ipynb`; MAE 5.6d vs 7.0d baseline |
| ML — artifacts | `models/*.joblib`, loaded by `agent.load_or_train_models`; re-fit + rewritten on version-skew or `--retrain` |
| Agent | Local LLM via **Ollama** (`llama3.1:8b`, `temp 0`) — diagnosis + drafting only, optional, rule-based fallback |
| Payments | `razorpay` PyPI SDK, test-mode keys — the one and only external network call; link carries `customer` + `notify` |
| Email | `notify.py` — dry-run by default; `live=True` sends one real SMTP email (`SMTP_USER`/`SMTP_PASSWORD`) for the demo invoice; personalized body from `agent.compose_email` |
| Clock | `pandas.date_range` in a `for` loop |
| Orchestration | `run.py` (`--no-llm` / `--fresh` / `--live-link`) |
| API | FastAPI — read-only `GET` over `results.duckdb`, `uvicorn` |
| UI | Vite + React SPA — three views, Razorpay Blade palette, fetches from `api.py` |
| Tests | `pytest` + `parametrize` (29) |

Cut: PySpark, dbt, Airflow, APScheduler, Razorpay MCP server, Smart Collect, `create_invoice`,
webhooks, XGBoost / LightGBM, Redux / React Router, any hosted/paid LLM API. Optional
Razorpay APIs for a production version are catalogued in `RAZORPAY_API.md`.

---

## Explicitly not in scope

- Tamper-proof audit (append-only by convention + timestamp only)
- Real payment settlement / webhook reconciliation (test-mode link proves connectivity, not settlement)
- Hourly simulation clock (day granularity + assigned timestamps covers the bound)
- Full test coverage (38 tests target silent-failure paths and the demo beats; ledger stats, brief formatting, React rendering untested by choice)
- Auth, write endpoints, or a webhook on `api.py` — read-only `GET` only
- A hosted LLM — local via Ollama; the only external call is one Razorpay test-mode link
- LLM fine-tuning — `llama3.1:8b` is used as-shipped; behavior is prompt + `temp 0` only
- Real email transmission — `notify.send(live=True)` is a guarded stub; no SES/SMTP wired. WhatsApp and inbound-reply handling are also out.
- Real buyer contact data — AP names / emails / phones are synthetic, `.example` TLD
- The prediction driving the ladder — Model 1/2 outputs are informational; the rung is a pure function of observed DBT
- CI/CD, packaging, deployment — deliverable is a repo + a video
- A live buyer-negotiation counterparty — the brief is the negotiating position, not a simulated negotiation

## Resolved

**Integration risk (previously open):** the pipeline is built and runs end to end;
`run.py --no-llm` → `results.duckdb` → `api.py` → `web/` all work, 38 tests green. No
blocking integration issues surfaced. Remaining polish: swap Model 2 to `Ridge`/Poisson for
the synthetic build, add the buyer-mean baseline to the metrics, record the video.

---

*Companion docs: `README.md` · `DATA_MODEL.md` · `MODEL_TRAINING.md` · `RAZORPAY_API.md`
Full design record: `~/.gstack/projects/Recoverly/shyamsundar-unknown-design-20260827-231916.md`*
