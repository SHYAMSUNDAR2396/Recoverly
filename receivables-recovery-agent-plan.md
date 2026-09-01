# Receivables Recovery Agent for SMEs

**Razorpay AI Buildathon 2026 — Track 03: AI Revenue Recovery**

**Source problem:** "Why can't SMEs negotiate favorable payment terms with large buyers?" — Itch score 82.8, the highest-scored B2B Services problem on Razorpay's Fix My Itch board.

---

## The pitch in one line

Large buyers pay SMEs late because they can, and SMEs have no leverage and no time to chase. This agent watches every open invoice, predicts which ones are about to slip, picks the right intervention, and executes a bounded recovery workflow — then reports how much cash it actually pulled forward, measured against a control group.

That last part is the differentiator. Most submissions will show "the agent sent a reminder and the money came in." This shows *causal* recovery: money that would not have arrived without the agent.

---

## Architecture

### Layer 1 — Ledger and data foundation

Synthetic dataset: ~250 invoices across ~20 buyers, 6 months of history.

Each invoice carries:

| Field | Notes |
|---|---|
| `invoice_id` | Primary key |
| `buyer_id` | FK to buyer dimension |
| `amount` | Invoice value |
| `issue_date` | |
| `due_date` | |
| `terms` | Net 30 / 45 / 60 |
| `status` | open / partial / paid / disputed |
| `payment_date` | Null until settled |
| `dispute_flag` | Boolean |

Each buyer gets a **behavioral profile**: historical days-beyond-terms distribution, seasonality (quarter-end squeezes), partial-payment tendency, dispute rate, and responsiveness to prior nudges.

Make it realistic. Some buyers are chronically 20 days late but always pay. Some pay on time then suddenly stop. Some dispute purely to stall. Realistic messiness is what makes the final metrics believable.

Build the generator in PySpark or plain Python, land it in DuckDB or Postgres, and model the ledger properly: invoice fact table, buyer dimension, payment event table. A dbt layer for the transforms makes the "defend your architecture" conversation straightforward.

### Layer 2 — Risk scoring

For each open invoice, predict `P(slips beyond terms)` and `expected days late`.

**Features:**

- Buyer's historical days-beyond-terms mean and variance
- Invoice size relative to that buyer's typical invoice
- Days remaining until due date
- Quarter-end flag
- Prior touch history on this invoice
- Open dispute flag
- Buyer's recent payment-velocity trend

**Model:** gradient boosting (XGBoost / LightGBM) or plain logistic regression. Simple is fine and more defensible.

Hold out the last 6 weeks and report precision, recall, and AUC. Track 03 doesn't mandate this — importing Track 02's rigor is what makes the submission look serious.

Output is not just a score but a **segmentation**: `on_track` / `at_risk` / `slipping` / `overdue` / `chronic`.

### Layer 3 — Diagnosis and policy engine

This is where the LLM earns its place. For each at-risk invoice, the agent reasons over the buyer profile, invoice context, prior touches, and any free-text notes (emails, remarks) to produce a **diagnosis**:

- Cash-flow issue on the buyer's side
- Process issue (invoice never entered their AP system)
- Genuine dispute
- Deliberate stretching

The diagnosis drives intervention selection from a **fixed ladder**:

| Stage | Trigger | Action |
|---|---|---|
| 0 | 5 days before due | Pre-due courtesy reminder + payment link |
| 1 | Due date passed | Polite follow-up, restate terms |
| 2 | DBT 7 | Firm reminder, escalate to AP contact |
| 3 | DBT 15 | Early-settlement offer (bounded discount, max 2%) |
| 4 | DBT 30 | Payment plan proposal / partial settlement |
| 5 | DBT 45 or dispute | **Stop** — hand to human with a full brief |

*DBT = days beyond terms.*

The LLM chooses **which rung** and **drafts the message**. It does not invent new actions. Deterministic ladder, generative language — an easy split to defend to a judge.

### Layer 4 — Execution via Razorpay MCP (test mode)

| Capability | Razorpay surface |
|---|---|
| Invoice-specific payment link with exact amount | `create_payment_link` / `create_payment_link_upi` |
| Auto-reconciled inbound NEFT/RTGS/UPI per buyer | Smart Collect virtual account |
| Close the loop the moment money lands | Webhooks: `payment.captured`, `payment_link.paid` |
| Initial issuance flow (optional) | `create_invoice` |

The Smart Collect virtual account per buyer is worth calling out in the pitch — it removes manual invoice matching entirely, which is a real SME pain in its own right.

### Layer 5 — Bounds, gates, and audit trail

Track 03's bar demands "compliant escalation, stopping rules, and an audit trail." Make these explicit and visible, not buried in code.

**Hard bounds:**

- Max 6 touches per invoice, ever
- Minimum 72 hours between touches
- Business hours only (09:00–19:00 IST, no weekends)
- Discount authority capped at 2% — anything beyond requires human approval
- No agent action on invoices above a set threshold without maker-checker sign-off

**Instant stop conditions:**

- Dispute raised
- Promise-to-pay recorded (until the promised date passes)
- Buyer opt-out
- Payment received

**Promise-to-pay state machine.** When a buyer says "we'll pay on the 15th," the agent records the promise, goes silent until the 15th, then either closes it out or escalates using the broken promise as leverage. Track `promise_kept_rate` per buyer and feed it back into the risk model.

This piece is genuinely underserved — the shipped Gnani.ai × Razorpay collections platform does not publicly claim a promise-to-pay tracking loop.

**Audit record** — one immutable row per agent action:

```
timestamp | invoice_id | buyer_id | risk_score | diagnosis |
ladder_stage | action_taken | message_sent | razorpay_object_id |
bounds_checked[] | human_gate_required | outcome | outcome_timestamp
```

Any judge should be able to pick a random charge or message and trace exactly why it happened.

---

## Measurement design

This is the strongest card in the submission.

Split the 250 invoices: **70% treatment, 30% control** (holdout, no agent contact). Match the groups on buyer profile and invoice size.

**Report:**

- ₹ recovered and % of at-risk value recovered — treatment vs control
- DSO reduction in days — treatment vs control
- Cash pulled forward (₹-days) — the real economic value
- Cost of intervention — discounts given, touches spent
- **Net benefit** = cash acceleration value − discount cost
- Promise-kept rate
- **Exception list**: invoices the agent could not resolve, and why

The control group converts "the agent sent emails and money arrived" into "the agent caused ₹N of recovery." Almost nobody else will do this. Say so explicitly in the pitch.

---

## Tech stack

| Component | Choice |
|---|---|
| Data / ledger | Python + DuckDB (or Postgres); PySpark for the generator |
| Transforms | dbt |
| ML | scikit-learn / LightGBM |
| Agent | Claude via API, policy ladder as structured tool definitions |
| Payments | Razorpay MCP server (`mcp.razorpay.com`), test-mode keys |
| Orchestration | APScheduler or cron simulating daily runs (Airflow is overkill) |
| UI | Streamlit dashboard — invoice queue, risk heatmap, action log, live metrics |
| Webhooks | FastAPI endpoint |

---

## Demo plan (5-minute video)

**0:00–0:35 — The problem, concretely.**
One SME, ₹42L in receivables, 6 large buyers, average 34 days beyond terms. Show the cash trapped. Name the Fix My Itch problem and its 82.8 score — this signals the problem was sourced from Razorpay's own research, not invented.

**0:35–1:15 — Architecture walkthrough.**
One clean diagram, 40 seconds. Data → risk → diagnosis → policy ladder → Razorpay execution → audit. State clearly: deterministic ladder, generative language, hard bounds.

**1:15–2:30 — Live single-invoice trace.**
Pick one invoice. Show the risk score rising, the diagnosis produced, the ladder rung chosen, the actual Razorpay payment link created (test mode, visible on screen), the drafted message, then the webhook firing when payment lands. Show the resulting audit row.

**2:30–3:15 — The failure and bounds demo.**
Deliberately trigger three things:

1. A buyer raises a dispute → agent stops immediately
2. A buyer promises to pay on a date → agent goes quiet, resumes only after
3. An invoice needs a 5% discount → agent refuses and escalates to human

This is the "bounded, gated, one failure handled gracefully" evidence the track explicitly asks for.

**3:15–4:30 — The batch results.**
Run all 250 invoices. Show the treatment vs control table: ₹ recovered, DSO delta, net benefit after discount cost. Then show the **exception list** — the invoices it couldn't resolve, honestly labeled. Judges reward the exception list more than the win rate.

**4:30–5:00 — What's next, and what you'd do with real data.**
Brief, confident, no overclaiming.

---

## Build sequence

Do these in order. Steps 1–4 alone constitute a working submission; everything after is upside.

1. Ledger generator + realistic buyer profiles ← *everything depends on this*
2. Razorpay MCP integration, one working payment link end to end
3. Policy ladder + bounds + audit log (deterministic, no LLM yet)
4. Webhook receiver + auto-reconciliation
5. Risk model
6. LLM diagnosis + message drafting
7. Control/treatment split + metrics
8. Dashboard
9. Failure scenarios
10. Video

Do not invert this order.

---

## Open decision

Decide early whether to frame this as a **collections agent** (chases late payments) or a **terms-negotiation agent** (proactively renegotiates payment terms with large buyers before invoices are even issued).

The Fix My Itch problem is literally about *negotiating terms*. Leaning into the proactive framing would be more faithful to the source problem and sits in less crowded territory than another collections bot.
