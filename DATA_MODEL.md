# Data model

How the synthetic dataset is shaped, why, and how it feeds the two risk models.
Everything here is produced by `ledger.py` (seeded, committed as parquet under
`data/ledger/`) and consumed by `agent.py` / `engine.py`.

---

## 1. Entities

`ledger.generate_ledger()` returns three tables:

| Entity | Table | Grain | Count |
|---|---|---|---|
| **Buyer** | `buyers` | one row per buyer | 20 |
| **Invoice** | `invoices` | one row per invoice | ~253 |
| **Event** | `events` | one dated thing that happens to an invoice | ~80 |

Two more frames are **derived** by the engine, not stored in the ledger:

| Frame | Produced by | Grain |
|---|---|---|
| `audit` | `engine.run` | one row per agent action |
| `final` | `engine.run` | `invoices` + outcome columns (`paid`, `paid_day`, `effective_dbt`, `escalated`, `touches`) |

### Payment is not an entity

A payment is a **field**, not a row: `natural_pay_date` at generation, `paid_day`
after the engine runs. Design-review decision — if payment were an event, the
authored response model (`config.RESPONSE_LIFT`) would leak into the control
group and the treatment effect could not be isolated. Payments are computed in
`engine.run`'s `settle_payments` step, never pre-joined.

### "Historical behavior" is not a table

It exists two ways:
- **Buyer parameters** the generator draws from (`dbt_mean`, `dispute_rate`, …).
- **Derivable** from the invoice rows: "this buyer's last 27 invoices averaged
  +17 days" is a `GROUP BY buyer_id` over `invoices` — exactly what `brief.py`
  computes for the leverage brief.

---

## 2. What each entity contains

### `buyers`

| Column | Type | Meaning |
|---|---|---|
| `buyer_id` | str (PK) | `BUY-00` … `BUY-19` |
| `name` | str | display name |
| `tier` | str | `heavy` (~25 invoices) / `light` (~8) |
| `dbt_mean` | float | historical mean days-beyond-terms |
| `dbt_sd` | float | its standard deviation |
| `qend_squeeze` | float | extra slowdown on invoices due at a quarter-end |
| `partial_rate` | float | tendency to pay partially |
| `dispute_rate` | float | share of invoices that hit a dispute |
| `will_opt_out` | bool | does this buyer opt out of automated contact |
| `promise_rate` | float | how often the buyer promises when nudged |
| `promise_keep_rate` | float | of those promises, how many are kept |
| `responsiveness` | float | reserved — how much the buyer reacts to touches |
| `ap_contact` | str | AP contact name (synthetic) |
| `email` | str | `ap@{slug}.example` — reserved TLD, undeliverable by design |
| `phone` | str | `+9198…` (synthetic) |

### `invoices`

| Column | Type | Meaning |
|---|---|---|
| `invoice_id` | str (PK) | `INV-2000` … |
| `buyer_id` | str (FK → buyers) | |
| `amount` | float | ₹, lognormal, scaled by tier |
| `issue_date` | date | spread across the first ~4 months |
| `due_date` | date | `issue_date + terms` |
| `terms` | int | 30 / 45 / 60 |
| `natural_pay_date` | date | **counterfactual** — when it pays with no agent |
| `natural_dbt` | int | `(natural_pay_date − due_date).days`, clipped ≥ −10 |
| `lift_roll` | float | one seeded uniform per invoice; the response model compares it to `RESPONSE_LIFT[stage]` probability |
| `group` | str | `treatment` (70%) / `control` (30%), stratified by buyer |
| `disputed` | bool | a dispute event exists for it |
| `held` | bool | disputed or opted-out → never settles |

### `events`

| Column | Type | Meaning |
|---|---|---|
| `invoice_id` | str (FK) | |
| `buyer_id` | str (FK) | |
| `day` | date | the date the event is *discovered* by the engine |
| `kind` | str | `dispute` / `opt_out` / `promise` |
| `promised_date` | date / NaT | set only for a **kept** promise |

Events are discovered on their `day` — never pre-joined to the invoice. A
`promise` row with a visible `kept` flag would hand the agent the future.

---

## 3. Relationships

```
buyer   1 ──< N  invoice     invoices.buyer_id  → buyers.buyer_id
invoice 1 ──< N  event       events.invoice_id  → invoices.invoice_id
```

No payment table. No many-to-many. `audit` and `final` both key back to
`invoice_id`.

---

## 4. Generating realistic payment behavior

Layered in `ledger.py`:

1. **Buyer archetypes** (`NAMED_BUYERS`):
   - *Meridian Retail* — chronic stretcher (mean +17, zero disputes)
   - *Harbour Textiles* — disputes-to-stall (22% dispute rate, opts out)
   - *Crest Pharma* — seasonal cash-flow (high `qend_squeeze`)
   - *Nandi Logistics* — pays early (mean −3)
   - *Blueleaf Foods* — process issues (invoices miss AP intake)
   - the other 15 buyers are random draws across the same parameter space.
2. **Seasonality** — `_month_end_squeeze`: invoices due in the last 10 days of
   Mar/Jun/Sep/Dec use `dbt_mean × (1 + qend_squeeze)`.
3. **Core draw** — `natural_dbt = round(normal(dbt_mean × squeeze, dbt_sd))`,
   clipped ≥ −10; `natural_pay_date = due_date + natural_dbt`.
4. **Amounts** — lognormal, tier-scaled (heavy ≈ ₹6L base, light ≈ ₹1.8L).
5. **Events** — `_build_events`:
   - dispute: `rng < buyer.dispute_rate`, dated `due + 3..18d`; marks `held`
   - opt-out: one buyer-level event on that buyer's worst (highest `natural_dbt`)
     invoice; marks `held`
   - promise: on invoices with `natural_dbt > 5`, dated `due + offset`, kept with
     probability `buyer.promise_keep_rate`

The result is deliberately messy: some buyers always 20 days late but always pay;
some dispute purely to stall; some fine until a quarter-end.

---

## 5. What becomes a feature

`agent._features(invoices, buyers)` → 7 columns, all knowable **before the due
date** (no post-hoc leakage):

| Feature | Definition |
|---|---|
| `buyer_dbt_mean`, `buyer_dbt_sd` | the buyer's historical lateness profile |
| `buyer_dispute_rate` | share of the buyer's invoices that hit a dispute |
| `buyer_promise_keep` | the buyer's promise-kept rate |
| `amount_rel` | this invoice ÷ the buyer's median invoice |
| `terms` | 30 / 45 / 60 |
| `quarter_end` | 1 if due in the last 10 days of a quarter month |

The same builder feeds both models and the live `make_risk_fn` closure.

---

## 6. Model 1's target

`(natural_dbt > 0)` — **binary**. *Will this invoice be paid beyond terms at
all?* Model 1 is the screening layer; only invoices it scores at or above
`config.RISK_THRESHOLD` (0.5) reach Model 2.

---

## 7. Model 2's target

`natural_dbt` itself — **the number of days late** — trained **only on the rows
where `natural_dbt > 0`**. *Given it's late, how late?* Forcing one regression to
also fit the on-time zeros makes it a poor zero-inflated fit; the late-only subset
gives a cleaner magnitude estimate.

---

## 8. Creating enough training examples

- ~253 invoices from 20 buyers (heavy ≈ 25 each, light ≈ 8), issued across ~5
  months.
- Model 1: **167 train / 86 test**.
- Model 2: **129 late rows / 68 late in the holdout**.

That is small — hence `Ridge` / `PoissonRegressor` is the defensible estimator
for Model 2 on this data (see `MODEL_TRAINING.md`). To get more without leaving
synthetic: raise `HEAVY_INVOICES` / `N_BUYERS` in `config.py`, or widen
`SIM_START … ISSUE_LAST`. The production answer is the Razorpay Settlements API
replacing the generator entirely.

---

## 9. Splitting history vs future without leakage

`agent.train_test_split_by_date`:

1. Sort invoices by `issue_date`.
2. Cutoff = `max(issue_date) − HOLDOUT_WEEKS (6)`. `≤ cutoff` → train;
   after → test.
3. Enforced: `train.issue_date.max() < test.issue_date.min()` —
   `test_engine.py::test_risk_model_has_no_holdout_leakage`.
4. Model 2's late-only filter is applied *within* each side; it never crosses the
   cutoff.

### Real-data caveat

Here the label (`natural_dbt`) is baked in at generation, so it is always
available. On **real** invoices the label only exists once the invoice is
*closed*. A real split must therefore also require "resolved as of the cutoff" —
otherwise you would train on invoices whose outcome had not happened yet. The
synthetic setup sidesteps this; a production version cannot.

---

*Companion docs: `README.md` · `PROJECT.md` · `MODEL_TRAINING.md` · `RAZORPAY_API.md`*
