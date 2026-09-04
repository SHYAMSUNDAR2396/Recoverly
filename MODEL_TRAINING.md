# Training the two risk models

These are the two **scikit-learn** models in the risk cascade (`agent.py`).
They are *not* the LLM — the local Ollama model does diagnosis/drafting and is
used as-shipped.

| | Model 1 | Model 2 |
|---|---|---|
| Name | `train_risk_model` | `train_delay_model` |
| Type | `LogisticRegression` (classifier) | `GradientBoostingRegressor` |
| Answers | *Will this invoice pay late at all?* `P(late)` | *How many days late?* |
| Runs | on every open invoice | only when Model 1 ≥ `RISK_THRESHOLD` (0.5) |
| Trained on | all invoices in the training window | **late invoices only** (`natural_dbt > 0`) |
| Metric | AUC / precision / recall / F1 | MAE vs a mean-prediction baseline |

Both are retrained on every `run.py` invocation (~2 s total). Nothing is
persisted to disk unless you add it (step 6).

---

## 0. Prerequisites

```bash
cd /Users/shyamsundar/Documents/Recoverly
./.venv/bin/python -c "import ledger; ledger.generate_ledger()"   # ensures data/ledger/*.parquet exists
```

The ledger is the training data. One row per invoice:
`buyer_id, amount, issue_date, due_date, terms`, and `natural_dbt` — the days
that invoice was paid beyond terms **with no agent involved**. `natural_dbt` is
the label both models learn.

---

## 1. Build the feature matrix (shared by both models)

`agent._features(invoices, buyers)` turns each invoice into 7 numeric columns:

| Feature | Source |
|---|---|
| `buyer_dbt_mean`, `buyer_dbt_sd` | the buyer's historical lateness profile |
| `buyer_dispute_rate`, `buyer_promise_keep` | buyer behaviour |
| `amount_rel` | this invoice ÷ the buyer's median invoice |
| `terms` | Net 30 / 45 / 60 |
| `quarter_end` | 1 if due in the last 10 days of Mar/Jun/Sep/Dec |

No feature uses anything known only *after* the due date — that is what keeps the
model honest. Add a column here and both models pick it up automatically.

---

## 2. Split by date (no leakage)

`agent.train_test_split_by_date(ledger)`:

- Sort by `issue_date`.
- Last **6 weeks** (`agent.HOLDOUT_WEEKS`) → test set. Everything earlier → train.
- Guarantee: every training invoice was issued before every test invoice.
  `test_engine.py::test_risk_model_has_no_holdout_leakage` asserts this.

On the committed ledger: **167 train / 86 test**.

---

## 3. Train Model 1 — late-payment classifier

`agent.train_risk_model(ledger)`:

1. `X_train = _features(train)`, `y_train = (train.natural_dbt > 0)` — binary.
2. Pipeline `StandardScaler → LogisticRegression(max_iter=1000)`, `.fit(...)`.
3. Predict probabilities on the test set, threshold at 0.5, compute
   **AUC / precision / recall / F1**.
4. Returns `(model, metrics)`.

Expected: `auc ≈ 0.87, precision ≈ 0.82, recall ≈ 0.96`.

Because the data is synthetic the model partly recovers its own generator — say
so on camera. The claim is the pipeline, not the AUC.

---

## 4. Train Model 2 — expected-delay regressor

`agent.train_delay_model(ledger)`:

1. Same date split, then **filter both sets to `natural_dbt > 0`** — Model 2 only
   ever sees invoices that were actually late (**129 train / 68 test**). This is
   the whole point of the two-model design: a single regression forced to also
   fit ~150 on-time zeros predicts magnitude badly (zero-inflated target).
2. `X_train = _features(late_train)`, `y_train = late_train.natural_dbt`.
3. Pipeline `StandardScaler → GradientBoostingRegressor(random_state=42)`, `.fit(...)`.
4. Predict on `late_test`; compute **MAE** and **baseline MAE** (predicting the
   training-set mean delay for everyone).
5. Returns `(model, metrics)`.

Expected: `mae_days ≈ 6.5` vs `baseline_mae_days ≈ 7.0` — it beats the naive
baseline. `test_engine.py::test_delay_model_trains_on_late_only` asserts both the
late-only filter and that MAE ≤ baseline.

---

## 5. Wire them into the cascade

`agent.make_risk_fn(clf, reg, buyers, threshold=config.RISK_THRESHOLD)` returns
one closure the engine calls per audit row:

```
p = clf.predict_proba(features)              # Model 1
if p < RISK_THRESHOLD (0.5):
    return {p_late: p, expected_delay_days: None,
            severity: "monitor", segment: "on_track"}    # Model 2 never runs
d = max(0, reg.predict(features))            # Model 2, only reached here
severity = "mild" if d < 7 else "moderate" if d < 21 else "severe"
segment  = "at_risk" if p < 0.7 else "slipping" if d < 21 else "overdue"
```

The prediction is **informational only** — the collections ladder rung is still a
pure function of *observed* days-beyond-terms. The signals feed the diagnosis
prompt, the severity label, and the dashboard.

---

## 6. Train + use everything in one shot

```bash
./.venv/bin/python agent.py            # trains both, prints metrics only
./.venv/bin/python run.py --no-llm     # trains both, runs the sim, writes results.duckdb
```

`run.py` stores both models' metrics in `results.duckdb`
(`meta.model1_metrics_json`, `meta.model2_metrics_json`). The dashboard header
reads them: `M1 AUC 0.87 · M2 MAE 6.5d`.

### Persisting the trained models (optional)

Models retrain on every `run.py` call (fine for the demo). To train once and
reuse:

```python
import joblib
joblib.dump(clf, "data/model1.joblib")
joblib.dump(reg, "data/model2.joblib")
# later
clf = joblib.load("data/model1.joblib")
reg = joblib.load("data/model2.joblib")
```

---

## 7. Tuning knobs

| Want to change | Where |
|---|---|
| When Model 2 fires | `config.RISK_THRESHOLD` (0.5) |
| Holdout window | `agent.HOLDOUT_WEEKS` (6) |
| Model 2 estimator / hyperparams | `train_delay_model` — swap `GradientBoostingRegressor(...)` args, or `Ridge()` |
| Features | `agent._features` (add a column → both models use it) |
| Severity / segment cutoffs | `agent.make_risk_fn` |
| Verify nothing broke | `./.venv/bin/python -m pytest -q` (28 tests) |
