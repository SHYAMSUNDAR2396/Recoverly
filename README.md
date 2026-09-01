# Recoverly

**Razorpay AI Buildathon 2026 · Track 03 — AI Revenue Recovery**
Source problem: *"Why can't SMEs negotiate favorable payment terms with large buyers?"*

A bounded, autonomous collections loop over every open invoice, whose output is a
**negotiating position** the SME can take into a room. Collections is the evidence
engine; the leverage brief is the product.

See `PROJECT.md` for the full feature list and architecture.

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Run

```bash
./.venv/bin/python run.py              # ledger -> risk model -> engine -> metrics -> results.duckdb
./.venv/bin/python run.py --no-llm     # skip Ollama; rule-based diagnosis only
./.venv/bin/python run.py --fresh      # regenerate the committed ledger

./.venv/bin/uvicorn api:app --reload   # read-only API over results.duckdb  (http://127.0.0.1:8000/docs)
./.venv/bin/python -m pytest -q        # 26 tests
```

Local LLM (optional): `ollama pull llama3.1:8b` then `ollama serve`. The pipeline
never blocks on it — `diagnose()` falls back to a rule-based classifier and
`draft_justification()` to a template.

## Modules

| File | Responsibility |
|---|---|
| `config.py` | Every constant. SEED, ladder thresholds, `BOUNDS` limits, `RESPONSE_LIFT`. |
| `ledger.py` | Synthetic ledger: buyers, ~250 invoices, dated event stream. Seeded, committed as parquet under `data/`. |
| `engine.py` | The day-stepping clock: ladder, `BOUNDS`, split stop conditions, promise loop, audit. |
| `agent.py` | **Model 1** `train_risk_model` — logistic classifier, P(late). **Model 2** `train_delay_model` — gradient-boosted regressor, expected days late, trained on late invoices only, run only when Model 1 clears `RISK_THRESHOLD`. `make_risk_fn` wires the cascade. `diagnose()` — 4 labels, rule-based + optional Ollama. Predictions are informational; the ladder still reads observed DBT only. |
| `brief.py` | Per-buyer leverage brief. Deterministic recommended terms, generative justification. |
| `razorpay_link.py` | Payment-link adapter. `live=False` simulated; `live=True` one real test-mode call. |
| `metrics.py` | Treatment vs control, cash pulled forward, net benefit, exception list. |
| `run.py` | Orchestrator → `results.duckdb`. |
| `api.py` | FastAPI, read-only `GET`. `/invoices` `/metrics` `/audit` `/exceptions` `/buyers` `/buyers/{id}/brief`. |
| `test_engine.py` | 26 tests: the 3 silent-failure gaps, every `BOUNDS` predicate, the ladder boundaries, the 4 demo beats, determinism. |
| `web/` | Vite + React SPA (not built yet). |

## On the numbers

The data is synthetic, so both risk models partly recover their own generator —
Model 1 AUC ~0.87, Model 2 MAE ~6.5 days vs a ~7.0-day mean baseline. The claim
is the two-stage pipeline, not the scores.

The treatment effect exists **only** because of the authored, benchmark-anchored
`RESPONSE_LIFT` assumption in `config.py`. The control group proves the
measurement machinery is correct; it does not prove the agent works. Headline
result: DSO −3 days from the free reminder rungs; the stage-3 discount rung runs
near break-even and is a tuning lever.
