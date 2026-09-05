# Recoverly

**A receivables leverage agent for SMEs.**
Built for the **Razorpay AI Buildathon 2026 · Track 03 — AI Revenue Recovery**.

![Razorpay AI Buildathon 2026](https://img.shields.io/badge/Razorpay%20AI%20Buildathon-2026%20Track%2003-1364F1)
![Python 3](https://img.shields.io/badge/python-3-blue)
![Tests](https://img.shields.io/badge/tests-36%20passing-brightgreen)
![License](https://img.shields.io/badge/license-unpublished-lightgrey)

Source problem: *"Why can't SMEs negotiate favorable payment terms with large buyers?"* —
[Fix My Itch](https://razorpay.com) score **82.8**, the highest-scored B2B Services
problem on Razorpay's own board.

---

## The pitch

Large buyers pay SMEs late because they can, and SMEs have no leverage and no time to
chase. Recoverly runs a bounded, fully-audited **autonomous collections loop** over
every open invoice — then turns what that loop learns into a **negotiating position**
the SME can take into a room:

> "Buyer X honored Net 60 twice and broke it four times — here is what we propose now,
> and here is the evidence."

**Collections is not the product. It's the evidence engine. The leverage is the
product.**

---

## Quick links

| | |
|---|---|
| 📄 Full feature list + architecture | [`PROJECT.md`](PROJECT.md) |
| 🧮 The two-model risk cascade, explained | [`MODEL_TRAINING.md`](MODEL_TRAINING.md) |
| 🗄️ Dataset / entity schema | [`DATA_MODEL.md`](DATA_MODEL.md) |
| 💳 Razorpay APIs used (and not used) | [`RAZORPAY_API.md`](RAZORPAY_API.md) |
| 🌐 Landing page | [`landing/index.html`](landing/index.html) |
| ▶️ Demo video | *link goes here once recorded* |

---

## Features

- **A two-model risk cascade.** A logistic classifier screens every invoice for
  `P(pays beyond terms)`; a gradient-boosted regressor estimates *how many days
  late*, but only runs for invoices the classifier flags. Informational only — see
  below.
- **A deterministic collections ladder.** Days-beyond-terms — not the model, not the
  LLM — picks the rung: courtesy reminder → firm reminder → early-settlement offer →
  payment plan → stop. Six stages, one pure function.
- **Local diagnosis and drafting.** A local Ollama model (no API key, no network
  egress) labels *why* an invoice is slipping and writes the buyer email in the right
  tone — with a deterministic template fallback so the pipeline never blocks on it.
- **Five hard bounds, defined once.** Max touches, minimum gap between touches,
  business hours only, a discount ceiling, and a maker-checker threshold — consumed
  by the engine, the tests, and the audit trail alike. Ask for a 5% discount and it
  refuses and escalates to a human.
- **A promise-to-pay feedback loop.** The agent goes quiet when a buyer promises to
  pay, and stops trusting a buyer's promises after the second broken one.
- **One real payment link.** Every batch invoice is simulated; exactly one Razorpay
  **test-mode** Payment Link is created live, carrying the buyer's contact so
  Razorpay texts them and auto-reminds until paid.
- **A dry-run buyer email.** Every touch composes a full, personalized email
  (LLM-written, template fallback) and "sends" it through a mock mailer that logs
  instead of transmitting — the message is real, the delivery is a stub you can wire
  up.
- **The leverage brief — the differentiator.** For any buyer with 20+ invoices: honored
  vs broken terms, mean days beyond terms, promise-kept rate (`NULL`, never a false
  `0.0`, for a buyer who's never promised), dispute rate, a **deterministic**
  recommended-terms rule, and a **generative** justification paragraph.
- **Honest measurement.** A 70/30 treatment/control split. The treatment effect is
  produced by a named, benchmark-anchored assumption (`RESPONSE_LIFT`), shown on
  screen as an assumption — the control group proves the measurement machinery is
  sound, not that the agent works.
- **A full audit trail.** One row per agent action — risk score, diagnosis, ladder
  stage, bounds checked, message, outcome. Pick any row, trace exactly why it
  happened.

---

## Quickstart

```bash
git clone https://github.com/SHYAMSUNDAR2396/Recoverly.git
cd Recoverly
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in RAZORPAY_KEY_ID/SECRET and SMTP_USER/PASSWORD if you
                       # want the live-link/demo-email paths; leave blank otherwise
```

`.env` is gitignored and loaded automatically by both `run.py` and `api.py` — fill it in
once and every terminal/process picks it up, no more manual `export` per shell.

Run the full pipeline (ledger → risk cascade → collections engine → metrics →
`results.duckdb`):

```bash
./.venv/bin/python run.py
```

Serve the dashboard:

```bash
./.venv/bin/uvicorn api:app --reload      # read-only API, http://127.0.0.1:8000/docs
cd web && npm install && npm run dev      # React SPA, http://localhost:5173
```

Open the landing page — no build step, just a file:

```bash
open landing/index.html
# or: python3 -m http.server --directory landing
```

Run the tests:

```bash
./.venv/bin/python -m pytest -q           # 36 tests
```

### Useful `run.py` flags

| Flag | Effect |
|---|---|
| *(none)* | Loads the committed model artifacts from `models/`, runs the full simulation |
| `--no-llm` | Skip Ollama — rule-based diagnosis, template emails, fully offline |
| `--fresh` | Regenerate the committed synthetic ledger |
| `--retrain` | Re-fit Model 1 + Model 2 and overwrite `models/*.joblib` |
| `--live-link INV-2032` | Create **one** real Razorpay test-mode payment link (needs `RAZORPAY_KEY_ID`/`RAZORPAY_KEY_SECRET`; invoice must be under the ₹5,00,000 test-mode cap) |
| `--live-link INV-2032 --demo-email you@gmail.com` | + actually **sends** that invoice's LLM-written email, containing the real link, to `you@gmail.com` over SMTP (needs `SMTP_USER`/`SMTP_PASSWORD` — a Gmail address + an [App Password](https://myaccount.google.com/apppasswords), not your normal password). Any failure falls back to dry-run; every other email in the run stays dry-run. |

Local LLM (optional, everything above works without it):

```bash
ollama pull llama3.1:8b
ollama serve
```

---

## Project structure

| Path | Responsibility |
|---|---|
| `config.py` | Every constant — `SEED`, ladder thresholds, `BOUNDS` limits, `RESPONSE_LIFT`, the risk-cascade threshold. |
| `ledger.py` | Synthetic ledger generator: ~253 invoices, 20 buyers, a dated event stream. Seeded, committed as parquet under `data/`. |
| `engine.py` | The day-stepping clock — ladder, `BOUNDS`, split stop conditions, promise loop, audit writes. |
| `agent.py` | **Model 1** (`train_risk_model`, logistic classifier) · **Model 2** (`train_delay_model`, gradient-boosted regressor, late-only) · `load_or_train_models` (loads `models/*.joblib`, re-fits on miss) · `make_risk_fn` cascade · `diagnose()` and `compose_email()` — rule/template-based with optional Ollama. |
| `models/` | Committed model artifacts (`model1_logistic_regression.joblib`, `model2.joblib`) and the notebooks that trained them. |
| `brief.py` | The per-buyer leverage brief. Deterministic recommendation, generative justification. |
| `razorpay_link.py` | Payment-link adapter. `live=False` → simulated; `live=True` → one real test-mode call, with `customer`/`notify`. |
| `notify.py` | Dry-run mailer — logs the composed buyer email instead of sending it. |
| `metrics.py` | Treatment vs control, cash pulled forward, net benefit, exception list. |
| `run.py` | Orchestrates all of the above → `results.duckdb`. |
| `api.py` | FastAPI, read-only `GET` over `results.duckdb` — `/invoices` `/metrics` `/audit` `/exceptions` `/buyers` `/buyers/{id}/brief`. One deliberate exception: `POST /demo/live-send` triggers one real Razorpay link + one real email for a chosen invoice/address (doesn't write to `results.duckdb`). |
| `web/` | Vite + React dashboard — Queue, Recovery results, Leverage brief. |
| `landing/` | Standalone marketing landing page (`index.html`) plus its design-canvas source. |
| `test_engine.py` | 36 tests — silent-failure gaps, every `BOUNDS` predicate, the risk-cascade gate, buyer-email rules, ladder boundaries, determinism. |

---

## Tech stack

Python · pandas · DuckDB · scikit-learn · a local Ollama LLM (no external API) ·
FastAPI · React (Vite) · pytest. The only outbound network call in the entire
project is the one Razorpay test-mode payment link.

## On the numbers

The data is synthetic, so both risk models partly recover their own generator —
Model 1 AUC ≈ 0.87, Model 2 MAE ≈ 5.6 days vs a ≈ 7.0-day mean baseline. The claim
is the two-stage pipeline, not the scores.

The treatment effect exists **only** because of the authored, benchmark-anchored
`RESPONSE_LIFT` assumption in `config.py`. The control group proves the measurement
machinery is correct; it does not prove the agent works. Headline result: DSO −3.1
days from the free reminder rungs; the stage-3 discount rung runs near break-even
and is a tuning lever, not hidden.

## License

Not yet published under a license — all rights reserved by the author pending the
buildathon submission.
