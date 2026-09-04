"""Risk cascade + diagnosis.

Two models, run in sequence:

  Model 1  train_risk_model   - logistic classifier, P(invoice paid beyond terms).
                                Screening layer.
  Model 2  train_delay_model  - gradient-boosted regressor, expected days beyond
                                terms.  Trained ONLY on late invoices, and only
                                *run* when Model 1 clears config.RISK_THRESHOLD.

make_risk_fn() wires them into one closure the engine calls per audit row; it
returns {p_late, expected_delay_days, severity, segment}.  Predictions are
informational - the collections ladder is still a pure function of *observed*
days-beyond-terms.

Data is synthetic, so both models partly recover their own generator; the claim
on camera is the pipeline, not the score.

diagnose()   - one of four labels.  Rule-based by default; if a local Ollama
               model is reachable it is asked instead, with the rule result as
               the fallback.  No hosted API, ever.
"""
from __future__ import annotations

import datetime as dt
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (mean_absolute_error, precision_recall_fscore_support,
                             roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import config

HOLDOUT_WEEKS = 6

# Committed model artifacts. Trained by models/model1.ipynb / models/Model2.ipynb;
# load_or_train_models() re-fits with the identical recipe when they are absent or
# unloadable (e.g. a scikit-learn version skew on the pickle).
MODELS_DIR = config.ROOT / "models"
M1_PATH = MODELS_DIR / "model1_logistic_regression.joblib"
M2_PATH = MODELS_DIR / "model2.joblib"
_LABELS = ("cashflow", "process", "dispute", "stretch")
_CACHE_PATH = config.DATA_DIR / "diagnosis_cache.json"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


# --------------------------------------------------------------------------- #
# Model 1 - late-payment classifier / Model 2 - expected-delay regressor
# --------------------------------------------------------------------------- #

def train_test_split_by_date(ledger: dict[str, pd.DataFrame]):
    """Split invoices strictly by issue_date. No row from the test window may
    predate any training row -> no holdout leakage."""
    inv = ledger["invoices"].copy()
    inv["issue_date"] = pd.to_datetime(inv["issue_date"])
    cutoff = inv.issue_date.max() - pd.Timedelta(weeks=HOLDOUT_WEEKS)
    train = inv[inv.issue_date <= cutoff].copy()
    test = inv[inv.issue_date > cutoff].copy()
    return train, test


def _features(inv: pd.DataFrame, buyers: pd.DataFrame) -> pd.DataFrame:
    b = buyers.set_index("buyer_id")
    med_amt = inv.groupby("buyer_id").amount.median()
    due = pd.to_datetime(inv.due_date)
    f = pd.DataFrame({
        "buyer_dbt_mean": inv.buyer_id.map(b.dbt_mean),
        "buyer_dbt_sd": inv.buyer_id.map(b.dbt_sd),
        "buyer_dispute_rate": inv.buyer_id.map(b.dispute_rate),
        "buyer_promise_keep": inv.buyer_id.map(b.promise_keep_rate),
        "amount_rel": inv.amount.values / inv.buyer_id.map(med_amt).values,
        "terms": inv.terms.values,
        "quarter_end": ((due.dt.month.isin([3, 6, 9, 12])) & (due.dt.day >= 21)).astype(int).values,
    }, index=inv.index)
    return f


# --- Model 1: late-payment classifier -------------------------------------- #
# Recipe from models/model1.ipynb: StandardScaler -> LogisticRegression, the 7
# features in _features(), target `natural_dbt > 0`, 6-week time split.

def _eval_clf(model, ledger: dict[str, pd.DataFrame]) -> dict:
    train, test = train_test_split_by_date(ledger)
    buyers = ledger["buyers"]
    Xte, yte = _features(test, buyers), (test.natural_dbt > 0).astype(int)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, pred, average="binary", zero_division=0)
    return {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "auc": round(float(roc_auc_score(yte, proba)), 3) if yte.nunique() > 1 else None,
        "precision": round(float(prec), 3), "recall": round(float(rec), 3),
        "f1": round(float(f1), 3),
        "note": "synthetic data - the model partly recovers its own generator",
    }


def train_risk_model(ledger: dict[str, pd.DataFrame]):
    """Model 1 - classifier for P(invoice paid beyond terms)."""
    train, _ = train_test_split_by_date(ledger)
    buyers = ledger["buyers"]
    Xtr, ytr = _features(train, buyers), (train.natural_dbt > 0).astype(int)
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(random_state=config.SEED, max_iter=1000))
    model.fit(Xtr, ytr)
    return model, _eval_clf(model, ledger)


# --- Model 2: expected-delay regressor ----------------------------------------- #
# Recipe from models/Model2.ipynb: GradientBoostingRegressor(n_estimators=100,
# learning_rate=0.03, max_depth=2) - no scaler (trees don't need one). Trained
# ONLY on invoices that were actually late (natural_dbt > 0): forcing one
# regression to also fit the on-time zeros is a poor zero-inflated fit.

def _eval_reg(model, ledger: dict[str, pd.DataFrame]) -> dict:
    train, test = train_test_split_by_date(ledger)
    buyers = ledger["buyers"]
    tr, te = train[train.natural_dbt > 0], test[test.natural_dbt > 0]
    Xte, yte = _features(te, buyers), te.natural_dbt.astype(float)
    pred = model.predict(Xte)
    baseline = np.full(len(yte), tr.natural_dbt.astype(float).mean())
    return {
        "n_train_late": int(len(tr)), "n_test_late": int(len(te)),
        "mae_days": round(float(mean_absolute_error(yte, pred)), 1) if len(te) else None,
        "baseline_mae_days": round(float(mean_absolute_error(yte, baseline)), 1) if len(te) else None,
        "note": "synthetic data - the model partly recovers its own generator",
    }


def train_delay_model(ledger: dict[str, pd.DataFrame]):
    train, _ = train_test_split_by_date(ledger)
    buyers = ledger["buyers"]
    tr = train[train.natural_dbt > 0]
    Xtr, ytr = _features(tr, buyers), tr.natural_dbt.astype(float)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.03,
                                      max_depth=2, random_state=config.SEED)
    model.fit(Xtr, ytr)
    return model, _eval_reg(model, ledger)


# --- load the committed artifacts, or re-fit with the same recipe ------------- #

def load_or_train_models(ledger: dict[str, pd.DataFrame], *, retrain: bool = False):
    """Return (clf, reg, m1_metrics, m2_metrics).

    Loads models/model1_logistic_regression.joblib + models/model2.joblib when
    present and loadable; otherwise (or with retrain=True) re-fits both with the
    notebook recipe and writes fresh artifacts back to models/. Metrics are
    always recomputed against this ledger's 6-week holdout so the number on the
    dashboard reflects the model actually in use.
    """
    if not retrain and M1_PATH.exists() and M2_PATH.exists():
        try:
            clf, reg = joblib.load(M1_PATH), joblib.load(M2_PATH)
            m1, m2 = _eval_clf(clf, ledger), _eval_reg(reg, ledger)
            m1["source"] = f"loaded {M1_PATH.name}"
            m2["source"] = f"loaded {M2_PATH.name}"
            return clf, reg, m1, m2
        except Exception as exc:                       # noqa: BLE001
            print(f"[agent] could not load committed models ({exc!r}); re-fitting from recipe")

    clf, m1 = train_risk_model(ledger)
    reg, m2 = train_delay_model(ledger)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, M1_PATH)
    joblib.dump(reg, M2_PATH)
    m1["source"] = m2["source"] = "re-fitted from notebook recipe"
    return clf, reg, m1, m2


def _row_features(state, b: pd.DataFrame, med_amt: dict[str, float] | None = None) -> pd.DataFrame:
    """One feature row for a live invoice state. `b` is buyers indexed by buyer_id."""
    due = pd.Timestamp(state.due_date)
    buyer_median = (med_amt or {}).get(state.buyer_id)
    if buyer_median and buyer_median > 0:
        amount_rel = float(state.amount) / float(buyer_median)
    else:
        amount_rel = 1.0

    return pd.DataFrame([{
        "buyer_dbt_mean": b.dbt_mean.get(state.buyer_id, 0.0),
        "buyer_dbt_sd": b.dbt_sd.get(state.buyer_id, 0.0),
        "buyer_dispute_rate": b.dispute_rate.get(state.buyer_id, 0.0),
        "buyer_promise_keep": b.promise_keep_rate.get(state.buyer_id, 0.5),
        "amount_rel": amount_rel,
        "terms": state.terms,
        "quarter_end": int(due.month in (3, 6, 9, 12) and due.day >= 21),
    }])


def make_risk_fn(clf, reg, buyers: pd.DataFrame, invoices: pd.DataFrame | None = None,
                 threshold: float = config.RISK_THRESHOLD):
    """The cascade as one closure: state -> {p_late, expected_delay_days, severity, segment}.

    Model 2 (reg) is only evaluated when Model 1 (clf) clears `threshold`.
    """
    b = buyers.set_index("buyer_id")
    med_amt = None
    if invoices is not None and "buyer_id" in invoices and "amount" in invoices:
        med_amt = invoices.groupby("buyer_id").amount.median().to_dict()

    def risk(state) -> dict:
        X = _row_features(state, b, med_amt)
        p = round(float(clf.predict_proba(X)[:, 1][0]), 3)
        if p < threshold:
            return {"p_late": p, "expected_delay_days": None,
                    "severity": "monitor", "segment": "on_track"}
        d = max(0.0, float(reg.predict(X)[0]))
        severity = "mild" if d < 7 else "moderate" if d < 21 else "severe"
        segment = "at_risk" if p < 0.7 else "slipping" if d < 21 else "overdue"
        return {"p_late": p, "expected_delay_days": round(d, 1),
                "severity": severity, "segment": segment}

    return risk


# --------------------------------------------------------------------------- #
# diagnosis
# --------------------------------------------------------------------------- #

def diagnose_rule(state, ctx: dict, buyer: pd.Series | None = None) -> str:
    """Deterministic fallback. Reads the buyer profile + invoice context."""
    if buyer is None:
        return "stretch"
    dbt = ctx.get("dbt", 0)
    exp = ctx.get("expected_delay")           # Model 2 output, may be None
    if buyer.get("dispute_rate", 0) >= 0.15:
        return "dispute"
    if buyer.get("qend_squeeze", 0) >= 0.4:
        return "cashflow"
    if exp is not None and exp < 7 and buyer.get("dbt_mean", 0) <= 6 and ctx.get("touches", 0) == 0:
        return "process"
    if buyer.get("dbt_mean", 0) <= 5 and ctx.get("touches", 0) == 0 and dbt <= 10:
        return "process"
    if exp is not None and exp >= 21 and buyer.get("dispute_rate", 0) < 0.15:
        return "stretch"
    if buyer.get("promise_keep_rate", 1) < 0.6 or buyer.get("dbt_mean", 0) > 12:
        return "stretch"
    return "cashflow"


def _ollama(prompt: str, timeout: float = 8.0) -> str | None:
    try:
        import requests

        r = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                  "options": {"temperature": 0, "seed": config.SEED}},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception:                       # noqa: BLE001 - never block the pipeline
        return None


_EMAIL_CACHE_PATH = config.DATA_DIR / "email_cache.json"


def _load_cache(path=_CACHE_PATH) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_cache(c: dict, path=_CACHE_PATH) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(c, indent=0))


def make_diagnose_fn(buyers: pd.DataFrame, use_ollama: bool = True):
    b = buyers.set_index("buyer_id")
    cache = _load_cache()

    def diagnose(state, ctx: dict) -> str:
        buyer = b.loc[state.buyer_id] if state.buyer_id in b.index else None
        rule = diagnose_rule(state, ctx, buyer)
        key = f"{state.invoice_id}:{ctx.get('dbt')}"
        if key in cache:
            return cache[key]
        label = rule
        if use_ollama and buyer is not None:
            prompt = (
                "You classify why a B2B invoice is being paid late. "
                f"Buyer mean days-late={buyer.dbt_mean}, dispute_rate={buyer.dispute_rate}, "
                f"promise_kept_rate={buyer.promise_keep_rate}, quarter_end_squeeze={buyer.qend_squeeze}. "
                f"This invoice is {ctx.get('dbt')} days beyond terms with {ctx.get('touches')} prior touches. "
                f"Answer with exactly one word from: {', '.join(_LABELS)}."
            )
            out = _ollama(prompt)
            if out:
                word = out.lower().split()[0].strip(".,")
                if word in _LABELS:
                    label = word
        cache[key] = label
        return label

    diagnose.flush = lambda: _save_cache(cache)      # call once after a run
    return diagnose


def draft_justification(facts: dict) -> str:
    """One paragraph for the leverage brief. Template fallback; Ollama if reachable."""
    if facts["mean_dbt"] <= 3:
        template = (
            f"{facts['buyer']} has settled reliably - {facts['honored']} of "
            f"{facts['honored'] + facts['broken']} invoices on or before the due date, "
            f"mean {facts['mean_dbt']:+.1f} days beyond terms, dispute rate "
            f"{facts['dispute_rate']:.0%}. The current terms are working; the record "
            f"is here to keep the relationship on the front foot at renewal."
        )
    else:
        template = (
            f"{facts['buyer']} agreed to Net {facts['terms']} but has settled "
            f"{facts['broken']} of {facts['honored'] + facts['broken']} invoices late, "
            f"averaging {facts['mean_dbt']:+.1f} days beyond terms "
            f"(dispute rate {facts['dispute_rate']:.0%}). "
            f"Peers on the same terms settle at {facts['peer_dbt']:+.1f} days. "
            f"The recommended terms neutralise the observed drift while keeping the "
            f"relationship intact; the honoured/broken record above is the evidence to present."
        )
    prompt = (
        "Write one persuasive but professional paragraph (<=90 words) a supplier "
        "would send to a large buyer to justify revising payment terms. Use only these facts: "
        + json.dumps(facts) + " Do not invent numbers."
    )
    out = _ollama(prompt, timeout=12.0)
    return out or template


# --------------------------------------------------------------------------- #
# the buyer-facing email
# --------------------------------------------------------------------------- #

# stage -> the intent that drives tone + the ask
_STAGE_INTENT = {
    0: "a courtesy reminder ahead of the due date",
    1: "a polite first follow-up now that the due date has passed",
    2: "a firm reminder; escalate to the AP lead",
    3: "an early-settlement offer",
    4: "propose a short payment plan",
}


def _email_template(facts: dict) -> str:
    """Deterministic fallback body (used when Ollama is unreachable)."""
    ap = facts.get("ap_first") or "there"
    stretch3 = (f"It is {facts['dbt']} days past terms again - please confirm payment today, "
                f"or we will escalate to your AP lead.")
    ask = {
        0: f"This is a reminder that it falls due on {facts['due_date']}.",
        1: f"It is now {facts['dbt']} days past the {facts['due_date']} due date - could you confirm a payment date?",
        2: f"It is {facts['dbt']} days beyond the Net {facts['terms']} terms. We are escalating this to your AP lead and would appreciate an update today.",
        3: facts.get("discount_line") or (stretch3 if facts.get("diagnosis") == "stretch"
            else f"It is {facts['dbt']} days overdue; we can offer an early-settlement discount if it is cleared this week."),
        4: f"It is {facts['dbt']} days overdue. We would like to agree a short payment plan - happy to set one up on a call.",
    }.get(facts["stage"], f"It is {facts['dbt']} days beyond terms.")
    lines = [
        f"Hi {ap},",
        "",
        f"Invoice {facts['invoice_id']} for {facts['amount_inr']} (Net {facts['terms']}, "
        f"due {facts['due_date']}). {ask}",
    ]
    if facts.get("pay_url"):
        lines += ["", f"Pay here: {facts['pay_url']}"]
    lines += ["", f"{facts['signatory']}", facts["sme_name"]]
    return "\n".join(lines)


def compose_email(facts: dict) -> str:
    """Personalised buyer email. LLM writes it from `facts`; template on fallback."""
    intent = _STAGE_INTENT.get(facts["stage"], "a reminder")
    if facts["stage"] == 3 and facts.get("diagnosis") == "stretch":
        intent = "a firm notice that the invoice is past terms again; ask for payment today or escalate"
    prompt = (
        "You write short, professional accounts-receivable emails for a supplier chasing "
        "a late B2B invoice. Write ONLY the email body, at most 110 words.\n"
        f"- Address the buyer contact by first name ({facts.get('ap_first') or 'the AP contact'}).\n"
        f"- State the invoice number, amount, terms and due date exactly as given.\n"
        f"- Tone and ask must match this intent: {intent}.\n"
        "- The internal diagnosis below guides WHAT to say; never name it or mention any "
        "internal analysis, model, score, or negotiation brief.\n"
        "- Keep every number and the payment URL verbatim. Invent nothing.\n"
        f"- Sign off as: {facts['signatory']}, {facts['sme_name']}.\n"
        "Facts: " + json.dumps({k: v for k, v in facts.items() if k != "sme_name"})
    )
    out = _ollama(prompt, timeout=15.0)
    if out and facts["invoice_id"] in out and (not facts.get("pay_url") or facts["pay_url"] in out):
        return out
    return _email_template(facts)


def make_compose_fn(use_ollama: bool = True):
    """Closure the engine calls per touch. Caches by (invoice_id, stage); temp 0."""
    cache = _load_cache(_EMAIL_CACHE_PATH)

    def compose(facts: dict) -> str:
        key = f"{facts['invoice_id']}:{facts['stage']}"
        if key in cache:
            return cache[key]
        body = compose_email(facts) if use_ollama else _email_template(facts)
        cache[key] = body
        return body

    compose.flush = lambda: _save_cache(cache, _EMAIL_CACHE_PATH)
    return compose


if __name__ == "__main__":
    import ledger

    L = ledger.generate_ledger()
    _, _, m1, m2 = load_or_train_models(L, retrain="--retrain" in __import__("sys").argv)
    print("Model 1 (late classifier):", m1)
    print("Model 2 (delay regressor):", m2)
