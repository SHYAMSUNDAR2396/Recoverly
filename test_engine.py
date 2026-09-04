"""~15 tests. Targets the silent-failure paths and the demo beats, not coverage.

Run: ./.venv/bin/python -m pytest -q
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

import config
import engine
import ledger

# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def L():
    return ledger.generate_ledger()


@pytest.fixture(scope="module")
def run_result(L):
    return engine.run(L)


def _mini(invoices, events=None):
    """Build a hand-made ledger. `invoices` rows are dicts with sensible defaults."""
    buyers = pd.DataFrame([dict(
        buyer_id="BUY-X", name="Test Co", tier="light", dbt_mean=10, dbt_sd=3,
        qend_squeeze=0.0, partial_rate=0.0, dispute_rate=0.0, will_opt_out=False,
        promise_rate=0.0, promise_keep_rate=0.5, responsiveness=0.5,
        ap_contact="Test Contact", email="ap@test-co.example", phone="+919800000000",
    )])
    base = dict(buyer_id="BUY-X", amount=200_000.0, terms=30, group="treatment",
                disputed=False, held=False)
    inv_rows = []
    for d in invoices:
        row = {**base, **d}
        row.setdefault("issue_date", row["due_date"] - dt.timedelta(days=row["terms"]))
        row.setdefault("natural_pay_date", row["due_date"] + dt.timedelta(days=999))
        row.setdefault("natural_dbt", 999)
        row.setdefault("lift_roll", 0.99)          # lift never lands unless a test wants it
        inv_rows.append(row)
    ev = pd.DataFrame(events or [], columns=["invoice_id", "buyer_id", "day", "kind", "promised_date"])
    return {"buyers": buyers, "invoices": pd.DataFrame(inv_rows), "events": ev}


# --------------------------------------------------------------------------- #
# CRITICAL GAP 1 - the response model must never touch the control group
# --------------------------------------------------------------------------- #

def test_control_group_pays_exactly_on_natural_date(run_result, L):
    _, fin = run_result
    ctrl = fin[(fin.group == "control") & fin.paid & fin.paid_day.notna()].copy()
    ctrl["nat"] = pd.to_datetime(ctrl.natural_pay_date)
    ctrl["pay"] = pd.to_datetime(ctrl.paid_day)
    ctrl = ctrl[ctrl.nat <= pd.Timestamp(config.SIM_END)]
    # every control invoice settles on its natural date - no acceleration, ever
    delta = (ctrl.pay - ctrl.nat).dt.days
    assert (delta == 0).all(), f"control invoices moved: {delta[delta != 0].to_dict()}"


def test_control_run_is_identical_with_and_without_agent(L):
    """Re-running with control invoices only must leave their pay dates unchanged."""
    _, fin_full = engine.run(L)
    ctrl_dates = fin_full[fin_full.group == "control"].set_index("invoice_id").paid_day
    # relabel everything control -> agent does nothing anywhere
    L2 = {**L, "invoices": L["invoices"].assign(group="control")}
    _, fin_none = engine.run(L2)
    none_dates = fin_none.set_index("invoice_id").paid_day.loc[ctrl_dates.index]
    assert ctrl_dates.equals(none_dates)


# --------------------------------------------------------------------------- #
# CRITICAL GAP 2 - holdout leakage in the risk cascade  (agent.py)
# --------------------------------------------------------------------------- #

def test_risk_model_has_no_holdout_leakage():
    agent = pytest.importorskip("agent")
    train, test = agent.train_test_split_by_date(ledger.generate_ledger())
    assert train.issue_date.max() < test.issue_date.min()


# --------------------------------------------------------------------------- #
# risk cascade - Model 2 runs only above the threshold; trained on late only
# --------------------------------------------------------------------------- #

def test_risk_cascade_gates_model2():
    agent = pytest.importorskip("agent")

    class Clf:
        def __init__(self, p): self.p = p
        def predict_proba(self, X): return np.array([[1 - self.p, self.p]])

    class Reg:
        def predict(self, X): return np.array([18.0])

    buyers = pd.DataFrame([dict(
        buyer_id="BUY-X", dbt_mean=10, dbt_sd=3, dispute_rate=0.0, promise_keep_rate=0.5)])
    s = _state()
    low = agent.make_risk_fn(Clf(0.10), Reg(), buyers, threshold=0.5)(s)
    hi = agent.make_risk_fn(Clf(0.90), Reg(), buyers, threshold=0.5)(s)
    assert low["expected_delay_days"] is None and low["segment"] == "on_track"
    assert hi["expected_delay_days"] == 18.0 and hi["severity"] == "moderate"


def test_risk_cascade_computes_dynamic_amount_rel():
    agent = pytest.importorskip("agent")

    captured_X = []
    class Clf:
        def predict_proba(self, X):
            captured_X.append(X.copy())
            return np.array([[0.5, 0.5]])

    class Reg:
        def predict(self, X):
            return np.array([10.0])

    buyers = pd.DataFrame([dict(
        buyer_id="BUY-X", dbt_mean=10, dbt_sd=3, dispute_rate=0.0, promise_keep_rate=0.5)])
    invoices = pd.DataFrame([
        dict(buyer_id="BUY-X", amount=50_000.0),
        dict(buyer_id="BUY-X", amount=150_000.0),
    ])  # median amount is 100,000.0

    s = _state(amount=250_000.0)  # 250k / 100k = 2.5
    fn = agent.make_risk_fn(Clf(), Reg(), buyers, invoices=invoices)
    fn(s)

    assert len(captured_X) == 1
    assert np.isclose(captured_X[0]["amount_rel"].iloc[0], 2.5)


def test_delay_model_trains_on_late_only(L):
    agent = pytest.importorskip("agent")
    train, _ = agent.train_test_split_by_date(L)
    _, m2 = agent.train_delay_model(L)
    assert m2["n_train_late"] < len(train)                 # zeros were excluded
    assert m2["mae_days"] <= m2["baseline_mae_days"] + 1e-9  # beats predicting the mean


def test_load_or_train_models_yields_usable_cascade(L):
    agent = pytest.importorskip("agent")
    clf, reg, m1, m2 = agent.load_or_train_models(L)        # loads models/ or re-fits
    assert hasattr(clf, "predict_proba") and hasattr(reg, "predict")
    assert {"auc", "source"} <= set(m1) and {"mae_days", "source"} <= set(m2)
    hi = agent.make_risk_fn(clf, reg, L["buyers"], invoices=L["invoices"])(_state())
    assert 0.0 <= hi["p_late"] <= 1.0


# --------------------------------------------------------------------------- #
# CRITICAL GAP 3 - promise_kept_rate with zero promises is NULL, not 0.0
# --------------------------------------------------------------------------- #

def test_promise_kept_rate_null_for_buyer_with_no_promises():
    brief = pytest.importorskip("brief")
    rate = brief.promise_kept_rate(pd.DataFrame(columns=["kind", "promised_date"]))
    assert rate is None or (isinstance(rate, float) and np.isnan(rate))


# --------------------------------------------------------------------------- #
# ladder - each stage fires at its exact DBT boundary
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("stage,threshold", list(config.LADDER.items()))
def test_ladder_stage_boundary(stage, threshold):
    assert engine.ladder_stage(threshold) >= stage
    assert engine.ladder_stage(threshold - 1) < stage or stage == 0


def test_ladder_is_monotonic_in_dbt():
    stages = [engine.ladder_stage(d) for d in range(-10, 60)]
    assert stages == sorted(stages)


# --------------------------------------------------------------------------- #
# BOUNDS - every predicate, pass and fail, driven off the one list
# --------------------------------------------------------------------------- #

def _state(**kw):
    d = dict(invoice_id="INV-1", buyer_id="BUY-X", amount=100_000.0,
             due_date=dt.date(2026, 1, 1), terms=30, group="treatment",
             natural_pay_date=dt.date(2026, 3, 1), lift_roll=0.9, held=False)
    d.update(kw)
    return engine._State(**d)

_MONDAY = dt.date(2026, 1, 5)
_SATURDAY = dt.date(2026, 1, 3)

BOUND_CASES = {
    "max_touches":      (_state(touches=0), _state(touches=config.MAX_TOUCHES)),
    "min_gap_72h":      (_state(last_touch_day=_MONDAY - dt.timedelta(days=3)),
                         _state(last_touch_day=_MONDAY - dt.timedelta(days=1))),
    "business_hours":   (_state(), _state()),          # pass on Mon, fail on Sat (day arg)
    "discount_cap_2pc": (_state(proposed_discount=0.02), _state(proposed_discount=0.05)),
    "maker_checker":    (_state(amount=999_999.0), _state(amount=1_500_000.0)),
}

@pytest.mark.parametrize("name,pred", engine.BOUNDS)
def test_bound_pass_and_fail(name, pred):
    ok_state, bad_state = BOUND_CASES[name]
    day_ok = _SATURDAY if name == "business_hours" else _MONDAY  # dummy, overridden below
    if name == "business_hours":
        assert pred(ok_state, _MONDAY) is True
        assert pred(bad_state, _SATURDAY) is False
    else:
        assert pred(ok_state, _MONDAY) is True
        assert pred(bad_state, _MONDAY) is False


# --------------------------------------------------------------------------- #
# demo beat 1 - dispute -> agent stops, escalation audit row appears
# --------------------------------------------------------------------------- #

def test_dispute_stops_agent_and_writes_stage5_row():
    due = dt.date(2026, 2, 1)
    L = _mini(
        [dict(invoice_id="INV-D", due_date=due)],
        [("INV-D", "BUY-X", due + dt.timedelta(days=5), "dispute", pd.NaT)],
    )
    audit, fin = engine.run(L)
    assert bool(fin.set_index("invoice_id").loc["INV-D", "escalated"])
    rows = audit[audit.invoice_id == "INV-D"]
    esc = rows[rows.action_taken == "escalate_to_human"]
    assert len(esc) == 1
    assert esc.iloc[0].ladder_stage == config.TERMINAL_STAGE
    assert esc.iloc[0].diagnosis == "dispute_raised"
    # no touch after the dispute day
    assert (pd.to_datetime(rows.timestamp).dt.date <= due + dt.timedelta(days=5)).all()


# --------------------------------------------------------------------------- #
# demo beat 2 - promise-to-pay -> agent silent until the date, then resolves
# --------------------------------------------------------------------------- #

def test_promise_silences_agent_until_promised_date():
    due = dt.date(2026, 2, 1)
    promise_day = due + dt.timedelta(days=6)
    promised = due + dt.timedelta(days=16)
    L = _mini(
        [dict(invoice_id="INV-P", due_date=due,
              natural_pay_date=due + dt.timedelta(days=40))],
        [("INV-P", "BUY-X", promise_day, "promise", promised)],
    )
    audit, fin = engine.run(L)
    rows = audit[audit.invoice_id == "INV-P"].copy()
    rows["d"] = pd.to_datetime(rows.timestamp).dt.date
    # nothing between the promise day and the promised date
    quiet = rows[(rows.d > promise_day) & (rows.d < promised)]
    assert quiet.empty, f"agent spoke during the quiet window: {quiet.action_taken.tolist()}"
    # kept promise -> invoice settles on the promised date
    assert bool(fin.set_index("invoice_id").loc["INV-P", "paid"])
    assert fin.set_index("invoice_id").loc["INV-P", "paid_day"] == promised


def test_second_broken_promise_stops_honouring_promises():
    due = dt.date(2026, 2, 1)
    # two early promises break (~day 15-16); the third arrives afterwards
    L = _mini(
        [dict(invoice_id=f"INV-B{i}", due_date=due,
              natural_pay_date=due + dt.timedelta(days=120)) for i in range(3)],
        [("INV-B0", "BUY-X", due + dt.timedelta(days=3), "promise", pd.NaT),
         ("INV-B1", "BUY-X", due + dt.timedelta(days=4), "promise", pd.NaT),
         ("INV-B2", "BUY-X", due + dt.timedelta(days=40), "promise", pd.NaT)],
    )
    audit, _ = engine.run(L)
    broken = audit[audit.diagnosis == "promise_broken"]
    assert len(broken) >= config.BROKEN_PROMISES_TO_STOP
    # the third promise lands after the buyer is over the limit -> recorded, not honoured
    assert (audit.action_taken == "promise_noted_ignored").any()


# --------------------------------------------------------------------------- #
# demo beat 3 - a 5% discount ask -> refused, escalated, human gate set
# --------------------------------------------------------------------------- #

def test_five_percent_discount_is_refused_and_escalated():
    due = dt.date(2026, 1, 1)
    L = _mini([dict(invoice_id="INV-5", due_date=due,
                    natural_pay_date=due + dt.timedelta(days=120))])
    audit, fin = engine.run(L, force_discount={"INV-5": 0.05})
    rows = audit[audit.invoice_id == "INV-5"]
    esc = rows[rows.action_taken == "escalate_to_human"]
    assert len(esc) == 1
    assert bool(esc.iloc[0].human_gate_required)
    assert esc.iloc[0].diagnosis == "discount_over_authority"
    assert bool(fin.set_index("invoice_id").loc["INV-5", "escalated"])


def test_invoice_above_maker_checker_threshold_never_acts_autonomously():
    due = dt.date(2026, 1, 1)
    L = _mini([dict(invoice_id="INV-BIG", amount=config.MAKER_CHECKER_THRESHOLD + 1,
                    due_date=due, natural_pay_date=due + dt.timedelta(days=120))])
    audit, _ = engine.run(L)
    acts = audit[(audit.invoice_id == "INV-BIG") &
                 (~audit.action_taken.isin(["escalate_to_human"]))]
    assert acts.empty
    assert (audit.diagnosis == "above_maker_checker").any()


# --------------------------------------------------------------------------- #
# demo beat 4 - determinism contract
# --------------------------------------------------------------------------- #

def test_identical_seed_gives_identical_ledger():
    a = ledger.generate_ledger(force=True)
    b = ledger.generate_ledger(force=True)
    for k in ("buyers", "invoices", "events"):
        pd.testing.assert_frame_equal(a[k], b[k])


def test_engine_run_is_deterministic(L):
    a1, f1 = engine.run(L)
    a2, f2 = engine.run(L)
    pd.testing.assert_frame_equal(a1, a2)
    pd.testing.assert_frame_equal(f1, f2)


# --------------------------------------------------------------------------- #
# audit trail - shape and traceability
# --------------------------------------------------------------------------- #

def test_audit_has_full_schema(run_result):
    audit, _ = run_result
    expected = {"timestamp", "invoice_id", "buyer_id", "risk_score", "expected_delay_days",
                "risk_severity", "risk_segment", "diagnosis", "ladder_stage", "action_taken",
                "message_sent", "razorpay_object_id", "bounds_checked", "human_gate_required",
                "email_to", "email_body", "email_message_id", "outcome", "outcome_timestamp"}
    assert expected <= set(audit.columns)
    assert audit.outcome.isin({"paid", "handed_off", "unresolved"}).all()


def test_every_autonomous_action_recorded_its_bounds(run_result):
    audit, _ = run_result
    acted = audit[audit.action_taken.isin(
        ["pre_due_courtesy", "polite_follow_up", "firm_reminder",
         "early_settlement_offer", "payment_plan"])]
    assert acted.bounds_checked.str.contains("business_hours").all()


def test_business_hour_timestamps(run_result):
    audit, _ = run_result
    ts = pd.to_datetime(audit.timestamp)
    assert ts.dt.hour.between(config.BUSINESS_HOUR_START, config.BUSINESS_HOUR_END - 1).all()
    acted = audit[audit.action_taken.str.contains("reminder|courtesy|follow_up|offer|plan")]
    assert (pd.to_datetime(acted.timestamp).dt.weekday < 5).all()


# --------------------------------------------------------------------------- #
# buyer-facing email - composed on active rungs, never on a terminal stop
# --------------------------------------------------------------------------- #

_ACTED = ["pre_due_courtesy", "polite_follow_up", "firm_reminder",
          "early_settlement_offer", "payment_plan"]


def test_email_composed_for_active_rungs(run_result):
    audit, _ = run_result
    acted = audit[audit.action_taken.isin(_ACTED)]
    assert (acted.email_body.str.len() > 0).all()
    assert (acted.email_to.str.endswith(".example")).all()
    assert (acted.email_message_id.str.startswith("msg_sim_")).all()


def test_no_buyer_email_on_terminal_stop():
    due = dt.date(2026, 2, 1)
    L = _mini(
        [dict(invoice_id="INV-D", due_date=due),
         dict(invoice_id="INV-O", due_date=due)],
        [("INV-D", "BUY-X", due + dt.timedelta(days=5), "dispute", pd.NaT),
         ("INV-O", "BUY-X", due + dt.timedelta(days=5), "opt_out", pd.NaT)],
    )
    audit, _ = engine.run(L)
    esc = audit[audit.action_taken == "escalate_to_human"]
    assert len(esc) == 2
    assert (esc.email_body == "").all()


def test_email_body_keeps_link_and_amount(run_result):
    audit, _ = run_result
    linked = audit[audit.action_taken.isin(["pre_due_courtesy", "early_settlement_offer"])
                   & (audit.razorpay_object_id.notna())].head(20)
    assert len(linked)
    for r in linked.itertuples():
        assert "₹" in r.email_body
        # the internal diagnosis label / brief must never leak to the buyer
        low = r.email_body.lower()
        assert not any(w in low for w in ("cashflow", "stretch", "leverage brief", "model"))
        assert len(r.email_body) <= 1400


def test_email_deterministic(L):
    a1, _ = engine.run(L)
    a2, _ = engine.run(L)
    pd.testing.assert_series_equal(a1.email_body, a2.email_body)


def test_razorpay_link_accepts_customer_kwargs():
    import razorpay_link
    res = razorpay_link.create_link("INV-9", 250_000.0, live=False, with_url=True,
                                    customer={"name": "A", "email": "a@x.example", "contact": "+910"},
                                    notify={"sms": True, "email": False})
    assert res["id"].startswith("plink_sim_")
    assert res["short_url"].startswith("https://rzp.io/i/")
    assert isinstance(razorpay_link.create_link("INV-9", 250_000.0, live=False), str)


def test_notify_send_simulated():
    import notify
    mid = notify.send("ap@harbour.example", "INV-1: payment reminder", "body text")
    assert mid.startswith("msg_sim_")
