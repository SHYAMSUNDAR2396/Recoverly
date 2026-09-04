"""The simulated clock - the agent.

A day-stepping loop over the ledger. Everything except the loop is a pure
function called from it. That is what makes this an agent and not a batch
report.

  for day in date_range(SIM_START, SIM_END):
      apply_events(day)              # promises / disputes / opt-outs arrive HERE
      for inv in open_invoices(day):
          if terminal stop:  escalate_to_human() -> audit (stage 5); continue
          if silent hold:    continue
          stage = ladder_stage(DBT)                 # pure fn of DBT + state
          if stage advanced and bounds_ok:
              act() -> audit
      settle_payments(day)           # natural_pay_date + RESPONSE_LIFT (treatment)
"""
from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config
import notify
import razorpay_link

# --- hard bounds: one list, consumed by the engine, the tests and the audit ---
# predicate(state, day) -> True means "this bound is satisfied".
BOUNDS = [
    ("max_touches",      lambda s, d: s.touches < config.MAX_TOUCHES),
    ("min_gap_72h",      lambda s, d: s.last_touch_day is None
                                     or (d - s.last_touch_day).days >= config.MIN_GAP_DAYS),
    ("business_hours",   lambda s, d: d.weekday() < 5),
    ("discount_cap_2pc", lambda s, d: s.proposed_discount <= config.DISCOUNT_CAP + 1e-9),
    ("maker_checker",    lambda s, d: s.amount <= config.MAKER_CHECKER_THRESHOLD),
]
# a failed bound in this set escalates to a human; the rest just defer the touch
_ESCALATING_BOUNDS = {"max_touches", "discount_cap_2pc", "maker_checker"}

_STAGE_ACTION = {
    0: "pre_due_courtesy",
    1: "polite_follow_up",
    2: "firm_reminder",
    3: "early_settlement_offer",
    4: "payment_plan",
    5: "escalate_to_human",
}


def ladder_stage(dbt: int) -> int:
    """Highest ladder rung whose DBT threshold is met. Pure function of DBT."""
    stage = -1
    for s, threshold in config.LADDER.items():
        if dbt >= threshold:
            stage = s
    return stage


def _proposed_discount(stage: int) -> float:
    return config.EARLY_SETTLEMENT_DISCOUNT if stage == 3 else 0.0


def _stamp(invoice_id: str, day: dt.date) -> dt.datetime:
    """Deterministic business-hour timestamp seeded from the invoice id."""
    r = random.Random(invoice_id)
    hour = r.randint(config.BUSINESS_HOUR_START, config.BUSINESS_HOUR_END - 1)
    return dt.datetime(day.year, day.month, day.day, hour, r.randint(0, 59))


@dataclass
class _State:
    invoice_id: str
    buyer_id: str
    amount: float
    due_date: dt.date
    terms: int
    group: str
    natural_pay_date: dt.date
    lift_roll: float
    held: bool
    touches: int = 0
    last_touch_day: dt.date | None = None
    last_stage: int = -1
    proposed_discount: float = 0.0
    active_until: dt.date | None = None      # silent hold: quiet while day < this
    promised_date: dt.date | None = None
    promise_kept: bool | None = None
    escalated: bool = False
    paid: bool = False
    paid_day: dt.date | None = None


_NO_RISK = {"p_late": float("nan"), "expected_delay_days": None,
            "severity": None, "segment": None}


def _fallback_compose(f: dict) -> str:
    return f"{f['invoice_id']} ({f['amount_inr']}) - {f.get('pay_url') or 'contact AR'}"


def run(ledger: dict[str, pd.DataFrame], *, diagnose_fn=None, risk_fn=None, compose_fn=None,
        force_discount: dict[str, float] | None = None,
        live_link_invoice: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the simulation. Returns (audit_df, invoices_final_df).

    risk_fn(state) -> {p_late, expected_delay_days, severity, segment}  (see
    agent.make_risk_fn). Informational only - the ladder does not read it.

    compose_fn(facts) -> the buyer-facing email body (see agent.make_compose_fn).

    live_link_invoice: if set, that one invoice's payment link is created for
    real (Razorpay test mode); every other link is simulated.
    """
    diagnose_fn = diagnose_fn or (lambda inv, ctx: "undiagnosed")
    risk_fn = risk_fn or (lambda inv: dict(_NO_RISK))
    compose_fn = compose_fn or _fallback_compose
    force_discount = force_discount or {}

    invoices = ledger["invoices"].copy()
    buyers = ledger["buyers"].set_index("buyer_id")
    events = ledger["events"]
    ev_by_day: dict[dt.date, pd.DataFrame] = {d: g for d, g in events.groupby("day")}

    st: dict[str, _State] = {
        r.invoice_id: _State(
            invoice_id=r.invoice_id, buyer_id=r.buyer_id, amount=float(r.amount),
            due_date=r.due_date, terms=int(r.terms), group=r.group,
            natural_pay_date=r.natural_pay_date, lift_roll=float(r.lift_roll),
            held=bool(r.held),
        )
        for r in invoices.itertuples()
    }
    broken_promises: dict[str, int] = {}     # buyer_id -> count
    stop_promises: set[str] = set()          # buyers whose promises no longer buy silence
    audit: list[dict] = []

    def escalate(s: _State, day: dt.date, reason: str) -> None:
        if s.escalated:
            return
        s.escalated = True
        audit.append(_row(s, day, stage=config.TERMINAL_STAGE, action="escalate_to_human",
                          diagnosis=reason, risk=risk_fn(s), link=None,
                          bounds_passed=[n for n, _ in BOUNDS], human_gate=True,
                          message=f"Handed to human: {reason}."))

    for day in pd.date_range(config.SIM_START, config.SIM_END, freq="D"):
        day = day.date()

        # 1. apply dated events
        for e in ev_by_day.get(day, pd.DataFrame()).itertuples():
            s = st[e.invoice_id]
            if s.paid or s.escalated:
                continue
            if e.kind in ("dispute", "opt_out"):
                s.held = True
                escalate(s, day, "dispute_raised" if e.kind == "dispute" else "buyer_opt_out")
            elif e.kind == "promise":
                if s.group == "control":
                    continue                     # control gets no agent-driven behaviour
                kept = not pd.isna(e.promised_date)
                if s.buyer_id in stop_promises:
                    audit.append(_row(s, day, stage=s.last_stage, action="promise_noted_ignored",
                                      diagnosis="promise_untrusted", risk=risk_fn(s), link=None,
                                      bounds_passed=[], human_gate=False,
                                      message="Promise recorded but not honoured (buyer over broken-promise limit)."))
                    continue
                s.promise_kept = kept
                s.promised_date = e.promised_date if kept else (day + dt.timedelta(config.PROMISE_WINDOW_DAYS))
                s.active_until = s.promised_date
                audit.append(_row(s, day, stage=s.last_stage, action="promise_recorded",
                                  diagnosis="promise_to_pay", risk=risk_fn(s), link=None,
                                  bounds_passed=[], human_gate=False,
                                  message=f"Promise-to-pay recorded for {s.promised_date}."))

        # 2. walk open invoices
        for s in st.values():
            if s.paid or s.escalated:
                continue

            # active promise: silent hold while the window is open
            if s.active_until is not None:
                if day < s.active_until:
                    continue
                s.active_until = None                     # window reached
                if not s.promise_kept:                    # broken -> escalate, count it
                    broken_promises[s.buyer_id] = broken_promises.get(s.buyer_id, 0) + 1
                    if broken_promises[s.buyer_id] >= config.BROKEN_PROMISES_TO_STOP:
                        stop_promises.add(s.buyer_id)
                    escalate(s, day, "promise_broken")
                continue                                  # kept -> settle() pays it below

            if s.held:               # disputed / opted-out but event not reached yet
                continue

            dbt = (day - s.due_date).days
            stage = ladder_stage(dbt)
            if stage <= s.last_stage or stage < 0:
                continue

            s.proposed_discount = force_discount.get(s.invoice_id, _proposed_discount(stage))
            passed, escalating_fail, soft_fail = [], False, False
            for name, pred in BOUNDS:
                if pred(s, day):
                    passed.append(name)
                elif name in _ESCALATING_BOUNDS:
                    escalating_fail = True
                else:
                    soft_fail = True

            if escalating_fail:
                reason = ("discount_over_authority" if s.proposed_discount > config.DISCOUNT_CAP
                          else "above_maker_checker" if s.amount > config.MAKER_CHECKER_THRESHOLD
                          else "touch_budget_exhausted")
                escalate(s, day, reason)
                continue
            if soft_fail:
                continue

            if stage == config.TERMINAL_STAGE:
                escalate(s, day, "dbt_45")
                continue

            # 3. act
            risk = risk_fn(s)
            ctx = {"dbt": dbt, "buyer_id": s.buyer_id, "amount": s.amount,
                   "touches": s.touches, "terms": s.terms,
                   "p_late": risk.get("p_late"), "expected_delay": risk.get("expected_delay_days")}
            diag = diagnose_fn(s, ctx)

            b = buyers.loc[s.buyer_id] if s.buyer_id in buyers.index else None
            customer = ({"name": b["ap_contact"], "email": b["email"], "contact": b["phone"]}
                        if b is not None else None)
            link = pay_url = None
            if stage in (0, 3):
                res = razorpay_link.create_link(
                    s.invoice_id, s.amount, live=(s.invoice_id == live_link_invoice),
                    customer=customer, notify=config.RAZORPAY_LINK_NOTIFY,
                    reminders=config.RAZORPAY_LINK_REMINDERS, with_url=True)
                link, pay_url = res["id"], res["short_url"]

            disc = None
            if stage == 3 and diag == "cashflow":
                net = s.amount * (1 - config.EARLY_SETTLEMENT_DISCOUNT)
                disc = (f"We can offer a {config.EARLY_SETTLEMENT_DISCOUNT:.1%} early-settlement "
                        f"discount if it is cleared within 7 days - net ₹{net:,.0f}.")

            facts = {
                "buyer": b["name"] if b is not None else s.buyer_id,
                "ap_first": b["ap_contact"].split()[0] if b is not None else None,
                "invoice_id": s.invoice_id, "amount_inr": f"₹{s.amount:,.0f}",
                "terms": s.terms, "due_date": s.due_date.isoformat(), "dbt": dbt,
                "stage": stage, "diagnosis": diag, "touch_no": s.touches + 1,
                "prior_promise_broken": s.promise_kept is False,
                "pay_url": pay_url, "discount_line": disc,
                "signatory": config.SME_SIGNATORY, "sme_name": config.SME_NAME,
            }
            body = compose_fn(facts)
            to = b["email"] if b is not None else "unknown@unknown.example"
            subject = f"{s.invoice_id}: payment {_STAGE_ACTION[stage].replace('_', ' ')}"
            msg_id = notify.send(to, subject, body, live=False)

            s.touches += 1
            s.last_touch_day = day
            s.last_stage = stage
            audit.append(_row(s, day, stage=stage, action=_STAGE_ACTION[stage],
                              diagnosis=diag, risk=risk, link=link,
                              bounds_passed=passed, human_gate=False,
                              message=_message(_STAGE_ACTION[stage], s, dbt),
                              email_to=to, email_body=body, email_message_id=msg_id))

        # 4. settle
        for s in st.values():
            if s.paid or s.escalated or s.held:
                continue
            if s.promise_kept and s.promised_date and day >= s.promised_date:
                s.paid, s.paid_day = True, day
                continue
            pay_date = s.natural_pay_date
            if s.group == "treatment" and s.last_stage >= 0:
                days_earlier, prob = config.RESPONSE_LIFT[s.last_stage]
                if s.lift_roll < prob:
                    earliest = (s.last_touch_day or s.due_date) + dt.timedelta(days=1)
                    pay_date = max(pay_date - dt.timedelta(days=days_earlier), earliest)
            if day >= pay_date:
                s.paid, s.paid_day = True, day

    audit_df = pd.DataFrame(audit)
    _finalise_outcomes(audit_df, st)

    fin = invoices.copy()
    fin["paid"] = fin.invoice_id.map(lambda i: st[i].paid)
    fin["paid_day"] = fin.invoice_id.map(lambda i: st[i].paid_day)
    fin["escalated"] = fin.invoice_id.map(lambda i: st[i].escalated)
    fin["touches"] = fin.invoice_id.map(lambda i: st[i].touches)
    fin["effective_dbt"] = fin.apply(
        lambda r: (st[r.invoice_id].paid_day - r.due_date).days
        if st[r.invoice_id].paid_day else np.nan, axis=1)
    return audit_df, fin


# --- helpers ---------------------------------------------------------------

def _row(s: _State, day, *, stage, action, diagnosis, risk, link, bounds_passed,
         human_gate, message, email_to="", email_body="", email_message_id="") -> dict:
    return dict(
        timestamp=_stamp(s.invoice_id, day), invoice_id=s.invoice_id, buyer_id=s.buyer_id,
        risk_score=risk.get("p_late"), expected_delay_days=risk.get("expected_delay_days"),
        risk_severity=risk.get("severity"), risk_segment=risk.get("segment"),
        diagnosis=diagnosis, ladder_stage=stage, action_taken=action,
        message_sent=message, razorpay_object_id=link,
        bounds_checked=",".join(bounds_passed), human_gate_required=human_gate,
        email_to=email_to, email_body=email_body, email_message_id=email_message_id,
        outcome="pending", outcome_timestamp=pd.NaT,
    )


def _message(action: str, s: _State, dbt: int) -> str:
    amt = f"₹{s.amount:,.0f}"
    return {
        "pre_due_courtesy": f"Courtesy note: {s.invoice_id} ({amt}) is due soon. Link attached.",
        "polite_follow_up": f"{s.invoice_id} ({amt}) is past its due date. Could you confirm a payment date?",
        "firm_reminder":    f"{s.invoice_id} ({amt}) is {dbt} days beyond terms. Escalating to your AP contact.",
        "early_settlement_offer": f"{s.invoice_id} ({amt}): 0.5% discount if cleared within 7 days.",
        "payment_plan":     f"{s.invoice_id} ({amt}): proposing a partial settlement / payment plan.",
        "escalate_to_human": f"{s.invoice_id} handed to a human.",
    }.get(action, action)


def _finalise_outcomes(audit_df: pd.DataFrame, st: dict[str, _State]) -> None:
    if audit_df.empty:
        return
    for i, r in audit_df.iterrows():
        s = st[r.invoice_id]
        if s.paid:
            audit_df.at[i, "outcome"] = "paid"
            audit_df.at[i, "outcome_timestamp"] = _stamp(s.invoice_id, s.paid_day)
        elif s.escalated:
            audit_df.at[i, "outcome"] = "handed_off"
        else:
            audit_df.at[i, "outcome"] = "unresolved"


if __name__ == "__main__":
    import ledger

    L = ledger.generate_ledger()
    a, f = run(L)
    print(f"audit rows: {len(a)}")
    print(a.action_taken.value_counts().to_string())
    print("\npaid:", int(f.paid.sum()), "/ ", len(f),
          " | escalated:", int(f.escalated.sum()),
          " | unresolved:", int((~f.paid & ~f.escalated).sum()))
    print("\ntreatment vs control mean effective DBT:")
    print(f.groupby("group").effective_dbt.mean().round(1).to_string())
