"""Treatment vs control. The control group proves the measurement machinery is
correct - it does not prove the agent works (the treatment effect only exists
because of the authored RESPONSE_LIFT assumption)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

COST_OF_CAPITAL = 0.18          # annual; SME working-capital financing rate
DISCOUNT_WINDOW_DAYS = 7        # the stage-3 offer is "0.5% if cleared within 7 days"
_ACTIONS = ("pre_due_courtesy", "polite_follow_up", "firm_reminder",
            "early_settlement_offer", "payment_plan")


def _group_block(g: pd.DataFrame) -> dict:
    paid = g[g.paid]
    at_risk_value = float(g.loc[g.natural_dbt > 0, "amount"].sum())
    recovered = float(paid.loc[paid.natural_dbt > 0, "amount"].sum())
    return {
        "n": int(len(g)),
        "rupees_recovered": round(recovered, 0),
        "pct_at_risk_recovered": round(recovered / at_risk_value, 3) if at_risk_value else None,
        "mean_dso_days": round(float((paid.terms + paid.effective_dbt).mean()), 1) if len(paid) else None,
        "mean_dbt_days": round(float(paid.effective_dbt.mean()), 1) if len(paid) else None,
        "paid": int(g.paid.sum()),
        "escalated": int(g.escalated.sum()),
        "unresolved": int((~g.paid & ~g.escalated).sum()),
    }


def compute_metrics(final: pd.DataFrame, audit: pd.DataFrame) -> dict:
    final = final.copy()
    by_group = {grp: _group_block(g) for grp, g in final.groupby("group")}

    treat = final[(final.group == "treatment") & final.paid].copy()
    pulled = (treat.natural_dbt - treat.effective_dbt).clip(lower=0)
    cash_days = float((treat.amount * pulled).sum())
    accel_value = cash_days * COST_OF_CAPITAL / 365.0

    # a discount only *costs* anything when it actually pulled the payment in:
    # the invoice settled within the 7-day offer window.
    offers = (audit[audit.action_taken == "early_settlement_offer"]
              .assign(offer_day=lambda d: pd.to_datetime(d.timestamp).dt.date)
              .groupby("invoice_id").offer_day.min())
    paid_day = final.set_index("invoice_id").paid_day
    took_discount = [
        iid for iid, oday in offers.items()
        if pd.notna(paid_day.get(iid)) and 0 <= (paid_day[iid] - oday).days <= DISCOUNT_WINDOW_DAYS
    ]
    discounted = final[final.invoice_id.isin(took_discount)]
    discount_cost = float((discounted.amount * config.EARLY_SETTLEMENT_DISCOUNT).sum())

    touches = int(audit.action_taken.isin(_ACTIONS).sum())
    promises = audit[audit.action_taken == "promise_recorded"]
    broken = audit[audit.diagnosis == "promise_broken"]
    pkr = round(1 - len(broken) / len(promises), 2) if len(promises) else None

    dso_delta = None
    if by_group.get("control", {}).get("mean_dso_days") and by_group.get("treatment", {}).get("mean_dso_days"):
        dso_delta = round(by_group["control"]["mean_dso_days"] - by_group["treatment"]["mean_dso_days"], 1)

    net = accel_value - discount_cost
    return {
        "by_group": by_group,
        "dso_reduction_days": dso_delta,
        "cash_pulled_forward_rupee_days": round(cash_days, 0),
        "cash_acceleration_value": round(accel_value, 0),
        "discount_cost": round(discount_cost, 0),
        "discounted_invoices": len(discounted),
        "touches_spent": touches,
        "net_benefit": round(net, 0),
        "promise_kept_rate": pkr,
        "cost_of_capital": COST_OF_CAPITAL,
        "response_model_assumption": config.RESPONSE_LIFT,
        "interpretation": (
            f"The free reminder rungs (0-2) pull DSO down {dso_delta} days, worth "
            f"~₹{accel_value:,.0f} at a {COST_OF_CAPITAL:.0%} cost of capital. The stage-3 "
            f"discount rung runs at roughly break-even (net ₹{net:,.0f} overall) - a tuning "
            f"lever an operator can tighten or disable."),
        "note": ("Treatment effect is produced by the authored RESPONSE_LIFT assumption. "
                 "Control proves the measurement machinery, not the agent."),
    }


def exception_list(final: pd.DataFrame, audit: pd.DataFrame) -> list[dict]:
    """Every invoice the agent could not resolve, honestly labelled."""
    stuck = final[(final.group == "treatment") & ~final.paid]
    last_reason = (audit.sort_values("timestamp").groupby("invoice_id").diagnosis.last())
    out = []
    for r in stuck.itertuples():
        reason = last_reason.get(r.invoice_id, "no_action_recorded")
        label = {
            "dispute_raised": "Escalated - dispute raised, handed to a human",
            "buyer_opt_out": "Opt-out on file - manual collections only",
            "promise_broken": "Escalated - promise-to-pay broken",
            "above_maker_checker": "Above maker-checker threshold - awaiting sign-off",
            "discount_over_authority": "Discount request over authority - escalated",
            "dbt_45": "Reached DBT 45 - handed to a human",
        }.get(reason, f"Unresolved ({reason})")
        out.append({"invoice_id": r.invoice_id, "buyer_id": r.buyer_id,
                    "amount": float(r.amount), "reason": label})
    return out
