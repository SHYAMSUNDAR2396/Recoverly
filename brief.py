"""The per-buyer leverage brief - the differentiator.

Built from the buyer's *natural* payment record (the ledger), not the engine
output, so it reflects how the buyer actually behaves, independent of anything
the agent did. Deterministic recommendation, generative justification - the
same split as the ladder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config

MIN_SAMPLE = 20          # enforced: no brief for a buyer with fewer invoices


def promise_kept_rate(events_for_buyer: pd.DataFrame):
    """Kept / total promises. None when the buyer has never promised -
    never 0.0, which would put a false claim on the brief."""
    promises = events_for_buyer[events_for_buyer.kind == "promise"]
    if len(promises) == 0:
        return None
    kept = promises.promised_date.notna().sum()
    return round(float(kept) / len(promises), 2)


def recommended_terms(honored_rate: float, mean_dbt: float, terms: int) -> str:
    """Pure function of the track record."""
    if mean_dbt <= 3:
        return f"Keep Net {terms}. The record supports the current terms."
    if honored_rate >= 0.6 and mean_dbt <= 12:
        return (f"Keep Net {terms} but add a {min(1.5, config.DISCOUNT_CAP * 100):.1f}% "
                f"early-settlement discount for payment within {max(10, terms - 25)} days.")
    tighter = 30 if terms > 45 else max(15, terms - 15)
    return (f"Move to Net {tighter} with a 1.5% early-settlement discount if paid by day "
            f"{max(10, tighter - 25)}. The shorter term absorbs the observed {mean_dbt:+.1f}-day drift.")


def buyer_brief(ledger: dict[str, pd.DataFrame], buyer_id: str, draft_fn=None) -> dict:
    inv = ledger["invoices"]
    ev = ledger["events"]
    buyers = ledger["buyers"].set_index("buyer_id")
    b_inv = inv[inv.buyer_id == buyer_id]
    if len(b_inv) < MIN_SAMPLE:
        raise ValueError(f"{buyer_id}: {len(b_inv)} invoices < MIN_SAMPLE {MIN_SAMPLE}")

    name = buyers.loc[buyer_id, "name"]
    terms = int(b_inv.terms.mode().iat[0])
    # honoured = cleared on or before the due date and never disputed
    honored = int(((b_inv.natural_dbt <= 0) & ~b_inv.disputed).sum())
    broken = int(len(b_inv) - honored)
    honored_rate = round(honored / len(b_inv), 2)
    mean_dbt = round(float(b_inv.natural_dbt.mean()), 1)
    sd_dbt = round(float(b_inv.natural_dbt.std()), 1)
    dispute_rate = round(float(b_inv.disputed.mean()), 2)

    peers = inv[(inv.terms == terms) & (inv.buyer_id != buyer_id)]
    peer_dbt = round(float(peers.natural_dbt.mean()), 1) if len(peers) else mean_dbt

    facts = dict(buyer=name, terms=terms, honored=honored, broken=broken,
                 honored_rate=honored_rate, mean_dbt=mean_dbt, sd_dbt=sd_dbt,
                 dispute_rate=dispute_rate, peer_dbt=peer_dbt,
                 promise_kept_rate=promise_kept_rate(ev[ev.buyer_id == buyer_id]))
    rec = recommended_terms(honored_rate, mean_dbt, terms)
    justification = (draft_fn or _default_draft)({**facts})
    return {
        **facts,
        "n_invoices": int(len(b_inv)),
        "recommended_terms": rec,
        "evidence_line": (f"Net {terms} agreed; honoured {honored}, broken {broken}; "
                          f"mean {mean_dbt:+.1f}d vs peer {peer_dbt:+.1f}d."),
        "justification": justification,
        "message": _message(name, facts, rec),
    }


def _default_draft(facts: dict) -> str:
    try:
        import agent
        return agent.draft_justification(facts)
    except Exception:                       # noqa: BLE001
        return (f"{facts['buyer']} agreed Net {facts['terms']} but settles "
                f"{facts['mean_dbt']:+.1f} days beyond terms on average, against a peer "
                f"average of {facts['peer_dbt']:+.1f}. The recommended terms reset the "
                f"anchor without threatening the relationship.")


def _message(name: str, facts: dict, rec: str) -> str:
    return (f"Hi,\n\nAhead of the next order I would like to align on payment terms. "
            f"Across our last {facts['honored'] + facts['broken']} invoices, settlement has "
            f"averaged {facts['mean_dbt']:+.0f} days beyond the Net {facts['terms']} we agreed. "
            f"Proposed: {rec}\n\nHappy to walk through the invoice history on a call.\n\nBest,\nRavi")


if __name__ == "__main__":
    import ledger

    L = ledger.generate_ledger()
    counts = L["invoices"].buyer_id.value_counts()
    for bid in counts[counts >= MIN_SAMPLE].index[:3]:
        br = buyer_brief(L, bid)
        print(f"\n== {br['buyer']}  (n={br['n_invoices']})")
        for k in ("evidence_line", "promise_kept_rate", "dispute_rate",
                  "recommended_terms", "justification"):
            print(f"  {k}: {br[k]}")
