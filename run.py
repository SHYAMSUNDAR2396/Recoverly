"""Orchestrator: ledger -> risk model -> engine -> metrics -> results.duckdb.

    ./.venv/bin/python run.py            # normal run
    ./.venv/bin/python run.py --no-llm   # skip Ollama, rule-based diagnosis only
    ./.venv/bin/python run.py --fresh    # regenerate the committed ledger
"""
from __future__ import annotations

import json
import sys

import duckdb
import pandas as pd

import agent
import brief as brief_mod
import config
import engine
import ledger
import metrics


def _write(con, name: str, df: pd.DataFrame) -> None:
    con.register("_t", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _t")
    con.unregister("_t")


def main(argv: list[str]) -> None:
    use_llm = "--no-llm" not in argv
    fresh = "--fresh" in argv

    L = ledger.generate_ledger(force=fresh)
    print(f"ledger: {len(L['invoices'])} invoices, {len(L['buyers'])} buyers, "
          f"{len(L['events'])} events")

    clf, m1 = agent.train_risk_model(L)
    reg, m2 = agent.train_delay_model(L)
    print("Model 1 (late classifier):", m1)
    print("Model 2 (delay regressor):", m2)

    risk_fn = agent.make_risk_fn(clf, reg, L["buyers"], invoices=L["invoices"])
    diagnose_fn = agent.make_diagnose_fn(L["buyers"], use_ollama=use_llm)

    audit, final = engine.run(L, diagnose_fn=diagnose_fn, risk_fn=risk_fn)
    if hasattr(diagnose_fn, "flush"):
        diagnose_fn.flush()

    m = metrics.compute_metrics(final, audit)
    exceptions = metrics.exception_list(final, audit)
    print(f"\npaid {int(final.paid.sum())}/{len(final)} | "
          f"escalated {int(final.escalated.sum())} | "
          f"net benefit ₹{m['net_benefit']:,.0f} | "
          f"DSO -{m['dso_reduction_days']}d | exceptions {len(exceptions)} | "
          f"Model2 MAE {m2['mae_days']}d (baseline {m2['baseline_mae_days']}d)")

    counts = L["invoices"].buyer_id.value_counts()
    briefs = []
    for bid in counts[counts >= brief_mod.MIN_SAMPLE].index:
        try:
            briefs.append({"buyer_id": bid, "json": json.dumps(brief_mod.buyer_brief(L, bid))})
        except ValueError:
            pass
    briefs_df = pd.DataFrame(briefs)

    con = duckdb.connect(str(config.RESULTS_DB))
    _write(con, "invoices", final)
    _write(con, "audit", audit)
    _write(con, "buyers", L["buyers"])
    _write(con, "events", L["events"])
    _write(con, "briefs", briefs_df)
    _write(con, "exceptions", pd.DataFrame(exceptions))
    _write(con, "meta", pd.DataFrame([{
        "metrics_json": json.dumps(m, default=str),
        "model1_metrics_json": json.dumps(m1),
        "model2_metrics_json": json.dumps(m2),
        "seed": config.SEED,
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
    }]))
    con.close()
    print(f"wrote {config.RESULTS_DB}  (tables: invoices, audit, buyers, events, "
          f"briefs, exceptions, meta)")


if __name__ == "__main__":
    main(sys.argv[1:])
