"""Orchestrator: ledger -> risk model -> engine -> metrics -> results.duckdb.

Credentials (RAZORPAY_KEY_ID/SECRET, SMTP_USER/PASSWORD) are loaded from a
.env file in the project root (see .env.example) - fill it in once, and every
run/every terminal picks it up automatically. No more manual `export` per shell.

    ./.venv/bin/python run.py                    # normal run (loads models/ artifacts)
    ./.venv/bin/python run.py --no-llm           # skip Ollama, rule-based diagnosis only
    ./.venv/bin/python run.py --fresh            # regenerate the committed ledger
    ./.venv/bin/python run.py --retrain          # re-fit Model 1 + Model 2, rewrite models/
    ./.venv/bin/python run.py --live-link INV-2032   # ONE real Razorpay test-mode link
                                                     # (needs RAZORPAY_KEY_ID / _SECRET env,
                                                     #  invoice amount < ₹5,00,000)
    ./.venv/bin/python run.py --live-link INV-2032 --demo-email you@gmail.com
                                                     # + sends that invoice's email for real
                                                     # (needs SMTP_USER / SMTP_PASSWORD env,
                                                     #  e.g. a Gmail address + App Password)
"""
from __future__ import annotations

import json
import sys

from dotenv import load_dotenv

load_dotenv()          # populate os.environ from .env before anything reads it

import duckdb
import pandas as pd

import agent
import brief as brief_mod
import config
import engine
import ledger
import metrics
import notify


def _write(con, name: str, df: pd.DataFrame) -> None:
    con.register("_t", df)
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _t")
    con.unregister("_t")


def main(argv: list[str]) -> None:
    use_llm = "--no-llm" not in argv
    fresh = "--fresh" in argv
    live_link_invoice = (argv[argv.index("--live-link") + 1]
                         if "--live-link" in argv and argv.index("--live-link") + 1 < len(argv)
                         else None)
    demo_email = (argv[argv.index("--demo-email") + 1]
                 if "--demo-email" in argv and argv.index("--demo-email") + 1 < len(argv)
                 else None)
    if demo_email and not live_link_invoice:
        print("[run] --demo-email given without --live-link; ignoring (no real link to send)")
        demo_email = None

    L = ledger.generate_ledger(force=fresh)
    print(f"ledger: {len(L['invoices'])} invoices, {len(L['buyers'])} buyers, "
          f"{len(L['events'])} events")

    clf, reg, m1, m2 = agent.load_or_train_models(L, retrain="--retrain" in argv)
    print("Model 1 (late classifier):", m1)
    print("Model 2 (delay regressor):", m2)

    risk_fn = agent.make_risk_fn(clf, reg, L["buyers"], invoices=L["invoices"])
    diagnose_fn = agent.make_diagnose_fn(L["buyers"], use_ollama=use_llm)
    compose_fn = agent.make_compose_fn(use_ollama=use_llm)

    audit, final = engine.run(L, diagnose_fn=diagnose_fn, risk_fn=risk_fn,
                              compose_fn=compose_fn, live_link_invoice=live_link_invoice,
                              live_email_to=demo_email)
    for fn in (diagnose_fn, compose_fn):
        if hasattr(fn, "flush"):
            fn.flush()

    if live_link_invoice:
        ids = set(audit.loc[audit.invoice_id == live_link_invoice, "razorpay_object_id"].dropna())
        real = [i for i in ids if i and not i.startswith("plink_sim_")]
        print(f"live link for {live_link_invoice}: "
              + (real[0] if real else f"FELL BACK to simulated {ids or '(no link stage reached)'}"))

    if demo_email:
        mids = set(audit.loc[audit.invoice_id == live_link_invoice, "email_message_id"].dropna())
        real_mail = [m for m in mids if m.startswith("msg_live_")]
        print(f"demo email to {demo_email}: "
              + (real_mail[0] if real_mail else f"FELL BACK to dry-run {mids or '(no email sent yet)'}"))

    m = metrics.compute_metrics(final, audit)
    exceptions = metrics.exception_list(final, audit)
    print(f"\npaid {int(final.paid.sum())}/{len(final)} | "
          f"escalated {int(final.escalated.sum())} | "
          f"net benefit ₹{m['net_benefit']:,.0f} | "
          f"DSO -{m['dso_reduction_days']}d | exceptions {len(exceptions)} | "
          f"Model2 MAE {m2['mae_days']}d (baseline {m2['baseline_mae_days']}d) | "
          f"emails composed {notify.sent_count()} (dry-run, .example)")

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
