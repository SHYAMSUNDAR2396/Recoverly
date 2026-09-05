"""Read-only API over results.duckdb, plus one deliberate exception.

Every endpoint is a `GET` over `results.duckdb` except `POST /demo/live-send`,
which performs one live external action (a real Razorpay test-mode link + a
real email) for a single invoice, on explicit request from the dashboard. It
does not write to results.duckdb - it returns the outcome directly - so the
"read-only over the database" claim for every other endpoint still holds.

    ./.venv/bin/uvicorn api:app --reload
"""
from __future__ import annotations

import json

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agent
import config
import notify
import razorpay_link

app = FastAPI(title="Recoverly", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def _con() -> duckdb.DuckDBPyConnection:
    if not config.RESULTS_DB.exists():
        raise HTTPException(503, "results.duckdb not found - run `python run.py` first")
    return duckdb.connect(str(config.RESULTS_DB), read_only=True)


def _rows(sql: str, params: list | None = None) -> list[dict]:
    with _con() as con:
        df = con.execute(sql, params or []).df()
    df = df.astype(object).where(pd.notna(df), None)      # NaN / NaT -> null, JSON-safe
    return df.to_dict(orient="records")


@app.get("/metrics")
def metrics():
    with _con() as con:
        row = con.execute("SELECT * FROM meta LIMIT 1").df().to_dict(orient="records")[0]
    return {
        "metrics": json.loads(row["metrics_json"]),
        "model1": json.loads(row["model1_metrics_json"]),
        "model2": json.loads(row["model2_metrics_json"]),
        "seed": row["seed"],
        "generated_at": row["generated_at"],
    }


@app.get("/invoices")
def invoices(group: str | None = None):
    sql = "SELECT * FROM invoices"
    params: list = []
    if group:
        sql += ' WHERE "group" = ?'
        params.append(group)
    return _rows(sql + " ORDER BY effective_dbt DESC NULLS FIRST", params)


@app.get("/audit")
def audit(invoice_id: str | None = None, limit: int = 500):
    if invoice_id:
        return _rows("SELECT * FROM audit WHERE invoice_id = ? ORDER BY timestamp", [invoice_id])
    return _rows("SELECT * FROM audit ORDER BY timestamp DESC LIMIT ?", [limit])


@app.get("/exceptions")
def exceptions():
    return _rows("SELECT * FROM exceptions")


@app.get("/buyers")
def buyers():
    return _rows("SELECT b.*, "
                 "(SELECT count(*) FROM invoices i WHERE i.buyer_id = b.buyer_id) AS n_invoices "
                 "FROM buyers b ORDER BY n_invoices DESC")


@app.get("/buyers/{buyer_id}/brief")
def buyer_brief(buyer_id: str):
    rows = _rows("SELECT json FROM briefs WHERE buyer_id = ?", [buyer_id])
    if not rows:
        raise HTTPException(404, f"no brief for {buyer_id} (needs >= 20 invoices)")
    return json.loads(rows[0]["json"])


class LiveSendRequest(BaseModel):
    invoice_id: str
    email: str


@app.post("/demo/live-send")
def demo_live_send(req: LiveSendRequest):
    """Create one real Razorpay test-mode link and send one real email for it.

    Reads credentials from the server's own environment only (RAZORPAY_KEY_ID/
    SECRET, SMTP_USER/PASSWORD) - never from the request. Falls back to a
    simulated link / dry-run email on any failure, same as the CLI path
    (`run.py --live-link ... --demo-email ...`); the response says which
    happened so the dashboard can show it honestly.
    """
    inv_rows = _rows("SELECT * FROM invoices WHERE invoice_id = ?", [req.invoice_id])
    if not inv_rows:
        raise HTTPException(404, f"unknown invoice {req.invoice_id}")
    inv = inv_rows[0]

    buyer_rows = _rows("SELECT * FROM buyers WHERE buyer_id = ?", [inv["buyer_id"]])
    if not buyer_rows:
        raise HTTPException(404, f"unknown buyer {inv['buyer_id']}")
    b = buyer_rows[0]

    warning = None
    if inv["amount"] >= 500_000:
        warning = ("Amount is at/above Razorpay's ₹5,00,000 test-mode cap; "
                  "the live link will likely fall back to simulated.")

    customer = {"name": b["ap_contact"], "email": b["email"], "contact": b["phone"]}
    link = razorpay_link.create_link(
        req.invoice_id, float(inv["amount"]), live=True, customer=customer,
        notify=config.RAZORPAY_LINK_NOTIFY, reminders=config.RAZORPAY_LINK_REMINDERS,
        with_url=True)

    facts = {
        "buyer": b["name"], "ap_first": (b["ap_contact"] or "").split(" ")[0] or None,
        "invoice_id": req.invoice_id, "amount_inr": f"₹{inv['amount']:,.0f}",
        "terms": inv["terms"], "due_date": str(inv["due_date"])[:10],
        "dbt": int(inv.get("effective_dbt") or inv.get("natural_dbt") or 0),
        "stage": 0, "diagnosis": "cashflow", "touch_no": 1,
        "prior_promise_broken": False, "pay_url": link["short_url"], "discount_line": None,
        "signatory": config.SME_SIGNATORY, "sme_name": config.SME_NAME,
    }
    body = agent.compose_email(facts)
    subject = f"{req.invoice_id}: payment reminder (demo)"
    msg_id = notify.send(req.email, subject, body, live=True)

    return {
        "invoice_id": req.invoice_id, "sent_to": req.email,
        "link_id": link["id"], "pay_url": link["short_url"],
        "link_live": not link["id"].startswith("plink_sim_"),
        "email_message_id": msg_id, "email_live": msg_id.startswith("msg_live_"),
        "email_body": body, "warning": warning,
    }
