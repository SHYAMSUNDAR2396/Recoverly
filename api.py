"""Read-only API over results.duckdb. No writes, no auth, no webhook.

    ./.venv/bin/uvicorn api:app --reload
"""
from __future__ import annotations

import json

import duckdb
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config

app = FastAPI(title="Recoverly", version="0.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])


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
