"""Synthetic receivables ledger.

Generates three tables and commits them as parquet so every run sees identical
data (determinism contract, config.SEED):

  buyers    - one behavioral profile per buyer
  invoices  - ~300 invoices, 6 months, deliberately messy
  events    - dated stream of disputes / opt-outs / promises, discovered on
              their date by the engine (never pre-joined -> no future leakage)

Payments are NOT events. The engine derives them from `natural_pay_date` plus
the response model (config.RESPONSE_LIFT), so the treatment effect stays
isolated and auditable.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import config

# --- named buyers so the demo has recognisable characters ---------------------
# tier, dbt_mean, dbt_sd, qend_squeeze, partial_rate, dispute_rate,
# will_opt_out, promise_rate, promise_keep_rate, responsiveness
NAMED_BUYERS = {
    "Meridian Retail":   ("heavy", 17, 6,  0.15, 0.10, 0.00, False, 0.55, 0.55, 0.55),
    "Harbour Textiles":  ("heavy", 30, 10, 0.20, 0.25, 0.22, True,  0.35, 0.30, 0.35),
    "Crest Pharma":      ("heavy", 9,  4,  0.70, 0.05, 0.00, False, 0.60, 0.85, 0.75),
    "Blueleaf Foods":    ("light", 4,  8,  0.10, 0.15, 0.05, False, 0.45, 0.60, 0.60),
    "Nandi Logistics":   ("light", -3, 3,  0.05, 0.02, 0.00, False, 0.20, 0.90, 0.80),
}
_FILLER_NAMES = [
    "Ashford Mills", "Corvid Systems", "Delta Provisions", "Everline Retail",
    "Fenwick Tools", "Grovewood Supply", "Halden Pharma", "Ironbark Foods",
    "Juniper Textiles", "Kettle & Co", "Larkspur Logistics", "Maple Freight",
    "Northgate Retail", "Oakhaven Supply", "Pinecrest Foods",
]

# synthetic AP contacts. `.example` is IANA-reserved -> no real inbox is reachable.
_AP_FIRST = ["Priya", "Deepak", "Anjali", "Farah", "Sunil", "Meera", "Arjun",
             "Kavya", "Rohan", "Nisha", "Vikram", "Sneha"]
_AP_LAST = ["Sharma", "Nair", "Iyer", "Patel", "Reddy", "Bose", "Khanna", "Menon"]


def _slug(name: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "-" for c in name)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-")


def _month_end_squeeze(due: dt.date, squeeze: float) -> float:
    """Quarter-end buyers pay slower on invoices due in the last 10 days of Mar/Jun/Sep/Dec."""
    if due.month in (3, 6, 9, 12) and due.day >= 21:
        return 1.0 + squeeze
    return 1.0


def _build_buyers(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    names = list(NAMED_BUYERS) + _FILLER_NAMES
    crng = np.random.default_rng(config.SEED ^ 0x5EED)   # separate stream: contacts don't
    for i, name in enumerate(names):                     # perturb the ledger's numbers
        if name in NAMED_BUYERS:
            (tier, dbt_mean, dbt_sd, qend, partial, disp, opt, prom, keep, resp) = NAMED_BUYERS[name]
        else:
            tier = "heavy" if i < config.HEAVY_BUYERS else "light"
            dbt_mean = float(rng.normal(10, 12))
            dbt_sd = float(rng.uniform(3, 12))
            qend = float(rng.uniform(0.0, 0.5))
            partial = float(rng.uniform(0.0, 0.25))
            disp = float(rng.choice([0.0, 0.0, 0.05, 0.12], p=[0.5, 0.2, 0.2, 0.1]))
            opt = bool(rng.random() < 0.08)
            prom = float(rng.uniform(0.2, 0.6))
            keep = float(rng.uniform(0.35, 0.9))
            resp = float(rng.uniform(0.3, 0.8))
        ap = f"{_AP_FIRST[crng.integers(len(_AP_FIRST))]} {_AP_LAST[crng.integers(len(_AP_LAST))]}"
        rows.append(dict(
            buyer_id=f"BUY-{i:02d}", name=name, tier=tier,
            dbt_mean=round(dbt_mean, 1), dbt_sd=round(dbt_sd, 1),
            qend_squeeze=round(qend, 2), partial_rate=round(partial, 2),
            dispute_rate=round(disp, 2), will_opt_out=opt,
            promise_rate=round(prom, 2), promise_keep_rate=round(keep, 2),
            responsiveness=round(resp, 2),
            ap_contact=ap, email=f"ap@{_slug(name)}.example",
            phone=f"+9198{int(crng.integers(10_000_000, 99_999_999))}",
        ))
    return pd.DataFrame(rows)


def _build_invoices(buyers: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    n = 0
    span_days = (config.ISSUE_LAST - config.SIM_START).days
    for _, b in buyers.iterrows():
        count = config.HEAVY_INVOICES if b.tier == "heavy" else config.LIGHT_INVOICES
        count = int(rng.integers(count - 3, count + 4))
        base_amt = 220_000 if b.tier == "heavy" else 70_000
        for _ in range(count):
            issue = config.SIM_START + dt.timedelta(days=int(rng.integers(0, span_days)))
            terms = int(rng.choice(config.TERMS_CHOICES))
            due = issue + dt.timedelta(days=terms)
            amount = float(np.round(rng.lognormal(mean=np.log(base_amt), sigma=0.35), -3))
            amount = min(amount, config.MAX_INVOICE_AMOUNT)   # every invoice stays clickable
            squeeze = _month_end_squeeze(due, b.qend_squeeze)
            natural_dbt = int(round(rng.normal(b.dbt_mean * squeeze, b.dbt_sd)))
            natural_dbt = max(natural_dbt, -10)
            rows.append(dict(
                invoice_id=f"INV-{2000 + n}", buyer_id=b.buyer_id, amount=amount,
                issue_date=issue, due_date=due, terms=terms,
                natural_pay_date=due + dt.timedelta(days=natural_dbt),
                natural_dbt=natural_dbt,
                lift_roll=float(rng.random()),      # one uniform per invoice, threshold applied by engine
            ))
            n += 1
    inv = pd.DataFrame(rows)
    # treatment / control split, stratified by buyer so groups are matched
    inv["group"] = "control"
    treat_idx = (
        inv.groupby("buyer_id", group_keys=False)
        .apply(lambda g: g.sample(frac=config.TREATMENT_FRAC, random_state=config.SEED))
        .index
    )
    inv.loc[treat_idx, "group"] = "treatment"
    inv["disputed"] = False
    inv["held"] = False          # disputed or opted-out -> never settles
    return inv.reset_index(drop=True)


def _build_events(buyers: pd.DataFrame, inv: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    bmap = buyers.set_index("buyer_id")
    events = []

    # disputes: buyer behaviour, both groups, invoice is held
    for idx, r in inv.iterrows():
        b = bmap.loc[r.buyer_id]
        if rng.random() < b.dispute_rate:
            day = r.due_date + dt.timedelta(days=int(rng.integers(3, 18)))
            if day <= config.SIM_END:
                events.append((r.invoice_id, r.buyer_id, day, "dispute", pd.NaT))
                inv.loc[idx, ["disputed", "held"]] = [True, True]

    # opt-out: one buyer-level event, lands on that buyer's worst open invoice
    for bid, b in bmap.iterrows():
        if not b.will_opt_out:
            continue
        cand = inv[(inv.buyer_id == bid) & (~inv.held)]
        if cand.empty:
            continue
        target = cand.loc[cand.natural_dbt.idxmax()]
        day = target.due_date + dt.timedelta(days=int(rng.integers(5, 15)))
        if day <= config.SIM_END:
            events.append((target.invoice_id, bid, day, "opt_out", pd.NaT))
            inv.loc[inv.invoice_id == target.invoice_id, "held"] = True

    # promises: generated for every buyer so the leverage brief has real data.
    # The engine only *acts* on promises for treatment invoices (see engine.run);
    # control invoices stay purely natural.
    for _, r in inv[~inv.held].iterrows():
        b = bmap.loc[r.buyer_id]
        if r.natural_dbt <= 5 or rng.random() >= b.promise_rate:
            continue
        offset = int(rng.integers(3, max(4, r.natural_dbt)))
        day = r.due_date + dt.timedelta(days=offset)
        promised = day + dt.timedelta(days=int(rng.integers(5, 15)))
        kept = rng.random() < b.promise_keep_rate
        if day <= config.SIM_END:
            events.append((r.invoice_id, r.buyer_id, day, "promise",
                           promised if kept else pd.NaT))

    df = pd.DataFrame(events, columns=["invoice_id", "buyer_id", "day", "kind", "promised_date"])
    return df.sort_values("day").reset_index(drop=True)


def generate_ledger(seed: int = config.SEED, force: bool = False) -> dict[str, pd.DataFrame]:
    """Build (or load) the ledger. Writes parquet on first build."""
    p = config.LEDGER_PARQUET
    if p.exists() and not force:
        return {name: pd.read_parquet(p / f"{name}.parquet")
                for name in ("buyers", "invoices", "events")}

    rng = np.random.default_rng(seed)
    buyers = _build_buyers(rng)
    invoices = _build_invoices(buyers, rng)
    events = _build_events(buyers, invoices, rng)

    p.mkdir(parents=True, exist_ok=True)
    buyers.to_parquet(p / "buyers.parquet", index=False)
    invoices.to_parquet(p / "invoices.parquet", index=False)
    events.to_parquet(p / "events.parquet", index=False)
    return {"buyers": buyers, "invoices": invoices, "events": events}


if __name__ == "__main__":
    L = generate_ledger(force=True)
    for name, df in L.items():
        print(f"\n== {name}  ({len(df)} rows)")
        print(df.head(6).to_string())
    inv = L["invoices"]
    print("\ngroups:", inv.group.value_counts().to_dict())
    print("held (dispute/opt-out):", int(inv.held.sum()))
    print("events:", L["events"].kind.value_counts().to_dict())
