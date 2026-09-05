"""Single source of truth for constants. Imported by every module."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

# --- determinism ---------------------------------------------------------------
SEED = 42

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
LEDGER_PARQUET = DATA_DIR / "ledger"          # dir: invoices.parquet, buyers.parquet, events.parquet
RESULTS_DB = ROOT / "results.duckdb"

# --- simulation window -----------------------------------------------------
# Invoices are issued across the first ~4 months so they can age inside the window.
SIM_START = dt.date(2026, 1, 1)
SIM_END = dt.date(2026, 6, 30)
ISSUE_LAST = dt.date(2026, 5, 1)

# --- ledger shape --------------------------------------------------------------
N_BUYERS = 20
HEAVY_BUYERS = 8          # ~25 invoices each
LIGHT_BUYERS = 12         # ~8 invoices each
HEAVY_INVOICES = 25
LIGHT_INVOICES = 8
TERMS_CHOICES = (30, 45, 60)
TREATMENT_FRAC = 0.70

# --- collections ladder: stage -> minimum days-beyond-terms (DBT) ----------
# Stage 0 fires before the due date; negative DBT.
LADDER = {
    0: -5,
    1: 0,
    2: 7,
    3: 15,
    4: 30,
    5: 45,
}
TERMINAL_STAGE = 5

# --- hard bounds -------------------------------------------------------------
MAX_TOUCHES = 6
MIN_GAP_DAYS = 3                     # >= 72h between touches
DISCOUNT_CAP = 0.02                  # 2% autonomous discount authority (hard ceiling)
EARLY_SETTLEMENT_DISCOUNT = 0.005    # what stage 3 actually offers; 2% is the cap, not the default

# --- risk cascade --------------------------------------------------------------
RISK_THRESHOLD = 0.5                 # Model 1 P(late) at/above which Model 2 (delay) runs
MAKER_CHECKER_THRESHOLD = 350_000    # above this, no autonomous action without sign-off
MAX_INVOICE_AMOUNT = 490_000         # every invoice stays under Razorpay's ₹5,00,000
                                     # test-mode Payment Links cap, so --live-link and
                                     # the dashboard's live-send work on ANY invoice
BUSINESS_HOUR_START = 9             # 09:00 IST
BUSINESS_HOUR_END = 19             # 19:00 IST

# --- response model (STATED ASSUMPTION, benchmark-anchored) ----------------
# stage -> (days a treatment invoice is pulled forward, probability the lift lands)
# Anchored loosely to published collections-automation benchmarks; NOT measured here.
RESPONSE_LIFT = {
    0: (1, 0.15),
    1: (3, 0.30),
    2: (6, 0.45),
    3: (9, 0.55),
    4: (11, 0.40),
    5: (0, 0.00),
}

# --- promise loop ----------------------------------------------------------
PROMISE_WINDOW_DAYS = 12          # a promise buys this many quiet days if none stated
BROKEN_PROMISES_TO_STOP = 2      # 2nd broken promise (buyer-level) -> stop honoring

# --- outbound / notifications --------------------------------------------------
SME_NAME = "Northwind Components"                    # the SME Recoverly runs collections for
SME_SIGNATORY = "Ravi Menon"                         # signs the buyer emails
RAZORPAY_LINK_NOTIFY = {"sms": True, "email": False}  # Razorpay sends the SMS nudge;
                                                     # the personalised email is notify.py's job
RAZORPAY_LINK_REMINDERS = True                       # Razorpay's own escalating link reminders
