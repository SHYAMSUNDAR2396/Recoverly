"""Payment-link adapter.

Named `razorpay_link` (not `razorpay`) so it does not shadow the `razorpay`
PyPI SDK. One `live` flag serves both paths:

  live=False -> simulated, zero network calls (the batch run)
  live=True  -> one real test-mode link via the SDK (the demo invoice)

The live path is intentionally the only outbound network call in the project.

When `customer` is supplied, the link carries recipient details so Razorpay
sends the SMS nudge and its own escalating reminders. The personalised email is
composed and (dry-run by default) sent by notify.py, not by Razorpay - hence
the default `notify` has email off.

Razorpay requires `reference_id` to be unique per link, and this adapter uses
the invoice id as that reference - so re-running --live-link on an invoice
that already has a real link raises "already exists". `create_link` recovers
by fetching that existing link instead of silently faking one.
"""
from __future__ import annotations

import hashlib
import os

_CACHE: dict[str, dict] = {}


def create_link(invoice_id: str, amount: float, *, live: bool = False,
                customer: dict | None = None, notify: dict | None = None,
                reminders: bool = True, with_url: bool = False):
    """Return a payment-link id (str), or `{"id", "short_url"}` when `with_url`.

    Deterministic on the simulated path. `customer` / `notify` / `reminders` are
    only sent to Razorpay on the live path; ignored when simulated. The
    simulated short_url is obviously fake (never a real rzp.io path) so it can
    never be mistaken for - or clicked as - a real link.
    """
    if not live:
        h = hashlib.sha1(invoice_id.encode()).hexdigest()[:10]
        rec = {"id": f"plink_sim_{h}", "short_url": f"https://simulated.invalid/pay/{h}"}
        return rec if with_url else rec["id"]

    if invoice_id not in _CACHE:                   # one real call per invoice; then cached
        try:
            import razorpay  # type: ignore

            client = razorpay.Client(auth=(os.environ["RAZORPAY_KEY_ID"],
                                           os.environ["RAZORPAY_KEY_SECRET"]))
            payload = {
                "amount": int(round(amount * 100)),   # paise
                "currency": "INR",
                "accept_partial": False,
                "reference_id": invoice_id,
                "description": f"Recoverly {invoice_id}",
            }
            if customer:
                payload["customer"] = customer
                payload["notify"] = notify or {"sms": True, "email": False}
                payload["reminder_enable"] = reminders
            try:
                link = client.payment_link.create(payload)
            except Exception as exc:                    # noqa: BLE001
                if "already exists" in str(exc) or "already attempted" in str(exc):
                    link = _find_by_reference(client, invoice_id)
                    if link is None:
                        raise
                    print(f"[razorpay_link] {invoice_id} already has a live link; reusing it")
                else:
                    raise
            _CACHE[invoice_id] = {"id": link["id"], "short_url": link.get("short_url", "")}
        except Exception as exc:                       # noqa: BLE001 - demo resilience
            print(f"[razorpay_link] live call failed ({exc!r}); using simulated id")
            return create_link(invoice_id, amount, live=False, with_url=with_url)

    rec = _CACHE[invoice_id]
    return rec if with_url else rec["id"]


def _find_by_reference(client, reference_id: str) -> dict | None:
    """Look up an already-created live link by its reference_id (one page, most
    recent first - fine for a demo account's link history)."""
    for link in client.payment_link.all({"count": 100}).get("payment_links", []):
        if link.get("reference_id") == reference_id:
            return link
    return None
