"""Payment-link adapter.

Named `razorpay_link` (not `razorpay`) so it does not shadow the `razorpay`
PyPI SDK. One `live` flag serves both paths:

  live=False -> simulated, zero network calls (the batch run)
  live=True  -> one real test-mode link via the SDK (the demo invoice)

The live path is intentionally the only outbound network call in the project.
"""
from __future__ import annotations

import hashlib
import os

_CACHE: dict[str, str] = {}


def create_link(invoice_id: str, amount: float, *, live: bool = False) -> str:
    """Return a payment-link id for `invoice_id`. Deterministic when simulated."""
    if not live:
        h = hashlib.sha1(invoice_id.encode()).hexdigest()[:10]
        return f"plink_sim_{h}"

    if invoice_id in _CACHE:                       # cached fallback on repeat / flaky network
        return _CACHE[invoice_id]
    try:
        import razorpay  # type: ignore

        client = razorpay.Client(auth=(os.environ["RAZORPAY_KEY_ID"],
                                       os.environ["RAZORPAY_KEY_SECRET"]))
        link = client.payment_link.create({
            "amount": int(round(amount * 100)),   # paise
            "currency": "INR",
            "accept_partial": False,
            "reference_id": invoice_id,
            "description": f"Recoverly {invoice_id}",
        })
        _CACHE[invoice_id] = link["id"]
        return link["id"]
    except Exception as exc:                       # noqa: BLE001 - demo resilience
        print(f"[razorpay_link] live call failed ({exc!r}); using simulated id")
        return create_link(invoice_id, amount, live=False)
