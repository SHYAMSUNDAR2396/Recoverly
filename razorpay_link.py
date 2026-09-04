"""Payment-link adapter.

Named `razorpay_link` (not `razorpay`) so it does not shadow the `razorpay`
PyPI SDK. One `live` flag serves both paths:

  live=False -> simulated, zero network calls (the batch run)
  live=True  -> one real test-mode link via the SDK (the demo invoice)

The live path is intentionally the only outbound network call in the project.

When `customer` is supplied, the link carries recipient details so Razorpay
sends the SMS nudge and its own escalating reminders. The personalised email is
composed and (dry-run) sent by notify.py, not by Razorpay - hence the default
`notify` has email off.
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
    only sent to Razorpay on the live path; ignored when simulated.
    """
    if not live:
        h = hashlib.sha1(invoice_id.encode()).hexdigest()[:10]
        rec = {"id": f"plink_sim_{h}", "short_url": f"https://rzp.io/i/{h}"}
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
            link = client.payment_link.create(payload)
            _CACHE[invoice_id] = {"id": link["id"], "short_url": link.get("short_url", "")}
        except Exception as exc:                       # noqa: BLE001 - demo resilience
            print(f"[razorpay_link] live call failed ({exc!r}); using simulated id")
            return create_link(invoice_id, amount, live=False, with_url=with_url)

    rec = _CACHE[invoice_id]
    return rec if with_url else rec["id"]
