"""Mailer.

Mirrors razorpay_link.py: simulated by default, so "send the buyer an email" is
a real call site the audit can record without a real email ever leaving the
machine. `live=True` sends for real over SMTP (stdlib smtplib), reading
credentials from the environment only (SMTP_USER / SMTP_PASSWORD, optionally
SMTP_HOST / SMTP_PORT for a non-Gmail provider) - same pattern as
razorpay_link.py's RAZORPAY_KEY_ID / RAZORPAY_KEY_SECRET. Any failure falls
back to the dry-run path so a demo never breaks.

Recipient addresses in the simulated ledger use the reserved `.example` TLD, so
the dry-run path could not reach a real inbox even if `live=True` were passed
for one of them by mistake. The live path is only ever used for one
operator-chosen demo address (engine.run(live_email_to=...)).
"""
from __future__ import annotations

import hashlib
import os

_SENT: list[tuple[str, str, str]] = []      # (to, subject, message_id) - send log


def send(to: str, subject: str, body: str, *, live: bool = False) -> str:
    """Return a message id. Deterministic and side-effect-free when simulated."""
    if not live:
        mid = "msg_sim_" + hashlib.sha1(f"{to}|{subject}".encode()).hexdigest()[:10]
        _SENT.append((to, subject, mid))
        return mid

    try:
        import smtplib
        from email.mime.text import MIMEText

        host = os.environ.get("SMTP_HOST", "smtp.gmail.com")
        port = int(os.environ.get("SMTP_PORT", "465"))
        user, password = os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"]

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to

        with smtplib.SMTP_SSL(host, port) as s:
            s.login(user, password)
            s.sendmail(user, [to], msg.as_string())

        mid = "msg_live_" + hashlib.sha1(f"{to}|{subject}".encode()).hexdigest()[:10]
        _SENT.append((to, subject, mid))
        return mid
    except Exception as exc:                       # noqa: BLE001 - demo resilience
        print(f"[notify] live send failed ({exc!r}); falling back to dry-run")
        return send(to, subject, body, live=False)


def sent_count() -> int:
    return len(_SENT)


def reset() -> None:
    _SENT.clear()
