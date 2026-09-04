"""Dry-run mailer.

Mirrors razorpay_link.py: simulated by default, so "send the buyer an email" is
a real call site the audit can record without a real email ever leaving the
machine. `live=True` is an intentionally unimplemented guard - wire SES/SMTP
there when the project is ready to actually send.

Recipient addresses in this project use the reserved `.example` TLD, so even a
live path could not reach a real inbox.
"""
from __future__ import annotations

import hashlib

_SENT: list[tuple[str, str, str]] = []      # (to, subject, message_id) - dry-run log


def send(to: str, subject: str, body: str, *, live: bool = False) -> str:
    """Return a message id. Deterministic and side-effect-free when simulated."""
    if not live:
        mid = "msg_sim_" + hashlib.sha1(f"{to}|{subject}".encode()).hexdigest()[:10]
        _SENT.append((to, subject, mid))
        return mid
    raise NotImplementedError(
        "live email is disabled; wire an SES/SMTP client here and remove this guard")


def sent_count() -> int:
    return len(_SENT)


def reset() -> None:
    _SENT.clear()
