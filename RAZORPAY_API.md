# Razorpay APIs — what Recoverly needs, and where to get it

Recoverly deliberately touches Razorpay **once**: a single test-mode Payment Link
created live on camera. Everything else in the pipeline is simulated. This doc
lists the one API you need now and the ones the "with real data" roadmap would
add.

---

## Getting credentials (do this once, ~5 minutes, no KYC)

1. Sign up / log in at **https://dashboard.razorpay.com**.
2. Flip the dashboard to **Test Mode** (toggle at the top). Test mode is free and
   works immediately — no business verification.
3. **Settings → API Keys → Generate Test Key.**
   You get a **Key ID** (`rzp_test_XXXXXXXXXXXXXX`) and a **Key Secret** (shown
   once — copy it now).
4. Export them exactly as `razorpay_link.py` expects:

   ```bash
   export RAZORPAY_KEY_ID=rzp_test_XXXXXXXXXXXXXX
   export RAZORPAY_KEY_SECRET=your_test_secret
   ```

5. Install the SDK (now in `requirements.txt`):

   ```bash
   ./.venv/bin/pip install razorpay
   ```

6. Fire the one live call:

   ```python
   import razorpay_link
   razorpay_link.create_link("INV-2041", 840000.0, live=True)   # returns a plink_... id
   ```

To simulate the buyer paying it: open the link, use test VPA `success@razorpay`
or test card `4111 1111 1111 1111`, any future expiry, any CVV.

---

## 1. Needed now — Payment Links API

| | |
|---|---|
| REST | `POST https://api.razorpay.com/v1/payment_links` |
| Python SDK | `client.payment_link.create({...})` |
| Auth | HTTP Basic: `Key ID` : `Key Secret` (test mode) |
| Used in | `razorpay_link.py` → `create_link(invoice_id, amount, live=True)` |
| Docs | https://razorpay.com/docs/api/payments/payment-links/ |

Payload Recoverly sends (`amount` is in **paise** — multiply rupees by 100):

```json
{
  "amount": 49700000,
  "currency": "INR",
  "accept_partial": false,
  "reference_id": "INV-2032",
  "description": "Recoverly INV-2032",
  "customer": { "name": "Rohan Reddy", "email": "ap@harbour-textiles.example", "contact": "+9198…" },
  "notify": { "sms": true, "email": false },
  "reminder_enable": true
}
```

- `customer` — the buyer's AP contact (synthetic; `.example` TLD so nothing real is contacted).
- `notify.sms: true` — Razorpay texts the link to the contact.
- `notify.email: false` — the **personalized** email is composed and sent by `notify.py`
  (dry-run in this build), not Razorpay's template. Set `email: true` if you'd rather
  Razorpay send its own plain email instead.
- `reminder_enable: true` — Razorpay auto-nudges the link until it's paid.

That is the entire Razorpay surface the current build requires.

---

## 2. Optional — the "real data" roadmap (all currently cut)

| Capability | API | SDK | Where to enable | Why it's cut |
|---|---|---|---|---|
| UPI-first links | `POST /v1/payment_links` with `upi_link: true` | `client.payment_link.create({..., "upi_link": true})` | same test key | the plain link already proves connectivity |
| Poll link status (no webhook) | `GET /v1/payment_links/{id}` | `client.payment_link.fetch(id)` | same test key | demo watches the link on screen instead |
| Poll a payment | `GET /v1/payments/{id}` | `client.payment.fetch(id)` | same test key | not needed for a scripted demo |
| Auto-reconciled inbound NEFT/RTGS/UPI, one virtual account per buyer | `POST /v1/virtual_accounts` (**Smart Collect**) | `client.virtual_account.create({...})` | Dashboard → **request Smart Collect activation** (not on by default, even in test) | scope; it was a pitch highlight, not built |
| Close the loop when money lands | **Webhooks**: `payment_link.paid`, `payment.captured` | n/a (inbound HTTP) | Dashboard → **Settings → Webhooks** → add URL + secret | `api.py` is read-only GET, no webhook server, by design |
| Issue the invoice itself | `POST /v1/invoices` | `client.invoice.create({...})` | same test key | Recoverly starts from an existing ledger |
| Read real settlement data (replace the synthetic ledger) | `GET /v1/settlements`, `GET /v1/transactions` | `client.settlement.all()` | same test key; richer data needs live mode | the whole ledger is synthetic in this build |
| Razorpay MCP server | tools over the same APIs at `mcp.razorpay.com` | MCP client | connect the MCP server | cut in favour of the direct SDK call |

---

## 3. What you do **not** need

- **Live mode / real money** — requires full business KYC and activation. Out of
  scope. The demo is test mode only.
- **Razorpay X / Payouts** (`POST /v1/payouts`) — that's paying money *out*.
  Recoverly only ever creates a request for money to come *in*.
- **Orders API** (`POST /v1/orders`) — needed for the embedded Checkout widget,
  not for hosted Payment Links.

---

## 4. Rate limits & safety

- Test mode has generous limits; the one call per demo is nowhere near them.
- Never commit the Key Secret. It stays in the environment (`RAZORPAY_KEY_SECRET`)
  — `razorpay_link.py` reads it from there and nowhere else.
- `razorpay_link.create_link(..., live=True)` catches any failure and falls back
  to a deterministic simulated id, so a flaky network never breaks the demo.
