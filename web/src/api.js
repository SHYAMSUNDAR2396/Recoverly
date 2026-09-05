const j = (r) => { if (!r.ok) throw new Error(r.status + " " + r.url); return r.json(); };
export const getInvoices = (group) =>
  fetch("/api/invoices" + (group ? `?group=${group}` : "")).then(j);
export const getMetrics = () => fetch("/api/metrics").then(j);
export const getExceptions = () => fetch("/api/exceptions").then(j);
export const getBuyers = () => fetch("/api/buyers").then(j);
export const getBrief = (id) => fetch(`/api/buyers/${id}/brief`).then(j);
export const getAudit = (invoiceId) =>
  fetch(`/api/audit?invoice_id=${encodeURIComponent(invoiceId)}`).then(j);

export const postLiveSend = async (invoiceId, email) => {
  const r = await fetch("/api/demo/live-send", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ invoice_id: invoiceId, email }),
  });
  const data = await r.json().catch(() => null);
  if (!r.ok) throw new Error((data && data.detail) || `${r.status} ${r.statusText}`);
  return data;
};
