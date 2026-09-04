import React, { useEffect, useState } from "react";
import { getInvoices, getMetrics, getExceptions, getBuyers, getBrief, getAudit } from "./api.js";

/* ---------- Razorpay Blade tokens ---------- */
const C = {
  navy: "#021331", blue: "#1364F1", blueDk: "#0E54CD", blueFaintBg: "#F5F8FF",
  blueFaintBd: "#D6E5FF", paper: "#F7F7F7", card: "#FFFFFF", bd: "#DEE1E3",
  bdSoft: "#EEF0F1", ink: "#191D1F", mute: "#7B878E", faint: "#96A0A6",
  navySub: "#8FA5C4", navyAccent: "#A8C8FF",
};
const SEG = {
  on_track: { bg: "#E6F4ED", fg: "#00753B", label: "On track" },
  at_risk:  { bg: "#FFF0E5", fg: "#C75300", label: "At risk" },
  slipping: { bg: "#FFE3CC", fg: "#AD4800", label: "Slipping" },
  overdue:  { bg: "#FBE3E1", fg: "#AA180E", label: "Overdue" },
  chronic:  { bg: "#F6CFCC", fg: "#7A100B", label: "Chronic" },
};
const seg = (dbt) =>
  dbt == null ? "at_risk" : dbt <= 0 ? "on_track" : dbt <= 7 ? "at_risk"
  : dbt <= 20 ? "slipping" : dbt <= 35 ? "overdue" : "chronic";

const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
const rupee = (n) => "₹" + inr.format(n);
const mono = { fontFamily: "Menlo, 'SF Mono', Consolas, monospace" };
const chip = (s) => ({ display: "inline-block", padding: "3px 10px", borderRadius: 9999,
  fontSize: 11, fontWeight: 600, background: s.bg, color: s.fg });
const card = { background: C.card, border: `1px solid ${C.bd}`, borderRadius: 12,
  boxShadow: "0 1px 2px rgba(2,19,49,.04)" };

/* ---------- shell ---------- */
export default function App() {
  const [view, setView] = useState("queue");
  const tab = (id, label) => (
    <button onClick={() => setView(id)} style={{
      background: "none", border: "none", cursor: "pointer",
      borderBottom: `2px solid ${view === id ? C.blue : "transparent"}`,
      padding: "16px 0", marginRight: 32, fontSize: 14,
      fontWeight: view === id ? 600 : 500, color: view === id ? C.blue : "#4F585F",
    }}>{label}</button>
  );
  return (
    <div style={{ minHeight: "100vh", background: C.paper }}>
      <header style={{ background: C.navy, padding: "22px 64px", display: "flex",
        justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <svg width="30" height="30" viewBox="0 0 24 24" aria-hidden="true">
            <rect x="1" y="1" width="22" height="22" rx="6" fill={C.blue} />
            <path d="M7 17.5 L11.5 6.5 L15 6.5 L10.5 17.5 Z" fill="#fff" />
            <path d="M12.5 12.5 L17 12.5 L14.5 17.5 L11 17.5 Z" fill={C.navyAccent} />
          </svg>
          <div>
            <div style={{ fontSize: 20, fontWeight: 600, color: "#fff", letterSpacing: "-.01em" }}>Recoverly</div>
            <div style={{ fontSize: 12, color: C.navyAccent, marginTop: 2 }}>Receivables leverage agent</div>
          </div>
        </div>
        <RunMeta />
      </header>
      <nav style={{ background: C.card, borderBottom: `1px solid ${C.bd}`, padding: "0 64px" }}>
        {tab("queue", "Queue")}
        {tab("results", "Recovery results")}
        {tab("leverage", "Leverage brief")}
      </nav>
      <main style={{ padding: "32px 64px 48px" }}>
        {view === "queue" && <Queue />}
        {view === "results" && <Results />}
        {view === "leverage" && <Leverage />}
      </main>
      <footer style={{ borderTop: `1px solid ${C.bd}`, margin: "0 64px", padding: "16px 0 40px",
        fontSize: 12, color: C.faint }}>
        Recoverly · Razorpay AI Buildathon 2026 · Track 03 — live over results.duckdb
      </footer>
    </div>
  );
}

function RunMeta() {
  const [m, setM] = useState(null);
  useEffect(() => { getMetrics().then(setM).catch(() => {}); }, []);
  const t = m?.metrics?.by_group?.treatment, c = m?.metrics?.by_group?.control;
  return (
    <div style={{ textAlign: "right", fontSize: 12, lineHeight: 1.6, color: C.navySub, ...mono }}>
      seed {m?.seed ?? "—"} · M1 AUC {m?.model1?.auc ?? "—"} · M2 MAE {m?.model2?.mae_days ?? "—"}d<br />
      {t && c ? `treatment ${t.n} / control ${c.n}` : "loading…"}
    </div>
  );
}

/* ---------- shared states ---------- */
const Loading = () => <div style={{ color: C.mute, fontSize: 14 }}>Loading…</div>;
const Err = ({ e }) => (
  <div style={{ ...card, padding: 20, color: "#AA180E", fontSize: 14 }}>
    Could not reach the API ({String(e.message || e)}). Is <code>uvicorn api:app</code> running?
  </div>
);
const H = ({ title, sub }) => (
  <div style={{ marginBottom: 22 }}>
    <div style={{ fontSize: 20, fontWeight: 600, letterSpacing: "-.01em" }}>{title}</div>
    {sub && <div style={{ fontSize: 14, color: C.mute, marginTop: 4 }}>{sub}</div>}
  </div>
);

/* ---------- QUEUE ---------- */
function Queue() {
  const [rows, setRows] = useState(null);
  const [err, setErr] = useState(null);
  const [sel, setSel] = useState(null);
  useEffect(() => {
    getInvoices().then((r) => {
      const open = r.filter((x) => !x.paid || x.escalated);
      const pool = (open.length ? open : r).slice().sort((a, b) => (b.natural_dbt) - (a.natural_dbt));
      setRows(pool);
      setSel(pool[0]?.invoice_id ?? null);
    }).catch(setErr);
  }, []);
  if (err) return <Err e={err} />;
  if (!rows) return <Loading />;

  const counts = rows.reduce((m, x) => { const s = seg(x.natural_dbt); m[s] = (m[s] || 0) + 1; return m; }, {});
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline" }}>
        <H title="Open invoices" sub={`${rows.length} invoices in the recovery queue, ordered by natural days-beyond-terms.`} />
        <div style={{ display: "flex", gap: 8 }}>
          {["chronic", "overdue", "slipping", "at_risk", "on_track"].filter((k) => counts[k])
            .map((k) => <span key={k} style={chip(SEG[k])}>{counts[k]} {SEG[k].label.toLowerCase()}</span>)}
        </div>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1.3fr 1fr", gap: 24 }}>
        <div style={{ ...card, overflow: "hidden", alignSelf: "start" }}>
          <div style={{ display: "grid", gridTemplateColumns: "104px 1.7fr 66px 96px", gap: 14,
            padding: "14px 20px", background: C.paper, borderBottom: `1px solid ${C.bd}`,
            fontSize: 11, fontWeight: 600, color: C.mute, letterSpacing: ".04em", textTransform: "uppercase" }}>
            <span>Invoice</span><span>Buyer</span><span>DBT</span><span>Segment</span>
          </div>
          <div style={{ maxHeight: 620, overflowY: "auto" }}>
            {rows.map((x) => {
              const s = SEG[seg(x.natural_dbt)];
              const on = x.invoice_id === sel;
              return (
                <div key={x.invoice_id} onClick={() => setSel(x.invoice_id)} style={{
                  display: "grid", gridTemplateColumns: "104px 1.7fr 66px 96px", gap: 14,
                  alignItems: "center", padding: "14px 20px", cursor: "pointer",
                  borderBottom: `1px solid ${C.bdSoft}`,
                  borderLeft: `3px solid ${on ? C.blue : "transparent"}`,
                  background: on ? C.blueFaintBg : C.card }}>
                  <span style={{ ...mono, fontSize: 13, color: "#4F585F" }}>{x.invoice_id}</span>
                  <span style={{ fontSize: 14, fontWeight: 500 }}>{x.buyer_id}</span>
                  <span style={{ ...mono, fontSize: 13, color: "#4F585F" }}>
                    {x.natural_dbt >= 0 ? "+" : ""}{x.natural_dbt}d</span>
                  <span style={chip(s)}>{s.label}</span>
                </div>
              );
            })}
          </div>
        </div>
        <InvoiceDetail id={sel} row={rows.find((r) => r.invoice_id === sel)} />
      </div>
    </div>
  );
}

function InvoiceDetail({ id, row }) {
  const [audit, setAudit] = useState(null);
  useEffect(() => { if (id) { setAudit(null); getAudit(id).then(setAudit).catch(() => setAudit([])); } }, [id]);
  if (!row) return <div style={{ ...card, padding: 24 }}><Loading /></div>;
  const last = audit && audit.length ? audit[audit.length - 1] : null;
  const risk = audit && audit.length
    ? Math.max(...audit.map((a) => a.risk_score ?? 0)) : null;
  const field = (k, v, m) => (
    <div><span style={{ display: "block", fontSize: 11, color: C.mute, marginBottom: 3 }}>{k}</span>
      <span style={{ fontSize: 14, ...(m ? mono : {}), fontWeight: m ? 500 : 400 }}>{v}</span></div>
  );
  return (
    <div style={{ ...card, padding: 24, alignSelf: "start" }}>
      <div style={{ fontSize: 18, fontWeight: 600 }}>{row.buyer_id} · {row.invoice_id}</div>
      <div style={{ fontSize: 12, color: C.mute, marginTop: 3 }}>
        {row.group} group · {row.touches} touches · {row.escalated ? "escalated" : row.paid ? "paid" : "open"}
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "14px 20px",
        marginTop: 20, paddingTop: 20, borderTop: `1px solid ${C.bd}` }}>
        {field("Amount", rupee(row.amount), true)}
        {field("Terms", "Net " + row.terms)}
        {field("Due", String(row.due_date).slice(0, 10))}
        {field("Natural DBT", (row.natural_dbt >= 0 ? "+" : "") + row.natural_dbt + " days", true)}
        {field("Paid on", row.paid_day ? String(row.paid_day).slice(0, 10) : "—")}
        {field("Effective DBT", row.effective_dbt == null ? "—"
          : (row.effective_dbt >= 0 ? "+" : "") + row.effective_dbt + " days", true)}
      </div>

      {risk != null && (
        <div style={{ marginTop: 20 }}>
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: C.mute }}>
            <span>Model 1 · P(late), max over touches</span><span style={mono}>{risk.toFixed(2)}</span>
          </div>
          <div style={{ height: 6, background: "#E9EBEC", borderRadius: 9999, marginTop: 6, overflow: "hidden" }}>
            <div style={{ height: 6, width: `${Math.round(risk * 100)}%`, background: C.blue, borderRadius: 9999 }} />
          </div>
        </div>
      )}

      {last && (
        <div style={{ marginTop: 18, padding: "14px 16px", background: C.blueFaintBg,
          border: `1px solid ${C.blueFaintBd}`, borderRadius: 8 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ ...mono, fontSize: 11, fontWeight: 500, color: "#fff",
              background: C.blue, borderRadius: 4, padding: "3px 7px" }}>Stage {last.ladder_stage}</span>
            <span style={{ fontSize: 13, fontWeight: 500, color: "#0A44A9" }}>{last.action_taken}</span>
          </div>
          <div style={{ fontSize: 12, color: "#4F585F", marginTop: 7 }}>diagnosis: <b>{last.diagnosis}</b></div>
          {last.expected_delay_days != null && (
            <div style={{ fontSize: 12, color: "#4F585F", marginTop: 3 }}>
              Model 2 predicted delay ~<b>{last.expected_delay_days}d</b> ({last.risk_severity})
            </div>
          )}
        </div>
      )}

      {(() => {
        const em = audit && [...audit].reverse().find((a) => a.email_body);
        return em ? (
          <div style={{ marginTop: 18, border: `1px solid ${C.bd}`, borderRadius: 8, overflow: "hidden" }}>
            <div style={{ padding: "8px 12px", background: C.paper, borderBottom: `1px solid ${C.bd}`,
              fontSize: 11, color: C.mute }}>
              Email to <b>{em.email_to}</b> · dry-run ({em.email_message_id})
            </div>
            <pre style={{ margin: 0, padding: "12px 14px", fontFamily: "inherit", fontSize: 13,
              lineHeight: 1.6, whiteSpace: "pre-wrap", color: C.ink }}>{em.email_body}</pre>
          </div>
        ) : null;
      })()}

      <div style={{ marginTop: 18, fontSize: 11, color: C.mute, marginBottom: 7 }}>Audit trail</div>
      {!audit ? <Loading /> : audit.length === 0
        ? <div style={{ fontSize: 13, color: C.mute }}>No agent actions recorded (control group, or never reached a rung).</div>
        : (
          <div style={{ display: "grid", gap: 8 }}>
            {audit.map((a, i) => (
              <div key={i} style={{ border: `1px solid ${C.bd}`, borderRadius: 8, padding: "10px 12px",
                background: "#FCFCFD" }}>
                <div style={{ ...mono, fontSize: 11, color: C.faint }}>
                  {String(a.timestamp).replace("T", " ").slice(0, 16)} · stage {a.ladder_stage} · {a.outcome}
                  {a.human_gate_required ? " · HUMAN GATE" : ""}
                </div>
                <div style={{ fontSize: 13, marginTop: 4 }}>{a.action_taken} — {a.message_sent}</div>
                {a.bounds_checked && (
                  <div style={{ ...mono, fontSize: 10.5, color: C.mute, marginTop: 4 }}>
                    bounds ✓ {a.bounds_checked}{a.razorpay_object_id ? ` · ${a.razorpay_object_id}` : ""}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
    </div>
  );
}

/* ---------- RESULTS ---------- */
function Results() {
  const [m, setM] = useState(null);
  const [ex, setEx] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    Promise.all([getMetrics(), getExceptions()]).then(([mm, ee]) => { setM(mm); setEx(ee); }).catch(setErr);
  }, []);
  if (err) return <Err e={err} />;
  if (!m) return <Loading />;
  const M = m.metrics, g = M.by_group;
  const stat = (label, value, sub, color) => (
    <div style={{ ...card, padding: 20 }}>
      <div style={{ fontSize: 11, color: C.mute }}>{label}</div>
      <div style={{ fontSize: 30, fontWeight: 600, marginTop: 8, letterSpacing: "-.02em", color: color || C.ink }}>{value}</div>
      <div style={{ fontSize: 12, color: C.mute, marginTop: 6 }}>{sub}</div>
    </div>
  );
  const rowsT = [
    ["Invoices", g.treatment.n, g.control.n],
    ["Paid", g.treatment.paid, g.control.paid],
    ["₹ recovered (at-risk)", rupee(g.treatment.rupees_recovered), rupee(g.control.rupees_recovered)],
    ["% of at-risk value recovered", pct(g.treatment.pct_at_risk_recovered), pct(g.control.pct_at_risk_recovered)],
    ["Mean DSO (days)", g.treatment.mean_dso_days, g.control.mean_dso_days],
    ["Mean DBT at payment (days)", g.treatment.mean_dbt_days, g.control.mean_dbt_days],
    ["Escalated to human", g.treatment.escalated, g.control.escalated],
  ];
  return (
    <div>
      <H title="Recovery outcome" sub="Treatment group measured against a matched, untouched control." />
      <div style={{ display: "grid", gridTemplateColumns: "repeat(4,minmax(0,1fr))", gap: 20 }}>
        {stat("DSO reduction", (M.dso_reduction_days ?? "—") + " days", "control minus treatment", C.ink)}
        {stat("Cash pulled forward", rupee(M.cash_pulled_forward_rupee_days) + "·d", "rupee-days of acceleration", C.ink)}
        {stat("Cash-acceleration value", rupee(M.cash_acceleration_value), `at ${pct(M.cost_of_capital)} cost of capital`, "#00753B")}
        {stat("Net benefit", rupee(M.net_benefit), `after ${rupee(M.discount_cost)} discount cost`, M.net_benefit >= 0 ? "#00753B" : "#AA180E")}
      </div>

      <div style={{ ...card, overflow: "hidden", marginTop: 22 }}>
        <div style={{ display: "grid", gridTemplateColumns: "1.8fr 1fr 1fr", gap: 16, padding: "14px 20px",
          background: C.paper, borderBottom: `1px solid ${C.bd}`, fontSize: 11, fontWeight: 600,
          color: C.mute, letterSpacing: ".04em", textTransform: "uppercase" }}>
          <span>Metric</span><span>Treatment</span><span>Control</span>
        </div>
        {rowsT.map(([k, t, c], i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "1.8fr 1fr 1fr", gap: 16,
            padding: "13px 20px", borderBottom: `1px solid ${C.bdSoft}`, fontSize: 14 }}>
            <span>{k}</span>
            <span style={{ ...mono, fontWeight: 500 }}>{t}</span>
            <span style={{ ...mono, color: C.mute }}>{c}</span>
          </div>
        ))}
      </div>

      <div style={{ display: "flex", gap: 12, background: "#E7F7FD", border: "1px solid #BEE7F8",
        borderRadius: 8, padding: "14px 16px", marginTop: 18, maxWidth: 900 }}>
        <div style={{ fontSize: 13, lineHeight: 1.65, color: "#00567F" }}>
          <b>{M.interpretation}</b><br />{M.note}
        </div>
      </div>

      <div style={{ fontSize: 16, fontWeight: 600, marginTop: 32 }}>Exceptions</div>
      <div style={{ fontSize: 14, color: C.mute, marginTop: 4 }}>
        {ex?.length ?? 0} invoices the agent could not resolve, honestly labelled.
      </div>
      <div style={{ ...card, overflow: "hidden", marginTop: 14 }}>
        {(ex || []).map((e, i) => (
          <div key={i} style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 20,
            padding: "14px 20px", borderBottom: `1px solid ${C.bdSoft}`, fontSize: 14, alignItems: "start" }}>
            <span style={{ ...mono, fontWeight: 500 }}>{rupee(e.amount)}</span>
            <span><b style={mono}>{e.invoice_id}</b> · {e.buyer_id} — {e.reason}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
const pct = (x) => x == null ? "—" : (x * 100).toFixed(0) + "%";

/* ---------- LEVERAGE ---------- */
function Leverage() {
  const [buyers, setBuyers] = useState(null);
  const [sel, setSel] = useState(null);
  const [brief, setBrief] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    getBuyers().then((b) => {
      const elig = b.filter((x) => x.n_invoices >= 20);
      setBuyers(elig);
      setSel(elig[0]?.buyer_id ?? null);
    }).catch(setErr);
  }, []);
  useEffect(() => { if (sel) { setBrief(null); getBrief(sel).then(setBrief).catch(setErr); } }, [sel]);
  if (err) return <Err e={err} />;
  if (!buyers) return <Loading />;

  return (
    <div>
      <H title="Leverage brief" sub="The negotiating position, built from this buyer's own payment record." />
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {buyers.map((b) => {
          const on = b.buyer_id === sel;
          return (
            <button key={b.buyer_id} onClick={() => setSel(b.buyer_id)} style={{
              fontSize: 14, fontWeight: on ? 600 : 500, padding: "9px 16px", borderRadius: 8,
              cursor: "pointer", border: `1px solid ${on ? C.blue : "#C8CDD0"}`,
              background: on ? C.blue : C.card, color: on ? "#fff" : "#4F585F" }}>
              {b.name}
            </button>
          );
        })}
      </div>

      {!brief ? <div style={{ marginTop: 20 }}><Loading /></div> : (
        <div style={{ maxWidth: 860, marginTop: 20, ...card, overflow: "hidden" }}>
          <div style={{ background: C.navy, padding: "26px 36px" }}>
            <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: ".06em", textTransform: "uppercase", color: C.navyAccent }}>
              Terms-revision brief · {brief.n_invoices} invoices
            </div>
            <div style={{ fontSize: 24, fontWeight: 600, color: "#fff", marginTop: 10, letterSpacing: "-.015em" }}>
              {brief.buyer} — {brief.recommended_terms.split(".")[0]}
            </div>
            <div style={{ fontSize: 13, color: C.navySub, marginTop: 6 }}>{brief.evidence_line}</div>
          </div>
          <div style={{ padding: "28px 36px 32px" }}>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(3,minmax(0,1fr))", gap: 16 }}>
              {metric("Mean days beyond terms", (brief.mean_dbt >= 0 ? "+" : "") + brief.mean_dbt + "d", "σ " + brief.sd_dbt + "d")}
              {metric("Promise-kept rate", brief.promise_kept_rate == null ? "—" : brief.promise_kept_rate,
                brief.promise_kept_rate == null ? "no promises on record" : "of recorded promises")}
              {metric("Dispute rate", pct(brief.dispute_rate), `${brief.honored} honoured / ${brief.broken} broken`)}
            </div>
            <div style={{ fontSize: 15, lineHeight: 1.7, marginTop: 24 }}>
              Peers on Net {brief.terms} settle at {brief.peer_dbt >= 0 ? "+" : ""}{brief.peer_dbt} days on average.
            </div>
            <div style={{ background: C.blueFaintBg, border: `1px solid ${C.blueFaintBd}`, borderRadius: 8,
              padding: "20px 22px", marginTop: 22 }}>
              <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: ".05em", textTransform: "uppercase", color: C.blueDk }}>
                Recommended terms
              </div>
              <div style={{ fontSize: 16, lineHeight: 1.65, color: C.navy, marginTop: 9, fontWeight: 500 }}>
                {brief.recommended_terms}
              </div>
            </div>
            <div style={{ marginTop: 22 }}>
              <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: ".04em", textTransform: "uppercase", color: C.mute }}>
                Justification
              </div>
              <div style={{ fontSize: 14, lineHeight: 1.75, color: "#4F585F", marginTop: 9 }}>{brief.justification}</div>
            </div>
            <div style={{ border: `1px solid ${C.bd}`, background: "#FCFCFD", borderRadius: 8,
              padding: "18px 20px", marginTop: 22 }}>
              <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: ".04em", textTransform: "uppercase",
                color: C.mute, marginBottom: 10 }}>Message to buyer's AP lead</div>
              <div style={{ fontSize: 13, lineHeight: 1.75, whiteSpace: "pre-line" }}>{brief.message}</div>
              <button style={{ marginTop: 16, fontSize: 14, fontWeight: 600, background: C.blue, color: "#fff",
                border: "none", borderRadius: 8, padding: "11px 20px", cursor: "pointer" }}>Approve &amp; send</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
function metric(label, value, sub) {
  return (
    <div style={{ background: C.paper, borderRadius: 8, padding: 16 }}>
      <div style={{ fontSize: 11, color: C.mute }}>{label}</div>
      <div style={{ fontSize: 24, fontWeight: 600, marginTop: 6, letterSpacing: "-.01em" }}>{value}</div>
      <div style={{ fontSize: 12, color: C.mute, marginTop: 3 }}>{sub}</div>
    </div>
  );
}
