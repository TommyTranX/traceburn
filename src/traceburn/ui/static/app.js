/* traceburn viewer. Plain JS on purpose: no build step, no dependencies,
   nothing fetched from outside 127.0.0.1. Edit and reload. */

"use strict";

const $ = (sel) => document.querySelector(sel);
// state.seq increments on every trace selection and tab switch; async
// renderers capture it and bail if it moved, so a slow response can never
// paint over a newer view.
const state = { traces: [], current: null, tab: "tree", spans: [], seq: 0 };

/* ---------- utilities ---------- */

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function fmtMs(ns) {
  if (ns === null || ns === undefined) return "-";
  const ms = ns / 1e6;
  if (ms < 9.95) return ms.toFixed(1) + "ms";
  if (ms < 999.5) return Math.round(ms) + "ms";
  const s = ms / 1000;
  if (Math.round(s * 10) / 10 < 60) return s.toFixed(1) + "s";
  const total = Math.round(s);
  return Math.floor(total / 60) + "m" + (total % 60) + "s";
}

function fmtCost(v) {
  if (!v) return "-";
  return v >= 1 ? "$" + v.toFixed(2) : "$" + v.toFixed(4);
}

function api(path) {
  return fetch(path).then((r) => {
    if (!r.ok) throw new Error(path + " -> " + r.status);
    return r.json();
  });
}

function showError(err) {
  const view = $("#view");
  view.textContent = "";
  const box = el("div", "finding high");
  box.appendChild(el("div", "f-head", "could not reach the traceburn server"));
  box.appendChild(el("div", "f-expl",
    String(err) + " (is `traceburn ui` still running? reload after restarting it)"));
  view.appendChild(box);
}

function spanTokens(attrs) {
  const inn = (attrs["gen_ai.usage.input_tokens"] || 0) +
    (attrs["cached_input_tokens"] || 0) + (attrs["cache_write_tokens"] || 0);
  const out = attrs["gen_ai.usage.output_tokens"] || 0;
  return inn || out ? `${inn}in/${out}out` : "";
}

/* ---------- sidebar ---------- */

async function loadTraces() {
  const data = await api("/api/traces");
  state.traces = data.traces;
  const list = $("#trace-list");
  list.textContent = "";
  for (const t of state.traces) {
    const li = el("li");
    li.dataset.id = t.trace_id;
    li.appendChild(el("div", "row1", t.name));
    const r2 = el("div", "row2");
    r2.appendChild(el("span", "", t.trace_id.slice(0, 10)));
    r2.appendChild(el("span", "", `${t.stats.span_count} spans`));
    r2.appendChild(el("span", "", fmtCost(t.stats.cost_usd)));
    li.appendChild(r2);
    li.onclick = () => selectTrace(t.trace_id);
    list.appendChild(li);
  }
}

async function selectTrace(traceId) {
  const seq = ++state.seq;
  state.current = state.traces.find((t) => t.trace_id === traceId);
  document.querySelectorAll("#trace-list li").forEach((li) =>
    li.classList.toggle("active", li.dataset.id === traceId));
  $("#empty").classList.add("hidden");
  $("#tabs").classList.remove("hidden");
  $("#trace-header").classList.remove("hidden");
  const t = state.current;
  $("#trace-title").textContent = t.name + "  (" + t.trace_id.slice(0, 12) + ")";
  const s = t.stats;
  $("#trace-stats").textContent =
    `${s.span_count} spans, ${s.llm_count || 0} llm calls, ${s.error_count || 0} errors, ` +
    `${fmtMs(t.end_ns ? t.end_ns - t.start_ns : null)}, ${fmtCost(s.cost_usd)} estimated`;
  try {
    const data = await api(`/api/traces/${t.trace_id}/spans`);
    if (seq !== state.seq) return;   // a newer selection won
    state.spans = data.spans;
    renderTab();
  } catch (err) {
    if (seq === state.seq) showError(err);
  }
}

/* ---------- tabs ---------- */

$("#tabs").addEventListener("click", (e) => {
  if (e.target.tagName !== "BUTTON") return;
  state.tab = e.target.dataset.tab;
  document.querySelectorAll("#tabs button").forEach((b) =>
    b.classList.toggle("active", b === e.target));
  renderTab();
});

function renderTab() {
  const view = $("#view");
  view.textContent = "";
  closeDetail();
  if (!state.current) return;
  const seq = ++state.seq;
  let job;
  if (state.tab === "tree") job = Promise.resolve(renderTree(view, seq));
  else if (state.tab === "flame") job = renderFlame(view, "latency", seq);
  else if (state.tab === "waterfall") job = renderWaterfall(view, seq);
  else if (state.tab === "waste") job = renderWaste(view, seq);
  else if (state.tab === "diff") job = Promise.resolve(renderDiff(view, seq));
  if (job) job.catch((err) => { if (seq === state.seq) showError(err); });
}

/* ---------- tree ---------- */

function renderTree(view, seq) {
  const byParent = new Map();
  const ids = new Set(state.spans.map((s) => s.span_id));
  for (const s of state.spans) {
    const parent = ids.has(s.parent_id) ? s.parent_id : null;
    if (!byParent.has(parent)) byParent.set(parent, []);
    byParent.get(parent).push(s);
  }
  const build = (parentId, container, depth) => {
    const kids = (byParent.get(parentId) || []).sort((a, b) => a.start_ns - b.start_ns);
    for (const s of kids) {
      const wrap = el("div", "tree-node" + (depth === 0 ? " root" : ""));
      const line = el("div", "span-line");
      const hasKids = byParent.has(s.span_id);
      const tog = el("span", "toggle", hasKids ? "▾" : " ");
      line.appendChild(tog);
      line.appendChild(el("span", "kind " + s.kind, s.kind));
      line.appendChild(el("span", "name", s.name));
      line.appendChild(el("span", "muted", fmtMs(s.end_ns ? s.end_ns - s.start_ns : null)));
      const tokens = spanTokens(s.attributes);
      if (tokens) line.appendChild(el("span", "muted", tokens));
      if (s.attributes.cost_usd) line.appendChild(el("span", "cost", fmtCost(s.attributes.cost_usd)));
      if (s.status === "error") line.appendChild(el("span", "err", "ERROR"));
      line.onclick = (e) => { e.stopPropagation(); showDetail(s); };
      wrap.appendChild(line);
      const childBox = el("div");
      wrap.appendChild(childBox);
      build(s.span_id, childBox, depth + 1);
      if (hasKids) {
        tog.onclick = (e) => {
          e.stopPropagation();
          const hidden = childBox.classList.toggle("hidden");
          tog.textContent = hidden ? "▸" : "▾";
        };
      }
      container.appendChild(wrap);
    }
  };
  build(null, view, 0);
}

/* ---------- flamegraph ---------- */

const KIND_COLORS = {
  llm: "#e8833a", tool: "#4c9be8", agent: "#8f7ae8",
  retrieval: "#3fae8a", custom: "#7d8697", trace: "#3a3f4a",
};

async function renderFlame(view, weight, seq) {
  if (seq === undefined) seq = state.seq;
  const tree = await api(`/api/traces/${state.current.trace_id}/flamegraph?weight=${weight}`);
  if (seq !== state.seq) return;
  view.textContent = "";
  const controls = el("div", "flame-controls");
  for (const w of ["latency", "cost"]) {
    const b = el("button", w === weight ? "active" : "", w);
    b.onclick = () => { renderFlame(view, w, ++state.seq).catch(showError); };
    controls.appendChild(b);
  }
  controls.appendChild(el("span", "muted", "click a frame to inspect; width is " + weight));
  view.appendChild(controls);

  if (!tree.value) { view.appendChild(el("p", "muted", "nothing to draw")); return; }
  const box = el("div");
  view.appendChild(box);
  const fmt = weight === "latency" ? fmtMs : fmtCost;

  // Icicle layout: one row per depth. A child row is confined to its
  // parent's extent: concurrent children can sum past the parent's
  // wall-clock, so they are scaled down to fit rather than overflowing.
  const MIN_W = 0.2;
  const rows = [];
  const place = (node, depth, offset, width) => {
    if (!rows[depth]) rows[depth] = [];
    const drawn = Math.max(width, MIN_W);
    rows[depth].push({ node, offset, width: drawn });
    const childSum = node.children.reduce((acc, c) => acc + c.value, 0);
    if (!childSum) return;
    const scale = drawn / Math.max(node.value, childSum);
    let childOffset = offset;
    for (const child of node.children) {
      const w = child.value * scale;
      place(child, depth + 1, childOffset, w);
      childOffset += Math.max(w, MIN_W);
    }
  };
  place(tree, 0, 0, 100);

  for (const row of rows) {
    const rowEl = el("div", "flame-row");
    let cursor = 0;
    for (const { node, offset, width } of row) {
      if (offset > cursor) {
        const gap = el("div", "frame spacer");
        gap.style.width = (offset - cursor) + "%";
        rowEl.appendChild(gap);
      }
      const frame = el("div", "frame", node.name);
      frame.style.width = width + "%";
      frame.style.background = KIND_COLORS[node.kind] || KIND_COLORS.custom;
      frame.title = `${node.name}\n${weight}: ${fmt(node.value)} (self ${fmt(node.self_value)})`;
      const span = state.spans.find((s) => s.span_id === node.span_id);
      if (span) frame.onclick = () => showDetail(span);
      rowEl.appendChild(frame);
      cursor = offset + width;
    }
    box.appendChild(rowEl);
  }
}

/* ---------- waterfall ---------- */

async function renderWaterfall(view, seq) {
  const data = await api(`/api/traces/${state.current.trace_id}/waterfall`);
  if (seq !== undefined && seq !== state.seq) return;
  const rows = data.rows;
  if (!rows.length) { view.appendChild(el("p", "muted", "no spans")); return; }
  const t0 = Math.min(...rows.map((r) => r.start_ns));
  const t1 = Math.max(...rows.map((r) => r.end_ns || r.start_ns));
  const total = Math.max(t1 - t0, 1);
  for (const r of rows) {
    const rowEl = el("div", "wf-row");
    const label = el("div", "wf-label");
    label.style.paddingLeft = r.depth * 14 + "px";
    label.textContent = r.name;
    rowEl.appendChild(label);
    const track = el("div", "wf-track");
    const bar = el("div", "wf-bar");
    bar.style.left = ((r.start_ns - t0) / total) * 100 + "%";
    bar.style.width = Math.max(((r.duration_ns || 0) / total) * 100, 0.3) + "%";
    bar.style.background = KIND_COLORS[r.kind] || KIND_COLORS.custom;
    track.appendChild(bar);
    rowEl.appendChild(track);
    const meta = el("div", "wf-meta", `${fmtMs(r.duration_ns)}  ${fmtCost(r.cost_usd)}`);
    rowEl.appendChild(meta);
    const span = state.spans.find((s) => s.span_id === r.span_id);
    if (span) rowEl.onclick = () => showDetail(span);
    view.appendChild(rowEl);
  }
}

/* ---------- waste ---------- */

async function renderWaste(view, seq) {
  const report = await api(`/api/traces/${state.current.trace_id}/waste`);
  if (seq !== undefined && seq !== state.seq) return;
  view.appendChild(el("div", "headline", report.headline));
  if (!report.findings.length) return;
  for (const f of report.findings) {
    const card = el("div", "finding " + f.severity);
    const head = el("div", "f-head");
    head.appendChild(el("span", "sev", f.severity));
    head.appendChild(el("span", "muted", f.rule_id));
    head.appendChild(el("span", "", f.summary));
    const amounts = [];
    if (f.avoidable_usd) amounts.push("~" + fmtCost(f.avoidable_usd));
    if (f.avoidable_tokens) amounts.push(f.avoidable_tokens + " tokens");
    if (f.avoidable_seconds) amounts.push(f.avoidable_seconds.toFixed(1) + "s");
    if (amounts.length) head.appendChild(el("span", "f-amounts", amounts.join(", ")));
    card.appendChild(head);
    card.appendChild(el("div", "f-expl", f.explanation));
    card.appendChild(el("div", "f-foot",
      `confidence: ${f.confidence}; figures are estimates. spans: ` +
      f.span_ids.slice(0, 6).map((s) => s.slice(0, 10)).join(", ") +
      (f.span_ids.length > 6 ? ` (+${f.span_ids.length - 6} more)` : "")));
    card.onclick = () => {
      const span = state.spans.find((s) => s.span_id === f.span_ids[0]);
      if (span) showDetail(span);
    };
    view.appendChild(card);
  }
}

/* ---------- diff ---------- */

function renderDiff(view, seq) {
  const pick = el("div", "diff-pick");
  const selA = el("select"), selB = el("select");
  for (const sel of [selA, selB]) {
    for (const t of state.traces) {
      const opt = el("option", "", `${t.name} (${t.trace_id.slice(0, 10)})`);
      opt.value = t.trace_id;
      sel.appendChild(opt);
    }
  }
  selA.value = state.current.trace_id;
  const others = state.traces.filter((t) => t.trace_id !== state.current.trace_id);
  if (others.length) selB.value = others[0].trace_id;
  const arrow = el("span", "muted", "vs");
  const go = el("button", "", "compare");
  pick.append(selA, arrow, selB, go);
  view.appendChild(pick);
  const out = el("div");
  view.appendChild(out);

  const run = async () => {
    out.textContent = "";
    let d;
    try {
      d = await api(`/api/diff?a=${selA.value}&b=${selB.value}`);
    } catch (err) {
      out.appendChild(el("p", "muted", "diff failed: " + err));
      return;
    }
    if (seq !== undefined && seq !== state.seq) return;
    const table = el("table", "totals");
    const header = el("tr");
    for (const h of ["", "spans", "llm", "errors", "time", "tokens in/out", "est. cost"])
      header.appendChild(el("th", "", h));
    table.appendChild(header);
    for (const [label, t] of [["a", d.totals_a], ["b", d.totals_b]]) {
      const tr = el("tr");
      tr.appendChild(el("td", "", label));
      tr.appendChild(el("td", "", t.spans));
      tr.appendChild(el("td", "", t.llm_calls));
      tr.appendChild(el("td", "", t.errors));
      tr.appendChild(el("td", "", (t.duration_ms / 1000).toFixed(1) + "s"));
      tr.appendChild(el("td", "", `${t.input_tokens}/${t.output_tokens}`));
      tr.appendChild(el("td", "", fmtCost(t.cost_usd)));
      table.appendChild(tr);
    }
    const dc = d.totals_b.cost_usd - d.totals_a.cost_usd;
    const dt = (d.totals_b.duration_ms - d.totals_a.duration_ms) / 1000;
    const tr = el("tr");
    tr.appendChild(el("td", "", "delta"));
    tr.append(el("td"), el("td"), el("td"));
    tr.appendChild(el("td", dt <= 0 ? "delta-neg" : "delta-pos", (dt >= 0 ? "+" : "") + dt.toFixed(1) + "s"));
    tr.appendChild(el("td"));
    tr.appendChild(el("td", dc <= 0 ? "delta-neg" : "delta-pos",
      Math.abs(dc) < 5e-5 ? "no change" : (dc > 0 ? "+" : "-") + fmtCost(Math.abs(dc))));
    table.appendChild(tr);
    out.appendChild(table);

    const changed = d.matched.filter((m) =>
      Math.abs(m.deltas.duration_ms) >= 50 || Math.abs(m.deltas.cost_usd) >= 5e-5 ||
      m.deltas.input_tokens || m.deltas.output_tokens || m.request_diff || m.response_diff);
    if (changed.length) out.appendChild(el("div", "side-head", "changed steps"));
    for (const m of changed) {
      const row = el("div", "diff-step");
      row.appendChild(el("span", "name", m.name));
      const bits = [];
      if (Math.abs(m.deltas.duration_ms) >= 50) bits.push((m.deltas.duration_ms / 1000).toFixed(2) + "s");
      if (Math.abs(m.deltas.cost_usd) >= 5e-5) bits.push((m.deltas.cost_usd > 0 ? "+" : "") + m.deltas.cost_usd.toFixed(4) + "$");
      if (m.deltas.input_tokens) bits.push(m.deltas.input_tokens + " in");
      if (m.deltas.output_tokens) bits.push(m.deltas.output_tokens + " out");
      row.appendChild(el("span", "muted", bits.join(", ") || "text changed"));
      out.appendChild(row);
      for (const key of ["request_diff", "response_diff"]) {
        if (!m[key]) continue;
        const pre = el("pre", "textdiff");
        for (const line of m[key].slice(0, 80)) {
          const ln = el("div", line.startsWith("+") ? "add" : line.startsWith("-") ? "del" : "", line);
          pre.appendChild(ln);
        }
        out.appendChild(pre);
      }
    }
    for (const [label, items] of [["only in b (added)", d.added], ["only in a (removed)", d.removed]]) {
      if (!items.length) continue;
      out.appendChild(el("div", "side-head", label));
      for (const s of items) {
        const row = el("div", "diff-step");
        row.appendChild(el("span", "name", s.name));
        row.appendChild(el("span", "muted", `${s.kind}  ${s.duration_ms ? s.duration_ms.toFixed(0) + "ms" : "-"}  ${fmtCost(s.cost_usd)}`));
        out.appendChild(row);
      }
    }
  };
  if (others.length) {
    go.onclick = run;
    run();
  } else {
    go.disabled = true;
    out.appendChild(el("p", "muted", "record a second trace to compare"));
  }
}

/* ---------- span detail drawer ---------- */

function showDetail(span) {
  $("#detail").classList.remove("hidden");
  $("#detail-title").textContent = span.name;
  const body = $("#detail-body");
  body.textContent = "";
  const attrs = span.attributes || {};
  const kv = el("div", "kv");
  const pairs = [
    ["span id", span.span_id], ["kind", span.kind], ["status", span.status],
    ["duration", fmtMs(span.end_ns ? span.end_ns - span.start_ns : null)],
    ["model", attrs["gen_ai.response.model"] || attrs["gen_ai.request.model"] || "-"],
    ["input tokens", attrs["gen_ai.usage.input_tokens"]],
    ["cached input", attrs["cached_input_tokens"]],
    ["cache write", attrs["cache_write_tokens"]],
    ["output tokens", attrs["gen_ai.usage.output_tokens"]],
    ["est. cost", fmtCost(attrs.cost_usd)],
    ["finish reason", attrs.finish_reason],
    ["usage estimated", attrs.usage_estimated ? "yes" : undefined],
    ["error", span.error],
  ];
  for (const [k, v] of pairs) {
    if (v === undefined || v === null) continue;
    kv.appendChild(el("div", "k", k));
    kv.appendChild(el("div", "v", String(v)));
  }
  body.appendChild(kv);
  for (const key of ["request", "response"]) {
    const value = attrs[key];
    if (value === undefined || value === null) continue;
    body.appendChild(el("h4", "", key));
    body.appendChild(el("pre", "", JSON.stringify(value, null, 2)));
  }
}

function closeDetail() { $("#detail").classList.add("hidden"); }
$("#detail-close").onclick = closeDetail;

/* ---------- boot ---------- */

api("/api/meta")
  .then((m) => { $("#meta").textContent = `v${m.version}  ${m.db}`; })
  .catch(() => { $("#meta").textContent = "server unreachable"; });
loadTraces().catch(showError);
