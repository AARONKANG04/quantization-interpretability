/* Trace browser for the quantization-interpretability evals and token KL dumps. Vanilla JS, no build step. */
"use strict";

// ---------------------------------------------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------------------------------------------

const $ = (sel, el = document) => el.querySelector(sel);
const LETTERS = "ABCDEFGHIJ";
const ROW_H = 58;                                   // sidebar row height, px (virtual list)
const KL_CAP = 1, KL_FLOOR = 0.01, MAX_ALPHA = 0.72;  // token colour: full colour at KL >= 1 nat
const STATUS = {
  all: "All", flip_to_wrong: "Flipped to wrong", flip_to_right: "Flipped to right",
  both_wrong: "Both wrong", both_correct: "Both right", correct: "Correct", wrong: "Wrong",
};
const SORTS = { doc: "Doc id", disagree: "Disagreement", length: "Output length", gold_drop: "Gold log-lik drop" };
const VARIANTS = { "strict-match": "strict", "flexible-extract": "flexible", acc: "acc", acc_norm: "acc_norm" };
const DOC_SORTS = [["mean_kl", "Mean KL"], ["max_kl", "Max KL"], ["flip_rate", "Top-1 flips"], ["length", "Length"], ["doc", "Doc id"]];

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmtInt = (n) => Number(n ?? 0).toLocaleString("en-US");
const fmtPct = (x) => (x == null ? "n/a" : `${(100 * x).toFixed(1)}%`);
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
function fmtNum(x) {
  if (x == null || Number.isNaN(x)) return "n/a";
  const a = Math.abs(x);
  if (a === 0) return "0";
  if (a >= 100) return x.toFixed(0);
  if (a >= 10) return x.toFixed(1);
  if (a >= 1) return x.toFixed(2);
  return x.toPrecision(3);
}
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };
const later = (fn, ms) => { const t = setTimeout(fn, ms); return () => clearTimeout(t); };

async function api(path, params = {}, signal) {
  const url = new URL(path, location.origin);
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") url.searchParams.set(k, v);
  const r = await fetch(url, { signal });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = (await r.json()).error || msg; } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return r.json();
}

// ---------------------------------------------------------------------------------------------------------------
// Preferences (localStorage) and view state (URL hash)
// ---------------------------------------------------------------------------------------------------------------

const PREFS_KEY = "qi-webui:prefs";
const prefs = { layout: "stacked", fewshot: false, divergence: true, font: 14, scale: "log", window: 2048, sidebar: true };
try { Object.assign(prefs, JSON.parse(localStorage.getItem(PREFS_KEY) || "{}")); } catch { /* ignore */ }
const savePrefs = () => { try { localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch { /* ignore */ } };

const state = {
  task: null,          // lm-eval task id, or "tokens"
  conds: [],           // selected conditions, the first one is the reference
  per: {},             // task -> filters and current document
  tokens: { dump: null, sort: "mean_kl", source: "", doc: null, start: 0 },
  meta: null,          // /api/conditions
  items: null,         // /api/items for the current task and filters
  item: null,          // /api/item for the current document
  cursor: -1,          // index of the current row in the sidebar list
  dumps: null, docs: null, doc: null,
};

function taskState(task = state.task) {
  if (!state.per[task]) state.per[task] = { filter: "", status: "all", q: "", subject: "", sort: "doc", compare: "", doc: null };
  return state.per[task];
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const task = p.get("task");
  if (!task) return;
  state.task = task;
  if (task === "tokens") {
    Object.assign(state.tokens, {
      dump: p.get("dump") || state.tokens.dump, doc: p.get("doc") || null, start: Number(p.get("at") || 0),
      sort: p.get("sort") || "mean_kl", source: p.get("src") || "",
    });
    return;
  }
  if (p.get("c")) state.conds = p.get("c").split(",").filter(Boolean);
  Object.assign(taskState(task), {
    filter: p.get("f") || "", status: p.get("s") || "all", q: p.get("q") || "", subject: p.get("sub") || "",
    sort: p.get("sort") || "doc", compare: p.get("cmp") || "", doc: p.get("doc"),
  });
}

function writeHash() {
  const p = new URLSearchParams();
  p.set("task", state.task);
  if (state.task === "tokens") {
    const t = state.tokens;
    if (t.dump) p.set("dump", t.dump);
    if (t.doc) p.set("doc", t.doc);
    if (t.start) p.set("at", t.start);
    if (t.sort !== "mean_kl") p.set("sort", t.sort);
    if (t.source !== "") p.set("src", t.source);
  } else {
    const f = taskState();
    p.set("c", state.conds.join(","));
    if (f.filter) p.set("f", f.filter);
    if (f.status !== "all") p.set("s", f.status);
    if (f.q) p.set("q", f.q);
    if (f.subject) p.set("sub", f.subject);
    if (f.sort !== "doc") p.set("sort", f.sort);
    if (f.compare) p.set("cmp", f.compare);
    if (f.doc != null) p.set("doc", f.doc);
  }
  const hash = "#" + p.toString().replaceAll("%2C", ",").replaceAll("%3A", ":");
  if (location.hash !== hash) history.replaceState(null, "", hash);
}

function applyPrefs() {
  document.documentElement.style.setProperty("--fs", `${prefs.font}px`);
  $("#app").classList.toggle("side-hidden", !prefs.sidebar);
  $("#btn-expand").classList.toggle("hidden", prefs.sidebar);
}

// ---------------------------------------------------------------------------------------------------------------
// Boot and routing
// ---------------------------------------------------------------------------------------------------------------

const condInfo = (id) => (state.meta?.conditions || []).find((c) => c.id === id);
const shortName = (id) => condInfo(id)?.short || id;
const taskLabel = (id) => (id === "tokens" ? "Tokens" : (state.meta?.tasks || []).find((t) => t.id === id)?.label || id);

function defaultConds() {
  const all = state.meta?.conditions || [];
  const pick = ["bf16", "NVFP4"].map((n) => all.find((c) => c.name === n)?.id).filter(Boolean);
  for (const c of all) if (pick.length < 2 && !pick.includes(c.id)) pick.push(c.id);
  return pick;
}

async function loadMeta(refresh) {
  try {
    state.meta = await api("/api/conditions", { refresh: refresh ? 1 : "" });
  } catch (e) {
    state.meta = { raw_dir: "", exists: false, tasks: [], conditions: [], error: e.message };
  }
}

async function boot() {
  applyPrefs();
  readHash();
  bindEvents();
  await loadMeta(false);
  if (!state.conds.length) state.conds = defaultConds();
  const tasks = state.meta.tasks.map((t) => t.id);
  if (!state.task || (state.task !== "tokens" && !tasks.includes(state.task))) state.task = tasks[0] || "tokens";
  route();
}

function route() {
  closePopover();
  hideTip();
  document.body.dataset.view = state.task === "tokens" ? "tokens" : "trace";
  renderTaskSwitch();
  renderSideTools();
  renderTopbar();
  if (state.task === "tokens") loadTokensView();
  else loadItems();
}

function switchTask(task) {
  if (task === state.task) return;
  state.task = task;
  state.cursor = -1;
  state.items = state.item = null;
  listMessage("");
  $("#list").scrollTop = 0;
  writeHash();
  route();
}

async function refreshData() {
  state.dumps = null;
  await loadMeta(true);
  route();
}

// ---------------------------------------------------------------------------------------------------------------
// Sidebar
// ---------------------------------------------------------------------------------------------------------------

function renderTaskSwitch() {
  const tasks = [...(state.meta?.tasks || []), { id: "tokens", label: "Tokens" }];
  $("#task-switch").innerHTML = tasks
    .map((t) => `<button type="button" data-task="${esc(t.id)}" class="${t.id === state.task ? "on" : ""}">${esc(t.label)}</button>`)
    .join("");
}

function renderSideTools() {
  const el = $("#side-tools");
  if (state.task !== "tokens" || !state.dumps) { el.innerHTML = ""; return; }
  const d = state.dumps.dumps.find((x) => x.name === state.tokens.dump);
  el.innerHTML = `<button type="button" class="btn wide" data-open="dumps"><span>${esc(d ? d.label : "Choose a dump")}</span><span class="muted">▾</span></button>`;
}

const currentRows = () => (state.task === "tokens" ? state.docs?.docs : state.items?.items) || [];

function listMessage(msg) {
  const spacer = $("#list-spacer");
  spacer.style.height = "";
  spacer.innerHTML = msg ? `<div class="list-empty">${esc(msg)}</div>` : "";
  $("#side-foot").textContent = "";
}

let listFrame = 0;
function renderList() {
  listFrame = 0;
  const rows = currentRows();
  const list = $("#list"), spacer = $("#list-spacer");
  if (!rows.length) return;
  spacer.style.height = `${rows.length * ROW_H}px`;
  const top = list.scrollTop, height = list.clientHeight || 800;
  const first = Math.max(0, Math.floor(top / ROW_H) - 6);
  const last = Math.min(rows.length, Math.ceil((top + height) / ROW_H) + 6);
  const rowHTML = state.task === "tokens" ? docRowHTML : itemRowHTML;
  let html = "";
  for (let i = first; i < last; i++) html += rowHTML(rows[i], i);
  spacer.innerHTML = `<div class="rows" style="transform:translateY(${first * ROW_H}px)">${html}</div>`;
  $("#side-foot").textContent = state.task === "tokens"
    ? `${fmtInt(rows.length)} documents · j/k to move · ←/→ window`
    : `${fmtInt(rows.length)} items · j/k to move · / to search`;
}

function scrollListTo(i, center) {
  const list = $("#list");
  if (i < 0) return;
  $("#list-spacer").style.height = `${currentRows().length * ROW_H}px`;  // size first, or scrollTop is clamped
  const top = i * ROW_H, h = list.clientHeight;
  if (center) list.scrollTop = Math.max(0, top - h / 2 + ROW_H / 2);
  else if (top < list.scrollTop) list.scrollTop = top;
  else if (top + ROW_H > list.scrollTop + h) list.scrollTop = top + ROW_H - h;
  renderList();
}

function itemRowHTML(it, i) {
  const conds = state.items.conditions;
  const dots = it.c.map((v, j) => {
    if (!conds[j].available) return "";
    const cls = v == null ? "na" : v ? "ok" : "bad";
    const word = v == null ? "n/a" : v ? "correct" : "wrong";
    return `<i class="dot ${cls}" title="${esc(conds[j].short)}: ${word}"></i>`;
  }).join("");
  const sub = it.subject ? `<span class="row-sub">${esc(it.subject.replaceAll("_", " "))}</span>` : "";
  return `<div class="row${i === state.cursor ? " active" : ""}" data-i="${i}">`
    + `<div class="row-top"><span class="row-id">#${esc(it.doc_id)}</span>${sub}<span class="dots">${dots}</span></div>`
    + `<div class="row-q">${esc(it.q)}</div></div>`;
}

function docRowHTML(d, i) {
  const top = state.docs.top || 1;
  const v = docMetricOf(state.tokens.sort)(d);
  const shown = state.tokens.sort === "flip_rate" ? fmtPct(v) : state.tokens.sort === "length" ? fmtInt(v) : fmtNum(v);
  const seqs = d.n_seq > 1 ? ` · ${d.n_seq} sequences` : "";
  return `<div class="row${i === state.cursor ? " active" : ""}" data-i="${i}">`
    + `<div class="row-top"><span class="row-id">doc ${d.doc_id}</span><span class="row-sub">${esc(d.source_name)}</span><span class="row-val">${shown}</span></div>`
    + `<div class="row-bar"><i style="width:${clamp((100 * v) / top, 0, 100).toFixed(1)}%"></i></div>`
    + `<div class="row-meta">${fmtInt(d.n)} tokens${seqs}</div></div>`;
}

function select(i) {
  const rows = currentRows();
  if (!rows.length) return;
  i = clamp(i, 0, rows.length - 1);
  if (i === state.cursor) return;
  state.cursor = i;
  if (state.task === "tokens") {
    state.tokens.doc = `${rows[i].source}:${rows[i].doc_id}`;
    state.tokens.start = 0;
    loadDoc();
  } else {
    taskState().doc = rows[i].key;
    loadItem();
  }
  scrollListTo(i, false);
  writeHash();
}

// ---------------------------------------------------------------------------------------------------------------
// Top bar
// ---------------------------------------------------------------------------------------------------------------

function activeFilterCount() {
  if (state.task === "tokens") return state.tokens.source !== "" ? 1 : 0;
  const f = taskState();
  return (f.status !== "all") + Boolean(f.q) + Boolean(f.subject);
}

function renderTopbar() {
  $("#crumb").textContent = taskLabel(state.task);
  const n = activeFilterCount();
  $("#btn-filters").innerHTML = `Filters${n ? `<span class="count">${n}</span>` : ""}`;
  const chips = $("#chips");
  if (state.task === "tokens") {
    const d = state.dumps?.dumps.find((x) => x.name === state.tokens.dump);
    if (!d) { chips.innerHTML = ""; return; }
    const s = d.summary || {};
    chips.innerHTML = [
      `<span class="chip"><span class="chip-name">${esc(d.cond_name)}</span>${esc(d.set_name)}</span>`,
      s.n_tokens != null ? `<span class="chip">${fmtInt(s.n_tokens)} tokens</span>` : "",
      state.docs ? `<span class="chip">${fmtInt(state.docs.n_docs)} documents</span>` : "",
      s.mean_kl != null ? `<span class="chip">mean KL <span class="chip-name">${fmtNum(s.mean_kl)}</span></span>` : "",
      s.median_kl != null ? `<span class="chip">median KL <span class="chip-name">${fmtNum(s.median_kl)}</span></span>` : "",
      s.top1_agree != null ? `<span class="chip">top-1 agrees <span class="chip-name">${fmtPct(s.top1_agree)}</span></span>` : "",
    ].join("");
    return;
  }
  const d = state.items;
  if (!d) {
    chips.innerHTML = state.conds.map((c) => `<button type="button" class="chip" data-open="conditions"><span class="chip-name">${esc(shortName(c))}</span></button>`).join("");
    return;
  }
  const s = d.summary;
  let html = d.conditions.map((c, j) => {
    const ref = c.id === d.reference ? `<span class="chip-ref">ref</span>` : "";
    const val = c.available ? fmtPct(s.accuracy[j]) : "no data";
    return `<button type="button" class="chip${c.available ? "" : " off"}" data-open="conditions" title="${esc(c.name)}">`
      + `<span class="chip-name">${esc(c.short)}</span>${ref}<span>${val}</span></button>`;
  }).join("");
  html += `<span class="chip">${fmtInt(s.n)} items</span>`;
  if (d.compare) {
    const many = d.conditions.filter((c) => c.available).length > 2;
    const vs = many ? `${esc(shortName(d.compare))}: ` : "";
    html += `<span class="chip" title="Flips from ${esc(shortName(d.reference))} to ${esc(shortName(d.compare))}">${vs}`
      + `<span class="bad-t">${fmtInt(s.flip_to_wrong)}</span> to wrong · <span class="ok-t">${fmtInt(s.flip_to_right)}</span> to right</span>`;
  }
  chips.innerHTML = html;
}

// ---------------------------------------------------------------------------------------------------------------
// Popovers: conditions, filters, settings, dump picker
// ---------------------------------------------------------------------------------------------------------------

let pop = null;  // { name, anchor }

function togglePopover(name, anchor, keepOpen) {
  if (pop && pop.name === name) { if (!keepOpen) closePopover(); return; }
  closePopover();
  pop = { name, anchor };
  anchor.setAttribute("aria-expanded", "true");
  refreshPopover();
}

function closePopover() {
  if (!pop) return;
  pop.anchor.removeAttribute("aria-expanded");
  pop = null;
  $("#popover").classList.add("hidden");
}

function refreshPopover() {
  if (!pop) return;
  const el = $("#popover");
  const focused = document.activeElement && el.contains(document.activeElement) ? document.activeElement : null;
  const focusId = focused?.id;
  const caret = focused && "selectionStart" in focused ? [focused.selectionStart, focused.selectionEnd] : null;
  const render = { conditions: popConditions, filters: popFilters, more: popMore, dumps: popDumps }[pop.name];
  el.innerHTML = render();
  el.classList.toggle("wide", pop.name === "conditions");
  el.classList.remove("hidden");
  placePopover();
  if (focusId) {
    const n = document.getElementById(focusId);
    if (n) { n.focus(); if (caret && n.setSelectionRange) try { n.setSelectionRange(...caret); } catch { /* not a text input */ } }
  }
}

function placePopover() {
  if (!pop) return;
  const el = $("#popover"), r = pop.anchor.getBoundingClientRect(), w = el.offsetWidth;
  const inSidebar = pop.anchor.closest(".sidebar");
  const left = inSidebar ? r.left : r.right - w;
  el.style.left = `${clamp(left, 8, innerWidth - w - 8)}px`;
  el.style.top = `${r.bottom + 6}px`;
}

const seg = (attr, options, current) => `<div class="seg wrap">${options.map(([v, label]) =>
  `<button type="button" data-${attr}="${esc(v)}" class="${String(current) === String(v) ? "on" : ""}">${esc(label)}</button>`).join("")}</div>`;

function popConditions() {
  const meta = state.meta;
  if (!meta.conditions.length) {
    return `<div class="pop-empty">No lm-eval samples found in ${esc(meta.raw_dir || "the raw directory")}.</div>`;
  }
  let html = `<div class="pop-head"><span>Conditions</span><span class="muted">first selected is the reference</span></div>`;
  let group = null;
  for (const c of meta.conditions) {
    if (c.group !== group) { group = c.group; html += `<div class="pop-group">${esc(group)}</div>`; }
    const idx = state.conds.indexOf(c.id);
    const t = c.tasks[state.task];
    const val = t ? (t.acc != null ? fmtPct(t.acc) : "") : `no ${esc(taskLabel(state.task))} run`;
    const make = idx > 0 ? `<button type="button" class="mini" data-ref="${esc(c.id)}">make reference</button>` : "";
    html += `<div class="opt-row${idx >= 0 ? " on" : ""}${t ? "" : " dim"}" data-cond="${esc(c.id)}" title="${esc(c.id)}">`
      + `<span class="ord">${idx === 0 ? "ref" : idx > 0 ? idx + 1 : ""}</span><span class="opt-name">${esc(c.name)}</span>`
      + `${make}<span class="opt-val">${val}</span></div>`;
  }
  html += `<div class="pop-foot"><button type="button" class="link" data-act="cond-default">Default</button>`
    + `<button type="button" class="link" data-act="cond-clear">Clear</button></div>`;
  return html;
}

function popFilters() {
  if (state.task === "tokens") return popTokenFilters();
  const d = state.items, f = taskState();
  if (!d) return `<div class="pop-empty">Loading…</div>`;
  const avail = d.conditions.filter((c) => c.available);
  const keys = avail.length >= 2 ? ["all", "flip_to_wrong", "flip_to_right", "both_wrong", "both_correct"] : ["all", "correct", "wrong"];
  const total = Object.values(d.counts || {}).reduce((a, b) => a + b, 0);
  const status = keys.map((k) => {
    const n = k === "all" ? total : d.counts?.[k] || 0;
    return `<div class="opt-row radio${f.status === k ? " on" : ""}" data-status="${k}"><span class="opt-name">${STATUS[k]}</span><span class="opt-val">${fmtInt(n)}</span></div>`;
  }).join("");
  const rel = d.compare ? `${esc(shortName(d.reference))} vs ${esc(shortName(d.compare))}` : "";
  let html = `<div class="pop-section"><div class="pop-label"><span>Status</span><span>${rel}</span></div>${status}</div>`;
  if (avail.length >= 3) {
    const opts = avail.filter((c) => c.id !== d.reference)
      .map((c) => `<option value="${esc(c.id)}"${c.id === d.compare ? " selected" : ""}>${esc(c.name)}</option>`).join("");
    html += `<div class="pop-section"><div class="pop-label">Compare the reference with</div><select id="f-compare">${opts}</select></div>`;
  }
  const where = d.kind === "generative" ? "question, answers and generations" : "question, choices and answers";
  html += `<div class="pop-section"><div class="pop-label">Search</div>`
    + `<input id="f-search" type="search" placeholder="Search ${where}" value="${esc(f.q)}" autocomplete="off" spellcheck="false"></div>`;
  if (d.subjects.length) {
    const opts = d.subjects.map((s) => `<option value="${esc(s.name)}"${s.name === f.subject ? " selected" : ""}>${esc(s.name.replaceAll("_", " "))} (${s.n})</option>`).join("");
    html += `<div class="pop-section"><div class="pop-label">Subject</div><select id="f-subject"><option value="">All subjects</option>${opts}</select></div>`;
  }
  if (d.filters.length > 1) {
    const label = d.kind === "generative" ? "Answer extraction" : "Scoring";
    const names = d.filters.map((v) => [v, (VARIANTS[v] || v).replace(/^[a-z]/, (ch) => (d.kind === "generative" ? ch.toUpperCase() : ch))]);
    html += `<div class="pop-section"><div class="pop-label">${label}</div>${seg("variant", names, d.filter)}</div>`;
  }
  html += `<div class="pop-section"><div class="pop-label">Sort</div>${seg("sort", d.sorts.map((s) => [s, SORTS[s] || s]), d.sort)}</div>`;
  html += `<div class="pop-foot"><button type="button" class="link" data-act="reset">Reset filters</button></div>`;
  return html;
}

function popTokenFilters() {
  const t = state.tokens, d = state.docs;
  let html = `<div class="pop-section"><div class="pop-label">Sort documents by</div>${seg("docsort", DOC_SORTS, t.sort)}</div>`;
  if (d && d.sources.length > 1) {
    const all = d.sources.reduce((a, s) => a + s.n, 0);
    const rows = [{ id: "", name: "All sources", n: all }, ...d.sources].map((s) =>
      `<div class="opt-row radio${String(t.source) === String(s.id) ? " on" : ""}" data-source="${s.id}"><span class="opt-name">${esc(s.name)}</span><span class="opt-val">${fmtInt(s.n)}</span></div>`).join("");
    html += `<div class="pop-section"><div class="pop-label">Source</div>${rows}</div>`;
  }
  return html;
}

function popMore() {
  const row = (label, key, options) => `<div class="pop-section"><div class="pop-label">${label}</div>${seg(`pref-${key}`, options, prefs[key])}</div>`;
  return row("Layout", "layout", [["stacked", "Stacked"], ["columns", "Side by side"]])
    + row("Few-shot prompt", "fewshot", [[false, "Collapsed"], [true, "Expanded"]])
    + row("Text shared with the reference", "divergence", [[true, "Dim and mark"], [false, "Plain"]])
    + row("Font size", "font", [[13, "13"], [14, "14"], [15, "15"], [16, "16"]])
    + row("Token colour scale", "scale", [["linear", "Linear"], ["log", "Log"]])
    + row("Token window", "window", [[1024, "1k"], [2048, "2k"], [4096, "4k"], [8192, "8k"]])
    + `<div class="pop-section"><button type="button" class="btn sm" data-act="refresh">Refresh data</button>`
    + `<div class="pop-note">${esc(state.meta?.raw_dir || "")}</div></div>`;
}

function popDumps() {
  const dumps = state.dumps?.dumps || [];
  if (!dumps.length) return `<div class="pop-empty">No token dumps found.</div>`;
  let html = "", set = null;
  for (const d of dumps) {
    if (d.set_name !== set) { set = d.set_name; html += `<div class="pop-group">${esc(set)}</div>`; }
    const mean = d.summary?.mean_kl != null ? `mean KL ${fmtNum(d.summary.mean_kl)}` : "";
    html += `<div class="opt-row radio${d.name === state.tokens.dump ? " on" : ""}" data-dump="${esc(d.name)}"><span class="opt-name">${esc(d.cond_name)}</span><span class="opt-val">${mean}</span></div>`;
  }
  return html;
}

function setPref(key, raw) {
  const value = raw === "true" ? true : raw === "false" ? false : /^\d+$/.test(raw) ? Number(raw) : raw;
  if (prefs[key] === value) return;
  prefs[key] = value;
  savePrefs();
  applyPrefs();
  refreshPopover();
  if (state.task === "tokens") {
    if (key === "window" && state.tokens.doc) loadDoc();
    else if ((key === "scale" || key === "font") && state.doc) renderDoc();
  } else if (state.item) {
    renderItem();
  }
}

function onPopoverClick(e) {
  const el = e.target.closest("button, .opt-row");
  if (!el || !pop) return;
  const ds = el.dataset;
  const f = taskState();
  if (ds.ref) {
    e.stopPropagation();
    state.conds = [ds.ref, ...state.conds.filter((c) => c !== ds.ref)];
    return condsChanged();
  }
  if (ds.cond) {
    const i = state.conds.indexOf(ds.cond);
    if (i >= 0) state.conds.splice(i, 1); else state.conds.push(ds.cond);
    return condsChanged();
  }
  if (ds.status) { f.status = ds.status; return loadItems(); }
  if (ds.variant) { f.filter = ds.variant; return loadItems(); }
  if (ds.sort) { f.sort = ds.sort; return loadItems(); }
  if (ds.dump) {
    Object.assign(state.tokens, { dump: ds.dump, doc: null, start: 0, source: "" });
    closePopover();
    renderSideTools();
    renderTopbar();
    return loadDocs();
  }
  if (ds.docsort) { state.tokens.sort = ds.docsort; return loadDocs(); }
  if ("source" in ds) { state.tokens.source = ds.source; return loadDocs(); }
  for (const key of Object.keys(ds)) {
    if (key.startsWith("pref")) return setPref(key.slice(4).toLowerCase(), ds[key]);
  }
  switch (ds.act) {
    case "cond-default": state.conds = defaultConds(); return condsChanged();
    case "cond-clear": state.conds = []; return condsChanged();
    case "reset": Object.assign(f, { status: "all", q: "", subject: "", sort: "doc", compare: "", filter: "" }); return loadItems();
    case "refresh": closePopover(); return refreshData();
    default: return undefined;
  }
}

function condsChanged() {
  writeHash();
  refreshPopover();
  loadItems();
}

const searchChanged = debounce(() => loadItems(), 250);

// ---------------------------------------------------------------------------------------------------------------
// Items (GSM8K, MMLU and any other lm-eval task)
// ---------------------------------------------------------------------------------------------------------------

let itemsSeq = 0;
async function loadItems() {
  const f = taskState();
  const seq = ++itemsSeq;
  const cancel = later(() => { if (seq === itemsSeq) listMessage("Loading…"); }, 200);
  let data;
  try {
    data = await api("/api/items", {
      task: state.task, conditions: state.conds.join(","), filter: f.filter, q: f.q, status: f.status,
      compare: f.compare, subject: f.subject, sort: f.sort,
    });
  } catch (e) {
    if (seq === itemsSeq) { listMessage(""); contentMessage(`Could not load items: ${e.message}`); }
    return;
  } finally {
    cancel();
  }
  if (seq !== itemsSeq) return;
  state.items = data;
  let i = f.doc != null ? data.items.findIndex((it) => it.key === String(f.doc)) : -1;
  if (i < 0 && data.items.length) i = 0;
  f.doc = i >= 0 ? data.items[i].key : f.doc;
  state.cursor = i;
  renderTopbar();
  refreshPopover();
  writeHash();
  if (i < 0) {
    listMessage(itemsEmptyReason(data, true));
    contentMessage(itemsEmptyReason(data, false));
    state.item = null;
    return;
  }
  scrollListTo(i, true);
  loadItem();
}

function itemsEmptyReason(d, short) {
  if (!state.meta?.conditions.length) return short ? "No samples found." : `No lm-eval samples found in ${state.meta?.raw_dir || "the raw directory"}.`;
  if (!state.conds.length) return short ? "No conditions selected." : "Choose conditions to compare.";
  if (!d.conditions.some((c) => c.available)) return short ? "No data." : `None of the selected conditions has ${taskLabel(state.task)} samples.`;
  if (!d.n_total) return short ? "No shared items." : "The selected conditions share no items.";
  return short ? "No matches." : "No items match these filters.";
}

let itemSeq = 0, itemCtl = null;
async function loadItem() {
  const f = taskState();
  const seq = ++itemSeq;
  if (itemCtl) itemCtl.abort();
  itemCtl = new AbortController();
  const cancel = later(() => { if (seq === itemSeq) contentMessage("Loading…"); }, 250);
  try {
    const d = await api("/api/item", {
      task: state.task, doc_id: f.doc, conditions: state.conds.join(","), filter: state.items?.filter,
    }, itemCtl.signal);
    if (seq !== itemSeq) return;
    state.item = d;
    renderItem();
    $("#content").scrollTop = 0;
  } catch (e) {
    if (e.name !== "AbortError" && seq === itemSeq) contentMessage(`Could not load this item: ${e.message}`);
  } finally {
    cancel();
  }
}

function contentMessage(msg) {
  $("#content").innerHTML = `<div class="empty">${esc(msg)}</div>`;
}

function badge(correct) {
  if (correct == null) return `<span class="badge na">n/a</span>`;
  return correct ? `<span class="badge ok">correct</span>` : `<span class="badge bad">wrong</span>`;
}

function renderItem() {
  const d = state.item;
  if (!d) return;
  const conds = d.conditions.filter((c) => c.available);  // no run at all: the top-bar chip says so
  const ref = conds.find((c) => c.present) || null;
  const columns = prefs.layout === "columns";
  const body = d.kind === "choice"
    ? conds.map((c) => choiceCard(c, d, ref)).join("")
    : conds.map((c) => generationCard(c, d, ref)).join("");
  $("#content").innerHTML = `<div class="thread${columns ? " columns" : ""}" style="--cols:${Math.max(1, conds.length)}">`
    + `${questionBubble(d)}<div class="answers">${body}</div></div>`;
}

function questionBubble(d) {
  const meta = [`#${esc(d.doc_id)}`, d.subject ? esc(d.subject.replaceAll("_", " ")) : ""].filter(Boolean).join(" · ");
  let choices = "";
  if (d.kind === "choice") {
    choices = `<div class="choices">${(d.choices || []).map((c, i) => {
      const gold = i === d.gold_index;
      return `<div class="choice"><span class="letter${gold ? " gold" : ""}">${LETTERS[i] || i}</span><span class="text">${esc(c)}</span>${gold ? `<span class="gold-tag">gold</span>` : ""}</div>`;
    }).join("")}</div>`;
  }
  const open = prefs.fewshot;
  const hasPrompt = Boolean(d.prompt_prefix || d.prompt_query);
  const actions = [
    hasPrompt ? `<button type="button" class="link" data-toggle="prompt">${open ? "Hide full prompt" : "Show full prompt"}</button>` : "",
    d.reasoning ? `<button type="button" class="link" data-toggle="solution">Show reference solution</button>` : "",
  ].filter(Boolean).join("");
  const prompt = hasPrompt
    ? `<div class="reveal" data-block="prompt"${open ? "" : " hidden"}><div class="reveal-label">Full prompt as sent to the model; the few-shot examples are dimmed</div>`
      + `<pre class="prompt-text"><span class="few">${esc(d.prompt_prefix)}</span>${esc(d.prompt_query)}</pre></div>`
    : "";
  const solution = d.reasoning
    ? `<div class="reveal" data-block="solution" hidden><div class="reveal-label">Reference solution</div><div class="text">${esc(d.reasoning)}</div></div>`
    : "";
  return `<div class="msg user"><div class="bubble"><div class="bubble-meta">${meta}</div><div class="text">${esc(d.question)}</div>`
    + `${choices}${actions ? `<div class="bubble-actions">${actions}</div>` : ""}${prompt}${solution}</div></div>`;
}

function cardHead(c, ref) {
  const tag = ref && c.id === ref.id ? `<span class="tag">reference</span>` : "";
  const right = c.present ? badge(c.correct) : `<span class="badge na">${c.available ? "missing" : "no data"}</span>`;
  return `<div class="a-head"><span class="a-name" title="${esc(c.name)}">${esc(c.name)}</span>${tag}${right}</div>`;
}

function otherVariants(c, d) {
  return Object.entries(c.variants || {})
    .filter(([v]) => v !== d.filter)
    .map(([v, x]) => {
      const verdict = x.correct == null ? "" : x.correct ? ` <span class="ok-t">correct</span>` : ` <span class="bad-t">wrong</span>`;
      return `<span>${esc(VARIANTS[v] || v)} <span class="val">${esc(x.answer || "none")}</span>${verdict}</span>`;
    })
    .join("");
}

function commonPrefix(a, b) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a.charCodeAt(i) === b.charCodeAt(i)) i++;
  return i;
}

function generationHTML(text, refText) {
  if (!prefs.divergence || refText == null || text === refText) return esc(text);
  let k = commonPrefix(text, refText);
  while (k > 0 && !/\s/.test(text[k - 1])) k--;  // back up to a word boundary
  if (k === 0) return esc(text);
  return `<span class="shared">${esc(text.slice(0, k))}</span><span class="div-mark" title="Diverges from the reference here"></span>${esc(text.slice(k))}`;
}

function generationCard(c, d, ref) {
  if (!c.present) return `<div class="msg assistant missing">${cardHead(c, ref)}<div class="gen muted">No sample for this item.</div></div>`;
  const tidy = (s) => (s || "").replace(/^\s+/, "").replace(/\s+$/, "");
  const text = tidy(c.generation);
  const refText = ref && c.id !== ref.id ? tidy(ref.generation) : null;
  const same = refText != null && text === refText ? `<span>identical to ${esc(ref.short)}</span>` : "";
  return `<div class="msg assistant">${cardHead(c, ref)}<div class="gen">${text ? generationHTML(text, refText) : `<span class="muted">(empty generation)</span>`}</div>`
    + `<div class="a-foot"><span>answer <span class="val">${esc(c.answer || "none")}</span></span><span>gold <span class="val">${esc(d.gold)}</span></span>`
    + `${otherVariants(c, d)}${same}</div></div>`;
}

function choiceCard(c, d, ref) {
  if (!c.present) return `<div class="msg assistant missing">${cardHead(c, ref)}<div class="muted">No sample for this item.</div></div>`;
  const ll = c.ll || [];
  const finite = ll.filter((v) => v != null);
  const top = finite.length ? Math.max(...finite) : 0;
  const ex = ll.map((v) => (v == null ? 0 : Math.exp(v - top)));
  const z = ex.reduce((a, b) => a + b, 0) || 1;
  const chosen = c.answer ? LETTERS.indexOf(c.answer) : -1;
  const rows = (d.choices || []).map((txt, i) => {
    const p = (ex[i] || 0) / z;
    const isChosen = i === chosen, isGold = i === d.gold_index;
    const cls = isChosen ? (c.correct ? " chosen ok" : " chosen bad") : "";
    const tip = `log-likelihood of the answer letter ${LETTERS[i] || i}: ${ll[i] == null ? "n/a" : ll[i].toFixed(4)}; p = ${p.toFixed(3)} after normalising over the ${d.choices.length} options`;
    return `<div class="opt${cls}" title="${esc(tip)}">`
      + `<span class="letter${isGold ? " gold" : ""}">${LETTERS[i] || i}</span><span class="opt-t">${esc(txt)}</span>`
      + `<span class="opt-bar"><i style="width:${(100 * p).toFixed(1)}%"></i></span>`
      + `<span class="opt-ll">${ll[i] == null ? "n/a" : ll[i].toFixed(3)}</span></div>`;
  }).join("");
  const pGold = d.gold_index != null && ex[d.gold_index] != null ? ex[d.gold_index] / z : null;
  return `<div class="msg assistant">${cardHead(c, ref)}<div class="opts">${rows}</div>`
    + `<div class="a-foot"><span>chosen <span class="val">${esc(c.answer || "none")}</span></span><span>gold <span class="val">${esc(d.gold)}</span></span>`
    + `<span title="probability of the gold letter, normalised over the options">p(gold) <span class="val">${pGold == null ? "n/a" : pGold.toFixed(2)}</span></span>`
    + `${otherVariants(c, d)}</div></div>`;
}

// ---------------------------------------------------------------------------------------------------------------
// Token view
// ---------------------------------------------------------------------------------------------------------------

function heatAlpha(kl) {
  if (!(kl > 0)) return 0;
  const x = prefs.scale === "log" ? Math.log(kl / KL_FLOOR) / Math.log(KL_CAP / KL_FLOOR) : kl / KL_CAP;
  return clamp(x, 0, 1) * MAX_ALPHA;
}

async function loadTokensView() {
  if (!state.dumps) {
    listMessage("Loading…");
    try {
      state.dumps = await api("/api/dumps");
    } catch (e) {
      listMessage("");
      contentMessage(`Could not list token dumps: ${e.message}`);
      return;
    }
  }
  if (state.task !== "tokens") return;
  const dumps = state.dumps.dumps;
  if (!dumps.length) {
    listMessage("No dumps.");
    contentMessage(`No token dumps found in ${state.dumps.dir}.`);
    return;
  }
  if (!dumps.some((d) => d.name === state.tokens.dump)) {
    state.tokens.dump = (dumps.find((d) => d.name === "q1_kl_2m") || dumps[0]).name;
  }
  renderSideTools();
  renderTopbar();
  loadDocs();
}

let docsSeq = 0;
async function loadDocs() {
  const t = state.tokens;
  const seq = ++docsSeq;
  const cancel = later(() => { if (seq === docsSeq) { listMessage("Loading dump…"); contentMessage("Loading dump…"); } }, 200);
  let data;
  try {
    data = await api("/api/dump/docs", { name: t.dump, sort: t.sort, limit: 100000, source: t.source });
  } catch (e) {
    if (seq === docsSeq) { listMessage(""); contentMessage(`Could not load the dump: ${e.message}`); }
    return;
  } finally {
    cancel();
  }
  if (seq !== docsSeq || state.task !== "tokens") return;
  data.top = Math.max(1e-9, ...data.docs.map(docMetricOf(t.sort)));
  state.docs = data;
  let i = t.doc ? data.docs.findIndex((d) => `${d.source}:${d.doc_id}` === t.doc) : -1;
  if (i < 0 && data.docs.length) { i = 0; t.start = 0; }
  t.doc = i >= 0 ? `${data.docs[i].source}:${data.docs[i].doc_id}` : null;
  state.cursor = i;
  renderTopbar();
  refreshPopover();
  writeHash();
  if (i < 0) { listMessage("No documents."); contentMessage("No documents in this dump."); return; }
  scrollListTo(i, true);
  loadDoc();
}

const docMetricOf = (sort) => (d) => ({ mean_kl: d.mean_kl, max_kl: d.max_kl, flip_rate: d.flip_rate, length: d.n }[sort] ?? d.mean_kl);

let docSeq = 0, docCtl = null;
async function loadDoc(flash) {
  const t = state.tokens;
  if (!t.doc) return;
  const [source, docId] = t.doc.split(":").map(Number);
  const seq = ++docSeq;
  if (docCtl) docCtl.abort();
  docCtl = new AbortController();
  const cancel = later(() => { if (seq === docSeq) contentMessage("Loading…"); }, 250);
  try {
    const d = await api("/api/dump/doc", { name: t.dump, doc_id: docId, source, start: t.start, limit: prefs.window }, docCtl.signal);
    if (seq !== docSeq) return;
    state.doc = d;
    t.start = d.start;
    writeHash();
    renderDoc(flash);
  } catch (e) {
    if (e.name !== "AbortError" && seq === docSeq) contentMessage(`Could not load this document: ${e.message}`);
  } finally {
    cancel();
  }
}

function tokenHTML(t, i) {
  let cls = "t";
  if (!t.agree[i]) cls += " x";
  let text = t.text[i], after = "";
  if (t.special[i]) {
    cls += " sp";
    if (/eos|end_of_turn/.test(text)) after = "\n";
  } else if (text === "") {
    cls += " e";
  } else if (/^[^\S\n]*\n\s*$/.test(text)) {  // a line break token: show a faint marker, keep the break
    cls += " nl";
    after = text.replace(/^[^\S\n]*/, "");
    text = "↵";
  }
  const a = heatAlpha(t.kl[i]);
  const style = a > 0.01 ? ` style="background:rgba(var(--heat),${a.toFixed(3)})"` : "";
  return `<span class="${cls}" data-i="${i}"${style}>${esc(text)}</span>${after}`;
}

function legendHTML() {
  const stops = prefs.scale === "log" ? [0.01, 0.03, 0.1, 0.3, 1] : [0.2, 0.4, 0.6, 0.8, 1];
  const sw = stops.map((v) => `<span class="sw" style="background:rgba(var(--heat),${heatAlpha(v).toFixed(3)})"></span><span>${v === KL_CAP ? `≥${v}` : v}</span>`).join("");
  return `<div class="legend muted"><span>KL(bf16 ‖ quantized) in nats, ${prefs.scale} scale, capped at ${KL_CAP}</span>`
    + `<span class="sws">${sw}</span><span><span class="x-demo">underline</span>: top-1 prediction differs</span></div>`;
}

function renderDoc(flash) {
  const d = state.doc;
  if (!d) return;
  const t = d.tokens, n = t.kl.length;
  let body = "";
  let prev = n ? t.seq[0] : 0;
  if (d.n_seq > 1 && n) {
    const cont = d.seq_starts.includes(d.start) ? "" : ", continued";
    body += `<span class="seq-break first">sequence ${t.seq[0] + 1} of ${d.n_seq}${cont}</span>`;
  }
  for (let i = 0; i < n; i++) {
    if (t.seq[i] !== prev) {
      body += `<span class="seq-break">sequence ${t.seq[i] + 1} of ${d.n_seq} · the model's context restarts here</span>`;
      prev = t.seq[i];
    }
    body += tokenHTML(t, i);
  }
  const seqs = d.n_seq > 1 ? ` in ${d.n_seq} sequences` : "";
  const tok = d.tokenizer || {};
  const note = tok.ok ? "" : `<div class="note">Tokenizer unavailable${tok.error ? ` (${esc(tok.error)})` : ""}: showing token ids instead of text.</div>`;
  const maxRange = Math.max(0, d.n - prefs.window);
  $("#content").innerHTML = `<div class="tokview">`
    + `<div class="tok-title">doc ${d.doc_id} <span class="muted">· ${esc(d.source_name)}</span></div>`
    + `<div class="tok-stats muted">${fmtInt(d.n)} tokens${seqs} · mean KL ${fmtNum(d.stats.mean_kl)} · top-1 differs on ${fmtPct(d.stats.flip_rate)}`
    + ` · max KL ${fmtNum(d.stats.max_kl)} <button type="button" class="link" data-act="jump-max">jump to it</button></div>${note}`
    + `<div class="tok-nav"><canvas id="strip" class="strip"></canvas><div class="win-row">`
    + `<button type="button" class="btn sm" data-act="win-prev"${d.start > 0 ? "" : " disabled"}>Previous</button>`
    + `<input id="win-range" type="range" min="0" max="${maxRange}" step="1" value="${Math.min(d.start, maxRange)}" aria-label="Window position">`
    + `<button type="button" class="btn sm" data-act="win-next"${d.end < d.n ? "" : " disabled"}>Next</button>`
    + `<span id="win-label" class="win-label muted">${windowLabel(d.start, d.end, d.n)}</span></div>${legendHTML()}</div>`
    + `<div id="tokens" class="tokens">${body}</div></div>`;
  drawStrip();
  if (flash != null) {
    const el = $(`#tokens .t[data-i="${flash - d.start}"]`);
    if (el) {
      el.classList.add("flash");
      el.scrollIntoView({ block: "center" });
      setTimeout(() => el.classList.remove("flash"), 2000);
    }
  } else {
    $("#content").scrollTop = 0;
  }
}

const windowLabel = (a, b, n) => `tokens ${fmtInt(a)}–${fmtInt(Math.max(a, b - 1))} of ${fmtInt(n)}`;

function drawStrip() {
  const canvas = $("#strip"), d = state.doc;
  if (!canvas || !d) return;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const heat = css.getPropertyValue("--heat").trim(), accent = css.getPropertyValue("--accent").trim();
  ctx.fillStyle = "#1a1a1a";
  ctx.fillRect(0, h - 1, w, 1);
  const { mean, edges } = d.profile;
  ctx.fillStyle = `rgba(${heat}, 0.85)`;
  for (let i = 0; i < mean.length; i++) {
    const x0 = (edges[i] / d.n) * w, x1 = ((i + 1 < edges.length ? edges[i + 1] : d.n) / d.n) * w;
    const bh = Math.max(1, (heatAlpha(mean[i]) / MAX_ALPHA) * (h - 4));
    ctx.fillRect(x0, h - bh, Math.max(1, x1 - x0 - 0.5), bh);
  }
  ctx.fillStyle = "#4a4a4a";
  for (const s of d.seq_starts) if (s > 0) ctx.fillRect(Math.round((s / d.n) * w), 0, 1, h);
  const x0 = (d.start / d.n) * w, x1 = (d.end / d.n) * w;
  ctx.strokeStyle = accent;
  ctx.lineWidth = 1;
  ctx.strokeRect(Math.round(x0) + 0.5, 0.5, Math.max(3, x1 - x0) - 1, h - 1);
}

function jumpTo(start, flash) {
  const d = state.doc;
  if (!d) return;
  state.tokens.start = clamp(Math.round(start), 0, Math.max(0, d.n - 1));
  loadDoc(flash);
}

function stepWindow(dir) {
  const d = state.doc;
  if (!d) return;
  const next = d.start + dir * prefs.window;
  if (next < 0 && d.start === 0) return;
  if (next >= d.n) return;
  jumpTo(Math.max(0, next));
}

function stripIndex(e) {
  const r = $("#strip").getBoundingClientRect();
  return Math.floor(clamp((e.clientX - r.left) / r.width, 0, 0.999999) * state.doc.n);
}

// ---------------------------------------------------------------------------------------------------------------
// Tooltip
// ---------------------------------------------------------------------------------------------------------------

function showTip(html, rect) {
  const tip = $("#tooltip");
  tip.innerHTML = html;
  tip.classList.remove("hidden");
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let top = rect.bottom + 6;
  if (top + h > innerHeight - 8) top = rect.top - h - 6;
  tip.style.left = `${clamp(rect.left, 8, innerWidth - w - 8)}px`;
  tip.style.top = `${Math.max(8, top)}px`;
}

function hideTip() { $("#tooltip").classList.add("hidden"); }

function tokenTip(i) {
  const d = state.doc, t = d.tokens;
  const seq = d.n_seq > 1 ? ` · sequence ${t.seq[i] + 1}` : "";
  const agree = t.agree[i] ? "top-1 prediction agrees" : `<span class="bad-t">top-1 prediction differs</span>`;
  return `<div>KL <b>${fmtNum(t.kl[i])}</b> nats</div><div>bf16 entropy <b>${fmtNum(t.ent[i])}</b> nats</div>`
    + `<div>${esc(d.classes[t.cls[i]] || "")} · position ${fmtInt(t.pos[i])}${seq}</div><div>${agree}</div>`
    + `<div class="muted">id ${t.id[i]} · ${esc(JSON.stringify(t.text[i]))}</div>`;
}

// ---------------------------------------------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------------------------------------------

function bindEvents() {
  $("#task-switch").addEventListener("click", (e) => {
    const b = e.target.closest("[data-task]");
    if (b) switchTask(b.dataset.task);
  });
  const sidebar = (show) => { prefs.sidebar = show; savePrefs(); applyPrefs(); drawStrip(); renderList(); };
  $("#btn-collapse").addEventListener("click", () => sidebar(false));
  $("#btn-expand").addEventListener("click", () => sidebar(true));
  $("#btn-conditions").addEventListener("click", (e) => togglePopover("conditions", e.currentTarget));
  $("#btn-filters").addEventListener("click", (e) => togglePopover("filters", e.currentTarget));
  $("#btn-more").addEventListener("click", (e) => togglePopover("more", e.currentTarget));
  $("#chips").addEventListener("click", (e) => {
    if (e.target.closest("[data-open='conditions']")) togglePopover("conditions", $("#btn-conditions"));
  });
  $("#side-tools").addEventListener("click", (e) => {
    const b = e.target.closest("[data-open='dumps']");
    if (b) togglePopover("dumps", b);
  });

  $("#list").addEventListener("scroll", () => { if (!listFrame) listFrame = requestAnimationFrame(renderList); });
  $("#list-spacer").addEventListener("click", (e) => {
    const r = e.target.closest(".row");
    if (r) select(Number(r.dataset.i));
  });

  const popEl = $("#popover");
  popEl.addEventListener("click", onPopoverClick);
  popEl.addEventListener("input", (e) => {
    if (e.target.id === "f-search") { taskState().q = e.target.value; searchChanged(); }
  });
  popEl.addEventListener("change", (e) => {
    const f = taskState();
    if (e.target.id === "f-subject") { f.subject = e.target.value; loadItems(); }
    if (e.target.id === "f-compare") { f.compare = e.target.value; loadItems(); }
  });

  const content = $("#content");
  content.addEventListener("click", (e) => {
    const toggle = e.target.closest("[data-toggle]");
    if (toggle) {
      const block = toggle.closest(".bubble").querySelector(`[data-block="${toggle.dataset.toggle}"]`);
      if (block) {
        block.hidden = !block.hidden;
        const what = toggle.dataset.toggle === "prompt" ? "full prompt" : "reference solution";
        toggle.textContent = `${block.hidden ? "Show" : "Hide"} ${what}`;
      }
      return;
    }
    const act = e.target.closest("[data-act]")?.dataset.act;
    if (act === "win-prev") stepWindow(-1);
    if (act === "win-next") stepWindow(1);
    if (act === "jump-max" && state.doc) {
      const at = state.doc.stats.argmax;
      jumpTo(at - Math.floor(prefs.window / 2), at);
    }
    if (e.target.id === "strip" && state.doc) jumpTo(stripIndex(e) - Math.floor(prefs.window / 2));
  });
  content.addEventListener("input", (e) => {
    if (e.target.id === "win-range" && state.doc) {
      const a = Number(e.target.value);
      $("#win-label").textContent = windowLabel(a, Math.min(a + prefs.window, state.doc.n), state.doc.n);
    }
  });
  content.addEventListener("change", (e) => {
    if (e.target.id === "win-range") jumpTo(Number(e.target.value));
  });
  content.addEventListener("mouseover", (e) => {
    const s = e.target.closest?.(".t");
    if (s && state.doc && state.task === "tokens") showTip(tokenTip(Number(s.dataset.i)), s.getBoundingClientRect());
  });
  content.addEventListener("mousemove", (e) => {
    if (e.target.id !== "strip" || !state.doc) return;
    const d = state.doc, at = stripIndex(e);
    const p = d.profile, b = Math.max(0, p.edges.findLastIndex((x) => x <= at));
    const end = b + 1 < p.edges.length ? p.edges[b + 1] : d.n;
    showTip(`<div>tokens ${fmtInt(p.edges[b])}–${fmtInt(end - 1)}</div><div>mean KL <b>${fmtNum(p.mean[b])}</b> · max <b>${fmtNum(p.max[b])}</b></div><div class="muted">click to view this region</div>`,
      { left: e.clientX - 20, right: e.clientX, top: e.clientY - 8, bottom: e.clientY + 8 });
  });
  content.addEventListener("mouseout", (e) => {
    const from = e.target.closest?.(".t, #strip");
    const to = e.relatedTarget?.closest?.(".t, #strip");
    if (from && from !== to) hideTip();
  });
  content.addEventListener("scroll", hideTip);

  document.addEventListener("mousedown", (e) => {
    if (!pop) return;
    if ($("#popover").contains(e.target) || pop.anchor.contains(e.target) || e.target.closest("[data-open]")) return;
    closePopover();
  });
  document.addEventListener("keydown", onKey);
  window.addEventListener("resize", () => { placePopover(); drawStrip(); renderList(); });
  window.addEventListener("hashchange", () => { readHash(); route(); });
}

function onKey(e) {
  const tag = (e.target.tagName || "").toLowerCase();
  const typing = tag === "input" || tag === "textarea" || tag === "select" || e.target.isContentEditable;
  if (e.key === "Escape") {
    hideTip();
    if (pop) { closePopover(); e.preventDefault(); } else if (typing) e.target.blur();
    return;
  }
  if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "j" || e.key === "ArrowDown") { e.preventDefault(); select(state.cursor + 1); }
  else if (e.key === "k" || e.key === "ArrowUp") { e.preventDefault(); select(state.cursor - 1); }
  else if (e.key === "/") {
    e.preventDefault();
    togglePopover("filters", $("#btn-filters"), true);
    const s = $("#f-search");
    if (s) { s.focus(); s.select(); }
  } else if (state.task === "tokens" && e.key === "ArrowRight") { e.preventDefault(); stepWindow(1); }
  else if (state.task === "tokens" && e.key === "ArrowLeft") { e.preventDefault(); stepWindow(-1); }
}

boot();
