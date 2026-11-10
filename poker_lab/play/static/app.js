// Poker-Lab play page. Vanilla JS, no external requests: everything goes to this server's /api.
"use strict";

const $ = (id) => document.getElementById(id);
const fill = (el, ...kids) => el.replaceChildren(...kids.flat().filter((k) => k !== null && k !== undefined));
let options = null;   // /api/options
let state = null;     // last /api/state
let shownToken = null; // turn token whose controls are on screen
let busy = false;
let pollTimer = null;
let showSetup = false; // after a session ends: report first, setup on request

function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k === "style") for (const d of v.split(";")) { const [p, val] = d.split(":"); if (val !== undefined) el.style.setProperty(p.trim(), val.trim()); }
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined) el.append(kid);
  return el;
}

const SUITS = { s: ["♠", "b", "spades"], h: ["♥", "r", "hearts"], d: ["♦", "r", "diamonds"], c: ["♣", "b", "clubs"] };
const RANKS = { T: "10", J: "J", Q: "Q", K: "K", A: "A" };
function card(c, small) {
  if (!c) return h("span", { class: "card back" + (small ? " small" : ""), "aria-label": "hidden card" });
  const [sym, color, name] = SUITS[c[1]];
  const rank = RANKS[c[0]] || c[0];
  return h("span", { class: `card ${color}` + (small ? " small" : ""), "aria-label": `${rank} of ${name}` }, rank, h("span", { "aria-hidden": "true", text: sym }));
}
const bbOf = (chips, big) => Math.round((chips / big) * 100) / 100;
const fmtBB = (x) => `${x > 0 ? "+" : ""}${Number(x).toFixed(2)} bb`;
const pct = (x) => `${(100 * x).toFixed(0)}%`;
const signed = (x, d = 3) => `${x > 0 ? "+" : ""}${Number(x).toFixed(d)}`;
const cls = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : "");

async function api(path, body) {
  const init = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, init);
  const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

function setStatus(text, error) {
  const el = $("status");
  el.textContent = text || "";
  el.classList.toggle("error", !!error);
}

// ---------------------------------------------------------------- setup
function mode() { return document.querySelector("input[name=mode]:checked").value; }
function syncSetup() {
  for (const el of document.querySelectorAll(".opts")) el.hidden = el.dataset.for !== mode();
  const adv = $("opponents").value === "advisor";
  $("engine-row").hidden = !adv;
  $("engine-desc").hidden = !adv;
  const e = options && options.engines.find((x) => x.name === $("engine").value);
  if (e) $("engine-desc").textContent = `${e.kind === "heuristic" ? "Heuristic: " : ""}${e.description}`;
}
async function start(ev) {
  if (ev) ev.preventDefault();
  const s = { mode: mode() };
  if (s.mode === "pushfold") { s.stack_bb = parseInt($("stack").value, 10); s.fresh = $("fresh").checked; }
  else { s.players = parseInt($("players").value, 10); s.opponents = $("opponents").value; if (s.opponents === "advisor") s.engine = $("engine").value; }
  if ($("seed").value !== "") s.seed = parseInt($("seed").value, 10);
  try { shownToken = null; showSetup = false; render(await api("/api/new", s)); } catch (e) { setStatus(e.message, true); }
}

// ---------------------------------------------------------------- actions
async function send(action, amount) {
  const t = state && state.turn;
  if (!t || busy) return;
  busy = true;
  const body = { token: t.token, action };
  if (amount !== undefined) body.amount = amount;
  try { render(await api("/api/act", body)); } catch (e) { setStatus(e.message, true); }
  finally { busy = false; }
}
function sendRaise() {
  const t = state && state.turn; if (!t) return;
  const big = t.big_blind, v = parseFloat($("raise-amount").value);
  const chips = Math.round(v * big);
  if (!isFinite(v) || chips < t.legal.min_raise_to || chips > t.legal.max_raise_to) {
    setStatus(`Raise to between ${bbOf(t.legal.min_raise_to, big)} and ${bbOf(t.legal.max_raise_to, big)} bb.`, true);
    $("raise-amount").focus();
    return;
  }
  send("raise", chips);
}
async function nextHand() {
  if (!state || !state.session || state.session.status !== "hand_over") return;
  try { render(await api("/api/next", {})); } catch (e) { setStatus(e.message, true); }
}
let endArmed = 0;
async function endSession() {
  if (!state || !state.session || state.session.status === "ended") return;
  if (Date.now() - endArmed > 4000) {  // two presses: no blocking dialog, still no accidents
    endArmed = Date.now();
    $("end-btn").firstChild.textContent = "Press again to end ";
    setTimeout(() => { $("end-btn").firstChild.textContent = "End session "; }, 4000);
    return;
  }
  endArmed = 0;
  $("end-btn").firstChild.textContent = "End session ";
  try { render(await api("/api/end", {})); } catch (e) { setStatus(e.message, true); }
}
async function toggleRead(on) {
  if (!state || !state.session) return;
  try { render(await api("/api/read", { on })); } catch (e) { setStatus(e.message, true); }
}

// ---------------------------------------------------------------- render
function render(s) {
  state = s;
  const sess = s.session;
  const ended = !sess || sess.status === "ended";
  $("setup").hidden = !(!sess || (ended && (showSetup || !s.report)));
  $("game").hidden = !sess || ended;
  $("results").hidden = !sess || !s.results || s.results.hands === 0;
  $("history").hidden = !sess || !s.log || s.log.length === 0;
  $("report").hidden = !s.report || showSetup;
  $("read-toggle").disabled = !sess || ended;
  $("read-toggle").checked = !!(sess && sess.read_on);
  $("end-btn").disabled = !sess || ended;
  $("read").hidden = !s.read || ended;
  if (!sess) { setStatus(""); schedule(); return; }

  const msg = { preparing: sess.message || "Preparing", playing: "", ready: "", hand_over: "", error: `Error: ${sess.error}`, ended: "Session ended." }[sess.status];
  setStatus(s.turn ? "Your move." : msg, sess.status === "error");
  renderTable(s);
  renderResults(s.results, sess);
  renderHistory(s.log);
  if (s.read && !ended) renderRead($("read-body"), s.read, sess.mode);
  if (s.report) renderReport(s.report);
  schedule();
}

function renderTable(s) {
  const t = s.turn, last = s.last_hand, sess = s.session;
  const over = sess.status === "hand_over" && last;
  const src = t || (over ? last : null);
  $("board").replaceChildren(...(src ? src.board.map((c) => card(c)) : []));
  $("pot").textContent = t ? `${bbOf(t.pot, t.big_blind)} bb` : "";
  $("your-cards").replaceChildren(...((src && src.your_cards) || [null, null]).map((c) => card(c)));
  // seats
  const seats = $("seats");
  if (t) {
    seats.replaceChildren(...t.seats.map((x) => h("li", { class: [x.actor && "actor", x.folded && "folded", x.you && "you"].filter(Boolean).join(" ") },
      h("div", {}, h("span", { class: "name", text: x.player }), " ", x.position, x.button ? h("span", { class: "btn", title: "button", text: "D" }) : null),
      h("div", { text: `${bbOf(x.stack, t.big_blind)} bb` + (x.bet ? ` · bet ${bbOf(x.bet, t.big_blind)}` : "") + (x.folded ? " · folded" : "") }))));
  } else if (over) {
    seats.replaceChildren(...last.seats.map((x) => h("li", { class: x.player === "You" ? "you" : "" },
      h("div", {}, h("span", { class: "name", text: x.player }), " ", x.position),
      h("div", { class: "cards" }, ...(x.cards ? x.cards.map((c) => card(c, true)) : [card(null, true), card(null, true)])),
      h("div", { class: cls(x.net), text: fmtBB(bbOf(x.net, last.big_blind)) }))));
  } else seats.replaceChildren();
  // log lines
  const lines = t ? t.actions : over ? last.actions : [];
  $("action-log").replaceChildren(...lines.map((l) => h("li", { text: l })));
  // controls
  if (t) {
    if (shownToken !== t.token) { shownToken = t.token; renderControls(t, sess.mode); }
  } else { shownToken = null; $("actions").replaceChildren(); $("raise-box").hidden = true; }
  $("hand-over").hidden = !over || !!t;
  if (over) {
    const r = last;
    fill($("hand-result"), 
      h("span", { class: cls(r.net), text: `Hand ${r.number}: ${fmtBB(bbOf(r.net, r.big_blind))}` }),
      r.adjusted ? h("span", { class: "note", text: ` · all-in EV ${fmtBB(bbOf(r.ev, r.big_blind))} (equity ${pct(r.equity)}, ${r.method === "exact" ? "exact" : "Monte Carlo"})` }) : null);
    const nb = $("next-btn");
    if (document.activeElement === document.body || document.activeElement === null) nb.focus({ preventScroll: true });
  }
}

const NUMBER_KEY = { allin: "1", fold: "2" };  // and 3 for the next hand

function renderControls(t, gameMode) {
  const big = t.big_blind, L = t.legal;
  const btns = t.buttons.map((b) => {
    let label, key, act = () => send(b.action);
    if (b.action === "fold") { label = "Fold"; key = "F"; }
    else if (b.action === "check") { label = "Check"; key = "C"; }
    else if (b.action === "call") { label = `Call ${bbOf(b.amount, big)} bb`; key = "C"; }
    else if (b.action === "allin") { label = b.label === "Push" ? `Push ${bbOf(L.max_raise_to, big)} bb` : `All-in ${bbOf(L.max_raise_to, big)} bb`; key = gameMode === "pushfold" ? "P" : "A"; }
    else { label = b.label || "Raise"; key = "R"; act = () => { $("raise-box").hidden = false; $("raise-amount").focus(); $("raise-amount").select(); }; }
    // Number keys first (1 push or all-in, 2 fold); the letter keys still work.
    const shown = NUMBER_KEY[b.action] || key;
    return h("button", { class: b.action === "allin" || b.action === "call" ? "primary" : "", "data-key": key, "data-action": b.action, onclick: act }, label, h("kbd", { text: shown }));
  });
  $("actions").replaceChildren(...btns);
  const box = $("raise-box");
  box.hidden = true;
  if (L.can_raise && gameMode !== "pushfold") {
    const inp = $("raise-amount");
    inp.min = bbOf(L.min_raise_to, big); inp.max = bbOf(L.max_raise_to, big);
    inp.step = big === 2 ? 0.5 : 0.01;
    inp.value = bbOf(L.min_raise_to, big);
    $("raise-range").textContent = `${inp.min}–${inp.max} bb`;
    const pot = t.pot, call = L.call_amount;
    const sized = (f) => Math.min(L.max_raise_to, Math.max(L.min_raise_to, Math.round(L.street_bet + call + f * (pot + call))));
    $("presets").replaceChildren(...[["Min", L.min_raise_to], ["½ pot", sized(0.5)], ["¾ pot", sized(0.75)], ["Pot", sized(1)]].map(([n, v]) =>
      h("button", { type: "button", onclick: () => { inp.value = bbOf(v, big); inp.focus(); } }, n)));
  }
  const first = $("actions").querySelector("button");
  if (first && !$("actions").contains(document.activeElement)) first.focus({ preventScroll: true });
}

function renderResults(r, sess) {
  if (!r || !r.hands) return;
  const items = [["Hands", r.hands], ["Won", fmtBB(r.bb)], ["All-in EV", fmtBB(r.allin_ev_bb)],
    ["Per hand", `${signed(r.bb_per_hand)} bb`], ["All-in EV per hand", `${signed(r.allin_ev_bb_per_hand)} bb`], ["All-in hands adjusted", r.adjusted_hands]];
  $("results-dl").replaceChildren(...items.map(([k, v]) => h("div", {}, h("dt", { text: k }), h("dd", { text: String(v) }))));
  $("results-dl").title = `Seed ${sess.seed}`;
}

function renderHistory(log) {
  $("history-body").replaceChildren(...(log || []).slice().reverse().map((x) => h("tr", {},
    h("td", { text: x.number }), h("td", {}, h("span", { class: "cards" }, ...(x.your_cards || []).map((c) => card(c, true)))),
    h("td", { class: cls(x.net_bb), text: fmtBB(x.net_bb) }), h("td", { class: cls(x.ev_bb), text: x.adjusted ? fmtBB(x.ev_bb) : "—" }))));
}

// ---------------------------------------------------------------- the read
function grid(title, probs, eq, cap) {
  const idx = Object.fromEntries(options.classes.map((c, i) => [c, i]));
  const R = "AKQJT98765432";
  const cells = [];
  for (let i = 0; i < 13; i++) for (let j = 0; j < 13; j++) {
    const name = i === j ? R[i] + R[j] : i < j ? R[i] + R[j] + "s" : R[j] + R[i] + "o";
    const k = idx[name], p = probs[k];
    cells.push(h("span", { style: `--a:${(0.08 + 0.85 * p).toFixed(2)}`, class: eq[k] > 0.5 ? "eq" : "", title: `${name}: ${pct(p)}${eq[k] > 0.5 ? " (equilibrium plays it)" : ""}`, text: name }));
  }
  return h("div", {}, h("div", { class: "label", text: title }), h("div", { class: "grid13", role: "img", "aria-label": `${title}, 13 by 13 hand grid` }, cells), h("p", { class: "note", text: cap }));
}
function iv(x, d = 3) { return `${signed(x.mean, d)} [${signed(x.lower, d)}, ${signed(x.upper, d)}]`; }
function dl(items) { return h("dl", { class: "stats" }, items.map(([k, v]) => h("div", {}, h("dt", { text: k }), h("dd", { text: String(v) })))); }

function renderRead(el, r, gameMode) {
  if (gameMode === "pushfold") return renderPushFoldRead(el, r);
  return renderHoldemRead(el, r);
}
function renderPushFoldRead(el, r) {
  const a = r.advised;
  const fit = r.fit ? h("table", {}, h("thead", {}, h("tr", {}, ["Check", "p-value", "Threshold", "Data", ""].map((x) => h("th", { text: x })))),
    h("tbody", {}, r.fit.map((c) => h("tr", {}, h("td", { text: c.name.replace("_", " ") }), h("td", { text: c.p_value.toFixed(3) }),
      h("td", { text: c.threshold === null ? "—" : c.threshold.toFixed(3) }), h("td", { text: c.n }),
      h("td", {}, h("span", { class: `badge ${c.failed ? "fail" : "ok"}`, text: c.failed ? "failed" : "passed" }))))))
    : h("p", { class: "note", text: r.checks_calibrated ? "No decisions of yours yet, so no fit checks." : "Fit checks not calibrated." });
  fill(el, 
    h("p", {}, `From ${r.hands} hand${r.hands === 1 ? "" : "s"} at ${r.stack} bb` + (r.earlier_hands ? ` (${r.earlier_hands} from earlier sessions)` : "") +
      `: you pushed ${r.pushes} of ${r.sb_decisions} small blinds and called ${r.calls} of ${r.bb_decisions} pushes.`),
    dl([["Your push range (est.)", `${pct(r.push_share.mean)} [${pct(r.push_share.lower)}, ${pct(r.push_share.upper)}]`],
      ["Equilibrium pushes", pct(r.eq_push_share)],
      ["Your call range (est.)", `${pct(r.call_share.mean)} [${pct(r.call_share.lower)}, ${pct(r.call_share.upper)}]`],
      ["Equilibrium calls", pct(r.eq_call_share)]]),
    h("h3", { text: "What the model plays" }),
    dl([["Choice", `${a.label} (rank ${a.rank} of ${r.ranking.length})`], [`Gain over equilibrium, ${Math.round(100 * r.level)}% interval`, `${iv(a.gain)} bb/hand`],
      ["P(gain > 0)", pct(a.prob_gain)], ["Fell back to equilibrium", r.falls_back ? "yes (stability check failed)" : "no"],
      ["Its value against your estimated play", `${signed(r.value_vs_you)} bb/hand`], ["Equilibrium's value against it", `${signed(r.eq_value_vs_you)} bb/hand`],
      ["Its exploitability", `${a.exploitability.toFixed(3)} bb/hand`], ["Your estimated exploitability", `${r.your_exploitability.toFixed(3)} bb/hand`]]),
    h("h3", { text: "Model-fit checks" }), fit,
    h("div", { class: "grids" }, grid("Your push range (estimated)", r.your_push, r.eq_push, "Shade: estimated push probability. Outlined: the equilibrium pushes it."),
      grid("Your call range (estimated)", r.your_call, r.eq_call, "Shade: estimated call probability facing a push. Outlined: the equilibrium calls.")),
    h("details", {}, h("summary", { text: "The model's explanation and full ranking" }), h("p", { class: "explain", text: r.explanation }),
      h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["#", "Response", "Gain (bb/hand)", "P(gain>0)", "Exploitability"].map((x) => h("th", { text: x })))),
        h("tbody", {}, r.ranking.map((x, i) => h("tr", {}, h("td", { text: i + 1 }), h("td", { text: x.label }), h("td", { text: iv(x.gain) }), h("td", { text: pct(x.prob_gain) }), h("td", { text: x.exploitability.toFixed(3) }))))))),
    h("p", { class: "note", text: "Estimates assume you play like the model's family: a range ordered by equity, with some noise. The fit checks test that assumption." }));
}
function renderHoldemRead(el, r) {
  const ex = r.exploit;
  fill(el, 
    h("p", { text: `Profile from ${r.hands} completed hand${r.hands === 1 ? "" : "s"} (public information only; your unshown cards are never used).` }),
    h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, h("tr", {}, ["Stat", "Estimate", "95% interval", "Chances"].map((x) => h("th", { text: x })))),
      h("tbody", {}, r.stats.map((s) => h("tr", {}, h("td", { text: s.label }), h("td", { text: s.rate ? pct(s.value) : s.value.toFixed(2) }),
        h("td", { text: s.interval ? `${pct(s.interval[0])}–${pct(s.interval[1])}` : "—" }), h("td", { text: s.opportunities ?? "—" })))))),
    h("h3", { text: "Closest bot archetype" }),
    h("div", {}, r.archetypes.map((a) => h("div", { class: "arch" }, h("span", { text: a.name.replace("_", " ") }), h("div", { class: "bar" }, h("i", { style: `width:${(100 * a.probability).toFixed(1)}%` })), h("span", { text: pct(a.probability) })))),
    h("p", { class: "note", text: r.caveat }),
    ex ? h("div", {}, h("h3", {}, "Exploit slot: ", ex.engine.replaceAll("_", " "), " ", h("span", { class: `badge ${ex.kind}`, text: ex.kind })),
      h("p", { class: "note", text: ex.description }),
      dl([["Advisor decisions", ex.decisions], ["Adjusted by the slot", ex.adjusted], ["Mean shift (total variation)", ex.mean_shift.toFixed(3)]]),
      ex.examples.length ? h("ul", { class: "lines" }, ex.examples.map((x) => h("li", { text: x }))) : null) : null);
}

function renderReport(rep) {
  const r = rep.results;
  const kids = [
    h("p", { text: `${rep.mode === "pushfold" ? `Push/fold at ${rep.settings.stack_bb} bb` : `No-limit hold'em, ${rep.settings.players} players, ${rep.settings.opponents}`} · ${r.hands} hands · seed ${rep.seed}` }),
    dl([["Won", fmtBB(r.bb)], ["All-in EV", fmtBB(r.allin_ev_bb)], ["Per hand", r.hands ? `${signed(r.bb_per_hand)} bb` : "—"], ["All-in EV per hand", r.hands ? `${signed(r.allin_ev_bb_per_hand)} bb` : "—"]])];
  if (rep.model) { const box = h("div", {}); renderRead(box, rep.model, rep.mode); kids.push(h("h3", { text: "The model's read at the end" }), box); }
  if (rep.model_choices) kids.push(h("p", { class: "note", text: `The model's choices over the session: ${Object.entries(rep.model_choices).map(([k, v]) => `${k.replaceAll("_", " ")} ×${v}`).join(", ")}; fallbacks: ${rep.fallbacks}.` }));
  if (rep.opponents) kids.push(h("h3", { text: "Your opponents" }), h("ul", {}, rep.opponents.map((o) => h("li", { text: `${o.player}: ${o.archetype ? o.archetype.replaceAll("_", " ") : `${o.engine.replaceAll("_", " ")} (${o.kind})`}, ${fmtBB(o.net_bb)}` }))));
  kids.push(h("ul", { class: "note" }, rep.notes.map((n) => h("li", { text: n }))));
  if (rep.report_file) kids.push(h("p", { class: "note" }, "Saved: ", h("code", { text: rep.report_file }), h("br"), "Hands: ", h("code", { text: rep.hands_file })));
  $("report-body").replaceChildren(...kids);
}

// ---------------------------------------------------------------- polling and keys
function schedule() {
  clearTimeout(pollTimer);
  const s = state && state.session;
  if (!s || s.status === "ended" || s.status === "error") return;
  const waiting = state.turn || s.status === "hand_over";
  pollTimer = setTimeout(poll, waiting ? 2000 : 300);
}
async function poll() {
  try { render(await api("/api/state")); } catch (e) { setStatus(`Lost the server: ${e.message}`, true); pollTimer = setTimeout(poll, 3000); }
}

document.addEventListener("keydown", (e) => {
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  const inInput = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName);
  const k = e.key.toLowerCase();
  if (inInput) {
    if (document.activeElement === $("raise-amount")) {
      if (e.key === "Enter") { e.preventDefault(); sendRaise(); }
      else if (e.key === "Escape") { $("raise-box").hidden = true; $("actions").querySelector("button")?.focus(); }
    }
    return;
  }
  const s = state && state.session;
  if (k === "m" && s && s.status !== "ended") { e.preventDefault(); toggleRead(!s.read_on); return; }
  if (k === "e" && s) { e.preventDefault(); endSession(); return; }
  if (k === "n" || k === "3") { e.preventDefault(); nextHand(); return; }
  if (state && state.turn && (k === "1" || k === "2")) {
    const b = $("actions").querySelector(`button[data-action="${k === "1" ? "allin" : "fold"}"]`);
    if (b) { e.preventDefault(); b.click(); }
    return;
  }
  if (state && state.turn) {
    const b = $("actions").querySelector(`button[data-key="${k.toUpperCase()}"]`);
    if (b) { e.preventDefault(); b.click(); }
  }
});

$("setup-form").addEventListener("submit", start);
$("setup-form").addEventListener("change", syncSetup);
$("raise-go").addEventListener("click", sendRaise);
$("next-btn").addEventListener("click", nextHand);
$("end-btn").addEventListener("click", endSession);
$("again-btn").addEventListener("click", () => { showSetup = true; render(state); document.querySelector("input[name=mode]:checked").focus(); });
$("read-toggle").addEventListener("change", (e) => toggleRead(e.target.checked));

(async function init() {
  try {
    options = await api("/api/options");
    $("data-dir").textContent = options.data_dir;
    $("stack").min = options.stack[0]; $("stack").max = options.stack[1];
    $("engine").replaceChildren(...options.engines.map((e) => h("option", { value: e.name, selected: e.name === options.default_engine }, `${e.name.replaceAll("_", " ")} (${e.kind})`)));
    syncSetup();
    render(await api("/api/state"));
  } catch (e) { setStatus(e.message, true); }
})();
