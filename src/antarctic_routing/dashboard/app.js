/* Antarctic Ice-Risk Routing dashboard - no external dependencies. */
"use strict";

const $ = (sel) => document.querySelector(sel);
// API location: same origin unless <meta name="antroute-api-base"> names another (separately hosted dashboard).
const API_BASE = (document.querySelector('meta[name="antroute-api-base"]')?.content || "").replace(/\/$/, "");
const apiUrl = (path) => API_BASE + path;
const css = (name) => getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(name).trim();
const SERIES = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5", "--series-6", "--series-7", "--series-8"];
const ICE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
const pct = (v, d = 1) => (v == null || !isFinite(v) ? "–" : (100 * v).toFixed(d) + "%");
const num = (v, d = 0) => (v == null || !isFinite(v) ? "–" : Number(v).toFixed(d));

async function api(path, opts = {}) {
  const res = await fetch(apiUrl(path), { headers: { "Content-Type": "application/json" }, ...opts });
  const text = await res.text();
  let body;
  try { body = JSON.parse(text); } catch { body = text; }
  if (!res.ok) throw new Error(typeof body === "object" && body.detail ? JSON.stringify(body.detail) : String(body));
  return body;
}

function el(tag, attrs = {}, ...children) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v; else if (k === "style") n.style.cssText = v; else n.setAttribute(k, v);
  }
  for (const c of children) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return n;
}

function hexToRgb(h) { const v = parseInt(h.slice(1), 16); return [(v >> 16) & 255, (v >> 8) & 255, v & 255]; }
function rampColor(p) {
  const t = Math.min(1, Math.max(0, p / 100)) * (ICE_RAMP.length - 1);
  const i = Math.min(ICE_RAMP.length - 2, Math.floor(t));
  const a = hexToRgb(ICE_RAMP[i]), b = hexToRgb(ICE_RAMP[i + 1]), f = t - i;
  return a.map((x, k) => Math.round(x + (b[k] - x) * f));
}

function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || canvas.width;
  const h = Math.round(w * (canvas.height / canvas.width));
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

/* ------------------------------------------------------- data status */
const STATUS_TEXT = { real: "Real", historical: "Historical", forecast: "Forecast", schematic: "Schematic",
  unavailable: "Unavailable" };
function statusBadge(status) {
  const s = STATUS_TEXT[status] ? status : "unavailable";
  return el("span", { class: "badge status-" + s }, STATUS_TEXT[s]);
}
const currentTab = () => document.querySelector('.tabs button[aria-selected="true"]').dataset.tab;

function setModeBadge(tab) {
  const b = $("#mode-badge"), st = window.__status;
  let status = "schematic", text = "CONTROLLED-SYNTHETIC DATA";
  if (!st) { status = "unavailable"; text = "DATA STATUS UNKNOWN"; }
  else if (tab === "real") {
    const ok = st.real_data.status === "available";
    status = ok ? "historical" : "unavailable";
    text = ok ? "REAL DATA · HISTORICAL REPLAY" : "REAL DATA UNAVAILABLE";
  } else if (isHist(tab)) {
    const ok = st.historical && st.historical.status === "available";
    status = ok ? "historical" : "unavailable";
    text = ok ? "REAL HISTORICAL DATA · HINDSIGHT FORCING" : "REAL HISTORICAL DATA UNAVAILABLE";
  } else if (tab === "validation") {
    const real = (window.__figsMode || st.figures.execution_mode) === "real";
    status = real ? "real" : "schematic";
    text = real ? "REAL-DATA FIGURES" : "CONTROLLED-SYNTHETIC FIGURES";
  }
  b.className = "badge status-" + status;
  b.textContent = text;
}

/* ------------------------------------------------------------- tabs */
const HIST_TABS = ["hroute", "hwindow", "hvoyage"];
const isHist = (tab) => HIST_TABS.includes(tab);
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  document.querySelectorAll(".tab").forEach((s) => { s.hidden = s.id !== "tab-" + b.dataset.tab; });
  if (b.dataset.tab === "validation") loadFigures();
  if (b.dataset.tab === "real") redrawReal();
  if (b.dataset.tab === "plan" && mapState.plan) drawMap(mapState.plan);
  if (b.dataset.tab === "window" && winState.result) drawWindow(winState.result, budget());
  if (isHist(b.dataset.tab)) redrawHist(b.dataset.tab);
  setModeBadge(b.dataset.tab);
}));
const budget = () => (window.__config ? window.__config.routing.risk_budget : 0.05);

/* ------------------------------------------------------------- map */
const mapState = { plan: null, toScreen: null, screenRoutes: [] };

function drawMap(plan, canvasSel = "#map", st = mapState, legendSel = "#map-legend") {
  const canvas = $(canvasSel);
  const { ctx, w, h } = fitCanvas(canvas);
  const m = plan.map;
  const c = Math.cos(m.rotation_rad), s = Math.sin(m.rotation_rad);
  const rot = (x, y) => [x * c - y * s, x * s + y * c];
  const X1 = m.x0_km + m.nx * m.res_km, Y1 = m.y0_km + m.ny * m.res_km;
  const corners = [[m.x0_km, m.y0_km], [X1, m.y0_km], [m.x0_km, Y1], [X1, Y1]].map(([x, y]) => rot(x, y));
  const minx = Math.min(...corners.map((p) => p[0])), maxx = Math.max(...corners.map((p) => p[0]));
  const miny = Math.min(...corners.map((p) => p[1])), maxy = Math.max(...corners.map((p) => p[1]));
  const pad = 8;
  const scale = Math.min((w - 2 * pad) / (maxx - minx), (h - 2 * pad) / (maxy - miny));
  const e0 = pad - scale * minx + ((w - 2 * pad) - scale * (maxx - minx)) / 2;
  const f0 = pad + scale * maxy + ((h - 2 * pad) - scale * (maxy - miny)) / 2;
  const toScreen = (x, y) => { const [rx, ry] = rot(x, y); return [e0 + scale * rx, f0 - scale * ry]; };
  st.toScreen = toScreen;

  ctx.fillStyle = css("--surface-1");
  ctx.fillRect(0, 0, w, h);

  // raster: one pixel per grid cell, row 0 of the grid is the bottom of the image
  const img = new ImageData(m.nx, m.ny);
  const land = hexToRgb(css("--land")), berg = hexToRgb(css("--berg")), surf = hexToRgb(css("--surface-1"));
  const missing = hexToRgb(css("--missing"));
  let anyMissing = false;
  for (let j = 0; j < m.ny; j++) {
    for (let i = 0; i < m.nx; i++) {
      const k = j * m.nx + i, o = ((m.ny - 1 - j) * m.nx + i) * 4;
      let rgb = surf, a = 255;
      if (m.land[k]) rgb = land;
      else if (m.p_ice[k] == null) { rgb = missing; anyMissing = true; }   // no data is never open water
      else if (m.p_ice[k] > 0) rgb = rampColor(m.p_ice[k]);
      if (!m.land[k] && m.p_berg && m.p_berg[k] >= 5) {
        const f = 0.35 + 0.5 * Math.min(1, m.p_berg[k] / 50);
        rgb = rgb.map((v, q) => Math.round(v * (1 - f) + berg[q] * f));
      }
      img.data.set([...rgb, a], o);
    }
  }
  const off = document.createElement("canvas");
  off.width = m.nx; off.height = m.ny;
  off.getContext("2d").putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  const r = m.res_km;
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr * scale * c * r, dpr * -scale * s * r, dpr * scale * s * r, dpr * scale * c * r,
    dpr * (e0 + scale * (c * m.x0_km - s * Y1)), dpr * (f0 - scale * (s * m.x0_km + c * Y1)));
  ctx.drawImage(off, 0, 0);
  ctx.restore();

  // routes: non-recommended first, recommended last and thicker
  st.screenRoutes = [];
  const order = plan.candidates.map((_, i) => i).sort((a, b) => (a === plan.recommended_index) - (b === plan.recommended_index));
  for (const i of order) {
    const cand = plan.candidates[i];
    const pts = cand.xy_km.map(([x, y]) => toScreen(x, y));
    const rec = i === plan.recommended_index;
    ctx.lineJoin = "round"; ctx.lineCap = "round";
    if (rec) { ctx.strokeStyle = css("--surface-1"); ctx.lineWidth = 6; strokePath(ctx, pts); }
    ctx.strokeStyle = css(SERIES[i % SERIES.length]);
    ctx.lineWidth = rec ? 3.5 : 2;
    ctx.setLineDash(cand.feasible ? [] : [6, 4]);
    strokePath(ctx, pts);
    ctx.setLineDash([]);
    st.screenRoutes.push({ i, pts });
  }
  if (plan.origin_xy_km) marker(ctx, toScreen(...plan.origin_xy_km), css("--series-3"), "Origin");
  if (plan.destination_xy_km) marker(ctx, toScreen(...plan.destination_xy_km), css("--series-4"), "Destination");

  const legend = $(legendSel);
  legend.replaceChildren(
    el("span", {}, el("i", { class: "box", style: `background:${css("--land")}` }), plan.land_label || "Land (schematic)"),
    ...(m.p_berg ? [el("span", {}, el("i", { class: "box", style: `background:${css("--berg")}` }), "Iceberg presence")] : []),
    ...(anyMissing ? [el("span", {}, el("i", { class: "box", style: `background:${css("--missing")}` }), "No data")] : []),
    el("span", {}, el("i", { style: "border-top-style:dashed;border-color:" + css("--muted") }), "Exceeds budget"),
  );
}

function strokePath(ctx, pts) {
  ctx.beginPath();
  pts.forEach(([x, y], k) => (k ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
}

function marker(ctx, [x, y], colour, label) {
  ctx.beginPath(); ctx.arc(x, y, 6, 0, 2 * Math.PI);
  ctx.fillStyle = colour; ctx.fill();
  ctx.lineWidth = 2; ctx.strokeStyle = css("--surface-1"); ctx.stroke();
  ctx.fillStyle = css("--text-primary"); ctx.font = "12px system-ui, sans-serif";
  ctx.fillText(label, x + 9, y - 8);
}

function distToSegment(px, py, [x1, y1], [x2, y2]) {
  const dx = x2 - x1, dy = y2 - y1, L = dx * dx + dy * dy;
  const t = L ? Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / L)) : 0;
  return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
}

function attachMapTip(canvasSel, tipSel, st) {
$(canvasSel).addEventListener("pointermove", (ev) => {
  const tip = $(tipSel);
  if (!st.plan) return;
  const rect = ev.target.getBoundingClientRect();
  const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
  let best = null;
  for (const r of st.screenRoutes) {
    for (let k = 1; k < r.pts.length; k++) {
      const d = distToSegment(px, py, r.pts[k - 1], r.pts[k]);
      if (d < 10 && (!best || d < best.d)) best = { d, i: r.i };
    }
  }
  if (!best) { tip.hidden = true; return; }
  const cand = st.plan.candidates[best.i];
  tip.replaceChildren(
    el("strong", {}, cand.labels.join(", ")),
    row("P(breach)", pct(cand.p_breach)), row("95% upper bound", pct(cand.p_breach_upper)),
    row("Expected time", num(cand.expected_hours, 1) + " h"), row("Fuel index", num(cand.expected_fuel)),
    row("Distance", num(cand.distance_km) + " km"),
  );
  tip.style.left = Math.min(px + 14, rect.width - 200) + "px";
  tip.style.top = (py + 40) + "px";
  tip.hidden = false;
});
$(canvasSel).addEventListener("pointerleave", () => { $(tipSel).hidden = true; });
}
attachMapTip("#map", "#map-tip", mapState);

function row(label, value) { return el("div", { class: "row" }, el("span", {}, label), el("b", {}, value)); }

function statusPill(ok) {
  return el("span", { class: "pill " + (ok ? "ok" : "bad") }, ok ? "✓ meets budget" : "✕ exceeds budget");
}

function renderPlan(plan) {
  mapState.plan = plan;
  drawMap(plan);
  $("#verdict").textContent = plan.explanation;
  const tbody = $("#cand-table tbody");
  tbody.replaceChildren(...plan.candidates.map((c, i) => {
    const tr = el("tr", { class: i === plan.recommended_index ? "rec" : "" },
      el("td", {}, el("span", { class: "swatch", style: `background:${css(SERIES[i % SERIES.length])}` })),
      el("td", {}, (i === plan.recommended_index ? "★ " : "") + c.labels.join(", ")),
      el("td", {}, statusPill(c.feasible)),
      el("td", { class: "num" }, num(c.distance_km)),
      el("td", { class: "num" }, num(c.expected_hours, 1)),
      el("td", { class: "num" }, num(c.expected_fuel)),
      el("td", { class: "num" }, pct(c.p_breach)),
      el("td", { class: "num" }, pct(c.p_breach_upper)));
    return tr;
  }));
}

$("#plan-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const body = {
    departure: f.get("departure"), scenarios: Number(f.get("scenarios")),
    resolution_km: Number(f.get("resolution_km")), scenario_routes: 2,
    icebergs: f.get("berg") ? [{ id: "BERG-1", lat: Number(f.get("berg_lat")), lon: Number(f.get("berg_lon")) }] : [],
  };
  const status = $("#plan-status");
  status.textContent = "Planning across joint scenarios…";
  $("#plan-btn").disabled = true;
  try {
    const job = await api("/routes?wait=true", { method: "POST", body: JSON.stringify(body) });
    if (job.status !== "done") throw new Error(job.error || "planning failed");
    renderPlan(job.result);
    status.textContent = `${job.result.n_scenarios} joint scenarios · ${job.result.execution_mode}`;
  } catch (e) {
    status.textContent = "Error: " + e.message;
  } finally {
    $("#plan-btn").disabled = false;
  }
});

/* ------------------------------------------------------------- window */
const winState = { result: null, bars: [] };

function drawWindow(result, budget, canvasSel = "#window-chart", st = winState) {
  const canvas = $(canvasSel);
  const { ctx, w, h } = fitCanvas(canvas);
  const opts = result.options;
  const left = 52, right = 12, top = 16, bottom = 46;
  const peak = Math.max(budget, ...opts.map((o) => Math.min(1, o.p_breach_upper)));
  const step = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25].find((st) => (1.15 * peak) / st <= 5) || 0.25;
  const ymax = Math.min(1, Math.max(step * 2, Math.ceil((1.15 * peak) / step) * step));
  const y = (v) => top + (h - top - bottom) * (1 - v / ymax);
  ctx.fillStyle = css("--surface-1"); ctx.fillRect(0, 0, w, h);
  ctx.font = "11px system-ui, sans-serif"; ctx.fillStyle = css("--muted"); ctx.strokeStyle = css("--grid"); ctx.lineWidth = 1;
  for (let v = 0; v <= ymax + 1e-9; v += step) {
    const yy = y(v);
    ctx.beginPath(); ctx.moveTo(left, yy); ctx.lineTo(w - right, yy); ctx.stroke();
    ctx.fillText(pct(v, 0), 8, yy + 4);
  }
  const bw = (w - left - right) / opts.length;
  st.bars = [];
  opts.forEach((o, k) => {
    const x = left + k * bw + bw * 0.15, bwid = bw * 0.7, v = Math.min(1, o.p_breach_upper);
    ctx.fillStyle = o.feasible ? css("--good") : css("--critical");
    const yy = y(v), r = Math.min(4, bwid / 2);
    ctx.beginPath();
    ctx.moveTo(x, y(0)); ctx.lineTo(x, yy + r); ctx.quadraticCurveTo(x, yy, x + r, yy);
    ctx.lineTo(x + bwid - r, yy); ctx.quadraticCurveTo(x + bwid, yy, x + bwid, yy + r); ctx.lineTo(x + bwid, y(0));
    ctx.fill();
    ctx.fillStyle = css("--text-primary");
    ctx.fillText(o.feasible ? "✓" : "✕", x + bwid / 2 - 4, yy - 4);
    ctx.fillStyle = css("--muted");
    ctx.save(); ctx.translate(x + bwid / 2, h - bottom + 12); ctx.rotate(-0.6);
    ctx.fillText(o.departure.slice(5), -18, 8); ctx.restore();
    st.bars.push({ x, w: bwid, o });
  });
  ctx.strokeStyle = css("--text-primary"); ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(left, y(budget)); ctx.lineTo(w - right, y(budget)); ctx.stroke(); ctx.setLineDash([]);
  const label = `risk budget ${pct(budget, 0)}`;
  ctx.fillStyle = css("--surface-1");
  ctx.fillRect(left + 2, y(budget) - 17, ctx.measureText(label).width + 6, 14);
  ctx.fillStyle = css("--text-primary"); ctx.fillText(label, left + 5, y(budget) - 6);
  if (result.selected) {
    const b = st.bars.find((q) => q.o.departure === result.selected);
    if (b) {
      ctx.fillStyle = css("--text-primary");
      ctx.fillText("▼ selected", b.x + b.w / 2 - 26, Math.max(top + 10, y(Math.min(1, b.o.p_breach_upper)) - 18));
    }
  }
}

function attachWindowTip(canvasSel, tipSel, st) {
$(canvasSel).addEventListener("pointermove", (ev) => {
  const tip = $(tipSel);
  const rect = ev.target.getBoundingClientRect();
  const px = ev.clientX - rect.left;
  const b = st.bars.find((q) => px >= q.x - 4 && px <= q.x + q.w + 4);
  if (!b) { tip.hidden = true; return; }
  tip.replaceChildren(el("strong", {}, b.o.departure), row("95% upper bound", pct(b.o.p_breach_upper)),
    row("P(breach)", pct(b.o.p_breach)), row("Expected time", num(b.o.expected_hours, 1) + " h"),
    row("Fuel index", num(b.o.expected_fuel)), row("Status", b.o.feasible ? "✓ meets budget" : "✕ exceeds budget"));
  tip.style.left = Math.min(px + 14, rect.width - 200) + "px"; tip.style.top = "60px"; tip.hidden = false;
});
$(canvasSel).addEventListener("pointerleave", () => { $(tipSel).hidden = true; });
}
attachWindowTip("#window-chart", "#window-tip", winState);

$("#window-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const status = $("#window-status");
  status.textContent = "Sweeping departure dates…";
  try {
    const job = await api("/departures?wait=true", { method: "POST", body: JSON.stringify({
      start: f.get("start"), end: f.get("end"), step_days: Number(f.get("step_days")),
      scenarios: Number(f.get("scenarios")), resolution_km: 25 }) });
    if (job.status !== "done") throw new Error(job.error || "sweep failed");
    winState.result = job.result;
    drawWindow(job.result, budget());
    $("#window-verdict").textContent = job.result.explanation + " Rule: " + job.result.rule + ".";
    $("#window-table tbody").replaceChildren(...job.result.options.map((o) => el("tr", {},
      el("td", {}, o.departure), el("td", {}, statusPill(o.feasible)), el("td", { class: "num" }, pct(o.p_breach_upper)),
      el("td", { class: "num" }, num(o.expected_hours, 1)), el("td", { class: "num" }, num(o.expected_fuel)))));
    status.textContent = "";
  } catch (e) { status.textContent = "Error: " + e.message; }
});

/* ------------------------------------------------------------- voyage */
const voyage = { id: null, latlon: [] };

function setRoute(route) {
  voyage.latlon = route.latlon;
  const slider = $("#wp-slider");
  slider.max = String(route.latlon.length - 2);
  slider.value = String(Math.min(Number(slider.value), route.latlon.length - 2));
  updateWp();
}
function updateWp() {
  const k = Number($("#wp-slider").value), p = voyage.latlon[k];
  $("#wp-out").textContent = p ? `#${k} (${p[0].toFixed(2)}, ${p[1].toFixed(2)})` : "–";
}
$("#wp-slider").addEventListener("input", updateWp);

async function refreshLog() {
  const hist = await api(`/voyages/${voyage.id}/history`);
  $("#voyage-log").replaceChildren(...hist.events.map((e) => el("li", {},
    el("div", {}, el("strong", {}, e.event === "planned" ? "Planned" : (e.action || "").toUpperCase()),
      e.alert ? " ⚠ alert" : "", e.triggers && e.triggers.length ? " · " + e.triggers.join(", ") : ""),
    el("div", {}, e.explanation || ""),
    el("div", { class: "meta" }, (e.logged_at || e.at || "") + (e.issued ? " · forecast " + e.issued : "")))));
}

$("#voyage-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const status = $("#voyage-status");
  status.textContent = "Creating voyage…";
  try {
    const v = await api("/voyages", { method: "POST", body: JSON.stringify({
      departure: f.get("departure"), scenarios: 120, resolution_km: 25, scenario_routes: 0 }) });
    voyage.id = v.voyage_id;
    setRoute(v.route);
    $("#replan-btn").disabled = false;
    for (const [id, fmt] of [["#export-geojson", "geojson"], ["#export-csv", "csv"]]) {
      const a = $(id); a.href = apiUrl(`/voyages/${voyage.id}/export?format=${fmt}`); a.hidden = false;
    }
    status.textContent = "Voyage " + voyage.id + " · " + v.explanation;
    await refreshLog();
  } catch (e) { status.textContent = "Error: " + e.message; }
});

$("#replan-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const p = voyage.latlon[Number($("#wp-slider").value)];
  const status = $("#voyage-status");
  status.textContent = "Replanning…";
  try {
    const d = await api(`/voyages/${voyage.id}/replan`, { method: "POST", body: JSON.stringify({
      lat: p[0], lon: p[1], issued: f.get("issued"), scenarios: 120, data_age_hours: Number(f.get("age")) }) });
    setRoute(d.route);
    $("#wp-slider").value = "0"; updateWp();
    status.textContent = d.explanation;
    await refreshLog();
  } catch (e) { status.textContent = "Error: " + e.message; }
});

/* ------------------------------------------------------------- validation */
async function loadFigures() {
  const box = $("#figs");
  if (box.dataset.loaded) return;
  try {
    const figs = await api("/figures");
    window.__figsMode = figs.execution_mode;
    const real = figs.execution_mode === "real";
    $("#figs-badge").replaceWith(Object.assign(statusBadge(real ? "real" : "schematic"), { id: "figs-badge" }));
    $("#figs-note").textContent = real ? "Figures from real-data runs." :
      "These figures come from controlled-synthetic runs, not real conditions.";
    setModeBadge(currentTab());
    box.replaceChildren(...figs.figures.map((fig) => el("figure", {},
      el("img", { src: apiUrl(fig.url), alt: fig.caption, loading: "lazy" }), el("figcaption", {}, fig.caption))));
    if (!figs.figures.length) box.replaceChildren(el("p", { class: "sub" }, "No figures yet - run the CLI reports."));
    box.dataset.loaded = "1";
  } catch (e) { box.replaceChildren(el("p", {}, "Could not load figures: " + e.message)); }
}

/* ------------------------------------------------------------- real data */
// Everything here is read from the backend's verified real-data bundle; the page computes nothing.
const realState = { plan: null, toScreen: null, screenRoutes: [], layers: {}, route: null, dates: null, k: 0,
  tracks: null };
const realWinState = { result: null, bars: [] };
attachMapTip("#real-map", "#real-map-tip", realState);
attachWindowTip("#real-window-chart", "#real-window-tip", realWinState);
const short = (h) => (h ? String(h).slice(0, 12) + "…" : "–");

function showRealUnavailable(reason) {
  $("#real-unavailable").hidden = false;
  $("#real-content").hidden = true;
  $("#real-reason").textContent = reason;
}

function kv(rows) {
  return rows.map(([k, v]) => el("tr", {}, el("th", {}, k), el("td", {}, v)));
}

async function showRealLayer(k) {
  const info = realState.dates.layers[k];
  const observed = k === 0 && info.source === "observed";
  if (!realState.layers[k]) realState.layers[k] = await api(observed ? "/real/sea-ice" : `/real/forecast-map?layer=${k}`);
  const L = realState.layers[k], g = L.grid, route = realState.route, xy = route.xy_km || [];
  const ev = route.evaluation || {};
  realState.k = k;
  realState.plan = {
    map: { nx: g.nx, ny: g.ny, res_km: g.res_km, x0_km: g.x0_km, y0_km: g.y0_km, rotation_rad: g.rotation_rad,
      land: L.land, p_ice: observed ? L.concentration_pct : L.p_ice_ge_limit_pct, p_berg: observed ? null : L.p_berg_pct },
    candidates: xy.length ? [{ ...ev, xy_km: xy, labels: [`selected route, departing ${route.departure}`], feasible: true }] : [],
    recommended_index: xy.length ? 0 : null,
    origin_xy_km: xy[0], destination_xy_km: xy[xy.length - 1],
    land_label: "Land (sea-ice product mask; not a navigational coastline)",
  };
  if (!$("#tab-real").hidden) { drawMap(realState.plan, "#real-map", realState, "#real-map-legend"); drawTracks(); }
  $("#real-layer-out").textContent = `${info.date} · ${observed ? "observed (Historical)" : info.source + " (Forecast)"}`;
  $("#real-ramp-label").textContent = observed ? "100% observed concentration"
    : `100% P(ice ≥ ${pct(L.tau, 0)}) across ${L.n_members} members`;
}

// Ensemble-mean iceberg drift (forecast) up to the shown day; a hollow ring marks the position on that day.
function drawTracks() {
  const tracks = realState.tracks;
  if (!tracks || !tracks.length || !realState.toScreen) return;
  const ctx = $("#real-map").getContext("2d"), col = css("--berg");
  ctx.save();
  ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
  for (const t of tracks) {
    const pts = t.daily_mean.filter((d) => d.layer <= realState.k && d.x_km != null)
      .map((d) => realState.toScreen(d.x_km, d.y_km));
    if (!pts.length) continue;
    strokePath(ctx, pts);
    const [x, y] = pts[pts.length - 1];
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.stroke();
    ctx.font = "10px system-ui, sans-serif"; ctx.fillText(t.id, x + 6, y + 3);
    ctx.setLineDash([4, 3]);
  }
  ctx.restore();
  const legend = $("#real-map-legend");
  if (!legend.querySelector(".track-key")) {
    legend.append(el("span", { class: "track-key" }, el("i", { style: `border-top-style:dashed;border-color:${col}` }),
      "Iceberg mean drift (forecast)"));
  }
}

function redrawReal() {
  if (realState.plan) { drawMap(realState.plan, "#real-map", realState, "#real-map-legend"); drawTracks(); }
  if (realWinState.result) drawWindow(realWinState.result, realWinState.result.risk_budget, "#real-window-chart", realWinState);
}

async function loadReal(status) {
  const real = status.real_data;
  if (real.status !== "available") { showRealUnavailable(real.reason || "The real-data bundle is unavailable."); return; }
  const [route, win, bergs, dates, prov] = await Promise.all([api("/real/route"), api("/real/departure-window"),
    api("/real/icebergs"), api("/real/forecast-dates"), api("/provenance")]);
  $("#real-unavailable").hidden = true;
  $("#real-content").hidden = false;
  $("#real-bundle").textContent = `Bundle ${real.bundle_id} · forecast issued ${real.issue} · execution mode ` +
    `"${real.execution_mode}". Historical replay with real inputs; not a live forecast.`;
  $("#real-sources").replaceChildren(...real.sources.map((s) => el("li", {}, statusBadge(s.status),
    el("span", { class: "src-name" }, s.name.replace(/_/g, " ")), el("span", {}, s.label))));
  $("#real-limits").replaceChildren(...(real.limitations || []).map((t) => el("li", {}, t)));

  realState.route = route; realState.dates = dates; realState.tracks = bergs.tracks;
  const ev = route.evaluation || {}, pres = route.iceberg_presence_on_route || {};
  $("#real-verdict").textContent = route.explanation || "";
  $("#real-summary tbody").replaceChildren(...kv(route.selected ? [
    ["Selected departure", route.selected],
    ["Expected time", `${num(ev.expected_hours, 1)} h (p10–p90 ${num(ev.hours_p10, 2)}–${num(ev.hours_p90, 2)})`],
    ["Expected fuel index", num(ev.expected_fuel, 1)],
    ["Distance", `${num(ev.distance_km, 1)} km`],
    ["Breaches", `${ev.breaches} of ${ev.n_scenarios} joint scenarios`],
    ["P(breach) 95% upper bound", `${pct(ev.p_breach_upper, 2)} (budget ${pct(win.risk_budget, 0)})`],
    ["Route cells with iceberg presence", `${pres.route_cells_with_any_presence ?? "–"} of ${(route.cells_row_col || []).length}`],
  ] : [["Selected departure", "none met the risk budget"]]));

  realWinState.result = win;
  $("#real-window-verdict").textContent = `${win.explanation} Rule: ${win.rule}.`;

  const src = bergs.source || {};
  $("#real-berg-badge").replaceWith(Object.assign(statusBadge(bergs.status), { id: "real-berg-badge" }));
  $("#real-berg-source").textContent = `${src.source || "–"}, list of ${(src.update_dates || []).join(", ")} ` +
    `(${src.filename || "–"}). Drift: beta ${bergs.drift?.beta}, spread factor ${bergs.drift?.spread_factor}, ` +
    `radius ${num((bergs.drift?.radius_m || 0) / 1000)} km.`;
  $("#real-bergs tbody").replaceChildren(
    ...bergs.drifted.map((b) => {
      const tr = (bergs.tracks || []).find((t) => t.id === b.id), end = tr && tr.daily_mean[tr.daily_mean.length - 1];
      const drift = end && end.lat != null ? ` → day ${end.layer}: ${num(end.lat, 2)}, ${num(end.lon, 2)} ` +
        `(±${num(end.spread_p90_km)} km p90)` : "";
      return el("tr", {}, el("td", {}, b.id), el("td", { class: "num" }, num(b.lat, 2)),
        el("td", { class: "num" }, num(b.lon, 2)), el("td", {}, "drifted" + drift));
    }),
    el("tr", {}, el("td", { colspan: "4" }, `${bergs.outside_grid.length} more outside the routing grid (not drifted)`)));

  const man = prov.manifest || {}, env = man.environment || {}, b = prov.bundle || {};
  $("#real-prov tbody").replaceChildren(...kv([
    ["Bundle", `${b.bundle_id} (created ${b.created_utc})`],
    ["Plan output (canonical sha256)", short(b.plan_window_canonical_sha256)],
    ["Forecast model", `${man.sea_ice?.forecast_model?.id || "–"} · ${short(man.sea_ice?.forecast_model?.sha256)}`],
    ...Object.entries(man.inputs || {}).map(([k, v]) => [`Input: ${k}`, `${short(v.sha256)} ${v.verified ? "✓ verified" : ""}`]),
    ["Code", `${short(env.git_commit)}${env.worktree_dirty ? " + uncommitted changes" : ""}`],
    ["Python / torch", `${env.python || "–"} / ${(env.packages || {}).torch || "–"}`],
  ]));
  $("#real-prov-link").href = apiUrl("/provenance");

  const slider = $("#real-layer");
  slider.max = String(dates.layers.length - 1);
  slider.value = "0";
  slider.addEventListener("input", () => showRealLayer(Number(slider.value)).catch((e) => {
    $("#real-layer-out").textContent = "Error: " + e.message; }));
  await showRealLayer(0);
  redrawReal();
}

/* ------------------------------------------------------------- real historical data */
// The backend runs the frozen real-data planner on past issue dates; the page only draws what comes back.
const hist = { status: null, dates: null };
const hRouteState = { plan: null, toScreen: null, screenRoutes: [], tracks: null, k: 0 };
const hWinState = { result: null, bars: [] };
const hWinMapState = { plan: null, toScreen: null, screenRoutes: [], tracks: null, k: 0 };
const hVoyState = { plan: null, toScreen: null, screenRoutes: [], tracks: null, k: 0 };
const hReplayState = { plan: null, toScreen: null, screenRoutes: [], bergs: null };
const hvoy = { id: null, latlon: [], lastIssue: null };
attachMapTip("#hroute-map", "#hroute-map-tip", hRouteState);
attachMapTip("#hwindow-map", "#hwindow-map-tip", hWinMapState);
attachMapTip("#hvoyage-map", "#hvoyage-map-tip", hVoyState);
attachWindowTip("#hwindow-chart", "#hwindow-tip", hWinState);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function errText(e) {
  try { const d = JSON.parse(e.message); return d.reason || e.message; } catch { return e.message; }
}

// Background job + polling, so long real-data runs never hold a request open.
async function runJob(path, body, statusEl, msg) {
  const first = await api(path, { method: "POST", body: JSON.stringify(body) });
  let job = first;
  const t0 = Date.now();
  while (job.status === "queued" || job.status === "running") {
    statusEl.textContent = `${msg} (${job.status}, ${Math.round((Date.now() - t0) / 1000)} s)`;
    await sleep(1500);
    job = await api(`/jobs/${first.job_id}`);
  }
  if (job.status !== "done") throw new Error(job.error || "the computation failed");
  return job.result;
}

function setupHistorical(st) {
  hist.status = st;
  for (const tab of HIST_TABS) {
    const sec = $("#tab-" + tab), slot = sec.querySelector(".hist-slot"), body = sec.querySelector(".hist-body");
    if (!st || st.status !== "available") {
      const card = $("#hist-unavailable-template").content.cloneNode(true);
      card.querySelector('[data-field="reason"]').textContent =
        st ? `${st.status}: ${st.reason || "no reason given"}` : "The data status could not be read.";
      slot.replaceChildren(card);
      body.hidden = true;
      continue;
    }
    const note = $("#hist-note-template").content.cloneNode(true);
    note.querySelector('[data-field="disclosure"]').textContent = st.hindsight_forcing;
    slot.replaceChildren(note);
    body.hidden = false;
  }
}

function seasonOf(kind, day) {
  return ((hist.dates && hist.dates.seasons[kind]) || []).find((s) => s.first <= day && day <= s.last) || null;
}

function seasonText(s) {
  if (!s) return { text: "", warn: false };
  if (s.out_of_sample) {
    return { text: `${s.season} season: out-of-sample for the sea-ice U-Net and the iceberg drift calibration.`,
      warn: false };
  }
  return { text: `${s.season} season is in-sample: ${s.in_sample_notes.join("; ")}. Results here are not ` +
    "independent evidence of skill.", warn: true };
}

function showSeasonFlag(sel, s) {
  const p = $(sel);
  if (!p) return;
  const t = seasonText(s);
  p.textContent = t.text;
  p.classList.toggle("in-sample", t.warn);
}

function flagForForm(form, kind) {
  const day = form.querySelector('input[type="date"]').value;
  const s = seasonOf(kind, day);
  const target = form.dataset.flag;
  if (!target) return;
  if (!s) {
    const ranges = hist.dates.seasons[kind].map((x) => `${x.first} to ${x.last}`).join(", ");
    $(target).textContent = day ? `No Real Historical Data for ${day}. Issue dates run ${ranges} ` +
      "(some single days inside these ranges are missing upstream)." : "";
    $(target).classList.add("in-sample");
  } else showSeasonFlag(target, s);
}

async function loadHistoricalDates() {
  const d = await api("/real/historical/dates");
  hist.dates = d;
  $("#hroute-form").dataset.flag = "#hroute-season";
  $("#hwindow-form").dataset.flag = "#hwindow-season";
  document.querySelectorAll("[data-season-pick]").forEach((sel) => {
    const kind = sel.dataset.seasonPick, seasons = d.seasons[kind], form = sel.form;
    const input = form.querySelector('input[type="date"]');
    const dates = kind === "route" ? d.route_dates : d.window_dates;
    sel.replaceChildren(...seasons.map((s) => el("option", { value: s.first },
      `${s.season} · ${s.out_of_sample ? "out-of-sample" : "in-sample"} · ${s.n_dates} dates`)));
    input.min = dates[0];
    input.max = dates[dates.length - 1];
    const pick = seasons.find((s) => s.season === "2023-24") || seasons[seasons.length - 1];
    sel.value = pick.first;
    input.value = pick.first;
    sel.addEventListener("change", () => { input.value = sel.value; flagForForm(form, kind); });
    input.addEventListener("change", () => flagForForm(form, kind));
    flagForForm(form, kind);
  });
}

// Ensemble-mean iceberg drift up to layer k (forecast); a ring marks the position on that layer.
function drawTrackLines(canvasSel, st, tracks, k, legendSel) {
  if (!tracks || !tracks.length || !st.toScreen) return;
  const ctx = $(canvasSel).getContext("2d"), col = css("--berg");
  ctx.save();
  ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.5;
  for (const t of tracks) {
    const pts = t.daily_mean.filter((d) => d.layer <= k && d.x_km != null).map((d) => st.toScreen(d.x_km, d.y_km));
    if (!pts.length) continue;
    ctx.setLineDash([4, 3]); strokePath(ctx, pts); ctx.setLineDash([]);
    const [x, y] = pts[pts.length - 1];
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.stroke();
    ctx.font = "10px system-ui, sans-serif"; ctx.fillText(t.id, x + 6, y + 3);
  }
  ctx.restore();
  const legend = $(legendSel);
  legend.append(el("span", {}, el("i", { style: `border-top-style:dashed;border-color:${col}` }),
    "Iceberg mean drift (forecast)"));
}

function drawHistMap(canvasSel, st, legendSel) {
  if (!st.plan) return;
  drawMap(st.plan, canvasSel, st, legendSel);
  drawTrackLines(canvasSel, st, st.tracks, st.k, legendSel);
}

function candRows(plan) {
  return plan.candidates.map((c, i) => el("tr", { class: i === plan.recommended_index ? "rec" : "" },
    el("td", {}, el("span", { class: "swatch", style: `background:${css(SERIES[i % SERIES.length])}` })),
    el("td", {}, (i === plan.recommended_index ? "★ " : "") + c.labels.join(", ")),
    el("td", {}, statusPill(c.feasible)),
    el("td", { class: "num" }, num(c.distance_km)), el("td", { class: "num" }, num(c.expected_hours, 1)),
    el("td", { class: "num" }, num(c.expected_fuel)), el("td", { class: "num" }, pct(c.p_breach)),
    el("td", { class: "num" }, pct(c.p_breach_upper))));
}

function inputRows(h) {
  const files = (f) => (f.files || []).map((x) => x.file).join(", ");
  const b = h.icebergs, p = h.parameters, s = h.season;
  return kv([
    ["Forecast issued", `${h.issue} (sea ice observed up to this day; scenario days ${h.scenario_days.join(" to ")})`],
    ["Season", s ? `${s.season} · ${s.out_of_sample ? "out-of-sample" : "in-sample: " + s.in_sample_notes.join("; ")}` : "–"],
    ["Sea ice", `${h.sea_ice.source} · ${h.sea_ice.file} (${short(h.sea_ice.sha256)})`],
    ["Forecast model", `${h.forecast_model.id} (${short(h.forecast_model.sha256)}), ${h.forecast_model.lead_days ?? "–"}-day leads`],
    ["Winds (hindsight)", `${h.forcing.winds.product}: ${files(h.forcing.winds)}`],
    ["Currents (hindsight)", `${h.forcing.currents.product}: ${files(h.forcing.currents)}`],
    ["Icebergs", b ? `USNIC list of ${b.list_date} (${b.age_days} d old): ${b.drifted.length} drifted` +
      `${b.drifted.length ? " (" + b.drifted.join(", ") + ")" : ""}; ${b.outside_grid.length} outside the grid` : "–"],
    ["Iceberg drift", `beta ${p.drift_beta}, alpha scale ${p.drift_alpha_scale}, spread factor ` +
      `${p.drift_spread_factor}, radius ${p.berg_radius_km} km`],
    ["Scenarios", `${p.members} joint scenarios, seed ${p.seed}`],
    ["Probability calibration", h.probability_calibration.applied_in_route_risk ? "applied" : "not applied to route risk"],
    ["Vessel", `${h.vessel.name}, ice class ${h.vessel.ice_class}, limit ${pct(h.vessel.max_ice_concentration, 0)} (${h.vessel.note})`],
  ]);
}

$("#hroute-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const issue = new FormData(ev.target).get("issue");
  const status = $("#hroute-status");
  $("#hroute-btn").disabled = true;
  try {
    const res = await runJob("/real/historical/routes", { issue }, status,
      "Running the real-data planner (the first run also loads the model)");
    hRouteState.plan = res;
    hRouteState.tracks = res.iceberg_tracks;
    hRouteState.k = res.map.layer_day;
    drawHistMap("#hroute-map", hRouteState, "#hroute-map-legend");
    $("#hroute-verdict").textContent = res.explanation;
    $("#hroute-table tbody").replaceChildren(...candRows(res));
    $("#hroute-inputs tbody").replaceChildren(...inputRows(res.historical));
    $("#hroute-ramp").textContent = `100% P(ice ≥ limit), day ${res.map.layer_day} after issue`;
    showSeasonFlag("#hroute-season", res.historical.season);
    status.textContent = `${res.n_scenarios} joint scenarios · ${res.data_label}`;
  } catch (e) {
    status.textContent = "Error: " + errText(e);
  } finally {
    $("#hroute-btn").disabled = false;
  }
});

$("#hwindow-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const status = $("#hwindow-status");
  $("#hwindow-btn").disabled = true;
  try {
    const res = await runJob("/real/historical/departures",
      { issue: f.get("issue"), window_days: Number(f.get("window_days")) }, status,
      "Planning every departure date from one real forecast");
    hWinState.result = res;
    drawWindow(res, res.risk_budget, "#hwindow-chart", hWinState);
    $("#hwindow-verdict").textContent = `${res.explanation} Rule: ${res.rule}.`;
    $("#hwindow-table tbody").replaceChildren(...res.options.map((o) => el("tr", {},
      el("td", {}, o.departure), el("td", { class: "num" }, o.lead_days ?? "–"), el("td", {}, o.support || "–"),
      el("td", {}, statusPill(o.feasible)), el("td", { class: "num" }, pct(o.p_breach_upper)),
      el("td", { class: "num" }, num(o.expected_hours, 1)), el("td", { class: "num" }, num(o.expected_fuel)))));
    const sel = res.selected_route;
    hWinMapState.plan = { map: res.map, candidates: sel ? [sel] : [], recommended_index: sel ? 0 : null,
      origin_xy_km: res.origin_xy_km, destination_xy_km: res.destination_xy_km, land_label: res.land_label };
    hWinMapState.tracks = res.iceberg_tracks;
    hWinMapState.k = res.map.layer_day;
    drawHistMap("#hwindow-map", hWinMapState, "#hwindow-map-legend");
    $("#hwindow-map-title").textContent = res.selected ?
      `Selected departure ${res.selected} (map: day ${res.map.layer_day} after issue)` : "No departure meets the budget";
    $("#hwindow-inputs tbody").replaceChildren(...inputRows(res.historical));
    showSeasonFlag("#hwindow-season", res.historical.season);
    status.textContent = `${res.n_scenarios} joint scenarios · ${res.data_label}`;
  } catch (e) {
    status.textContent = "Error: " + errText(e);
  } finally {
    $("#hwindow-btn").disabled = false;
  }
});

/* voyage */
function setHRoute(route) {
  hvoy.latlon = route.latlon;
  const slider = $("#hwp-slider");
  slider.max = String(route.latlon.length - 2);
  slider.value = String(Math.min(Number(slider.value), route.latlon.length - 2));
  updateHWp();
}
function updateHWp() {
  const k = Number($("#hwp-slider").value), p = hvoy.latlon[k];
  $("#hwp-out").textContent = p ? `#${k} (${p[0].toFixed(2)}, ${p[1].toFixed(2)})` : "–";
}
$("#hwp-slider").addEventListener("input", updateHWp);

function showVoyageMap(res, route) {
  hVoyState.plan = { map: res.map, candidates: [route], recommended_index: 0, origin_xy_km: route.xy_km[0],
    destination_xy_km: route.xy_km[route.xy_km.length - 1], land_label: res.land_label };
  hVoyState.tracks = res.iceberg_tracks;
  hVoyState.k = res.map.layer_day;
  if (!$("#tab-hvoyage").hidden) drawHistMap("#hvoyage-map", hVoyState, "#hvoyage-map-legend");
}

function renderLog(sel, events) {
  $(sel).replaceChildren(...events.map((e) => el("li", {},
    el("div", {}, el("strong", {}, e.event === "planned" ? "Planned" : (e.action || e.event || "").toUpperCase()),
      e.alert ? " ⚠ alert" : "", e.triggers && e.triggers.length ? " · " + e.triggers.join(", ") : ""),
    el("div", {}, e.explanation || ""),
    el("div", { class: "meta" }, [e.logged_at || e.at || e.day || "", e.issued ? "forecast " + e.issued : "",
      e.usnic_list ? "USNIC list " + e.usnic_list : ""].filter(Boolean).join(" · ")))));
}

$("#hvoyage-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const issue = new FormData(ev.target).get("issue");
  const status = $("#hvoyage-status");
  status.textContent = "Planning the voyage with the real-data planner…";
  try {
    const v = await api("/real/historical/voyages", { method: "POST", body: JSON.stringify({ issue }) });
    hvoy.id = v.voyage_id;
    hvoy.lastIssue = issue;
    setHRoute(v.route);
    showVoyageMap(v, v.route);
    const input = $("#hreplan-form").querySelector('input[name="issued"]');
    input.min = issue;
    input.max = hist.dates.route_dates[hist.dates.route_dates.length - 1];
    const next = new Date(issue + "T00:00:00Z");
    next.setUTCDate(next.getUTCDate() + 1);
    input.value = next.toISOString().slice(0, 10);
    $("#hreplan-btn").disabled = false;
    for (const [id, fmt] of [["#hexport-geojson", "geojson"], ["#hexport-csv", "csv"]]) {
      const a = $(id); a.href = apiUrl(`/voyages/${hvoy.id}/export?format=${fmt}`); a.hidden = false;
    }
    const s = seasonText(v.historical.season);
    status.textContent = `Voyage ${hvoy.id} · ${v.explanation}`;
    $("#hvoyage-age").textContent = `Icebergs: USNIC list of ${v.icebergs.list_date} (${v.icebergs.age_days} d old). ${s.text}`;
    renderLog("#hvoyage-log", (await api(`/voyages/${hvoy.id}/history`)).events);
  } catch (e) { status.textContent = "Error: " + errText(e); }
});

$("#hreplan-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const issued = new FormData(ev.target).get("issued");
  const p = hvoy.latlon[Number($("#hwp-slider").value)];
  const status = $("#hvoyage-status");
  status.textContent = "Replanning with the forecast issued " + issued + "…";
  $("#hreplan-btn").disabled = true;
  try {
    const d = await api(`/real/historical/voyages/${hvoy.id}/replan`, { method: "POST",
      body: JSON.stringify({ lat: p[0], lon: p[1], issued }) });
    hvoy.lastIssue = issued;
    $("#hreplan-form").querySelector('input[name="issued"]').min = issued;
    setHRoute(d.route);
    $("#hwp-slider").value = "0"; updateHWp();
    showVoyageMap(d, d.route);
    status.textContent = d.explanation;
    $("#hvoyage-age").textContent = `${d.data_age_basis} USNIC list of ${d.usnic_list} (${d.usnic_age_days} d old).`;
    renderLog("#hvoyage-log", (await api(`/voyages/${hvoy.id}/history`)).events);
  } catch (e) { status.textContent = "Error: " + errText(e); }
  finally { $("#hreplan-btn").disabled = false; }
});

/* replay */
function drawReplay() {
  const st = hReplayState;
  if (!st.plan) return;
  drawMap(st.plan, "#hreplay-map", st, "#hreplay-map-legend");
  const ctx = $("#hreplay-map").getContext("2d"), col = css("--berg");
  ctx.save(); ctx.strokeStyle = col; ctx.fillStyle = col; ctx.lineWidth = 1.5;
  for (const b of st.bergs || []) {
    const [x, y] = st.toScreen(b.xy_km[0], b.xy_km[1]);
    ctx.beginPath(); ctx.arc(x, y, 4, 0, 2 * Math.PI); ctx.stroke();
    ctx.font = "10px system-ui, sans-serif"; ctx.fillText(b.id, x + 6, y + 3);
  }
  ctx.restore();
  $("#hreplay-map-legend").replaceChildren(
    el("span", {}, el("i", { class: "box", style: `background:${css("--land")}` }), st.plan.land_label),
    el("span", {}, el("i", { style: `border-color:${css(SERIES[0])}` }), "Sailed (planner)"),
    el("span", {}, el("i", { style: `border-color:${css(SERIES[1])}` }), "Naive (shortest, left on day 1)"),
    el("span", {}, el("i", { class: "box", style: `background:${css("--berg")}` }), "USNIC-reported berg"));
}

$("#hreplay-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const status = $("#hreplay-status");
  $("#hreplay-btn").disabled = true;
  try {
    const res = await runJob("/real/historical/replay",
      { start: f.get("start"), max_wait_days: Number(f.get("max_wait_days")) }, status,
      "Replaying day by day (each day in port plans a 14-day window)");
    $("#hreplay-result").hidden = false;
    const cands = [];
    if (res.sailed_xy_km) cands.push({ xy_km: res.sailed_xy_km, labels: ["sailed (planner)"], feasible: true });
    if (res.naive_xy_km) cands.push({ xy_km: res.naive_xy_km, labels: ["naive"], feasible: true });
    hReplayState.plan = { map: res.map, candidates: cands, recommended_index: cands.length ? 0 : null,
      origin_xy_km: res.origin_xy_km, destination_xy_km: res.destination_xy_km, land_label: res.land_label };
    hReplayState.bergs = res.observed_bergs;
    $("#hreplay-map-title").textContent = `Observed sea ice on ${res.map.observed_date}`;
    drawReplay();
    $("#hreplay-verdict").textContent = res.departure ?
      `Departed ${res.departure} after waiting ${res.days_waited} day(s); ` +
      (res.arrived ? `arrived ${res.arrival_day}.` : "did not arrive within the replay.") :
      "No departure met the risk budget within the wait limit.";
    const t = res.truth;
    $("#hreplay-table tbody").replaceChildren(...(t ? [
      ["Left on", t.planner.departure, t.naive.departure],
      ["Hours (observed ice)", num(t.planner.hours, 1), num(t.naive.hours, 1)],
      ["Fuel index", num(t.planner.fuel_index), num(t.naive.fuel_index)],
      ["Distance (km)", num(t.planner.distance_km), num(t.naive.distance_km)],
      ["Cells in observed ice ≥ limit", t.planner.observed_breach_cells, t.naive.observed_breach_cells],
      ["Hours in observed ice ≥ limit", num(t.planner.hazard_hours, 1), num(t.naive.hazard_hours, 1)],
      ["Cells in a reported berg's footprint", t.planner.berg_footprint_cells, t.naive.berg_footprint_cells],
      ["Nearest reported berg (km)", `${num(t.planner.berg_min_distance_km)} (${t.planner.berg_nearest || "–"})`,
        `${num(t.naive.berg_min_distance_km)} (${t.naive.berg_nearest || "–"})`],
    ] : []).map(([k, a, b]) => el("tr", {}, el("td", {}, k), el("td", { class: "num" }, a), el("td", { class: "num" }, b))));
    renderLog("#hreplay-log", res.days.map((d) => ({ ...d, event: d.phase })));
    const s = seasonText(res.historical.season);
    status.textContent = `${res.data_label} · icebergs included · ${s.text}`;
  } catch (e) {
    status.textContent = "Error: " + errText(e);
  } finally {
    $("#hreplay-btn").disabled = false;
  }
});

function redrawHist(tab) {
  if (tab === "hroute") drawHistMap("#hroute-map", hRouteState, "#hroute-map-legend");
  if (tab === "hwindow") {
    if (hWinState.result) drawWindow(hWinState.result, hWinState.result.risk_budget, "#hwindow-chart", hWinState);
    drawHistMap("#hwindow-map", hWinMapState, "#hwindow-map-legend");
  }
  if (tab === "hvoyage") { drawHistMap("#hvoyage-map", hVoyState, "#hvoyage-map-legend"); drawReplay(); }
}

/* ------------------------------------------------------------- boot */
(async () => {
  try {
    const h = await api("/health");
    $("#disclaimer").textContent = h.disclaimer;
    $("#version-badge").textContent = "v" + h.version;
    const c = await api("/config");
    window.__config = c.config;
    window.__status = await api("/status");
  } catch (e) {
    $("#disclaimer").textContent = "API unavailable: " + e.message;
    showRealUnavailable("API unavailable: " + e.message);
  }
  setModeBadge(currentTab());
  setupHistorical(window.__status ? window.__status.historical : null);
  if (window.__status) {
    try { await loadReal(window.__status); } catch (e) { showRealUnavailable("Could not load real data: " + e.message); }
    if (window.__status.historical && window.__status.historical.status === "available") {
      try { await loadHistoricalDates(); } catch (e) {
        setupHistorical({ status: "failed", reason: "could not load the available dates: " + errText(e) });
      }
    }
  }
  window.addEventListener("resize", () => {
    if (mapState.plan && !$("#tab-plan").hidden) drawMap(mapState.plan);
    if (winState.result && !$("#tab-window").hidden) drawWindow(winState.result, budget());
    if (!$("#tab-real").hidden) redrawReal();
    const tab = currentTab();
    if (isHist(tab)) redrawHist(tab);
  });
})();
