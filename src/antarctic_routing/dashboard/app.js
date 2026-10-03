/* Antarctic Ice-Risk Routing dashboard - no external dependencies. */
"use strict";

const $ = (sel) => document.querySelector(sel);
const css = (name) => getComputedStyle(document.querySelector(".viz-root")).getPropertyValue(name).trim();
const SERIES = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5", "--series-6", "--series-7", "--series-8"];
const ICE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"];
const pct = (v, d = 1) => (v == null || !isFinite(v) ? "–" : (100 * v).toFixed(d) + "%");
const num = (v, d = 0) => (v == null || !isFinite(v) ? "–" : Number(v).toFixed(d));

async function api(path, opts = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...opts });
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

/* ------------------------------------------------------------- tabs */
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
  document.querySelectorAll(".tab").forEach((s) => { s.hidden = s.id !== "tab-" + b.dataset.tab; });
  if (b.dataset.tab === "validation") loadFigures();
}));

/* ------------------------------------------------------------- map */
const mapState = { plan: null, toScreen: null, screenRoutes: [] };

function drawMap(plan) {
  const canvas = $("#map");
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
  mapState.toScreen = toScreen;

  ctx.fillStyle = css("--surface-1");
  ctx.fillRect(0, 0, w, h);

  // raster: one pixel per grid cell, row 0 of the grid is the bottom of the image
  const img = new ImageData(m.nx, m.ny);
  const land = hexToRgb(css("--land")), berg = hexToRgb(css("--berg")), surf = hexToRgb(css("--surface-1"));
  for (let j = 0; j < m.ny; j++) {
    for (let i = 0; i < m.nx; i++) {
      const k = j * m.nx + i, o = ((m.ny - 1 - j) * m.nx + i) * 4;
      let rgb = surf, a = 255;
      if (m.land[k]) rgb = land;
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
  mapState.screenRoutes = [];
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
    mapState.screenRoutes.push({ i, pts });
  }
  marker(ctx, toScreen(...plan.origin_xy_km), css("--series-3"), "Origin");
  marker(ctx, toScreen(...plan.destination_xy_km), css("--series-4"), "Destination");

  const legend = $("#map-legend");
  legend.replaceChildren(
    el("span", {}, el("i", { class: "box", style: `background:${css("--land")}` }), "Land (schematic)"),
    ...(m.p_berg ? [el("span", {}, el("i", { class: "box", style: `background:${css("--berg")}` }), "Iceberg presence")] : []),
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

$("#map").addEventListener("pointermove", (ev) => {
  const tip = $("#map-tip");
  if (!mapState.plan) return;
  const rect = ev.target.getBoundingClientRect();
  const px = ev.clientX - rect.left, py = ev.clientY - rect.top;
  let best = null;
  for (const r of mapState.screenRoutes) {
    for (let k = 1; k < r.pts.length; k++) {
      const d = distToSegment(px, py, r.pts[k - 1], r.pts[k]);
      if (d < 10 && (!best || d < best.d)) best = { d, i: r.i };
    }
  }
  if (!best) { tip.hidden = true; return; }
  const cand = mapState.plan.candidates[best.i];
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
$("#map").addEventListener("pointerleave", () => { $("#map-tip").hidden = true; });

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

function drawWindow(result, budget) {
  const canvas = $("#window-chart");
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
  winState.bars = [];
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
    winState.bars.push({ x, w: bwid, o });
  });
  ctx.strokeStyle = css("--text-primary"); ctx.setLineDash([5, 4]);
  ctx.beginPath(); ctx.moveTo(left, y(budget)); ctx.lineTo(w - right, y(budget)); ctx.stroke(); ctx.setLineDash([]);
  const label = `risk budget ${pct(budget, 0)}`;
  ctx.fillStyle = css("--surface-1");
  ctx.fillRect(left + 2, y(budget) - 17, ctx.measureText(label).width + 6, 14);
  ctx.fillStyle = css("--text-primary"); ctx.fillText(label, left + 5, y(budget) - 6);
  if (result.selected) {
    const b = winState.bars.find((q) => q.o.departure === result.selected);
    if (b) {
      ctx.fillStyle = css("--text-primary");
      ctx.fillText("▼ selected", b.x + b.w / 2 - 26, Math.max(top + 10, y(Math.min(1, b.o.p_breach_upper)) - 18));
    }
  }
}

$("#window-chart").addEventListener("pointermove", (ev) => {
  const tip = $("#window-tip");
  const rect = ev.target.getBoundingClientRect();
  const px = ev.clientX - rect.left;
  const b = winState.bars.find((q) => px >= q.x - 4 && px <= q.x + q.w + 4);
  if (!b) { tip.hidden = true; return; }
  tip.replaceChildren(el("strong", {}, b.o.departure), row("95% upper bound", pct(b.o.p_breach_upper)),
    row("P(breach)", pct(b.o.p_breach)), row("Expected time", num(b.o.expected_hours, 1) + " h"),
    row("Fuel index", num(b.o.expected_fuel)), row("Status", b.o.feasible ? "✓ meets budget" : "✕ exceeds budget"));
  tip.style.left = Math.min(px + 14, rect.width - 200) + "px"; tip.style.top = "60px"; tip.hidden = false;
});
$("#window-chart").addEventListener("pointerleave", () => { $("#window-tip").hidden = true; });

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
    const budget = window.__config ? window.__config.routing.risk_budget : 0.05;
    drawWindow(job.result, budget);
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
      const a = $(id); a.href = `/voyages/${voyage.id}/export?format=${fmt}`; a.hidden = false;
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
    box.replaceChildren(...figs.figures.map((fig) => el("figure", {},
      el("img", { src: fig.url, alt: fig.caption, loading: "lazy" }), el("figcaption", {}, fig.caption))));
    if (!figs.figures.length) box.replaceChildren(el("p", { class: "sub" }, "No figures yet - run the CLI reports."));
    box.dataset.loaded = "1";
  } catch (e) { box.replaceChildren(el("p", {}, "Could not load figures: " + e.message)); }
}

/* ------------------------------------------------------------- boot */
(async () => {
  try {
    const h = await api("/health");
    $("#disclaimer").textContent = h.disclaimer;
    $("#version-badge").textContent = "v" + h.version;
    const c = await api("/config");
    window.__config = c.config;
  } catch (e) { $("#disclaimer").textContent = "API unavailable: " + e.message; }
  window.addEventListener("resize", () => {
    if (mapState.plan) drawMap(mapState.plan);
    if (winState.result) drawWindow(winState.result, window.__config ? window.__config.routing.risk_budget : 0.05);
  });
})();
