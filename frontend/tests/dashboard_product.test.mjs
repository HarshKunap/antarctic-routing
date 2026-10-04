// Browser tests for the dashboard's Plan Route product view, against a mocked API.
//
//   node --test frontend/tests/            (needs Playwright with Chromium; skipped when it is not installed)
//
// Every request the page makes is answered here: the dashboard files from src/antarctic_routing/dashboard and the
// API from small stand-in payloads shaped like GET /real/locations, GET /real/historical/dates and POST /real/plan.
// The stand-in plan is a 6 x 5 cell test grid, not real data; tests/test_historical_real.py and the S3 smoke run
// cover the real archive. Set ANTROUTE_SMOKE_URL=http://host:port to also run one real Plan Route end to end.
import { test } from "node:test";
import assert from "node:assert/strict";
import { execSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const DASH = resolve(here, "..", "..", "src", "antarctic_routing", "dashboard");
const ORIGIN = "http://dashboard.test";

function loadPlaywright() {
  const require = createRequire(import.meta.url);
  try { return require("playwright"); } catch { /* fall through to a global install */ }
  try {
    const root = execSync("npm root -g", { stdio: ["ignore", "pipe", "ignore"] }).toString().trim();
    return require(join(root, "playwright"));
  } catch { return null; }
}
const pw = loadPlaywright();
const skip = pw ? false : "Playwright is not installed";

/* ------------------------------------------------------------- stand-in API payloads */
const DISCLOSURE = "Hindsight forcing: test disclosure text for the stand-in plan.";
const PRESETS = [
  { id: "alpha", name: "Alpha Point", region: "North", note: "", requested: { lat: -60, lon: -60 }, available: true,
    resolved: { lat: -60.01, lon: -60.02, row: 1, col: 1 }, snapped: false, distance_km: 2.0, reason: "in a navigable cell" },
  { id: "bravo", name: "Bravo Bay", region: "North", note: "", requested: { lat: -61, lon: -59 }, available: true,
    resolved: { lat: -61.0, lon: -59.0, row: 3, col: 4 }, snapped: false, distance_km: 1.0, reason: "in a navigable cell" },
  { id: "charlie", name: "Charlie Station", region: "South", note: "", requested: { lat: -62, lon: -61 }, available: true,
    resolved: { lat: -61.9, lon: -60.7 , row: 2, col: 3 }, snapped: true, distance_km: 12.4,
    reason: "the requested point lies on land in the grid's land mask; moved to the nearest navigable cell (12 km away)" },
];
const DATES = (() => {
  const out = [];
  for (let d = new Date("2023-11-01T00:00:00Z"); d <= new Date("2023-12-20T00:00:00Z"); d.setUTCDate(d.getUTCDate() + 1)) {
    out.push(d.toISOString().slice(0, 10));
  }
  return out;
})();
const datesPayload = {
  label: "Real Historical Data", horizon_days: 6, window_days_max: 14, route_dates: DATES, window_dates: DATES,
  seasons: { route: [{ season: "2023-24", first: DATES[0], last: DATES.at(-1), n_dates: DATES.length, out_of_sample: true, in_sample_notes: [] }],
    window: [{ season: "2023-24", first: DATES[0], last: DATES.at(-1), n_dates: DATES.length, out_of_sample: true, in_sample_notes: [] }] },
};

const NX = 6, NY = 5;
const grid = { crs: "EPSG:3031", nx: NX, ny: NY, res_km: 25, x0_km: 0, y0_km: 0, rotation_rad: 0,
  x_km: [...Array(NX).keys()].map((i) => 12.5 + 25 * i), y_km: [...Array(NY).keys()].map((j) => 12.5 + 25 * j) };
const land = Array(NX * NY).fill(0); land[NX * NY - 1] = 1;
const layer = (k, ice) => ({ scenario_layer: k, date: `2023-11-${14 + k}`, source: k ? "forecast" : "observed",
  p_ice_ge_limit_pct: land.map((l, i) => (l ? null : (i * ice) % 100)), p_berg_pct: land.map((l, i) => (l ? null : i === 8 ? 40 : 0)) });

function riskPart(breaches, upper) {
  return { breaches, n_scenarios: 200, p_breach: breaches / 200, p_breach_upper: upper, segment_breach_prob: [0, 0, 0, 0],
    impassable_scenarios: 0 };
}
function option(day, feasible, upper, hours) {
  return { departure: `2023-11-${day}`, status: feasible ? "feasible" : "exceeds_budget", feasible, expected_hours: hours,
    expected_fuel: 100 + day, p_breach: upper / 2, p_breach_upper: upper, beyond_horizon_fraction: 0, route_labels: [],
    lead_days: day - 14, forecast_fraction: 1, support: "forecast-supported", within_trust_horizon: null };
}

function makePlan(status = "recommended") {
  const rec = status === "recommended";
  const loc = (p) => ({ id: p.id, name: p.name, cell: [p.resolved.row, p.resolved.col], snapped: p.snapped,
    distance_km: p.distance_km, reason: p.reason, requested: p.requested, resolved: p.resolved });
  return {
    status,
    explanation: rec ? "Stand-in explanation: selected 2023-11-14." : "Stand-in explanation: no departure date satisfies the budget.",
    metadata: { mode: "historical", label: "Real Historical Data", execution_mode: "real", data_status: "historical",
      issue_date: "2023-11-14", hindsight_forcing: true, hindsight_disclosure: DISCLOSURE,
      banners: ["HISTORICAL MODE", "ERA5/CMEMS hindsight forcing", "Research estimate, not certified navigation"],
      disclaimer: "Stand-in disclaimer.", window_days: 14, horizon_days: 6, scenario_days: 20, n_scenarios: 200,
      layer_source: ["observed", "forecast"],
      provenance: { limitations: ["Stand-in limitation one."], season: datesPayload.seasons.window[0] } },
    locations: { origin: loc(PRESETS[0]), destination: loc(PRESETS[1]) },
    horizon: { great_circle_km: 120, planning_hours: 20, required_days: 6, horizon_days: 6, lead_days: 21, supported: true },
    route: { recommended: rec, departure_date: rec ? "2023-11-14" : "2023-11-16", departure_utc: rec ? "2023-11-14T00:00:00Z" : "2023-11-16T00:00:00Z",
      time_resolution: "24 h forecast layers", lead_days: rec ? 0 : 2, expected_hours: 30.25, hours_p10: 29.5, hours_p90: 31.0,
      eta_utc: rec ? "2023-11-15T06:15Z" : "2023-11-17T06:15Z", distance_km: 123.4,
      fuel_index: { expected: 456.7, p10: 450, p90: 460, unit: "relative fuel index" }, labels: [], tags: [],
      forecast_fraction: 1, support: "forecast-supported", beyond_horizon_fraction: 0, impassable_scenarios: 0,
      cells: [[1, 1], [1, 2], [2, 3], [3, 4]], latlon: [[-60, -60], [-60.3, -59.8], [-60.6, -59.5], [-61, -59]],
      xy_km: [[37.5, 37.5], [62.5, 37.5], [87.5, 62.5], [112.5, 87.5]] },
    risk: { authoritative: "combined", risk_budget: 0.05, confidence: 0.95, estimator: "wilson_upper",
      combined: { ...riskPart(rec ? 2 : 150, rec ? 0.0311 : 0.7911), within_budget: rec },
      sea_ice: riskPart(rec ? 1 : 140, rec ? 0.0222 : 0.7422), iceberg: riskPart(rec ? 1 : 10, rec ? 0.0133 : 0.0833),
      iceberg_only_increment: riskPart(1, 0.02),
      definitions: { combined: "Combined definition.", sea_ice: "Sea-ice definition.", iceberg: "Iceberg definition." } },
    departure: { recommended: rec ? "2023-11-14" : null, rule: "stand-in rule", explanation: "x",
      options: [option(14, rec, rec ? 0.0311 : 0.81, 30.25), option(15, false, 0.2, 31), option(16, false, rec ? 0.3 : 0.7911, 30.25)],
      depart_on_issue_date: option(14, rec, 0.0311, 30.25) },
    alternatives: [],
    daily: [
      { date: "2023-11-14", day_of_voyage: 1, scenario_layer: 0, layer_source: "observed", nominal_hours_since_departure: [0, 24],
        route_cell_index: [0, 2], position_end_of_day: [-60.6, -59.5], distance_km: 90.1,
        sea_ice: { mean_concentration: 0.01, max_concentration: 0.02, max_p_ge_vessel_limit: 0 }, iceberg: { max_p_presence: 0 },
        risk: { combined: { max_cell_breach_prob: 0 }, sea_ice: { max_cell_breach_prob: 0 }, iceberg: { max_cell_breach_prob: 0 } } },
      { date: "2023-11-15", day_of_voyage: 2, scenario_layer: 1, layer_source: "forecast", nominal_hours_since_departure: [24, 30.25],
        route_cell_index: [3, 3], position_end_of_day: [-61, -59], distance_km: 33.3,
        sea_ice: { mean_concentration: 0.2, max_concentration: 0.3, max_p_ge_vessel_limit: 0.07 }, iceberg: { max_p_presence: 0.4 },
        risk: { combined: { max_cell_breach_prob: 0.0777 }, sea_ice: { max_cell_breach_prob: 0.07 }, iceberg: { max_cell_breach_prob: 0.01 } } },
    ],
    daily_note: "Stand-in nominal timeline note.",
    layers: { grid, land, vessel_limit: 0.15, layers: [layer(0, 3), layer(1, 7)] },
    icebergs: { list_date: "2023-11-09", drifted: ["T1"] },
    iceberg_tracks: [{ id: "T1", daily_mean: [{ layer: 0, x_km: 62.5, y_km: 112.5, lat: -60.5, lon: -60.5 },
      { layer: 1, x_km: 87.5, y_km: 112.5, lat: -60.6, lon: -60.4 }] }],
  };
}

/* ------------------------------------------------------------- stand-in simulation */
// Shaped like POST /real/simulate's job result; the "replan" variant switches route on day 2.
function makeSim(replan = true) {
  const plan = makePlan();
  const g = { ...grid, land };
  const planned = plan.route.xy_km, cells = plan.route.cells;
  const alt = { cells: [[2, 3], [2, 4], [3, 4]], xy_km: [[87.5, 62.5], [112.5, 62.5], [112.5, 87.5]] };
  const pos = (k, xy, c) => ({ lat: -60 - k / 2, lon: -60 + k / 2, row: c[0], col: c[1], xy_km: xy });
  const risk = (u) => ({ combined: { breaches: 1, n_scenarios: 200, p_breach: 0.005, p_breach_upper: u, within_budget: u <= 0.05 },
    sea_ice: { p_breach_upper: u }, iceberg: { p_breach_upper: 0.0188 }, risk_budget: 0.05 });
  const frame = (k, phase, ts, xy, cell, track, route, version, extra = {}) => ({
    index: k, day_of_voyage: k, phase, timestamp_utc: ts, date: ts.slice(0, 10), position: pos(k, xy, cell),
    progress: { sailed_km: 40 * k, remaining_km: phase === "arrived" ? 0 : 120 - 40 * k, fraction: phase === "arrived" ? 1 : k / 3,
      sailed_hours: 20 * k, cells_sailed: k },
    segment: { cells: [], xy_km: [], latlon: [], sailed_on: null }, track,
    route: { ...route, version }, route_version: version,
    forecast: phase === "arrived" ? null : { issued: ts.slice(0, 10), route_ahead: { expected_hours: 30 - 10 * k,
      expected_fuel: 400 - 100 * k, distance_km: 120 - 40 * k, p_breach_upper: 0.0311 }, risk: risk(k === 1 && replan ? 0.081 : 0.0311),
      eta_utc: "2023-11-15T06:15Z" },
    observed: { sea_ice_on_segment: k ? { date: ts.slice(0, 10), mean_concentration: 0.01, max_concentration: 0.02 } : null,
      icebergs: { list_date: "2023-11-09", age_days: 5, in_grid: [{ id: "T1", lat: -60.5, lon: -60.5, xy_km: [62.5, 112.5] }],
        nearest_km: 55.5, nearest_id: "T1" } },
    map: { date: ts.slice(0, 10), source: "observed", concentration_pct: land.map((l, i) => (l ? null : (i * (k + 2)) % 100)) },
    decision: null, replanned: false, ...extra,
  });
  const keep = { action: "keep", triggers: [], explanation: "Keep current route: stand-in keep.", alert: false, deviation_km: 0, reasons: [] };
  const sw = { action: "switch", triggers: ["previous_route_exceeds_budget"], alert: true, deviation_km: 31.2,
    explanation: "Switch route (previous_route_exceeds_budget): stand-in switch.",
    reasons: ["The route ahead no longer met the risk budget under the new forecast."] };
  const frames = replan ? [
    frame(0, "departure", "2023-11-14T00:00Z", planned[0], cells[0], [planned[0]], { cells, xy_km: planned }, 0),
    frame(1, "at_sea", "2023-11-15T00:00Z", planned[2], cells[2], planned.slice(0, 3), { ...alt }, 1,
      { decision: sw, replanned: true }),
    frame(2, "at_sea", "2023-11-16T00:00Z", alt.xy_km[1], alt.cells[1], [...planned.slice(0, 3), alt.xy_km[1]],
      { cells: alt.cells.slice(1), xy_km: alt.xy_km.slice(1) }, 1, { decision: keep }),
    frame(3, "arrived", "2023-11-16T09:00Z", alt.xy_km[2], alt.cells[2], [...planned.slice(0, 3), ...alt.xy_km.slice(1)],
      { cells: [alt.cells[2]], xy_km: [alt.xy_km[2]] }, 1),
  ] : [
    frame(0, "departure", "2023-11-14T00:00Z", planned[0], cells[0], [planned[0]], { cells, xy_km: planned }, 0),
    frame(1, "at_sea", "2023-11-15T00:00Z", planned[2], cells[2], planned.slice(0, 3), { cells: cells.slice(2), xy_km: planned.slice(2) }, 0,
      { decision: keep }),
    frame(2, "arrived", "2023-11-15T06:15Z", planned[3], cells[3], planned, { cells: [cells[3]], xy_km: [planned[3]] }, 0),
  ];
  const last = frames.at(-1);
  const events = [{ type: "departed", frame: 0, timestamp_utc: frames[0].timestamp_utc, explanation: "Departed on the planned route." }];
  if (replan) {
    events.push({ type: "replan", frame: 1, timestamp_utc: frames[1].timestamp_utc, ...sw, issued: "2023-11-15",
      old_route: { cells: cells.slice(2), xy_km: planned.slice(2), p_breach_upper: 0.081, expected_hours: 10.5, expected_fuel: 210, distance_km: 50 },
      new_route: { ...alt, p_breach_upper: 0.0311, expected_hours: 12.0, expected_fuel: 230, distance_km: 61 },
      change: { p_breach_upper: -0.0499, expected_hours: 1.5, expected_fuel: 20, distance_km: 11 } });
  }
  events.push({ type: "arrived", frame: last.index, timestamp_utc: last.timestamp_utc, explanation: "Destination reached." });
  return {
    status: "simulated", label: "Real Historical Data",
    metadata: { ...plan.metadata, departure_date: "2023-11-14", simulation_note: "Stand-in simulation note." },
    locations: plan.locations, plan: { status: "recommended", explanation: "x", route: plan.route, risk: plan.risk },
    grid: g, frames, events, sailed_track: { cells: [], xy_km: last.track, latlon: [] },
    summary: { status: "arrived", reason: null, arrived: true, departure_utc: "2023-11-14T00:00Z", arrival_utc: last.timestamp_utc,
      days_at_sea: frames.length - 1, simulated_hours: replan ? 57 : 30.25, held_hours: 1.5, replans: replan ? 1 : 0,
      replan_note: replan ? null : "No replan was required during this voyage.", no_feasible_route_alerts: 0,
      sailed: { hours_through_observed_ice: replan ? 55.5 : 30.2, distance_km: replan ? 131 : 123.4, fuel_index: replan ? 470 : 456.7,
        observed_breach_cells: 0, hazard_hours: 0, berg_footprint_cells: 0, berg_min_distance_km: 55.5, berg_nearest: "T1" },
      planned: { distance_km: 123.4, expected_fuel: 456.7, expected_hours: 30.25, p_breach_upper: 0.0311 },
      final_forecast_risk: risk(0.0311), final_forecast_issued: replan ? "2023-11-16" : "2023-11-15",
      notes: { sailed: "Stand-in sailed note." } },
  };
}

/* ------------------------------------------------------------- harness */
const TYPES = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };

async function openDashboard(browser, { plan = () => ({ status: 200, body: makePlan() }), historical = "available",
  simulate = () => ({ status: 200, body: makeSim(true) }), dates = datesPayload } = {}) {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const calls = [];
  const errors = [];
  const jobs = new Map();
  page.on("pageerror", (e) => errors.push(e.message));
  await page.route("**/*", async (route) => {
    const req = route.request(), url = new URL(req.url());
    if (url.origin !== ORIGIN) return route.abort();
    const json = (body, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    const p = url.pathname;
    if (p === "/" || p.startsWith("/static/")) {
      const file = p === "/" ? "index.html" : p.slice("/static/".length);
      return route.fulfill({ status: 200, contentType: TYPES[file.slice(file.lastIndexOf("."))], body: readFileSync(join(DASH, file)) });
    }
    calls.push({ method: req.method(), path: p, search: url.search, body: req.postData() });
    if (p === "/health") return json({ version: "test", disclaimer: "Test disclaimer." });
    if (p === "/config") return json({ config: { routing: { risk_budget: 0.05 } } });
    if (p === "/status") {
      return json({ real_data: { status: "unavailable", reason: "not used in this test" },
        historical: historical === "available" ? { status: "available", hindsight_forcing: DISCLOSURE }
          : { status: "unavailable", reason: "no archive configured" },
        figures: { execution_mode: "controlled_synthetic" } });
    }
    if (p === "/real/locations") return json({ label: "Real Historical Data", presets: PRESETS, max_snap_km: 60 });
    if (p === "/real/historical/dates") return json(dates);
    if (p === "/real/plan") {
      const r = await plan(JSON.parse(req.postData() || "{}"));
      if (r.abort) return route.abort("failed");
      if (r.delay) await new Promise((ok) => setTimeout(ok, r.delay));
      return route.fulfill({ status: r.status, contentType: "application/json", body: r.raw ?? JSON.stringify(r.body) });
    }
    if (p === "/real/simulate") {               // a background job: queued here, done on the first poll
      const r = await simulate(JSON.parse(req.postData() || "{}"));
      if (r.status !== 200) return json(r.body, r.status);
      jobs.set("sim1", r);
      return json({ job_id: "sim1", kind: "real_simulate", status: "queued", result: null, error: null });
    }
    if (p.startsWith("/jobs/")) {
      const r = jobs.get(p.slice(6));
      if (r.delay) await new Promise((ok) => setTimeout(ok, r.delay));
      return json(r.failed ? { job_id: "sim1", status: "failed", error: r.failed, result: null }
        : { job_id: "sim1", status: "done", result: r.body, error: null });
    }
    return json({ detail: "not mocked" }, 404);
  });
  await page.goto(ORIGIN + "/");
  return { page, calls, errors, planCalls: () => calls.filter((c) => c.path === "/real/plan") };
}

const ready = (page) => page.waitForFunction(() => !document.querySelector("#pr-submit").disabled);
const planned = (page) => page.waitForSelector("#pr-result:not([hidden])");
const text = (page, sel) => page.locator(sel).innerText();

let browser;
test.before(async () => { if (pw) browser = await pw.chromium.launch(); });
test.after(async () => { if (browser) await browser.close(); });

/* ------------------------------------------------------------- tests */
test("Plan Route is the default view and shows a clean landing, no precomputed result", { skip }, async () => {
  const { page, errors, planCalls } = await openDashboard(browser);
  await ready(page);
  assert.equal(await page.locator('.tabs button[aria-selected="true"]').getAttribute("data-tab"), "product");
  assert.ok(await page.locator("#pr-landing").isVisible());
  assert.ok(!(await page.locator("#pr-result").isVisible()));
  assert.match(await text(page, "#pr-landing"), /Origin[\s\S]*Destination[\s\S]*Date[\s\S]*Plan Route/);
  assert.equal(planCalls().length, 0);
  assert.match(await text(page, ".tabs"), /Research & legacy views/i);
  assert.match(await text(page, ".tabs"), /Synthetic sandbox \(demo only\)/i);
  assert.deepEqual(errors, []);
  await page.close();
});

test("locations come from GET /real/locations and dates from the route's supported list", { skip }, async () => {
  const { page, calls } = await openDashboard(browser);
  await ready(page);
  const ids = await page.locator("#pr-origin option").evaluateAll((os) => os.map((o) => o.value));
  assert.deepEqual(ids, PRESETS.map((p) => p.id));
  assert.equal(await page.inputValue("#pr-origin"), "alpha");
  assert.equal(await page.inputValue("#pr-destination"), "bravo");
  const d = calls.find((c) => c.path === "/real/historical/dates" && c.search);
  assert.equal(d.search, "?origin=alpha&destination=bravo");
  assert.equal(await page.getAttribute("#pr-issue", "min"), DATES[0]);
  assert.equal(await page.getAttribute("#pr-issue", "max"), DATES.at(-1));
  assert.equal(await page.inputValue("#pr-issue"), "2023-11-14");
  await page.close();
});

test("a snapped location shows a non-blocking note with requested and routing coordinates", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await ready(page);
  await page.selectOption("#pr-destination", "charlie");
  await ready(page);
  const notes = await text(page, "#pr-notes");
  assert.match(notes, /Destination snapped 12\.4 km to the nearest open-water grid cell/);
  assert.match(notes, /requested -62\.000°, -61\.000°, routing from -61\.900°, -60\.700°/);
  assert.equal(await page.isDisabled("#pr-submit"), false);
  await page.close();
});

test("unsupported dates and identical ends block Plan Route with a reason", { skip }, async () => {
  const { page, planCalls } = await openDashboard(browser);
  await ready(page);
  await page.fill("#pr-issue", "2023-06-01");
  assert.equal(await page.isDisabled("#pr-submit"), true);
  assert.match(await text(page, "#pr-notes"), /No Real Historical Data for 2023-06-01/);
  await page.fill("#pr-issue", "2023-11-14");
  await page.selectOption("#pr-destination", "alpha");
  assert.equal(await page.isDisabled("#pr-submit"), true);
  assert.match(await text(page, "#pr-notes"), /same place/);
  assert.equal(planCalls().length, 0);
  await page.close();
});

test("Plan Route sends exactly one POST /real/plan with the chosen inputs and shows a loading state", { skip }, async () => {
  const { page, planCalls } = await openDashboard(browser, { plan: () => ({ status: 200, body: makePlan(), delay: 1200 }) });
  await ready(page);
  await page.selectOption("#pr-destination", "charlie");
  await ready(page);
  await page.fill("#pr-issue", "2023-11-20");
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-loading:not([hidden])");
  const loading = await text(page, "#pr-loading");
  assert.match(loading, /Running the real historical model/);
  assert.doesNotMatch(loading, /\d+\s*%/);                 // elapsed seconds only, no fake progress percentage
  assert.equal(await page.isDisabled("#pr-submit"), true);
  await planned(page);
  const calls = planCalls();
  assert.equal(calls.length, 1);
  assert.equal(calls[0].method, "POST");
  assert.deepEqual(JSON.parse(calls[0].body),
    { origin: { preset: "alpha" }, destination: { preset: "charlie" }, issue: "2023-11-20" });
  assert.ok(!(await page.locator("#pr-loading").isVisible()));
  await page.close();
});

test("a recommended result renders the verdict, metrics and three separate risk cards", { skip }, async () => {
  const { page, errors } = await openDashboard(browser);
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  assert.match(await text(page, "#pr-verdict-badge"), /Recommended/);
  assert.match(await text(page, "#pr-verdict-title"), /within the 5% risk budget/);
  const metrics = await text(page, "#pr-metrics");
  for (const s of ["14 Nov 2023, 00:00 UTC", "15 Nov 2023, 06:15 UTC", "30.3 h", "123 km", "457"]) assert.ok(metrics.includes(s), s);
  assert.match(await text(page, '[data-risk="combined"]'), /within budget[\s\S]*3\.1%[\s\S]*2 of 200 scenarios/);
  assert.match(await text(page, '[data-risk="sea_ice"]'), /2\.2%[\s\S]*1 of 200 scenarios/);
  assert.match(await text(page, '[data-risk="iceberg"]'), /1\.3%[\s\S]*1 of 200 scenarios/);
  assert.match(await text(page, '[data-risk="combined"]'), /95% upper bound \(Wilson\)/);
  assert.doesNotMatch(await text(page, "#pr-result"), /catastroph|probability of (sinking|loss)/i);
  assert.deepEqual(errors, []);
  await page.close();
});

test("no recommended departure: says so and shows the least-risky option as not recommended", { skip }, async () => {
  const { page } = await openDashboard(browser, { plan: () => ({ status: 200, body: makePlan("no_feasible_departure") }) });
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  assert.match(await text(page, "#pr-verdict-badge"), /Not recommended/);
  assert.equal(await text(page, "#pr-verdict-title"), "No departure in the selected window meets the 5% risk budget.");
  assert.match(await text(page, "#pr-verdict-text"), /least-risky option for reference only: depart 2023-11-16/);
  assert.match(await text(page, "#pr-metrics"), /Least-risky departure/i);
  assert.match(await text(page, '[data-risk="combined"]'), /exceeds budget[\s\S]*79\.1%/);
  assert.match(await text(page, "#pr-options tbody tr.rec"), /◆ 2023-11-16/);
  await page.close();
});

test("the map draws the returned layers and route, and the day strip changes the day without a new request", { skip }, async () => {
  const { page, planCalls } = await openDashboard(browser);
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  const plan = makePlan();
  const drawn = await page.evaluate(() => ({ ice: prMapState.plan.map.p_ice, land: prMapState.plan.map.land,
    route: prMapState.plan.candidates[0].xy_km, origin: prMapState.plan.origin_xy_km, k: prMapState.k }));
  assert.deepEqual(drawn.ice, plan.layers.layers[0].p_ice_ge_limit_pct);
  assert.deepEqual(drawn.land, plan.layers.land);
  assert.deepEqual(drawn.route, plan.route.xy_km);
  assert.deepEqual(drawn.origin, [grid.x_km[1], grid.y_km[1]]);
  assert.equal(drawn.k, 0);
  const before = await page.locator("#pr-map").screenshot();
  assert.match(await text(page, "#pr-day-title"), /Day 1 of 2/);
  await page.click('#pr-days [data-day="1"]');
  assert.match(await text(page, "#pr-day-title"), /Day 2 of 2/);
  const day = await text(page, "#pr-day-table");
  assert.match(day, /-61\.000°, -59\.000°/);
  assert.match(day, /7\.8%/);                               // max cell breach probability of that day
  assert.match(day, /40\.0%/);                              // max iceberg presence of that day
  assert.equal(await page.evaluate(() => prMapState.k), 1);
  assert.deepEqual(await page.evaluate(() => prMapState.plan.map.p_ice), plan.layers.layers[1].p_ice_ge_limit_pct);
  assert.notDeepEqual(await page.locator("#pr-map").screenshot(), before);
  await page.keyboard.press("ArrowLeft");                   // focus is on the strip
  assert.match(await text(page, "#pr-day-title"), /Day 1 of 2/);
  assert.equal(planCalls().length, 1);
  await page.close();
});

test("departure options come straight from the response, with the shown one marked", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  const rows = await page.locator("#pr-options tbody tr").allInnerTexts();
  assert.equal(rows.length, 3);
  assert.match(rows[0], /★ 2023-11-14[\s\S]*meets budget[\s\S]*3\.1%[\s\S]*shown/);
  assert.match(rows[1], /2023-11-15[\s\S]*exceeds budget[\s\S]*20\.0%[\s\S]*\+0\.8 h/);
  assert.match(await text(page, "#pr-window-verdict"), /1 of 3 departure dates meet the 5% budget/);
  await page.close();
});

test("the hindsight disclosure and Data & Confidence panel come from the response", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  assert.ok(await page.locator("#pr-disclosure").isVisible());
  assert.equal(await text(page, "#pr-disclosure"), DISCLOSURE);
  assert.match(await text(page, "#pr-banners"), /HISTORICAL MODE[\s\S]*ERA5\/CMEMS hindsight forcing[\s\S]*Research estimate, not certified navigation/i);
  await page.click('.pr-steps button[data-view="data"]');
  assert.ok(await page.locator("#pr-data").isVisible());
  assert.ok(!(await page.locator("#pr-result").isVisible()));
  const data = await text(page, "#pr-data");
  assert.ok(data.includes(DISCLOSURE));
  assert.match(data, /Stand-in limitation one\./);
  assert.match(data, /Research estimate, not certified navigation/);
  assert.doesNotMatch(data, /confidence[^\n]*\d+\s*%/i);   // no invented confidence percentages
  await page.close();
});

for (const [name, reply, expect] of [
  ["out of coverage (422)", { status: 422, body: { detail: { status: "out_of_coverage", reason: "no USNIC list within 14 days" } } },
    /archive does not cover this date[\s\S]*no USNIC list within 14 days/],
  ["invalid location (422)", { status: 422, body: { detail: { status: "invalid_location", reason: "outside the routing grid" } } },
    /cannot be routed[\s\S]*outside the routing grid/],
  ["model unavailable (503)", { status: 503, body: { detail: { status: "blocked", reason: "model checkpoint missing" } } },
    /unavailable right now[\s\S]*model checkpoint missing[\s\S]*Nothing synthetic/],
  ["malformed response", { status: 200, body: { status: "recommended", metadata: { mode: "historical", execution_mode: "real" } } },
    /could not be read[\s\S]*Malformed/],
  ["non-JSON response", { status: 200, raw: "<html>proxy error</html>" }, /could not be read/],
  ["unlabelled synthetic response", { status: 200, body: { ...makePlan(), metadata: { ...makePlan().metadata, execution_mode: "controlled_synthetic" } } },
    /not labelled as real historical data/],
  ["network failure", { abort: true }, /Could not reach the planning server/],
]) {
  test(`errors are shown in words, inputs kept, nothing substituted: ${name}`, { skip }, async () => {
    const { page, planCalls } = await openDashboard(browser, { plan: () => reply });
    await ready(page);
    await page.selectOption("#pr-destination", "charlie");
    await ready(page);
    await page.fill("#pr-issue", "2023-11-21");
    await page.click("#pr-submit");
    await page.waitForSelector("#pr-error:not([hidden])");
    assert.match(await text(page, "#pr-error"), expect);
    assert.ok(!(await page.locator("#pr-result").isVisible()));
    assert.equal(await page.evaluate(() => prod.result), null);
    assert.equal(await page.inputValue("#pr-destination"), "charlie");
    assert.equal(await page.inputValue("#pr-issue"), "2023-11-21");
    assert.equal(await page.isDisabled("#pr-submit"), false);
    assert.equal(planCalls().length, 1);
    await page.close();
  });
}

test("a slow request can be cancelled, leaving no result", { skip }, async () => {
  const { page } = await openDashboard(browser, { plan: () => ({ status: 200, body: makePlan(), delay: 4000 }) });
  await ready(page);
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-loading:not([hidden])");
  await page.waitForFunction(() => Number(document.querySelector("#pr-elapsed").textContent) >= 1);
  await page.click("#pr-cancel");
  await page.waitForSelector("#pr-error:not([hidden])");
  assert.match(await text(page, "#pr-error"), /cancelled/i);
  assert.ok(!(await page.locator("#pr-result").isVisible()));
  await page.close();
});

test("without the real archive the product says it is unavailable and never plans", { skip }, async () => {
  const { page, planCalls } = await openDashboard(browser, { historical: "unavailable" });
  await page.waitForSelector("#pr-unavailable:not([hidden])");
  assert.match(await text(page, "#pr-unavailable"), /no archive configured[\s\S]*Nothing synthetic/);
  assert.equal(await page.isDisabled("#pr-submit"), true);
  assert.ok(!(await page.locator("#pr-landing").isVisible()));
  assert.equal(planCalls().length, 0);
  await page.close();
});

test("the result fits a phone-width screen without horizontal page scroll", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await page.setViewportSize({ width: 390, height: 844 });
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await page.close();
});

/* ------------------------------------------------------------- Simulate Voyage */
async function simulated(page) {
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  await page.click("#pr-simulate");
  await page.waitForSelector("#sim-body:not([hidden])");
}
const simCalls = (calls) => calls.filter((c) => c.path === "/real/simulate");
const vessel = (page) => page.evaluate(() => sim.data.frames[sim.i].position.xy_km);

test("Simulate Voyage starts from the planned route without planning again", { skip }, async () => {
  const { page, calls, planCalls, errors } = await openDashboard(browser);
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  assert.ok(await page.locator("#pr-simulate").isVisible());
  await page.click("#pr-simulate");
  await page.waitForSelector("#sim-body:not([hidden])");
  assert.equal(planCalls().length, 1);
  const sc = simCalls(calls);
  assert.equal(sc.length, 1);
  assert.deepEqual(JSON.parse(sc[0].body), { origin: { preset: "alpha" }, destination: { preset: "bravo" },
    issue: "2023-11-14", departure: "2023-11-14" });
  assert.equal(await page.locator('.pr-steps button[aria-selected="true"]').getAttribute("data-view"), "sim");
  // back to the result and again: the frames are reused, nothing is recomputed
  await page.click('.pr-steps button[data-view="result"]');
  await page.click('.pr-steps button[data-view="sim"]');
  assert.equal(simCalls(calls).length, 1);
  assert.deepEqual(errors, []);
  await page.close();
});

test("the simulation shows a loading state while the server computes, without fake progress", { skip }, async () => {
  const { page } = await openDashboard(browser, { simulate: () => ({ status: 200, body: makeSim(true), delay: 1500 }) });
  await ready(page);
  await page.click("#pr-submit");
  await planned(page);
  await page.click("#pr-simulate");
  await page.waitForSelector("#sim-loading:not([hidden])");
  const t = await text(page, "#sim-loading");
  assert.match(t, /Simulating the voyage on real historical data/);
  assert.doesNotMatch(t, /\d+\s*%/);
  await page.waitForSelector("#sim-body:not([hidden])");
  assert.ok(!(await page.locator("#sim-loading").isVisible()));
  await page.close();
});

test("next, previous and the slider change the frame and move the vessel", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await simulated(page);
  assert.match(await text(page, "#sim-slider-out"), /2023-11-14 \(0 of 3\)/);
  const p0 = await vessel(page), shot0 = await page.locator("#sim-map").screenshot();
  assert.equal(await page.isDisabled("#sim-prev"), true);
  await page.click("#sim-next");
  assert.match(await text(page, "#sim-slider-out"), /2023-11-15 \(1 of 3\)/);
  assert.notDeepEqual(await vessel(page), p0);
  assert.notDeepEqual(await page.locator("#sim-map").screenshot(), shot0);
  await page.click("#sim-prev");
  assert.match(await text(page, "#sim-slider-out"), /\(0 of 3\)/);
  await page.locator("#sim-slider").fill("2");
  assert.match(await text(page, "#sim-slider-out"), /2023-11-16 \(2 of 3\)/);
  assert.match(await text(page, "#sim-status"), /Route replanned \(1×\)/);
  await page.click('#sim-events [data-frame="0"]');
  assert.match(await text(page, "#sim-status"), /On planned route/);
  await page.close();
});

test("Play advances through the precomputed frames and Pause stops it", { skip }, async () => {
  const { page, calls } = await openDashboard(browser);
  await simulated(page);
  await page.click("#sim-play");
  assert.match(await text(page, "#sim-play"), /Pause/);
  await page.waitForFunction(() => sim.i >= 1, null, { timeout: 5000 });
  await page.click("#sim-play");                                   // pause
  const at = await page.evaluate(() => sim.i);
  await page.waitForTimeout(1800);
  assert.equal(await page.evaluate(() => sim.i), at);
  assert.match(await text(page, "#sim-play"), /Play/);
  await page.click("#sim-play");
  await page.waitForFunction(() => sim.i === sim.data.frames.length - 1, null, { timeout: 8000 });
  await page.waitForFunction(() => sim.timer === null);
  assert.match(await text(page, "#sim-play"), /Replay/);
  assert.equal(simCalls(calls).length, 1);                         // playback never calls the server
  await page.close();
});

test("a real replan event is visible: old vs new route, reason and changes", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await simulated(page);
  const chip = page.locator('#sim-events [data-frame="1"]');
  assert.match(await chip.innerText(), /REPLAN/);
  assert.match(await chip.getAttribute("class"), /replan/);
  const before = await page.evaluate(() => sim.data.frames[0].route.xy_km);
  await chip.click();
  assert.match(await text(page, "#sim-decision-title"), /REPLAN: new route selected/);
  const box = await text(page, "#sim-decision-card");
  assert.match(box, /stand-in switch/);
  assert.match(box, /no longer met the risk budget/);
  assert.match(box, /8\.1%[\s\S]*3\.1%[\s\S]*-5\.0 pp/);
  assert.match(box, /\+1\.5 h/);
  assert.match(box, /\+11 km/);
  assert.match(await text(page, "#sim-map-legend"), /Original planned route[\s\S]*New route ahead \(after replan\)/);
  const after = await page.evaluate(() => sim.data.frames[sim.i].route.xy_km);
  assert.notDeepEqual(after, before);
  assert.match(await text(page, "#sim-subtitle"), /route replanned 1×/);
  await page.close();
});

test("a voyage without a replan says so", { skip }, async () => {
  const { page } = await openDashboard(browser, { simulate: () => ({ status: 200, body: makeSim(false) }) });
  await simulated(page);
  assert.match(await text(page, "#sim-subtitle"), /No replan was required during this voyage\./);
  assert.doesNotMatch(await text(page, "#sim-events"), /REPLAN/);
  await page.click('#sim-events [data-frame="2"]');
  assert.match(await text(page, "#sim-summary"), /No replan was required during this voyage\./);
  await page.close();
});

test("the final summary appears at the last frame with the server's numbers", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await simulated(page);
  assert.ok(!(await page.locator("#sim-summary").isVisible()));
  await page.click('#sim-events [data-frame="3"]');
  assert.ok(await page.locator("#sim-summary").isVisible());
  const s = await text(page, "#sim-summary");
  assert.match(s, /Voyage completed: arrived 16 Nov 2023, 09:00 UTC\. Route replanned 1 time/);
  for (const v of ["57.0 h", "131 km", "470", "3.1%", "Arrived"]) assert.ok(s.includes(v), v);
  assert.match(s, /REPLAN[\s\S]*stand-in switch/);
  await page.close();
});

test("historical disclosure stays visible during the simulation", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await simulated(page);
  assert.match(await text(page, "#sim-banners"), /HISTORICAL MODE[\s\S]*ERA5\/CMEMS hindsight forcing[\s\S]*Research estimate, not certified navigation/i);
  assert.ok(await page.locator("#sim-disclosure").isVisible());
  assert.match(await text(page, "#sim-disclosure"), /not live vessel tracking/);
  assert.equal(await text(page, "#mode-badge"), "REAL HISTORICAL DATA · HINDSIGHT FORCING");
  await page.close();
});

for (const [name, opts, expect] of [
  ["not simulated", { simulate: () => ({ status: 200, body: { status: "not_simulated", reason: "the archive cannot issue the daily forecasts" } }) },
    /cannot be simulated[\s\S]*cannot issue the daily forecasts/],
  ["job failed", { simulate: () => ({ status: 200, body: null, failed: "HTTPException: model missing" }) }, /failed on the server[\s\S]*model missing/],
  ["synthetic-labelled result", { simulate: () => ({ status: 200, body: { ...makeSim(true), metadata: { ...makeSim(true).metadata, execution_mode: "controlled_synthetic" } } }) },
    /not labelled as real historical data/],
  ["frames out of order", { simulate: () => { const s = makeSim(true); s.frames.reverse(); return { status: 200, body: s }; } }, /out of order/],
  ["503 on submit", { simulate: () => ({ status: 503, body: { detail: { status: "blocked", reason: "no archive" } } }) }, /unavailable[\s\S]*no archive/],
]) {
  test(`simulation errors show without playback or substitution: ${name}`, { skip }, async () => {
    const { page } = await openDashboard(browser, opts);
    await ready(page);
    await page.click("#pr-submit");
    await planned(page);
    await page.click("#pr-simulate");
    await page.waitForSelector("#sim-error:not([hidden])");
    assert.match(await text(page, "#sim-error"), expect);
    assert.ok(!(await page.locator("#sim-body").isVisible()));
    assert.equal(await page.evaluate(() => sim.data), null);
    await page.close();
  });
}

test("a new plan clears the previous simulation", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await simulated(page);
  await page.click('.pr-steps button[data-view="plan"]');
  await page.click("#pr-submit");
  await planned(page);
  assert.equal(await page.evaluate(() => sim.data), null);
  await page.close();
});

test("the simulation fits a phone-width screen", { skip }, async () => {
  const { page } = await openDashboard(browser);
  await page.setViewportSize({ width: 390, height: 844 });
  await simulated(page);
  await page.click('#sim-events [data-frame="3"]');
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth), true);
  await page.close();
});

test("an additional season from the archive is selectable with its evaluation label, no special path", { skip }, async () => {
  const NEW = ["2024-11-14", "2024-11-15", "2024-11-16"];
  const s25 = { season: "2024-25", first: NEW[0], last: NEW.at(-1), n_dates: NEW.length, out_of_sample: true,
    in_sample_notes: [], independent_evaluation: true,
    evaluation_notes: ["the frozen sea-ice U-Net and its calibration were scored on this season after freezing (nothing fitted)"] };
  const dates = { ...datesPayload, route_dates: [...DATES, ...NEW], window_dates: [...DATES, ...NEW],
    seasons: { route: [...datesPayload.seasons.route, s25], window: [...datesPayload.seasons.window, s25] } };
  const bodies = [];
  const { page } = await openDashboard(browser, { dates, plan: (b) => { bodies.push(b); return { status: 200, body: makePlan() }; } });
  await ready(page);
  const opts = await page.locator("#pr-season option").allInnerTexts();
  assert.deepEqual(opts, ["2023-24 (50 dates)", "2024-25 (3 dates, independent evaluation)"]);
  await page.selectOption("#pr-season", "2024-25");
  assert.equal(await page.inputValue("#pr-issue"), "2024-11-14");
  assert.equal(await page.getAttribute("#pr-issue", "max"), "2024-11-16");
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-result:not([hidden])");
  assert.equal(bodies.length, 1);
  assert.equal(bodies[0].issue, "2024-11-14");
  await page.close();
});

/* ------------------------------------------------------------- optional real smoke */
const SMOKE = process.env.ANTROUTE_SMOKE_URL;
test("real smoke: Drake Passage to Bransfield Strait, 2023-11-14, reproduces the frozen plan", {
  skip: skip || (SMOKE ? false : "set ANTROUTE_SMOKE_URL to run against a live API with the real archive"), timeout: 240000,
}, async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const posts = [];
  page.on("request", (r) => { if (new URL(r.url()).pathname === "/real/plan") posts.push(r.postData()); });
  await page.goto(SMOKE);
  await ready(page);
  await page.selectOption("#pr-origin", "drake_passage");
  await page.selectOption("#pr-destination", "bransfield_strait");
  await ready(page);
  await page.fill("#pr-issue", "2023-11-14");
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-result:not([hidden])", { timeout: 220000 });
  assert.equal(posts.length, 1);
  assert.match(await text(page, "#pr-verdict-badge"), /Recommended/);
  const metrics = await text(page, "#pr-metrics");
  for (const s of ["14 Nov 2023, 00:00 UTC", "15 Nov 2023, 13:30 UTC", "37.5 h", "838 km", "838"]) assert.ok(metrics.includes(s), s);
  assert.match(await text(page, '[data-risk="combined"]'), /1\.9%[\s\S]*0 of 200 scenarios/);
  assert.equal((await page.locator("#pr-options tbody tr").allInnerTexts()).length, 14);
  assert.match(await text(page, "#pr-disclosure"), /Hindsight forcing/);
  await page.close();
});

test("real smoke: Simulate Voyage sails the frozen 2023-11-14 plan with no replan", {
  skip: skip || (SMOKE ? false : "set ANTROUTE_SMOKE_URL to run against a live API with the real archive"), timeout: 480000,
}, async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const calls = { plan: 0, simulate: 0 };
  page.on("request", (r) => {
    const p = new URL(r.url()).pathname;
    if (r.method() === "POST" && p === "/real/plan") calls.plan += 1;
    if (r.method() === "POST" && p === "/real/simulate") calls.simulate += 1;
  });
  await page.goto(SMOKE);
  await ready(page);
  await page.selectOption("#pr-origin", "drake_passage");
  await page.selectOption("#pr-destination", "bransfield_strait");
  await ready(page);
  await page.fill("#pr-issue", "2023-11-14");
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-result:not([hidden])", { timeout: 220000 });
  await page.click("#pr-simulate");
  await page.waitForSelector("#sim-body:not([hidden])", { timeout: 220000 });
  assert.deepEqual(calls, { plan: 1, simulate: 1 });
  assert.match(await text(page, "#sim-slider-out"), /2023-11-14 \(0 of 2\)/);
  assert.match(await text(page, "#sim-subtitle"), /No replan was required during this voyage\./);
  assert.match(await text(page, "#sim-banners"), /HISTORICAL MODE[\s\S]*hindsight[\s\S]*not certified navigation/i);
  await page.click('#sim-events [data-frame="2"]');
  assert.ok(await page.locator("#sim-summary").isVisible());
  assert.match(await text(page, "#sim-summary"), /No replan was required during this voyage\./);
  await page.close();
});

test("real smoke: a 2024-25 issue date plans and simulates through the normal flow", {
  skip: skip || (SMOKE ? false : "set ANTROUTE_SMOKE_URL to run against a live API with the real archive"), timeout: 480000,
}, async () => {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  await page.goto(SMOKE);
  await ready(page);
  await page.selectOption("#pr-origin", "drake_passage");
  await page.selectOption("#pr-destination", "bransfield_strait");
  await ready(page);
  assert.ok((await page.locator("#pr-season option").allInnerTexts()).includes("2024-25 (88 dates, independent evaluation)"));
  await page.selectOption("#pr-season", "2024-25");
  assert.equal(await page.inputValue("#pr-issue"), "2024-11-14");
  await page.click("#pr-submit");
  await page.waitForSelector("#pr-result:not([hidden])", { timeout: 220000 });
  assert.match(await text(page, "#pr-verdict-badge"), /Recommended/);
  assert.match(await text(page, "#pr-metrics"), /14 Nov 2024, 00:00 UTC/);
  await page.click("#pr-simulate");
  await page.waitForSelector("#sim-body:not([hidden])", { timeout: 220000 });
  assert.match(await text(page, "#sim-slider-out"), /2024-11-14 \(0 of 2\)/);
  assert.match(await text(page, "#sim-subtitle"), /No replan was required during this voyage\./);
  assert.match(await text(page, "#sim-banners"), /HISTORICAL MODE[\s\S]*hindsight[\s\S]*not certified navigation/i);
  await page.close();
});
