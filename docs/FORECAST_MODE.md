# Forecast mode (hackathon estimate for future dates)

> **Forecast estimate:** future-year Antarctic observations/forcing are not fully available in this demo runtime.
> Results use the latest available real seasonal data and explicitly labelled proxy/estimated forcing where
> required. This is a research/hackathon estimate, not a certified navigation route.

## What it is

`POST /real/plan` and `POST /real/simulate` accept a departure date after the end of the real archive, such as
2026-11-19. The mode is chosen from the date alone:

| requested date | mode | `metadata.mode` | `execution_mode` | `data_status` |
|---|---|---|---|---|
| inside the real archive (last day 2025-02-28) | Real Historical Data | `historical` | `real` | `historical` |
| after the archive, in season (Nov–Feb) | Forecast / hackathon estimate | `forecast` | `modelled` | `forecast_estimate` |

Historical requests run exactly as before. Their output is unchanged byte for byte.

```bash
curl -s -X POST localhost:8000/real/plan -H 'content-type: application/json' \
  -d '{"origin":{"preset":"drake_passage"},"destination":{"preset":"bransfield_strait"},"issue":"2026-11-19"}'
```

Forecast mode adds no new routing, risk or departure-window logic. The same planner, Wilson risk, A* routing,
departure window and simulation run unchanged, from an **analogue start**:

1. **Analogue date.** The analogue is the requested calendar day in the latest archive season that can run the
   whole window. For 2026-11-19 this is 2024-11-19, an offset of 730 days.
2. **Engine run.** The engine runs on the analogue days, using:
   - the frozen U-Net with 200 members and seed 42;
   - ERA5 and CMEMS reanalysis of those days.
3. **Icebergs.** The latest official USNIC list on or before the requested date is used, and it may be at most
   120 days old. Each berg is held at its last reported position, then drifted with the calibrated ensemble
   (beta 0.1, alpha scale 0.1, spread factor 0.6053).
4. **Date shift.** Engine dates are shown on the requested timeline (engine day + offset). Real-world dates,
   such as the iceberg list date and the provenance of the analogue inputs, are never shifted.

`GET /real/historical/dates` returns `forecast.{dates, first, last}`. For Drake Passage → Bransfield Strait this
is 2026-11-14 … 2027-01-29 (77 dates). Any other in-season date after the archive is refused with HTTP 422
`forecast_unavailable` and the reason. The reasons are:
- no analogue season covers the run;
- no official iceberg list is recent enough.

An off-season date (March–October) stays in historical mode and is refused there as before (`out_of_coverage`).

## What is real and what is proxy

`metadata.forecast` records every input with its status and the dates actually used:

| input | status | what is used |
|---|---|---|
| sea-ice starting state | `proxy_analogue` | real OSI SAF observations of the analogue days (`observed_window_used`), **not** observations of the requested year |
| sea-ice forecast | `forecast` | the frozen U-Net and residual-bank scenarios from that proxy start |
| winds / currents | `proxy_analogue_reanalysis` | ERA5 / CMEMS reanalysis of the analogue days (`dates_used`, files and SHA-256). Not a forecast, and not a measurement of the requested year |
| icebergs | `observed_snapshot_held` | an official USNIC list (`snapshot_date`, `file`, `sha256`, age in days, number of source bergs, ids in the grid), held and then drifted |

Three places in the output that would otherwise say "observed" are relabelled `proxy_analogue_observed`:
- the scenario layer 0;
- the sailed ice in a simulation;
- each simulated day's `sea_ice_on_segment`, which also carries its `analogue_date`.

`observations_for_requested_dates` states that no observation dated on or after the requested date is used.
`metadata.forecast.limitations` repeats the engine's limitations, with the historical iceberg-age line ("up to 14 days
old") restated for the held snapshot: it may be up to 120 days old, and its actual age is in `icebergs`. This is
wording only; no rule or calculation changes.

The two 2026 USNIC lists are pinned in `config/real_historical.json` under `inputs.forecast_icebergs`:
- `AntarcticIcebergs_20260924.csv`;
- `AntarcticIcebergs_20261001.csv`.

They are verified only when forecast mode is used, so historical mode does not depend on them.

The dashboard shows a **REAL HISTORICAL DATA** or **FORECAST / HACKATHON ESTIMATE** badge next to the date. The
date field is the only date control: it takes any date the server lists, past or future, and the mode follows from the
date. Its default is 2026-11-19 when the route's forecast range includes it; a date the user chose is never replaced.
A forecast result carries:
- the disclosure above;
- the forecast banners;
- a Data & Confidence panel that names every proxy and its dates.

Nothing in this mode changes the frozen datasets, checkpoint, calibration, parameters or historical results.

## Why a 2026 result is not an operational forecast

- **Sea ice and weather come from another year.** The ice, winds and currents are those of 2024-25. They show
  what a typical season can look like on that calendar day, not what will happen in 2026.
- **The bergs are not tracked to the departure date.** Their positions are weeks old, and the drift between the
  list date and the departure is not modelled. The bergs are only held in place over that gap.
- **The method has not been validated.** The analogue approach has not been checked against what actually
  happened in any season.
- **The vessel is a placeholder.** The vessel and the fuel index are generic, and nothing is certified.

## What true live forecasting would need

1. **Near-real-time sea ice.** Daily OSI SAF OSI-401 sea-ice concentration up to the issue date, which the
   OSI-401-d compatibility work prepares. This would replace the analogue start.
2. **Operational weather and ocean forecasts.** Winds from ECMWF HRES/ENS and currents from the CMEMS analysis &
   forecast products, issued on the issue date, instead of reanalysis.
3. **Current iceberg positions.** The current USNIC list, or satellite iceberg detections, each issue day, with
   drift starting from that list's date.
4. **Live validation.** Validation of the whole chain on live, issue-time inputs, run through a season, before
   any skill claim is made.
5. **Operational infrastructure.** Data ingestion, monitoring and fall-back handling, together with
   vessel-specific limits and certification, and qualified human oversight.
