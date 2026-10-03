# Assumptions register

Stage 1 output: every value the system depends on, its status, and what must replace it before any operational interpretation.

**Status key:** 🟥 placeholder (must be replaced) · 🟧 modelling assumption (documented, test sensitivity) · 🟩 derived / standard

## Scenario

| Item | Value | Status | Justification / replacement |
|---|---|---|---|
| Region | Drake Passage → Bransfield Strait, lat −66…−55, lon −72…−52 | 🟧 | Open-water approach first; narrow Peninsula channels (Gerlache) need < 5 km grids |
| Season | Nov-Feb (austral summer) | 🟩 | Main Antarctic shipping season; seasons labelled by start year |
| Origin | (−56.3, −66.0) south of Cape Horn | 🟧 | Illustrative waypoint, not a port |
| Destination | (−63.0, −59.0) central Bransfield Strait | 🟧 | Illustrative waypoint |
| CRS | EPSG:3031 | 🟩 | Standard Antarctic polar stereographic; distances computed geodesically |
| Grid | 10 km, 24 h | 🟧 | Matches OSI SAF OSI-401-b (10 km). NSIDC CDR is 25 km and should not be presented as 10 km detail |

## Vessel

| Item | Value | Status | Replacement |
|---|---|---|---|
| `ice_class` | `REPLACE_WITH_VERIFIED_CLASS` | 🟥 | Vessel's verified Polar Class / ice class |
| Limit τ<sub>v</sub> | 0.15 concentration | 🟥 | Derived from vessel capability and operating guidance (e.g. POLARIS RIO), *not* from ice class alone |
| Cruise speed | 12 kn | 🟥 | Vessel data |
| Speed in ice | v(1 − 0.7C), floor 3 kn | 🟧 | Replace with an ice-performance curve |
| Fuel model | relative index d[1 + 4C²] | 🟧 | Relative only; replace with validated power/fuel-rate curve |

## Risk

| Item | Value | Status | Notes |
|---|---|---|---|
| Breach definition | any route cell reaches C ≥ τ<sub>v</sub> (or a tracked iceberg) at arrival time | 🟧 | Extend with ice type/thickness when available |
| Risk budget | 5% | 🟥 | Operator decision |
| Estimator | Wilson upper bound, 95% | 🟩 | Conservative; requires ≥ 73 scenarios for 5% (enforced by config validation) |
| Route scenarios | 200 | 🟩 | Resolution 0.5%; ML ensemble members can be expanded with forcing perturbations |
| Missing data | treated as hazardous | 🟩 | Conservative default |

## Forecasting and icebergs

| Item | Value | Status | Notes |
|---|---|---|---|
| Input / lead window | 14 d in, 1-7 d out | 🟧 | Extend leads only after beating baselines |
| Season split | chronological, val 3 / test 3 seasons | 🟩 | Climatology, ρ, residual bank: train only; calibration: validation only |
| Ensemble | forecast + training-season residual fields, 20 members | 🟧 | Residuals assume the error statistics are stationary |
| Iceberg wind factor α | U(0.01, 0.03) | 🟧 | Depends on berg size/shape; fit per berg class from tracks |
| Iceberg position error | σ = 2 km | 🟥 | Use the reported USNIC position accuracy |
| Iceberg safety radius | 10 km (CLI default) | 🟥 | Operator decision; should include berg size |
| Iceberg coverage | USNIC giant bergs only | 🟥 | Small bergs/growlers need SAR detection |

## Synthetic world (development only)

| Item | Status | Notes |
|---|---|---|
| Land mask | 🟥 | Hand-drawn schematic polygons; replace with NSIDC/ADD coastline mask |
| Ice edge | 🟥 | Retreats 0.06°/day from 61.6°S on 1 Nov with Weddell tongue and uncertain tongues; replace with forecasts |
| Currents | 🟥 | Schematic eastward ACC jet; replace with CMEMS `uo`/`vo` |
| Winds | 🟥 | Schematic westerlies peaking near 57°S; replace with ERA5 u10/v10 |
| Training history | 🟥 | `synthetic_history`: retreating edge, AR(1) season anomaly (0.97/day), tongues drifting 0.25° lon/day east |

## Known simplifications

- Ground speed uses only the along-track current; cross-track drift (crabbing) is ignored.
- Search uses conditions at edge departure (speed) and arrival (ice); exact for travel time only under FIFO.
- Conditions beyond the generated horizon reuse the last layer and are flagged (`beyond_horizon_fraction`). Phase 3 replaces this with climatology-anomaly scenarios.
- Candidate routes are evaluated on the same scenarios used to plan them (in-sample). Independent validation comes from historical replay (Phase 4).
