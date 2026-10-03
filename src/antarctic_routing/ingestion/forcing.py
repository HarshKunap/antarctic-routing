"""Real forcing fields: ERA5 10 m wind and CMEMS surface currents on the polar grid.

Both products arrive on regular lat/lon grids with east/north vector
components. They are time-averaged over the downloaded period, interpolated to
the EPSG:3031 cell centres and *rotated* into grid x/y components (see
``preprocessing.grid``). Cells outside the source coverage (or on land in
CMEMS) are filled with zero velocity and the filled fraction is recorded.

The result feeds :class:`forecasting.scenarios.ForecastContext` (``--forcing``)
in place of the schematic currents and westerlies. A time mean is a deliberate
first step; per-day forcing can be added because ScenarioSet already accepts
time-varying (T, ny, nx) fields.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from antarctic_routing.common.provenance import sha256_file
from antarctic_routing.preprocessing.grid import CRS, PolarGrid
from antarctic_routing.preprocessing.harmonize import regrid_vector_latlon

_TIME_NAMES = ("valid_time", "time")
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon")


def _name(ds: xr.Dataset, options) -> str:
    for n in options:
        if n in ds.dims or n in ds.coords:
            return n
    raise ValueError(f"none of {options} found in dataset")


def read_latlon_vector(path: str | Path, u_name: str, v_name: str):
    """Time-mean (and surface-level) east/north components with their lat/lon axes."""
    with xr.open_dataset(path) as ds:
        u, v = ds[u_name], ds[v_name]
        if "depth" in u.dims:
            u, v = u.isel(depth=0), v.isel(depth=0)
        t = next((n for n in _TIME_NAMES if n in u.dims), None)
        if t:
            u, v = u.mean(t, skipna=True), v.mean(t, skipna=True)
        lat = ds[_name(ds, _LAT_NAMES)].values.astype(float)
        lon = ds[_name(ds, _LON_NAMES)].values.astype(float)
        lon = np.where(lon > 180, lon - 360, lon)
        return u.values.astype(float), v.values.astype(float), lat, lon


def _regrid(path, u_name, v_name, grid: PolarGrid):
    u, v, lat, lon = read_latlon_vector(path, u_name, v_name)
    order = np.argsort(lon)
    x, y = regrid_vector_latlon(u[:, order], v[:, order], lat, lon[order], grid)
    missing = ~(np.isfinite(x) & np.isfinite(y))
    return np.where(missing, 0.0, x), np.where(missing, 0.0, y), float(missing.mean())


def build_forcing(grid: PolarGrid, era5_path: str | Path | None = None,
                  cmems_path: str | Path | None = None) -> xr.Dataset:
    if era5_path is None and cmems_path is None:
        raise ValueError("provide an ERA5 and/or a CMEMS file")
    data_vars, attrs, sources = {}, {}, []
    if era5_path is not None:
        wx, wy, filled = _regrid(era5_path, "u10", "v10", grid)
        data_vars.update(wind_x=(("y", "x"), wx), wind_y=(("y", "x"), wy))
        attrs["wind_filled_fraction"] = filled
        sources.append(Path(era5_path))
    if cmems_path is not None:
        cx, cy, filled = _regrid(cmems_path, "uo", "vo", grid)
        data_vars.update(current_x=(("y", "x"), cx), current_y=(("y", "x"), cy))
        attrs["current_filled_fraction"] = filled
        sources.append(Path(cmems_path))
    for name in data_vars:
        data_vars[name] = (*data_vars[name], {"units": "m s-1", "long_name": f"{name} (EPSG:3031 grid component)"})
    return xr.Dataset(
        data_vars,
        coords={"y": grid.y, "x": grid.x},
        attrs={**attrs, "crs": CRS, "resolution_m": grid.resolution_m, "execution_mode": "real",
               "averaging": "time mean over the source files",
               "source_files": ",".join(p.name for p in sources),
               "source_sha256": ",".join(sha256_file(p) for p in sources)},
    )


def load_forcing(path: str | Path, expected_shape: tuple[int, int]):
    """Return ``(currents, winds)`` tuples (each ``(x, y)`` arrays or None) for ForecastContext."""
    with xr.open_dataset(path) as ds:
        shape = (ds.sizes["y"], ds.sizes["x"])
        if shape != tuple(expected_shape):
            raise ValueError(f"forcing grid {shape} does not match the dataset grid {tuple(expected_shape)}; "
                             "rebuild it with the same --resolution-km")
        currents = (ds["current_x"].values, ds["current_y"].values) if "current_x" in ds else None
        winds = (ds["wind_x"].values, ds["wind_y"].values) if "wind_x" in ds else None
    return currents, winds
