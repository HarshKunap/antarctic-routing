"""Read EUMETSAT OSI SAF OSI-401-b sea-ice concentration and harmonise it.

OSI-401-b is distributed on its *own* polar stereographic grid (true scale at
70S, Hughes 1980 ellipsoid, coordinates in km), not EPSG:3031. The CRS is read
from the file's CF ``grid_mapping`` metadata, falling back to the documented
proj4 string. Concentration is in percent, land/lake come from the CF
flag-coded ``status_flag`` variable, and missing ocean values are filled and
flagged by :func:`preprocessing.harmonize.fill_missing`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from scipy.interpolate import RegularGridInterpolator

from antarctic_routing.common.provenance import sha256_file
from antarctic_routing.preprocessing.grid import CRS as GRID_CRS
from antarctic_routing.preprocessing.grid import PolarGrid
from antarctic_routing.preprocessing.harmonize import fill_missing, normalize_concentration, to_dataset

OSI_PROJ4 = "+proj=stere +a=6378273 +b=6356889.44891 +lat_0=-90 +lat_ts=-70 +lon_0=0 +units=m +no_defs"
NON_NAVIGABLE_FLAGS = ("land", "lake")


@dataclass
class SourceField:
    day: date
    x: np.ndarray        # ascending, metres in ``crs``
    y: np.ndarray        # ascending, metres in ``crs``
    conc: np.ndarray     # (y, x) fraction, NaN where missing
    land: np.ndarray     # (y, x) bool
    crs: CRS
    path: Path
    sha256: str

    @property
    def date(self) -> date:
        return self.day


def _crs_of(ds: xr.Dataset, var: xr.DataArray) -> CRS:
    gm = var.attrs.get("grid_mapping")
    if gm and gm in ds.variables:
        attrs = dict(ds[gm].attrs)
        try:
            return CRS.from_cf(attrs)
        except Exception:  # noqa: BLE001 - fall back to the proj4 string the product also ships
            if "proj4_string" in attrs:
                return CRS.from_proj4(attrs["proj4_string"])
    return CRS.from_proj4(OSI_PROJ4)


def _flag_mask(flag: xr.DataArray, names: Sequence[str]) -> np.ndarray:
    values = np.asarray(flag.values)
    meanings = str(flag.attrs.get("flag_meanings", "")).split()
    out = np.zeros(values.shape, bool)
    if "flag_masks" in flag.attrs:
        for mask, meaning in zip(np.atleast_1d(flag.attrs["flag_masks"]), meanings, strict=False):
            if meaning in names:
                out |= (values.astype(np.int64) & int(mask)) != 0
    elif "flag_values" in flag.attrs:
        for val, meaning in zip(np.atleast_1d(flag.attrs["flag_values"]), meanings, strict=False):
            if meaning in names:
                out |= values == val
    return out


def _metres(coord: xr.DataArray) -> np.ndarray:
    scale = 1000.0 if str(coord.attrs.get("units", "m")).lower() in ("km", "kilometers", "kilometres") else 1.0
    return np.asarray(coord.values, float) * scale


def read_osisaf(path: str | Path) -> SourceField:
    path = Path(path)
    with xr.open_dataset(path) as ds:
        var = ds["ice_conc"]
        if "time" in var.dims:
            var = var.isel(time=0)
        crs = _crs_of(ds, ds["ice_conc"])
        units = str(var.attrs.get("units", "%")).lower()
        conc, _ = normalize_concentration(var.values, "percent" if units in ("%", "percent") else "fraction")
        if "status_flag" in ds:
            flag = ds["status_flag"].isel(time=0) if "time" in ds["status_flag"].dims else ds["status_flag"]
            land = _flag_mask(flag, NON_NAVIGABLE_FLAGS)
        else:
            land = np.zeros(conc.shape, bool)
        x, y = _metres(ds["xc"]), _metres(ds["yc"])
        day = pd.Timestamp(ds["time"].values[0]).date() if "time" in ds.coords else None
    if day is None:
        raise ValueError(f"{path}: no time coordinate")
    if x[0] > x[-1]:
        x, conc, land = x[::-1], conc[:, ::-1], land[:, ::-1]
    if y[0] > y[-1]:
        y, conc, land = y[::-1], conc[::-1], land[::-1]
    conc = np.where(land, np.nan, conc)
    return SourceField(day, x, y, conc, land, crs, path, sha256_file(path))


def regrid_projected(
    values: np.ndarray, src_x: np.ndarray, src_y: np.ndarray, src_crs: CRS, grid: PolarGrid, method: str = "linear"
) -> np.ndarray:
    """Sample a field on any projected regular grid at the EPSG:3031 cell centres."""
    to_src = Transformer.from_crs(GRID_CRS, src_crs, always_xy=True)
    xx, yy = np.meshgrid(grid.x, grid.y)
    sx, sy = to_src.transform(xx, yy)
    interp = RegularGridInterpolator((src_y, src_x), np.asarray(values, float), method=method,
                                     bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([np.ravel(sy), np.ravel(sx)])).reshape(grid.shape)


def build_sea_ice_dataset(paths: Sequence[str | Path], grid: PolarGrid) -> xr.Dataset:
    """Read daily OSI SAF files, regrid to ``grid`` and stack them by date."""
    fields = sorted((read_osisaf(p) for p in paths), key=lambda f: f.day)
    days = [f.day for f in fields]
    if len(set(days)) != len(days):
        raise ValueError("duplicate dates in input files")
    if not fields:
        raise ValueError("no input files")

    land = np.zeros(grid.shape, bool)
    for f in fields:
        land |= regrid_projected(f.land.astype(float), f.x, f.y, f.crs, grid, "nearest") > 0.5
    conc_stack, imputed_stack = [], []
    for f in fields:
        conc = regrid_projected(f.conc, f.x, f.y, f.crs, grid, "linear")
        filled, imputed = fill_missing(conc, ~land)
        conc_stack.append(filled)
        imputed_stack.append(imputed)

    ds = to_dataset(
        grid, [pd.Timestamp(d).to_pydatetime() for d in days],
        ice_concentration=np.stack(conc_stack).astype("float32"),
        imputed_mask=np.stack(imputed_stack),
        land_mask=land,
    )
    ds["ice_concentration"].attrs.update(units="1", long_name="sea ice area fraction")
    ds.attrs.update(
        source_product="OSI-401-b",
        source_files=",".join(f.path.name for f in fields),
        source_sha256=",".join(f.sha256 for f in fields),
        execution_mode="real",
    )
    return ds
