"""Stage 3 - harmonise heterogeneous sources onto the common polar grid.

Interpolation choices follow the variable type:

=========================  =====================================================
Variable                   Method
=========================  =====================================================
Ice concentration          area-mean (same-projection, finer source) or bilinear
Wind / current vectors     interpolate east/north components, then *rotate*
Temperature                bilinear
Land / categorical masks   nearest neighbour
=========================  =====================================================

Missing values are filled only over ocean, and every filled cell is flagged so
downstream models and the router know which values were imputed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt

from antarctic_routing.preprocessing.grid import CRS, PolarGrid


def _ascending(axis: np.ndarray, values: np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    if axis[0] > axis[-1]:
        return axis[::-1], np.flip(values, axis=dim)
    return axis, values


def regrid_latlon(
    values: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    grid: PolarGrid,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate a regular lat/lon field (ERA5, CMEMS) onto the polar grid.

    Target cells outside the source coverage become NaN rather than being
    extrapolated.
    """
    if method not in {"linear", "nearest"}:
        raise ValueError("method must be 'linear' or 'nearest'")
    lat, vals = _ascending(np.asarray(src_lat, float), np.asarray(values, float), 0)
    lon, vals = _ascending(np.asarray(src_lon, float), vals, 1)
    interp = RegularGridInterpolator((lat, lon), vals, method=method, bounds_error=False, fill_value=np.nan)
    return interp(np.column_stack([grid.lat2d.ravel(), grid.lon2d.ravel()])).reshape(grid.shape)


def regrid_vector_latlon(
    u_east: np.ndarray,
    v_north: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    grid: PolarGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Regrid an east/north vector field and return grid-aligned (x, y) components."""
    ue = regrid_latlon(u_east, src_lat, src_lon, grid)
    vn = regrid_latlon(v_north, src_lat, src_lon, grid)
    return grid.rotate_en_to_xy(ue, vn, grid.lon2d)


def area_mean_regrid(values: np.ndarray, src_x: np.ndarray, src_y: np.ndarray, target: PolarGrid) -> np.ndarray:
    """Area-aware coarsening for a finer source already in EPSG:3031.

    Each target cell receives the mean of the valid source cells whose centres
    fall inside it, so missing source cells are ignored rather than counted as
    open water.
    """
    vals = np.asarray(values, float)
    res = target.resolution_m
    cols = np.floor((np.asarray(src_x) - (target.x[0] - res / 2)) / res).astype(int)
    rows = np.floor((np.asarray(src_y) - (target.y[0] - res / 2)) / res).astype(int)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    keep = (rr >= 0) & (rr < target.y.size) & (cc >= 0) & (cc < target.x.size) & np.isfinite(vals)
    sums = np.zeros(target.shape)
    counts = np.zeros(target.shape)
    np.add.at(sums, (rr[keep], cc[keep]), vals[keep])
    np.add.at(counts, (rr[keep], cc[keep]), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def fill_missing(values: np.ndarray, ocean_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-valid-neighbour fill over ocean only.

    Returns ``(filled, imputed)`` where ``imputed`` flags every filled cell.
    Land cells are set to NaN and never imputed.
    """
    vals = np.asarray(values, float)
    ocean = np.asarray(ocean_mask, bool)
    valid = ocean & np.isfinite(vals)
    if not valid.any():
        raise ValueError("no valid ocean values to fill from")
    imputed = ocean & ~np.isfinite(vals)
    _, indices = distance_transform_edt(~valid, return_indices=True)
    nearest = vals[tuple(indices)]
    filled = np.where(imputed, nearest, vals)
    filled[~ocean] = np.nan
    return filled, imputed


def normalize_concentration(values: np.ndarray, units: str) -> tuple[np.ndarray, dict[str, int]]:
    """Convert sea-ice concentration to a 0-1 fraction and clip out-of-range values."""
    vals = np.asarray(values, float)
    if units == "percent":
        vals = vals / 100.0
    elif units != "fraction":
        raise ValueError("units must be 'percent' or 'fraction'")
    finite = np.isfinite(vals)
    clipped = int(np.count_nonzero(finite & ((vals < 0) | (vals > 1))))
    out = np.where(finite, np.clip(vals, 0.0, 1.0), np.nan)
    return out, {"clipped": clipped, "missing": int(np.count_nonzero(~finite))}


def to_dataset(grid: PolarGrid, times: Sequence[datetime], **fields: np.ndarray) -> xr.Dataset:
    """Assemble harmonised fields into an ``xarray.Dataset`` on the polar grid."""
    data_vars = {}
    for name, arr in fields.items():
        arr = np.asarray(arr)
        if arr.ndim == 3:
            data_vars[name] = (("time", "y", "x"), arr)
        elif arr.ndim == 2:
            data_vars[name] = (("y", "x"), arr)
        else:
            raise ValueError(f"{name}: expected 2-D or 3-D array, got {arr.ndim}-D")
    return xr.Dataset(
        data_vars,
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "y": grid.y,
            "x": grid.x,
            "lat": (("y", "x"), grid.lat2d),
            "lon": (("y", "x"), grid.lon2d),
        },
        attrs={"crs": CRS, "resolution_m": grid.resolution_m},
    )
