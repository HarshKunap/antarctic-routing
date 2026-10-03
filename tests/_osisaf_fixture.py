"""OSI SAF OSI-401-b fixture files following the documented product layout (not real data)."""

from datetime import date

import numpy as np
import xarray as xr
from pyproj import Transformer

from antarctic_routing.ingestion.osisaf_reader import OSI_PROJ4


def _fixture_conc(lat):
    """Percent concentration, linear in latitude: 0% at -58, 100% at -64."""
    return np.clip((-58.0 - lat) / 6.0 * 100.0, 0, 100)


def write_osisaf(path, day: date, land_box=None, missing_box=None):
    to_osi = Transformer.from_crs("EPSG:4326", OSI_PROJ4, always_xy=True)
    xs, ys = to_osi.transform([-80, -80, -40, -40, -60], [-70, -50, -70, -50, -60])
    xc = np.arange(np.floor(min(xs) / 1e4) * 10, np.ceil(max(xs) / 1e4) * 10 + 10, 10.0)  # km
    yc = np.arange(np.ceil(max(ys) / 1e4) * 10, np.floor(min(ys) / 1e4) * 10 - 10, -10.0)  # descending, km
    xx, yy = np.meshgrid(xc * 1000, yc * 1000)
    lon, lat = Transformer.from_crs(OSI_PROJ4, "EPSG:4326", always_xy=True).transform(xx, yy)
    conc = _fixture_conc(lat)
    status = np.zeros(conc.shape, np.int8)
    def box(b):  # (lat_min, lat_max, lon_min, lon_max)
        return (lat >= b[0]) & (lat <= b[1]) & (lon >= b[2]) & (lon <= b[3])

    if land_box:
        status[box(land_box)] = 1
        conc[box(land_box)] = np.nan
    if missing_box:
        conc[box(missing_box)] = np.nan
    ds = xr.Dataset(
        {
            "ice_conc": (("time", "yc", "xc"), conc[None].astype("float32"),
                         {"units": "%", "grid_mapping": "Polar_Stereographic_Grid"}),
            "status_flag": (("time", "yc", "xc"), status[None],
                            {"flag_values": np.array([0, 1, 2], np.int8),
                             "flag_meanings": "nominal land lake"}),
            "Polar_Stereographic_Grid": ((), np.int32(0), {
                "grid_mapping_name": "polar_stereographic",
                "straight_vertical_longitude_from_pole": 0.0,
                "latitude_of_projection_origin": -90.0,
                "standard_parallel": -70.0,
                "false_easting": 0.0, "false_northing": 0.0,
                "semi_major_axis": 6378273.0, "semi_minor_axis": 6356889.44891,
                "proj4_string": OSI_PROJ4,
            }),
        },
        coords={"time": [np.datetime64(f"{day.isoformat()}T12:00")],
                "xc": ("xc", xc, {"units": "km"}), "yc": ("yc", yc, {"units": "km"})},
        attrs={"product_id": "OSI-401-b"},
    )
    ds.to_netcdf(path)
    return path
