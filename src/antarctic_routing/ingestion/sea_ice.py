"""Sea-ice concentration: EUMETSAT OSI SAF OSI-401-b (Southern Hemisphere, 10 km).

Daily NetCDF files are public over HTTPS from the MET Norway THREDDS server; no
credentials are needed. Concentration is in percent and must be converted with
``preprocessing.harmonize.normalize_concentration(..., units="percent")``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from antarctic_routing.ingestion.base import DownloadRequest, MissingDependency

PRODUCT = "OSI-401-b"
BASE = "https://thredds.met.no/thredds/fileServer/osisaf/met.no/ice/conc"


def osisaf_url(day: date) -> str:
    return f"{BASE}/{day:%Y}/{day:%m}/ice_conc_sh_polstere-100_multi_{day:%Y%m%d}1200.nc"


def fetch_osisaf(request: DownloadRequest, dest: Path, timeout: float = 60.0) -> dict:
    """Download a single day (``request.start == request.end``) to ``dest``."""
    if request.start != request.end:
        raise ValueError("OSI SAF requests are one file per day; split the date range")
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependency("requests is not installed (pip install '.[ingest]')") from exc
    url = osisaf_url(request.start)
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in resp.iter_content(1 << 20):
                fh.write(chunk)
    return {"source_url": url, "product_version": PRODUCT}


def daily_requests(start: date, end: date, bbox) -> list[DownloadRequest]:
    days = (end - start).days + 1
    return [
        DownloadRequest("sea_ice", PRODUCT, ("ice_conc", "status_flag"), d, d, tuple(bbox))
        for d in (start + timedelta(days=i) for i in range(days))
    ]
