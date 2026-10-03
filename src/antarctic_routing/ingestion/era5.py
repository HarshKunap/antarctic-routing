"""Atmospheric forcing: ERA5 single levels from the Copernicus Climate Data Store.

Requires the ``cdsapi`` client and a CDS API key (``~/.cdsapirc`` or the
``CDSAPI_URL``/``CDSAPI_KEY`` environment variables). ERA5 is hourly; daily
summaries are derived downstream.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

from antarctic_routing.config import DomainSection
from antarctic_routing.ingestion.base import DownloadRequest, MissingCredentials, MissingDependency

DATASET = "reanalysis-era5-single-levels"
VARIABLES = (
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "mean_sea_level_pressure",
)


def era5_request(domain: DomainSection, start: date, end: date, hours: tuple[int, ...] = (0, 6, 12, 18)) -> dict:
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return {
        "product_type": ["reanalysis"],
        "variable": list(VARIABLES),
        "year": sorted({f"{d:%Y}" for d in days}),
        "month": sorted({f"{d:%m}" for d in days}),
        "day": sorted({f"{d:%d}" for d in days}),
        "time": [f"{h:02d}:00" for h in hours],
        "area": [domain.lat_max, domain.lon_min, domain.lat_min, domain.lon_max],  # N, W, S, E
        "data_format": "netcdf",
    }


def _has_credentials() -> bool:
    return bool(os.environ.get("CDSAPI_KEY")) or Path.home().joinpath(".cdsapirc").is_file()


def make_fetcher(domain: DomainSection):
    def fetch(request: DownloadRequest, dest: Path) -> dict:
        try:
            import cdsapi
        except ImportError as exc:
            raise MissingDependency("cdsapi is not installed (pip install '.[ingest]')") from exc
        if not _has_credentials():
            raise MissingCredentials("CDS API key not configured (~/.cdsapirc or CDSAPI_KEY)")
        body = era5_request(domain, request.start, request.end)
        cdsapi.Client().retrieve(DATASET, body, str(dest))
        return {"source_url": f"https://cds.climate.copernicus.eu/datasets/{DATASET}", "product_version": DATASET}

    return fetch


def request_for(domain: DomainSection, start: date, end: date) -> DownloadRequest:
    return DownloadRequest("era5", DATASET, VARIABLES, start, end,
                           (domain.lat_min, domain.lat_max, domain.lon_min, domain.lon_max))
