"""Ocean currents and temperature: Copernicus Marine Service (CMEMS).

Uses the ``copernicusmarine`` toolbox. Credentials come from
``COPERNICUSMARINE_SERVICE_USERNAME``/``COPERNICUSMARINE_SERVICE_PASSWORD`` or a
prior ``copernicusmarine login``. Record whether the chosen dataset is a
reanalysis (``_my_``) or analysis/forecast (``_anfc_``) product.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from antarctic_routing.config import DomainSection
from antarctic_routing.ingestion.base import DownloadRequest, MissingCredentials, MissingDependency

REANALYSIS = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
ANALYSIS_FORECAST = "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m"
VARIABLES = ("uo", "vo", "thetao")


def cmems_subset_kwargs(domain: DomainSection, start: date, end: date, output_filename: str,
                        dataset_id: str = REANALYSIS) -> dict:
    return {
        "dataset_id": dataset_id,
        "variables": list(VARIABLES),
        "minimum_latitude": domain.lat_min,
        "maximum_latitude": domain.lat_max,
        "minimum_longitude": domain.lon_min,
        "maximum_longitude": domain.lon_max,
        "minimum_depth": 0.0,
        "maximum_depth": 1.0,
        "start_datetime": f"{start.isoformat()}T00:00:00",
        "end_datetime": f"{end.isoformat()}T23:59:59",
        "output_filename": output_filename,
    }


def make_fetcher(domain: DomainSection, dataset_id: str = REANALYSIS):
    def fetch(request: DownloadRequest, dest: Path) -> dict:
        try:
            import copernicusmarine
        except ImportError as exc:
            raise MissingDependency("copernicusmarine is not installed (pip install '.[ingest]')") from exc
        has_env = os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME") and os.environ.get(
            "COPERNICUSMARINE_SERVICE_PASSWORD")
        has_file = Path.home().joinpath(".copernicusmarine").exists()
        if not (has_env or has_file):
            raise MissingCredentials("Copernicus Marine credentials not configured")
        kwargs = cmems_subset_kwargs(domain, request.start, request.end, dest.name, dataset_id)
        copernicusmarine.subset(**kwargs, output_directory=str(dest.parent))
        return {"source_url": f"https://data.marine.copernicus.eu/product/{dataset_id}",
                "product_version": dataset_id,
                "product_kind": "reanalysis" if "_my_" in dataset_id else "analysis_forecast"}

    return fetch


def request_for(domain: DomainSection, start: date, end: date, dataset_id: str = REANALYSIS) -> DownloadRequest:
    return DownloadRequest("cmems", dataset_id, VARIABLES, start, end,
                           (domain.lat_min, domain.lat_max, domain.lon_min, domain.lon_max))
