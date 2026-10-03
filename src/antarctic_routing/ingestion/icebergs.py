"""Iceberg positions from U.S. National Ice Center (USNIC) Antarctic iceberg lists.

USNIC tracks only *large* icebergs (roughly > 10 nautical miles on the long
axis, e.g. A23A). Smaller bergs, bergy bits and growlers - the ones most
relevant to ship strikes - are NOT included, so any layer built from this data
is a "tracked large-iceberg hazard", not a complete encounter risk.

The USNIC download location changes over time, so this module imports a file
the operator has already downloaded; it never fetches a URL itself.
"""

from __future__ import annotations

import csv
import re
from datetime import date, datetime
from pathlib import Path

NM_TO_KM = 1.852
_DMS = re.compile(r"^\s*(-?\d+(?:\.\d+)?)(?:\s+(\d+(?:\.\d+)?)'?)?\s*([NSEW])?\s*$", re.IGNORECASE)


def parse_coordinate(text: str) -> float:
    """Parse ``-65.5``, ``65.5S``, ``65 30S`` or ``65 30'S`` to signed decimal degrees."""
    m = _DMS.match(str(text))
    if not m:
        raise ValueError(f"unrecognised coordinate: {text!r}")
    deg, minutes, hemi = float(m.group(1)), m.group(2), (m.group(3) or "").upper()
    value = abs(deg) + (float(minutes) / 60.0 if minutes else 0.0)
    negative = deg < 0 or hemi in ("S", "W")
    return -value if negative else value


def _parse_date(text: str) -> date:
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%d-%b-%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            continue
    raise ValueError(f"unrecognised date: {text!r}")


def read_iceberg_positions(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        for raw in csv.DictReader(fh):
            r = {k.strip().lower(): (v or "").strip() for k, v in raw.items() if k}
            rows.append({
                "iceberg_id": r["iceberg"].upper(),
                "date": _parse_date(r.get("updated") or r.get("date", "")),
                "lat": parse_coordinate(r["latitude"]),
                "lon": parse_coordinate(r["longitude"]),
                "length_km": float(r["length (nm)"]) * NM_TO_KM if r.get("length (nm)") else None,
                "width_km": float(r["width (nm)"]) * NM_TO_KM if r.get("width (nm)") else None,
            })
    return rows
