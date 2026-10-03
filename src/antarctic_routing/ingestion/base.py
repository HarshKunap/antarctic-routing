"""Stage 2 - reproducible, resumable, provenance-recording downloads.

Every download is described by a :class:`DownloadRequest`. Its deterministic key
names the output file, so the same request is never downloaded twice. Each file
gets a ``.manifest.json`` sidecar with the query parameters, retrieval time,
product version, source URL and SHA-256.

Data is written to a ``.part`` file and renamed only on success, so an
interrupted download never leaves a file that looks complete. Missing
credentials or client libraries produce a ``blocked`` StageResult - never
placeholder data.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from antarctic_routing import __version__
from antarctic_routing.common.provenance import StageResult, sha256_file, utc_now, write_json_artifact

SOFTWARE = {"name": "antarctic-routing", "version": __version__}


class MissingCredentials(RuntimeError):
    """Credentials for a data service are not configured."""


class MissingDependency(RuntimeError):
    """A client library needed for a data service is not installed."""


@dataclass(frozen=True)
class DownloadRequest:
    source: str
    product: str
    variables: tuple[str, ...]
    start: date
    end: date
    bbox: tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max
    extra: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["variables"] = list(self.variables)
        d["start"], d["end"] = self.start.isoformat(), self.end.isoformat()
        d["bbox"] = list(self.bbox)
        d["extra"] = dict(self.extra)
        return d

    def key(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return f"{self.start:%Y%m%d}-{self.end:%Y%m%d}-{hashlib.sha256(blob).hexdigest()[:12]}"


Fetcher = Callable[[DownloadRequest, Path], dict[str, Any]]


def target_path(request: DownloadRequest, root: Path, suffix: str = ".nc") -> Path:
    return Path(root) / request.source / request.product / f"{request.key()}{suffix}"


def run_download(request: DownloadRequest, root: Path, fetch: Fetcher, suffix: str = ".nc") -> StageResult:
    """Run ``fetch`` into a temp file, then atomically publish it with a manifest."""
    t0, started = time.time(), utc_now()
    target = target_path(request, root, suffix)
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    stage = f"ingest_{request.source}"
    params = request.to_dict()

    def result(status: str, outputs=(), warnings=(), metrics=None) -> StageResult:
        return StageResult(
            stage=stage, status=status, execution_mode="real", software=SOFTWARE,
            parameters=params, outputs=list(outputs), metrics=metrics or {},
            command=f"download {request.source}/{request.product}", started_at=started,
            finished_at=utc_now(), duration_seconds=round(time.time() - t0, 3), warnings=list(warnings),
        )

    if target.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") == sha256_file(target):
            return result("passed", [{"path": str(target), "sha256": manifest["sha256"]}],
                          [f"cached: {target.name} already downloaded and verified"])

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    try:
        meta = fetch(request, part) or {}
        if not part.is_file() or part.stat().st_size == 0:
            raise RuntimeError("fetcher produced no data")
        part.replace(target)
    except (MissingCredentials, MissingDependency) as exc:
        part.unlink(missing_ok=True)
        return result("blocked", warnings=[str(exc)])
    except Exception as exc:  # noqa: BLE001 - any failure must be reported, not raised mid-pipeline
        part.unlink(missing_ok=True)
        _prune_empty_dirs(target.parent, Path(root))
        return result("failed", warnings=[f"{type(exc).__name__}: {exc}"])

    digest = sha256_file(target)
    write_json_artifact(manifest_path, {
        "request": params, "retrieved_at": utc_now(), "sha256": digest,
        "bytes": target.stat().st_size, **meta,
    })
    return result("passed", [{"path": str(target), "sha256": digest}],
                  metrics={"bytes": target.stat().st_size})


def _prune_empty_dirs(path: Path, root: Path) -> None:
    path, root = path.resolve(), root.resolve()
    while path != root and path.is_dir() and not any(path.iterdir()):
        path.rmdir()
        path = path.parent
