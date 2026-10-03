"""Truth-preserving result envelope shared by every pipeline stage.

Every stage records whether it ran on real data, controlled synthetic data or
modelled assumptions. A missing dataset, credential or dependency yields
``blocked`` - never a fabricated success.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VALID_STATUSES = frozenset({"passed", "failed", "blocked", "fallback"})
VALID_EXECUTION_MODES = frozenset({"real", "controlled_synthetic", "modelled"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(slots=True)
class StageResult:
    stage: str
    status: str
    execution_mode: str
    software: dict[str, str]
    schema_version: str = "1.0"
    inputs: list[dict[str, Any]] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    outputs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    command: str = ""
    started_at: str = field(default_factory=utc_now)
    finished_at: str = field(default_factory=utc_now)
    duration_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValueError("stage must be non-empty")
        if self.status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        if self.execution_mode not in VALID_EXECUTION_MODES:
            raise ValueError(f"execution_mode must be one of {sorted(VALID_EXECUTION_MODES)}")
        if not isinstance(self.software, dict) or not all(
            self.software.get(key) for key in ("name", "version")
        ):
            raise ValueError("software must contain non-empty name and version")
        if self.duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, source: str) -> dict[str, str]:
    resolved = Path(path)
    return {"path": str(resolved), "source": source, "sha256": sha256_file(resolved)}


def write_json_artifact(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON atomically (temp file + rename) with sorted keys."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n"
    temporary.write_text(encoded, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    return destination
