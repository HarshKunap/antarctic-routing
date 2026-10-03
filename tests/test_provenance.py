import json

import pytest

from antarctic_routing.common.provenance import (
    StageResult,
    file_record,
    sha256_file,
    write_json_artifact,
)


def _ok(**overrides):
    base = dict(
        stage="routing",
        status="passed",
        execution_mode="controlled_synthetic",
        software={"name": "antarctic-routing", "version": "0.1.0"},
    )
    base.update(overrides)
    return StageResult(**base)


def test_valid_stage_result_serialises():
    payload = _ok(metrics={"p_breach": 0.01}).to_dict()
    assert payload["status"] == "passed"
    assert payload["schema_version"] == "1.0"
    assert payload["metrics"] == {"p_breach": 0.01}


@pytest.mark.parametrize(
    "field,value",
    [("status", "success"), ("execution_mode", "fake"), ("stage", "  "), ("duration_seconds", -1)],
)
def test_invalid_fields_rejected(field, value):
    with pytest.raises(ValueError):
        _ok(**{field: value})


def test_software_requires_name_and_version():
    with pytest.raises(ValueError):
        _ok(software={"name": "x"})


def test_sha256_matches_known_digest(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"abc")
    assert sha256_file(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert file_record(path, "unit")["sha256"] == sha256_file(path)


def test_write_json_artifact_is_atomic_and_sorted(tmp_path):
    out = write_json_artifact(tmp_path / "nested" / "r.json", {"b": 1, "a": 2})
    text = out.read_text()
    assert text.index('"a"') < text.index('"b"')
    assert json.loads(text) == {"a": 2, "b": 1}
    assert not list(tmp_path.rglob("*.tmp"))
