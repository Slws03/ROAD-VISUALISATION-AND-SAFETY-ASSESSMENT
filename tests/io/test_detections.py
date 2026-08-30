"""Detections cross a process boundary, so they are validated on entry."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from irap_screening.io.detections import load_detection_set, load_detections

LANE_RECORD = {
    "frame_id": "f_000001",
    "image_path": "frames/f_000001.jpg",
    "sha256": "aaaa1111",
    "type": "lane",
    "transform": {
        "source_wh": [1920, 1080],
        "crop_top": 0.25,
        "crop_bottom": 0.20,
        "model_wh": [800, 320],
    },
    "instances": [
        {"id": "f_000001#lane#0", "points": [[100.0, 700.0], [110.0, 650.0]], "conf": 0.87}
    ],
}

SIGN_RECORD = {
    "frame_id": "f_000001",
    "image_path": "frames/f_000001.jpg",
    "sha256": "aaaa1111",
    "type": "sign",
    "transform": {
        "source_wh": [1920, 1080],
        "crop_top": 0.0,
        "crop_bottom": 0.0,
        "model_wh": [640, 640],
    },
    "instances": [
        {"id": "f_000001#sign#0", "bbox": [10.0, 20.0, 60.0, 80.0], "conf": 0.91, "cls": "Bend"}
    ],
}


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )


def test_loads_records_keyed_by_frame_id(tmp_path: Path):
    _write_jsonl(tmp_path / "lane.jsonl", [LANE_RECORD])
    loaded = load_detections(tmp_path / "lane.jsonl")
    assert set(loaded) == {"f_000001"}
    assert loaded["f_000001"].instances[0].conf == pytest.approx(0.87)


def test_transform_is_preserved_for_audit(tmp_path: Path):
    _write_jsonl(tmp_path / "lane.jsonl", [LANE_RECORD])
    record = load_detections(tmp_path / "lane.jsonl")["f_000001"]
    assert record.transform.crop_top == pytest.approx(0.25)
    assert record.transform.model_wh == (800, 320)


def test_malformed_record_fails_loudly(tmp_path: Path):
    bad = dict(LANE_RECORD)
    bad["instances"] = [{"id": "x", "points": "not-a-list", "conf": 0.5}]
    _write_jsonl(tmp_path / "lane.jsonl", [bad])
    with pytest.raises(ValidationError):
        load_detections(tmp_path / "lane.jsonl")


def test_blank_lines_are_skipped(tmp_path: Path):
    path = tmp_path / "lane.jsonl"
    path.write_text(json.dumps(LANE_RECORD) + "\n\n\n", encoding="utf-8")
    assert len(load_detections(path)) == 1


def test_detection_set_loads_all_three_streams(tmp_path: Path):
    _write_jsonl(tmp_path / "lane.jsonl", [LANE_RECORD])
    _write_jsonl(tmp_path / "sign.jsonl", [SIGN_RECORD])
    detections = load_detection_set(tmp_path)
    assert set(detections.lane) == {"f_000001"}
    assert set(detections.sign) == {"f_000001"}


def test_missing_pothole_file_is_empty_not_an_error(tmp_path: Path):
    """Road Condition is stubbable; its absence must not break the run."""
    _write_jsonl(tmp_path / "lane.jsonl", [LANE_RECORD])
    detections = load_detection_set(tmp_path)
    assert detections.pothole == {}
