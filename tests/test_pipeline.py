"""End-to-end: a route directory in, a full report out, no GPU involved."""

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from irap_screening.config import load_config
from irap_screening.pipeline import run_pipeline
from tests.factories import curved_route


def _write_route(root: Path) -> None:
    frames, gps, imu = curved_route(radius_m=312.5, arc_length_m=300.0)
    (root / "frames").mkdir(parents=True, exist_ok=True)

    image = np.full((480, 640, 3), 100, dtype=np.uint8)
    image[:, 317:324] = 220
    for frame in frames:
        cv2.imwrite(str(root / frame.path), image)

    with (root / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "path", "timestamp_utc", "sha256"])
        for frame in frames:
            writer.writerow(
                [frame.frame_id, frame.path, frame.timestamp_utc.isoformat(), frame.sha256]
            )

    with (root / "gps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "lat", "lon", "speed_mps"])
        for fix in gps:
            writer.writerow(
                [
                    fix.timestamp_utc.isoformat(),
                    fix.lat,
                    fix.lon,
                    fix.speed_mps,
                ]
            )

    with (root / "imu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "ax", "ay", "az", "gx", "gy", "gz"])
        for sample in imu:
            writer.writerow(
                [
                    sample.timestamp_utc.isoformat(),
                    sample.ax, sample.ay, sample.az,
                    sample.gx, sample.gy, sample.gz,
                ]
            )

    detections = root / "detections"
    detections.mkdir(exist_ok=True)
    records = [
        {
            "frame_id": frame.frame_id,
            "image_path": frame.path,
            "sha256": frame.sha256,
            "type": "lane",
            "transform": {
                "source_wh": [640, 480],
                "crop_top": 0.0,
                "crop_bottom": 0.0,
                "model_wh": [800, 320],
            },
            "instances": [
                {
                    "id": f"{frame.frame_id}#lane#0",
                    "points": [[320.0, 300.0], [320.0, 350.0]],
                    "conf": 0.9,
                }
            ],
        }
        for frame in frames
    ]
    (detections / "lane.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )


def test_pipeline_produces_a_report_without_a_gpu(tmp_path: Path):
    _write_route(tmp_path)
    report = run_pipeline(tmp_path, load_config())
    assert report["schema_version"] == "1.0"
    assert report["segments"]


def test_pipeline_flags_the_curve_as_critical(tmp_path: Path):
    _write_route(tmp_path)
    report = run_pipeline(tmp_path, load_config())
    assert report["segments"][0]["curvature"]["flag"] == "critical"


def test_pipeline_recommends_the_missing_curve_warning(tmp_path: Path):
    """No signs were detected: chevrons are missing. The curve starts with the
    route, so the advance sign's position was never photographed: unchecked."""
    _write_route(tmp_path)
    report = run_pipeline(tmp_path, load_config())
    entry = report["segments"][0]["recommendations"][0]
    assert entry["rule"] == "curve_warning_incomplete"
    assert entry["missing"] == ["chevron"]
    assert entry["unchecked"] == ["advance_bend_sign"]


def test_pipeline_codes_the_unsigned_curve_as_poor_quality(tmp_path: Path):
    _write_route(tmp_path)
    report = run_pipeline(tmp_path, load_config())
    assert report["segments"][0]["quality_of_curve"]["code"] == 2


def test_threshold_override_changes_the_verdict(tmp_path: Path):
    _write_route(tmp_path)
    cfg = load_config(overrides=["curvature.threshold_m=200"])
    report = run_pipeline(tmp_path, cfg)
    assert report["segments"][0]["curvature"]["flag"] == "normal"
    assert report["config"]["curvature"]["threshold_m"] == 200


def test_missing_module_results_raise_rather_than_silently_report_nothing():
    """A dropped module must fail loudly, not vanish into "no findings".

    gap-check treats an absent module key the same as a segment that could not be
    assessed, so without this guard a wiring mistake would produce a clean-looking
    report instead of an error.
    """
    import pytest

    from irap_screening.decision.report import SCORING_MODULES
    from irap_screening.pipeline import _require_all_modules

    with pytest.raises(ValueError, match="curvature"):
        _require_all_modules({name: [] for name in SCORING_MODULES if name != "curvature"})
