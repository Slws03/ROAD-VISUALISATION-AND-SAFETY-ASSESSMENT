"""Weber contrast grades marking degradation; CLRerNet only says present or not."""

from datetime import timedelta
from pathlib import Path

import cv2
import numpy as np
import pytest

from irap_screening.config import load_config
from irap_screening.io.align import align_route
from irap_screening.io.detections import DetectionSet
from irap_screening.io.schema import DetectionRecord, Flag, LaneInstance, Transform
from irap_screening.modules.lane import analyse_lane, weber_contrast_at
from irap_screening.segment import build_segments
from tests.factories import straight_route

BG_LEVEL = 100
LINE_X = 320


def _write_frame(path: Path, line_level: int | None) -> None:
    """Dark road with one bright vertical stripe, or no stripe at all."""
    image = np.full((480, 640, 3), BG_LEVEL, dtype=np.uint8)
    if line_level is not None:
        image[:, LINE_X - 3 : LINE_X + 4] = line_level
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _lane_record(frame_id: str, line_present: bool) -> DetectionRecord:
    instances = (
        [LaneInstance(id=f"{frame_id}#lane#0", points=[(float(LINE_X), 300.0)], conf=0.9)]
        if line_present
        else []
    )
    return DetectionRecord(
        frame_id=frame_id,
        image_path=f"frames/{frame_id}.jpg",
        sha256="0" * 16,
        type="lane",
        transform=Transform(
            source_wh=(640, 480), crop_top=0.0, crop_bottom=0.0, model_wh=(800, 320)
        ),
        instances=instances,
    )


def _build(tmp_path: Path, line_level: int | None, line_present: bool = True):
    frames, gps, imu = straight_route(length_m=200.0)
    for frame in frames:
        _write_frame(tmp_path / frame.path, line_level)
    route = align_route(frames, gps, imu)
    cfg = load_config()
    segments = build_segments(route, cfg)
    detections = DetectionSet(
        lane={f.frame_id: _lane_record(f.frame_id, line_present) for f in frames}
    )
    return route, segments, detections, cfg


def test_weber_contrast_matches_hand_computation():
    """C = (I - I_bg) / I_bg. A 200 stripe on a 100 background gives 1.0."""
    gray = np.full((480, 640), BG_LEVEL, dtype=np.uint8)
    gray[:, LINE_X - 3 : LINE_X + 4] = 200
    contrast = weber_contrast_at(gray, float(LINE_X), 300.0, bg_window=81)
    assert contrast == pytest.approx(1.0, rel=0.05)


def test_weber_contrast_is_near_zero_for_a_faded_line():
    gray = np.full((480, 640), BG_LEVEL, dtype=np.uint8)
    gray[:, LINE_X - 3 : LINE_X + 4] = 105
    contrast = weber_contrast_at(gray, float(LINE_X), 300.0, bg_window=81)
    assert contrast == pytest.approx(0.05, abs=0.03)


def test_bright_marking_is_normal(tmp_path: Path):
    results = analyse_lane(*_build(tmp_path, line_level=200), image_root=tmp_path)
    assert results[0].flag is Flag.NORMAL
    assert results[0].metrics["weber_contrast"] > 0.35


def test_mid_contrast_marking_is_watch(tmp_path: Path):
    results = analyse_lane(*_build(tmp_path, line_level=128), image_root=tmp_path)
    assert results[0].flag is Flag.WATCH


def test_faded_marking_is_critical(tmp_path: Path):
    results = analyse_lane(*_build(tmp_path, line_level=110), image_root=tmp_path)
    assert results[0].flag is Flag.CRITICAL


def test_no_detection_at_all_is_critical(tmp_path: Path):
    """No line detected is the worst case, not a missing value."""
    results = analyse_lane(
        *_build(tmp_path, line_level=None, line_present=False), image_root=tmp_path
    )
    assert results[0].flag is Flag.CRITICAL
    assert results[0].metrics["detection_rate"] == pytest.approx(0.0)


def test_detection_rate_is_reported(tmp_path: Path):
    results = analyse_lane(*_build(tmp_path, line_level=200), image_root=tmp_path)
    assert results[0].metrics["detection_rate"] == pytest.approx(1.0)
    assert results[0].metrics["total_frames"] > 0


def test_evidence_lists_contributing_instance_ids(tmp_path: Path):
    results = analyse_lane(*_build(tmp_path, line_level=200), image_root=tmp_path)
    assert results[0].evidence
    assert all("#lane#" in instance_id for instance_id in results[0].evidence)


def test_aggregation_takes_the_worst_frame(tmp_path: Path):
    """max_risk: one faded frame among bright ones codes the whole segment."""
    frames, gps, imu = straight_route(length_m=200.0)
    for index, frame in enumerate(frames):
        _write_frame(tmp_path / frame.path, 110 if index == 0 else 200)
    route = align_route(frames, gps, imu)
    cfg = load_config()
    segments = build_segments(route, cfg)
    detections = DetectionSet(
        lane={f.frame_id: _lane_record(f.frame_id, True) for f in frames}
    )
    results = analyse_lane(route, segments, detections, cfg, image_root=tmp_path)
    assert results[0].flag is Flag.CRITICAL


def test_segment_without_frames_is_not_assessed(tmp_path: Path):
    frames, gps, imu = straight_route(length_m=200.0)
    for frame in frames:
        _write_frame(tmp_path / frame.path, 200)
    # Push every frame timestamp past the route so no segment claims one.
    for frame in frames:
        frame.timestamp_utc = frame.timestamp_utc + timedelta(hours=1)
    route = align_route(frames, gps, imu)
    cfg = load_config()
    segments = build_segments(route, cfg)
    detections = DetectionSet(lane={})
    results = analyse_lane(route, segments, detections, cfg, image_root=tmp_path)
    assert results[0].flag is Flag.NOT_ASSESSED
