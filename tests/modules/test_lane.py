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
from irap_screening.modules.lane import (
    analyse_lane,
    frame_contrasts,
    irap_delineation_category,
    lane_contrasts,
    weber_contrast_at,
)
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


def test_a_marking_a_few_pixels_off_the_lane_point_is_still_found_on_a_large_frame():
    """CLRerNet's points can sit 8 px off the paint on a 1920-wide frame."""
    gray = np.full((1080, 1920), BG_LEVEL, dtype=np.uint8)
    gray[:, 1008:1015] = 200  # stripe centred 11 px right of the point at 1000
    assert weber_contrast_at(gray, 1000.0, 500.0, 81) == pytest.approx(0.0, abs=0.01)
    assert weber_contrast_at(gray, 1000.0, 500.0, 81, peak_frac=0.005) == pytest.approx(
        1.0, rel=0.05
    )


def test_the_peak_window_never_shrinks_below_4_px_on_a_small_frame():
    gray = np.full((480, 640), BG_LEVEL, dtype=np.uint8)
    gray[:, LINE_X + 4] = 200  # 0.005 * 640 rounds to 3 px; the 4 px floor still reaches it
    assert weber_contrast_at(gray, float(LINE_X), 300.0, 81, peak_frac=0.005) == pytest.approx(
        1.0, rel=0.05
    )


def test_bright_marking_is_normal(tmp_path: Path):
    results = analyse_lane(*_build(tmp_path, line_level=200), image_root=tmp_path)
    assert results[0].flag is Flag.NORMAL
    assert results[0].metrics["weber_contrast"] > 0.35


def test_mid_contrast_marking_is_critical(tmp_path: Path):
    """Two bands, not three: iRAP's Delineation is Adequate/Poor, no middle
    ground, and this project no longer keeps its own -- a marking faded past
    "clearly visible" grades critical/poor rather than a separate watch band.
    """
    results = analyse_lane(*_build(tmp_path, line_level=128), image_root=tmp_path)
    assert results[0].flag is Flag.CRITICAL


def test_irap_delineation_category_matches_the_flag():
    assert irap_delineation_category(Flag.NORMAL) == 1
    assert irap_delineation_category(Flag.CRITICAL) == 2


def test_analysis_reports_the_irap_category(tmp_path: Path):
    normal = analyse_lane(*_build(tmp_path, line_level=200), image_root=tmp_path)
    assert normal[0].metrics["irap_category"] == 1.0

    critical = analyse_lane(*_build(tmp_path, line_level=110), image_root=tmp_path)
    assert critical[0].metrics["irap_category"] == 2.0


def test_faded_marking_is_critical(tmp_path: Path):
    results = analyse_lane(*_build(tmp_path, line_level=110), image_root=tmp_path)
    assert results[0].flag is Flag.CRITICAL


def test_no_detection_at_all_is_not_assessed(tmp_path: Path):
    """Replaces an earlier rule that graded this critical. The stance changed.

    "The detector found no lane here" and "this road has no markings" are
    different claims, and CLRerNet's output cannot tell them apart. Calling it
    critical blames the road for a detector limitation; on KITTI, where a
    CULane-trained model reaches only 61 percent detection, that turned every
    segment red and the flag stopped carrying information.

    not_assessed says what is actually known. It is never collapsed into normal,
    so the segment appears in the report's coverage statistics rather than
    receiving a clean bill of health -- and it is the condition the planned
    review stage is meant to resolve.

    The cost is real: a road that genuinely has no markings is no longer flagged
    critical on the first pass. It is not called normal either.
    """
    route, segments, detections, cfg = _build(tmp_path, line_level=None, line_present=False)
    results = analyse_lane(route, segments, detections, cfg, image_root=tmp_path)
    assert results[0].flag is Flag.NOT_ASSESSED
    assert results[0].metrics["detection_rate"] == 0.0
    assert "unmarked" in (results[0].notes or "").lower()

def test_detection_rate_is_reported(tmp_path: Path):
    results = analyse_lane(*_build(tmp_path, line_level=200), image_root=tmp_path)
    assert results[0].metrics["detection_rate"] == pytest.approx(1.0)
    assert results[0].metrics["total_frames"] > 0


def test_evidence_lists_contributing_instance_ids(tmp_path: Path):
    results = analyse_lane(*_build(tmp_path, line_level=200), image_root=tmp_path)
    assert results[0].evidence
    assert all("#lane#" in instance_id for instance_id in results[0].evidence)


def _mixed(tmp_path: Path, faded) -> list:
    frames, gps, imu = straight_route(length_m=200.0)
    for index, frame in enumerate(frames):
        _write_frame(tmp_path / frame.path, 110 if faded(index) else 200)
    route = align_route(frames, gps, imu)
    cfg = load_config()
    segments = build_segments(route, cfg)
    detections = DetectionSet(
        lane={f.frame_id: _lane_record(f.frame_id, True) for f in frames}
    )
    return analyse_lane(route, segments, detections, cfg, image_root=tmp_path)


def test_one_faded_line_does_not_condemn_a_generally_visible_segment(tmp_path: Path):
    """iRAP Adequate asks for markings "generally present and visible", not every line."""
    results = _mixed(tmp_path, lambda index: index == 0)
    assert results[0].flag is Flag.NORMAL
    assert results[0].metrics["lines_visible"] == results[0].metrics["lines_total"] - 1
    assert results[0].metrics["weber_contrast"] < 0.35


def test_mostly_faded_lines_are_critical(tmp_path: Path):
    results = _mixed(tmp_path, lambda index: index != 0)
    assert results[0].flag is Flag.CRITICAL


def test_the_viewer_contrasts_are_the_ones_graded(tmp_path: Path):
    """The labels on the frames must be the numbers the flag came from."""
    route, segments, detections, cfg = _build(tmp_path, line_level=200)
    frame_ids = set(segments[0].frame_ids)

    contrasts = lane_contrasts(route, detections, cfg, tmp_path, frame_ids)
    result = analyse_lane(route, segments, detections, cfg, image_root=tmp_path)[0]

    assert set(contrasts) == set(result.evidence)
    assert min(contrasts.values()) == pytest.approx(result.metrics["weber_contrast"])


def _verge_frame() -> np.ndarray:
    """Asphalt (100) with dark verge (40) left of x=162; paint (200) at 162-164,
    320-322 and 478-480. A lane point at x=160 sits on the verge, 2 px off the
    paint, so most of its full background window is verge."""
    gray = np.full((480, 640), 100, dtype=np.uint8)
    gray[:, :162] = 40
    for x in (162, 320, 478):
        gray[:, x : x + 3] = 200
    return gray


def _vertical(lane_id: str, x: float) -> LaneInstance:
    return LaneInstance(id=lane_id, points=[(x, y) for y in (300.0, 350.0, 400.0)], conf=0.9)


def test_an_edge_line_takes_its_background_from_the_road_side():
    """Verge in the window pulled the median down and the contrast up (delineation_review §8)."""
    cfg = load_config()
    edge, other = _vertical("edge", 160.0), _vertical("other", 479.0)

    paired = frame_contrasts(_verge_frame(), [edge, other], cfg)
    alone = frame_contrasts(_verge_frame(), [edge], cfg)

    assert paired["edge"] == pytest.approx(1.0)  # (200 - 100) / 100: road side only
    assert alone["edge"] == pytest.approx(4.0)  # (200 - 40) / 40: one line, full window


def test_a_line_between_two_others_keeps_the_full_window():
    """Road on both sides; one side alone would be thrown by a vehicle or shadow."""
    cfg = load_config()
    gray = _verge_frame()
    gray[:, 280:320] = 40  # a shadow on the middle line's left only
    lanes = [_vertical("left", 160.0), _vertical("middle", 321.0), _vertical("right", 479.0)]

    contrasts = frame_contrasts(gray, lanes, cfg)

    assert contrasts["middle"] == pytest.approx(
        weber_contrast_at(gray, 321.0, 350.0, cfg.lane.bg_window, cfg.lane.peak_window_frac)
    )


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


def _dashed_frame(path: Path, paint_level: int, dash_period: int = 60, duty: int = 20) -> None:
    """A dashed stripe: paint for `duty` rows out of every `dash_period`."""
    image = np.full((480, 640, 3), BG_LEVEL, dtype=np.uint8)
    for top in range(0, 480, dash_period):
        image[top : top + duty, LINE_X - 3 : LINE_X + 4] = paint_level
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _line_record(frame_id: str, ys: list[float]) -> DetectionRecord:
    """One lane whose points run down the stripe, gaps included.

    CLRerNet returns points along the line's path, not along its paint, so a
    dashed marking is sampled through its gaps as well.
    """
    return DetectionRecord(
        frame_id=frame_id,
        image_path=f"frames/{frame_id}.jpg",
        sha256="0" * 16,
        type="lane",
        transform=Transform(
            source_wh=(640, 480), crop_top=0.0, crop_bottom=0.0, model_wh=(800, 320)
        ),
        instances=[
            LaneInstance(
                id=f"{frame_id}#lane#0", points=[(float(LINE_X), y) for y in ys], conf=0.9
            )
        ],
    )


def _build_dashed(tmp_path: Path, paint_level: int):
    frames, gps, imu = straight_route(length_m=200.0)
    ys = [float(y) for y in range(10, 470, 10)]
    for frame in frames:
        _dashed_frame(tmp_path / frame.path, paint_level)
    route = align_route(frames, gps, imu)
    cfg = load_config()
    return (
        route,
        build_segments(route, cfg),
        DetectionSet(lane={f.frame_id: _line_record(f.frame_id, ys) for f in frames}),
        cfg,
    )


def test_a_healthy_dashed_marking_is_not_read_as_worn(tmp_path: Path):
    """Most of a dashed line's length is gap, so most samples measure asphalt.

    Summarising a lane by the median therefore reports the gaps rather than the
    paint, and a perfectly good dashed marking grades as invisible. On KITTI
    this put 22 of 25 segments at critical on a road whose markings are crisp.
    """
    route, segments, detections, cfg = _build_dashed(tmp_path, paint_level=240)

    results = analyse_lane(route, segments, detections, cfg, image_root=tmp_path)

    assert results[0].flag is Flag.NORMAL
    assert results[0].metrics["weber_contrast"] > cfg.lane.weber_adequate


def test_the_median_would_have_called_that_dashed_marking_critical(tmp_path: Path):
    """Pins that the test above is checking real work, not a loosened band."""
    route, segments, detections, cfg = _build_dashed(tmp_path, paint_level=240)
    image = cv2.imread(str(tmp_path / route.frames[0].path), cv2.IMREAD_GRAYSCALE)
    samples = [
        weber_contrast_at(image, float(LINE_X), y, cfg.lane.bg_window)
        for y in range(10, 470, 10)
    ]
    samples = [s for s in samples if s is not None]

    assert float(np.median(samples)) < cfg.lane.weber_adequate
    assert float(np.percentile(samples, cfg.lane.contrast_percentile)) > cfg.lane.weber_adequate


def test_a_uniformly_worn_marking_is_still_critical(tmp_path: Path):
    """The percentile must not rescue a line that is faint along its whole length."""
    route, segments, detections, cfg = _build_dashed(tmp_path, paint_level=BG_LEVEL + 8)

    results = analyse_lane(route, segments, detections, cfg, image_root=tmp_path)

    assert results[0].flag is Flag.CRITICAL


def test_the_percentile_is_configurable(tmp_path: Path):
    route, segments, detections, _ = _build_dashed(tmp_path, paint_level=240)
    cfg = load_config(overrides=["lane.contrast_percentile=50"])

    results = analyse_lane(route, segments, detections, cfg, image_root=tmp_path)

    assert results[0].flag is Flag.CRITICAL


def _build_partial(tmp_path: Path, keep):
    """A segment where only the frames satisfying keep(index) carry a lane."""
    frames, gps, imu = straight_route(length_m=200.0)
    for frame in frames:
        _write_frame(tmp_path / frame.path, line_level=240)
    route = align_route(frames, gps, imu)
    cfg = load_config()
    return (
        route,
        build_segments(route, cfg),
        DetectionSet(
            lane={
                frame.frame_id: _lane_record(frame.frame_id, keep(index))
                for index, frame in enumerate(frames)
            }
        ),
        cfg,
    )


def test_a_single_missed_frame_does_not_condemn_the_segment(tmp_path: Path):
    """Markings do not vanish for 20 m and come back; a gap in detection is noise.

    The old rule scored a frame with no lane as zero contrast and aggregated by
    the worst frame, so one miss made the whole 100 m critical.
    """
    route, segments, detections, cfg = _build_partial(tmp_path, keep=lambda i: i != 1)

    results = analyse_lane(route, segments, detections, cfg, image_root=tmp_path)

    assert results[0].flag is Flag.NORMAL
    assert results[0].metrics["detection_rate"] < 1.0


def test_sparse_detection_is_not_assessed_rather_than_scored(tmp_path: Path):
    """Below the floor there is too little evidence to grade the road either way."""
    route, segments, detections, cfg = _build_partial(tmp_path, keep=lambda i: i % 5 == 0)

    results = analyse_lane(route, segments, detections, cfg, image_root=tmp_path)

    assert results[0].flag is Flag.NOT_ASSESSED
    assert results[0].metrics["detection_rate"] < cfg.lane.min_detection_rate
    assert results[0].metrics["detection_rate"] > 0.0


def test_the_detection_floor_is_configurable(tmp_path: Path):
    route, segments, detections, _ = _build_partial(tmp_path, keep=lambda i: i % 5 == 0)
    cfg = load_config(overrides=["lane.min_detection_rate=0.1"])

    results = analyse_lane(route, segments, detections, cfg, image_root=tmp_path)

    assert results[0].flag is Flag.NORMAL
