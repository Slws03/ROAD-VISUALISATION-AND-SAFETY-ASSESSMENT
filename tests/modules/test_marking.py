"""Existence facts for gap-check. No flags are produced here by design."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from irap_screening.config import load_config
from irap_screening.io.align import align_route
from irap_screening.io.detections import DetectionSet
from irap_screening.io.schema import (
    BoxInstance,
    DetectionRecord,
    ExistenceResult,
    LaneInstance,
    Transform,
)
from irap_screening.modules.marking import analyse_marking, scan_near_lane_point, weber_blobs
from irap_screening.segment import build_segments
from tests.factories import straight_route

BG = 100
ANCHOR_X = 320
ANCHOR_Y = 300


def _double_line_image(gap_px: int) -> np.ndarray:
    gray = np.full((480, 640), BG, dtype=np.uint8)
    left = ANCHOR_X - gap_px // 2
    right = ANCHOR_X + gap_px // 2
    gray[:, left - 2 : left + 3] = 230
    gray[:, right - 2 : right + 3] = 230
    return gray


def _single_line_image() -> np.ndarray:
    gray = np.full((480, 640), BG, dtype=np.uint8)
    gray[:, ANCHOR_X - 2 : ANCHOR_X + 3] = 230
    return gray


def _decoy_image() -> np.ndarray:
    """A lone true stroke on the anchor, plus a bright pair far to the left.

    Correct code never returns the decoy pair: at every window width the pair is
    either outside the window, rejected by anchor_tol, or rejected by max_gap.
    A naive full-row scan that grabs the first two blobs returns the decoy's gap.
    """
    gray = np.full((480, 640), BG, dtype=np.uint8)
    gray[:, ANCHOR_X - 2 : ANCHOR_X + 3] = 230   # the real, single stroke
    gray[:, 98:103] = 230                        # decoy pair, 60 px apart,
    gray[:, 158:163] = 230                       # both far from the anchor
    return gray


def test_weber_blobs_finds_two_bright_spans():
    gray = _double_line_image(gap_px=28)
    blobs = weber_blobs(gray[ANCHOR_Y], load_config())
    assert len(blobs) == 2


def test_weber_blobs_finds_one_span_for_single_line():
    blobs = weber_blobs(_single_line_image()[ANCHOR_Y], load_config())
    assert len(blobs) == 1


def test_scan_detects_double_line_and_returns_gap():
    gap = scan_near_lane_point(
        _double_line_image(gap_px=28), float(ANCHOR_X), float(ANCHOR_Y), load_config()
    )
    assert gap == pytest.approx(28.0, abs=4.0)


def test_scan_returns_none_for_single_line():
    gap = scan_near_lane_point(
        _single_line_image(), float(ANCHOR_X), float(ANCHOR_Y), load_config()
    )
    assert gap is None


def test_scan_widens_window_to_find_far_apart_strokes():
    """A 120 px gap exceeds the base 50 px half-width and needs widening."""
    gap = scan_near_lane_point(
        _double_line_image(gap_px=120), float(ANCHOR_X), float(ANCHOR_Y), load_config()
    )
    assert gap == pytest.approx(120.0, abs=10.0)


def test_scan_rejects_a_bright_pair_far_from_the_anchor():
    """Widening the window must not let an unrelated marking pair up.

    This is what anchor_tol_px exists for. Without the window and the tolerance
    check, the decoy pair at x=100/160 would be reported as a double line even
    though the anchored lane point sits at x=320 with a single stroke.
    """
    assert scan_near_lane_point(
        _decoy_image(), float(ANCHOR_X), float(ANCHOR_Y), load_config()
    ) is None


def test_a_stroke_split_by_a_one_pixel_dropout_still_counts_as_one_blob():
    """Worn paint drops pixels; merging must happen before the width filter.

    The two fragments are 2 px each, below min_width=3, so a filter-first
    implementation discards both and finds nothing. Merging first joins them
    into a single 5 px span that passes.
    """
    gray = np.full((480, 640), BG, dtype=np.uint8)
    gray[:, 318:320] = 230      # fragment one
    gray[:, 321:323] = 230      # fragment two, separated by a single dark pixel
    assert len(weber_blobs(gray[ANCHOR_Y], load_config())) == 1


def _build(tmp_path: Path, gray: np.ndarray, sign_cls: str | None):
    frames, gps, imu = straight_route(length_m=200.0)
    for frame in frames:
        (tmp_path / frame.path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(tmp_path / frame.path), gray)
    route = align_route(frames, gps, imu)
    cfg = load_config()
    segments = build_segments(route, cfg)

    lane = {
        f.frame_id: DetectionRecord(
            frame_id=f.frame_id,
            image_path=f.path,
            sha256="0" * 16,
            type="lane",
            transform=Transform(
                source_wh=(640, 480), crop_top=0.0, crop_bottom=0.0, model_wh=(800, 320)
            ),
            instances=[
                LaneInstance(
                    id=f"{f.frame_id}#lane#0",
                    points=[(float(ANCHOR_X), float(ANCHOR_Y))],
                    conf=0.9,
                )
            ],
        )
        for f in frames
    }
    sign = {}
    if sign_cls is not None:
        mid_frame = frames[len(frames) // 2]
        sign[mid_frame.frame_id] = DetectionRecord(
            frame_id=mid_frame.frame_id,
            image_path=mid_frame.path,
            sha256="0" * 16,
            type="sign",
            transform=Transform(
                source_wh=(640, 480), crop_top=0.0, crop_bottom=0.0, model_wh=(640, 640)
            ),
            instances=[
                BoxInstance(
                    id=f"{mid_frame.frame_id}#sign#0",
                    bbox=(10.0, 20.0, 60.0, 80.0),
                    conf=0.91,
                    cls=sign_cls,
                )
            ],
        )
    return route, segments, DetectionSet(lane=lane, sign=sign), cfg


def test_double_line_present_is_reported(tmp_path: Path):
    results = analyse_marking(
        *_build(tmp_path, _double_line_image(28), None), image_root=tmp_path
    )
    assert results[0].double_line.present is True


def test_single_line_reports_no_double_line(tmp_path: Path):
    results = analyse_marking(
        *_build(tmp_path, _single_line_image(), None), image_root=tmp_path
    )
    assert results[0].double_line.present is False


def test_sign_is_recorded_with_distance_along_route(tmp_path: Path):
    route, segments, detections, cfg = _build(tmp_path, _single_line_image(), "Bend")
    results = analyse_marking(route, segments, detections, cfg, image_root=tmp_path)

    signs = [sign for result in results for sign in result.signs]
    assert len(signs) == 1

    # The sign sits on a mid-route frame, so a zero here would mean the distance
    # lookup silently fell through to its default rather than reading frame_dist_m.
    expected = float(route.frame_dist_m[len(route.frames) // 2])
    assert expected > 0.0
    assert signs[0].at_m == pytest.approx(expected, rel=1e-6)


def test_module_returns_existence_results_not_flags(tmp_path: Path):
    """Giving this module a flag would invite folding it into the rating."""
    results = analyse_marking(
        *_build(tmp_path, _single_line_image(), None), image_root=tmp_path
    )
    assert all(isinstance(r, ExistenceResult) for r in results)
    assert not any(hasattr(r, "flag") for r in results)
