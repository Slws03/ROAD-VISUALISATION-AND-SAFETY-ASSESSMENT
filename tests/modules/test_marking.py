"""Existence facts for gap-check. No flags are produced here by design."""

import pytest

from irap_screening.config import load_config
from irap_screening.io.align import align_route
from irap_screening.io.detections import DetectionSet
from irap_screening.io.schema import (
    BoxInstance,
    DetectionRecord,
    ExistenceResult,
    Transform,
)
from irap_screening.modules.marking import analyse_marking
from irap_screening.segment import build_segments
from tests.factories import straight_route


def _build(sign_cls: str | None):
    frames, gps, imu = straight_route(length_m=200.0)
    route = align_route(frames, gps, imu)
    cfg = load_config()
    segments = build_segments(route, cfg)

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
    return route, segments, DetectionSet(sign=sign), cfg


def test_sign_is_recorded_with_distance_along_route():
    route, segments, detections, cfg = _build("Bend")
    results = analyse_marking(route, segments, detections, cfg)

    signs = [sign for result in results for sign in result.signs]
    assert len(signs) == 1

    # The sign sits on a mid-route frame, so a zero here would mean the distance
    # lookup silently fell through to its default rather than reading frame_dist_m.
    expected = float(route.frame_dist_m[len(route.frames) // 2])
    assert expected > 0.0
    assert signs[0].at_m == pytest.approx(expected, rel=1e-6)


def test_module_returns_existence_results_not_flags():
    """Giving this module a flag would invite folding it into the rating."""
    route, segments, detections, cfg = _build(None)
    results = analyse_marking(route, segments, detections, cfg)
    assert all(isinstance(r, ExistenceResult) for r in results)
    assert not any(hasattr(r, "flag") for r in results)
