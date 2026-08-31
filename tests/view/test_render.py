"""Evidence rendering: detections drawn back onto the frame they came from.

The point of the viewer is to check a flag against the pixels that produced it,
so a drawing that lands in the wrong place is worse than no drawing at all -- it
would confirm a detection that never happened there. These tests assert where
the ink actually goes, not merely that the image changed.
"""

import numpy as np
import pytest

from irap_screening.io.schema import (
    BoxInstance,
    DetectionRecord,
    LaneInstance,
    Transform,
)
from irap_screening.view.render import (
    LANE_COLOR,
    SIGN_COLOR,
    render_evidence,
    scale_factor,
)

SOURCE_WH = (1280, 720)


def _blank(width=1280, height=720):
    return np.zeros((height, width, 3), dtype=np.uint8)


def _transform():
    return Transform(
        source_wh=SOURCE_WH, crop_top=0.0, crop_bottom=0.0, model_wh=(800, 320)
    )


def _lane_record(points):
    return DetectionRecord(
        frame_id="f_000000",
        image_path="frames/f_000000.jpg",
        sha256="0" * 16,
        type="lane",
        transform=_transform(),
        instances=[LaneInstance(id="f_000000#lane#0", points=points, conf=0.9)],
    )


def _sign_record(bbox, cls="Bend", conf=0.93):
    return DetectionRecord(
        frame_id="f_000000",
        image_path="frames/f_000000.jpg",
        sha256="0" * 16,
        type="sign",
        transform=_transform(),
        instances=[BoxInstance(id="f_000000#sign#0", bbox=bbox, conf=conf, cls=cls)],
    )


def _has_colour(image, colour, tolerance=40):
    distance = np.abs(image.astype(int) - np.array(colour, dtype=int)).sum(axis=2)
    return distance < tolerance


def test_scale_factor_never_upscales():
    """A frame narrower than the target stays at its own size.

    Blowing a small frame up wastes bytes and invents detail that is not there.
    """
    assert scale_factor((1280, 720), 640) == pytest.approx(0.5)
    assert scale_factor((320, 240), 640) == 1.0


def test_output_is_scaled_to_the_target_width():
    rendered = render_evidence(_blank(), None, None, target_width=640)

    assert rendered.shape[1] == 640
    assert rendered.shape[0] == 360


def test_lane_points_land_at_the_scaled_position():
    """The load-bearing assertion: ink at source (x, y) must appear at (x/2, y/2)."""
    points = [(400.0, 300.0), (400.0, 500.0)]
    rendered = render_evidence(_blank(), _lane_record(points), None, target_width=640)

    hits = _has_colour(rendered, LANE_COLOR)
    assert hits[200, 200], "lane should pass through the scaled midpoint (200, 200)"
    assert hits[150, 200], "and through (200, 150)"
    assert not hits[200, 400], "nothing should be drawn far from the lane"


def test_sign_box_lands_at_the_scaled_position():
    rendered = render_evidence(
        _blank(), None, _sign_record((600.0, 200.0, 700.0, 300.0)), target_width=640
    )

    hits = _has_colour(rendered, SIGN_COLOR)
    # Box corners at source (600, 200)-(700, 300) scale to (300, 100)-(350, 150).
    assert hits[100, 300], "top-left corner"
    assert hits[150, 350], "bottom-right corner"
    assert not hits[125, 325], "the box interior must stay transparent"


def test_off_frame_lane_points_are_skipped_not_clamped():
    """CLRerNet extends curved lanes past the edge; clamping would draw a lie.

    A clamped point would put a green line along the image border where no
    marking was ever detected.
    """
    rendered = render_evidence(
        _blank(), _lane_record([(-200.0, 300.0), (-100.0, 500.0)]), None, target_width=640
    )

    assert not _has_colour(rendered, LANE_COLOR).any()


def test_sign_label_carries_class_and_confidence():
    """The class string is what gap-check matches on, so it must be visible."""
    rendered = render_evidence(
        _blank(), None, _sign_record((600.0, 200.0, 700.0, 300.0), cls="Chevron"), 640
    )

    # Label text is drawn above the box; assert ink exists there rather than
    # trying to OCR it.
    band = rendered[80:100, 300:640]
    assert band.any(), "a label should be drawn above the box"


def test_missing_records_render_a_clean_frame():
    """A frame with no detections is a real state and must still be viewable."""
    rendered = render_evidence(_blank(), None, None, target_width=640)

    assert rendered.shape == (360, 640, 3)
    assert not _has_colour(rendered, LANE_COLOR).any()
    assert not _has_colour(rendered, SIGN_COLOR).any()


def test_a_lane_with_a_single_point_draws_a_marker_not_nothing():
    """CLRerNet can return a one-point lane; silently dropping it hides evidence."""
    rendered = render_evidence(_blank(), _lane_record([(400.0, 300.0)]), None, 640)

    assert _has_colour(rendered, LANE_COLOR).any()
