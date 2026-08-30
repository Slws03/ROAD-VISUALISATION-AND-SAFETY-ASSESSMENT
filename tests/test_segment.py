"""Segments are the shared coordinate system; every module keys results by them."""

import numpy as np
import pytest

from irap_screening.config import load_config
from irap_screening.io.align import align_route
from irap_screening.segment import build_segments
from tests.factories import straight_route


def test_five_hundred_metre_route_yields_five_segments():
    frames, gps, imu = straight_route(length_m=500.0)
    route = align_route(frames, gps, imu)
    segments = build_segments(route, load_config())
    assert len(segments) == 5


def test_segment_ids_are_zero_padded_and_ordered():
    frames, gps, imu = straight_route(length_m=500.0)
    route = align_route(frames, gps, imu)
    segments = build_segments(route, load_config(), route_id="R07")
    assert segments[0].segment_id == "R07-0000"
    assert segments[4].segment_id == "R07-0004"


def test_segment_boundaries_are_contiguous_and_100m_long():
    frames, gps, imu = straight_route(length_m=500.0)
    route = align_route(frames, gps, imu)
    segments = build_segments(route, load_config())
    for segment in segments:
        assert segment.end_m - segment.start_m == pytest.approx(100.0)
    for earlier, later in zip(segments, segments[1:], strict=False):
        assert earlier.end_m == pytest.approx(later.start_m)


def test_every_frame_is_assigned_to_exactly_one_segment():
    frames, gps, imu = straight_route(length_m=500.0)
    route = align_route(frames, gps, imu)
    segments = build_segments(route, load_config())
    assigned = [fid for segment in segments for fid in segment.frame_ids]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) <= {frame.frame_id for frame in frames}


def test_imu_index_ranges_are_non_overlapping_and_cover_the_route():
    frames, gps, imu = straight_route(length_m=500.0)
    route = align_route(frames, gps, imu)
    segments = build_segments(route, load_config())
    for earlier, later in zip(segments, segments[1:], strict=False):
        assert earlier.imu_end <= later.imu_start
    assert segments[0].imu_start == 0


def test_trailing_partial_segment_is_dropped():
    """A 250 m route yields two full segments; the leftover 50 m is not a coding unit."""
    frames, gps, imu = straight_route(length_m=250.0)
    route = align_route(frames, gps, imu)
    segments = build_segments(route, load_config())
    assert len(segments) == 2


def test_custom_segment_length_is_honoured():
    frames, gps, imu = straight_route(length_m=500.0)
    route = align_route(frames, gps, imu)
    cfg = load_config(overrides=["segment.length_m=250"])
    segments = build_segments(route, cfg)
    assert len(segments) == 2


def test_segment_without_imu_samples_keeps_ranges_monotonic():
    """An IMU dropout must not reset a mid-route segment's range to zero.

    Real logs drop samples. A gap segment reporting (0, 0) would sit below the
    previous segment's imu_end, breaking the ordering the test above asserts.
    """
    frames, gps, imu = straight_route(length_m=500.0)
    route = align_route(frames, gps, imu)

    # Drop every IMU sample falling inside the third segment (200-300 m).
    kept = [
        (sample, distance)
        for sample, distance in zip(route.imu, route.imu_dist_m, strict=False)
        if not (200.0 <= distance < 300.0)
    ]
    route.imu = [sample for sample, _ in kept]
    route.imu_dist_m = np.array([distance for _, distance in kept])
    route.imu_speed_mps = np.full(len(route.imu), 16.7)

    segments = build_segments(route, load_config())

    gap = segments[2]
    assert gap.imu_start == gap.imu_end, "the gap segment must yield an empty range"
    for earlier, later in zip(segments, segments[1:], strict=False):
        assert earlier.imu_end <= later.imu_start
