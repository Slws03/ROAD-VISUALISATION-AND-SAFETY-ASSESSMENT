"""Distance and speed are shared foundations; every module depends on them."""

import numpy as np
import pytest

from irap_screening.io.align import align_route, cumulative_distance, haversine_m, resolve_speed
from tests.factories import straight_route


def test_haversine_matches_known_distance():
    """One degree of latitude is about 111.2 km."""
    d = haversine_m(0.0, 0.0, 1.0, 0.0)
    assert d == pytest.approx(111_195.0, rel=1e-3)


def test_haversine_is_zero_for_identical_points():
    assert haversine_m(1.55, 110.35, 1.55, 110.35) == pytest.approx(0.0, abs=1e-9)


def test_cumulative_distance_starts_at_zero_and_is_monotonic():
    _, gps, _ = straight_route(length_m=500.0)
    dist = cumulative_distance(gps)
    assert dist[0] == pytest.approx(0.0)
    assert np.all(np.diff(dist) >= 0.0)


def test_cumulative_distance_recovers_route_length():
    _, gps, _ = straight_route(length_m=500.0, speed_mps=16.7)
    dist = cumulative_distance(gps)
    assert dist[-1] == pytest.approx(500.0, rel=0.02)


def test_resolve_speed_prefers_doppler_field():
    _, gps, _ = straight_route(speed_mps=16.7, include_speed=True)
    speeds, source = resolve_speed(gps)
    assert source == "doppler"
    assert speeds[0] == pytest.approx(16.7)


def test_resolve_speed_falls_back_to_differencing():
    _, gps, _ = straight_route(speed_mps=16.7, include_speed=False)
    speeds, source = resolve_speed(gps)
    assert source == "differenced"
    assert speeds[1:-1] == pytest.approx(16.7, rel=0.05)


def test_interpolated_speed_covers_every_imu_sample():
    frames, gps, imu = straight_route(length_m=500.0)
    route = align_route(frames, gps, imu)
    assert len(route.imu_speed_mps) == len(imu)
    assert np.all(route.imu_speed_mps > 0.0)


def test_frame_distance_is_assigned_for_every_frame():
    frames, gps, imu = straight_route(length_m=500.0)
    route = align_route(frames, gps, imu)
    assert len(route.frame_dist_m) == len(frames)
    assert route.frame_dist_m[0] == pytest.approx(0.0, abs=1.0)


def test_speed_source_is_recorded_on_the_route():
    frames, gps, imu = straight_route(include_speed=False)
    route = align_route(frames, gps, imu)
    assert route.speed_source == "differenced"
