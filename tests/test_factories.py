"""The factories are the ground truth every module test asserts against.

A silent geometry error here would make those downstream tests pass against a
wrong reference, so the factories get their own regression guard.
"""

import math

import pytest

from tests.factories import (
    EARTH_RADIUS_M,
    curved_route,
    graded_route,
    spiked_grade_route,
    straight_route,
)


def test_curved_route_gyro_recovers_the_requested_radius():
    """radius = speed / |gz| must return the radius that was asked for."""
    speed, radius = 16.7, 312.5
    _, _, imu = curved_route(radius_m=radius, speed_mps=speed, arc_length_m=300.0)
    assert speed / abs(imu[0].gz) == pytest.approx(radius, rel=1e-9)


def test_curved_route_traces_a_circle_of_that_radius():
    """Every fix must sit on the circle centred one radius to the east."""
    radius = 312.5
    _, gps, _ = curved_route(radius_m=radius, arc_length_m=300.0)
    lat0, lon0 = gps[0].lat, gps[0].lon
    for fix in gps:
        north = math.radians(fix.lat - lat0) * EARTH_RADIUS_M
        east = math.radians(fix.lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
        distance_from_centre = math.hypot(north, east - radius)
        assert distance_from_centre == pytest.approx(radius, rel=1e-6)


def test_graded_route_accelerometer_recovers_the_requested_grade():
    """atan2(ax, sqrt(ay^2+az^2)) then tan()*100 must return the grade asked for."""
    grade = 8.0
    _, _, imu = graded_route(grade_pct=grade, length_m=300.0)
    sample = imu[0]
    pitch = math.atan2(sample.ax, math.hypot(sample.ay, sample.az))
    assert math.tan(pitch) * 100.0 == pytest.approx(grade, rel=1e-9)


def test_spiked_grade_route_recovers_background_and_spike_grades():
    """Samples outside the spike window read the background grade; inside, the spike."""
    _, _, imu = spiked_grade_route(
        background_pct=0.0,
        spike_pct=20.0,
        spike_start_m=40.0,
        spike_length_m=3.0,
        speed_mps=16.7,
    )

    def grade_of(sample) -> float:
        pitch = math.atan2(sample.ax, math.hypot(sample.ay, sample.az))
        return math.tan(pitch) * 100.0

    # 40 m / 16.7 m/s ~= 2.40 s; well before that, still background.
    assert grade_of(imu[0]) == pytest.approx(0.0, abs=1e-9)
    # 41.5 m / 16.7 m/s ~= 2.49 s, inside [40, 43) m.
    spike_sample = imu[int(round(41.5 / 16.7 * 50.0))]
    assert grade_of(spike_sample) == pytest.approx(20.0, rel=1e-9)
    # Well past the 3 m spike, back to background.
    assert grade_of(imu[-1]) == pytest.approx(0.0, abs=1e-9)


def test_straight_route_has_no_yaw_and_no_pitch():
    _, _, imu = straight_route(length_m=300.0)
    assert all(sample.gz == 0.0 for sample in imu)
    assert all(sample.ax == 0.0 for sample in imu)


def test_stationary_prefix_never_overlaps_the_driving_section():
    """Two IMU rows at one timestamp with different gz would corrupt interpolation."""
    for prefix_s in (0.0, 2.0, 2.5, 3.7):
        _, _, imu = curved_route(
            radius_m=800.0, arc_length_m=200.0, stationary_prefix_s=prefix_s
        )
        stamps = [sample.timestamp_utc for sample in imu]
        assert len(stamps) == len(set(stamps)), f"duplicate timestamp at prefix={prefix_s}"
        assert stamps == sorted(stamps), f"timestamps out of order at prefix={prefix_s}"
