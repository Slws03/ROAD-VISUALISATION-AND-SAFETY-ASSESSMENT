"""Curvature is omega / v. Every test here pins one of its failure modes."""

import math

import pytest

from irap_screening.config import load_config
from irap_screening.io.align import align_route
from irap_screening.io.schema import Flag
from irap_screening.modules.curvature import analyse_curvature, estimate_gyro_bias
from irap_screening.segment import build_segments
from tests.factories import curved_route, straight_route


def _analyse(frames, gps, imu, cfg=None):
    cfg = cfg or load_config()
    route = align_route(frames, gps, imu)
    segments = build_segments(route, cfg)
    return analyse_curvature(route, segments, cfg), segments


def test_straight_road_is_normal_with_large_radius():
    results, _ = _analyse(*straight_route(length_m=500.0))
    assert all(r.flag is Flag.NORMAL for r in results)


def test_known_radius_is_recovered_within_five_percent():
    """A 312.5 m constant-radius curve must read back as roughly 312.5 m."""
    results, _ = _analyse(*curved_route(radius_m=312.5, arc_length_m=300.0))
    assert results, "expected at least one segment"
    assert results[0].metrics["radius_m"] == pytest.approx(312.5, rel=0.05)


def test_radius_below_threshold_is_critical():
    results, _ = _analyse(*curved_route(radius_m=312.5, arc_length_m=300.0))
    assert results[0].flag is Flag.CRITICAL


def test_radius_above_threshold_is_normal():
    results, _ = _analyse(*curved_route(radius_m=800.0, arc_length_m=300.0))
    assert results[0].flag is Flag.NORMAL


def test_curvature_needs_speed_not_just_gyro():
    """Same yaw rate at double the speed is double the radius.

    Integrating gyro alone would report the two as identical, which is exactly
    the error this assertion guards against.
    """
    slow, _ = _analyse(*curved_route(radius_m=300.0, speed_mps=10.0, arc_length_m=300.0))
    fast_frames, fast_gps, fast_imu = curved_route(
        radius_m=600.0, speed_mps=20.0, arc_length_m=600.0
    )
    fast, _ = _analyse(fast_frames, fast_gps, fast_imu)
    assert slow[0].metrics["yaw_rate_rad_s"] == pytest.approx(
        fast[0].metrics["yaw_rate_rad_s"], rel=0.05
    )
    assert fast[0].metrics["radius_m"] > 1.8 * slow[0].metrics["radius_m"]


def test_low_speed_segment_is_not_assessed_never_critical():
    """A slow crawl produces a tiny radius numerically; it must not be reported."""
    frames, gps, imu = curved_route(radius_m=300.0, speed_mps=2.0, arc_length_m=300.0)
    results, _ = _analyse(frames, gps, imu)
    assert results[0].flag is Flag.NOT_ASSESSED
    assert results[0].notes is not None
    assert "speed" in results[0].notes.lower()


def test_gyro_bias_is_estimated_from_stationary_samples():
    """The zero reference comes from a parked interval, not from the drive."""
    bias = math.radians(2.4)
    frames, gps, imu = curved_route(
        radius_m=800.0, arc_length_m=300.0, gyro_bias_rad_s=bias,
        stationary_prefix_s=3.0,
    )
    route = align_route(frames, gps, imu)
    assert estimate_gyro_bias(route, load_config()) == pytest.approx(bias, rel=0.02)


def test_gyro_bias_is_zero_without_a_stationary_interval():
    """No parked samples means no reference; report zero rather than guessing."""
    frames, gps, imu = straight_route(length_m=500.0)
    route = align_route(frames, gps, imu)
    assert estimate_gyro_bias(route, load_config()) == pytest.approx(0.0, abs=1e-6)


def test_injected_gyro_bias_is_removed():
    """2.4 deg/s of drift is the same size as the signal at the 400 m boundary."""
    bias = math.radians(2.4)
    frames, gps, imu = curved_route(
        radius_m=800.0, arc_length_m=300.0, gyro_bias_rad_s=bias,
        stationary_prefix_s=3.0,
    )
    uncorrected_radius = 16.7 / (16.7 / 800.0 + bias)
    assert uncorrected_radius < 400.0, "test setup must actually flip the verdict"

    results, _ = _analyse(frames, gps, imu)
    assert results[0].metrics["radius_m"] == pytest.approx(800.0, rel=0.10)
    assert results[0].flag is Flag.NORMAL


def test_stopped_samples_do_not_manufacture_a_hairpin():
    """A junction stop inside a straight segment must not read as a tight curve.

    Speed near zero divides into a radius near zero. Gating on the segment mean
    alone would let those samples through, so the gate is per sample.
    """
    frames, gps, imu = curved_route(
        radius_m=800.0, arc_length_m=300.0, stationary_prefix_s=3.0
    )
    results, _ = _analyse(frames, gps, imu)
    assert results[0].metrics["radius_m"] > 400.0
    assert results[0].flag is Flag.NORMAL


def test_threshold_is_configurable():
    cfg = load_config(overrides=["curvature.threshold_m=250"])
    results, _ = _analyse(*curved_route(radius_m=312.5, arc_length_m=300.0), cfg=cfg)
    assert results[0].flag is Flag.NORMAL


def test_segment_aggregation_takes_the_worst_radius():
    """max_risk for curvature means the smallest radius in the segment wins."""
    results, _ = _analyse(*curved_route(radius_m=312.5, arc_length_m=300.0))
    for result in results:
        assert result.metrics["radius_m"] <= 400.0


def test_slow_driving_is_not_mistaken_for_stationary():
    """A vehicle crawling through a curve still has a genuine yaw rate.

    Gating the bias estimate on v_min_mps would absorb that yaw rate into the
    bias, and since bias is estimated once per route and applied to every
    segment, a single junction turn would make straight road read as a hairpin.
    """
    frames, gps, imu = curved_route(radius_m=20.0, speed_mps=3.0, arc_length_m=60.0)
    route = align_route(frames, gps, imu)
    assert estimate_gyro_bias(route, load_config()) == pytest.approx(0.0, abs=1e-9)
