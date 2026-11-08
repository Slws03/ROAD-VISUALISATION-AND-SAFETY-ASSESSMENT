"""Curvature is omega / v. Every test here pins one of its failure modes."""

import math
from datetime import timedelta

import numpy as np
import pytest

from irap_screening.config import load_config
from irap_screening.io.align import align_route
from irap_screening.io.schema import Flag, ImuSample
from irap_screening.modules.curvature import (
    analyse_curvature,
    estimate_gyro_bias,
    irap_curvature_category,
)
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


def test_irap_curvature_category_boundaries():
    """iRAP Curvature Risk Factor: >=800->1, 400-800->2, 200-400->3, <200->4."""
    assert irap_curvature_category(1000.0) == 1
    assert irap_curvature_category(800.0) == 1
    assert irap_curvature_category(799.9) == 2
    assert irap_curvature_category(400.0) == 2
    assert irap_curvature_category(399.9) == 3
    assert irap_curvature_category(200.0) == 3
    assert irap_curvature_category(199.9) == 4
    assert irap_curvature_category(50.0) == 4


def test_analysis_reports_the_irap_category():
    results, _ = _analyse(*curved_route(radius_m=312.5, arc_length_m=300.0))
    assert results[0].metrics["irap_category"] == 3


def test_radius_below_threshold_is_critical():
    results, _ = _analyse(*curved_route(radius_m=312.5, arc_length_m=300.0))
    assert results[0].flag is Flag.CRITICAL


def test_radius_above_threshold_is_normal():
    # Not exactly 800: a windowed estimate is accurate to a fraction of a
    # percent, so a fixture sitting on the threshold would flip on rounding.
    results, _ = _analyse(*curved_route(radius_m=900.0, arc_length_m=300.0))
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
        radius_m=900.0, arc_length_m=300.0, gyro_bias_rad_s=bias,
        stationary_prefix_s=3.0,
    )
    uncorrected_radius = 16.7 / (16.7 / 900.0 + bias)
    assert uncorrected_radius < 400.0, "test setup must actually flip the verdict"

    results, _ = _analyse(frames, gps, imu)
    assert results[0].metrics["radius_m"] == pytest.approx(900.0, rel=0.10)
    assert results[0].flag is Flag.NORMAL


def test_stopped_samples_do_not_manufacture_a_hairpin():
    """A junction stop inside a straight segment must not read as a tight curve.

    Speed near zero divides into a radius near zero. Gating on the segment mean
    alone would let those samples through, so the gate is per sample.
    """
    frames, gps, imu = curved_route(
        radius_m=900.0, arc_length_m=300.0, stationary_prefix_s=3.0
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


def test_gyro_noise_on_a_straight_road_is_not_a_curve():
    """Phone gyro noise (0.06 rad/s, as on the Samarahan routes) is not geometry.

    Per sample, v / |noise| reads as a ~100 m curve on every segment; over a
    window of road the noise averages out and the road reads straight.
    """
    frames, gps, imu = straight_route(length_m=500.0)
    noise = np.random.default_rng(0).normal(0.0, 0.06, len(imu))
    imu = [s.model_copy(update={"gz": s.gz + n}) for s, n in zip(imu, noise, strict=True)]
    results, _ = _analyse(frames, gps, imu)
    assert all(r.flag is Flag.NORMAL for r in results)


def test_a_window_longer_than_the_route_is_not_assessed():
    """No window of road fits: report not_assessed, never crash or guess."""
    cfg = load_config(overrides=["curvature.window_m=1000"])
    results, _ = _analyse(*straight_route(length_m=500.0), cfg=cfg)
    assert results and all(r.flag is Flag.NOT_ASSESSED for r in results)
    assert "window" in results[0].notes


def test_a_curve_in_the_next_segment_does_not_code_this_one():
    """The road turns 2 m into the second segment; the first stays straight.

    A window centred 10 m before the boundary would reach 15 m into the curve
    and read about 770 m, below the threshold. Only windows lying wholly
    inside a segment may code it.
    """
    frames, gps, imu = curved_route(radius_m=200.0, arc_length_m=300.0)
    speed = 16.7
    t0 = imu[0].timestamp_utc
    imu = [
        s.model_copy(update={"gz": 0.0})
        if (s.timestamp_utc - t0).total_seconds() * speed < 102.0 else s
        for s in imu
    ]
    results, _ = _analyse(frames, gps, imu)
    assert results[0].flag is Flag.NORMAL
    assert results[1].flag is Flag.CRITICAL


def _with_bearing(gps, bearing_at):
    t0 = gps[0].timestamp_utc
    return [
        fix.model_copy(update={"bearing_deg": bearing_at((fix.timestamp_utc - t0).total_seconds())})
        for fix in gps
    ]


def test_gps_bearing_that_agrees_does_not_ask_for_review():
    frames, gps, imu = curved_route(radius_m=312.5, arc_length_m=300.0)
    gps = _with_bearing(gps, lambda t: math.degrees(16.7 / 312.5 * t))
    results, _ = _analyse(frames, gps, imu)
    assert results[0].metrics["gps_radius_m"] == pytest.approx(312.5, rel=0.10)
    assert results[0].metrics["needs_review"] == 0.0


def test_gyro_and_gps_disagreeing_asks_for_review_but_the_gyro_decides():
    """The gyro turns, the GPS course holds north: flagged, still coded by the gyro."""
    frames, gps, imu = straight_route(length_m=500.0)
    imu = [s.model_copy(update={"gz": 16.7 / 300.0}) for s in imu]
    gps = _with_bearing(gps, lambda t: 0.0)
    results, _ = _analyse(frames, gps, imu)
    assert results[0].flag is Flag.CRITICAL
    assert results[0].metrics["gps_radius_m"] > 800.0
    assert results[0].metrics["needs_review"] == 1.0


def test_no_gps_bearing_means_no_cross_check():
    """KITTI's GPS carries no course: report the gyro alone, not a false agreement."""
    results, _ = _analyse(*curved_route(radius_m=312.5, arc_length_m=300.0))
    assert "gps_radius_m" not in results[0].metrics
    assert "needs_review" not in results[0].metrics


def test_yaw_read_while_stopped_is_not_a_turn():
    """A straight road with a minute parked mid-way and the gyro drifting all the while.

    The accelerometer shakes during the stop, as a handled phone does, so the bias
    estimator declines and the drift stays in the signal.
    """
    frames, gps, imu = straight_route(length_m=500.0)
    middle = gps[len(gps) // 2]
    stop = timedelta(seconds=60)

    def shifted(items):
        return [
            item.model_copy(update={"timestamp_utc": item.timestamp_utc + stop})
            if item.timestamp_utc > middle.timestamp_utc else item
            for item in items
        ]

    parked_gps = [
        middle.model_copy(update={
            "timestamp_utc": middle.timestamp_utc + timedelta(seconds=s), "speed_mps": 0.0,
        })
        for s in range(1, 60)
    ]
    parked_imu = [
        ImuSample(
            timestamp_utc=middle.timestamp_utc + timedelta(seconds=s / 10.0),
            ax=1.0 if s % 2 else -1.0, ay=0.0, az=9.81, gx=0.0, gy=0.0, gz=0.02,
        )
        for s in range(10, 590)
    ]
    by_time = lambda item: item.timestamp_utc  # noqa: E731
    gps = sorted(shifted(gps) + parked_gps, key=by_time)
    imu = sorted(shifted(imu) + parked_imu, key=by_time)
    results, _ = _analyse(frames, gps, imu)
    assert all(r.flag is not Flag.CRITICAL for r in results)
