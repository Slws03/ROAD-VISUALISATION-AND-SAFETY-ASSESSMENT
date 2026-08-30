"""Grade comes from accelerometer tilt, smoothed by a complementary filter."""

from datetime import timedelta

import numpy as np
import pytest

from irap_screening.config import load_config
from irap_screening.io.align import align_route
from irap_screening.io.schema import Flag, ImuSample
from irap_screening.modules.grade import analyse_grade, pitch_series
from irap_screening.segment import build_segments
from tests.factories import graded_route, straight_route


def _analyse(frames, gps, imu, cfg=None):
    cfg = cfg or load_config()
    route = align_route(frames, gps, imu)
    segments = build_segments(route, cfg)
    return analyse_grade(route, segments, cfg)


def test_level_road_reports_zero_grade():
    results = _analyse(*straight_route(length_m=500.0))
    assert results[0].metrics["grade_pct"] == pytest.approx(0.0, abs=0.5)
    assert results[0].flag is Flag.NORMAL


def test_known_grade_is_recovered():
    results = _analyse(*graded_route(grade_pct=8.0, length_m=300.0))
    assert results[0].metrics["grade_pct"] == pytest.approx(8.0, rel=0.05)


def test_grade_above_threshold_is_critical():
    results = _analyse(*graded_route(grade_pct=8.0, length_m=300.0))
    assert results[0].flag is Flag.CRITICAL


def test_grade_below_threshold_is_normal():
    results = _analyse(*graded_route(grade_pct=4.0, length_m=300.0))
    assert results[0].flag is Flag.NORMAL


def test_downgrade_is_flagged_the_same_as_upgrade():
    """Steepness matters, direction does not; the flag keys on magnitude."""
    results = _analyse(*graded_route(grade_pct=-8.0, length_m=300.0))
    assert results[0].flag is Flag.CRITICAL
    assert results[0].metrics["grade_pct"] == pytest.approx(-8.0, rel=0.05)


def test_pitch_series_has_one_value_per_imu_sample():
    frames, gps, imu = graded_route(grade_pct=8.0, length_m=300.0)
    route = align_route(frames, gps, imu)
    assert len(pitch_series(route, load_config())) == len(imu)


def test_threshold_is_configurable():
    cfg = load_config(overrides=["grade.threshold_pct=10.0"])
    results = _analyse(*graded_route(grade_pct=8.0, length_m=300.0), cfg=cfg)
    assert results[0].flag is Flag.NORMAL


def test_segment_aggregation_takes_the_steepest_sample():
    results = _analyse(*graded_route(grade_pct=8.0, length_m=300.0))
    for result in results:
        assert abs(result.metrics["grade_pct"]) >= 7.0


def test_gyro_term_actually_contributes():
    """A rotation the accelerometer cannot see must still move the estimate.

    Every factory route has gy == 0 and a constant accelerometer pitch, which
    makes the recurrence degenerate to the identity — so without this case a
    broken gyro term would be invisible to the whole suite.
    """
    frames, gps, _ = straight_route(length_m=200.0)
    t0 = frames[0].timestamp_utc
    imu = [
        ImuSample(
            timestamp_utc=t0 + timedelta(seconds=i / 50.0),
            ax=0.0,
            ay=0.0,
            az=9.81,          # accelerometer reports dead level throughout
            gx=0.0,
            gy=0.1,           # but the gyro reports a steady pitch-up rate
            gz=0.0,
        )
        for i in range(200)
    ]
    route = align_route(frames, gps, imu)
    pitch = pitch_series(route, load_config())

    # Steady state is alpha*gy*dt/(1-alpha) = 0.098 rad; anything above 0.01 rad
    # proves the gyro term is wired in. Dropping it would pin this at 0.0.
    assert pitch[-1] > 0.01


def test_segment_without_imu_samples_is_not_assessed():
    """An unevaluable segment must not be reported as level road."""
    frames, gps, imu = graded_route(grade_pct=8.0, length_m=300.0)
    route = align_route(frames, gps, imu)
    cfg = load_config()
    segments = build_segments(route, cfg)

    route.imu = []
    route.imu_dist_m = np.array([])
    for segment in segments:
        segment.imu_start = 0
        segment.imu_end = 0

    results = analyse_grade(route, segments, cfg)
    assert results
    assert all(result.flag is Flag.NOT_ASSESSED for result in results)
    assert all(result.notes for result in results)
