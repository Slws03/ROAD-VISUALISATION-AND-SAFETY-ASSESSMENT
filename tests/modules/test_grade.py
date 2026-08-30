"""Grade comes from accelerometer tilt, smoothed by a complementary filter."""

import pytest

from irap_screening.config import load_config
from irap_screening.io.align import align_route
from irap_screening.io.schema import Flag
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
