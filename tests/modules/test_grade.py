"""Grade comes from accelerometer tilt, smoothed by a complementary filter."""

from datetime import timedelta

import numpy as np
import pytest

from irap_screening.config import load_config
from irap_screening.io.align import align_route
from irap_screening.io.schema import Flag, ImuSample
from irap_screening.modules.grade import (
    _sustained_steepest_grade,
    analyse_grade,
    longitudinal_accel,
    pitch_series,
)
from irap_screening.segment import build_segments
from tests.factories import braking_route, graded_route, spiked_grade_route, straight_route


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


def test_sustained_steepest_grade_ignores_a_short_spike():
    """Coding Manual p.54 (attribute AO): a grade shorter than 10 m (one tenth
    of the 100 m segment) must not be recorded as the segment's grade -- a
    culvert or speed bump must not read as the segment's slope. Works on
    synthetic (grade_pct, dist_m) arrays directly, bypassing pitch_series's
    complementary filter, whose own slow convergence would otherwise be a
    second, unrelated reason a short spike reads low.
    """
    dist_m = np.linspace(0.0, 20.0, 61)
    grade_pct = np.zeros_like(dist_m)
    spike = (dist_m >= 9.5) & (dist_m < 10.5)  # a 1 m wide spike
    grade_pct[spike] = 20.0

    value, _ = _sustained_steepest_grade(grade_pct, dist_m, min_length_m=10.0)
    assert value == pytest.approx(0.0, abs=1e-9)


def test_sustained_steepest_grade_picks_up_a_run_at_least_as_long_as_the_floor():
    dist_m = np.linspace(0.0, 30.0, 91)
    grade_pct = np.zeros_like(dist_m)
    run = (dist_m >= 8.0) & (dist_m < 20.0)  # a 12 m run
    grade_pct[run] = 15.0

    value, _ = _sustained_steepest_grade(grade_pct, dist_m, min_length_m=10.0)
    assert value == pytest.approx(15.0, abs=1e-9)


def test_sustained_steepest_grade_preserves_sign():
    dist_m = np.linspace(0.0, 30.0, 91)
    grade_pct = np.zeros_like(dist_m)
    run = (dist_m >= 8.0) & (dist_m < 20.0)
    grade_pct[run] = -15.0

    value, _ = _sustained_steepest_grade(grade_pct, dist_m, min_length_m=10.0)
    assert value == pytest.approx(-15.0, abs=1e-9)


def test_sustained_steepest_grade_falls_back_when_the_segment_itself_is_too_short():
    """A segment whose IMU span never reaches min_length_m has no qualifying
    window at all; fall back to the single worst sample rather than reporting
    nothing.
    """
    dist_m = np.array([0.0, 1.0, 2.0])
    grade_pct = np.array([1.0, 9.0, 2.0])

    value, index = _sustained_steepest_grade(grade_pct, dist_m, min_length_m=10.0)
    assert value == pytest.approx(9.0)
    assert index == 1


def test_short_spike_reads_far_flatter_than_a_sustained_grade_end_to_end():
    """Same peak steepness, same route, only the run length differs -- and
    both go through the same slow complementary filter, so any difference in
    the reported grade is attributable to the 10 m duration floor, not to
    filter dynamics.
    """
    short = _analyse(
        *spiked_grade_route(
            background_pct=0.0, spike_pct=20.0, spike_start_m=40.0,
            spike_length_m=2.0, length_m=150.0,
        )
    )
    long = _analyse(
        *spiked_grade_route(
            background_pct=0.0, spike_pct=20.0, spike_start_m=40.0,
            spike_length_m=40.0, length_m=150.0,
        )
    )
    assert abs(short[0].metrics["grade_pct"]) < abs(long[0].metrics["grade_pct"]) / 2


def test_gyro_term_actually_contributes():
    """A rotation the accelerometer cannot see must still move the estimate.

    Every factory route has gy == 0 and a constant accelerometer pitch, which
    makes the recurrence degenerate to the identity -- so without this case a
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


def test_longitudinal_accel_recovers_a_steady_deceleration():
    frames, gps, imu = braking_route(v_start=28.0, v_end=8.0, length_m=300.0)
    route = align_route(frames, gps, imu)
    duration = 2.0 * 300.0 / (28.0 + 8.0)

    accel = longitudinal_accel(route)

    assert len(accel) == len(imu)
    assert np.median(accel) == pytest.approx((8.0 - 28.0) / duration, rel=0.05)


def test_braking_on_a_level_road_is_not_a_grade():
    """The defect KITTI exposed: an accelerometer cannot tell braking from a slope.

    Its longitudinal channel carries the vehicle's own acceleration on top of the
    gravity projection, and a complementary filter does not separate them --
    sustained braking sits inside the band the accelerometer anchors. Against
    KITTI's INS ground truth this produced grades of -12 percent on a road whose
    true maximum was about 5 percent, flagging five segments critical that were
    not.
    """
    results = _analyse(*braking_route(grade_pct=0.0, v_start=28.0, v_end=8.0))

    for result in results:
        assert result.metrics["grade_pct"] == pytest.approx(0.0, abs=1.0)
        assert result.flag is Flag.NORMAL


def test_the_uncorrected_reading_would_have_flagged_that_road_critical():
    """Pins that the previous test is not passing for a trivial reason.

    Without the correction the same accelerometer reading yields a grade well
    past the threshold, so the test above is checking real work.
    """
    _, _, imu = braking_route(grade_pct=0.0, v_start=28.0, v_end=8.0)
    sample = imu[len(imu) // 2]

    uncorrected_pct = (
        np.tan(np.arctan2(-sample.ax, np.hypot(sample.ay, sample.az))) * 100.0
    )

    assert abs(uncorrected_pct) > load_config().grade.threshold_pct


def test_a_real_grade_survives_braking_on_top_of_it():
    """Correcting for the vehicle must not flatten a slope that is really there."""
    results = _analyse(*braking_route(grade_pct=8.0, v_start=28.0, v_end=8.0))

    assert results[0].metrics["grade_pct"] == pytest.approx(8.0, rel=0.05)
    assert results[0].flag is Flag.CRITICAL


def test_non_increasing_imu_timestamps_are_rejected():
    """Differentiating speed needs strictly increasing time.

    A duplicate timestamp divides by zero, and the NaN that follows compares
    false against every threshold -- so the segment would report normal no
    matter how steep it was.
    """
    frames, gps, imu = straight_route(length_m=200.0)
    imu[5] = ImuSample(
        timestamp_utc=imu[4].timestamp_utc,
        ax=imu[5].ax, ay=imu[5].ay, az=imu[5].az,
        gx=imu[5].gx, gy=imu[5].gy, gz=imu[5].gz,
    )
    route = align_route(frames, gps, imu)

    with pytest.raises(ValueError, match="strictly increasing"):
        pitch_series(route, load_config())
