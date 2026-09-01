"""Grade from accelerometer tilt, corrected for the vehicle's own acceleration.

An accelerometer measures specific force, so its longitudinal channel carries
the vehicle's braking and acceleration on top of the gravity projection. On a
level road the two are indistinguishable from the accelerometer alone: braking
at 1.2 m/s^2 reads exactly like a 12 percent downgrade. A complementary filter
does not separate them either -- sustained braking lasts seconds, which is
inside the band the accelerometer anchors, not the band the gyro carries.

The correction subtracts the longitudinal acceleration derived from recorded
speed before taking the gravity direction. Measured against KITTI drive
2011_10_03_0042, whose INS supplies ground-truth pitch, this took the
accelerometer-only error from 3.31 degrees RMSE to 0.38 degrees, and removed
five spurious critical segments on a road whose true maximum grade was about
5 percent.

Differentiating speed amplifies its noise. KITTI's recorded velocity is clean;
a phone's 1 Hz GPS speed is not, and this may need smoothing before provider
data is trusted here.
"""

from __future__ import annotations

import numpy as np

from irap_screening.config import Settings
from irap_screening.io.align import AlignedRoute
from irap_screening.io.schema import Flag, ModuleResult, Segment

MODULE_NAME = "grade"


def _accel_pitch(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    """Pitch in radians from the gravity direction, positive nose-up."""
    return np.arctan2(-ax, np.sqrt(ay**2 + az**2))


def longitudinal_accel(route: AlignedRoute) -> np.ndarray:
    """Vehicle acceleration along its own axis, from recorded speed.

    This is the term the accelerometer cannot separate from a slope, so it is
    computed from a source that does not confuse the two.
    """
    times = np.array([sample.timestamp_utc.timestamp() for sample in route.imu])
    if times.size < 2:
        return np.zeros(times.size)
    if np.any(np.diff(times) <= 0.0):
        raise ValueError(
            "IMU timestamps must be strictly increasing to differentiate speed; "
            "a repeated timestamp divides by zero, and the NaN that follows "
            "compares false against every threshold, so the segment would report "
            "normal however steep it is"
        )
    return np.gradient(route.imu_speed_mps, times)


def pitch_series(route: AlignedRoute, cfg: Settings) -> np.ndarray:
    """Complementary-filtered pitch in radians, one value per IMU sample.

    Note: alpha is applied as a fixed per-step weight, so the filter's effective
    time constant scales with the IMU sample interval. This is correct for the
    regular sampling this pipeline assumes; under irregular sampling alpha would
    need deriving per step from a time constant and the actual dt.
    """
    if not route.imu:
        return np.zeros(0)

    ax = np.array([s.ax for s in route.imu], dtype=float)
    ay = np.array([s.ay for s in route.imu], dtype=float)
    az = np.array([s.az for s in route.imu], dtype=float)
    gy = np.array([s.gy for s in route.imu], dtype=float)
    times = np.array([s.timestamp_utc.timestamp() for s in route.imu], dtype=float)

    # Remove the vehicle's own longitudinal acceleration, which the sensor
    # reports in the same channel as the gravity projection.
    accel_pitch = _accel_pitch(ax - longitudinal_accel(route), ay, az)
    alpha = cfg.grade.comp_filter_alpha

    fused = np.empty_like(accel_pitch)
    fused[0] = accel_pitch[0]
    for i in range(1, len(fused)):
        dt = times[i] - times[i - 1]
        predicted = fused[i - 1] + gy[i] * dt
        fused[i] = alpha * predicted + (1.0 - alpha) * accel_pitch[i]
    return fused


def analyse_grade(
    route: AlignedRoute, segments: list[Segment], cfg: Settings
) -> list[ModuleResult]:
    """One ModuleResult per segment, aggregating by steepest absolute grade."""
    pitch = pitch_series(route, cfg)
    grade_pct = np.tan(pitch) * 100.0

    results: list[ModuleResult] = []
    for segment in segments:
        window = grade_pct[segment.imu_start : segment.imu_end]
        if window.size == 0:
            results.append(
                ModuleResult(
                    module=MODULE_NAME,
                    segment_id=segment.segment_id,
                    flag=Flag.NOT_ASSESSED,
                    notes="No IMU samples fall within this segment.",
                )
            )
            continue

        steepest_index = int(np.argmax(np.abs(window)))
        steepest = float(window[steepest_index])
        flag = (
            Flag.CRITICAL
            if abs(steepest) >= cfg.grade.threshold_pct
            else Flag.NORMAL
        )
        results.append(
            ModuleResult(
                module=MODULE_NAME,
                segment_id=segment.segment_id,
                flag=flag,
                metrics={
                    "grade_pct": steepest,
                    "pitch_deg": float(
                        np.degrees(pitch[segment.imu_start + steepest_index])
                    ),
                },
            )
        )
    return results
