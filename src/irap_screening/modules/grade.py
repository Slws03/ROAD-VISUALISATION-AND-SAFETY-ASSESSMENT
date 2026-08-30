"""Longitudinal grade from accelerometer tilt with complementary filtering.

The accelerometer gives an absolute pitch reference but is noisy under vehicle
acceleration; the gyro is smooth but drifts. A complementary filter combines
them: the gyro carries the short term, the accelerometer anchors the long term.

Grade is reported signed, so uphill and downhill remain distinguishable in the
report, but the flag keys on magnitude because steepness is what carries risk.
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


def pitch_series(route: AlignedRoute, cfg: Settings) -> np.ndarray:
    """Complementary-filtered pitch in radians, one value per IMU sample."""
    if not route.imu:
        return np.zeros(0)

    ax = np.array([s.ax for s in route.imu], dtype=float)
    ay = np.array([s.ay for s in route.imu], dtype=float)
    az = np.array([s.az for s in route.imu], dtype=float)
    gy = np.array([s.gy for s in route.imu], dtype=float)
    times = np.array([s.timestamp_utc.timestamp() for s in route.imu], dtype=float)

    accel_pitch = _accel_pitch(ax, ay, az)
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
