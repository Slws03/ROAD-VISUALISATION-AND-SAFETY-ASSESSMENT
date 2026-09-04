"""Curvature from gyroscope yaw rate and vehicle speed.

Curvature is kappa = omega / v. Integrating the gyro z axis alone yields heading
change, not curvature: the same yaw rate at 30 km/h and at 80 km/h corresponds to
radii that differ by nearly a factor of three.

Two guards are mandatory rather than optional:

- Low-speed gating. As v approaches zero, kappa diverges. Traffic lights and
  junction stops would otherwise manufacture extreme curvature.
- Bias correction. A 400 m radius at 60 km/h is omega = 0.042 rad/s, about
  2.4 deg/s, the same order as consumer phone gyro bias drift. Without removing
  the bias the threshold decision is dominated by drift.
"""

from __future__ import annotations

import numpy as np

from irap_screening.config import Settings
from irap_screening.io.align import AlignedRoute
from irap_screening.io.schema import Flag, ModuleResult, Segment

MODULE_NAME = "curvature"
_HUGE_RADIUS_M = 1.0e6


def estimate_gyro_bias(route: AlignedRoute, cfg: Settings) -> float:
    """Mean yaw rate over stationary samples, used as the zero reference.

    A sample counts as stationary when interpolated speed is below the stationary
    threshold and local accelerometer variance is small. When no stationary window
    exists, the bias is reported as zero and the caller proceeds uncorrected.
    """
    if not route.imu:
        return 0.0

    gz = np.array([sample.gz for sample in route.imu], dtype=float)
    accel = np.array(
        [[sample.ax, sample.ay, sample.az] for sample in route.imu], dtype=float
    )

    # Stationary, not merely slow. v_min_mps is the floor for curvature being
    # assessable; a vehicle below it is still moving and still turning. Folding a
    # crawling turn into the bias would corrupt every segment on the route, since
    # the bias is estimated once and applied everywhere.
    stationary = route.imu_speed_mps < cfg.curvature.stationary_speed_mps
    if not np.any(stationary):
        return 0.0

    variance = accel[stationary].var(axis=0).max()
    if variance > cfg.curvature.stationary_accel_var:
        return 0.0
    return float(gz[stationary].mean())


def irap_curvature_category(radius_m: float) -> int:
    """iRAP Curvature Risk Factor descriptive category, by radius band.

    Display only: independent of threshold_m, which drives the flag. >=800->1
    (straight/gently curving), 400-800->2 (moderate), 200-400->3 (sharp),
    <200->4 (very sharp).
    """
    if radius_m >= 800.0:
        return 1
    if radius_m >= 400.0:
        return 2
    if radius_m >= 200.0:
        return 3
    return 4


def analyse_curvature(
    route: AlignedRoute, segments: list[Segment], cfg: Settings
) -> list[ModuleResult]:
    """Produce one ModuleResult per segment, aggregating by worst (smallest) radius."""
    bias = estimate_gyro_bias(route, cfg)
    gz = np.array([sample.gz for sample in route.imu], dtype=float) - bias
    speed = route.imu_speed_mps

    results: list[ModuleResult] = []
    for segment in segments:
        lo, hi = segment.imu_start, segment.imu_end
        window_gz = gz[lo:hi]
        window_speed = speed[lo:hi]

        if window_gz.size == 0:
            results.append(
                ModuleResult(
                    module=MODULE_NAME,
                    segment_id=segment.segment_id,
                    flag=Flag.NOT_ASSESSED,
                    metrics={"gyro_bias_rad_s": bias},
                    notes="No IMU samples fall within this segment.",
                )
            )
            continue

        # Gate per sample, not per segment average. A segment that mixes a
        # junction stop with normal driving would otherwise keep the stopped
        # samples, and speed near zero yields a radius near zero: a fake hairpin
        # that survives into the report as critical.
        usable = window_speed >= cfg.curvature.v_min_mps
        mean_speed = float(window_speed[usable].mean()) if usable.any() else float(
            window_speed.mean()
        )

        if not usable.any():
            results.append(
                ModuleResult(
                    module=MODULE_NAME,
                    segment_id=segment.segment_id,
                    flag=Flag.NOT_ASSESSED,
                    metrics={
                        "mean_speed_mps": mean_speed,
                        "gyro_bias_rad_s": bias,
                    },
                    notes=(
                        f"Mean speed {mean_speed:.1f} m/s is below the minimum "
                        f"{cfg.curvature.v_min_mps} m/s; curvature is undefined at "
                        "low speed and is not reported."
                    ),
                )
            )
            continue

        usable_gz = window_gz[usable]
        usable_speed = window_speed[usable]

        # Radius per usable sample, then take the worst (smallest) as the segment
        # value, matching the iRAP rule that the highest-risk value codes the
        # segment.
        with np.errstate(divide="ignore", invalid="ignore"):
            radii = np.where(
                np.abs(usable_gz) > 1e-9,
                usable_speed / np.abs(usable_gz),
                _HUGE_RADIUS_M,
            )
        radii = np.clip(radii, 0.0, _HUGE_RADIUS_M)
        worst_index = int(np.argmin(radii))
        radius_m = float(radii[worst_index])

        flag = Flag.CRITICAL if radius_m < cfg.curvature.threshold_m else Flag.NORMAL
        results.append(
            ModuleResult(
                module=MODULE_NAME,
                segment_id=segment.segment_id,
                flag=flag,
                metrics={
                    "radius_m": radius_m,
                    "yaw_rate_rad_s": float(abs(usable_gz[worst_index])),
                    "mean_speed_mps": mean_speed,
                    "gyro_bias_rad_s": bias,
                    "irap_category": float(irap_curvature_category(radius_m)),
                },
            )
        )
    return results
