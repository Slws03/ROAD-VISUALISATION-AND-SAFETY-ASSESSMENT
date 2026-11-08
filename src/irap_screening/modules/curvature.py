"""Curvature from gyroscope yaw rate and vehicle speed.

Curvature is kappa = omega / v. Integrating the gyro z axis alone yields heading
change, not curvature: the same yaw rate at 30 km/h and at 80 km/h corresponds to
radii that differ by nearly a factor of three.

Three guards are mandatory rather than optional:

- Low-speed gating. As v approaches zero, kappa diverges. Traffic lights and
  junction stops would otherwise manufacture extreme curvature.
- Bias correction. A 400 m radius at 60 km/h is omega = 0.042 rad/s, about
  2.4 deg/s, the same order as consumer phone gyro bias drift. Without removing
  the bias the threshold decision is dominated by drift.
- A distance window. iRAP codes the road's geometry (curve radius), not the
  vehicle's instantaneous path (Tsai et al. 2021, GDOT 19-17, keep the two
  apart). v / |omega| per sample is the latter: on a phone gyro its noise
  alone reads straight road as a 100 m curve, and on any gyro a steering
  correction reads as a bend. Curvature is therefore the heading change over
  window_m of road divided by window_m (deflection angle over arc length),
  with heading integrated from the bias-corrected gyro.

A segment is coded by the sharpest window lying wholly inside it: the
sharpest window_m of road within the 100 m, as iRAP codes the most severe
curve in a segment and as its grade rule takes the steepest gradient held
over >= 10 m. A window centred near a segment's end would otherwise reach
into the next segment and code this one with its neighbour's curve, which
on the Samarahan routes did so to about 12% of segments, equally with gyro,
GPS and OSM curvature (D:/FYP/curvature_review/window_spill.md). Against
OSM centreline curvature under the same rule, 83% of segment flags agree.

Where the route carries a GPS bearing, the same window and rule are applied
to it as an independent cross-check. The gyro still decides the flag; a
segment where the two disagree on it is marked needs_review, for a person to
look at. The two agreed with each other far more often than either with OSM,
and where they disagreed neither was reliably right
(D:/FYP/curvature_review/osm_three_way.md).
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


def radius_series(route: AlignedRoute, cfg: Settings, bias: float) -> np.ndarray:
    """Turn radius in metres, one value per IMU sample.

    The radius at a sample is window_m divided by the heading change over the
    window_m of road centred on it. NaN where speed is below v_min_mps
    (curvature is undefined there, not large) and within window_m / 2 of either
    end of the route, where the window runs off the data. Shared by the segment
    flag and the viewer's evidence chart, so the chart plots exactly the values
    the flag was decided on.
    """
    if len(route.imu) < 2:
        return np.full(len(route.imu), np.nan)
    times = np.array([sample.timestamp_utc.timestamp() for sample in route.imu])
    gz = np.array([sample.gz for sample in route.imu], dtype=float) - bias
    # A stopped vehicle does not turn, so whatever the gyro reads meanwhile is
    # drift; integrated it would all land at one point of road. Crawling is left
    # in: it can be a real turn (curvature_review/stop_yaw.md).
    gz[route.imu_speed_mps < cfg.curvature.stationary_speed_mps] = 0.0
    heading = np.concatenate([[0.0], np.cumsum(0.5 * (gz[1:] + gz[:-1]) * np.diff(times))])

    # Heading as a function of distance needs distance to increase. While
    # stopped it does not; the first sample of each stop stands for it.
    dist = route.imu_dist_m
    moving = np.concatenate([[True], np.diff(dist) > 0.0])
    return _window_radius(route, dist[moving], heading[moving], cfg)


def gps_radius_series(route: AlignedRoute, cfg: Settings) -> np.ndarray | None:
    """The same windowed radius as radius_series, from GPS bearing instead of the gyro.

    One value per IMU sample, so the two line up. Only fixes at or above
    v_min_mps count: below it a receiver's course wanders. None when the
    route carries no usable bearing (KITTI's GPS does not).
    """
    bearing = np.array(
        [np.nan if fix.bearing_deg is None else fix.bearing_deg for fix in route.gps], dtype=float
    )
    usable = (route.gps_speed_mps >= cfg.curvature.v_min_mps) & np.isfinite(bearing)
    usable &= np.concatenate([[True], np.diff(route.gps_dist_m) > 0.0])
    if usable.sum() < 2:
        return None
    # Bearing is clockwise; negate to the gyro's counter-clockwise sense.
    heading = -np.unwrap(np.radians(bearing[usable]))
    return _window_radius(route, route.gps_dist_m[usable], heading, cfg)


def _window_radius(
    route: AlignedRoute, along_m: np.ndarray, heading: np.ndarray, cfg: Settings
) -> np.ndarray:
    """window_m over the heading change across the window centred at each IMU sample.

    heading (radians) is given at distances along_m. NaN where the window runs
    past either end of along_m, and where speed is below v_min_mps.
    """
    dist = route.imu_dist_m
    half = cfg.curvature.window_m / 2.0
    radii = np.full(dist.size, np.nan)
    inside = (dist - half >= along_m[0]) & (dist + half <= along_m[-1])
    turn = np.interp(dist + half, along_m, heading) - np.interp(dist - half, along_m, heading)
    with np.errstate(divide="ignore"):
        window = np.where(np.abs(turn) > 1e-12, cfg.curvature.window_m / np.abs(turn), _HUGE_RADIUS_M)
    radii[inside] = np.clip(window[inside], 0.0, _HUGE_RADIUS_M)
    return np.where(route.imu_speed_mps >= cfg.curvature.v_min_mps, radii, np.nan)


def analyse_curvature(
    route: AlignedRoute, segments: list[Segment], cfg: Settings
) -> list[ModuleResult]:
    """Produce one ModuleResult per segment, aggregating by worst (smallest) radius."""
    bias = estimate_gyro_bias(route, cfg)
    speed = route.imu_speed_mps
    radii = radius_series(route, cfg, bias)
    gps_radii = gps_radius_series(route, cfg)

    results: list[ModuleResult] = []
    for segment in segments:
        lo, hi = segment.imu_start, segment.imu_end
        window_speed = speed[lo:hi]

        if window_speed.size == 0:
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

        # Only windows lying wholly inside the segment code it; one centred
        # nearer than window_m / 2 to an end would measure the neighbour's road.
        dist = route.imu_dist_m[lo:hi]
        half = cfg.curvature.window_m / 2.0
        inner = (dist - half >= segment.start_m) & (dist + half <= segment.end_m)
        window_radii = np.where(inner, radii[lo:hi], np.nan)
        if np.isnan(window_radii).all():
            results.append(
                ModuleResult(
                    module=MODULE_NAME,
                    segment_id=segment.segment_id,
                    flag=Flag.NOT_ASSESSED,
                    metrics={"mean_speed_mps": mean_speed, "gyro_bias_rad_s": bias},
                    notes=(
                        f"No {cfg.curvature.window_m:.0f} m window of road driven at "
                        f"or above {cfg.curvature.v_min_mps} m/s fits inside this "
                        "segment."
                    ),
                )
            )
            continue

        # The worst (smallest) radius codes the segment, matching the iRAP rule
        # that the highest-risk value codes the segment. Gated samples are NaN.
        worst_index = int(np.nanargmin(window_radii))
        radius_m = float(window_radii[worst_index])

        critical = radius_m < cfg.curvature.threshold_m
        metrics = {
            "radius_m": radius_m,
            # The yaw rate that radius implies at that speed: the
            # window's mean, not one (noisy) gyro sample.
            "yaw_rate_rad_s": float(window_speed[worst_index] / radius_m),
            "mean_speed_mps": mean_speed,
            "gyro_bias_rad_s": bias,
            "irap_category": float(irap_curvature_category(radius_m)),
        }
        if gps_radii is not None:
            gps_window = np.where(inner, gps_radii[lo:hi], np.nan)
            if not np.isnan(gps_window).all():
                gps_radius_m = float(np.nanmin(gps_window))
                metrics["gps_radius_m"] = gps_radius_m
                gps_critical = gps_radius_m < cfg.curvature.threshold_m
                metrics["needs_review"] = float(gps_critical != critical)

        results.append(
            ModuleResult(
                module=MODULE_NAME,
                segment_id=segment.segment_id,
                flag=Flag.CRITICAL if critical else Flag.NORMAL,
                metrics=metrics,
            )
        )
    return results
