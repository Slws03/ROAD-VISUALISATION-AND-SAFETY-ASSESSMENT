"""Time, distance and speed alignment across the three input streams.

Speed is a hard dependency of the curvature module: curvature is omega / v, and
integrating gyro alone yields heading change, not curvature. GPS runs near 1 Hz
while IMU runs far faster, so speed is interpolated up to IMU rate here rather
than inside each module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from irap_screening.io.schema import Frame, GpsFix, ImuSample

EARTH_RADIUS_M = 6_371_000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def cumulative_distance(fixes: list[GpsFix]) -> np.ndarray:
    """Distance along the route at each GPS fix, starting at zero."""
    if not fixes:
        return np.zeros(0)
    steps = [0.0]
    for previous, current in zip(fixes, fixes[1:], strict=False):
        steps.append(haversine_m(previous.lat, previous.lon, current.lat, current.lon))
    return np.cumsum(steps)


def _epoch_seconds(items: list[GpsFix] | list[ImuSample] | list[Frame]) -> np.ndarray:
    return np.array([item.timestamp_utc.timestamp() for item in items], dtype=float)


def resolve_speed(fixes: list[GpsFix]) -> tuple[np.ndarray, str]:
    """Return speed per GPS fix and which source produced it.

    A recorded speed field comes from carrier Doppler and is roughly an order of
    magnitude more accurate than differencing positions, so it is always
    preferred when present.
    """
    if fixes and all(fix.speed_mps is not None for fix in fixes):
        return np.array([fix.speed_mps for fix in fixes], dtype=float), "doppler"

    times = _epoch_seconds(fixes)
    dist = cumulative_distance(fixes)
    if len(fixes) < 2:
        return np.zeros(len(fixes)), "differenced"
    speeds = np.gradient(dist, times)
    return np.clip(speeds, 0.0, None), "differenced"


@dataclass
class AlignedRoute:
    """Everything the analyser modules need, on a shared distance axis."""

    frames: list[Frame]
    gps: list[GpsFix]
    imu: list[ImuSample]
    gps_dist_m: np.ndarray
    gps_speed_mps: np.ndarray
    speed_source: str
    imu_speed_mps: np.ndarray
    imu_dist_m: np.ndarray
    frame_dist_m: np.ndarray

    @property
    def total_length_m(self) -> float:
        return float(self.gps_dist_m[-1]) if len(self.gps_dist_m) else 0.0


def align_route(
    frames: list[Frame], gps: list[GpsFix], imu: list[ImuSample]
) -> AlignedRoute:
    """Put frames, GPS and IMU onto a common time and distance axis."""
    gps_times = _epoch_seconds(gps)
    gps_dist = cumulative_distance(gps)
    gps_speed, speed_source = resolve_speed(gps)

    imu_times = _epoch_seconds(imu)
    imu_speed = np.interp(imu_times, gps_times, gps_speed)
    imu_dist = np.interp(imu_times, gps_times, gps_dist)

    frame_times = _epoch_seconds(frames)
    frame_dist = np.interp(frame_times, gps_times, gps_dist)

    return AlignedRoute(
        frames=frames,
        gps=gps,
        imu=imu,
        gps_dist_m=gps_dist,
        gps_speed_mps=gps_speed,
        speed_source=speed_source,
        imu_speed_mps=imu_speed,
        imu_dist_m=imu_dist,
        frame_dist_m=frame_dist,
    )
