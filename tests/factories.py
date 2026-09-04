"""Synthetic routes with analytically known ground truth.

Module tests need a route whose true radius or true grade is known exactly.
Deriving those from real data would make the tests circular, so they are
constructed here from first principles.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

from irap_screening.io.schema import Frame, GpsFix, ImuSample

EARTH_RADIUS_M = 6_371_000.0
T0 = datetime(2026, 8, 31, tzinfo=UTC)


def _offset_latlon(lat: float, lon: float, north_m: float, east_m: float) -> tuple[float, float]:
    dlat = math.degrees(north_m / EARTH_RADIUS_M)
    dlon = math.degrees(east_m / (EARTH_RADIUS_M * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


def straight_route(
    length_m: float = 500.0,
    speed_mps: float = 16.7,
    gps_hz: float = 1.0,
    imu_hz: float = 50.0,
    include_speed: bool = True,
) -> tuple[list[Frame], list[GpsFix], list[ImuSample]]:
    """A straight, level road driven at constant speed. Yaw rate and pitch are zero."""
    duration_s = length_m / speed_mps
    lat0, lon0 = 1.55, 110.35

    gps: list[GpsFix] = []
    # ceil, not truncation: a 500 m route at 16.7 m/s lasts 29.94 s, and
    # truncating to 29 fixes covers only 484 m -- the factory would not deliver
    # the length it was asked for, and segment-count assertions would be off by one.
    n_gps = math.ceil(duration_s * gps_hz) + 1
    for i in range(n_gps):
        t = i / gps_hz
        lat, lon = _offset_latlon(lat0, lon0, north_m=speed_mps * t, east_m=0.0)
        gps.append(
            GpsFix(
                timestamp_utc=T0 + timedelta(seconds=t),
                lat=lat,
                lon=lon,
                speed_mps=speed_mps if include_speed else None,
            )
        )

    imu: list[ImuSample] = []
    n_imu = int(duration_s * imu_hz) + 1
    for i in range(n_imu):
        t = i / imu_hz
        imu.append(
            ImuSample(
                timestamp_utc=T0 + timedelta(seconds=t),
                ax=0.0, ay=0.0, az=9.81, gx=0.0, gy=0.0, gz=0.0,
            )
        )

    frames = [
        Frame(
            frame_id=f"f_{i:06d}",
            path=f"frames/f_{i:06d}.jpg",
            timestamp_utc=T0 + timedelta(seconds=i),
            sha256=f"{i:016x}",
        )
        for i in range(int(duration_s) + 1)
    ]
    return frames, gps, imu


def curved_route(
    radius_m: float = 312.5,
    speed_mps: float = 16.7,
    arc_length_m: float = 300.0,
    imu_hz: float = 50.0,
    gyro_bias_rad_s: float = 0.0,
    stationary_prefix_s: float = 0.0,
) -> tuple[list[Frame], list[GpsFix], list[ImuSample]]:
    """A constant-radius curve. True yaw rate is speed / radius throughout.

    gyro_bias_rad_s injects a constant offset so bias-correction can be tested.
    stationary_prefix_s prepends a parked interval, which is what the bias
    estimator needs a sample of; without it there is no zero reference.
    """
    omega = speed_mps / radius_m
    duration_s = arc_length_m / speed_mps
    lat0, lon0 = 1.55, 110.35

    gps: list[GpsFix] = []
    imu: list[ImuSample] = []

    # Parked at the start: no motion, so the gyro reads pure bias.
    n_prefix_gps = int(stationary_prefix_s)
    for i in range(n_prefix_gps):
        gps.append(
            GpsFix(
                timestamp_utc=T0 + timedelta(seconds=float(i)),
                lat=lat0,
                lon=lon0,
                speed_mps=0.0,
            )
        )
    for i in range(int(stationary_prefix_s * imu_hz)):
        imu.append(
            ImuSample(
                timestamp_utc=T0 + timedelta(seconds=i / imu_hz),
                ax=0.0, ay=0.0, az=9.81,
                gx=0.0, gy=0.0, gz=gyro_bias_rad_s,
            )
        )

    offset = stationary_prefix_s
    n_gps = math.ceil(duration_s) + 1
    for i in range(n_gps):
        t = float(i)
        theta = omega * t
        north = radius_m * math.sin(theta)
        east = radius_m * (1.0 - math.cos(theta))
        lat, lon = _offset_latlon(lat0, lon0, north_m=north, east_m=east)
        gps.append(
            GpsFix(
                timestamp_utc=T0 + timedelta(seconds=offset + t),
                lat=lat,
                lon=lon,
                speed_mps=speed_mps,
            )
        )

    n_imu = int(duration_s * imu_hz) + 1
    for i in range(n_imu):
        t = i / imu_hz
        imu.append(
            ImuSample(
                timestamp_utc=T0 + timedelta(seconds=offset + t),
                ax=0.0, ay=0.0, az=9.81,
                gx=0.0, gy=0.0, gz=omega + gyro_bias_rad_s,
            )
        )

    frames = [
        Frame(
            frame_id=f"f_{i:06d}",
            path=f"frames/f_{i:06d}.jpg",
            timestamp_utc=T0 + timedelta(seconds=offset + i),
            sha256=f"{i:016x}",
        )
        for i in range(int(duration_s) + 1)
    ]
    return frames, gps, imu


def graded_route(
    grade_pct: float = 8.0,
    speed_mps: float = 16.7,
    length_m: float = 300.0,
    imu_hz: float = 50.0,
) -> tuple[list[Frame], list[GpsFix], list[ImuSample]]:
    """A constant-grade slope. Accelerometer sees gravity tilted by the pitch angle."""
    pitch_rad = math.atan(grade_pct / 100.0)
    duration_s = length_m / speed_mps
    lat0, lon0 = 1.55, 110.35

    gps: list[GpsFix] = []
    for i in range(math.ceil(duration_s) + 1):
        t = float(i)
        lat, lon = _offset_latlon(lat0, lon0, north_m=speed_mps * t, east_m=0.0)
        gps.append(
            GpsFix(
                timestamp_utc=T0 + timedelta(seconds=t),
                lat=lat,
                lon=lon,
                speed_mps=speed_mps,
                altitude=speed_mps * t * math.sin(pitch_rad),
            )
        )

    imu: list[ImuSample] = []
    n_imu = int(duration_s * imu_hz) + 1
    for i in range(n_imu):
        t = i / imu_hz
        imu.append(
            ImuSample(
                timestamp_utc=T0 + timedelta(seconds=t),
                ax=-9.81 * math.sin(pitch_rad),
                ay=0.0,
                az=9.81 * math.cos(pitch_rad),
                gx=0.0, gy=0.0, gz=0.0,
            )
        )

    frames = [
        Frame(
            frame_id=f"f_{i:06d}",
            path=f"frames/f_{i:06d}.jpg",
            timestamp_utc=T0 + timedelta(seconds=i),
            sha256=f"{i:016x}",
        )
        for i in range(int(duration_s) + 1)
    ]
    return frames, gps, imu


def spiked_grade_route(
    background_pct: float = 0.0,
    spike_pct: float = 20.0,
    spike_start_m: float = 40.0,
    spike_length_m: float = 3.0,
    speed_mps: float = 16.7,
    length_m: float = 300.0,
    imu_hz: float = 50.0,
) -> tuple[list[Frame], list[GpsFix], list[ImuSample]]:
    """A background_pct road with one short, steep spike.

    Models a culvert or speed bump: a brief pitch excursion covering
    spike_length_m of distance, embedded in an otherwise constant grade. Used
    to test the iRAP rule that a grade shorter than one-tenth of the 100 m
    coding segment must not be recorded as the segment's grade (Coding
    Manual p.54, attribute AO: "record the steepest gradient which is 10m in
    length or more").
    """
    duration_s = length_m / speed_mps
    lat0, lon0 = 1.55, 110.35

    def pitch_rad_at(distance_m: float) -> float:
        in_spike = spike_start_m <= distance_m < spike_start_m + spike_length_m
        return math.atan((spike_pct if in_spike else background_pct) / 100.0)

    gps: list[GpsFix] = []
    for i in range(math.ceil(duration_s) + 1):
        t = float(i)
        distance = speed_mps * t
        lat, lon = _offset_latlon(lat0, lon0, north_m=distance, east_m=0.0)
        gps.append(
            GpsFix(
                timestamp_utc=T0 + timedelta(seconds=t),
                lat=lat,
                lon=lon,
                speed_mps=speed_mps,
                altitude=distance * math.sin(pitch_rad_at(distance)),
            )
        )

    imu: list[ImuSample] = []
    n_imu = int(duration_s * imu_hz) + 1
    for i in range(n_imu):
        t = i / imu_hz
        pitch_rad = pitch_rad_at(speed_mps * t)
        imu.append(
            ImuSample(
                timestamp_utc=T0 + timedelta(seconds=t),
                ax=-9.81 * math.sin(pitch_rad),
                ay=0.0,
                az=9.81 * math.cos(pitch_rad),
                gx=0.0, gy=0.0, gz=0.0,
            )
        )

    frames = [
        Frame(
            frame_id=f"f_{i:06d}",
            path=f"frames/f_{i:06d}.jpg",
            timestamp_utc=T0 + timedelta(seconds=i),
            sha256=f"{i:016x}",
        )
        for i in range(int(duration_s) + 1)
    ]
    return frames, gps, imu


def braking_route(
    grade_pct: float = 0.0,
    v_start: float = 28.0,
    v_end: float = 8.0,
    length_m: float = 300.0,
    imu_hz: float = 50.0,
) -> tuple[list[Frame], list[GpsFix], list[ImuSample]]:
    """A constant-grade road driven with a steady deceleration.

    An accelerometer measures specific force, so its longitudinal channel carries
    the vehicle's own acceleration on top of the gravity projection. On a level
    road the two are indistinguishable from the accelerometer alone: braking
    reads exactly like a downgrade. This factory produces that confusion
    deliberately.
    """
    pitch_rad = math.atan(grade_pct / 100.0)
    duration_s = 2.0 * length_m / (v_start + v_end)
    accel = (v_end - v_start) / duration_s
    lat0, lon0 = 1.55, 110.35

    def speed_at(t: float) -> float:
        return v_start + accel * t

    def distance_at(t: float) -> float:
        return v_start * t + 0.5 * accel * t * t

    gps: list[GpsFix] = []
    for i in range(math.ceil(duration_s) + 1):
        t = min(float(i), duration_s)
        lat, lon = _offset_latlon(lat0, lon0, north_m=distance_at(t), east_m=0.0)
        gps.append(
            GpsFix(
                timestamp_utc=T0 + timedelta(seconds=t),
                lat=lat,
                lon=lon,
                speed_mps=speed_at(t),
                altitude=distance_at(t) * math.sin(pitch_rad),
            )
        )

    imu: list[ImuSample] = []
    for i in range(int(duration_s * imu_hz) + 1):
        t = i / imu_hz
        imu.append(
            ImuSample(
                timestamp_utc=T0 + timedelta(seconds=t),
                # Gravity projection plus the vehicle's own longitudinal
                # acceleration, which is what the sensor actually reports.
                ax=-9.81 * math.sin(pitch_rad) + accel,
                ay=0.0,
                az=9.81 * math.cos(pitch_rad),
                gx=0.0, gy=0.0, gz=0.0,
            )
        )

    frames = [
        Frame(
            frame_id=f"f_{i:06d}",
            path=f"frames/f_{i:06d}.jpg",
            timestamp_utc=T0 + timedelta(seconds=i),
            sha256=f"{i:016x}",
        )
        for i in range(int(duration_s) + 1)
    ]
    return frames, gps, imu
