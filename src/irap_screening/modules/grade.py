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

The remaining, fast error -- vibration, bumps -- is averaged out over
window_m of road: pitch at a sample is the mean corrected accelerometer pitch
over the window_m centred on it. Against KITTI's INS this took the error from
0.38 to 0.26 degrees RMSE at 30 m. A complementary filter with the gyro used
to do this job and did it worse: with no gyro bias correction it gave 1.05
degrees on KITTI, and a bracket-mounted phone's 10 Hz pitch rate is mount
vibration aliased into the band, which read 208 of 266 Samarahan segments as
critical (D:/FYP/grade_review). The gyro is not read here.

The three axes are averaged over the window before the pitch is taken, not
the per-sample pitch. A rocking mount shakes the forward and vertical axes
together, and the mean of their per-sample ratio falls short of the ratio of
their means by about cov(ax, az)/g^2: a downgrade that grows with vibration.
Against 7 GSDC 2023 phone drives (NovAtel SPAN altitude as reference, 2,711
segments) this took segment RMSE from 1.85 to 1.10 percent and false critical
segments from 30 to 6, with the 34 true ones found either way; KITTI 0042 was
unchanged (0.26 degrees RMSE). Those drives carry SPAN speed, cleaner than a
phone's GPS, so the dv/dt correction was tested at its best.

A segment is coded only from windows lying wholly inside it, as curvature is,
so a slope in the next segment cannot set this one's grade.

Where the route carries GPS altitude, the climb across the same window, over
its length, is an independent cross-check, read under the same rule. The
accelerometer still decides the flag; a segment where the two fall on
opposite sides of the threshold is marked needs_review, as curvature marks
its gyro/GPS disagreements. Altitude's 1 Hz fixes (about 9 m apart) cannot
see a short pitch the accelerometer can, which is why it does not decide.
"""

from __future__ import annotations

import numpy as np

from irap_screening.config import Settings
from irap_screening.io.align import AlignedRoute
from irap_screening.io.schema import Flag, ModuleResult, Segment

MODULE_NAME = "grade"


def _accel_pitch(ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
    """Pitch in radians from the gravity direction, positive nose-up.

    Nose up, the forward axis reads +g*sin(pitch): the specific force holding
    the vehicle against gravity. KITTI's OXTS and phones both report it so.
    """
    return np.arctan2(ax, np.sqrt(ay**2 + az**2))


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
    """Pitch in radians, one value per IMU sample.

    The accelerometer's pitch with the vehicle's own acceleration removed,
    over the window_m of road centred on each sample: each axis is averaged
    over the window first, then one pitch is taken from the means. Taking the
    pitch per sample and averaging that would turn vibration shared by the
    forward and vertical axes into a downgrade (see the module docstring).
    Near either end of the route the window holds whatever samples it can.
    """
    if not route.imu:
        return np.zeros(0)

    ax = np.array([s.ax for s in route.imu], dtype=float)
    ay = np.array([s.ay for s in route.imu], dtype=float)
    az = np.array([s.az for s in route.imu], dtype=float)

    # Remove the vehicle's own longitudinal acceleration, which the sensor
    # reports in the same channel as the gravity projection.
    ax = ax - longitudinal_accel(route)

    # IMU distance is interpolated from the GPS track, so it never decreases.
    dist = route.imu_dist_m
    half = cfg.grade.window_m / 2.0
    lo = np.searchsorted(dist, dist - half, side="left")
    hi = np.searchsorted(dist, dist + half, side="right")

    def window_mean(values: np.ndarray) -> np.ndarray:
        total = np.concatenate([[0.0], np.cumsum(values)])
        return (total[hi] - total[lo]) / (hi - lo)

    return _accel_pitch(window_mean(ax), window_mean(ay), window_mean(az))


def gps_grade_series(route: AlignedRoute, cfg: Settings) -> np.ndarray | None:
    """Grade (%) from GPS altitude: the climb across the window_m centred on each
    IMU sample, over window_m. NaN where the window runs past the altitude
    record; None when the route carries no altitude.
    """
    altitude = np.array(
        [np.nan if fix.altitude is None else fix.altitude for fix in route.gps], dtype=float
    )
    usable = np.isfinite(altitude) & np.concatenate([[True], np.diff(route.gps_dist_m) > 0.0])
    if usable.sum() < 2 or not route.imu:
        return None
    along = route.gps_dist_m[usable]
    # A climb between consecutive fixes steeper than any road is the receiver,
    # not the road: a phone's altitude jumps by metres in a second (36 such
    # steps over 1.5 m on one Samarahan route). Drop those steps and rebuild the
    # profile from the rest, so a jump that persists does not offset what follows.
    climb = np.diff(altitude[usable])
    implausible = np.abs(climb) > cfg.grade.gps_max_step_pct / 100.0 * np.diff(along)
    climb[implausible] = 0.0
    height = altitude[usable][0] + np.concatenate([[0.0], np.cumsum(climb)])
    dist = route.imu_dist_m
    half = cfg.grade.window_m / 2.0
    climb = np.interp(dist + half, along, height) - np.interp(dist - half, along, height)
    inside = (dist - half >= along[0]) & (dist + half <= along[-1])
    return np.where(inside, climb / cfg.grade.window_m * 100.0, np.nan)


def inner_samples(route: AlignedRoute, segment: Segment, cfg: Settings) -> np.ndarray:
    """Indices of the segment's IMU samples whose window_m lies wholly inside it.

    Shared by the flag and the viewer, so both read the same samples.
    """
    index = np.arange(segment.imu_start, segment.imu_end)
    dist = route.imu_dist_m[segment.imu_start : segment.imu_end]
    half = cfg.grade.window_m / 2.0
    return index[(dist - half >= segment.start_m) & (dist + half <= segment.end_m)]


def _sustained_steepest_grade(
    grade_pct: np.ndarray, dist_m: np.ndarray, min_length_m: float
) -> tuple[float, int]:
    """The steepest grade sustained for at least min_length_m of distance.

    Coding Manual p.54 (attribute AO): "Where there are variable grades
    within a 100m coding segment, record the steepest gradient which is 10m
    in length or more." A spike shorter than that -- a culvert, a speed
    bump -- must not be recorded as the segment's grade.

    For each sample, considers the smallest window of consecutive samples,
    ending at that sample, that still spans >= min_length_m, and takes the
    minimum |grade| within it: the level the gradient is guaranteed to hold
    for the whole window. Widening the window can only lower or preserve
    that minimum, never raise it, so the smallest qualifying window is also
    the one that maximises it -- the minimal window is optimal, not an
    approximation. The steepest such floor, across every sample, is the
    segment's value; its sign is taken from the sample that set it.

    Falls back to the single steepest sample when the segment's own IMU
    span never reaches min_length_m (sparse data), matching the prior,
    unfiltered behaviour rather than reporting nothing.
    """
    index, _, _, _ = sustained_grade_window(grade_pct, dist_m, min_length_m)
    return float(grade_pct[index]), index


def sustained_grade_window(
    grade_pct: np.ndarray, dist_m: np.ndarray, min_length_m: float
) -> tuple[int, int, int, bool]:
    """The search behind _sustained_steepest_grade, with the window it chose.

    Returns (index, left, right, fallback): the sample whose value is the
    segment's grade, the first and last sample of the window it was held over,
    and whether the segment was too short and the single steepest sample was
    taken instead (then left == right == index). Exposed so the viewer shows
    the exact window the flag was decided on.
    """
    n = grade_pct.size
    abs_grade = np.abs(grade_pct)

    best_floor = -1.0
    best = (0, 0, 0)
    left = 0
    for right in range(n):
        while left < right and dist_m[right] - dist_m[left + 1] >= min_length_m:
            left += 1
        if dist_m[right] - dist_m[left] < min_length_m:
            continue
        window = abs_grade[left : right + 1]
        window_index = left + int(np.argmin(window))
        floor = float(window[window_index - left])
        if floor > best_floor:
            best_floor = floor
            best = (window_index, left, right)

    if best_floor < 0:
        index = int(np.argmax(abs_grade))
        return index, index, index, True
    return best[0], best[1], best[2], False


def irap_grade_category(grade_pct: float) -> int:
    """iRAP Grade Risk Factor descriptive category.

    Display only: independent of threshold_pct, which drives the flag. Codes
    are 1, 4, 5 -- not sequential -- per the Coding Manual's own attribute
    value table: <7.5%->1 (flat/gentle), 7.5-<10%->4 (moderate), >=10%->5
    (steep).
    """
    magnitude = abs(grade_pct)
    if magnitude < 7.5:
        return 1
    if magnitude < 10.0:
        return 4
    return 5


def analyse_grade(
    route: AlignedRoute, segments: list[Segment], cfg: Settings
) -> list[ModuleResult]:
    """One ModuleResult per segment, aggregating by steepest sustained grade."""
    pitch = pitch_series(route, cfg)
    grade_pct = np.tan(pitch) * 100.0
    gps_grade = gps_grade_series(route, cfg)
    min_length_m = cfg.segment.length_m / 10.0  # iRAP: one tenth of the coding segment

    results: list[ModuleResult] = []
    for segment in segments:
        if segment.imu_end <= segment.imu_start:
            results.append(
                ModuleResult(
                    module=MODULE_NAME,
                    segment_id=segment.segment_id,
                    flag=Flag.NOT_ASSESSED,
                    notes="No IMU samples fall within this segment.",
                )
            )
            continue
        samples = inner_samples(route, segment, cfg)
        if samples.size == 0:
            results.append(
                ModuleResult(
                    module=MODULE_NAME,
                    segment_id=segment.segment_id,
                    flag=Flag.NOT_ASSESSED,
                    notes=(
                        f"No {cfg.grade.window_m:.0f} m window of road fits inside "
                        "this segment."
                    ),
                )
            )
            continue

        steepest, steepest_index = _sustained_steepest_grade(
            grade_pct[samples], route.imu_dist_m[samples], min_length_m
        )
        critical = abs(steepest) >= cfg.grade.threshold_pct
        metrics = {
            "grade_pct": steepest,
            "pitch_deg": float(np.degrees(pitch[samples[steepest_index]])),
            "irap_category": float(irap_grade_category(steepest)),
        }
        if gps_grade is not None:
            checked = samples[np.isfinite(gps_grade[samples])]
            if checked.size:
                gps_steepest, _ = _sustained_steepest_grade(
                    gps_grade[checked], route.imu_dist_m[checked], min_length_m
                )
                metrics["gps_grade_pct"] = gps_steepest
                gps_critical = abs(gps_steepest) >= cfg.grade.threshold_pct
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
