"""Convert one Neoun road report (RPT-*) into the canonical route directory layout.

A Neoun report is a phone survey: position.csv (1 Hz fused GPS, with speed and
bearing), acceleration.csv (about 10 Hz, accelerometer only) and spots/ (forward
camera stills, 4K .jpg and 960x544 .webp, timestamped by spots.csv). Three gaps
against what the pipeline assumes are bridged here, and each one is a stand-in,
not a measurement:

- No gyroscope. Curvature is taken from the GPS bearing differentiated over
  distance (see curvature_profile), pooled by median across the other runs of
  the same road passed with --also, and written as gz = v * kappa so the
  curvature module's v / |gz| returns 1 / |kappa|. It is zeroed below
  curvature.v_min_mps, where it would otherwise be folded into the gyro bias.
  gx and gy are zero.
- Phone axes. The phone is mounted landscape: x points up, -z points forward
  (checked per route against GPS dv/dt, and flipped if the data says so).
- Mount tilt. The phone sits pitched by several degrees, which reads as a
  constant 10+ percent grade. The mean gravity direction over the whole report
  is rotated onto vertical, which assumes the road's mean grade over the report
  is zero.

Usage:
    uv run python scripts/neoun_to_route.py --report "<...>/Q13-1A/RPT-973c0525-abd-AI" \
        --also "<...>/Q13-1A/RPT-d3a8d3f5-5bd-AI" --out routes/q13_1a
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

V_MIN_MPS = 5.0  # mirrors curvature.v_min_mps in configs/default.yaml
MAX_WIDTH = 1920  # 4K stills are downscaled; detectors resize far below this anyway
EARTH_M = 6371000.0
STEP_M = 5.0  # curvature grid spacing along the road
WINDOW_M = 50.0  # heading-derivative window; curves shorter than this are flattened
MATCH_M = 15.0  # another run's point counts as the same place within this distance
MAX_HEADING_DEG = 30.0  # ...and only if it travels along the same line
# Another run's first and last stretch is not pooled. Q13-1A's second run was
# still merging off a roundabout 100 m in, inside the heading tolerance, and
# with two runs the median is a mean and cannot reject it.
EDGE_M = 2 * WINDOW_M


def parse_time(raw: str) -> datetime:
    """Parse Neoun's '2026-02-14 2:02:25.42 +00:00:00' (single-digit hour, ragged fraction)."""
    stamp = raw.split(" +")[0]
    fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in stamp else "%Y-%m-%d %H:%M:%S"
    return datetime.strptime(stamp, fmt).replace(tzinfo=UTC)


def read_csv(report: Path, suffix: str) -> list[dict[str, str]]:
    (path,) = report.glob(f"*{suffix}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def by_time(rows: list[dict[str, str]], key: str) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Sort by timestamp and drop repeated timestamps, which break differentiation downstream."""
    rows = sorted(rows, key=lambda r: parse_time(r[key]))
    times = np.array([parse_time(r[key]).timestamp() for r in rows])
    keep = np.concatenate([[True], np.diff(times) > 0])
    return times[keep], [r for r, k in zip(rows, keep, strict=True) if k]


def read_track(report: Path) -> tuple[np.ndarray, list[dict[str, str]], dict[str, np.ndarray]]:
    t_gps, pos = by_time(read_csv(report, ".position.csv"), "date")
    columns = {
        "lat": np.array([float(r["latitude"]) for r in pos]),
        "lon": np.array([float(r["longitude"]) for r in pos]),
        "speed": np.array([float(r["speed"]) for r in pos]),
        # A fix carries no bearing when stopped.
        "bearing": np.array([float(r["bearing"] or "nan") for r in pos]),
    }
    return t_gps, pos, columns


def distance_along(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Cumulative haversine distance, the same axis the pipeline's alignment uses."""
    la, lo = np.radians(lat), np.radians(lon)
    a = (
        np.sin(np.diff(la) / 2) ** 2
        + np.cos(la[:-1]) * np.cos(la[1:]) * np.sin(np.diff(lo) / 2) ** 2
    )
    return np.concatenate([[0.0], np.cumsum(2 * EARTH_M * np.arcsin(np.sqrt(a)))])


def curvature_profile(track: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Signed curvature (1/m, clockwise positive) on a STEP_M grid of distance travelled.

    Heading is differentiated over distance, not time. Over time, R = v / omega
    divides a fixed bearing noise by speed, so at 6 m/s a 1-2 deg/s jitter reads
    as a 100-300 m curve on a straight road. Over distance the noise floor does
    not depend on speed. Only fixes at or above V_MIN_MPS are used: below it a
    fused bearing wanders.
    """
    s = distance_along(track["lat"], track["lon"])
    ok = (track["speed"] >= V_MIN_MPS) & ~np.isnan(track["bearing"])
    grid = np.arange(s[ok][0], s[ok][-1], STEP_M)
    heading = np.interp(grid, s[ok], np.unwrap(np.radians(track["bearing"][ok])))
    return {
        "s": s,
        "grid": grid,
        "heading": heading,
        "kappa": savgol_filter(heading, int(WINDOW_M / STEP_M) | 1, 2, deriv=1, delta=STEP_M),
        "lat": np.interp(grid, s, track["lat"]),
        "lon": np.interp(grid, s, track["lon"]),
    }


def pool_runs(ref: dict[str, np.ndarray], others: list[dict[str, np.ndarray]]) -> np.ndarray:
    """Median |curvature| across runs of the same road, at each point of the reference run.

    A point of another run counts when it lies within MATCH_M of the reference
    point and runs along the same line, in either direction; |kappa| is pooled
    so an opposite-direction run still counts. The first and last EDGE_M of
    another run are left out: there the derivative window runs off the data,
    and a run often joins the road from a side street. The reference run's
    sign is kept. Returns the number of runs per point.
    """
    def xy(p: dict[str, np.ndarray]) -> np.ndarray:
        scale = EARTH_M * np.cos(np.radians(ref["lat"].mean()))
        return np.column_stack([np.radians(p["lon"]) * scale, np.radians(p["lat"]) * EARTH_M])

    samples = [np.abs(ref["kappa"])]
    for other in others:
        grid = other["grid"]
        inner = (grid >= grid[0] + EDGE_M) & (grid <= grid[-1] - EDGE_M)
        if not inner.any():
            continue
        dist, index = cKDTree(xy(other)[inner]).query(xy(ref))
        # Angle between the two directions of travel, folded so opposite counts as aligned.
        turn = np.abs(np.angle(np.exp(2j * (other["heading"][inner][index] - ref["heading"]))))
        aligned = (dist <= MATCH_M) & (turn / 2 <= np.radians(MAX_HEADING_DEG))
        samples.append(np.where(aligned, np.abs(other["kappa"][inner][index]), np.nan))
    stacked = np.vstack(samples)
    ref["kappa"] = np.sign(ref["kappa"]) * np.nanmedian(stacked, axis=0)
    return (~np.isnan(stacked)).sum(axis=0)


def write_sensors(
    report: Path, out: Path, also: list[Path]
) -> tuple[float, float, tuple[float, float], np.ndarray]:
    t_gps, pos, track = read_track(report)
    t_imu, acc = by_time(read_csv(report, ".acceleration.csv"), "date")
    speed = track["speed"]

    with (out / "gps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "lat", "lon", "speed_mps", "altitude"])
        for t, r in zip(t_gps, pos, strict=True):
            writer.writerow(
                [datetime.fromtimestamp(t, UTC).isoformat(), r["latitude"], r["longitude"],
                 r["speed"], r["altitude"]]
            )

    phone = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in acc])
    # Landscape mount: phone x is up, y is left, -z is forward.
    vehicle = np.column_stack([-phone[:, 2], phone[:, 1], phone[:, 0]])

    # The forward sign must agree with the speed record, or grade's
    # longitudinal correction doubles the error instead of cancelling it.
    v_imu = np.interp(t_imu, t_gps, speed)
    dvdt = np.gradient(savgol_filter(v_imu, 31, 2), t_imu)
    fwd = savgol_filter(vehicle[:, 0], 31, 2)
    corr = float(np.corrcoef(fwd, dvdt)[0, 1])
    if corr < 0:
        vehicle[:, 0] *= -1
        vehicle[:, 1] *= -1  # keep the frame right-handed
        corr = -corr

    # Rotate the mean gravity direction onto vertical: removes mount tilt.
    mean_g = vehicle.mean(axis=0)
    tilt = float(np.degrees(np.arccos(mean_g[2] / np.linalg.norm(mean_g))))
    rot, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [mean_g])
    vehicle = rot.apply(vehicle)

    profile = curvature_profile(track)
    runs = pool_runs(profile, [curvature_profile(read_track(p)[2]) for p in also])
    # gz = v * kappa, so the curvature module's v / |gz| recovers 1 / |kappa|.
    # Bearing is clockwise; gz is about the up axis, counter-clockwise.
    s_imu = np.interp(t_imu, t_gps, profile["s"])
    gz = -v_imu * np.interp(s_imu, profile["grid"], profile["kappa"], left=0.0, right=0.0)
    gz[v_imu < V_MIN_MPS] = 0.0

    with (out / "imu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "ax", "ay", "az", "gx", "gy", "gz"])
        for t, (ax, ay, az), g in zip(t_imu, vehicle, gz, strict=True):
            writer.writerow([datetime.fromtimestamp(t, UTC).isoformat(), ax, ay, az, 0.0, 0.0, g])

    return corr, tilt, (t_gps[0], t_gps[-1]), runs


def write_frames(report: Path, out: Path, span: tuple[float, float], quality: int) -> int:
    """Every spot still inside the GPS span, in capture order, as JPEG."""
    # A spot can ship as both a 4K .jpg and a 960-wide .webp; sorting .jpg
    # last makes it win the stem, so detectors get the full-resolution still.
    files = {
        p.name.split(".")[0]: p
        for p in sorted((report / "spots").iterdir(), key=lambda p: p.suffix == ".jpg")
    }
    spots = sorted(
        (parse_time(r["createdAt"]), files[r["name"].split(".")[0]], r["id"])
        for r in read_csv(report, "-spots.csv")
        if r["name"].split(".")[0] in files
    )
    return write_spots(spots, out, span, quality)


def write_spots(
    spots: list[tuple[datetime, Path, str]], out: Path, span: tuple[float, float], quality: int
) -> int:
    """Write (time, image, spot id) stills inside the GPS span, sorted, as frames/ + frames.csv."""
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    last = None
    with (out / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "path", "timestamp_utc", "sha256", "source_frame"])
        for when, source, spot_id in sorted(spots):
            if not span[0] <= when.timestamp() <= span[1] or when == last:
                continue
            image = cv2.imread(str(source))
            if image is None:
                raise SystemExit(f"could not read {source}")
            if image.shape[1] > MAX_WIDTH:
                scale = MAX_WIDTH / image.shape[1]
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frame_id = f"f_{written:06d}"
            target = frames_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
            writer.writerow([frame_id, f"frames/{frame_id}.jpg", when.isoformat(), digest, spot_id])
            written += 1
            last = when
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path, help="Neoun RPT-* directory")
    parser.add_argument("--out", required=True, type=Path, help="route directory to write")
    parser.add_argument(
        "--also", nargs="*", type=Path, default=[],
        help="other RPT-* runs of the same road; their curvature is pooled by median",
    )
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "detections").mkdir(exist_ok=True)

    corr, tilt, span, runs = write_sensors(args.report, args.out, args.also)
    frames = write_frames(args.report, args.out, span, args.quality)

    print(f"Wrote {args.out}")
    print(f"  forward axis vs GPS dv/dt correlation : {corr:.2f}")
    print(f"  mount tilt removed                    : {tilt:.1f} deg")
    counts = np.bincount(runs)
    print(f"  curvature points by runs pooled       : {dict(enumerate(counts.tolist()))}")
    print(f"  frames                                : {frames}")


if __name__ == "__main__":
    main()
