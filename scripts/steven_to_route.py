"""Convert a Neoun multi-report export into one canonical route directory per report.

The export holds positions.csv, accelerations.csv and rotations.csv for every
report together (keyed by reportId), and reports/<id>/ with <id>.meta.json and
spots/. Unlike the single RPT-* layout neoun_to_route.py reads, it carries a
gyroscope, so curvature comes from the measured yaw rate instead of the GPS
bearing stand-in.

- Phone axes as in neoun_to_route: landscape mount, x up, -z forward, the
  forward sign checked against GPS dv/dt. Checked on this export: gyro x
  tracks the GPS bearing rate with slope -0.99 (rad/s, counter-clockwise
  about up), so it is gz once mapped.
- Mount tilt: the mean gravity direction is rotated onto vertical, and the
  same rotation is applied to the gyro, so gz stays the yaw rate about the
  vertical. Gyro bias is left in: the curvature module estimates it.
- The gyro is interpolated onto the accelerometer's timestamps; the two
  streams run at the same rate but are stamped separately.
- Spot photos ship without an extension (WebP or JPEG); OpenCV reads either
  by content.

Usage:
    uv run python scripts/steven_to_route.py \
        --export D:/FYP_RESOURCE/steven_extract/steven_scanned_routes --out routes         --names D:/FYP_RESOURCE/steven_extract/route_names.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from neoun_to_route import parse_time, write_spots


def iso_time(raw: str) -> float:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()


def by_report(path: Path) -> dict[str, list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[row["reportId"]].append(row)
    return groups


def by_time(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[dict[str, str]]]:
    """Sort by timestamp and drop repeated timestamps, which break differentiation downstream."""
    rows = sorted(rows, key=lambda r: iso_time(r["date"]))
    times = np.array([iso_time(r["date"]) for r in rows])
    keep = np.concatenate([[True], np.diff(times) > 0])
    return times[keep], [r for r, k in zip(rows, keep, strict=True) if k]


def xyz(rows: list[dict[str, str]]) -> np.ndarray:
    return np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])


def write_sensors(
    out: Path, pos: list[dict[str, str]], acc: list[dict[str, str]], rot: list[dict[str, str]]
) -> tuple[float, float, tuple[float, float]]:
    t_gps, pos = by_time(pos)
    t_imu, acc = by_time(acc)
    t_rot, rot = by_time(rot)
    speed = np.array([float(r["speedMeta.speed"]) for r in pos])

    with (out / "gps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "lat", "lon", "speed_mps", "altitude", "bearing_deg"])
        for t, r in zip(t_gps, pos, strict=True):
            writer.writerow(
                [datetime.fromtimestamp(t, UTC).isoformat(), r["point.coordinates[1]"],
                 r["point.coordinates[0]"], r["speedMeta.speed"], r["point.coordinates[2]"],
                 r["bearingMeta.bearing"]]
            )

    phone_acc = xyz(acc)
    phone_gyro = np.column_stack([np.interp(t_imu, t_rot, axis) for axis in xyz(rot).T])
    # Landscape mount: phone x is up, y is left, -z is forward.
    accel = np.column_stack([-phone_acc[:, 2], phone_acc[:, 1], phone_acc[:, 0]])
    gyro = np.column_stack([-phone_gyro[:, 2], phone_gyro[:, 1], phone_gyro[:, 0]])

    # The forward sign must agree with the speed record, or grade's
    # longitudinal correction doubles the error instead of cancelling it.
    v_imu = np.interp(t_imu, t_gps, speed)
    dvdt = np.gradient(savgol_filter(v_imu, 31, 2), t_imu)
    corr = float(np.corrcoef(savgol_filter(accel[:, 0], 31, 2), dvdt)[0, 1])
    if corr < 0:
        # Turning about the up axis keeps the frame right-handed and gz unchanged.
        accel[:, :2] *= -1
        gyro[:, :2] *= -1
        corr = -corr

    # Rotate the mean gravity direction onto vertical: removes mount tilt.
    mean_g = accel.mean(axis=0)
    tilt = float(np.degrees(np.arccos(mean_g[2] / np.linalg.norm(mean_g))))
    level, _ = Rotation.align_vectors([[0.0, 0.0, 1.0]], [mean_g])
    accel, gyro = level.apply(accel), level.apply(gyro)

    with (out / "imu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "ax", "ay", "az", "gx", "gy", "gz"])
        for t, a, g in zip(t_imu, accel, gyro, strict=True):
            writer.writerow([datetime.fromtimestamp(t, UTC).isoformat(), *a, *g])

    return corr, tilt, (t_gps[0], t_gps[-1])


def route_name(meta: dict, names: dict[str, str]) -> str:
    """RPT-e8227eed-13f-AI -> e8227eed, prefixed by its road name when one is listed."""
    code = meta["name"].split("-")[1]
    return f"{names[code]}_{code}" if code in names else code


def read_names(path: Path | None) -> dict[str, str]:
    """code -> road name, from a CSV with columns code,name (reverse-geocoded by hand)."""
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["code"]: row["name"] for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", required=True, type=Path, help="folder holding positions.csv")
    parser.add_argument("--out", required=True, type=Path, help="parent directory of the routes")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality")
    parser.add_argument("--names", type=Path, help="CSV of code,name: road name per report")
    args = parser.parse_args()
    names = read_names(args.names)

    pos = by_report(args.export / "positions.csv")
    acc = by_report(args.export / "accelerations.csv")
    rot = by_report(args.export / "rotations.csv")

    for report in sorted((args.export / "reports").iterdir()):
        meta = json.loads((report / f"{report.name}.meta.json").read_text(encoding="utf-8"))
        out = args.out / route_name(meta, names)
        (out / "detections").mkdir(parents=True, exist_ok=True)

        corr, tilt, span = write_sensors(out, pos[meta["id"]], acc[meta["id"]], rot[meta["id"]])
        spots = [
            (parse_time(s["createdAt"]), report / "spots" / s["photo"]["name"], s["id"])
            for s in meta["spots"]
        ]
        frames = write_spots(spots, out, span, args.quality)
        print(f"{out.name} ({meta['name']}): forward corr {corr:.2f}, "
              f"tilt {tilt:.1f} deg, frames {frames}")


if __name__ == "__main__":
    main()
