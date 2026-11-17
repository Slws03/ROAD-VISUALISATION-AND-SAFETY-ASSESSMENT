"""Convert a KITTI raw drive into the canonical route directory layout.

KITTI's oxts stream matches this pipeline's assumptions unusually well. Its
accelerations are given "in direction of vehicle front / left / top" and its
angular rates about those same axes, which is exactly the fixed-mount frame the
curvature and grade modules assume -- the assumption a loose phone breaks. Its
forward velocity vf is a recorded speed, so the pipeline uses it directly
instead of differencing positions.

Only the frames are subsampled. GPS and IMU keep every oxts row: alignment runs
on its own time axis, and thinning it would only coarsen the curvature and grade
windows for no benefit.

KITTI is continuous 10 Hz video. The default keeps every tenth frame, giving
roughly 1 Hz -- about 20 m of ground between frames at highway speed. Frames are renumbered sequentially from
zero; the original KITTI index is kept in a source_frame column so a frame can
still be found in the drive it came from.

Usage:
    uv run python scripts/kitti_to_route.py \
        --drive "D:/.../2011_10_03_drive_0042_sync" --out routes/kitti_0042
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import cv2

# Column indices in oxts/data/*.txt, per the drive's own dataformat.txt.
LAT, LON, ALT = 0, 1, 2
VF = 8
AX, AY, AZ = 11, 12, 13
WX, WY, WZ = 17, 18, 19


def parse_timestamp(raw: str) -> datetime:
    """Parse a KITTI timestamp, which carries nanoseconds and no timezone.

    fromisoformat accepts at most microseconds, so the sub-microsecond digits are
    dropped. KITTI records local capture time; it is read as UTC because the
    pipeline only ever uses differences and the offset cancels.
    """
    stamp, _, fraction = raw.strip().partition(".")
    micros = (fraction + "000000")[:6]
    return datetime.strptime(f"{stamp}.{micros}", "%Y-%m-%d %H:%M:%S.%f").replace(
        tzinfo=UTC
    )


def read_timestamps(path: Path) -> list[datetime]:
    return [parse_timestamp(line) for line in path.read_text().splitlines() if line.strip()]


def write_oxts(drive: Path, out: Path) -> int:
    """Write gps.csv and imu.csv from every oxts row."""
    rows = [
        [float(value) for value in path.read_text().split()]
        for path in sorted((drive / "oxts" / "data").glob("*.txt"))
    ]
    times = read_timestamps(drive / "oxts" / "timestamps.txt")
    if len(times) != len(rows):
        raise SystemExit(
            f"oxts has {len(rows)} samples but {len(times)} timestamps; "
            "the drive is inconsistent and alignment would be silently wrong"
        )

    with (out / "gps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "lat", "lon", "speed_mps", "altitude"])
        for time, row in zip(times, rows, strict=True):
            writer.writerow(
                [time.isoformat(), row[LAT], row[LON], row[VF], row[ALT]]
            )

    with (out / "imu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "ax", "ay", "az", "gx", "gy", "gz"])
        for time, row in zip(times, rows, strict=True):
            # wx/wy/wz are rates about the front/left/top axes, which is what
            # gx/gy/gz mean here: gz is yaw rate, gy is pitch rate.
            writer.writerow(
                [
                    time.isoformat(),
                    row[AX], row[AY], row[AZ],
                    row[WX], row[WY], row[WZ],
                ]
            )
    return len(rows)


def write_frames(drive: Path, out: Path, camera: str, every: int, quality: int) -> int:
    """Copy every Nth camera frame as JPEG and write frames.csv."""
    source = drive / camera / "data"
    # The drive ships a preview .gif alongside the frames; globbing *.png keeps
    # it out rather than letting it become a frame with no oxts row.
    images = sorted(source.glob("*.png"))
    times = read_timestamps(drive / camera / "timestamps.txt")
    if len(times) != len(images):
        raise SystemExit(
            f"{camera} has {len(images)} images but {len(times)} timestamps"
        )

    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    with (out / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        # source_frame is extra provenance; the route reader ignores unknown
        # columns, so it costs nothing downstream.
        writer.writerow(["frame_id", "path", "timestamp_utc", "sha256", "source_frame"])
        for index in range(0, len(images), every):
            image = cv2.imread(str(images[index]))
            if image is None:
                raise SystemExit(f"could not read {images[index]}")
            frame_id = f"f_{written:06d}"
            target = frames_dir / f"{frame_id}.jpg"
            cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            digest = hashlib.sha256(target.read_bytes()).hexdigest()[:16]
            writer.writerow(
                [
                    frame_id,
                    f"frames/{frame_id}.jpg",
                    times[index].isoformat(),
                    digest,
                    index,
                ]
            )
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive", required=True, type=Path, help="KITTI *_sync directory")
    parser.add_argument("--out", required=True, type=Path, help="route directory to write")
    parser.add_argument("--camera", default="image_02", help="camera stream (default: left colour)")
    parser.add_argument(
        "--every", type=int, default=10,
        help="keep every Nth frame; KITTI is 10 Hz, so 10 gives about 1 Hz",
    )
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality")
    args = parser.parse_args()

    if not (args.drive / "oxts" / "data").is_dir():
        raise SystemExit(f"no oxts/data under {args.drive}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "detections").mkdir(exist_ok=True)

    samples = write_oxts(args.drive, args.out)
    frames = write_frames(args.drive, args.out, args.camera, args.every, args.quality)

    print(f"Wrote {args.out}")
    print(f"  gps.csv / imu.csv : {samples} samples (every oxts row)")
    print(f"  frames            : {frames} of {samples} ({args.camera}, every {args.every})")
    print()
    print("Next: produce detections/lane.jsonl and detections/sign.jsonl, then")
    print(f"  uv run irap report --route {args.out} --out out")


if __name__ == "__main__":
    main()
