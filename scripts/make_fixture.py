"""Generate the committed demo route.

Produces a synthetic but complete route: a curve tight enough to trip the
curvature threshold, a lane marking faint enough to be gradeable, and cached
lane detections. Regenerating is deterministic, so the fixture can be rebuilt
if the schema changes.

Run: uv run python scripts/make_fixture.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.factories import curved_route  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "route_demo"
LINE_X = 320
LINE_Y_SAMPLES = [280.0, 320.0, 360.0, 400.0]


def _make_image(line_level: int) -> np.ndarray:
    image = np.full((480, 640, 3), 100, dtype=np.uint8)
    image[:, LINE_X - 3 : LINE_X + 4] = line_level
    return image


def main() -> None:
    frames, gps, imu = curved_route(radius_m=312.5, arc_length_m=400.0)
    (ROOT / "frames").mkdir(parents=True, exist_ok=True)
    (ROOT / "detections").mkdir(parents=True, exist_ok=True)

    digests: dict[str, str] = {}
    for index, frame in enumerate(frames):
        # Alternate a clear and a worn marking so the contrast bands are exercised.
        image = _make_image(210 if index % 3 else 118)
        path = ROOT / frame.path
        cv2.imwrite(str(path), image)
        digests[frame.frame_id] = hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    with (ROOT / "frames.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_id", "path", "timestamp_utc", "sha256"])
        for frame in frames:
            writer.writerow(
                [
                    frame.frame_id,
                    frame.path,
                    frame.timestamp_utc.isoformat(),
                    digests[frame.frame_id],
                ]
            )

    with (ROOT / "gps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "lat", "lon", "speed_mps"])
        for fix in gps:
            writer.writerow([fix.timestamp_utc.isoformat(), fix.lat, fix.lon, fix.speed_mps])

    with (ROOT / "imu.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_utc", "ax", "ay", "az", "gx", "gy", "gz"])
        for sample in imu:
            writer.writerow(
                [
                    sample.timestamp_utc.isoformat(),
                    sample.ax, sample.ay, sample.az,
                    sample.gx, sample.gy, sample.gz,
                ]
            )

    records = [
        {
            "frame_id": frame.frame_id,
            "image_path": frame.path,
            "sha256": digests[frame.frame_id],
            "type": "lane",
            "transform": {
                "source_wh": [640, 480],
                "crop_top": 0.0,
                "crop_bottom": 0.0,
                "model_wh": [800, 320],
            },
            "instances": [
                {
                    "id": f"{frame.frame_id}#lane#0",
                    "points": [[float(LINE_X), y] for y in LINE_Y_SAMPLES],
                    "conf": 0.9,
                }
            ],
        }
        for frame in frames
    ]
    (ROOT / "detections" / "lane.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8"
    )

    print(f"Wrote fixture with {len(frames)} frames to {ROOT}")


if __name__ == "__main__":
    main()
