"""Composite warning signs into a copy of a route, for a synthetic scenario.

A drive with no warning signage cannot exercise the gap-check decision layer:
every curve reports the same missing mitigations no matter what the rules do.
Pasting signs into a copy gives a second variant to compare against, so the
layer can be shown to change its verdict for the right reason.

Anything produced from a route this wrote is a synthetic scenario. It says
nothing about detector accuracy on real imagery, and a report built on it must
be labelled as such.

Placement is not decoration. The gap-check rules are spatial: an advance bend
sign belongs at a point 230 m before the curve entry, a chevron belongs inside
the curve. Pasting a bend sign at the curve leaves the gap-check reporting it
missing, which is correct behaviour and looks like a bug.

Usage:
    uv run python scripts/paste_signs.py \
        --route routes/kitti_0042 --out routes/kitti_0042_signed \
        --spec scripts/kitti_0042_signs.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def diamond_mask(height: int, width: int) -> np.ndarray:
    """Alpha for a diamond sign, so its corners keep the destination background.

    A rectangular paste of a diamond sign carries the corners of whatever scene
    it was cut from, which shows up against sky. Letting the destination through
    also gives the detector the sign boundary it needs to classify confidently.
    """
    yy, xx = np.mgrid[0:height, 0:width]
    inside = np.abs(xx / (width - 1) - 0.5) + np.abs(yy / (height - 1) - 0.5) <= 0.52
    return inside.astype(np.float32)


def load_sign(path: Path, mask_kind: str) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"could not read sign image {path}")
    height, width = image.shape[:2]
    if mask_kind == "diamond":
        return image, diamond_mask(height, width)
    if mask_kind == "none":
        return image, np.ones((height, width), np.float32)
    raise SystemExit(f"unknown mask {mask_kind!r}; use 'diamond' or 'none'")


def composite(
    base: np.ndarray,
    sign: np.ndarray,
    mask: np.ndarray,
    centre_x: float,
    centre_y: float,
    height: int,
) -> None:
    """Draw the sign into base at the given centre and pixel height, in place."""
    sign_h, sign_w = sign.shape[:2]
    width = max(2, int(round(height * sign_w / sign_h)))
    scaled = cv2.resize(sign, (width, height), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(mask, (width, height), interpolation=cv2.INTER_AREA)[..., None]

    left, top = int(round(centre_x - width / 2)), int(round(centre_y - height / 2))
    frame_h, frame_w = base.shape[:2]
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(frame_w, left + width), min(frame_h, top + height)
    if x1 <= x0 or y1 <= y0:
        return

    scaled = scaled[y0 - top : y1 - top, x0 - left : x1 - left]
    alpha = alpha[y0 - top : y1 - top, x0 - left : x1 - left]
    base[y0:y1, x0:x1] = (alpha * scaled + (1.0 - alpha) * base[y0:y1, x0:x1]).astype(
        np.uint8
    )


def interpolate(start: float, end: float, index: int, count: int) -> float:
    """Linear ramp across a run of frames; a single frame takes the start value."""
    if count < 2:
        return start
    return start + (end - start) * index / (count - 1)


def apply_spec(out_route: Path, spec: dict, spec_dir: Path) -> dict[str, list[str]]:
    """Paste every placement, returning which signs landed on which frames."""
    signs = {
        name: load_sign(
            (spec_dir / entry["image"]).resolve(), entry.get("mask", "none")
        )
        for name, entry in spec["signs"].items()
    }
    frames_dir = out_route / "frames"
    touched: dict[str, list[str]] = {}

    for placement in spec["placements"]:
        sign, mask = signs[placement["sign"]]
        frames = placement["frames"]
        start, end = placement["start"], placement.get("end", placement["start"])
        for index, frame_index in enumerate(frames):
            frame_id = f"f_{frame_index:06d}"
            path = frames_dir / f"{frame_id}.jpg"
            if not path.exists():
                raise SystemExit(
                    f"{path} does not exist; the spec names a frame the route does "
                    "not contain, probably because of the subsampling step"
                )
            image = cv2.imread(str(path))
            composite(
                image,
                sign,
                mask,
                interpolate(start["x"], end["x"], index, len(frames)),
                interpolate(start["y"], end["y"], index, len(frames)),
                int(round(interpolate(start["h"], end["h"], index, len(frames)))),
            )
            cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
            touched.setdefault(placement["sign"], []).append(frame_id)
    return touched


def rewrite_digests(route: Path) -> None:
    """Recompute frames.csv checksums for the frames that were edited.

    Nothing validates them today, but leaving the originals in place would make
    the manifest quietly describe images that no longer exist.
    """
    path = route / "frames.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["sha256"] = hashlib.sha256(
            (route / row["path"]).read_bytes()
        ).hexdigest()[:16]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["frame_id", "path", "timestamp_utc", "sha256"]
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, type=Path, help="source route")
    parser.add_argument("--out", required=True, type=Path, help="route copy to write")
    parser.add_argument("--spec", required=True, type=Path, help="placement spec JSON")
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))

    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(args.route, args.out)
    # Detections describe the original imagery, so they cannot carry over.
    detections = args.out / "detections"
    if detections.exists():
        shutil.rmtree(detections)
    detections.mkdir()

    touched = apply_spec(args.out, spec, args.spec.parent)
    rewrite_digests(args.out)

    print(f"Wrote {args.out}")
    for sign, frames in touched.items():
        print(f"  {sign:10} {len(frames):3} frames  {frames[0]} .. {frames[-1]}")
    print()
    print("This route is a synthetic scenario. Re-run the sign detector on it:")
    print(f"  uv run irap detect-signs --route {args.out} --weights <best.pt>")


if __name__ == "__main__":
    main()
