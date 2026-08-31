"""Traffic-sign runner built on a fine-tuned YOLO11m detector.

Writes one `sign.jsonl` record per frame listed in the route's frames.csv, in
the canonical DetectionRecord schema. Two properties matter and are enforced
rather than assumed:

- Frame identity comes from frames.csv, never from globbing the image
  directory. The fusion layer looks detections up by frame_id, so ids invented
  from filenames would produce records nothing ever reads -- and no error.
- Class names must be exactly the ones the decision layer matches on. A
  retrained weight whose names drift ("bend" for "Bend") would flow through
  every layer cleanly and then match no gap-check rule, reporting missing
  signage that is actually present. That failure is silent, so the check is
  loud.

Coordinates are original-image pixels. Ultralytics letterboxes internally and
maps its boxes back before returning them, so no inverse transform is needed
here; the Transform record documents the letterbox for audit only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from irap_screening.config import Settings
from irap_screening.io.schema import BoxInstance, DetectionRecord, Frame, Transform
from irap_screening.io.sources.fixture import FixtureSource

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch out of the CPU layer
    from collections.abc import Iterable

# Ultralytics pads the scaled image up to a multiple of this stride.
LETTERBOX_STRIDE = 32

# The exact strings irap_screening.decision.gapcheck matches on, plus the
# catch-all the training run merged every other sign into.
EXPECTED_CLASSES = ("Bend", "Double Bend", "Chevron", "No overtaking", "other_sign")


def validate_class_names(names: dict[int, str]) -> None:
    """Fail loudly when a weight's classes are not the ones the decision layer knows."""
    actual = tuple(names[key] for key in sorted(names))
    if actual != EXPECTED_CLASSES:
        raise ValueError(
            "weight class names do not match the decision layer's vocabulary.\n"
            f"  expected: {list(EXPECTED_CLASSES)}\n"
            f"  weight:   {list(actual)}\n"
            "Gap-check matches these strings exactly; a mismatch would silently "
            "report present signage as missing."
        )


def letterbox_wh(source_wh: tuple[int, int], imgsz: int) -> tuple[int, int]:
    """Return the shape the model actually saw, not the requested budget.

    Ultralytics uses rectangular inference: it scales the long side to imgsz and
    pads the short side up to a stride multiple, so a 4592x3448 frame is seen as
    roughly 1280x960, never 1280x1280. Recording the square would put a
    per-image falsehood into the one field that exists to make the coordinate
    mapping auditable.
    """
    width, height = source_wh
    ratio = imgsz / max(width, height)
    unpadded = (round(width * ratio), round(height * ratio))
    # Ultralytics pads by (imgsz - unpadded) modulo stride, which is not the same
    # as rounding up to a stride multiple unless imgsz is itself one.
    padded = tuple(value + (imgsz - value) % LETTERBOX_STRIDE for value in unpadded)
    return padded[0], padded[1]


def build_record(
    frame: Frame,
    boxes: Iterable[tuple[tuple[float, float, float, float], float, str]],
    source_wh: tuple[int, int],
    imgsz: int,
) -> DetectionRecord:
    """Assemble one frame's detections into a canonical record.

    Instance ids follow `{frame_id}#sign#{index}` so a finding in the report can
    be traced back to a single box in a single frame.
    """
    instances = [
        BoxInstance(
            id=f"{frame.frame_id}#sign#{index}",
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            conf=float(conf),
            cls=cls,
        )
        for index, ((x1, y1, x2, y2), conf, cls) in enumerate(boxes)
    ]
    return DetectionRecord(
        frame_id=frame.frame_id,
        image_path=frame.path,
        sha256=frame.sha256,
        type="sign",
        transform=Transform(
            source_wh=source_wh,
            crop_top=0.0,
            crop_bottom=0.0,
            model_wh=letterbox_wh(source_wh, imgsz),
        ),
        instances=instances,
    )


def _boxes_of(result: Any) -> list[tuple[tuple[float, float, float, float], float, str]]:
    """Extract (bbox, conf, class name) triples from one ultralytics Results object."""
    names = result.names
    boxes = []
    for box in result.boxes:
        x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
        boxes.append(((x1, y1, x2, y2), float(box.conf[0]), names[int(box.cls[0])]))
    return boxes


def run(
    route_root: Path,
    weights: Path,
    cfg: Settings,
    out_path: Path | None = None,
    device: str = "cpu",
) -> Path:
    """Detect signs on every frame of a route and write detections/sign.jsonl.

    Returns the path written. Inference runs frame by frame rather than as a
    batch: a screening route is a few hundred frames, and per-frame iteration
    keeps the frames.csv row and its result unambiguously paired.
    """
    from ultralytics import YOLO  # imported lazily: torch is an optional extra

    route_root = Path(route_root)
    frames = FixtureSource(route_root).frames()

    model = YOLO(str(weights))
    validate_class_names(model.names)

    out_path = Path(out_path) if out_path else route_root / "detections" / "sign.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            image_path = route_root / frame.path
            if not image_path.exists():
                raise FileNotFoundError(
                    f"frames.csv lists {frame.path} for {frame.frame_id}, but "
                    f"{image_path} does not exist"
                )
            result = model.predict(
                str(image_path),
                imgsz=cfg.sign.imgsz,
                conf=cfg.sign.runner_conf,
                device=device,
                verbose=False,
            )[0]
            height, width = result.orig_shape
            record = build_record(frame, _boxes_of(result), (width, height), cfg.sign.imgsz)
            handle.write(json.dumps(record.model_dump(mode="json")) + "\n")

    return out_path
