"""Loading and validating detection artifacts written by the GPU runners.

This is the boundary between processes. Validating here means a malformed
runner output fails immediately with a clear message, rather than surfacing
much later as a quietly wrong number.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from irap_screening.io.schema import DetectionRecord


def load_detections(path: Path) -> dict[str, DetectionRecord]:
    """Read one JSONL detection file, keyed by frame id.

    A duplicate frame_id within one file overwrites the earlier record; a runner
    writing one record per frame per detector type should never produce one.
    """
    path = Path(path)
    if not path.exists():
        return {}

    records: dict[str, DetectionRecord] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = DetectionRecord.model_validate(json.loads(line))
            records[record.frame_id] = record
    return records


@dataclass
class DetectionSet:
    """All cached detections for one route, grouped by detector type."""

    lane: dict[str, DetectionRecord] = field(default_factory=dict)
    sign: dict[str, DetectionRecord] = field(default_factory=dict)
    pothole: dict[str, DetectionRecord] = field(default_factory=dict)


def load_detection_set(root: Path) -> DetectionSet:
    """Read lane, sign and pothole artifacts from a detections directory.

    A missing file yields an empty mapping rather than an error: the Road
    Condition module is explicitly stubbable, and a run without it is valid.
    """
    root = Path(root)
    return DetectionSet(
        lane=load_detections(root / "lane.jsonl"),
        sign=load_detections(root / "sign.jsonl"),
        pothole=load_detections(root / "pothole.jsonl"),
    )
