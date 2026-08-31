"""Build the local viewer directory for one route.

The viewer is a verification tool, not a second analysis: it re-runs the same
pipeline and renders the same report, adding only what the report deliberately
leaves out -- the frames behind each segment, with their detections drawn on.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2

from irap_screening.config import Settings
from irap_screening.io.align import align_route
from irap_screening.io.detections import load_detection_set
from irap_screening.io.sources.fixture import FixtureSource
from irap_screening.pipeline import run_pipeline
from irap_screening.segment import build_segments
from irap_screening.view.render import EVIDENCE_WIDTH, render_evidence

TEMPLATE = Path(__file__).parent / "template.html"
JPEG_QUALITY = 85


def export_view(
    route_dir: Path,
    cfg: Settings,
    out_dir: Path,
    route_id: str = "R01",
    target_width: int = EVIDENCE_WIDTH,
) -> Path:
    """Render a route into a self-describing directory the page can serve."""
    route_dir, out_dir = Path(route_dir), Path(out_dir)
    report = run_pipeline(route_dir, cfg, route_id=route_id)

    source = FixtureSource(route_dir)
    route = align_route(source.frames(), source.gps(), source.imu())
    segments = build_segments(route, cfg, route_id=route_id)
    detections = load_detection_set(route_dir / "detections")
    frames_by_id = {frame.frame_id: frame for frame in route.frames}

    frame_ids_by_segment = {
        segment.segment_id: list(segment.frame_ids) for segment in segments
    }

    evidence_dir = out_dir / "evidence"
    # Wipe rather than overwrite: leftovers from a previous route would be
    # served as if they belonged to this one.
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    evidence_dir.mkdir(parents=True)

    for segment in segments:
        for frame_id in segment.frame_ids:
            frame = frames_by_id.get(frame_id)
            if frame is None:
                continue
            image = cv2.imread(str(route_dir / frame.path))
            if image is None:
                continue
            rendered = render_evidence(
                image,
                detections.lane.get(frame_id),
                detections.sign.get(frame_id),
                target_width=target_width,
            )
            cv2.imwrite(
                str(evidence_dir / f"{frame_id}.jpg"),
                rendered,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )

    data: dict[str, Any] = dict(report)
    data["segments"] = [
        {**segment, "frame_ids": frame_ids_by_segment.get(segment["segment_id"], [])}
        for segment in report["segments"]
    ]
    (out_dir / "data.json").write_text(
        json.dumps(data, indent=1, default=str), encoding="utf-8"
    )
    shutil.copyfile(TEMPLATE, out_dir / "index.html")
    return out_dir
