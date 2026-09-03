"""Build the local viewer directory for one route.

The viewer is a verification tool, not a second analysis: it re-runs the same
pipeline and renders the same report, adding only what the report deliberately
leaves out -- the frames behind each segment, with their detections drawn on.

Detections are drawn in the browser, on the frame at its native resolution,
so this module's only job is to say where each frame lives. It does not read
or write a single pixel.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from irap_screening.config import Settings
from irap_screening.io.align import align_route
from irap_screening.io.sources.fixture import FixtureSource
from irap_screening.pipeline import run_pipeline
from irap_screening.segment import build_segments

TEMPLATE = Path(__file__).parent / "template.html"


def export_view(
    route_dir: Path,
    cfg: Settings,
    out_dir: Path,
    route_id: str = "R01",
) -> Path:
    """Render a route into a self-describing directory the page can serve."""
    route_dir, out_dir = Path(route_dir), Path(out_dir)
    report = run_pipeline(route_dir, cfg, route_id=route_id)

    source = FixtureSource(route_dir)
    route = align_route(source.frames(), source.gps(), source.imu())
    segments = build_segments(route, cfg, route_id=route_id)

    frame_ids_by_segment = {
        segment.segment_id: list(segment.frame_ids) for segment in segments
    }
    referenced_ids = {fid for ids in frame_ids_by_segment.values() for fid in ids}
    frames_by_id = {frame.frame_id: frame for frame in route.frames}

    out_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, Any] = dict(report)
    data["segments"] = [
        {**segment, "frame_ids": frame_ids_by_segment.get(segment["segment_id"], [])}
        for segment in report["segments"]
    ]
    data["frames"] = {
        frame_id: frames_by_id[frame_id].path
        for frame_id in referenced_ids
        if frame_id in frames_by_id
    }
    (out_dir / "data.json").write_text(
        json.dumps(data, indent=1, default=str), encoding="utf-8"
    )
    shutil.copyfile(TEMPLATE, out_dir / "index.html")
    return out_dir
