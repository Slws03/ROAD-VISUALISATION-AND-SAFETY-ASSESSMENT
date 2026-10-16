"""Orchestration of the CPU fusion layer.

Kept separate from cli.py so the whole flow can be tested without invoking a
command-line runner. Grade runs before Road Condition because the latter reuses
the former's pitch estimate for ground-plane projection, which removes the need
for a separate camera calibration step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from irap_screening.config import Settings
from irap_screening.decision.curve_quality import code_quality_of_curve
from irap_screening.decision.gapcheck import run_gapcheck
from irap_screening.decision.report import SCORING_MODULES, build_report
from irap_screening.io.align import align_route
from irap_screening.io.detections import load_detection_set
from irap_screening.io.schema import ModuleResult
from irap_screening.io.sources.fixture import FixtureSource
from irap_screening.modules.condition import analyse_condition
from irap_screening.modules.curvature import analyse_curvature
from irap_screening.modules.grade import analyse_grade, pitch_series
from irap_screening.modules.lane import analyse_lane
from irap_screening.modules.marking import analyse_marking
from irap_screening.segment import build_segments


def _require_all_modules(module_results: dict[str, list[ModuleResult]]) -> None:
    """Fail loudly if a scoring module is absent from the results dict.

    gap-check reads a missing module key as not_assessed, identical to a segment it
    genuinely could not evaluate. A wiring mistake would therefore stop every rule
    from firing and produce a report with no findings rather than an error.
    """
    missing = set(SCORING_MODULES) - set(module_results)
    if missing:
        raise ValueError(
            f"module_results is missing {sorted(missing)}; gap-check would silently "
            "treat those segments as unassessed and report no findings"
        )


def run_pipeline(route_dir: Path, cfg: Settings, route_id: str = "R01") -> dict[str, Any]:
    """Run the full CPU layer over one route directory."""
    route_dir = Path(route_dir)
    source = FixtureSource(route_dir)
    route = align_route(source.frames(), source.gps(), source.imu())
    segments = build_segments(route, cfg, route_id=route_id)
    detections = load_detection_set(route_dir / "detections")

    pitch = pitch_series(route, cfg)
    mean_pitch = float(np.mean(pitch)) if pitch.size else 0.0

    module_results = {
        "curvature": analyse_curvature(route, segments, cfg),
        "grade": analyse_grade(route, segments, cfg),
        "delineation": analyse_lane(route, segments, detections, cfg, image_root=route_dir),
        "condition": analyse_condition(
            route, segments, detections, cfg, pitch_rad=mean_pitch
        ),
    }
    _require_all_modules(module_results)
    existence = analyse_marking(
        route,
        segments,
        detections,
        cfg,
        image_root=route_dir,
        grade_results=module_results["grade"],
    )
    recommendations = run_gapcheck(segments, module_results, existence, cfg)
    quality = code_quality_of_curve(segments, module_results, existence, cfg)

    return build_report(
        route_id, segments, module_results, existence, recommendations, cfg,
        quality_of_curve=quality,
    )

