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

    return build_report(
        route_id, segments, module_results, existence, recommendations, cfg
    )


def _rate_hz(timestamps: list[float]) -> float:
    if len(timestamps) < 2:
        return 0.0
    span = timestamps[-1] - timestamps[0]
    return (len(timestamps) - 1) / span if span > 0 else 0.0


def validate_route(route_dir: Path, cfg: Settings) -> dict[str, Any]:
    """Health-check a route before writing any adapter code.

    Separates format problems from assumption problems: the former are cheap to
    fix in an adapter, the latter can invalidate a module's physical premise.
    """
    route_dir = Path(route_dir)
    source = FixtureSource(route_dir)
    frames, gps, imu = source.frames(), source.gps(), source.imu()
    route = align_route(frames, gps, imu)
    segments = build_segments(route, cfg)

    gps_times = [fix.timestamp_utc.timestamp() for fix in gps]
    imu_times = [sample.timestamp_utc.timestamp() for sample in imu]

    accel = np.array([[s.ax, s.ay, s.az] for s in imu], dtype=float)
    # Whichever axis carries the largest mean magnitude is the one aligned with
    # gravity. A fixed mount keeps this stable; a loose phone does not.
    gravity_axis = ("ax", "ay", "az")[int(np.argmax(np.abs(accel.mean(axis=0))))]

    return {
        "frames": len(frames),
        "gps_fixes": len(gps),
        "imu_samples": len(imu),
        "gps_hz": _rate_hz(gps_times),
        "imu_hz": _rate_hz(imu_times),
        "speed_source": route.speed_source,
        "route_length_m": route.total_length_m,
        "segments": len(segments),
        "frames_per_segment": (len(frames) / len(segments)) if segments else 0.0,
        "gravity_axis": gravity_axis,
        "gravity_magnitude": float(np.linalg.norm(accel.mean(axis=0))),
        "has_detections": (route_dir / "detections").exists(),
    }
