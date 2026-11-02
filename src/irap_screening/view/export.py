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

import numpy as np

from irap_screening.config import Settings
from irap_screening.io.align import AlignedRoute, align_route
from irap_screening.io.schema import Segment
from irap_screening.io.sources.fixture import FixtureSource
from irap_screening.modules.curvature import (
    estimate_gyro_bias,
    gps_radius_series,
    radius_series,
)
from irap_screening.modules.grade import (
    _accel_pitch,
    gps_grade_series,
    longitudinal_accel,
    pitch_series,
    inner_samples,
    sustained_grade_window,
)
from irap_screening.pipeline import run_pipeline
from irap_screening.segment import build_segments
from irap_screening.view.review import load_review

TEMPLATE = Path(__file__).parent / "template.html"


def _rounded(values: np.ndarray, digits: int) -> list[float | None]:
    # JSON has no NaN; a gated sample becomes null, which the chart draws as a gap.
    return [None if np.isnan(v) else round(float(v), digits) for v in values]


def build_profile(route: AlignedRoute, cfg: Settings) -> dict[str, list[float | None]]:
    """Per-IMU-sample series behind the curvature and grade flags.

    Computed by the same functions the modules use, so the evidence chart
    shows the values the flags were decided on, not a re-derivation.
    """
    # ponytail: every sample is exported (~1.2k on KITTI); downsample if a
    # long 100 Hz phone route makes data.json too heavy, keeping each min/max.
    radius = radius_series(route, cfg, estimate_gyro_bias(route, cfg))
    pitch = pitch_series(route, cfg)
    grade = np.tan(pitch) * 100.0
    times = np.array([s.timestamp_utc.timestamp() for s in route.imu], dtype=float)
    yaw_rate = np.array([s.gz for s in route.imu], dtype=float)  # raw, before bias removal
    ax, ay, az = np.array([[s.ax, s.ay, s.az] for s in route.imu], dtype=float).reshape(-1, 3).T
    a_long = longitudinal_accel(route)
    # The cross-checks, per sample like the series they check; all null where
    # the route carries no GPS bearing or altitude.
    missing = np.full(len(route.imu), np.nan)
    gps_radius = gps_radius_series(route, cfg)
    gps_grade = gps_grade_series(route, cfg)
    return {
        "dist_m": _rounded(route.imu_dist_m, 1),
        "time_s": _rounded(times - times[0], 2) if times.size else [],
        "yaw_rate_rad_s": _rounded(yaw_rate, 5),
        "speed_mps": _rounded(route.imu_speed_mps, 2),
        "radius_m": _rounded(radius, 1),
        "gps_radius_m": _rounded(missing if gps_radius is None else gps_radius, 1),
        "grade_pct": _rounded(grade, 2),
        "gps_grade_pct": _rounded(missing if gps_grade is None else gps_grade, 2),
        # Grade's chain: the accelerometer's gravity direction before and after
        # removing the vehicle's own acceleration, then fused with the gyro.
        "accel_x_mps2": _rounded(ax, 3),
        "long_accel_mps2": _rounded(a_long, 3),
        "pitch_accel_raw_deg": _rounded(np.degrees(_accel_pitch(ax, ay, az)), 3),
        "pitch_accel_deg": _rounded(np.degrees(_accel_pitch(ax - a_long, ay, az)), 3),
        "pitch_deg": _rounded(np.degrees(pitch), 3),
    }


def grade_windows(
    route: AlignedRoute, segments: list[Segment], cfg: Settings
) -> dict[str, dict[str, Any]]:
    """Per segment, the window its grade was held over, as analyse_grade chose it.

    Indices are into the route's IMU samples (the profile), not the segment.
    """
    grade = np.tan(pitch_series(route, cfg)) * 100.0
    min_length_m = cfg.segment.length_m / 10.0  # as analyse_grade
    windows = {}
    for segment in segments:
        samples = inner_samples(route, segment, cfg)
        if samples.size == 0:
            continue
        index, left, right, fallback = sustained_grade_window(
            grade[samples], route.imu_dist_m[samples], min_length_m
        )
        windows[segment.segment_id] = {
            "index": int(samples[index]),
            "start_m": round(float(route.imu_dist_m[samples[left]]), 1),
            "end_m": round(float(route.imu_dist_m[samples[right]]), 1),
            "fallback": fallback,
        }
    return windows


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
    # Interpolated from the frame's timestamp between GPS fixes: an estimate of
    # where the photo was taken, not a measurement.
    dist_by_id = {
        frame.frame_id: float(dist)
        for frame, dist in zip(route.frames, route.frame_dist_m, strict=True)
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    review_by_segment = load_review(route_dir)
    windows = grade_windows(route, segments, cfg)

    data: dict[str, Any] = dict(report)
    data["segments"] = [
        {
            **segment,
            "frame_ids": frame_ids_by_segment.get(segment["segment_id"], []),
            "frame_m": [
                round(dist_by_id[fid], 1)
                for fid in frame_ids_by_segment.get(segment["segment_id"], [])
            ],
            "review": review_by_segment.get(segment["segment_id"], {}),
            "grade_window": windows.get(segment["segment_id"]),
        }
        for segment in report["segments"]
    ]
    data["profile"] = build_profile(route, cfg)
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


_ROUTE_FILES = ("frames.csv", "gps.csv", "imu.csv")


def is_route_dir(path: Path) -> bool:
    """Whether `path` itself holds the raw CSVs a route source reads."""
    return all((path / name).exists() for name in _ROUTE_FILES)


def discover_routes(root: Path) -> dict[str, Path]:
    """Map route id to route directory under `root`.

    `root` is itself one route if it directly holds the raw CSVs; otherwise
    its immediate subdirectories that do are each treated as one route, keyed
    by directory name.
    """
    root = Path(root)
    if is_route_dir(root):
        return {root.name: root}
    return {
        child.name: child
        for child in sorted(root.iterdir())
        if child.is_dir() and is_route_dir(child)
    }


def export_routes(routes: dict[str, Path], cfg: Settings, out_dir: Path) -> Path:
    """Render every route into its own subdirectory under one shared page.

    Each route gets the same treatment as export_view, just nested under its
    route id, so the page can fetch <route_id>/data.json and switch between
    routes without a rebuild.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for route_id in sorted(routes):
        route_out = export_view(routes[route_id], cfg, out_dir / route_id, route_id=route_id)
        (route_out / "index.html").unlink()  # only the shared page at out_dir is served
        index.append({"id": route_id, "label": route_id})
    (out_dir / "routes.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
    shutil.copyfile(TEMPLATE, out_dir / "index.html")
    return out_dir


def add_route(
    route_id: str, route_dir: Path, routes: dict[str, Path], cfg: Settings, out_dir: Path
) -> None:
    """Render one more route into a directory export_routes already built.

    `routes` is updated in place. A running server resolves requests through
    that same dict, which is what makes the new route reachable without a
    restart.
    """
    out_dir = Path(out_dir)
    route_out = export_view(route_dir, cfg, out_dir / route_id, route_id=route_id)
    (route_out / "index.html").unlink()
    routes[route_id] = Path(route_dir).resolve()
    index = [{"id": known, "label": known} for known in sorted(routes)]
    (out_dir / "routes.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
