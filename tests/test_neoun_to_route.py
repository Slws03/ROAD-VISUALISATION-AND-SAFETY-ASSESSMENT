"""The Neoun converter's GPS curvature must read a straight as straight at low speed.

The time-based yaw rate it replaced turned 1-2 degrees of bearing jitter at
6 m/s into 100-300 m "curves" on a straight residential road.
"""

import importlib.util
from pathlib import Path

import cv2
import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "neoun_to_route.py"
_spec = importlib.util.spec_from_file_location("neoun_to_route", _PATH)
neoun = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(neoun)

LAT0 = 1.5
SPEED = 6.0
STRAIGHT_M = 300.0
RADIUS_M = 200.0
ARC_M = 400.0


def _track(seed: int) -> dict[str, np.ndarray]:
    """1 Hz fixes along a northbound straight, then a right-hand arc, with integer-degree jitter."""
    s = np.arange(0.0, STRAIGHT_M + ARC_M, SPEED)
    on_arc = np.clip(s - STRAIGHT_M, 0.0, None)
    heading = on_arc / RADIUS_M  # clockwise from north
    east = np.where(s > STRAIGHT_M, RADIUS_M * (1 - np.cos(heading)), 0.0)
    north = np.where(s > STRAIGHT_M, STRAIGHT_M + RADIUS_M * np.sin(heading), s)
    jitter = np.random.default_rng(seed).normal(0.0, 1.5, s.size)
    return {
        "lat": LAT0 + np.degrees(north / neoun.EARTH_M),
        "lon": 110.0 + np.degrees(east / (neoun.EARTH_M * np.cos(np.radians(LAT0)))),
        "speed": np.full(s.size, SPEED),
        "bearing": np.round(np.degrees(heading) + jitter) % 360,
    }


def _radius(profile, lo, hi):
    window = (profile["grid"] >= lo) & (profile["grid"] < hi)
    return 1.0 / np.abs(profile["kappa"][window])


def test_straight_at_low_speed_is_not_a_curve():
    profile = neoun.curvature_profile(_track(seed=0))
    # Keep clear of the transition, which the 50 m window spreads either side.
    assert _radius(profile, 0, STRAIGHT_M - neoun.WINDOW_M).min() > 400.0


def test_arc_radius_is_recovered():
    profile = neoun.curvature_profile(_track(seed=0))
    middle = np.median(_radius(profile, STRAIGHT_M + 100, STRAIGHT_M + ARC_M - 100))
    assert abs(middle - RADIUS_M) / RADIUS_M < 0.1


def test_pooling_counts_every_overlapping_run_and_keeps_the_sign():
    ref = neoun.curvature_profile(_track(seed=0))
    others = [neoun.curvature_profile(_track(seed=k)) for k in (1, 2)]
    runs = neoun.pool_runs(ref, others)
    # Each run's own ends are left out of the pool, so only the interior has all three.
    margin = neoun.EDGE_M + 10
    interior = (ref["grid"] > margin) & (ref["grid"] < ref["grid"][-1] - margin)
    assert (runs[interior] == 3).all()
    arc = ref["grid"] > STRAIGHT_M + 100
    assert (ref["kappa"][arc] > 0).all()  # right-hand bend, clockwise positive


def test_a_run_crossing_at_an_angle_is_not_pooled():
    ref = neoun.curvature_profile(_track(seed=0))
    crossing = _track(seed=1)
    crossing["bearing"] = (crossing["bearing"] + 90) % 360  # same places, perpendicular travel
    runs = neoun.pool_runs(ref, [neoun.curvature_profile(crossing)])
    assert (runs == 1).all()


def test_a_spot_with_both_formats_uses_the_full_resolution_jpg(tmp_path):
    report, out = tmp_path / "RPT-x", tmp_path / "route"
    (report / "spots").mkdir(parents=True)
    cv2.imwrite(str(report / "spots" / "s1.webp"), np.zeros((54, 96, 3), np.uint8))
    cv2.imwrite(str(report / "spots" / "s1.jpg"), np.zeros((108, 192, 3), np.uint8))
    (report / "r-spots.csv").write_text(
        "id,name,createdAt\nA,s1.jpg,2026-02-14 2:02:25.42 +00:00:00\n", encoding="utf-8"
    )
    when = neoun.parse_time("2026-02-14 2:02:25.42 +00:00:00").timestamp()

    assert neoun.write_frames(report, out, (when - 1, when + 1), 90) == 1
    assert cv2.imread(str(out / "frames" / "f_000000.jpg")).shape[1] == 192
