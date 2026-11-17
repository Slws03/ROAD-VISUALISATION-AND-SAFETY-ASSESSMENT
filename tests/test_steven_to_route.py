"""The multi-report converter must put the phone's yaw rate on gz, with gravity on az."""

import csv
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from irap_screening.io.sources.fixture import FixtureSource

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_spec = importlib.util.spec_from_file_location("steven_to_route", _SCRIPTS / "steven_to_route.py")
steven = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(steven)

T0 = 1_760_000_000.0
YAW = 0.1  # rad/s, counter-clockwise


def _stamp(t: float) -> str:
    return datetime.fromtimestamp(T0 + t, UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _export(root: Path, duration: float = 60.0, yaw: float = YAW, grade=None) -> None:
    """Landscape-mounted drive at about 11 m/s; `grade(t)` (fraction) tilts the road."""
    rid = "r1"
    t_gps = np.arange(0.0, duration, 1.0)
    t_imu = np.arange(0.0, duration, 0.1)
    speed = lambda t: 8 + 2 * np.sin(2 * np.pi * t / 15)  # noqa: E731
    accel = lambda t: 2 * 2 * np.pi / 15 * np.cos(2 * np.pi * t / 15)  # noqa: E731
    pitch = np.arctan(grade(t_imu)) if grade else np.zeros(t_imu.size)
    step = 1e-4 * 111_195.0  # metres per GPS fix along the meridian
    climb = np.tan(np.interp(t_gps[1:], t_imu, pitch)) * step
    altitude = 30.0 + np.concatenate([[0.0], np.cumsum(climb)])

    def write(name, header, rows):
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    write("positions.csv",
          ["_id", "date", "reportId", "point.coordinates[0]", "point.coordinates[1]",
           "point.coordinates[2]", "speedMeta.speed", "bearingMeta.bearing"],
          [[i, _stamp(t), rid, 110.0, 1.5 + t * 1e-4, h, speed(t), 0]
           for i, (t, h) in enumerate(zip(t_gps, altitude, strict=True))])
    # Landscape mount: gravity on phone x, forward is -z. On a grade the
    # specific force tips forward by g sin(pitch).
    write("accelerations.csv", ["_id", "date", "reportId", "x", "y", "z"],
          [[i, _stamp(t), rid, 9.81 * np.cos(p), 0.0, -(accel(t) + 9.81 * np.sin(p))]
           for i, (t, p) in enumerate(zip(t_imu, pitch, strict=True))])
    # Stamped apart from the accelerometer, as in the real export.
    write("rotations.csv", ["_id", "date", "reportId", "x", "y", "z"],
          [[i, _stamp(t + 0.05), rid, yaw, 0.0, 0.0] for i, t in enumerate(t_imu)])

    spots = root / "reports" / rid / "spots"
    spots.mkdir(parents=True)
    ok, jpg = cv2.imencode(".jpg", np.zeros((72, 128, 3), np.uint8))
    (spots / "s1").write_bytes(jpg.tobytes())  # no extension, as shipped
    meta = {"id": rid, "name": "RPT-abc12345-13f-AI", "spots": [
        {"id": "s1", "createdAt": datetime.fromtimestamp(T0 + 30, UTC).strftime(
            "%Y-%m-%d %H:%M:%S.%f")[:-3] + " +00:00:00", "photo": {"name": "s1"}}]}
    (root / "reports" / rid / f"{rid}.meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_yaw_rate_lands_on_gz_and_frames_are_written(tmp_path, monkeypatch):
    export, out = tmp_path / "export", tmp_path / "routes"
    export.mkdir()
    _export(export)
    monkeypatch.setattr(sys, "argv", ["x", "--export", str(export), "--out", str(out)])
    steven.main()

    route = out / "abc12345"
    with (route / "imu.csv").open(newline="", encoding="utf-8") as handle:
        imu = np.array([[float(v) for v in list(r.values())[1:]] for r in csv.DictReader(handle)])
    ax, az, gx, gy, gz = imu[:, 0], imu[:, 2], imu[:, 3], imu[:, 4], imu[:, 5]

    assert np.allclose(gz, YAW, atol=1e-3)
    assert np.allclose(gx, 0, atol=1e-3) and np.allclose(gy, 0, atol=1e-3)
    assert np.allclose(az.mean(), 9.81, atol=0.05)
    assert np.corrcoef(ax, np.gradient(np.interp(np.arange(0, 60, 0.1), np.arange(60.0),
                       8 + 2 * np.sin(2 * np.pi * np.arange(60.0) / 15))))[0, 1] > 0.9
    assert (route / "frames" / "f_000000.jpg").exists()
    assert FixtureSource(route).gps()[0].bearing_deg == 0.0


def test_a_route_that_climbs_is_levelled_on_its_flat_road(tmp_path, monkeypatch):
    # Flat for a minute, then a 10 % climb: the drive's mean gravity is tipped
    # by the climb, so levelling on it would read the flat road as -5 %.
    export, out = tmp_path / "export", tmp_path / "routes"
    export.mkdir()
    _export(export, duration=120.0, yaw=0.0, grade=lambda t: np.where(t < 60.0, 0.0, 0.10))
    monkeypatch.setattr(sys, "argv", ["x", "--export", str(export), "--out", str(out)])
    steven.main()

    with (out / "abc12345" / "imu.csv").open(newline="", encoding="utf-8") as handle:
        imu = np.array([[float(v) for v in list(r.values())[1:]] for r in csv.DictReader(handle)])
    t = np.arange(0.0, 120.0, 0.1)[: len(imu)]
    forward = imu[:, 0] - 2 * 2 * np.pi / 15 * np.cos(2 * np.pi * t / 15)
    grade = forward / np.hypot(imu[:, 1], imu[:, 2]) * 100.0
    assert abs(grade[(t > 5) & (t < 55)].mean()) < 1.0
    assert abs(grade[(t > 65) & (t < 115)].mean() - 10.0) < 1.0


def test_a_listed_road_name_prefixes_the_route(tmp_path):
    names = tmp_path / "names.csv"
    names.write_text("code,name\nabc12345,jalan_endap\n", encoding="utf-8")
    meta = {"name": "RPT-abc12345-13f-AI"}
    assert steven.route_name(meta, steven.read_names(names)) == "jalan_endap_abc12345"
    assert steven.route_name(meta, {}) == "abc12345"
