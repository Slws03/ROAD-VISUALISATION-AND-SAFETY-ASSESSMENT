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


def _export(root: Path) -> None:
    rid = "r1"
    t_gps = np.arange(0.0, 60.0, 1.0)
    t_imu = np.arange(0.0, 60.0, 0.1)
    speed = lambda t: 8 + 2 * np.sin(2 * np.pi * t / 15)  # noqa: E731
    accel = lambda t: 2 * 2 * np.pi / 15 * np.cos(2 * np.pi * t / 15)  # noqa: E731

    def write(name, header, rows):
        with (root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    write("positions.csv",
          ["_id", "date", "reportId", "point.coordinates[0]", "point.coordinates[1]",
           "point.coordinates[2]", "speedMeta.speed", "bearingMeta.bearing"],
          [[i, _stamp(t), rid, 110.0, 1.5 + t * 1e-4, 30.0, speed(t), 0] for i, t in enumerate(t_gps)])
    # Landscape mount: gravity on phone x, forward is -z.
    write("accelerations.csv", ["_id", "date", "reportId", "x", "y", "z"],
          [[i, _stamp(t), rid, 9.81, 0.0, -accel(t)] for i, t in enumerate(t_imu)])
    # Stamped apart from the accelerometer, as in the real export.
    write("rotations.csv", ["_id", "date", "reportId", "x", "y", "z"],
          [[i, _stamp(t + 0.05), rid, YAW, 0.0, 0.0] for i, t in enumerate(t_imu)])

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


def test_a_listed_road_name_prefixes_the_route(tmp_path):
    names = tmp_path / "names.csv"
    names.write_text("code,name\nabc12345,jalan_endap\n", encoding="utf-8")
    meta = {"name": "RPT-abc12345-13f-AI"}
    assert steven.route_name(meta, steven.read_names(names)) == "jalan_endap_abc12345"
    assert steven.route_name(meta, {}) == "abc12345"
