"""CLI surface. The main entry point must run without a GPU."""

import json
from pathlib import Path

from typer.testing import CliRunner

from irap_screening.cli import app
from tests.test_pipeline import _write_route

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "route_demo"

runner = CliRunner()


def test_report_command_writes_all_outputs(tmp_path: Path):
    route = tmp_path / "route"
    route.mkdir()
    _write_route(route)
    out = tmp_path / "out"

    result = runner.invoke(app, ["report", "--route", str(route), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert (out / "report.json").exists()
    assert (out / "segments.geojson").exists()
    assert (out / "report.md").exists()


def test_report_accepts_inline_threshold_override(tmp_path: Path):
    route = tmp_path / "route"
    route.mkdir()
    _write_route(route)
    out = tmp_path / "out"

    result = runner.invoke(
        app,
        [
            "report", "--route", str(route), "--out", str(out),
            "--set", "curvature.threshold_m=200",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["config"]["curvature"]["threshold_m"] == 200


def test_missing_route_exits_with_an_error(tmp_path: Path):
    result = runner.invoke(app, ["report", "--route", str(tmp_path / "nope")])
    assert result.exit_code != 0


def test_detect_signs_is_registered_without_importing_torch():
    """The runner command must be reachable from a base install.

    Its ultralytics import is deliberately inside the command body, so listing
    and validating the CLI never pulls in the optional extra. If that import
    moved to module scope, this fails on any environment without it.
    """
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "detect-signs" in result.output


def test_import_lanes_is_registered():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "import-lanes" in result.output


def test_import_lanes_reports_a_missing_dump(tmp_path: Path):
    route = tmp_path / "route"
    route.mkdir()
    _write_route(route)

    result = runner.invoke(
        app, ["import-lanes", "--route", str(route), "--raw", str(tmp_path / "nope.json")]
    )
    assert result.exit_code != 0


def test_view_builds_without_serving(tmp_path: Path):
    """--no-serve must produce the directory and return, not block on a server."""
    import json

    out = tmp_path / "view"
    result = runner.invoke(
        app,
        ["view", "--route", str(FIXTURE), "--out", str(out), "--no-serve"],
    )

    assert result.exit_code == 0, result.output
    assert (out / "index.html").exists()
    assert (out / "data.json").exists()
    data = json.loads((out / "data.json").read_text(encoding="utf-8"))
    assert data["frames"]


def test_view_serves_the_route_directory_for_frames_and_detections(
    tmp_path: Path, monkeypatch
):
    """The static server must be able to reach route_dir/frames and
    route_dir/detections, not just the viewer's own output directory."""
    calls = {}

    def fake_serve_directory(directory, route_dir, port=8000, open_browser=True):
        calls["directory"] = directory
        calls["route_dir"] = route_dir

    monkeypatch.setattr(
        "irap_screening.view.serve.serve_directory", fake_serve_directory
    )

    out = tmp_path / "view"
    result = runner.invoke(app, ["view", "--route", str(FIXTURE), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert calls["route_dir"] == FIXTURE
    assert calls["directory"] == out


def test_view_scans_a_directory_of_routes(tmp_path: Path):
    """Pointing --route at a directory of routes (not a route itself) builds
    every route it finds under one shared page, instead of erroring out."""
    parent = tmp_path / "routes"
    _write_route(parent / "route_a")
    _write_route(parent / "route_b")
    out = tmp_path / "view"

    result = runner.invoke(app, ["view", "--route", str(parent), "--out", str(out), "--no-serve"])

    assert result.exit_code == 0, result.output
    assert (out / "index.html").exists()
    routes = json.loads((out / "routes.json").read_text(encoding="utf-8"))
    assert {r["id"] for r in routes} == {"route_a", "route_b"}
    assert (out / "route_a" / "data.json").exists()
    assert (out / "route_b" / "data.json").exists()


def test_view_scan_serves_each_routes_own_directory(tmp_path: Path, monkeypatch):
    parent = tmp_path / "routes"
    _write_route(parent / "route_a")
    _write_route(parent / "route_b")
    calls = {}

    def fake_serve_routes(directory, route_dirs, port=8000, open_browser=True):
        calls["directory"] = directory
        calls["route_dirs"] = route_dirs

    monkeypatch.setattr("irap_screening.view.serve.serve_routes", fake_serve_routes)

    out = tmp_path / "view"
    result = runner.invoke(app, ["view", "--route", str(parent), "--out", str(out)])

    assert result.exit_code == 0, result.output
    assert calls["directory"] == out
    assert calls["route_dirs"] == {"route_a": parent / "route_a", "route_b": parent / "route_b"}


def test_view_scan_rejects_an_empty_directory(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()

    result = runner.invoke(app, ["view", "--route", str(empty), "--no-serve"])

    assert result.exit_code != 0


def test_report_defaults_to_an_out_inside_the_route(tmp_path: Path):
    """Two routes must not write over each other's report by default.

    A fixed default meant every route's report landed in the same directory and
    the last run silently replaced the rest.
    """
    route = tmp_path / "route"
    route.mkdir()
    _write_route(route)

    result = runner.invoke(app, ["report", "--route", str(route)])

    assert result.exit_code == 0, result.output
    assert (route / "out" / "report.json").exists()
    assert not (Path("out") / "report.json").resolve().is_relative_to(tmp_path)


def test_view_defaults_to_a_view_inside_the_route(tmp_path: Path):
    route = tmp_path / "route"
    route.mkdir()
    _write_route(route)

    result = runner.invoke(app, ["view", "--route", str(route), "--no-serve"])

    assert result.exit_code == 0, result.output
    assert (route / "view" / "index.html").exists()


def test_an_explicit_out_still_wins(tmp_path: Path):
    route = tmp_path / "route"
    route.mkdir()
    _write_route(route)
    elsewhere = tmp_path / "elsewhere"

    result = runner.invoke(
        app, ["report", "--route", str(route), "--out", str(elsewhere)]
    )

    assert result.exit_code == 0, result.output
    assert (elsewhere / "report.json").exists()
    assert not (route / "out").exists()
