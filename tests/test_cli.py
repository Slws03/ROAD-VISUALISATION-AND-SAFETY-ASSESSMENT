"""CLI surface. The main entry point must run without a GPU."""

import json
from pathlib import Path

from typer.testing import CliRunner

from irap_screening.cli import app
from tests.test_pipeline import _write_route

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
