"""Command-line interface.

`irap report` is the main entry point. It consumes cached detection artifacts
and therefore needs no GPU, which is what lets the decision logic and every
sensitivity sweep be reproduced on an ordinary laptop.
"""

from __future__ import annotations

from pathlib import Path

import typer

from irap_screening.config import load_config
from irap_screening.decision.report import write_report
from irap_screening.pipeline import run_pipeline

app = typer.Typer(help="Dashcam and IMU based iRAP road safety screening.")


@app.callback()
def _main() -> None:
    """Force multi-command mode.

    Typer collapses an app with a single registered command into a bare command,
    which breaks subcommand dispatch. Registering any callback keeps it a group.
    Do not add invoke_without_command=True: that also disables help-on-no-args,
    so a bare `irap` would exit silently.
    """


RouteOption = typer.Option(..., "--route", help="Route directory to analyse.")
OutOption = typer.Option(Path("out"), "--out", help="Output directory.")
SetOption = typer.Option(None, "--set", help="Override a threshold, as key=value.")
ProfileOption = typer.Option(None, "--config", help="Profile YAML overriding defaults.")
RouteIdOption = typer.Option("R01", "--route-id", help="Identifier used in segment ids.")
WeightsOption = typer.Option(..., "--weights", help="Path to the YOLO sign weight (.pt).")
DeviceOption = typer.Option("cpu", "--device", help="Inference device, e.g. cpu or 0.")
RawOption = typer.Option(..., "--raw", help="CLRerNet dump written by the Colab cell.")


def _require_route(route: Path) -> Path:
    route = Path(route)
    if not route.exists():
        typer.secho(f"Route directory not found: {route}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    return route


@app.command(name="report")
def report(
    route: Path = RouteOption,
    out: Path = OutOption,
    config: Path | None = ProfileOption,
    set_: list[str] | None = SetOption,
    route_id: str = RouteIdOption,
) -> None:
    """Run the full CPU layer and write report.json, segments.geojson and report.md."""
    route = _require_route(route)
    cfg = load_config(profile=config, overrides=set_)
    result = run_pipeline(route, cfg, route_id=route_id)
    write_report(out, result)
    typer.echo(f"Wrote {len(result['segments'])} segments to {out}")


@app.command(name="detect-signs")
def detect_signs(
    route: Path = RouteOption,
    weights: Path = WeightsOption,
    config: Path | None = ProfileOption,
    set_: list[str] | None = SetOption,
    device: str = DeviceOption,
) -> None:
    """Run the YOLO sign detector over a route and cache detections/sign.jsonl.

    Requires the optional runners extra (`uv sync --extra runners`). This is the
    only command that needs a model; once it has written its artifact, every
    later `irap report` and threshold sweep runs from the cache.
    """
    from irap_screening.runners.sign import run as run_sign_runner

    route = _require_route(route)
    cfg = load_config(profile=config, overrides=set_)
    out_path = run_sign_runner(route, weights, cfg, device=device)
    typer.echo(f"Wrote {out_path}")


@app.command(name="import-lanes")
def import_lanes(
    route: Path = RouteOption,
    raw: Path = RawOption,
) -> None:
    """Convert a CLRerNet dump into detections/lane.jsonl.

    CLRerNet runs in an mmcv-pinned environment this project does not reproduce,
    so inference happens on Colab and only the numbers come back. The coordinate
    mapping back to original-image pixels happens here, where it is tested.
    """
    from irap_screening.runners.lane import export

    route = _require_route(route)
    if not Path(raw).exists():
        typer.secho(f"Dump not found: {raw}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    try:
        out_path = export(raw, route)
    except ValueError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote {out_path}")


if __name__ == "__main__":
    app()
