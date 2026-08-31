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


@app.callback(invoke_without_command=True)
def main() -> None:
    """Dashcam and IMU based iRAP road safety screening."""
    pass


RouteOption = typer.Option(..., "--route", help="Route directory to analyse.")
OutOption = typer.Option(Path("out"), "--out", help="Output directory.")
SetOption = typer.Option(None, "--set", help="Override a threshold, as key=value.")
ProfileOption = typer.Option(None, "--config", help="Profile YAML overriding defaults.")
RouteIdOption = typer.Option("R01", "--route-id", help="Identifier used in segment ids.")


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


if __name__ == "__main__":
    app()
