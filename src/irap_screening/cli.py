"""Command-line interface.

`irap report` is the main entry point. It consumes cached detection artifacts
and therefore needs no GPU, which is what lets the decision logic and every
sensitivity sweep be reproduced on an ordinary laptop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from irap_screening.config import Settings, load_config
from irap_screening.decision.report import write_report
from irap_screening.pipeline import run_pipeline
from irap_screening.view.jobs import Rebuild, Upload

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
OutOption = typer.Option(
    None, "--out", help="Output directory. Default: an out/ inside the route."
)
SetOption = typer.Option(None, "--set", help="Override a threshold, as key=value.")
ProfileOption = typer.Option(None, "--config", help="Profile YAML overriding defaults.")
RouteIdOption = typer.Option("R01", "--route-id", help="Identifier used in segment ids.")
WeightsOption = typer.Option(..., "--weights", help="Path to the YOLO sign weight (.pt).")
DeviceOption = typer.Option("cpu", "--device", help="Inference device, e.g. cpu or 0.")
RawOption = typer.Option(..., "--raw", help="CLRerNet dump written by scripts/clrernet/dump.py.")
ViewOutOption = typer.Option(
    None, "--out", help="Viewer output directory. Default: a view/ inside the route."
)
PortOption = typer.Option(8000, "--port", help="Port for the local viewer server.")
NoServeOption = typer.Option(False, "--no-serve", help="Build the viewer, do not serve it.")
NoOpenOption = typer.Option(False, "--no-open", help="Serve without opening a browser.")
UploadWeightsOption = typer.Option(
    Path("best.pt"), "--weights", help="YOLO sign weight used for uploaded routes."
)
ClrernetPythonOption = typer.Option(
    Path(".clrernet/venv/Scripts/python.exe"),
    "--clrernet-python",
    help="Interpreter of the CLRerNet environment (scripts/clrernet/setup.ps1).",
)
SignDeviceOption = typer.Option(
    "cpu", "--sign-device", help="Sign detector device for uploaded routes, e.g. cpu or 0."
)
LANE_SCRIPT = Path("scripts/clrernet/dump.py")


def _require_route(route: Path) -> Path:
    route = Path(route)
    if not route.exists():
        typer.secho(f"Route directory not found: {route}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    return route


@app.command(name="report")
def report(
    route: Path = RouteOption,
    out: Path | None = OutOption,
    config: Path | None = ProfileOption,
    set_: list[str] | None = SetOption,
    route_id: str = RouteIdOption,
) -> None:
    """Run the full CPU layer and write report.json, segments.geojson and report.md."""
    route = _require_route(route)
    # Default beside the data it describes: a fixed directory meant every route
    # wrote to the same place and the last run replaced the rest.
    out = out or route / "out"
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
    config: Path | None = ProfileOption,
    set_: list[str] | None = SetOption,
) -> None:
    """Convert a CLRerNet dump into detections/lane.jsonl.

    CLRerNet runs in an mmcv-pinned environment of its own (scripts/clrernet,
    or the Colab notebook), and only the numbers come back. The coordinate
    mapping back to original-image pixels happens here, where it is tested.
    """
    from irap_screening.runners.lane import export

    route = _require_route(route)
    if not Path(raw).exists():
        typer.secho(f"Dump not found: {raw}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    try:
        out_path = export(raw, route, load_config(profile=config, overrides=set_))
    except ValueError as error:
        typer.secho(str(error), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote {out_path}")


def _upload_settings(
    routes_root: Path,
    out_dir: Path,
    cfg: Settings,
    profile: Path | None,
    overrides: list[str] | None,
    weights: Path,
    clrernet_python: Path,
    sign_device: str,
) -> Upload | None:
    """What an uploaded route's job will run, or None where it could not run.

    Decided once at startup: a page that offered the upload control and then
    failed halfway through a job would be worse than one that never offered it.
    """
    missing = [path for path in (weights, clrernet_python, LANE_SCRIPT) if not path.exists()]
    if missing:
        typer.echo(
            "Upload is off, missing: "
            + ", ".join(str(path) for path in missing)
            + ". Run from the repository root; scripts/clrernet/setup.ps1 builds the "
            "CLRerNet environment."
        )
        return None
    # The sign detector is started with this interpreter rather than through
    # `uv run`, and given the same configuration this server was started with.
    sign_command = [sys.executable, "-m", "irap_screening.cli", "detect-signs"]
    sign_command += ["--weights", str(weights.resolve()), "--device", sign_device]
    if profile:
        sign_command += ["--config", str(profile.resolve())]
    for override in overrides or []:
        sign_command += ["--set", override]
    return Upload(
        routes_root=routes_root,
        out_dir=out_dir,
        cfg=cfg,
        lane_command=[str(clrernet_python.resolve()), str(LANE_SCRIPT.resolve())],
        sign_command=sign_command,
    )


@app.command(name="view")
def view(
    route: Path = RouteOption,
    out: Path | None = ViewOutOption,
    config: Path | None = ProfileOption,
    set_: list[str] | None = SetOption,
    route_id: str = RouteIdOption,
    port: int = PortOption,
    no_serve: bool = NoServeOption,
    no_open: bool = NoOpenOption,
    weights: Path = UploadWeightsOption,
    clrernet_python: Path = ClrernetPythonOption,
    sign_device: str = SignDeviceOption,
) -> None:
    """Build and serve a local page for inspecting a route's results.

    A verification tool: it puts each segment's flags next to the frames that
    produced them, with the detections drawn on. It serves over HTTP rather than
    writing a file to double-click because the page fetches its data, which the
    browser refuses to do from a file:// URL.

    --route may also name a directory of routes (each an immediate
    subdirectory with its own frames.csv/gps.csv/imu.csv): every route found
    is built, and the page gets a switcher to jump between them.

    Serving a directory of routes also puts an "Upload route" control on the
    page: a zip of a prepared route is unpacked beside the others, both
    detectors and the report run over it, and it joins the switcher. That
    needs the sign weight and the local CLRerNet environment; without them
    the page is served read-only.
    """
    from irap_screening.view.export import discover_routes, export_routes, export_view, is_route_dir
    from irap_screening.view.serve import serve_directory, serve_routes

    route = _require_route(route)
    out = out or route / "view"
    cfg = load_config(profile=config, overrides=set_)

    if is_route_dir(route):
        out_dir = export_view(route, cfg, out, route_id=route_id)
        typer.echo(f"Built {out_dir}")
        if no_serve:
            typer.echo(f"Serve it with: python -m http.server -d {out_dir}")
            return
        serve_directory(out_dir, route, port=port, open_browser=not no_open)
        return

    routes = discover_routes(route)
    if not routes:
        typer.secho(f"No routes found under: {route}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    out_dir = export_routes(routes, cfg, out)
    typer.echo(f"Built {out_dir} ({len(routes)} route(s): {', '.join(sorted(routes))})")

    if no_serve:
        typer.echo(f"Serve it with: python -m http.server -d {out_dir}")
        return
    upload = _upload_settings(
        route, out_dir, cfg, config, set_, weights, clrernet_python, sign_device
    )
    rebuild = Rebuild(out_dir, cfg, profile=config, overrides=list(set_ or []), upload=upload)
    serve_routes(
        out_dir, routes, port=port, open_browser=not no_open, upload=upload, rebuild=rebuild
    )


if __name__ == "__main__":
    app()
