# iRAP Screening Pipeline

A low-cost screening tool that turns dashcam imagery and smartphone IMU logs
into per-segment iRAP mid-block attribute flags and gap-based recommendations.

## Architecture

GPU detectors and the CPU fusion layer are decoupled through on-disk artifacts.
Detectors write `detections/*.jsonl`; everything downstream runs CPU-only. This
keeps CLRerNet's mmcv-pinned environment away from ultralytics, and lets the
decision logic be reproduced without a GPU. Because detections are cached on
disk, re-running the decision layer against different thresholds costs seconds
rather than re-running inference.

## Quick start

```bash
uv sync --extra dev
uv run irap report --route fixtures/route_demo
```

This runs the full decision layer against the committed fixture. No GPU, no
external data.

## Commands

| Command | Purpose |
| --- | --- |
| `irap report --route DIR` | Full CPU layer; writes JSON, GeoJSON and Markdown into `DIR/out` |
| `irap detect-signs --route DIR --weights W.pt` | Runs the YOLO sign detector; writes `detections/sign.jsonl` |
| `irap import-lanes --route DIR --raw dump.json` | Converts a CLRerNet dump; writes `detections/lane.jsonl` |
| `irap view --route DIR` | Builds and serves a local page for inspecting the results |

## Detection runners

`irap detect-signs` is the only command that needs a model. It lives behind an
optional extra so the CPU layer never installs torch:

```bash
uv sync --extra runners
uv run irap detect-signs --route DIR --weights path/to/best.pt
```

`scripts/run_detectors.ps1` wraps both detectors and the report in one command:

```powershell
.\scripts\run_detectors.ps1 -Route D:\routes\kuching_01 -Report
```

It refuses to run without a CLRerNet dump unless `-SkipLanes` is passed, because
a route with no `lane.jsonl` is not an error downstream -- it reads as a road
with no markings anywhere and grades critical throughout.

CPU inference is adequate for screening-rate imagery. The runner refuses to run
against a weight whose class names are not exactly the five the decision layer
matches on -- a drifted name would flow through every layer cleanly and then
report present signage as missing.

Lane detections come from CLRerNet, which pins torch 2.2 and mmcv 2.2.0 and so
gets an environment of its own under `.clrernet/`, never the project's:

```powershell
.\scripts\clrernet\setup.ps1
.\.clrernet\venv\Scripts\python.exe scripts\clrernet\dump.py --route DIR
uv run irap import-lanes --route DIR --raw DIR\clrernet_raw.json
```

`setup.ps1` installs prebuilt wheels only. CLRerNet's one compiled piece, a CUDA
lane NMS that needs `nvcc`, is replaced by the NumPy port in
`scripts/clrernet/nms`, which `tests/test_clrernet_nms.py` checks against a
statement-by-statement transliteration of the original kernel.
`notebooks/clrernet_export.ipynb` does the same job on Colab and writes the same
`clrernet_raw.json`.

The split is deliberate. `dump.py` and the notebook only write numbers down; the
mapping from CLRerNet's normalised coordinates back to original-image pixels is
the part that can be plausibly wrong, so it lives in the tested codebase rather
than beside the model. `tests/test_notebook_export.py` pins the notebook to the
exporter: it executes the notebook's dump and overlay cells against stubs and
fails if either drifts.

## Viewer

`irap view` is a verification tool: it puts each segment's flags next to the
frames that produced them, with the detections drawn on.

```bash
uv run irap view --route fixtures/route_demo
```

It builds `view/`, serves it on localhost and opens a browser. Arrow keys step
through a segment's frames. Overlays are rendered in Python rather than on a
canvas in the browser, so the coordinate arithmetic keeps a single tested
implementation instead of a second, unwatched one in JavaScript.

### Upload and run

Pointing `irap view` at a directory of routes also puts an "Upload route"
control on the page:

```powershell
uv run irap view --route routes
```

Choose a zip of a prepared route (`frames/`, `frames.csv`, `gps.csv`, `imu.csv`,
either at the top level or inside one folder). The server unpacks it beside the
other routes under the zip's file name, runs CLRerNet, the lane import, the sign
detector and the report over it, and the page switches to the new route when
the job ends. Everything is written into the route's own directory, exactly as
a manual run leaves it, so `irap report` can be re-run on it afterwards.

The control appears only when `best.pt`, the `.clrernet` environment and
`scripts/clrernet/dump.py` are found from the working directory; otherwise the
server says what is missing and serves the page read-only. One job runs at a
time, and a route name is never overwritten: delete its directory to retry.

The sign detector runs on the CPU unless `--sign-device 0` is passed, which
needs a CUDA build of torch in the project environment. On a 117-frame route
that is about 55 s of a 70 s job.

## Configuration

All thresholds live in `configs/default.yaml`. Override one inline:

```bash
uv run irap report --route fixtures/route_demo --set curvature.threshold_m=350
```

Output defaults beside the data it describes -- `DIR/out` for the report,
`DIR/view` for the viewer -- so two routes cannot overwrite each other. Pass
`--out` to put it elsewhere.

## Modules

| Module | iRAP attribute | Method |
| --- | --- | --- |
| Lane marking | Delineation | Weber contrast at CLRerNet lane points |
| Curvature | Curvature | kappa = omega / v, gyro bias corrected, low-speed gated |
| Grade | Grade | Complementary-filter pitch from accelerometer |
| Road condition | Road Condition | Declared stub: reports not_assessed pending camera intrinsics and mounting height |
| Sign and marking | (supporting) | Existence facts for gap-check only; produces no flag |

## Known limitations

- Curvature (400 m) and grade (7.5 percent) thresholds are provisional, pending
  sensitivity analysis.
- Pothole concavity depth cannot be measured from a single camera, so iRAP's
  "one defect deeper than 5 cm is Poor" rule is not implemented.
- Road Condition is a declared stub in this build. It reports `not_assessed` for
  every segment so the gap stays visible in the coverage statistics rather than
  passing unnoticed; defect sizing needs camera intrinsics and mounting height.
- Rutting is out of scope; it needs continuous deformation measurement along the
  wheel path.
- Advance warning signs are checked at a point 230 m upstream of the hazard, so a
  hazard within the first 230 m of a recorded route has no observable upstream
  window and its advance sign is always reported missing. This is conservative
  rather than wrong, but it means route-start segments carry a known false
  positive for `advance_bend_sign`.
- The committed fixture contains no `sign.jsonl`, and its `lane.jsonl` is
  synthesised rather than produced by CLRerNet. The fixture exercises the
  decision layer; it is not evidence about detector accuracy.
- This project does not reproduce full iRAP Star Ratings: there is no crash data
  available for FSI calibration.
