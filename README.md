# iRAP Screening Pipeline

A low-cost screening tool that turns one smartphone's camera stills, GPS and
IMU logs into per-segment iRAP mid-block codes (delineation, curvature, grade,
quality of curve) and gap-based recommendations that follow JKR ATJ 2B/85.

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

## Route layout

A route directory holds `frames/`, `frames.csv`, `gps.csv` and `imu.csv`. The
detectors add `detections/lane.jsonl` and `detections/sign.jsonl`. Converters
in `scripts/` produce this layout from raw data:

| Script | Source |
| --- | --- |
| `steven_to_route.py` | Neoun multi-report export with gyroscope; levels the IMU on flat road |
| `neoun_to_route.py` | Single Neoun RPT-* report, accelerometer only; curvature from GPS bearing |
| `kitti_to_route.py` | KITTI raw drive (oxts) |
| `make_fixture.py` | Rebuilds `fixtures/route_demo` |
| `paste_signs.py` | Composites signs into a copy of a route (synthetic scenario only) |

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
| `modules/lane.py` | Delineation | Weber contrast at CLRerNet lane points. Poor when under half of a segment's lines reach 0.25, or the median frame shows fewer than 2 lines |
| `modules/curvature.py` | Curvature | Heading change from the bias-corrected gyro over a 50 m window, sharpest window inside the segment; critical below 400 m; GPS bearing cross-check |
| `modules/grade.py` | Grade | Accelerometer pitch with dv/dt from speed removed, axes averaged over 30 m; critical at 7.5% held over 10 m; no gyro; GPS altitude cross-check |
| `modules/marking.py` | (supporting) | Accepted sign records per segment for gap-check; produces no flag |
| `decision/curve_quality.py` | Quality of curve | N/A where the radius is 800 m or more; Poor where delineation is poor; else Adequate with an advance sign or chevrons |
| `decision/gapcheck.py` | (recommendations) | Rule 1: curve series (< 180 m apart) need one JKR sign 230 m ahead; chevrons are checked on each curve. Rule 2: adjacent poor-delineation segments form one stretch to repaint |

## Known limitations

- The curvature window (50 m), grade window (30 m) and delineation thresholds
  were tuned on the 14 Samarahan routes and KITTI drive 0042 and have not been
  tested on held-out routes. `configs/default.yaml` records the evidence for
  each value.
- Delineation counts lines but does not identify the centre line, so a narrow
  road with two edge lines and no centre line reads Adequate.
- The sign weight has no Winding Road (WD.11) class; such signs are detected as
  Other and do not count as an advance sign.
- Advance warning signs are checked at a point 230 m before the first curve of a
  series, so a series within the first 230 m of a recorded route has no photos
  of its sign's position. Its advance sign is reported `unchecked`, not missing,
  and the recommendation says it could not be checked.
- The committed fixture contains no `sign.jsonl`, and its `lane.jsonl` is
  synthesised rather than produced by CLRerNet. The fixture exercises the
  decision layer; it is not evidence about detector accuracy.
- This project does not reproduce full iRAP Star Ratings: there is no crash data
  available for FSI calibration.
