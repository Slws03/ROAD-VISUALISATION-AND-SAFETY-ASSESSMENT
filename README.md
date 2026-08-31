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
uv run irap report --route fixtures/route_demo --out out
```

This runs the full decision layer against the committed fixture. No GPU, no
external data.

## Commands

| Command | Purpose |
| --- | --- |
| `irap report --route DIR --out DIR` | Full CPU layer; writes JSON, GeoJSON and Markdown |
| `irap detect-signs --route DIR --weights W.pt` | Runs the YOLO sign detector; writes `detections/sign.jsonl` |
| `irap import-lanes --route DIR --raw dump.json` | Converts a CLRerNet dump; writes `detections/lane.jsonl` |

`irap sweep`, `irap validate` and `irap config show` are designed but not yet
implemented in this build.

## Detection runners

`irap detect-signs` is the only command that needs a model. It lives behind an
optional extra so the CPU layer never installs torch:

```bash
uv sync --extra runners
uv run irap detect-signs --route DIR --weights path/to/best.pt
```

CPU inference is adequate for screening-rate imagery. The runner refuses to run
against a weight whose class names are not exactly the five the decision layer
matches on -- a drifted name would flow through every layer cleanly and then
report present signage as missing.

Lane detections come from CLRerNet, whose mmcv-pinned environment is not
reproduced here. Inference runs on Colab; `scripts/clrernet_colab_export.py` is
a cell that serialises its raw output, and `irap import-lanes` converts that
dump locally:

```bash
uv run irap import-lanes --route DIR --raw clrernet_raw.json
```

The split is deliberate. The Colab cell only writes numbers down; the mapping
from CLRerNet's normalised coordinates back to original-image pixels is the part
that can be plausibly wrong, so it lives in the tested codebase rather than in
an environment that cannot be reproduced.

## Configuration

All thresholds live in `configs/default.yaml`. Override one inline:

```bash
uv run irap report --route fixtures/route_demo --set curvature.threshold_m=350
```

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
