"""Run CLRerNet over a route's frames and write clrernet_raw.json, locally.

The local counterpart of notebooks/clrernet_export.ipynb (Blocks 3-7): same
preprocessing, same geometry, same payload, so `irap import-lanes` cannot tell
which one produced a dump. Like the notebook it does no coordinate arithmetic;
that stays in irap_screening.runners.lane.

Runs in the separate CLRerNet environment (torch 2.2 + mmcv 2.2.0), never the
project's own:

    .clrernet/venv/Scripts/python scripts/clrernet/dump.py --route ROUTE_DIR
    uv run irap import-lanes --route ROUTE_DIR --raw ROUTE_DIR/clrernet_raw.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_CLRERNET = HERE.parents[1] / ".clrernet" / "CLRerNet"

WEIGHTS_NAME = "clrernet_culane_dla34_ema.pth"
WEIGHTS_URL = f"https://github.com/hirotomusiker/CLRerNet/releases/download/v0.1.0/{WEIGHTS_NAME}"
CONFIG = "configs/clrernet/culane/clrernet_culane_dla34_ema.py"

MODEL_W, MODEL_H = 800, 320  # network input size the crop is resized to
ORI_IMG_W, ORI_IMG_H = 1241, 376  # CLRerNet's reference geometry
# Permissive on purpose, and it must stay at or below the pipeline's
# lane.clrernet_conf (0.41): the floor is baked into the cached dump, and
# irap import-lanes rejects a dump that violates this.
CONF_THRESHOLD = 0.10
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def build_payload(frame_results: list[dict], meta: dict) -> dict:
    """The dump RawLaneDump validates. Mirrors the notebook's Block 7."""
    frames = []
    for entry in frame_results:
        lanes, scores = entry["lanes"], entry["scores"]
        # lanes and scores are parallel arrays; a length mismatch would mis-pair
        # confidences, which the lane module filters on, undetectably downstream.
        assert len(lanes) == len(scores), (
            f"{entry['filename']}: {len(lanes)} lanes but {len(scores)} scores"
        )
        frames.append({
            "filename": entry["filename"],
            "source_wh": entry["source_wh"],
            "lanes": [
                {
                    "conf": float(score),
                    "points_norm": [
                        [float(x), float(y)]
                        for x, y in (lane.points if hasattr(lane, "points") else lane)
                    ],
                }
                for lane, score in zip(lanes, scores, strict=True)
            ],
        })
    return {"meta": meta, "frames": frames}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--route", type=Path, required=True, help="directory holding frames/")
    parser.add_argument("--out", type=Path, help="default: ROUTE/clrernet_raw.json")
    parser.add_argument("--clrernet", type=Path, default=DEFAULT_CLRERNET)
    parser.add_argument("--crop-top", type=float, default=0.25,
                        help="fraction of image height removed from the top")
    parser.add_argument("--crop-bottom", type=float, default=0.0,
                        help="fraction removed from the bottom (bonnet, dashboard)")
    parser.add_argument("--device", default=None, help="default: cuda:0 if available, else cpu")
    args = parser.parse_args()

    route = args.route.resolve()
    clrernet = args.clrernet.resolve()
    out = (args.out or route / "clrernet_raw.json").resolve()
    images = sorted(p for p in (route / "frames").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images found in {route / 'frames'}")
    if not (clrernet / CONFIG).exists():
        raise SystemExit(f"No CLRerNet checkout at {clrernet}; run scripts/clrernet/setup.ps1")

    # The pure-NumPy `nms` package beside this file must win over anything else
    # of that name; CLRerNet's head imports it by bare name.
    sys.path[:0] = [str(HERE), str(clrernet), str(clrernet / "libs")]
    import cv2
    import numpy as np
    import torch
    from mmdet.registry import MODELS
    from mmdet.structures import DetDataSample
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import load_checkpoint

    # Ampere and newer GPUs default to TF32 convolutions, which makes scores
    # depend on which GPU ran the model. A cached dump should not.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    weights = clrernet.parent / WEIGHTS_NAME
    if not weights.exists():
        print(f"Downloading {WEIGHTS_NAME} (~63 MB) ...")
        urllib.request.urlretrieve(WEIGHTS_URL, weights)

    # cut_height is DERIVED, never typed in: CLRerNet spreads its output over
    # the rows it believes were kept, so cut_height/ORI_IMG_H must equal crop_top.
    cut_height = int(round(ORI_IMG_H * args.crop_top))
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    # mmdet's init_detector does exactly this and then builds the CULane test
    # dataset to pick a display palette, which needs the dataset on disk and
    # albumentations installed. Neither is used for inference.
    config = Config.fromfile(str(clrernet / CONFIG))
    init_default_scope(config.get("default_scope", "mmdet"))
    model = MODELS.build(config.model)
    load_checkpoint(model, str(weights), map_location="cpu")
    model.to(device).eval()
    model.bbox_head.test_cfg.conf_threshold = CONF_THRESHOLD
    model.bbox_head.test_cfg.ori_img_h = ORI_IMG_H
    model.bbox_head.test_cfg.ori_img_w = ORI_IMG_W
    model.bbox_head.test_cfg.cut_height = cut_height
    print(f"Model on {device}; {len(images)} frames; crop top {args.crop_top:.0%}, "
          f"bottom {args.crop_bottom:.0%}; cut_height {cut_height}")

    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

    def run(img_bgr):
        h, w = img_bgr.shape[:2]
        crop_top_px = int(h * args.crop_top)
        crop_bottom_px = int(h * args.crop_bottom)
        crop_end = h - crop_bottom_px if crop_bottom_px > 0 else h
        cropped = img_bgr[crop_top_px:crop_end, 0:w]
        resized = cv2.resize(cropped, (MODEL_W, MODEL_H), interpolation=cv2.INTER_LINEAR)
        normalised = (resized.astype(np.float32) - mean) / std
        tensor = torch.from_numpy(normalised).permute(2, 0, 1).unsqueeze(0).to(device)
        sample = DetDataSample()
        sample.set_metainfo({
            "img_shape": (MODEL_H, MODEL_W),
            "ori_shape": (MODEL_H, MODEL_W),
            "scale_factor": 1.0,
        })
        with torch.no_grad():
            return model.predict(tensor, [sample])[0]

    frame_results = []
    started = time.perf_counter()
    for index, path in enumerate(images):
        # CLRerNet was trained on colour CULane; force 3-channel BGR.
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"Failed to read image: {path}")
        result = run(img)
        scores = result.get("scores", [])
        if hasattr(scores, "cpu"):
            scores = scores.cpu().tolist()
        frame_results.append({
            "filename": path.name,
            "source_wh": [int(img.shape[1]), int(img.shape[0])],
            "lanes": result.get("lanes", []) or [],
            "scores": [float(s) for s in scores],
        })
        if (index + 1) % 50 == 0 or index == len(images) - 1:
            print(f"  [{index + 1}/{len(images)}]", flush=True)
    elapsed = time.perf_counter() - started

    payload = build_payload(frame_results, {
        "crop_top": float(args.crop_top),
        "crop_bottom": float(args.crop_bottom),
        "cut_height": cut_height,
        "ori_img_h": ORI_IMG_H,
        "ori_img_w": ORI_IMG_W,
        "conf_threshold": CONF_THRESHOLD,
        "model_wh": [MODEL_W, MODEL_H],
        "model": "clrernet_culane_dla34_ema",
        "sampling_interval": 1,
    })
    total_lanes = sum(len(f["lanes"]) for f in payload["frames"])
    if total_lanes == 0:
        raise SystemExit(
            "no lanes on any frame; the local exporter rejects such a dump because "
            "zero lanes grades every segment as the worst contrast band"
        )
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"Wrote {out}: {len(images)} frames, {total_lanes} lanes, "
          f"{elapsed:.1f}s ({len(images) / elapsed:.1f} frames/s)")


if __name__ == "__main__":
    main()
