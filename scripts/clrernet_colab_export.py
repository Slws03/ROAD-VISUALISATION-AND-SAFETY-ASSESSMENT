"""Colab-side dump cell for CLRerNet. Paste this into the demo notebook.

Run it after the notebook's inference block, with `frame_results`, `model` and
the crop constants already defined. It writes one JSON file that
`irap import-lanes` converts into canonical detections locally.

This cell deliberately does nothing but serialise numbers. Every coordinate
transform lives in `irap_screening.runners.lane`, where it is unit tested; the
mmcv-pinned Colab environment is not reproducible, so no arithmetic that could
be wrong should live in it.

Usage in the notebook:

    exec(open("clrernet_colab_export.py").read())

or simply copy the body of `dump_clrernet_results` into a cell and call it.
"""

import json
from pathlib import Path


def dump_clrernet_results(
    frame_results, model, crop_top, crop_bottom, image_dir, out_path, load_bgr,
    model_w=800, model_h=320,
):
    """Serialise CLRerNet output plus the geometry it was produced with.

    frame_results: the notebook's list of per-frame dicts, each carrying
        "filename" and the raw "_result_obj" returned by model.predict.
    load_bgr: the notebook's image loader, used only to record each frame's
        original size. The exporter needs it and it cannot be recovered later.
    model_w, model_h: the resize the notebook applied before inference. Pass
        them if the cv2.resize literal in run_clrernet was changed.
    """
    test_cfg = model.bbox_head.test_cfg
    frames = []

    for entry in frame_results:
        result = entry.get("_result_obj", {})
        lanes = result.get("lanes", []) or []
        scores = result.get("scores", [])
        if hasattr(scores, "cpu"):
            scores = scores.cpu().tolist()

        image = load_bgr(str(Path(image_dir) / entry["filename"]))
        height, width = image.shape[:2]

        frames.append(
            {
                "filename": entry["filename"],
                "source_wh": [int(width), int(height)],
                "lanes": [
                    {
                        "conf": float(score),
                        "points_norm": [
                            [float(x), float(y)]
                            for x, y in (
                                lane.points if hasattr(lane, "points") else lane
                            )
                        ],
                    }
                    # strict: lanes and scores are parallel arrays from one
                    # result. A length mismatch would mis-pair confidences,
                    # which modules/lane.py filters on and nothing can detect
                    # downstream. Fail here, where the user is watching.
                    for lane, score in zip(lanes, scores, strict=True)
                ],
            }
        )

    payload = {
        "meta": {
            "crop_top": float(crop_top),
            "crop_bottom": float(crop_bottom),
            "cut_height": int(test_cfg.cut_height),
            "ori_img_h": int(test_cfg.ori_img_h),
            "ori_img_w": int(test_cfg.ori_img_w),
            "conf_threshold": float(test_cfg.conf_threshold),
            "model_wh": [int(model_w), int(model_h)],
            "model": "clrernet_culane_dla34_ema",
        },
        "frames": frames,
    }

    out_path = Path(out_path)
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    total = sum(len(frame["lanes"]) for frame in frames)
    print(f"Wrote {out_path}: {len(frames)} frames, {total} lanes")
    return out_path


# Example call inside the notebook, after Block 6:
#
# dump_clrernet_results(
#     frame_results, model, CROP_TOP, CROP_BOTTOM, IMAGE_DIR,
#     OUTPUT_DIR / "clrernet_raw.json", load_bgr,
# )
