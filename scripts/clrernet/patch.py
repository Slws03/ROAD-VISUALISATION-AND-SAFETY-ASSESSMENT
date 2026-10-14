"""Patch CLRerNet and mmdet for inference-only use. Idempotent.

The same four patches as Block 1 of notebooks/clrernet_export.ipynb. They edit
installed files, so reinstalling mmdet or re-cloning CLRerNet undoes them; run
this again afterwards. A patch whose target text is neither present nor already
patched fails loudly: upstream has changed and the edit needs a human.

    .clrernet/venv/Scripts/python scripts/clrernet/patch.py .clrernet/CLRerNet
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    # read_text normalises CRLF, which a Windows git checkout may have introduced
    # and which would otherwise defeat the multi-line match below.
    text = path.read_text(encoding="utf-8")
    if new in text:
        print(f"[ok]      {label} (already patched)")
    elif old in text:
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"[patched] {label}")
    else:
        raise SystemExit(f"[FAIL] {label}: expected text not found in {path}")


def main(clrernet: Path) -> None:
    # Located without importing: mmdet's __init__ carries the very assert that
    # patch 1 removes.
    spec = importlib.util.find_spec("mmdet")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit("mmdet is not installed in this interpreter")
    mmdet = Path(spec.submodule_search_locations[0])
    if not (clrernet / "libs").is_dir():
        raise SystemExit(f"{clrernet} is not a CLRerNet checkout")

    # 1. mmdet asserts a narrow mmcv range; mmcv 2.2.0 is newer on purpose.
    replace_once(
        mmdet / "__init__.py",
        "assert (mmcv_version >= digit_version(mmcv_minimum_version)",
        "assert (True or mmcv_version >= digit_version(mmcv_minimum_version)",
        "mmdet mmcv version check",
    )

    # 2. The CULane metric is evaluation-only and pulls in extra dependencies.
    replace_once(
        clrernet / "libs/datasets/__init__.py",
        "from .metrics.culane_metric import CULaneMetric  # noqa: F401",
        "# from .metrics.culane_metric import CULaneMetric  # disabled for inference",
        "culane_metric import",
    )

    # 3. The DLA backbone downloads ImageNet weights the checkpoint already has.
    dla = clrernet / "libs/models/backbones/dla.py"
    replace_once(
        dla,
        """    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        # fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        self.load_state_dict(model_weights, strict=False)
        # self.fc = fc""",
        """    def load_pretrained_model(self, data='imagenet', name='dla34', hash='ba72cf86'):
        pass  # disabled for inference-only mode""",
        "DLA load_pretrained_model",
    )
    replace_once(
        dla,
        "            pretrained=pretrained, levels=levels, in_channels=in_channels",
        "            pretrained=False, levels=levels, in_channels=in_channels",
        "DLA pretrained=False",
    )

    # 4. mmdet imports transformers for GroundingDINO, which breaks on
    #    torch < 2.4 when transformers happens to be installed.
    vlfuse = mmdet / "models/utils/vlfuse_helper.py"
    if vlfuse.exists():
        replace_once(
            vlfuse,
            "except ImportError:",
            "except (ImportError, NameError, AttributeError):",
            "vlfuse_helper import guard",
        )

    for cache in clrernet.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    print("CLRerNet patched for inference.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(Path(sys.argv[1]).resolve())
