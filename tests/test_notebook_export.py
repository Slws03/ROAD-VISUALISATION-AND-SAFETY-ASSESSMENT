"""The Colab notebook is the only producer of dumps, and it lives outside CI.

Nothing runs it here -- it needs mmcv, a GPU and Google Drive. But two of its
cells are coupled to this codebase and will rot silently if that coupling is not
pinned: the cell that builds the dump payload must satisfy RawLaneDump, and the
sanity-overlay cell reimplements the inverse mapping, so it must agree with
to_original_points. A drifted overlay is worse than none: it is the check the
user trusts to confirm the mapping is right.

The cells are extracted by their block markers and executed against stubs.
"""

import ast
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from irap_screening.runners.lane import RawLaneDump, to_original_points

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "clrernet_export.ipynb"

CROP_TOP = 0.25
CROP_BOTTOM = 0.20
ORI_IMG_W, ORI_IMG_H = 1241, 376
CUT_HEIGHT = int(round(ORI_IMG_H * CROP_TOP))
SOURCE_W, SOURCE_H = 1920, 1080
# The last point sits off the left edge, which CLRerNet does on curves.
POINTS = [(0.30, 0.30), (0.35, 0.60), (0.42, 0.95), (-0.05, 1.00)]


class _FakeLane:
    """Stands in for CLRerNet's Lane, which exposes normalised points."""

    def __init__(self, points):
        self.points = np.array(points)


def _code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _block(marker):
    matches = [source for source in _code_cells() if marker in source]
    assert len(matches) == 1, f"expected exactly one cell containing {marker!r}"
    return matches[0]


def _base_namespace(out_dir):
    return {
        "frame_results": [
            {
                "filename": "a.png",
                "source_wh": [SOURCE_W, SOURCE_H],
                "lanes": [_FakeLane(POINTS)],
                "scores": [0.83],
            }
        ],
        "CROP_TOP": CROP_TOP,
        "CROP_BOTTOM": CROP_BOTTOM,
        "CUT_HEIGHT": CUT_HEIGHT,
        "ORI_IMG_H": ORI_IMG_H,
        "ORI_IMG_W": ORI_IMG_W,
        "CONF_THRESHOLD": 0.10,
        "MODEL_W": 800,
        "MODEL_H": 320,
        "SAMPLING_INTERVAL": 1,
        "OUTPUT_DIR": out_dir,
    }


def test_notebook_is_valid_and_carries_no_outputs():
    """Committed notebooks must be diffable; stored outputs are megabytes of noise."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    assert not any(cell.get("outputs") for cell in notebook["cells"])


def test_every_code_cell_parses():
    """A syntax error would only surface after the multi-minute install block."""
    for index, source in enumerate(_code_cells()):
        try:
            ast.parse(source)
        except SyntaxError as error:  # pragma: no cover - failure path
            pytest.fail(f"code cell {index} does not parse: {error}")


def test_dump_cell_produces_a_payload_the_exporter_accepts(tmp_path):
    """The notebook's only job is to emit something irap import-lanes can read."""
    namespace = _base_namespace(tmp_path)
    exec(compile(_block("Block 7"), "block7", "exec"), namespace)  # noqa: S102

    raw = json.loads((tmp_path / "clrernet_raw.json").read_text(encoding="utf-8"))
    dump = RawLaneDump.model_validate(raw)

    assert dump.meta.cut_height == CUT_HEIGHT
    assert dump.meta.crop_top == CROP_TOP
    assert dump.meta.model_wh == (800, 320)
    assert len(dump.frames) == 1
    assert dump.frames[0].lanes[0].conf == pytest.approx(0.83)


def test_sanity_overlay_matches_the_exporter_mapping(tmp_path, monkeypatch):
    """The overlay is the user's evidence that the mapping is right.

    If it drifted from to_original_points it would keep looking convincing while
    proving nothing about the dump actually being converted.
    """
    namespace = _base_namespace(tmp_path)
    exec(compile(_block("Block 7"), "block7", "exec"), namespace)  # noqa: S102
    dump = RawLaneDump.model_validate(
        json.loads((tmp_path / "clrernet_raw.json").read_text(encoding="utf-8"))
    )

    drawn: list[tuple[int, int]] = []

    def _spy(image, point_a, point_b, *args, **kwargs):
        drawn.extend([point_a, point_b])
        return image

    monkeypatch.setattr(cv2, "line", _spy)
    monkeypatch.setitem(
        sys.modules,
        "matplotlib.pyplot",
        types.SimpleNamespace(
            figure=lambda **kwargs: None,
            imshow=lambda *args, **kwargs: None,
            title=lambda *args, **kwargs: None,
            axis=lambda *args, **kwargs: None,
            show=lambda: None,
        ),
    )
    namespace.update(
        cv2=cv2,
        np=np,
        load_bgr=lambda path: np.zeros((SOURCE_H, SOURCE_W, 3), np.uint8),
        IMAGE_DIR=tmp_path,
    )
    exec(compile(_block("Block 8"), "block8", "exec"), namespace)  # noqa: S102

    expected = to_original_points(POINTS, (SOURCE_W, SOURCE_H), dump.meta)
    in_frame = {(int(x), int(y)) for x, y in expected if 0 <= x < SOURCE_W}

    assert in_frame <= set(drawn)
    off_frame = [(x, y) for x, y in expected if not 0 <= x < SOURCE_W]
    assert off_frame, "the fixture must include an off-frame point to exercise the skip"
    assert not any(x < 0 for x, _ in drawn), "off-frame points must not be drawn"
