"""The automatic crop must give CLRerNet the framing it was trained on, at any size."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "clrernet" / "dump.py"
_spec = importlib.util.spec_from_file_location("clrernet_dump", _PATH)
dump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dump)


@pytest.mark.parametrize("wh", [(1920, 1080), (1280, 720), (960, 544), (1241, 376), (1640, 590)])
def test_the_kept_band_has_culanes_aspect_and_passes_the_importers_check(wh):
    width, height = wh
    crop_top = dump.auto_crop_top(width, height)
    kept_rows = height * (1 - crop_top)
    # Snapping to a reference row moves the band by at most one such row.
    assert kept_rows == pytest.approx(width / dump.TRAIN_ASPECT, abs=height / dump.ORI_IMG_H)
    cut_height = int(round(dump.ORI_IMG_H * crop_top))  # as main() derives it
    assert abs(cut_height / dump.ORI_IMG_H - crop_top) <= 1e-3  # RawLaneMeta's rule


def test_culane_itself_is_cut_where_culane_was_cut():
    assert dump.auto_crop_top(1640, 590) == pytest.approx(270 / 590, abs=1 / dump.ORI_IMG_H)


def test_a_bonnet_cut_from_the_bottom_moves_the_band_up_not_its_height():
    plain, bonnet = dump.auto_crop_top(1920, 1080), dump.auto_crop_top(1920, 1080, 0.1)
    assert plain - bonnet == pytest.approx(0.1, abs=1 / dump.ORI_IMG_H)


def test_a_frame_wider_than_the_band_keeps_every_row():
    assert dump.auto_crop_top(4000, 500) == 0.0


def test_portrait_footage_is_refused_rather_than_cut_to_a_sliver():
    with pytest.raises(SystemExit):
        dump.auto_crop_top(1080, 1920)
