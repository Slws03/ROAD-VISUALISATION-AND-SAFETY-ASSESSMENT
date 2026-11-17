"""The automatic crop: CLRerNet's training framing plus a margin above, at any size."""

import importlib.util
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "clrernet" / "dump.py"
_spec = importlib.util.spec_from_file_location("clrernet_dump", _PATH)
dump = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dump)


@pytest.mark.parametrize("wh", [(1920, 1080), (1280, 720), (960, 544), (1241, 376), (1640, 590)])
def test_the_kept_band_is_culanes_plus_the_margin_and_passes_the_importers_check(wh):
    width, height = wh
    crop_top = dump.auto_crop_top(width, height)
    kept_rows = height * (1 - crop_top)
    # Snapping to a reference row moves the band by at most one such row.
    expected = width / dump.TRAIN_ASPECT + dump.HORIZON_MARGIN * height
    assert kept_rows == pytest.approx(expected, abs=height / dump.ORI_IMG_H)
    cut_height = int(round(dump.ORI_IMG_H * crop_top))  # as main() derives it
    assert abs(cut_height / dump.ORI_IMG_H - crop_top) <= 1e-3  # RawLaneMeta's rule


def test_culane_itself_is_cut_where_culane_was_cut_less_the_margin():
    expected = 270 / 590 - dump.HORIZON_MARGIN
    assert dump.auto_crop_top(1640, 590) == pytest.approx(expected, abs=1 / dump.ORI_IMG_H)


def test_neuon_phone_frames_are_cut_near_the_tuned_055():
    for wh in [(1920, 1080), (1280, 720), (960, 544)]:
        assert dump.auto_crop_top(*wh) == pytest.approx(0.55, abs=0.01)


def test_a_bonnet_cut_from_the_bottom_moves_the_band_up_not_its_height():
    plain, bonnet = dump.auto_crop_top(1920, 1080), dump.auto_crop_top(1920, 1080, 0.1)
    assert plain - bonnet == pytest.approx(0.1, abs=1 / dump.ORI_IMG_H)


def test_a_frame_wider_than_the_band_keeps_every_row():
    assert dump.auto_crop_top(4000, 500) == 0.0


def test_portrait_footage_is_refused_rather_than_cut_to_a_sliver():
    with pytest.raises(SystemExit):
        dump.auto_crop_top(1080, 1920)
