"""FixtureSource reads the canonical CSV layout defined in the spec."""

from pathlib import Path

import pytest

from irap_screening.io.sources.fixture import FixtureSource


def _write_route(root: Path) -> None:
    (root / "frames").mkdir(parents=True)
    (root / "frames.csv").write_text(
        "frame_id,path,timestamp_utc,sha256\n"
        "f_000001,frames/f_000001.jpg,2026-08-31T00:00:00Z,aaaa1111\n"
        "f_000002,frames/f_000002.jpg,2026-08-31T00:00:01Z,bbbb2222\n",
        encoding="utf-8",
    )
    (root / "gps.csv").write_text(
        "timestamp_utc,lat,lon,speed_mps\n"
        "2026-08-31T00:00:00Z,1.5500000,110.3500000,16.7\n"
        "2026-08-31T00:00:01Z,1.5501500,110.3500000,16.7\n",
        encoding="utf-8",
    )
    (root / "imu.csv").write_text(
        "timestamp_utc,ax,ay,az,gx,gy,gz\n"
        "2026-08-31T00:00:00.00Z,0.0,0.0,9.81,0.0,0.0,0.0\n"
        "2026-08-31T00:00:00.02Z,0.0,0.0,9.81,0.0,0.0,0.0\n",
        encoding="utf-8",
    )


def test_reads_all_three_streams(tmp_path: Path):
    _write_route(tmp_path)
    source = FixtureSource(tmp_path)
    assert len(source.frames()) == 2
    assert len(source.gps()) == 2
    assert len(source.imu()) == 2


def test_speed_column_is_preserved_when_present(tmp_path: Path):
    _write_route(tmp_path)
    source = FixtureSource(tmp_path)
    assert source.gps()[0].speed_mps == pytest.approx(16.7)


def test_missing_speed_column_yields_none(tmp_path: Path):
    _write_route(tmp_path)
    (tmp_path / "gps.csv").write_text(
        "timestamp_utc,lat,lon\n"
        "2026-08-31T00:00:00Z,1.5500000,110.3500000\n",
        encoding="utf-8",
    )
    source = FixtureSource(tmp_path)
    assert source.gps()[0].speed_mps is None


def test_missing_file_raises_with_useful_path(tmp_path: Path):
    source = FixtureSource(tmp_path)
    with pytest.raises(FileNotFoundError, match="gps.csv"):
        source.gps()


def test_timestamps_are_timezone_aware(tmp_path: Path):
    _write_route(tmp_path)
    source = FixtureSource(tmp_path)
    assert source.frames()[0].timestamp_utc.tzinfo is not None


def test_empty_speed_value_yields_none(tmp_path: Path):
    """An empty cell must not be read as 0.0 -- that would look like a stopped vehicle."""
    _write_route(tmp_path)
    (tmp_path / "gps.csv").write_text(
        "timestamp_utc,lat,lon,speed_mps\n"
        "2026-08-31T00:00:00Z,1.5500000,110.3500000,\n",
        encoding="utf-8",
    )
    assert FixtureSource(tmp_path).gps()[0].speed_mps is None
