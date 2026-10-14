"""Upload handling and job sequencing for the viewer's upload control.

No models and no GPU: the two detector commands are plain argument lists, so
the tests substitute a Python one-liner for each.
"""

import zipfile
from pathlib import Path

import pytest

from irap_screening.view.jobs import Conflict, UploadError, extract_route, valid_route_id

ROUTE_FILES = {
    "frames.csv": b"x",
    "gps.csv": b"x",
    "imu.csv": b"x",
    "frames/f_000000.jpg": b"x",
}


def _zip(path: Path, files: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path


def test_a_route_zip_is_extracted_in_place(tmp_path):
    route = tmp_path / "routes" / "r1"

    extract_route(_zip(tmp_path / "r1.zip", ROUTE_FILES), route)

    assert (route / "frames.csv").exists()
    assert (route / "frames" / "f_000000.jpg").exists()


def test_a_single_wrapping_folder_is_stripped(tmp_path):
    """Explorer's "compress folder" nests the route one level down."""
    nested = {f"kuching_01/{name}": content for name, content in ROUTE_FILES.items()}
    route = tmp_path / "routes" / "r1"

    extract_route(_zip(tmp_path / "r1.zip", nested), route)

    assert (route / "frames.csv").exists()
    assert not (route / "kuching_01").exists()


def test_a_member_escaping_the_route_is_rejected_and_nothing_is_left(tmp_path):
    route = tmp_path / "routes" / "r1"
    files = {**ROUTE_FILES, "../../evil.txt": b"x"}

    with pytest.raises(UploadError, match="escapes"):
        extract_route(_zip(tmp_path / "r1.zip", files), route)

    assert not route.exists()
    assert not (tmp_path / "evil.txt").exists()


def test_a_zip_that_is_not_a_route_is_rejected_and_cleaned_up(tmp_path):
    route = tmp_path / "routes" / "r1"

    with pytest.raises(UploadError, match="not a route"):
        extract_route(_zip(tmp_path / "r1.zip", {"readme.txt": b"x"}), route)

    assert not route.exists()


def test_a_route_with_an_empty_frames_directory_is_rejected(tmp_path):
    files = {name: c for name, c in ROUTE_FILES.items() if not name.startswith("frames/")}
    route = tmp_path / "routes" / "r1"

    with pytest.raises(UploadError, match="not a route"):
        extract_route(_zip(tmp_path / "r1.zip", files), route)


def test_a_file_that_is_not_a_zip_is_rejected(tmp_path):
    bogus = tmp_path / "r1.zip"
    bogus.write_bytes(b"this is not a zip")

    with pytest.raises(UploadError, match="not a zip"):
        extract_route(bogus, tmp_path / "routes" / "r1")


def test_an_existing_route_is_never_overwritten(tmp_path):
    route = tmp_path / "routes" / "r1"
    route.mkdir(parents=True)
    (route / "keep.txt").write_text("mine")

    with pytest.raises(Conflict, match="already exists"):
        extract_route(_zip(tmp_path / "r1.zip", ROUTE_FILES), route)

    assert (route / "keep.txt").read_text() == "mine"


@pytest.mark.parametrize("name", ["kuching_01", "R-2", "a"])
def test_plain_names_are_valid_route_ids(name):
    assert valid_route_id(name)


@pytest.mark.parametrize("name", ["", "kuching 01", "a/b", "..", "路线", "a.b"])
def test_anything_else_is_not_a_route_id(name):
    assert not valid_route_id(name)
