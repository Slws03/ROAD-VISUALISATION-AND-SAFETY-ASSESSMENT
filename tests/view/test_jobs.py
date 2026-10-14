"""Upload handling and job sequencing for the viewer's upload control.

No models and no GPU: the two detector commands are plain argument lists, so
the tests substitute a Python one-liner for each.
"""

import sys
import threading
import zipfile
from pathlib import Path

import pytest

from irap_screening.config import load_config
from irap_screening.view.jobs import (
    Conflict,
    JobRunner,
    Upload,
    UploadError,
    extract_route,
    pipeline_steps,
    run_command,
    valid_route_id,
)

FAIL = [sys.executable, "-c", "print('boom'); raise SystemExit(3)"]

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


def _finished(runner: JobRunner) -> dict:
    runner._thread.join(timeout=20)
    return runner.status()


def _upload(tmp_path: Path, lane=FAIL, sign=FAIL) -> Upload:
    routes = tmp_path / "routes"
    routes.mkdir()
    return Upload(
        routes_root=routes,
        out_dir=routes / "view",
        cfg=load_config(),
        lane_command=lane,
        sign_command=sign,
    )


def test_run_command_feeds_output_lines_to_the_log():
    lines = []

    run_command([sys.executable, "-c", "print('one'); print(); print('two')"], lines.append)

    assert lines == ["one", "two"]


def test_run_command_raises_with_the_last_line_the_command_printed():
    with pytest.raises(RuntimeError, match="boom"):
        run_command(FAIL, lambda line: None)


def test_run_command_survives_output_that_is_not_utf8():
    lines = []
    script = "import sys; sys.stdout.buffer.write(b'caf\\xe9\\n')"

    run_command([sys.executable, "-c", script], lines.append)

    assert lines == ["caf\ufffd"]


def test_an_idle_runner_says_so():
    assert JobRunner().status()["state"] == "idle"


def test_steps_run_in_order_and_the_job_ends_done():
    ran = []
    runner = JobRunner()
    steps = [
        ("First", lambda log: (ran.append("first"), log("from first"))),
        ("Second", lambda log: (ran.append("second"), log("from second"))),
    ]

    runner.start("r1", steps)
    status = _finished(runner)

    assert ran == ["first", "second"]
    assert status["state"] == "done"
    assert (status["route_id"], status["step"], status["steps"]) == ("r1", 2, 2)
    # The log is the current step's own: a line left over from the step before
    # would be shown beside the wrong label.
    assert status["log"] == ["from second"]


def test_a_failing_step_stops_the_job_and_reports_why():
    ran = []

    def explode(log):
        raise RuntimeError("no lanes on any frame")

    runner = JobRunner()
    runner.start("r1", [("Lane detection", explode), ("Next", lambda log: ran.append(1))])
    status = _finished(runner)

    assert status["state"] == "failed"
    assert status["label"] == "Lane detection"
    assert status["error"] == "no lanes on any frame"
    assert ran == []


def test_a_second_job_is_refused_while_one_runs_and_accepted_after():
    release = threading.Event()
    runner = JobRunner()
    runner.start("r1", [("Wait", lambda log: release.wait(20))])

    with pytest.raises(Conflict, match="already running"):
        runner.start("r2", [])

    release.set()
    _finished(runner)
    runner.start("r2", [])
    assert _finished(runner)["route_id"] == "r2"


def test_a_failed_detector_leaves_the_route_on_disk_but_unpublished(tmp_path):
    upload = _upload(tmp_path)
    archive = _zip(tmp_path / "r1.zip", ROUTE_FILES)

    upload.runner.start("r1", pipeline_steps(upload, "r1", archive))
    status = _finished(upload.runner)

    assert (status["state"], status["label"], status["error"]) == (
        "failed",
        "Lane detection",
        "boom",
    )
    assert (upload.routes_root / "r1" / "frames.csv").exists()
    assert upload.route_dirs == {}
    assert not archive.exists()  # the temporary upload is always removed


def test_a_bad_zip_fails_the_job_at_the_first_step(tmp_path):
    upload = _upload(tmp_path)
    archive = _zip(tmp_path / "r1.zip", {"readme.txt": b"x"})

    upload.runner.start("r1", pipeline_steps(upload, "r1", archive))
    status = _finished(upload.runner)

    assert (status["state"], status["step"], status["label"]) == ("failed", 1, "Extracting")
    assert "not a route" in status["error"]
    assert not (upload.routes_root / "r1").exists()
