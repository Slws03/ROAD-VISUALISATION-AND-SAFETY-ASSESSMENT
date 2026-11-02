"""Request-path mapping for the viewer's static server.

The server has to reach two directories the viewer's own output directory
does not contain: the route's frames and its raw detection files. These
tests pin exactly which prefixes escape the output directory and where they
land, since a mistake here is a silent 404 or, worse, a path escaping the
route directory entirely.
"""

import json
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pytest

from irap_screening.config import load_config
from irap_screening.view.jobs import Rebuild, Upload
from irap_screening.view.review import load_review
from irap_screening.view.serve import (
    _make_server,
    _resolve_multi_path,
    _resolve_path,
    _review_write_target,
    _review_write_target_multi,
)

OUT_DIR = Path("/srv/route/view")
ROUTE_DIR = Path("/srv/route")
ROUTE_DIRS = {"route_a": Path("/data/route_a"), "route_b": Path("/data/route_b")}


def test_an_ordinary_path_resolves_under_out_dir():
    assert _resolve_path("/data.json", OUT_DIR, ROUTE_DIR) == OUT_DIR / "data.json"


def test_the_root_path_resolves_to_out_dir_itself():
    assert _resolve_path("/", OUT_DIR, ROUTE_DIR) == OUT_DIR


def test_frames_prefix_resolves_under_route_dir():
    assert (
        _resolve_path("/frames/f_000000.jpg", OUT_DIR, ROUTE_DIR)
        == ROUTE_DIR / "frames" / "f_000000.jpg"
    )


def test_detections_prefix_resolves_under_route_dir():
    assert (
        _resolve_path("/detections/lane.jsonl", OUT_DIR, ROUTE_DIR)
        == ROUTE_DIR / "detections" / "lane.jsonl"
    )


def test_query_and_fragment_are_stripped():
    assert (
        _resolve_path("/frames/f_000000.jpg?t=1#x", OUT_DIR, ROUTE_DIR)
        == ROUTE_DIR / "frames" / "f_000000.jpg"
    )


def test_parent_segments_cannot_escape_the_exposed_directory():
    """A crafted path must not walk out of frames/ or route_dir."""
    assert (
        _resolve_path("/frames/../../secret.txt", OUT_DIR, ROUTE_DIR)
        == ROUTE_DIR / "frames" / "secret.txt"
    )
    assert (
        _resolve_path("/../secret.txt", OUT_DIR, ROUTE_DIR) == OUT_DIR / "secret.txt"
    )


def test_a_segment_with_an_embedded_separator_cannot_escape_route_dir():
    """A single URL segment can smuggle a path separator (e.g. a literal
    backslash on Windows, where os.path.dirname treats it as one), which
    would otherwise let a crafted request walk out of frames/ even though
    "." and ".." alone are filtered. _resolve_path receives the path after
    percent-decoding (that happens inside this function), so the raw
    encoded request "/frames%5C..%5C..%5Cgps.csv" arrives here already
    decoded as a literal backslash sequence.
    """
    assert (
        _resolve_path("/frames/..\\..\\gps.csv", OUT_DIR, ROUTE_DIR)
        == ROUTE_DIR / "frames"
    )


def test_a_segment_with_an_embedded_separator_cannot_escape_out_dir():
    """Same trick, but aimed at escaping out_dir entirely."""
    assert _resolve_path("/..\\..\\secret.txt", OUT_DIR, ROUTE_DIR) == OUT_DIR


def test_multi_route_data_json_resolves_under_out_dir():
    """<route_id>/data.json is an ordinary file export_routes already wrote there."""
    assert (
        _resolve_multi_path("/route_a/data.json", OUT_DIR, ROUTE_DIRS)
        == OUT_DIR / "route_a" / "data.json"
    )


def test_multi_route_frames_prefix_resolves_under_that_routes_dir():
    assert (
        _resolve_multi_path("/route_a/frames/f_000000.jpg", OUT_DIR, ROUTE_DIRS)
        == ROUTE_DIRS["route_a"] / "frames" / "f_000000.jpg"
    )


def test_multi_route_detections_prefix_resolves_under_that_routes_dir():
    assert (
        _resolve_multi_path("/route_b/detections/lane.jsonl", OUT_DIR, ROUTE_DIRS)
        == ROUTE_DIRS["route_b"] / "detections" / "lane.jsonl"
    )


def test_multi_route_unknown_route_id_falls_back_to_out_dir():
    """A path that merely looks like /<id>/frames/... but names no known route
    is just an ordinary path under out_dir, not an error."""
    assert (
        _resolve_multi_path("/nope/frames/x.jpg", OUT_DIR, ROUTE_DIRS)
        == OUT_DIR / "nope" / "frames" / "x.jpg"
    )


def test_multi_route_root_resolves_to_out_dir_itself():
    assert _resolve_multi_path("/", OUT_DIR, ROUTE_DIRS) == OUT_DIR
    assert _resolve_multi_path("/routes.json", OUT_DIR, ROUTE_DIRS) == OUT_DIR / "routes.json"


def test_review_write_target_accepts_only_the_review_path():
    assert _review_write_target("/review", ROUTE_DIR) == ROUTE_DIR


def test_review_write_target_rejects_anything_else():
    assert _review_write_target("/data.json", ROUTE_DIR) is None
    assert _review_write_target("/frames/review", ROUTE_DIR) is None
    assert _review_write_target("/", ROUTE_DIR) is None


def test_multi_review_write_target_resolves_the_named_route():
    assert (
        _review_write_target_multi("/route_a/review", ROUTE_DIRS)
        == ROUTE_DIRS["route_a"]
    )


def test_multi_review_write_target_rejects_an_unknown_route():
    assert _review_write_target_multi("/nope/review", ROUTE_DIRS) is None


def test_multi_review_write_target_rejects_a_bare_route_id():
    assert _review_write_target_multi("/route_a", ROUTE_DIRS) is None


FAIL = [sys.executable, "-c", "print('boom'); raise SystemExit(3)"]
ROUTE_FILES = {
    "frames.csv": b"x",
    "gps.csv": b"x",
    "imu.csv": b"x",
    "frames/f_000000.jpg": b"x",
}


def _zip_bytes(tmp_path, files):
    path = tmp_path / "upload.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return path.read_bytes()


@pytest.fixture
def server(tmp_path):
    """A live server on a free port; yields (base_url, upload)."""
    routes = tmp_path / "routes"
    routes.mkdir()
    upload = Upload(
        routes_root=routes,
        out_dir=routes / "view",
        cfg=load_config(),
        lane_command=FAIL,
        sign_command=FAIL,
    )
    httpd = _make_server(
        tmp_path,
        lambda path, out: _resolve_multi_path(path, out, upload.route_dirs),
        lambda path: None,
        0,
        upload,
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", upload
    httpd.shutdown()
    httpd.server_close()


def _request(url, body=None):
    request = urllib.request.Request(url, data=body, method="POST" if body is not None else "GET")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read()), response.reason
    except urllib.error.HTTPError as error:
        return error.code, None, error.reason


def _settled(base):
    for _ in range(200):
        status = _request(f"{base}/job")[1]
        if status["state"] != "running":
            return status
        time.sleep(0.05)
    raise AssertionError("job never settled")


def test_job_status_is_not_served_without_upload(tmp_path):
    httpd = _make_server(tmp_path, lambda path, out: out / "nothing", lambda path: None, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        assert _request(f"http://127.0.0.1:{httpd.server_address[1]}/job")[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_job_status_starts_idle(server):
    base, _ = server
    code, status, _ = _request(f"{base}/job")
    assert (code, status["state"]) == (200, "idle")


def test_an_upload_starts_a_job_and_extracts_the_route(server, tmp_path):
    base, upload = server

    code, status, _ = _request(f"{base}/upload?name=r1", _zip_bytes(tmp_path, ROUTE_FILES))

    assert (code, status["route_id"]) == (202, "r1")
    final = _settled(base)
    # The stub lane detector fails, which is as far as a test without models goes.
    assert (final["state"], final["label"], final["error"]) == ("failed", "Lane detection", "boom")
    assert (upload.routes_root / "r1" / "frames" / "f_000000.jpg").exists()


def test_a_body_larger_than_one_chunk_arrives_intact(server, tmp_path):
    base, upload = server
    big = bytes(range(256)) * (3 * 4096)  # 3 MiB, stored uncompressed
    body = _zip_bytes(tmp_path, {**ROUTE_FILES, "frames/big.bin": big})

    assert _request(f"{base}/upload?name=big", body)[0] == 202
    _settled(base)

    assert (upload.routes_root / "big" / "frames" / "big.bin").read_bytes() == big


@pytest.mark.parametrize("name", ["kuching%2001", "a%2Fb", "", "%E8%B7%AF"])
def test_a_name_that_is_not_a_route_id_is_refused_before_anything_is_written(
    server, tmp_path, name
):
    base, upload = server

    code, _, reason = _request(f"{base}/upload?name={name}", _zip_bytes(tmp_path, ROUTE_FILES))

    assert code == 400
    assert "letters, digits" in reason
    assert list(upload.routes_root.iterdir()) == []


def test_a_name_already_taken_is_refused_and_says_what_to_delete(server, tmp_path):
    base, upload = server
    (upload.routes_root / "r1").mkdir()

    code, _, reason = _request(f"{base}/upload?name=r1", _zip_bytes(tmp_path, ROUTE_FILES))

    assert code == 409
    assert "already exists" in reason and "delete" in reason


def test_a_route_left_by_a_failed_job_blocks_a_retry_under_the_same_name(server, tmp_path):
    base, _ = server
    body = _zip_bytes(tmp_path, ROUTE_FILES)
    _request(f"{base}/upload?name=r1", body)
    _settled(base)

    assert _request(f"{base}/upload?name=r1", body)[0] == 409


@pytest.mark.parametrize("length", ["abc", "-1", "0"])
def test_a_content_length_that_is_not_a_positive_number_is_refused(server, length):
    import http.client

    base, upload = server
    connection = http.client.HTTPConnection(base.removeprefix("http://"), timeout=20)
    connection.putrequest("POST", "/upload?name=r1")
    connection.putheader("Content-Length", length)
    connection.endheaders()

    assert connection.getresponse().status == 400
    assert list(upload.routes_root.iterdir()) == []


def test_a_connection_reset_mid_upload_leaves_no_temporary_zip(server, tmp_path, monkeypatch):
    import socket
    import struct
    import tempfile

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    base, _ = server
    host, port = base.removeprefix("http://").split(":")

    client = socket.create_connection((host, int(port)))
    client.sendall(b"POST /upload?name=r1 HTTP/1.1\r\nContent-Length: 1000000\r\n\r\npartial")
    time.sleep(0.3)  # let the handler create its temporary file and block reading
    # Linger 0 makes close() send a reset rather than a clean end of stream.
    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("hh", 1, 0))
    client.close()

    for _ in range(100):
        if not list(scratch.iterdir()):
            break
        time.sleep(0.05)
    assert list(scratch.iterdir()) == []


@pytest.fixture
def config_server(tmp_path):
    """A live server with a config panel and no routes; yields (base_url, rebuild)."""
    upload = Upload(
        routes_root=tmp_path, out_dir=tmp_path, cfg=load_config(),
        lane_command=FAIL, sign_command=FAIL,
    )
    startup = ["grade.threshold_pct=8"]  # as `irap view --set` would pass it
    rebuild = Rebuild(tmp_path, load_config(overrides=startup), overrides=startup, upload=upload)
    httpd = _make_server(
        tmp_path, lambda path, out: out / "x", lambda path: None, 0, upload, rebuild
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", rebuild
    httpd.shutdown()
    httpd.server_close()


def _post_json(url, payload):
    return _request(url, json.dumps(payload).encode("utf-8"))


def test_config_lists_every_threshold_with_its_source(config_server):
    base, _ = config_server
    code, state, _ = _request(f"{base}/config")
    assert code == 200
    assert state["values"]["curvature.threshold_m"] == 400
    assert state["sources"]["curvature.threshold_m"] == "default"
    assert state["sources"]["grade.threshold_pct"] == "cli"


@pytest.mark.parametrize("changes", [
    {"curvature.no_such_key": "1"},
    {"sign.imgsz": "640"},              # baked into the cached detections
    {"grade.threshold_pct": "-1"},      # fails the configuration's own range check
    {"sign.conf": "0.05"},              # below sign.runner_conf
])
def test_a_change_the_report_cannot_take_is_refused_and_nothing_changes(config_server, changes):
    base, rebuild = config_server
    code, _, _ = _post_json(f"{base}/config", changes)
    assert code == 400
    assert rebuild.panel == {} and rebuild.runner.status()["state"] == "idle"


def test_a_valid_change_starts_a_rebuild_and_reaches_uploads_too(config_server):
    base, rebuild = config_server
    code, status, _ = _post_json(f"{base}/config", {"curvature.threshold_m": "350"})
    assert (code, status["kind"]) == (202, "config")
    assert _settled(base)["state"] == "done"

    state = _request(f"{base}/config")[1]
    assert state["values"]["curvature.threshold_m"] == 350
    assert state["sources"]["curvature.threshold_m"] == "viewer"
    assert state["values"]["grade.threshold_pct"] == 8  # the startup --set still holds
    assert rebuild.upload.cfg.curvature.threshold_m == 350


def test_a_recommendation_edit_is_saved_through_post_review(tmp_path):
    httpd = _make_server(
        tmp_path, lambda path, out: out, lambda path: _review_write_target(path, tmp_path), 0
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        body = {"segment_id": "R01-0002", "module": "recommendation", "text": "Repaint."}
        code, entry, _ = _request(f"{base}/review", json.dumps(body).encode())
        assert (code, entry["text"]) == (200, "Repaint.")
        assert load_review(tmp_path)["R01-0002"]["recommendation"]["text"] == "Repaint."
        # The page reads it back, so it survives a reload before data.json is rebuilt.
        assert _request(f"{base}/review")[1]["R01-0002"]["recommendation"]["text"] == "Repaint."

        # The text field is required for this kind of entry.
        del body["text"]
        assert _request(f"{base}/review", json.dumps(body).encode())[0] == 400
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_category_review_is_saved_with_its_flag_and_bad_codes_are_refused(tmp_path):
    httpd = _make_server(
        tmp_path, lambda path, out: out, lambda path: _review_write_target(path, tmp_path), 0
    )
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        body = {"segment_id": "R01-0002", "module": "quality_of_curve", "category": 2}
        code, entry, _ = _request(f"{base}/review", json.dumps(body).encode())
        assert (code, entry["flag"]) == (200, "critical")

        # Grade has no category 2; an unknown module has no categories at all.
        body = {"segment_id": "R01-0002", "module": "grade", "category": 2}
        assert _request(f"{base}/review", json.dumps(body).encode())[0] == 400
        body = {"segment_id": "R01-0002", "module": "speed", "category": None}
        assert _request(f"{base}/review", json.dumps(body).encode())[0] == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
