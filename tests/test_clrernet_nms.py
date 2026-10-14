"""The NumPy lane NMS must agree with CLRerNet's CUDA kernel.

The kernel cannot be built here (it needs nvcc), so the reference is a scalar
transliteration of nms_kernel.cu, written one C statement at a time. The
vectorised port in scripts/clrernet/nms is checked against it on random
proposals and on the rounding edges where a careless port goes wrong.
"""

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "clrernet" / "nms" / "__init__.py"
_spec = importlib.util.spec_from_file_location("clrernet_nms", _PATH)
nms_port = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(nms_port)

N_OFFSETS, N_STRIPS, PROP_SIZE = 72, 71, 77
f32 = np.float32


def _dev_iou(a, b, threshold):
    start_a = int(np.float64(a[2] * f32(N_STRIPS)) + 0.5)
    start_b = int(np.float64(b[2] * f32(N_STRIPS)) + 0.5)
    start = max(start_a, start_b)
    end_a = int(np.float64(f32(start_a) + a[4] - f32(1)) + 0.5 - ((a[4] - 1) < 0))
    end_b = int(np.float64(f32(start_b) + b[4] - f32(1)) + 0.5 - ((b[4] - 1) < 0))
    end = min(min(end_a, end_b), N_OFFSETS - 1)
    if end < start:
        return False
    dist = f32(0)
    for i in range(5 + start, 5 + end + 1):
        dist = f32(dist + abs(a[i] - b[i]))
    return bool(dist < f32(threshold) * f32(end - start + 1))


def _reference_nms(boxes, scores, overlap, top_k):
    order = np.argsort(-scores, kind="stable")
    removed = [False] * len(order)
    keep = []
    for i in range(len(order)):
        if removed[i]:
            continue
        keep.append(int(order[i]))
        if len(keep) == top_k:
            break
        for j in range(i + 1, len(order)):
            if _dev_iou(boxes[order[i]], boxes[order[j]], overlap):
                removed[j] = True
    return keep


def _proposals(rng, n, spread):
    """Clusters of near-duplicate lanes, so suppression actually happens."""
    boxes = np.zeros((n, PROP_SIZE), dtype=f32)
    centres = rng.uniform(100, 700, size=4)
    boxes[:, 2] = rng.uniform(0.0, 0.9, size=n)
    boxes[:, 4] = rng.uniform(0.0, 80.0, size=n)
    slope = rng.uniform(-3, 3, size=(n, 1))
    boxes[:, 5:] = (
        rng.choice(centres, size=(n, 1))
        + slope * np.arange(N_OFFSETS)
        + rng.normal(0, spread, size=(n, N_OFFSETS))
    )
    return boxes, rng.uniform(0.1, 1.0, size=n).astype(f32)


def test_matches_the_kernel_on_random_proposals():
    rng = np.random.default_rng(0)
    suppressed_something = False
    for trial in range(40):
        n = int(rng.integers(1, 60))
        boxes, scores = _proposals(rng, n, spread=rng.choice([1.0, 20.0, 60.0]))
        top_k = int(rng.choice([1, 4, 1000]))
        expected = _reference_nms(boxes, scores, 50.0, top_k)
        assert nms_port.lane_nms(boxes, scores, 50.0, top_k) == expected, f"trial {trial}"
        suppressed_something |= len(expected) < min(n, top_k)
    assert suppressed_something, "the fixture never exercised suppression"


def test_pairwise_decision_on_rounding_edges():
    rng = np.random.default_rng(1)
    starts = [0.0, 0.007, 0.0071, 0.5, 0.993, 1.0, 1.02, -0.01]
    lengths = [0.0, 0.3, 0.49, 0.5, 0.51, 1.0, 1.5, 71.0, 72.4, 200.0]
    boxes = []
    for start in starts:
        for length in lengths:
            row = np.zeros(PROP_SIZE, dtype=f32)
            row[2], row[4] = start, length
            row[5:] = 400 + rng.normal(0, 30, size=N_OFFSETS)
            boxes.append(row)
    boxes = np.stack(boxes)
    got = nms_port.suppression_matrix(boxes, 50.0)
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            assert got[i, j] == _dev_iou(boxes[i], boxes[j], 50.0), (i, j)
    assert got.any() and not got.all()


def test_disjoint_vertical_ranges_never_suppress():
    low = np.zeros(PROP_SIZE, dtype=f32)
    high = np.zeros(PROP_SIZE, dtype=f32)
    low[2], low[4] = 0.0, 10.0
    high[2], high[4] = 0.8, 10.0
    boxes = np.stack([low, high])  # identical x, but they never share a row
    assert nms_port.lane_nms(boxes, np.array([0.9, 0.8], dtype=f32), 50.0, 4) == [0, 1]


def test_top_k_caps_the_result_and_empty_input_is_empty():
    rng = np.random.default_rng(2)
    boxes, scores = _proposals(rng, 30, spread=80.0)
    assert len(nms_port.lane_nms(boxes, scores, 50.0, 2)) == 2
    empty = np.zeros((0, PROP_SIZE), dtype=f32)
    assert nms_port.lane_nms(empty, np.zeros(0, dtype=f32), 50.0, 4) == []
