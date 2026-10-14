"""Lane NMS for CLRerNet without the compiled CUDA extension.

CLRerNet's head does ``from nms import nms`` and expects the LaneATT CUDA
kernel (libs/models/layers/nms/src/nms_kernel.cu), which needs nvcc and a C++
compiler to build. This is the same algorithm in NumPy, so the environment
installs from wheels alone and runs on Windows. Putting this directory's parent
on sys.path ahead of CLRerNet is the whole hookup; CLRerNet itself is not edited.

The arithmetic mirrors the kernel line for line, including where it rounds and
where float32 is promoted to double, because a port that is "about right"
still produces lane-shaped output. tests/test_clrernet_nms.py checks it against
a scalar transliteration of the kernel; it needs no torch, so it runs in CI.

A proposal row is PROP_SIZE = 77 values: two class logits, start_y, start_x,
length in strips, then 72 x coordinates in pixels.
"""

from __future__ import annotations

import numpy as np

N_OFFSETS = 72
N_STRIPS = N_OFFSETS - 1
PROP_SIZE = 5 + N_OFFSETS


def suppression_matrix(boxes: np.ndarray, overlap: float) -> np.ndarray:
    """Pairwise "these two proposals are the same lane", the kernel's devIoU."""
    b = np.ascontiguousarray(boxes, dtype=np.float32)
    if b.ndim != 2 or b.shape[1] != PROP_SIZE:
        raise ValueError(f"expected (N, {PROP_SIZE}) proposals, got {b.shape}")

    # (int)(a[2] * N_STRIPS + 0.5): float32 product, promoted to double by the
    # 0.5 literal, truncated toward zero.
    start = np.trunc((b[:, 2] * N_STRIPS).astype(np.float64) + 0.5).astype(np.int64)
    # (int)(start + a[4] - 1 + 0.5 - ((a[4] - 1) < 0)): float32 up to the 0.5.
    length = b[:, 4]
    end = np.trunc(
        (start.astype(np.float32) + length - 1).astype(np.float64) + 0.5 - ((length - 1) < 0)
    ).astype(np.int64)

    s = np.maximum(start[:, None], start[None, :])
    e = np.minimum(np.minimum(end[:, None], end[None, :]), N_OFFSETS - 1)

    # The kernel sums |a[i] - b[i]| for i in [5 + start, 5 + end].
    offset = np.arange(PROP_SIZE) - 5
    in_range = (offset >= s[..., None]) & (offset <= e[..., None])
    dist = (np.abs(b[:, None, :] - b[None, :, :]) * in_range).sum(axis=-1, dtype=np.float32)

    return (e >= s) & (dist < np.float32(overlap) * (e - s + 1).astype(np.float32))


def lane_nms(boxes: np.ndarray, scores: np.ndarray, overlap: float, top_k: int) -> list[int]:
    """Indices of the proposals kept, best first, at most ``top_k`` of them."""
    order = np.argsort(-np.asarray(scores), kind="stable")
    suppress = suppression_matrix(np.asarray(boxes)[order], overlap)
    removed = np.zeros(len(order), dtype=bool)
    keep: list[int] = []
    for i in range(len(order)):
        if removed[i]:
            continue
        keep.append(int(order[i]))
        if len(keep) == top_k:
            break
        removed[i + 1 :] |= suppress[i, i + 1 :]
    return keep


def nms(boxes, scores, overlap, top_k):
    """Drop-in for the extension's ``nms``: returns (keep, num_to_keep, parents).

    ``keep`` is padded to the proposal count, as the kernel's is; the caller
    slices it with ``num_to_keep``. CLRerNet never reads the third value.
    """
    import torch

    kept = lane_nms(
        boxes.detach().cpu().numpy(), scores.detach().cpu().numpy(), float(overlap), int(top_k)
    )
    keep = torch.zeros(boxes.shape[0], dtype=torch.long, device=boxes.device)
    keep[: len(kept)] = torch.as_tensor(kept, dtype=torch.long, device=boxes.device)
    num_to_keep = torch.tensor(len(kept), dtype=torch.long, device=boxes.device)
    return keep, num_to_keep, torch.zeros_like(keep)
