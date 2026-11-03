"""The five perturbations of Sec. 5.3 applied to the ACDC reference segmentations (labels 0-3).

Boundary noise, erosion and dilation are those of the benchmark; the blob takes one of the three
other labels and is placed in a uniform 4x4 window; the majority filter is vectorized.
The random generator of slice i and perturbation k is default_rng(20261027 + 10 i + k).
"""
import numpy as np
import scipy.ndimage as ndi

import benchmark

LABELS = (0, 1, 2, 3)
KINDS = ["noise", "erode", "dilate", "blob", "smooth"]


def majority(G):
    pad = np.pad(G, 1, mode="edge")
    cnt = np.stack([ndi.uniform_filter((pad == l).astype(float), 3, mode="constant")[1:-1, 1:-1] for l in LABELS])
    cnt = np.rint(cnt * 9).astype(int)
    best = cnt == cnt.max(0, keepdims=True)
    keep = np.take_along_axis(best, G[None].astype(int), 0)[0]
    first = np.array(LABELS)[best.argmax(0)]
    return np.where(keep, G, first).astype(G.dtype)


def perturb(G, kind, rng):
    if kind == "smooth":
        return majority(G)
    if kind == "blob":
        h, w = G.shape; P = G.copy()
        win = np.lib.stride_tricks.sliding_window_view(G, (4, 4))[: h - 4, : w - 4]
        uni = (win == win[..., :1, :1]).all((-1, -2))
        ys, xs = np.nonzero(uni)
        if len(ys) == 0:
            return P
        k = rng.integers(len(ys)); y, x = ys[k] + 1, xs[k] + 1
        host = G[y, x]; others = [c for c in LABELS if c != host]
        P[y:y + 2, x:x + 2] = others[rng.integers(len(others))]
        return P
    return benchmark.perturb(G, kind, rng)


def rng_for(i, kind):
    return np.random.default_rng(20261027 + 10 * i + KINDS.index(kind))
