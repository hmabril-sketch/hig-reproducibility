"""Controlled synthetic benchmark of Sec. 5.1 and the five perturbations of Sec. 5.3.

benchmark()      the 1298 distinct images of the benchmark (r = 1..8 with n = 0, and r = 5 with
                 n = 0..4, 100 seeds each, duplicates removed), as (condition, seed, r, n);
image(seed, r, n, up)  the 10x10 segmentation, enlarged `up` times by pixel replication;
perturb(G, kind, rng)  boundary noise, erosion, dilation, 2x2 blob, 3x3 majority filter;
evaluate_pair(...)     Dice, per-class Betti numbers, RAG with OUT, typed HIG and localized
                 differences of a reference G and a perturbed segmentation P.
"""
import numpy as np
import scipy.ndimage as ndi

import topoqc
from rectgen import generate_rectangular_segmentation

BASE_SEED, N, UP = 20260927, 100, 4
PERTURBATIONS = ["noise", "erode", "dilate", "blob", "smooth"]


def benchmark():
    imgs, seen = [], set()
    for r in range(1, 9):
        for j in range(N):
            imgs.append((f"r={r},n=0", BASE_SEED + r * 1000 + j, r, 0))
    for n in range(5):
        for j in range(N):
            imgs.append((f"r=5,n={n}", BASE_SEED + n * 1000 + j, 5, n))
    out = []
    for cond, seed, r, n in imgs:
        A = generate_rectangular_segmentation(r, n, seed)[0].astype(np.int32)
        k = A.tobytes()
        if k in seen:
            continue
        seen.add(k)
        out.append((cond, seed, r, n))
    return out


def boundary_mask(G):
    b = np.zeros(G.shape, bool)
    b[:-1, :] |= G[:-1, :] != G[1:, :]
    b[1:, :] |= G[:-1, :] != G[1:, :]
    b[:, :-1] |= G[:, :-1] != G[:, 1:]
    b[:, 1:] |= G[:, :-1] != G[:, 1:]
    return b


def perturb(G, kind, rng):
    P = G.copy()
    h, w = G.shape
    if kind == "noise":
        ys, xs = np.nonzero(boundary_mask(G))
        for y, x in zip(ys, xs):
            if rng.random() < 0.15:
                nb = [G[yy, xx] for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                      if 0 <= yy < h and 0 <= xx < w and G[yy, xx] != G[y, x]]
                if nb:
                    P[y, x] = nb[rng.integers(len(nb))]
    elif kind in ("erode", "dilate"):
        labels = np.unique(G)
        if len(labels) < 2:
            return P
        l = labels[rng.integers(len(labels))]
        m = G == l
        if kind == "erode":
            keep = ndi.binary_erosion(m, structure=topoqc.S4, border_value=1)
            removed = m & ~keep
            _, (iy, ix) = ndi.distance_transform_edt(m, return_indices=True)
            P[removed] = G[iy[removed], ix[removed]]
        else:
            grow = ndi.binary_dilation(m, structure=topoqc.S4) & ~m
            P[grow] = l
    elif kind == "blob":
        cand = []
        for y in range(1, h - 3):
            for x in range(1, w - 3):
                win = G[y - 1:y + 3, x - 1:x + 3]
                if (win == win[0, 0]).all():
                    cand.append((y, x))
        if not cand:
            return P
        y, x = cand[rng.integers(len(cand))]
        host = G[y, x]
        others = [c for c in range(5) if c != host]
        P[y:y + 2, x:x + 2] = others[rng.integers(len(others))]
    elif kind == "smooth":
        pad = np.pad(G, 1, mode="edge")
        for y in range(h):
            for x in range(w):
                win = pad[y:y + 3, x:x + 3].ravel()
                vals, cnt = np.unique(win, return_counts=True)
                best = vals[cnt == cnt.max()]
                P[y, x] = G[y, x] if G[y, x] in best else best[0]
    return P


def evaluate_pair(G, hG, gG, rG, bG, P):
    hP = topoqc.HIG(P)
    changed = G != P
    rec = {
        "changed_px": float(changed.mean()),
        "dice": topoqc.dice_macro(G, P),
        "betti_changed": topoqc.betti_per_label(P) != bG,
        "rag_changed": not topoqc.same_graph(rG, hP.rag_out()),
        "hig_changed": not topoqc.same_graph(gG, hP.graph(), edge_attr=True),
    }
    diffs = topoqc.localized_differences(hG, hP)
    near = ndi.binary_dilation(changed, structure=topoqc.S8, iterations=1)
    ok = 0
    kinds = []
    for side, key, sup in diffs:
        h = hP if side == "P" else hG
        pts = topoqc.support_pixels(h, sup)
        if any(near[p] for p in pts):
            ok += 1
        kinds.append(key[0])
    rec["n_diffs"] = len(diffs)
    rec["n_diffs_near_change"] = ok
    for k in ("region", "contact", "hole", "junction"):
        rec[f"diff_{k}"] = kinds.count(k)
    return rec


def image(seed, r, n, up=1):
    """The benchmark segmentation (10x10), enlarged `up` times by pixel replication."""
    A = generate_rectangular_segmentation(r, n, seed)[0].astype(np.int32)
    return A if up == 1 else np.kron(A, np.ones((up, up), dtype=np.int32))


def perturbation_rng(seed, kind):
    return np.random.default_rng(seed * 10 + PERTURBATIONS.index(kind))
