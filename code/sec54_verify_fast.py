"""Sec. 5.4: the compiled construction (hig_fast.construct_fast) returns exactly the same HIG as
the reference implementation (hig_construct.construct): same regions, typed geometric 1-nodes
with their region pairs, cut nodes with their carriers, and both incidence matrices.

usage: python3 sec54_verify_fast.py [RUN_DIR]
Checks the 1300 benchmark images and 400 random label images, and with the U-Net run directory
also 345 ACDC segmentations (references and epochs 2 and 16, every eighth slice): 2045 in all.
"""
import sys, os, numpy as np
from collections import Counter
from hig_construct import construct
from hig_fast import construct_fast
from rectgen import generate_rectangular_segmentation

def edge_coords_fast(e, h, w):
    NV = h * (w + 1)
    if e < NV:
        y, x = divmod(e, w + 1); return ((y, x), (y + 1, x))
    y, x = divmod(e - NV, w); return ((y, x), (y, x + 1))

def canon_ref(I):
    ids, edges, geom, cuts, carriers, holes, m01, m12 = construct(I)
    ek = lambda i: tuple(sorted(edges[i][:2]))
    gkeys = [frozenset(ek(i) for i in g["edges"]) for g in geom]
    ckeys = [frozenset(ek(i) for i in g) for g in cuts]
    keys1 = gkeys + ckeys
    junctions = sorted(v for v, inc in __import__('collections').Counter(
        p for e in edges for p in e[:2]).items() if inc >= 3)
    return ids, Counter((g["type"], g["pair"]) for g in geom), dict(zip(ckeys, carriers)), _mat(m01, m12, keys1, junctions, geom)

def _mat(m01, m12, keys1, junctions, gtypes):
    c12 = m12.tocoo(); M12 = Counter()
    for k, r, v in zip(c12.row, c12.col, c12.data): M12[(keys1[k], int(r))] += int(v)
    c01 = m01.tocoo(); M01 = Counter(); nj = len(junctions)
    for vv, k, v in zip(c01.row, c01.col, c01.data):
        vk = junctions[vv] if vv < nj else ("virt", keys1[k])
        M01[(vk, keys1[k])] += int(v)
    return M01, M12

def canon_fast(I):
    d = construct_fast(I); h, w = d["ids"].shape
    fe = d["frontier_edges"]
    cls = [[] for _ in range(len(d["gtype"]))]
    for e, c in zip(fe, d["edge_class"]): cls[c].append(edge_coords_fast(int(e), h, w))
    gkeys = [frozenset(c) for c in cls]
    comp_edges = {}
    for e, k in zip(fe, d["edge_comp"]): comp_edges.setdefault(int(k), []).append(edge_coords_fast(int(e), h, w))
    ckeys = [frozenset(comp_edges[int(k)]) for k in d["cut_comp"]]
    keys1 = gkeys + ckeys
    names = {0: "enc0", 1: "enc1", 2: "seg"}
    types = Counter((names[int(t)], (int(a), int(b))) for t, (a, b) in zip(d["gtype"], d["gpair"]))
    deg = Counter(p for e in fe for p in edge_coords_fast(int(e), h, w))
    junctions = sorted(v for v, n in deg.items() if n >= 3)
    return d["ids"], types, dict(zip(ckeys, [int(c) for c in d["carriers"]])), _mat(d["m01"], d["m12"], keys1, junctions, None)

def same(I):
    a = canon_ref(I); b = canon_fast(I)
    return np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2] and a[3] == b[3]

imgs = []
BASE_SEED, N = 20260927, 100
for r in range(1, 9):
    for j in range(N): imgs.append(generate_rectangular_segmentation(r, 0, BASE_SEED + r * 1000 + j)[0])
for n in range(5):
    for j in range(N): imgs.append(generate_rectangular_segmentation(5, n, BASE_SEED + n * 1000 + j)[0])
rng = np.random.default_rng(3)
for t in range(400):
    h, w = rng.integers(2, 30, size=2); imgs.append(rng.integers(0, int(rng.integers(2, 6)), size=(h, w)))
bad = sum(not same(np.asarray(I, np.int32)) for I in imgs)
print("benchmark + random:", len(imgs), "images, mismatches", bad)
if len(sys.argv) < 2:
    sys.exit(0)
run = sys.argv[1]
GT = np.load(run + '/test_gt.npy', mmap_mode='r'); P16 = np.load(run + '/test_pred_ep16.npy', mmap_mode='r'); P02 = np.load(run + '/test_pred_ep02.npy', mmap_mode='r')
idx = list(range(0, len(GT), 8))
bad = 0; n = 0
for i in idx:
    for A in (GT[i], P16[i], P02[i]):
        n += 1; bad += not same(np.asarray(A, np.int32))
print("ACDC:", n, "segmentations, mismatches", bad)
