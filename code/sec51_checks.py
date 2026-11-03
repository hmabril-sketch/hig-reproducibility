"""Sec. 5.1 implementation checks: in every segmentation used in Sec. 5, every region carries as
many cut nodes as it has (4,8) holes, computed independently from its digital Euler number with
Gray's bit quads, and the graded Euler identity |V2| - |V1| + |V0| = 2 holds.

usage: python3 sec51_checks.py [RUN_DIR ...]
Without arguments only the benchmark and its perturbations are checked (9063 segmentations).
With the ACDC run directories (the first one must contain test_gt.npy and test_img.npy), every
test_pred_epXX.npy in them, the ACDC references, their five perturbations and the 900 quantized
images of Sec. 5.4 are checked as well. With the U-Net run, the cleaned U-Net run, the DeepLabv3+
run and the cleaned DeepLabv3+ run this gives the 21 919 segmentations of the paper.
Reference output: results/sec51/checks.txt.
"""
import glob
import os
import sys

import numpy as np
from scipy import ndimage as ndi

import acdc_perturbations as AP
import benchmark as B
from euler_regions import holes_from_euler
from hig_fast import construct_fast

tot = dict(n=0, euler=0, holes=0, h=0)


def check(Sg):
    d = construct_fast(np.ascontiguousarray(Sg, dtype=np.int32))
    chi = (d["nreg"] + 1) - (len(d["gtype"]) + len(d["cut_comp"])) + d["m01"].shape[0]
    hh = np.bincount(d["carriers"], minlength=d["nreg"] + 1)[:d["nreg"]]
    he = holes_from_euler(d["ids"], d["nreg"])
    tot["n"] += 1; tot["euler"] += chi != 2; tot["holes"] += not np.array_equal(hh, he); tot["h"] += int(he.sum())


def report(name):
    print(f"{name}: segmentations {tot['n']}, Euler identity fails {tot['euler']}, "
          f"regions with cut nodes != holes {tot['holes']}, holes {tot['h']}", flush=True)


def quantize(I, sigma, L):
    J = ndi.gaussian_filter(I, sigma) if sigma else I
    return np.digitize(J, np.quantile(J, np.linspace(0, 1, L + 1)[1:-1])).astype(np.int32)


for idx, (cond, seed, r, n) in enumerate(B.benchmark()):
    A = B.image(seed, r, n); G = B.image(seed, r, n, B.UP)
    check(A); check(G)
    for kind in B.PERTURBATIONS:
        P = B.perturb(G, kind, B.perturbation_rng(seed, kind))
        if not np.array_equal(P, G):
            check(P)
report("benchmark (10x10 and enlarged) + perturbations")
runs = sys.argv[1:]
if runs:
    GT = np.load(os.path.join(runs[0], "test_gt.npy"))
    sets = [np.load(f, mmap_mode="r") for run in runs for f in sorted(glob.glob(os.path.join(run, "test_pred_ep*.npy")))]
    for i in range(len(GT)):
        if not (GT[i] > 0).any():
            continue
        G = GT[i].astype(np.int32); check(G)
        for A in sets:
            check(np.asarray(A[i]))
        for kind in AP.KINDS:
            check(AP.perturb(G, kind, AP.rng_for(i, kind)))
    report(f"+ ACDC references, {len(sets)} prediction sets and perturbed references")
    IMG = np.load(os.path.join(runs[0], "test_img.npy"), mmap_mode="r")
    for i in np.random.default_rng(20261027).choice(len(IMG), 60, replace=False):
        I = np.asarray(IMG[i], float)
        for sigma in (0, 1, 2):
            for L in (4, 8, 16, 32, 64):
                check(quantize(I, sigma, L))
    report("+ 900 quantized images")
