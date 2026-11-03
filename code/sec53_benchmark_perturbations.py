"""Table 2, benchmark rows (Sec. 5.3): the 1298 benchmark segmentations, enlarged four times,
under five perturbations; per pair, Dice, per-class Betti numbers, both augmented RAGs, the typed
HIG, the localized differences and the per-class Betti matching error.

usage: python3 sec53_benchmark_perturbations.py OUT_CSV [--no-bm] [--max-images N]
Reference output: results/sec53/benchmark_perturbations.csv (6467 pairs; 23 perturbations that
leave the image unchanged are discarded). About 3 min on two cores.
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import benchmark as B
import topoqc
from rag8 import rag8_out

BM = True


def init(bm):
    global BM
    BM = bm


def one(item):
    idx, (cond, seed, r, n) = item
    G = B.image(seed, r, n, B.UP)
    hG = topoqc.HIG(G)
    gG, rG, bG, r8G = hG.graph(), hG.rag_out(), topoqc.betti_per_label(G), rag8_out(G)
    rows, same = [], 0
    for kind in B.PERTURBATIONS:
        P = B.perturb(G, kind, B.perturbation_rng(seed, kind))
        if np.array_equal(P, G):
            same += 1
            continue
        rec = B.evaluate_pair(G, hG, gG, rG, bG, P)
        rec.update(image=idx, condition=cond, seed=seed, perturbation=kind, identical=False)
        rec["rag8_changed"] = not topoqc.same_graph(r8G, rag8_out(P))
        if BM:
            import bettimatch
            e = bettimatch.errors(P, G)
            rec.update(bm0=e["bm0"], bm1=e["bm1"], bm=e["bm"])
        rows.append(rec)
    return rows, same


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--no-bm", action="store_true"); ap.add_argument("--max-images", type=int)
    a = ap.parse_args()
    items = list(enumerate(B.benchmark()))[: a.max_images]
    t0 = time.time()
    with Pool(int(os.environ.get("HIG_WORKERS", os.cpu_count())), initializer=init, initargs=(not a.no_bm,)) as pool:
        res = pool.map(one, items, chunksize=8)
    df = pd.DataFrame([r for rows, _ in res for r in rows])
    df.to_csv(a.out, index=False)
    print(f"{len(items)} images, {len(df)} pairs, {sum(s for _, s in res)} unchanged perturbations discarded, "
          f"{time.time() - t0:.0f} s")
