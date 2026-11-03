"""Table 2, ACDC rows (Sec. 5.3): the five perturbations applied to the 854 ACDC reference
segmentations that contain a cardiac structure, with the same comparisons as the benchmark.

usage: python3 sec53_acdc_reference_perturbations.py RUN_DIR OUT_CSV [--no-bm] [--max-slices N]
RUN_DIR is any directory with test_gt.npy (written by acdc_unet.py or acdc_predict.py).
Reference output: results/sec53/acdc_reference_perturbations.csv. About 10 min on two cores.
"""
import argparse
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

import acdc_perturbations as AP
import benchmark as B
import topoqc
from rag8 import rag8_out

BM, GT = True, None


def init(run, bm):
    global BM, GT
    BM, GT = bm, np.load(os.path.join(run, "test_gt.npy"), mmap_mode="r")


def one(i):
    G = GT[i].astype(np.int32)
    hG = topoqc.HIG(G)
    gG, rG, bG, r8G = hG.graph(), hG.rag_out(), topoqc.betti_per_label(G), rag8_out(G)
    rows = []
    for kind in AP.KINDS:
        P = AP.perturb(G, kind, AP.rng_for(i, kind))
        if np.array_equal(P, G):
            rows.append(dict(slice=i, perturbation=kind, identical=True))
            continue
        rec = B.evaluate_pair(G, hG, gG, rG, bG, P)
        rec["rag8_changed"] = not topoqc.same_graph(r8G, rag8_out(P))
        if BM:
            import bettimatch
            e = bettimatch.errors(P, G)
            rec.update(bm0=e["bm0"], bm1=e["bm1"], bm=e["bm"])
        rec.update(slice=i, perturbation=kind, identical=False)
        rows.append(rec)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("out")
    ap.add_argument("--no-bm", action="store_true"); ap.add_argument("--max-slices", type=int)
    a = ap.parse_args()
    gt = np.load(os.path.join(a.run, "test_gt.npy"))
    idx = [i for i in range(len(gt)) if (gt[i] > 0).any()][: a.max_slices]
    t0 = time.time()
    with Pool(int(os.environ.get("HIG_WORKERS", os.cpu_count())), initializer=init, initargs=(a.run, not a.no_bm)) as pool:
        res = pool.map(one, idx, chunksize=4)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.to_csv(a.out, index=False)
    print(f"{len(idx)} slices, {len(df)} rows, {time.time() - t0:.0f} s")
