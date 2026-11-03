"""Table 3 and the ACDC text of Sec. 5.3: every stored prediction of one network compared with
the reference, slice by slice (Dice, per-class Betti numbers, both augmented RAGs, typed HIG,
localized differences, anatomical constraints, per-class Betti matching error).

usage: python3 sec53_acdc_predictions.py RUN_DIR TAG NET CLEAN OUT_CSV [--no-bm]
  RUN_DIR  directory with test_gt.npy, test_meta.tsv and test_pred_epXX.npy
  TAG, NET, CLEAN  labels written in the output (e.g. unet "U-Net" 0)
Reference outputs: results/sec53/acdc_predictions_{unet,unet_clean,deeplab,deeplab_clean}.csv.
About 2.5 min per prediction file on two cores.
"""
import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.ndimage as ndi

import topoqc
from rag8 import rag8_out

RV, MYO, LV = 1, 2, 3
RUN, BM = None, True


def anatomy(h):
    """Constraints read from the typed nodes of one HIG: exactly one LV and one MYO region,
    every geometric 1-node of the LV also incident with the MYO, and a MYO cut node whose hole
    contains the LV."""
    lab = h.lab
    nLV = int(np.sum(lab[:h.out] == LV))
    nMYO = int(np.sum(lab[:h.out] == MYO))
    lv_bad = rv_lv = 0
    for g in h.geom:
        a, b = g["pair"]
        la, lb = int(lab[a]), int(lab[b])
        if LV in (la, lb):
            other = lb if la == LV else la
            lv_bad += other != MYO
            rv_lv += other == RV
    myo_holes_with_lv = 0
    for k, group in enumerate(h.cuts):
        car = h.carriers[k]
        if lab[car] != MYO:
            continue
        inner = set()
        for i in group:
            inner.update(h.edges[i][2])
        inner.discard(car)
        myo_holes_with_lv += any(lab[r] == LV for r in inner)
    return {"nLV": nLV, "nMYO": nMYO, "lv_contacts_not_myo": lv_bad, "rv_lv_contacts": rv_lv,
            "myo_holes_with_lv": myo_holes_with_lv,
            "plausible": int(nLV == 1 and nMYO == 1 and lv_bad == 0 and myo_holes_with_lv == 1)}


def init(run, bm):
    global RUN, BM, GT, META, PREDS
    RUN, BM = run, bm
    GT = np.load(os.path.join(run, "test_gt.npy"), mmap_mode="r")
    META = [l.rstrip("\n").split("\t") for l in open(os.path.join(run, "test_meta.tsv"))]
    PREDS = {os.path.basename(p): np.load(p, mmap_mode="r") for p in sorted(glob.glob(os.path.join(run, "test_pred_ep*.npy")))}


def one(i):
    G = np.asarray(GT[i]).astype(np.int32)
    if not (G > 0).any():
        return []
    hG = topoqc.HIG(G)
    gG, rG, bG, r8G = hG.graph(), hG.rag_out(), topoqc.betti_per_label(G), rag8_out(G)
    aG = anatomy(hG)
    rows = []
    for name, PR in PREDS.items():
        P = np.asarray(PR[i]).astype(np.int32)
        hP = topoqc.HIG(P)
        changed = G != P
        rec = {"slice": i, "case": META[i][0], "z": int(META[i][1]), "nz": int(META[i][2]),
               "epoch": int(name[12:14]), "dice": topoqc.dice_macro(G, P),
               "changed_px": float(changed.mean()),
               "betti_changed": topoqc.betti_per_label(P) != bG,
               "rag_changed": not topoqc.same_graph(rG, hP.rag_out()),
               "hig_changed": not topoqc.same_graph(gG, hP.graph(), edge_attr=True)}
        diffs = topoqc.localized_differences(hG, hP)
        near = ndi.binary_dilation(changed, structure=topoqc.S8)
        ok, kinds = 0, []
        for side, key, sup in diffs:
            h = hP if side == "P" else hG
            ok += any(near[p] for p in topoqc.support_pixels(h, sup))
            kinds.append(key[0])
        rec["n_diffs"], rec["n_diffs_near_change"] = len(diffs), ok
        for k in ("region", "contact", "hole", "junction"):
            rec[f"diff_{k}"] = kinds.count(k)
        aP = anatomy(hP)
        rec.update({f"gt_{k}": v for k, v in aG.items()})
        rec.update({f"pred_{k}": v for k, v in aP.items()})
        rec["rag8_changed"] = not topoqc.same_graph(r8G, rag8_out(P))
        rec["file"] = name
        if BM:
            import bettimatch
            rec.update(bettimatch.errors(P, G))
        rows.append(rec)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for k in ("run", "tag", "net", "clean", "out"):
        ap.add_argument(k)
    ap.add_argument("--no-bm", action="store_true")
    a = ap.parse_args()
    n = len(np.load(os.path.join(a.run, "test_gt.npy"), mmap_mode="r"))
    with Pool(int(os.environ.get("HIG_WORKERS", os.cpu_count())), initializer=init, initargs=(a.run, not a.no_bm)) as pool:
        res = pool.map(one, range(n), chunksize=8)
    df = pd.DataFrame([r for rows in res for r in rows])
    df.insert(df.columns.get_loc("file") + 1, "tag", a.tag)
    if not a.no_bm:
        for l in range(4):
            for d in (0, 1):
                if f"bm{d}_L{l}" not in df:
                    df[f"bm{d}_L{l}"] = 0
        per = [f"bm{d}_L{l}" for l in range(4) for d in (0, 1)]
        df[per] = df[per].fillna(0).astype(int)
        df = df[[c for c in df.columns if c not in per + ["bm0", "bm1", "bm"]] + per + ["bm0", "bm1", "bm"]]
        df["bm_changed"] = df.bm > 0
    df["net"], df["clean"] = a.net, bool(int(a.clean))
    df.sort_values(["epoch", "slice"]).to_csv(a.out, index=False)
    print(len(df), "rows")
