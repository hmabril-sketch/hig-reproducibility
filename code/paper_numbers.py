"""Recomputes the numbers of Sec. 5 (Tables 1-4 and the text) from the CSV files in results/ and
compares each one with the value printed in the paper, formatted with the same precision.

usage: python3 paper_numbers.py [RESULTS_DIR]      (default: ../results)
Exit status 0 when every number agrees. Run it on the committed results, or on a results
directory regenerated with the scripts listed in the README.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
N_OK = N_BAD = 0


def chk(what, value, expected, fmt="{}"):
    """Compare value, formatted like the paper, with the printed string `expected`."""
    global N_OK, N_BAD
    got = fmt.format(value) if not isinstance(value, str) else value
    ok = got == expected
    N_OK += ok; N_BAD += not ok
    print(f"  {'ok ' if ok else 'BAD'} {what}: {got}" + ("" if ok else f"   (paper: {expected})"))


def pct(x):
    return 100.0 * float(np.mean(x))


def section(title):
    print(f"\n{title}")


def boolcols(x):
    for c in ("betti_changed", "rag_changed", "rag8_changed", "hig_changed", "identical"):
        if c in x:
            x[c] = x[c].astype(bool)
    if "bm" in x:
        x["bm_changed"] = x.bm > 0
    return x


# --------------------------------------------------------------------------- Sec. 5.1
section("Sec. 5.1  benchmark node counts and implementation checks")
nc = pd.read_csv(os.path.join(R, "sec51", "node_counts.csv"))
nest = nc[nc.series == "nesting"].set_index("n")
for col, a, b in (("seg", "12.02", "2.28"), ("junctions", "8.04", "1.52"), ("cut", "1.07", "4.00")):
    chk(f"mean {col} nodes at r=5, n=0", nest.loc[0, col], a, "{:.2f}")
    chk(f"mean {col} nodes at r=5, n=4", nest.loc[4, col], b, "{:.2f}")
ck = os.path.join(R, "sec51", "checks.txt")
if os.path.exists(ck):
    last = open(ck).read().strip().splitlines()[-1]
    n, e, h = map(int, re.search(r"segmentations (\d+), Euler identity fails (\d+), regions with cut nodes != holes (\d+)", last).groups())
    chk("segmentations checked", n, "21919")
    chk("Euler identity failures", e, "0")
    chk("regions whose cut nodes differ from their holes", h, "0")

# --------------------------------------------------------------------------- Table 1
section("Table 1  isomorphism classes on the 1298 distinct benchmark images")
t1 = pd.read_csv(os.path.join(R, "sec52", "table1.csv")).set_index("graph")
paper = {"Simple RAG": (894, 1913, "1166 (61.0%)"), "RAG with OUT": (1088, 761, "14 (1.8%)"),
         "RAG with corner contacts": (927, 1799, "1052 (58.5%)"),
         "RAG with corner contacts and OUT": (1096, 750, "3 (0.4%)")}
for g, (cl, pairs, sep) in paper.items():
    r = t1.loc[g]
    chk(f"{g}: classes", int(r.classes), str(cl))
    chk(f"{g}: pairs with the same graph", int(r.pairs_same_graph), str(pairs))
    chk(f"{g}: separated by the HIG", f"{int(r.separated_by_hig)} ({100 * r.separated_by_hig / r.pairs_same_graph:.1f}%)", sep)
    chk(f"{g}: same HIG but different graph", int(r.same_hig_different_graph), "0")
chk("HIG classes", int(t1.loc["HIG", "classes"]), "1099")
s = t1.loc["Simple RAG"]
chk("simple-RAG pairs separated by holes / multiplicities / junction or enclosing",
    f"{int(s.sep_holes)} / {int(s.sep_multiplicities)} / {int(s.sep_junction_or_enclosing)}", "1010 / 69 / 87")
chk("of them already separated by the RAG with OUT", int(s.separated_by_hig - t1.loc["RAG with OUT", "separated_by_hig"]), "1152")
chk("pairs separated by the HIG but by neither augmented RAG", int(t1.loc["Both RAGs with OUT", "separated_by_hig"]), "0")

# --------------------------------------------------------------------------- Table 2
TABLE2 = {  # pairs, pixels %, Dice, Betti, BM, RAG, RAG8, HIG, only vs Betti, RAG, RAG8
    "bench": {"noise": "1298 2.98 0.953 94.6 94.6 94.7 94.7 94.7 0.1 0.0 0.0",
              "erode": "1298 4.99 0.917 11.1 11.1 10.9 13.8 15.7 4.6 4.8 1.9",
              "dilate": "1298 5.13 0.916 7.0 7.0 11.9 11.8 15.7 8.7 3.9 3.9",
              "blob": "1298 0.25 0.996 100.0 100.0 100.0 100.0 100.0 0.0 0.0 0.0",
              "smooth": "1275 0.63 0.985 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0",
              "All": "6467 2.81 0.953 42.7 42.7 43.7 44.2 45.4 2.7 1.7 1.2"},
    "acdc": {"noise": "854 0.36 0.980 99.3 99.3 99.3 99.3 99.3 0.0 0.0 0.0",
             "erode": "854 0.61 0.964 10.4 10.5 9.5 10.5 10.5 0.1 1.1 0.0",
             "dilate": "854 0.64 0.964 9.4 9.6 7.6 10.4 10.4 1.1 2.8 0.0",
             "blob": "854 0.01 0.999 100.0 100.0 100.0 100.0 100.0 0.0 0.0 0.0",
             "smooth": "854 0.25 0.986 6.8 7.1 7.0 6.9 7.1 0.4 0.1 0.2",
             "All": "4270 0.38 0.979 45.2 45.3 44.7 45.4 45.5 0.3 0.8 0.0"}}


def row2(z):
    h = z.hig_changed
    return " ".join([str(len(z)), f"{100 * z.changed_px.mean():.2f}", f"{z.dice.mean():.3f}"] +
                    [f"{pct(z[c]):.1f}" for c in ("betti_changed", "bm_changed", "rag_changed", "rag8_changed", "hig_changed")] +
                    [f"{pct(h & ~z[c]):.1f}" for c in ("betti_changed", "rag_changed", "rag8_changed")])


a = boolcols(pd.read_csv(os.path.join(R, "sec53", "benchmark_perturbations.csv")))
a = a[~a.identical]
b = boolcols(pd.read_csv(os.path.join(R, "sec53", "acdc_reference_perturbations.csv")))
b_all = b
b = b[~b.identical]
for name, x in (("bench", a), ("acdc", b)):
    section(f"Table 2  {'benchmark' if name == 'bench' else 'ACDC references'} (pairs, pixels, Dice, Betti, BM, RAG, RAG8, HIG, HIG only vs Betti/RAG/RAG8)")
    for k, exp in TABLE2[name].items():
        z = x if k == "All" else x[x.perturbation == k]
        chk(k, row2(z), exp)

section("Sec. 5.3  text on the perturbations")
chk("benchmark perturbations that left the image unchanged", 1298 * 5 - len(a), "23")
chk("ACDC reference perturbations that left the image unchanged", int(b_all.identical.sum()), "0")
for name, x in (("benchmark", a), ("ACDC", b)):
    others = x.betti_changed | x.bm_changed | x.rag_changed | x.rag8_changed
    chk(f"{name}: changes of Betti, RAG or RAG8 missed by the HIG", int(((x.betti_changed | x.rag_changed | x.rag8_changed) & ~x.hig_changed).sum()), "0")
    chk(f"{name}: HIG changes missed by both augmented RAGs", int((x.hig_changed & ~x.rag_changed & ~x.rag8_changed).sum()), "0")
    chk(f"{name}: localized differences exactly when the HIGs are non-isomorphic", bool(((x.n_diffs > 0) == x.hig_changed).all()), "True")
chk("benchmark: pairs where Betti matching and the Betti numbers disagree", int((a.bm_changed != a.betti_changed).sum()), "0")
chk("ACDC: pairs flagged by Betti matching with unchanged Betti numbers", int((b.bm_changed & ~b.betti_changed).sum()), "6")
chk("ACDC: of them with isomorphic HIGs", int((b.bm_changed & ~b.betti_changed & ~b.hig_changed).sum()), "3")
hi = a[a.dice >= 0.95]; lo = a[a.dice < 0.90]
chk("benchmark pairs with Dice >= 0.95 / HIG changed", f"{len(hi)} / {pct(hi.hig_changed):.1f}%", "4033 / 51.9%")
chk("benchmark pairs with Dice < 0.90 / HIG unchanged", f"{len(lo)} / {pct(~lo.hig_changed):.1f}%", "741 / 79.5%")
hb = b[b.dice >= 0.95]
chk("ACDC pairs with Dice >= 0.95 / HIG changed", f"{len(hb)} / {pct(hb.hig_changed):.1f}%", "3954 / 47.3%")
chk("benchmark: localized differences within one pixel of a changed pixel", f"{100 * a.n_diffs_near_change.sum() / a.n_diffs.sum():.2f}%", "99.97%")
chk("ACDC: localized differences within one pixel of a changed pixel", f"{100 * b.n_diffs_near_change.sum() / b.n_diffs.sum():.0f}%", "100%")

# --------------------------------------------------------------------------- Table 3
q = pd.concat([boolcols(pd.read_csv(os.path.join(R, "sec53", f"acdc_predictions_{t}.csv")))
               for t in ("unet", "unet_clean", "deeplab", "deeplab_clean")])
q["set"] = q.tag + "_" + q.epoch.astype(str)
TABLE3 = {"unet_2": "0.760 96.4 96.4 95.8 90.6 94.1", "unet_5": "0.841 87.2 86.9 76.7 47.0 54.2",
          "unet_16": "0.851 71.8 71.7 58.2 33.4 47.0", "unet_clean_16": "0.850 47.1 47.0 28.5 26.0 56.5",
          "deeplab_16": "0.899 30.7 30.4 21.6 6.6 22.4", "deeplab_clean_16": "0.899 23.3 22.7 14.8 5.8 26.4"}
section("Table 3  ACDC predictions (Dice, BM, HIG differs all / Dice >= 0.9, constraints violated / errors found)")
S3 = {}
for st, z in q.groupby("set"):
    zz = z[z.gt_plausible == 1]; flagged = zz.pred_plausible == 0
    hi = z[z.dice >= 0.9]
    S3[st] = dict(n=len(z), n09=len(hi), nplaus=len(zz),
                  row=f"{z.dice.mean():.3f} {pct(z.bm_changed):.1f} {pct(z.hig_changed):.1f} {pct(hi.hig_changed):.1f} "
                      f"{pct(flagged):.1f} {pct(flagged[zz.hig_changed]):.1f}",
                  hig_not_bm=int((z.hig_changed & ~z.bm_changed).sum()), bm_not_hig=int((z.bm_changed & ~z.hig_changed).sum()),
                  missed=int(((z.betti_changed | z.rag_changed | z.rag8_changed) & ~z.hig_changed).sum()),
                  only_rag=int((z.hig_changed & ~z.rag_changed).sum()), only_rag8=int((z.hig_changed & ~z.rag8_changed).sum()),
                  only_both=int((z.hig_changed & ~z.rag_changed & ~z.rag8_changed).sum()),
                  loc_iff=bool(((z.n_diffs > 0) == z.hig_changed).all()))
for st, exp in TABLE3.items():
    chk(st, S3[st]["row"], exp)

section("Sec. 5.3  text on the ACDC predictions")
chk("slices per prediction set", sorted({v["n"] for v in S3.values()}), "[854]")
chk("U-Net epoch 16: slices with Dice >= 0.9", S3["unet_16"]["n09"], "469")
chk("U-Net epoch 16 cleaned: slices with Dice >= 0.9", S3["unet_clean_16"]["n09"], "471")
chk("DeepLabv3+ epoch 16: slices with Dice >= 0.9", S3["deeplab_16"]["n09"], "615")
chk("slices with a plausible reference", f"{S3['unet_16']['nplaus']} ({100 * S3['unet_16']['nplaus'] / 854:.1f}%)", "832 (97.4%)")
chk("changes of Betti, RAG or RAG8 missed by the HIG, any stored prediction", sum(v["missed"] for v in S3.values()), "0")
un = [k for k in S3 if k.startswith("unet")]
chk("U-Net: largest share of slices detected only by the HIG against the RAG is below 0.5%",
    max(S3[k]["only_rag"] for k in un) / 854 < 0.005, "True")
chk("U-Net: largest share against the RAG with corner contacts", f"{100 * max(S3[k]['only_rag8'] for k in un) / 854:.2f}%", "0.23%")
chk("U-Net: slices detected only by the HIG against both RAGs", sum(S3[k]["only_both"] for k in un), "0")
chk("U-Net: localized differences exactly when the HIGs are non-isomorphic", all(S3[k]["loc_iff"] for k in un), "True")
chk("U-Net epoch 16: slices flagged by only one of HIG and Betti matching",
    f"{S3['unet_16']['hig_not_bm'] + S3['unet_16']['bm_not_hig']} (HIG only {S3['unet_16']['hig_not_bm']})", "3 (HIG only 1)")
x = q[(q.set == "unet_16") & q.bm_changed & ~q.hig_changed]
chk("  Betti-matching-only slices with unchanged Betti numbers", int((~x.betti_changed).sum()), "2")
chk("largest disagreement between HIG and Betti matching, any stored prediction",
    max(v["hig_not_bm"] + v["bm_not_hig"] for v in S3.values()), "9")
f1 = q[(q.set == "unet_clean_16") & (q.slice == 266)].iloc[0]
chk("Fig. 6, slice 266: Dice", f1.dice, "0.935", "{:.3f}")
chk("Fig. 6, slice 266: Betti / BM / RAG8 / RAG / HIG changed",
    " ".join(str(bool(f1[c])) for c in ("betti_changed", "bm_changed", "rag8_changed", "rag_changed", "hig_changed")),
    "False False False True True")
chk("Fig. 6, slice 266: reference and prediction plausible", f"{int(f1.gt_plausible)} {int(f1.pred_plausible)}", "0 0")
f2 = q[(q.set == "unet_16") & (q.slice == 865)].iloc[0]
chk("Fig. 6, slice 865: Dice", f2.dice, "0.965", "{:.3f}")

# --------------------------------------------------------------------------- Table 4 and Sec. 5.4
section("Table 4 and Sec. 5.4  sizes and times")
e1 = pd.read_csv(os.path.join(R, "sec54", "e1_query.csv"))
e2 = pd.read_csv(os.path.join(R, "sec54", "e2_scaling.csv"))
e3 = pd.read_csv(os.path.join(R, "sec54", "e3_sizes_acdc.csv"))
cols = ["rag_nodes", "complex_incid", "frontier_edges", "hig_gen", "hig_nnz", "rag8_edges", "tmap_darts"]
fmt = lambda s: " ".join(f"{int(round(v))}" for v in s)
m3 = e3.groupby("seg")[cols].median()
chk("ACDC reference", fmt(m3.loc["reference"]), "4 295680 1339 16 20 5 10")
chk("ACDC, epoch 16", fmt(m3.loc["epoch 16"]), "5 295680 1356 20 24 6 12")
e1["bin"] = pd.cut(e1.regions, [0, 300, 1000, 3000, 10000, 1e9], labels=["<300", "300-1k", "1k-3k", "3k-10k", ">10k"])
m1 = e1.groupby("bin", observed=True)[cols].median()
chk("Quantized, < 300", fmt(m1.loc["<300"]), "119 295680 6691 520 794 201 400")
chk("Quantized, 1k-3k", fmt(m1.loc["1k-3k"]), "1864 295680 17286 8438 16334 6124 8186")
chk("Quantized, > 10k", fmt(m1.loc[">10k"]), "14808 295680 45504 65562 130855 56011 65462")
chk("quantized segmentations / regions", f"{len(e1)} / {e1.regions.min()} to {e1.regions.max()}", "900 / 35 to 23000")
big = m1.loc[">10k"]
chk("nodes and incidences per region with many regions", f"{big.hig_gen / big.rag_nodes:.0f} {big.hig_nnz / big.rag_nodes:.0f}", "4 9")
g2 = e2.groupby("side").t_hig_build.median()
chk("construction time, 128 x 128", f"{1e3 * g2.loc[128]:.1f} ms", "3.8 ms")
chk("construction time, 2048 x 2048", f"{g2.loc[2048]:.2f} s", "0.31 s")
nspp = [1e9 * g2.loc[s] / s ** 2 for s in (512, 1024, 2048)]
chk("ns per pixel from 512 x 512 on", f"{min(nspp):.0f} to {max(nspp):.0f}", "63 to 75")
chk("holes of every region from the HIG, slowest case below 0.3 ms", bool(e1.t_hig_query.max() < 3e-4), "True")
sm = e1[e1.t_rag8_naive.notna()]
ratio = max((sm.t_rag8_naive / sm.t_hig_query).median(), (sm.t_pixel_naive / sm.t_hig_query).median())
chk("region-by-region search, times longer (median, rounded to hundreds)", f"{round(ratio, -2):.0f}", "7700")
chk("low-link search of the RAG8, slowest case below 2 ms", bool(e1.t_rag8_lowlink.max() < 2e-3), "True")
chk("all hole computations agree", bool(e1.agree.all() and e2.agree.all()), "True")

print(f"\n{N_OK} numbers agree with the paper, {N_BAD} differ.")
sys.exit(1 if N_BAD else 0)
