"""Sec. 5.4: summaries of the measurements of sec54_size_cost.py and the log-log figure
(fig_size_cost.pdf/png: holes of every region against the number of regions, and construction
time against the number of pixels).

usage: python3 sec54_analyse.py OUT_DIR
Reference output: results/sec54/analysis.txt and fig_size_cost.pdf.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1]
e1 = pd.read_csv(os.path.join(OUT, "e1_query.csv"))
e2 = pd.read_csv(os.path.join(OUT, "e2_scaling.csv"))
e3 = pd.read_csv(os.path.join(OUT, "e3_sizes_acdc.csv"))
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

print("E1 segmentations", len(e1), "regions", e1.regions.min(), "-", e1.regions.max(),
      "| all methods agree:", bool(e1.agree.all()))
bins = [0, 300, 1000, 3000, 10000, 1e9]
labels = ["<300", "300-1k", "1k-3k", "3k-10k", ">10k"]
e1["bin"] = pd.cut(e1.regions, bins, labels=labels)
ms = lambda s: 1e3 * s
g = e1.groupby("bin", observed=True)
t1 = pd.DataFrame({"n": g.size(), "regions": g.regions.median(), "holes": g.holes.median(),
                   "HIG build": ms(g.t_hig_build.median()), "HIG query": ms(g.t_hig_query.median()),
                   "RAG8 build": ms(g.t_rag8_build.median()), "RAG8 lowlink": ms(g.t_rag8_lowlink.median()),
                   "RAG8 per-region": ms(g.t_rag8_naive.median()), "pixel per-region": ms(g.t_pixel_naive.median())})
print(t1.round(2).to_string())
for c in ("t_rag8_naive", "t_pixel_naive", "t_hig_query", "t_rag8_lowlink", "t_hig_build", "t_rag8_build"):
    m = e1[c].notna() & (e1.regions >= 50)
    k = np.polyfit(np.log(e1.regions[m]), np.log(e1[c][m]), 1)[0]
    print(f"log-log slope vs regions, {c}: {k:.2f}")
small = e1[e1.t_rag8_naive.notna()]
print("speed-up of HIG query over per-region RAG8 search, median:",
      round(float((small.t_rag8_naive / small.t_hig_query).median()), 0),
      "| over per-region pixel flood fill:", round(float((small.t_pixel_naive / small.t_hig_query).median()), 0))
print("build+query, HIG vs RAG8+lowlink (median ratio):",
      round(float(((e1.t_hig_build + e1.t_hig_query) / (e1.t_rag8_build + e1.t_rag8_lowlink)).median()), 2))

print("\nE2 scaling, all methods agree:", bool(e2.agree.all()))
g2 = e2.groupby("side")
t2 = pd.DataFrame({"pixels": g2.pixels.first(), "regions": g2.regions.median(), "HIG build ms": ms(g2.t_hig_build.median()),
                   "HIG query ms": ms(g2.t_hig_query.median()), "RAG8 build ms": ms(g2.t_rag8_build.median()),
                   "RAG8 lowlink ms": ms(g2.t_rag8_lowlink.median())})
print(t2.round(2).to_string())
for c in ("t_hig_build", "t_rag8_build", "t_rag8_lowlink", "t_hig_query"):
    k = np.polyfit(np.log(e2.pixels), np.log(e2[c]), 1)[0]
    print(f"log-log slope vs pixels, {c}: {k:.2f}")

print("\nE3 sizes (medians) on ACDC segmentations")
cols = ["pixels", "complex_cells", "complex_incid", "frontier_edges", "hig_gen", "hig_nnz", "hig_cuts",
        "rag_nodes", "rag_edges", "rag8_nodes", "rag8_edges", "tmap_darts"]
print(e3.groupby("seg")[cols].median().round(1).to_string())
print("\nE1 sizes by number of regions (medians)")
print(e1.groupby("bin", observed=True)[cols].median().round(0).to_string())

# ------------------------------------------------------------------ figure
blue, orange, aqua, yellow = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
ink, muted, grid = "#1f1f1e", "#6b6a64", "#e4e3de"
plt.rcParams.update({"font.size": 7, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                     "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(5.0, 2.9))
a = ax[0]
for c, col, mk, lab in (("t_pixel_naive", yellow, "v", "pixel flood fill per region"),
                        ("t_rag8_naive", aqua, "s", "RAG$_8$ search per region"),
                        ("t_rag8_lowlink", orange, "^", "RAG$_8$ low-link DFS"),
                        ("t_hig_query", blue, "o", "HIG cut layer")):
    m = e1[c].notna()
    a.scatter(e1.regions[m], 1e3 * e1[c][m], s=7, color=col, marker=mk, linewidths=0, alpha=0.75, label=lab)
a.set_xscale("log"); a.set_yscale("log")
a.set_xlabel("regions"); a.set_ylabel("time for all holes (ms)")
a.grid(True, which="major", color=grid, lw=0.6); a.set_axisbelow(True)
a.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.26), ncol=1, markerscale=1.6, columnspacing=1.0, handletextpad=0.3)
a.set_title("(a) holes of every region", fontsize=7.5, color=ink, loc="left")
b = ax[1]
g2m = e2.groupby("pixels").median(numeric_only=True)
for c, col, mk, lab in (("t_hig_build", blue, "o", "HIG construction"),
                        ("t_rag8_build", orange, "^", "RAG$_8$ construction")):
    b.plot(g2m.index, 1e3 * g2m[c], color=col, marker=mk, ms=4, lw=1.5, label=lab)
    b.scatter(e2.pixels, 1e3 * e2[c], s=5, color=col, alpha=0.35, linewidths=0)
xs = np.array([g2m.index.min(), g2m.index.max()])
y0 = 1e3 * g2m.t_hig_build.iloc[0]
b.plot(xs, y0 * xs / xs[0], color=muted, lw=0.8, ls="--", label="linear")
b.set_xscale("log"); b.set_yscale("log")
b.set_xlabel("pixels"); b.set_ylabel("time (ms)")
b.grid(True, which="major", color=grid, lw=0.6); b.set_axisbelow(True)
b.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=1, columnspacing=1.0, handletextpad=0.4)
b.set_title("(b) construction", fontsize=7.5, color=ink, loc="left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_size_cost.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "fig_size_cost.png"), dpi=200, bbox_inches="tight")
print("figure saved")
