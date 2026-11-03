"""Sec. 5.1: typed node counts of the controlled benchmark (100 images per condition), as in the
text ("At r = 5, increasing the nesting parameter from n = 0 to n = 4 ...").

usage: python3 sec51_node_counts.py OUT_CSV
Reference output: results/sec51/node_counts.csv. About 10 s.
"""
import sys

import numpy as np
import pandas as pd

from benchmark import BASE_SEED, N
from hig_construct import construct
from rectgen import generate_rectangular_segmentation

# two series, as in the paper: complexity (n = 0, r = 1..8) and nesting (r = 5, n = 0..4)
conds = [(r, 0, "complexity", lambda j, r=r: BASE_SEED + r * 1000 + j) for r in range(1, 9)]
conds += [(5, n, "nesting", lambda j, n=n: BASE_SEED + n * 1000 + j) for n in range(5)]
rows = []
for r, n, name, seed in conds:
    for j in range(N):
        A = generate_rectangular_segmentation(r, n, seed(j))[0].astype(np.int32)
        ids, edges, geom, cuts, carriers, holes, m01, m12 = construct(A)
        nenc0 = sum(g["type"] == "enc0" for g in geom)
        n0, n1, n2 = m01.shape[0], m01.shape[1], m12.shape[1]
        rows.append(dict(series=name, r=r, n=n, image=j, regions=n2 - 1,
                         seg=sum(g["type"] == "seg" for g in geom), enc1=sum(g["type"] == "enc1" for g in geom),
                         enc0=nenc0, cut=len(cuts), junctions=n0 - nenc0, virtual=nenc0,
                         chi=n2 - n1 + n0))
d = pd.DataFrame(rows)
t = d.groupby(["series", "r", "n"], sort=False).mean(numeric_only=True).drop(columns="image").reset_index()
t.to_csv(sys.argv[1], index=False)
print(t.round(2).to_string(index=False))
print("chi = 2 in", int((d.chi == 2).sum()), "of", len(d), "segmentations")
