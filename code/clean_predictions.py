"""Post-processing of Sec. 5.3: every 4-connected component smaller than 20 pixels takes the most
frequent label around it.

usage: python3 clean_predictions.py RUN_DIR OUT_DIR [FILE]   (FILE defaults to test_pred_ep16.npy)
Copies test_gt.npy, test_meta.tsv (and test_img.npy if present) and writes the cleaned FILE.
"""
import os
import shutil
import sys

import numpy as np
import scipy.ndimage as ndi

S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
MIN = 20


def clean(s):
    s = s.copy()
    for l in np.unique(s):
        lab, n = ndi.label(s == l, structure=S4)
        if n <= 1 and l == 0:
            continue
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        for k, sz in enumerate(sizes, 1):
            if sz < MIN:
                m = lab == k
                ring = ndi.binary_dilation(m, structure=S4) & ~m
                vals, cnt = np.unique(s[ring], return_counts=True)
                if len(vals):
                    s[m] = vals[np.argmax(cnt)]
    return s


if __name__ == "__main__":
    run, out = sys.argv[1], sys.argv[2]
    name = sys.argv[3] if len(sys.argv) > 3 else "test_pred_ep16.npy"
    os.makedirs(out, exist_ok=True)
    for f in ("test_gt.npy", "test_meta.tsv", "test_img.npy"):
        if os.path.exists(os.path.join(run, f)):
            shutil.copy(os.path.join(run, f), out)
    P = np.load(os.path.join(run, name))
    Q = np.stack([clean(P[i]) for i in range(len(P))])
    np.save(os.path.join(out, name), Q)
    print("changed pixels:", int((Q != P).sum()))
