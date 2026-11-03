"""Per-class Betti matching error (Stucki et al., ICML 2023; per class as in Berger et al., 2024).

Uses the Betti-Matching-3D library (https://github.com/nstucki/Betti-Matching-3D), whose Python
module `betti_matching` must be importable: either installed, or built with cmake and its build
directory given in the environment variable BETTI_MATCHING_BUILD.

V-construction with sublevel sets of 1 - mask: each class is 4-connected and its holes are the
bounded 8-connected components of its complement, as for the Betti numbers of the paper. The
image is padded with one background pixel and a dummy foreground pixel is put in the padded
corner of both images, so that the essential component is the dummy and every real component
is a finite interval (otherwise a class present in one image and absent from the other would
not be counted).
"""
import os
import sys

import numpy as np

if os.environ.get("BETTI_MATCHING_BUILD"):
    sys.path.insert(0, os.environ["BETTI_MATCHING_BUILD"])
try:
    import betti_matching as bm
except ImportError as e:  # pragma: no cover
    raise ImportError("Betti matching needs the module betti_matching from "
                      "https://github.com/nstucki/Betti-Matching-3D; build it and set "
                      "BETTI_MATCHING_BUILD to its build directory (see README).") from e


def _vol(mask):
    v = np.ones((mask.shape[0] + 2, mask.shape[1] + 2), np.float64)
    v[1:-1, 1:-1] = 1.0 - mask
    v[0, 0] = 0.0
    return v


def barcode_counts(mask):
    r = bm.compute_barcode(_vol(mask))
    return len(r.birth_coordinates[0]), len(r.birth_coordinates[1])


def match_counts(P, G, labels=None):
    """Per class: (matched0, matched1, unmatchedP0, unmatchedP1, unmatchedG0, unmatchedG1)."""
    if labels is None:
        labels = sorted(set(np.unique(P).tolist()) | set(np.unique(G).tolist()))
    out = {}
    for l in labels:
        r = bm.compute_matching(_vol(P == l), _vol(G == l))
        m, uP, uG = np.asarray(r.num_matched), np.asarray(r.num_unmatched_input1), np.asarray(r.num_unmatched_input2)
        out[int(l)] = (int(m[0]), int(m[1]) if len(m) > 1 else 0, int(uP[0]), int(uP[1]) if len(uP) > 1 else 0,
                       int(uG[0]), int(uG[1]) if len(uG) > 1 else 0)
    return out


def errors(P, G):
    """Betti matching error split by dimension and by class: dict with bm0, bm1, bm, bm0_L<l>, bm1_L<l>."""
    rec, e0, e1 = {}, 0, 0
    for l, (m0, m1, p0, p1, g0, g1) in match_counts(P, G).items():
        rec[f"bm0_L{l}"] = p0 + g0; rec[f"bm1_L{l}"] = p1 + g1
        e0 += p0 + g0; e1 += p1 + g1
    rec.update(bm0=e0, bm1=e1, bm=e0 + e1)
    return rec
