"""Topological quality control of labelled segmentations with the HIG.

For a reference segmentation G and a test segmentation P (same label set):
  * Dice per label (macro mean);
  * per-label Betti numbers (b0 = 4-components of the label set,
    b1 = bounded 8-components of its complement), the usual "Betti error";
  * labelled RAG with OUT (side adjacency), compared up to isomorphism;
  * typed HIG (regions with labels, typed 1-nodes, junctions,
    cut nodes, integer multiplicities), compared up to isomorphism;
  * a label-level HIG signature whose differing entries are localized by
    matching nodes of P and G through their geometric support.
"""
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import scipy.ndimage as ndi

from hig_construct import construct

S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)
S8 = np.ones((3, 3), bool)
OUTL = -1


# ------------------------------------------------------------------ HIG
class HIG:
    def __init__(self, seg):
        seg = np.asarray(seg).astype(np.int32)
        self.seg = seg
        (self.ids, self.edges, self.geom, self.cuts, self.carriers,
         self.holes, self.m01, self.m12) = construct(seg)
        self.out = self.m12.shape[1] - 1
        lab = np.full(self.out + 1, OUTL, dtype=np.int64)
        lab[self.ids.ravel()] = seg.ravel()
        self.lab = lab                       # label of every 2-node (OUT -> -1)
        h, w = seg.shape
        self.h, self.w = h, w
        # junction vertices = endpoints of seg/enc1 nodes
        at = defaultdict(set)
        for i, (a, b, pair, _) in enumerate(self.edges):
            at[a].update(pair)
            at[b].update(pair)
        self.vertex_regions = at
        self.junctions = sorted({v for g in self.geom for v in g["endpoints"]})

    # ---- typed incidence multigraph (for exact isomorphism)
    def graph(self):
        G = nx.Graph()
        n0, n1 = self.m01.shape
        nj = n0 - sum(g["type"] == "enc0" for g in self.geom)
        for v in range(n0):
            G.add_node(("v", v), a="junc" if v < nj else "v0enc")
        for k in range(n1):
            G.add_node(("e", k), a=self.geom[k]["type"] if k < len(self.geom) else "cut")
        for r in range(self.out + 1):
            G.add_node(("R", r), a=("OUT" if r == self.out else f"L{self.lab[r]}"))
        c = self.m01.tocoo()
        for v, k, m in zip(c.row, c.col, c.data):
            G.add_edge(("v", int(v)), ("e", int(k)), m=int(m))
        c = self.m12.tocoo()
        for k, r, m in zip(c.row, c.col, c.data):
            G.add_edge(("e", int(k)), ("R", int(r)), m=int(m))
        return G

    def rag_out(self):
        G = nx.Graph()
        for r in range(self.out + 1):
            G.add_node(r, a=("OUT" if r == self.out else f"L{self.lab[r]}"))
        for (a, b, pair, _) in self.edges:
            G.add_edge(*pair)
        return G

    # ---- label-level signature with geometric support for localization
    def items(self):
        """List of (key, support) for every node, keyed by labels only."""
        L = lambda r: int(self.lab[r])
        out = []
        for r in range(self.out):
            out.append((("region", L(r)), ("pixels", r)))
        for g in self.geom:
            a, b = g["pair"]
            la, lb = sorted((L(a), L(b)))
            sup = frozenset(self.edges[i][:2] for i in g["edges"])
            out.append((("contact", la, lb, g["type"]), ("edges", sup)))
        for k, group in enumerate(self.cuts):
            car = self.carriers[k]
            inner = set()
            for i in group:
                inner.update(self.edges[i][2])
            inner.discard(car)
            key = ("hole", L(car), tuple(sorted({L(r) for r in inner})))
            sup = frozenset(self.edges[i][:2] for i in group)
            out.append((key, ("edges", sup)))
        for v in self.junctions:
            key = ("junction", tuple(sorted(L(r) for r in self.vertex_regions[v])))
            out.append((key, ("vertex", v)))
        return out


# ------------------------------------------------------------------ baselines
def dice_macro(G, P):
    labels = np.unique(G)            # classes present in the reference
    d = []
    for l in labels:
        a, b = G == l, P == l
        s = a.sum() + b.sum()
        d.append(2.0 * np.logical_and(a, b).sum() / s if s else 1.0)
    return float(np.mean(d))


def betti_per_label(seg):
    res = {}
    for l in np.unique(seg):
        m = seg == l
        b0 = ndi.label(m, structure=S4)[1]
        comp, n = ndi.label(np.pad(~m, 1, constant_values=True), structure=S8)
        b1 = n - 1  # the padded frame joins every complement component that meets it
        res[int(l)] = (int(b0), int(b1))
    return res


def same_graph(G1, G2, edge_attr=False):
    if G1.number_of_nodes() != G2.number_of_nodes() or G1.number_of_edges() != G2.number_of_edges():
        return False
    nm = lambda x, y: x["a"] == y["a"]
    em = (lambda x, y: x.get("m") == y.get("m")) if edge_attr else None
    return nx.is_isomorphic(G1, G2, node_match=nm, edge_match=em)


# ------------------------------------------------------------------ localization
def _support_overlap(s1, s2, tol=1):
    """Supports match when they share an element or, for small shifts,
    lie within tol grid units of each other."""
    k1, v1 = s1
    k2, v2 = s2
    if k1 != k2:
        return False
    if k1 == "vertex":
        return abs(v1[0] - v2[0]) <= tol and abs(v1[1] - v2[1]) <= tol
    if k1 == "edges":
        if v1 & v2:
            return True
        p1 = {p for e in v1 for p in e}
        p2 = {p for e in v2 for p in e}
        return any((y + dy, x + dx) in p2 for (y, x) in p1
                   for dy in range(-tol, tol + 1) for dx in range(-tol, tol + 1))
    return False


def _centroid(h, sup):
    kind, v = sup
    if kind == "pixels":
        ys, xs = np.nonzero(h.ids == v)
        return np.array([ys.mean(), xs.mean()])
    if kind == "vertex":
        return np.array(v, dtype=float) - 0.5
    pts = np.array([p for e in v for p in e], dtype=float)
    return pts.mean(axis=0) - 0.5


def localized_differences(hG, hP):
    """Nodes that have no counterpart of the same label-level key.

    For every key (region label; contact label pair and type; hole carrier
    label and inner labels; junction label multiset) the nodes of P and
    G are paired by a minimum-total-distance assignment of their supports,
    so that only the excess nodes of each key are reported, at their
    own location. Equal counts for every key give no difference, however
    far the boundaries moved."""
    from scipy.optimize import linear_sum_assignment

    byP, byG = defaultdict(list), defaultdict(list)
    for key, sup in hP.items():
        byP[key].append(sup)
    for key, sup in hG.items():
        byG[key].append(sup)
    diffs = []
    for key in set(byP) | set(byG):
        sp, sg = byP.get(key, []), byG.get(key, [])
        if len(sp) == len(sg):
            continue
        if sp and sg:
            cp = np.array([_centroid(hP, s) for s in sp])
            cg = np.array([_centroid(hG, s) for s in sg])
            cost = np.linalg.norm(cp[:, None, :] - cg[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(cost)
            mp, mg = set(ri.tolist()), set(ci.tolist())
        else:
            mp, mg = set(), set()
        diffs += [("P", key, s) for i, s in enumerate(sp) if i not in mp]
        diffs += [("G", key, s) for j, s in enumerate(sg) if j not in mg]
    return diffs


def support_pixels(h, sup):
    """Pixel coordinates near a support, for plotting and for checking
    that a difference lies where pixels were changed."""
    kind, v = sup
    if kind == "pixels":
        ys, xs = np.nonzero(h.ids == v)
        return list(zip(ys.tolist(), xs.tolist()))
    pts = set()
    verts = [v] if kind == "vertex" else [p for e in v for p in e]
    for (y, x) in verts:
        for py in (y - 1, y):
            for px in (x - 1, x):
                if 0 <= py < h.h and 0 <= px < h.w:
                    pts.add((py, px))
    return sorted(pts)
