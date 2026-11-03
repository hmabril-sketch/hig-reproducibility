"""Labelled region adjacency graph with corner contacts and OUT (RAG8 in Tables 2-4).

Regions are the 4-connected components of each label; two regions are adjacent when two of
their pixels are 8-adjacent, and a region is adjacent to OUT when it contains a frame pixel.
This is the regional 8-contact graph of Definition 4.14, computed here from the pixels.
"""
import networkx as nx
import numpy as np
import scipy.ndimage as ndi

S4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def rag8_out(seg):
    seg = np.asarray(seg)
    ids = np.zeros(seg.shape, np.int64); lab = []; n = 0
    for l in np.unique(seg):
        c, k = ndi.label(seg == l, structure=S4)
        m = c > 0
        ids[m] = c[m] - 1 + n; lab += [int(l)] * k; n += k
    G = nx.Graph()
    G.add_nodes_from((r, {"a": f"L{lab[r]}"}) for r in range(n)); G.add_node(n, a="OUT")
    for a, b in ((ids[:, :-1], ids[:, 1:]), (ids[:-1, :], ids[1:, :]),
                 (ids[:-1, :-1], ids[1:, 1:]), (ids[:-1, 1:], ids[1:, :-1])):
        d = a != b
        G.add_edges_from(set(zip(a[d].tolist(), b[d].tolist())))
    frame = set(ids[0].tolist()) | set(ids[-1].tolist()) | set(ids[:, 0].tolist()) | set(ids[:, -1].tolist())
    G.add_edges_from((r, n) for r in frame)
    return G
