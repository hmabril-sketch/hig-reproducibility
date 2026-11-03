"""Ways of obtaining, for every region, its (4,8) holes and the geometric nodes of the
interface with each hole.

  hig_query      : read from the HIG (cut layer + node-to-component map of the construction)
  rag8_build     : RAG with corner contacts and OUT (sparse adjacency), from the label image
  rag8_naive     : Lemma 4.15 applied region by region (delete R, connected components)
  rag8_lowlink   : one depth-first search with low-links rooted at OUT (block-cut tree): the holes
                   of R are the DFS children v of R with low[v] >= disc[R]
  pixel_naive    : flood fill of the complement of every region (8-adjacency), as in the
                   binary per-region approach
Every function returns holes as an int array indexed by region.
"""
import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy import ndimage as ndi


def hig_query(d):
    """Holes per region and, for each cut, the geometric nodes of its inner interface."""
    nr = d["nreg"]
    holes = np.bincount(d["carriers"], minlength=nr + 1)[:nr]
    # carrier of the component of every geometric node (-1 if the component is not a cut)
    comp_car = np.full(len(d["comp_is_cut"]), -1, np.int64)
    comp_car[d["cut_comp"]] = d["carriers"]
    car = comp_car[d["class_comp"]]
    inner = (car >= 0) & ((d["gpair"][:, 0] == car) | (d["gpair"][:, 1] == car))
    gens = np.nonzero(inner)[0]
    order = np.argsort(d["class_comp"][gens], kind="stable")   # grouped by cut
    return holes, gens[order]


def rag8_build(ids, nreg):
    h, w = ids.shape
    a = []; b = []
    for s1, s2 in ((np.s_[:, :-1], np.s_[:, 1:]), (np.s_[:-1, :], np.s_[1:, :]),
                   (np.s_[:-1, :-1], np.s_[1:, 1:]), (np.s_[:-1, 1:], np.s_[1:, :-1])):
        x, y = ids[s1].ravel(), ids[s2].ravel(); m = x != y
        a.append(x[m]); b.append(y[m])
    frame = np.unique(np.concatenate([ids[0], ids[-1], ids[:, 0], ids[:, -1]]))
    a.append(frame); b.append(np.full(len(frame), nreg, ids.dtype))
    a = np.concatenate(a).astype(np.int64); b = np.concatenate(b).astype(np.int64)
    key = np.unique(np.concatenate([np.minimum(a, b) * (nreg + 1) + np.maximum(a, b)]))
    u, v = key // (nreg + 1), key % (nreg + 1)
    n = nreg + 1
    A = sparse.csr_matrix((np.ones(2 * len(u), np.int32), (np.concatenate([u, v]), np.concatenate([v, u]))),
                          shape=(n, n))
    return A


def rag8_naive(A, nreg):
    holes = np.zeros(nreg, np.int64)
    n = nreg + 1
    keep_all = np.arange(n)
    for r in range(nreg):
        keep = np.delete(keep_all, r)
        sub = A[keep][:, keep]
        nc, lab = connected_components(sub, directed=False)
        holes[r] = nc - 1
    return holes


@njit(cache=True)
def _lowlink(indptr, indices, n, root):
    disc = np.full(n, -1, np.int64); low = np.zeros(n, np.int64); parent = np.full(n, -1, np.int64)
    holes = np.zeros(n, np.int64)
    it = np.zeros(n, np.int64)
    stack = np.empty(n, np.int64); sp = 0
    t = 0
    disc[root] = t; low[root] = t; t += 1
    stack[sp] = root; sp += 1; it[root] = indptr[root]
    while sp > 0:
        u = stack[sp - 1]
        if it[u] < indptr[u + 1]:
            v = indices[it[u]]; it[u] += 1
            if disc[v] == -1:
                parent[v] = u; disc[v] = t; low[v] = t; t += 1
                stack[sp] = v; sp += 1; it[v] = indptr[v]
            elif v != parent[u]:
                if disc[v] < low[u]:
                    low[u] = disc[v]
        else:
            sp -= 1
            p = parent[u]
            if p >= 0:
                if low[u] < low[p]:
                    low[p] = low[u]
                if p != root and low[u] >= disc[p]:
                    holes[p] += 1
    return holes


def rag8_lowlink(A, nreg):
    return _lowlink(A.indptr.astype(np.int64), A.indices.astype(np.int64), nreg + 1, nreg)[:nreg]


S8 = np.ones((3, 3), bool)


def pixel_naive(ids, nreg):
    holes = np.zeros(nreg, np.int64)
    for r in range(nreg):
        comp, n = ndi.label(np.pad(ids != r, 1, constant_values=True), structure=S8)
        holes[r] = n - 1
    return holes
