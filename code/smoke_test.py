"""Quick checks that need neither the ACDC data nor the Betti matching library (a few seconds).

1. The HIG of the example of Fig. 1 has the incidence matrices of Fig. 4 (up to the order of
   the nodes) and satisfies |V2| - |V1| + |V0| = 2.
2. The compiled construction (hig_fast) gives the same HIG as the reference implementation on
   200 random label images.
3. On 300 random label images, every region carries as many cut nodes as it has (4,8) holes,
   computed from its Euler number.
usage: python3 smoke_test.py
"""
import networkx as nx
import numpy as np

from euler_regions import holes_from_euler
from hig_construct import construct
from hig_fast import construct_fast

# ---------------------------------------------------------------- 1. Fig. 1 and Fig. 4
FIG1 = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 2, 0], [0, 1, 0, 3, 0], [0, 0, 0, 3, 0],
                 [0, 4, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]], np.int32)
FIG4_M01 = """0 0 0 0 0 0 0 0 0 0 0 2
0 1 1 1 0 0 0 0 0 0 0 0
0 1 0 0 1 0 2 0 0 0 0 0
0 0 1 0 1 1 0 1 0 0 0 0
0 0 0 1 0 1 0 1 0 0 0 0
0 0 0 0 0 0 0 0 0 2 2 0"""
FIG4_M12 = """0 0 0 0 0 0 2 0
1 0 0 0 0 0 1 0
1 1 0 0 0 0 0 0
0 1 0 0 0 0 1 0
1 0 0 0 0 0 1 0
0 1 0 1 0 0 0 0
0 0 1 0 0 0 1 0
0 0 0 1 0 0 1 0
0 0 0 0 0 0 2 0
0 0 0 0 1 0 1 0
0 0 0 0 0 1 1 0
0 0 0 0 0 0 1 1"""


def incidence_graph(m01, m12):
    """Graded incidence multigraph; the last column of m12 is OUT."""
    G = nx.Graph()
    n0, n1 = m01.shape; n2 = m12.shape[1]
    G.add_nodes_from((("v", i), {"d": 0}) for i in range(n0))
    G.add_nodes_from((("e", i), {"d": 1}) for i in range(n1))
    G.add_nodes_from((("R", i), {"d": "OUT" if i == n2 - 1 else 2}) for i in range(n2))
    for i, j in zip(*np.nonzero(m01)):
        G.add_edge(("v", i), ("e", j), m=int(m01[i, j]))
    for i, j in zip(*np.nonzero(m12)):
        G.add_edge(("e", i), ("R", j), m=int(m12[i, j]))
    return G


mat = lambda s: np.array([[int(v) for v in l.split()] for l in s.splitlines()])
ids, edges, geom, cuts, carriers, holes, m01, m12 = construct(FIG1)
m01, m12 = m01.toarray(), m12.toarray()
assert (m01.shape, m12.shape) == ((6, 12), (12, 8)), (m01.shape, m12.shape)
assert m12.shape[1] - m12.shape[0] + m01.shape[0] == 2
same = nx.is_isomorphic(incidence_graph(m01, m12), incidence_graph(mat(FIG4_M01), mat(FIG4_M12)),
                        node_match=lambda a, b: a["d"] == b["d"], edge_match=lambda a, b: a["m"] == b["m"])
assert same, "the HIG of Fig. 1 does not have the matrices of Fig. 4"
print("Fig. 1 / Fig. 4: matrices agree up to the order of the nodes; |V0|, |V1|, |V2| = 6, 12, 8")

# ---------------------------------------------------------------- 2-3. random label images
rng = np.random.default_rng(0)
for t in range(300):
    h, w = rng.integers(2, 25, size=2)
    A = rng.integers(0, int(rng.integers(2, 6)), size=(h, w)).astype(np.int32)
    d = construct_fast(A)
    hh = np.bincount(d["carriers"], minlength=d["nreg"] + 1)[:d["nreg"]]
    assert np.array_equal(hh, holes_from_euler(d["ids"], d["nreg"])), "cut nodes != holes"
    chi = (d["nreg"] + 1) - (len(d["gtype"]) + len(d["cut_comp"])) + d["m01"].shape[0]
    assert chi == 2
    if t < 200:
        ids, edges, geom, cuts, carriers, holes, r01, r12 = construct(A)
        assert np.array_equal(ids, d["ids"])
        assert sorted(carriers) == sorted(int(c) for c in d["carriers"])
        assert r01.nnz == d["m01"].nnz and r12.nnz == d["m12"].nnz and r01.sum() == d["m01"].sum()
print("random images: compiled and reference constructions agree; cut nodes = (4,8) holes; chi = 2")
print("smoke test passed")
