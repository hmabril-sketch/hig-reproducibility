"""Table 1 (Sec. 5.2): isomorphism classes of the simple RAG, the RAG with OUT, the RAG with corner
contacts, the RAG with corner contacts and OUT, and the typed HIG over the 1298 distinct images of
the benchmark, and the pairs of images that share a RAG class but not a HIG class.
Classes are computed exactly: Weisfeiler-Lehman hashes, then an exact isomorphism test per group.
It also checks, on every image, that the regional 8-contact graph read from the HIG matrices
(Definition 4.14) equals the RAG with corner contacts and OUT built from the pixels.

usage: python3 sec52_rag_classes.py OUT_CSV
Reference output: results/sec52/table1.csv. About 15 s.
"""
import sys, itertools, collections
import numpy as np, networkx as nx
import pandas as pd
from rectgen import generate_rectangular_segmentation
from hig_construct import construct

BASE_SEED = 20260927; N = 100
imgs = []
for r in range(1, 9):
    for j in range(N):
        imgs.append((f"r={r},n=0", generate_rectangular_segmentation(r, 0, BASE_SEED + r*1000 + j)[0]))
for n in range(5):
    for j in range(N):
        imgs.append((f"r=5,n={n}", generate_rectangular_segmentation(5, n, BASE_SEED + n*1000 + j)[0]))

def graphs(A):
    A = A.astype(np.int32)
    ids, edges, geom, cuts, carriers, holes, m01, m12 = construct(A)
    out = m12.shape[1] - 1
    lab = {r: int(A[tuple(np.argwhere(ids == r)[0])]) for r in range(out)}
    node = lambda G: G.add_nodes_from((r, {"a": f"L{lab[r]}"}) for r in range(out))
    rag, ragO, rag8, rag8O = nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()
    for G in (rag, ragO, rag8, rag8O): node(G)
    ragO.add_node(out, a="OUT"); rag8O.add_node(out, a="OUT")
    pairmult = collections.Counter()
    for g in geom:
        a, b = g["pair"]; ragO.add_edge(a, b)
        if out not in (a, b): rag.add_edge(a, b); pairmult[(a, b)] += 1
    # 8-adjacency between pixels of different regions (any two pixels of a 2x2 block)
    h, w = ids.shape
    for y in range(h):
        for x in range(w):
            for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and ids[y, x] != ids[yy, xx]:
                    rag8.add_edge(int(ids[y, x]), int(ids[yy, xx])); rag8O.add_edge(int(ids[y, x]), int(ids[yy, xx]))
    for r in set(ids[0]) | set(ids[-1]) | set(ids[:, 0]) | set(ids[:, -1]):
        rag8O.add_edge(int(r), out)
    # the same graph from the HIG matrices (Definition 4.14): sanity check
    ng = len(geom)
    B12 = (m12[:ng] > 0).astype(int); B01 = (m01[:, :ng] > 0).astype(int)
    Jm = ((B01 @ B12) > 0).astype(int)
    Adj = ((B12.T @ B12 + Jm.T @ Jm) > 0).tocoo()
    g8 = {(min(i, j), max(i, j)) for i, j in zip(Adj.row, Adj.col) if i != j}
    assert g8 == {(min(a, b), max(a, b)) for a, b in rag8O.edges()}, "G8 from HIG differs from pixel RAG8+OUT"
    hig = nx.Graph()
    n0, n1 = m01.shape
    nj = n0 - sum(g["type"] == "enc0" for g in geom)
    for v in range(n0): hig.add_node(("v", v), a="junc" if v < nj else "v0enc")
    for k in range(n1): hig.add_node(("e", k), a=geom[k]["type"] if k < len(geom) else "cut")
    for r in range(out): hig.add_node(("R", r), a=f"L{lab[r]}")
    hig.add_node(("R", out), a="OUT")
    m01c = m01.tocoo(); m12c = m12.tocoo()
    for v, k, m in zip(m01c.row, m01c.col, m01c.data): hig.add_edge(("v", int(v)), ("e", int(k)), m=int(m))
    for k, r, m in zip(m12c.row, m12c.col, m12c.data): hig.add_edge(("e", int(k)), ("R", int(r)), m=int(m))
    hole_reg = tuple(sorted((lab[r], c) for r, c in collections.Counter(carriers).items()))
    mult_sig = tuple(sorted(((lab[a], lab[b]) if lab[a] <= lab[b] else (lab[b], lab[a]), m) for (a, b), m in pairmult.items()))
    return [rag, ragO, rag8, rag8O, hig], dict(holes=hole_reg, mult=mult_sig, J=nj)

nm = lambda x, y: x["a"] == y["a"]
em = lambda x, y: x.get("m") == y.get("m")
def classes(G, edge_attr=None):
    buckets = collections.defaultdict(list)
    for i, g in enumerate(G):
        buckets[nx.weisfeiler_lehman_graph_hash(g, node_attr="a", edge_attr=edge_attr, iterations=4)].append(i)
    cls = [None] * len(G); c = 0
    for idx in buckets.values():
        reps = []
        for i in idx:
            for rep, cid in reps:
                if nx.is_isomorphic(G[i], G[rep], node_match=nm, edge_match=(em if edge_attr else None)):
                    cls[i] = cid; break
            else:
                cls[i] = c; reps.append((i, c)); c += 1
    return cls, c

keys = {}; uniq = []
for i, (_, A) in enumerate(imgs):
    k = A.tobytes()
    if k not in keys: keys[k] = i; uniq.append(i)
print("images", len(imgs), "distinct", len(uniq))
GR = [[] for _ in range(5)]; META = []
for i in uniq:
    gs, meta = graphs(imgs[i][1]); META.append(meta)
    for t in range(5): GR[t].append(gs[t])
names = ["Simple RAG", "RAG with OUT", "RAG with corner contacts", "RAG with corner contacts and OUT", "HIG"]
ROWS = []
CL = []
for t in range(5):
    cl, n = classes(GR[t], "m" if t == 4 else None); CL.append(cl); print(names[t], "classes", n)
cH = CL[4]
for t in range(4):
    viol = sum(1 for i, j in itertools.combinations(range(len(uniq)), 2) if cH[i] == cH[j] and CL[t][i] != CL[t][j])
    groups = collections.defaultdict(list)
    for i, c in enumerate(CL[t]): groups[c].append(i)
    tot = sep = 0; cause = collections.Counter(); ex = []
    for g in groups.values():
        for i, j in itertools.combinations(g, 2):
            tot += 1
            if cH[i] != cH[j]:
                sep += 1; ex.append((i, j))
                if META[i]["holes"] != META[j]["holes"]: cause["holes"] += 1
                elif META[i]["mult"] != META[j]["mult"]: cause["mult"] += 1
                elif META[i]["J"] != META[j]["J"]: cause["junction count"] += 1
                else: cause["other junction/enclosing"] += 1
    print(f"{names[t]}: pairs with same graph {tot}, separated by HIG {sep} ({100*sep/max(tot,1):.1f}%), causes {dict(cause)}, same-HIG-different-graph {viol}")
    ROWS.append(dict(graph=names[t], classes=len(set(CL[t])), pairs_same_graph=tot, separated_by_hig=sep,
                     sep_holes=cause["holes"], sep_multiplicities=cause["mult"],
                     sep_junction_or_enclosing=cause["junction count"] + cause["other junction/enclosing"],
                     same_hig_different_graph=viol))

# pairs that share both the side-adjacency RAG with OUT and the RAG with corner contacts and OUT
both = collections.defaultdict(list)
for i in range(len(uniq)): both[(CL[1][i], CL[3][i])].append(i)
tot = sep = 0
for g in both.values():
    for i, j in itertools.combinations(g, 2):
        tot += 1; sep += cH[i] != cH[j]
print(f"Both RAGs with OUT: classes {len(both)}, pairs with the same two graphs {tot}, separated by HIG {sep}")
ROWS.append(dict(graph="Both RAGs with OUT", classes=len(both), pairs_same_graph=tot, separated_by_hig=sep))
ROWS.append(dict(graph="HIG", classes=len(set(cH))))
pd.DataFrame(ROWS).to_csv(sys.argv[1], index=False)
print("images", len(imgs), "distinct", len(uniq), "->", sys.argv[1])
