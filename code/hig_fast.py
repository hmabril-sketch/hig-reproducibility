"""Fast (numba) intrinsic construction of the HIG, equivalent to hig_construct.construct.

Same conventions:
  * regions = 4-connected same-label components, numbered in raster order of their first
    pixel; OUT receives the id nreg;
  * frontier edges = grid edges whose two incident compactified 2-cells lie in different regions;
  * junctions = frontier vertices of degree >= 3;
  * geometric 1-nodes = continuation classes (edges joined through degree-2 vertices);
  * cut nodes = frame-free connected components of the frontier graph, carrier = region of
    the pixel above the uppermost, then leftmost, horizontal edge of the component.

Edge numbering: vertical edge (y, x), 0 <= y < h, 0 <= x <= w, separates pixels (y, x-1) and (y, x):
index y*(w+1)+x.  Horizontal edge (y, x), 0 <= y <= h, 0 <= x < w, separates (y-1, x) and (y, x):
index NV + y*w + x, with NV = h*(w+1).  Vertex (y, x), 0 <= y <= h, 0 <= x <= w: index y*(w+1)+x.
"""
import numpy as np
from numba import njit
from scipy import sparse


@njit(cache=True)
def _regions(img):
    h, w = img.shape
    ids = np.full((h, w), -1, np.int32)
    qy = np.empty(h * w, np.int32); qx = np.empty(h * w, np.int32)
    nr = 0
    for y in range(h):
        for x in range(w):
            if ids[y, x] != -1:
                continue
            lab = img[y, x]; ids[y, x] = nr
            head = 0; tail = 0
            qy[tail] = y; qx[tail] = x; tail += 1
            while head < tail:
                cy = qy[head]; cx = qx[head]; head += 1
                for k in range(4):
                    ny = cy + (-1 if k == 0 else (1 if k == 1 else 0))
                    nx = cx + (-1 if k == 2 else (1 if k == 3 else 0))
                    if 0 <= ny < h and 0 <= nx < w and ids[ny, nx] == -1 and img[ny, nx] == lab:
                        ids[ny, nx] = nr; qy[tail] = ny; qx[tail] = nx; tail += 1
            nr += 1
    return ids, nr


@njit(cache=True)
def _find(p, a):
    r = a
    while p[r] != r:
        r = p[r]
    while p[a] != r:
        n = p[a]; p[a] = r; a = n
    return r


@njit(cache=True)
def _union(p, a, b):
    ra = _find(p, a); rb = _find(p, b)
    if ra != rb:
        if ra < rb:
            p[rb] = ra
        else:
            p[ra] = rb


@njit(cache=True)
def _frontier(ids, nr):
    h, w = ids.shape
    NV = h * (w + 1); NE = NV + (h + 1) * w
    out = nr
    ea = np.full(NE, -1, np.int32); eb = np.full(NE, -1, np.int32)   # region pair (sorted)
    isf = np.zeros(NE, np.bool_); frame = np.zeros(NE, np.bool_)
    for y in range(h):
        for x in range(w + 1):
            e = y * (w + 1) + x
            a = ids[y, x - 1] if x > 0 else out
            b = ids[y, x] if x < w else out
            if a != b:
                isf[e] = True; ea[e] = min(a, b); eb[e] = max(a, b); frame[e] = (x == 0 or x == w)
    for y in range(h + 1):
        for x in range(w):
            e = NV + y * w + x
            a = ids[y - 1, x] if y > 0 else out
            b = ids[y, x] if y < h else out
            if a != b:
                isf[e] = True; ea[e] = min(a, b); eb[e] = max(a, b); frame[e] = (y == 0 or y == h)
    return isf, ea, eb, frame


@njit(cache=True)
def _vertex_edges(y, x, h, w, buf):
    """Frontier-graph edges incident with vertex (y, x) (all grid edges, checked by caller)."""
    NV = h * (w + 1); n = 0
    if y - 1 >= 0:
        buf[n] = (y - 1) * (w + 1) + x; n += 1          # vertical edge above
    if y < h:
        buf[n] = y * (w + 1) + x; n += 1                # vertical edge below
    if x - 1 >= 0:
        buf[n] = NV + y * w + (x - 1); n += 1           # horizontal edge to the left
    if x < w:
        buf[n] = NV + y * w + x; n += 1                 # horizontal edge to the right
    return n


@njit(cache=True)
def _classes(isf, h, w):
    NV = h * (w + 1); NE = NV + (h + 1) * w; NVert = (h + 1) * (w + 1)
    pc = np.arange(NE).astype(np.int32)   # continuation classes
    pk = np.arange(NE).astype(np.int32)   # frontier components
    deg = np.zeros(NVert, np.int32)
    buf = np.empty(4, np.int64); fe = np.empty(4, np.int64)
    for y in range(h + 1):
        for x in range(w + 1):
            n = _vertex_edges(y, x, h, w, buf); m = 0
            for i in range(n):
                if isf[buf[i]]:
                    fe[m] = buf[i]; m += 1
            v = y * (w + 1) + x
            deg[v] = m
            for i in range(1, m):
                _union(pk, fe[0], fe[i])
            if m == 2:
                _union(pc, fe[0], fe[1])
    return pc, pk, deg


def construct_fast(image):
    image = np.ascontiguousarray(image)
    ids, nr = _regions(image)
    h, w = ids.shape
    NV = h * (w + 1)
    isf, ea, eb, frame = _frontier(ids, nr)
    pc, pk, deg = _classes(isf, h, w)
    fe = np.nonzero(isf)[0]
    # roots (compress)
    croot = _roots(pc, fe); kroot = _roots(pk, fe)
    # endpoints of every frontier edge
    vert = fe < NV
    y = np.where(vert, fe // (w + 1), (fe - NV) // w)
    x = np.where(vert, fe % (w + 1), (fe - NV) % w)
    v1 = y * (w + 1) + x
    v2 = np.where(vert, (y + 1) * (w + 1) + x, y * (w + 1) + x + 1)
    junc = deg >= 3
    jlist = np.nonzero(junc)[0]
    jindex = np.full(deg.shape[0], -1, np.int64); jindex[jlist] = np.arange(len(jlist))
    # continuation classes -> geometric nodes
    cls, cinv = np.unique(croot, return_inverse=True)
    ng = len(cls)
    ep_cnt = np.zeros(ng, np.int64)
    rows01, cols01 = [], []
    for vv in (v1, v2):
        m = junc[vv]
        np.add.at(ep_cnt, cinv[m], 1)
        rows01.append(jindex[vv[m]]); cols01.append(cinv[m])
    rows01 = np.concatenate(rows01); cols01 = np.concatenate(cols01)
    # type: 0 endpoints -> enc0; 2 endpoints at the same junction -> enc1; else seg
    first_j = np.full(ng, -1, np.int64); second_j = np.full(ng, -1, np.int64)
    order = np.argsort(cols01, kind="stable")
    c_sorted = cols01[order]; r_sorted = rows01[order]
    starts = np.searchsorted(c_sorted, np.arange(ng))
    has = ep_cnt > 0
    first_j[has] = r_sorted[starts[has]]
    second_j[has] = r_sorted[starts[has] + 1]
    if np.any(ep_cnt[has] != 2):
        raise ValueError("continuation class with endpoint count != 0, 2")
    gtype = np.where(~has, 0, np.where(first_j == second_j, 1, 2))   # 0 enc0, 1 enc1, 2 seg
    rep = np.zeros(ng, np.int64); rep[cinv] = fe          # any edge of the class
    ga, gb = ea[rep], eb[rep]
    # frontier components -> cuts
    comps, kinv = np.unique(kroot, return_inverse=True)
    nk = len(comps)
    kframe = np.zeros(nk, bool); np.logical_or.at(kframe, kinv, frame[fe])
    cut_ids = np.nonzero(~kframe)[0]
    # carrier: uppermost, then leftmost horizontal edge (smallest horizontal index) of each cut
    horiz = ~vert
    best = np.full(nk, np.iinfo(np.int64).max, np.int64)
    np.minimum.at(best, kinv[horiz], fe[horiz])
    hb = best[cut_ids] - NV
    carriers = ids[hb // w - 1, hb % w].astype(np.int64)
    # matrices
    nenc0 = int(np.sum(gtype == 0))
    n0 = len(jlist) + nenc0
    n1 = ng + len(cut_ids)
    n2 = nr + 1
    virt = np.full(ng, -1, np.int64); virt[gtype == 0] = len(jlist) + np.arange(nenc0)
    r01 = np.concatenate([rows01, virt[gtype == 0]])
    c01 = np.concatenate([cols01, np.nonzero(gtype == 0)[0]])
    d01 = np.concatenate([np.ones(len(rows01), np.int32), np.full(nenc0, 2, np.int32)])
    m01 = sparse.csr_matrix((d01, (r01, c01)), shape=(n0, n1), dtype=np.int32)   # duplicates summed
    r12 = np.concatenate([np.arange(ng), np.arange(ng), ng + np.arange(len(cut_ids))])
    c12 = np.concatenate([ga, gb, carriers])
    d12 = np.concatenate([np.ones(2 * ng, np.int32), np.full(len(cut_ids), 2, np.int32)])
    m12 = sparse.csr_matrix((d12, (r12, c12)), shape=(n1, n2), dtype=np.int32)
    return dict(ids=ids, nreg=nr, gtype=gtype, gpair=np.stack([ga, gb], 1),
                edge_class=cinv, edge_comp=kinv, frontier_edges=fe, comp_is_cut=~kframe,
                cut_comp=cut_ids, carriers=carriers, m01=m01, m12=m12,
                class_comp=_class_comp(cinv, kinv, ng))


@njit(cache=True)
def _roots(p, fe):
    r = np.empty(len(fe), np.int64)
    for i in range(len(fe)):
        r[i] = _find(p, fe[i])
    return r


def _class_comp(cinv, kinv, ng):
    cc = np.zeros(ng, np.int64); cc[cinv] = kinv
    return cc
