"""Sec. 5.4 and Table 4: query cost of the holes of every region (E1), construction time against
the number of pixels (E2) and representation sizes on the ACDC segmentations (E3).

usage: python3 sec54_size_cost.py RUN_DIR OUT_DIR
RUN_DIR is the U-Net run (test_img.npy, test_gt.npy, test_pred_ep02/05/16.npy).
Reference outputs: results/sec54/e1_query.csv, e2_scaling.csv, e3_sizes_acdc.csv, measured on one
core of an Intel Xeon at 2.1 GHz (times depend on the machine). Less than 1 h.
"""
import os, sys, time
import numpy as np, pandas as pd
from scipy import ndimage as ndi
from skimage.transform import resize
from hig_fast import construct_fast
import queries as Q

RUN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
IMG = np.load(os.path.join(RUN, 'test_img.npy'), mmap_mode='r')
GT = np.load(os.path.join(RUN, 'test_gt.npy'), mmap_mode='r')
PRED = {ep: np.load(os.path.join(RUN, f'test_pred_ep{ep}.npy'), mmap_mode='r') for ep in ('02', '05', '16')}

def quantize(I, sigma, L):
    J = ndi.gaussian_filter(I, sigma) if sigma else I
    return np.digitize(J, np.quantile(J, np.linspace(0, 1, L + 1)[1:-1])).astype(np.int32)

def best(f, *a, rep=3):
    t = []
    for _ in range(rep):
        t0 = time.perf_counter(); r = f(*a); t.append(time.perf_counter() - t0)
    return r, min(t)

def sizes(d, A, h, w):
    ng = len(d['gtype']); nc = len(d['cut_comp'])
    side = d['gpair'][(d['gpair'] < d['nreg']).all(1)]
    rag_edges = len(np.unique(side[:, 0] * (d['nreg'] + 1) + side[:, 1]))
    return dict(pixels=h * w,
                complex_cells=(h + 1) * (w + 1) + h * (w + 1) + (h + 1) * w + h * w,
                complex_incid=2 * (h * (w + 1) + (h + 1) * w) + 4 * h * w,
                frontier_edges=len(d['frontier_edges']),
                hig_gen=d['m01'].shape[0] + d['m01'].shape[1] + d['m12'].shape[1],
                hig_nnz=d['m01'].nnz + d['m12'].nnz,
                hig_geom=ng, hig_cuts=nc,
                rag_nodes=d['nreg'], rag_edges=rag_edges,
                rag8_nodes=d['nreg'] + 1, rag8_edges=A.nnz // 2,
                tmap_darts=2 * ng)

# warm-up (numba compilation)
_d = construct_fast(np.zeros((4, 4), np.int32)); Q.rag8_lowlink(Q.rag8_build(_d['ids'], _d['nreg']), _d['nreg'])

# ---------------------------------------------------------------- E1: query cost vs number of regions
rng = np.random.default_rng(20261027)
sel = rng.choice(len(IMG), 60, replace=False)
rows = []
for i in sel:
    I = np.asarray(IMG[i], float)
    for sigma in (0, 1, 2):
        for L in (4, 8, 16, 32, 64):
            Sg = quantize(I, sigma, L); h, w = Sg.shape
            d, t_hb = best(construct_fast, Sg)
            (hh, gens), t_hq = best(Q.hig_query, d)
            A, t_rb = best(Q.rag8_build, d['ids'], d['nreg'])
            hl, t_ll = best(Q.rag8_lowlink, A, d['nreg'])
            r = dict(image=int(i), sigma=sigma, L=L, regions=d['nreg'], holes=int(hh.sum()),
                     t_hig_build=t_hb, t_hig_query=t_hq, t_rag8_build=t_rb, t_rag8_lowlink=t_ll,
                     t_rag8_naive=np.nan, t_pixel_naive=np.nan, agree=bool(np.array_equal(hh, hl)))
            if d['nreg'] <= 3000:
                hn, r['t_rag8_naive'] = best(Q.rag8_naive, A, d['nreg'], rep=1)
                hp, r['t_pixel_naive'] = best(Q.pixel_naive, d['ids'], d['nreg'], rep=1)
                r['agree'] = r['agree'] and bool(np.array_equal(hh, hn) and np.array_equal(hh, hp))
            r.update(sizes(d, A, h, w)); rows.append(r)
    print('E1 image', i, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'e1_query.csv'), index=False)

# ---------------------------------------------------------------- E2: construction time vs number of pixels
rows = []
for i in sel[:20]:
    I = np.asarray(IMG[i], float)
    for s in (128, 256, 512, 1024, 2048):
        J = resize(I, (s, s), order=1, anti_aliasing=False)
        Sg = quantize(J, s / 192.0, 8)
        d, t_hb = best(construct_fast, Sg, rep=2)
        (hh, gens), t_hq = best(Q.hig_query, d, rep=2)
        A, t_rb = best(Q.rag8_build, d['ids'], d['nreg'], rep=2)
        hl, t_ll = best(Q.rag8_lowlink, A, d['nreg'], rep=2)
        r = dict(image=int(i), side=s, regions=d['nreg'], holes=int(hh.sum()), t_hig_build=t_hb,
                 t_hig_query=t_hq, t_rag8_build=t_rb, t_rag8_lowlink=t_ll, agree=bool(np.array_equal(hh, hl)))
        r.update(sizes(d, A, s, s)); rows.append(r)
    print('E2 image', i, flush=True)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'e2_scaling.csv'), index=False)

# ---------------------------------------------------------------- E3: sizes on the ACDC segmentations
rows = []
for i in range(len(GT)):
    G = np.asarray(GT[i], np.int32)
    if not (G > 0).any():
        continue
    for name, Sg in [('reference', G)] + [(f'epoch {ep}', np.asarray(P[i], np.int32)) for ep, P in PRED.items()]:
        d = construct_fast(Sg); A = Q.rag8_build(d['ids'], d['nreg'])
        r = dict(slice=i, seg=name); r.update(sizes(d, A, *Sg.shape)); rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'e3_sizes_acdc.csv'), index=False)
print('done')
