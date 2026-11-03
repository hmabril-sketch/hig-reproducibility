"""Holes of every 4-connected region from its digital Euler number (Gray's bit-quads,
4-connectivity): holes(R) = 1 - E4(R), E4 = (Q1 - Q3 + 2 QD) / 4, independent of the HIG."""
import numpy as np

def holes_from_euler(ids, nreg):
    p = np.pad(ids, 1, constant_values=-1)
    a, b, c, d = p[:-1, :-1], p[:-1, 1:], p[1:, :-1], p[1:, 1:]
    Q1 = np.zeros(nreg, np.int64); Q3 = np.zeros(nreg, np.int64); QD = np.zeros(nreg, np.int64)
    W = [a, b, c, d]
    for k, x in enumerate(W):
        first = np.ones(x.shape, bool)
        for y in W[:k]:
            first &= x != y
        m = first & (x >= 0)
        cnt = sum((x == y).astype(np.int64) for y in W)
        diag = (cnt == 2) & (((k == 0) & (a == d)) | ((k == 1) & (b == c)))
        np.add.at(Q1, x[m & (cnt == 1)], 1)
        np.add.at(Q3, x[m & (cnt == 3)], 1)
        np.add.at(QD, x[m & diag], 1)
    E = (Q1 - Q3 + 2 * QD)
    assert (E % 4 == 0).all()
    return 1 - E // 4
