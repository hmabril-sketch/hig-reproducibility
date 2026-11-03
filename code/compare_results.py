"""Compares regenerated results with the reference results, file by file and row by row.

usage: python3 compare_results.py NEW_DIR [REF_DIR]      (REF_DIR defaults to ../results)
Every CSV present in both directories (same relative path) is compared on its common columns,
after aligning rows on their identifying columns; timing columns (t_*) are skipped, since they
depend on the machine.
"""
import os
import sys

import numpy as np
import pandas as pd

NEW = sys.argv[1]
REF = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
KEYS = ["tag", "file", "epoch", "slice", "image", "perturbation", "graph", "series", "r", "n", "seg", "sigma", "L", "side"]
bad = 0
for root, _, files in os.walk(NEW):
    for f in sorted(files):
        if not f.endswith(".csv"):
            continue
        rel = os.path.relpath(os.path.join(root, f), NEW)
        ref_path = os.path.join(REF, rel)
        if not os.path.exists(ref_path):
            print(f"  --  {rel}: no reference file"); continue
        a, b = pd.read_csv(os.path.join(NEW, rel)), pd.read_csv(ref_path)
        keys = [k for k in KEYS if k in a and k in b]
        cols = [c for c in a.columns if c in b.columns and c not in keys and not c.startswith("t_")]
        if keys:
            a, b = a.sort_values(keys).reset_index(drop=True), b.sort_values(keys).reset_index(drop=True)
        problems = []
        if len(a) != len(b):
            problems.append(f"{len(a)} rows against {len(b)}")
        else:
            for k in keys:
                if not (a[k].astype(str).values == b[k].astype(str).values).all():
                    problems.append(f"rows differ in {k}")
            for c in cols:
                x, y = a[c], b[c]
                num = pd.api.types.is_numeric_dtype
                if not (num(x) and num(y)) or pd.api.types.is_bool_dtype(x) or pd.api.types.is_bool_dtype(y):
                    same = (x.astype(str).str.lower() == y.astype(str).str.lower()) | (x.isna() & y.isna())
                else:
                    same = np.isclose(x.astype(float), y.astype(float), rtol=1e-9, atol=1e-12, equal_nan=True)
                if not np.all(same):
                    problems.append(f"{c} ({int((~np.asarray(same)).sum())} rows)")
        bad += bool(problems)
        print(f"  {'BAD' if problems else 'ok '} {rel}: {len(a)} rows, {len(cols)} columns" + (f" -> {', '.join(problems)}" if problems else ""))
print("all regenerated files agree with the reference" if not bad else f"{bad} files differ")
sys.exit(1 if bad else 0)
