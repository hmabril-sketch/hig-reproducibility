import os, math, tempfile, zipfile
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


# ============================================================
# Global configuration
# ============================================================
SIZE = 10
OUT = -1
COLOR_NAMES = ["red", "blue", "green", "yellow", "purple"]
COLOR_RGB = [
    (231, 76, 60),
    (52, 152, 219),
    (46, 204, 113),
    (241, 196, 15),
    (155, 89, 182),
]
DIR4 = [(-1,0),(1,0),(0,-1),(0,1)]
DIR8 = [(dr,dc) for dr in (-1,0,1) for dc in (-1,0,1) if (dr,dc)!=(0,0)]

# Cubical 10x10 model sizes used for the compression ratios.
K0 = (SIZE + 1) * (SIZE + 1)   # 121
K1 = SIZE * (SIZE + 1) + (SIZE + 1) * SIZE  # 220
K2 = SIZE * SIZE               # 100
CUBICAL_CELLS = K0 + K1 + K2   # 441
# Support size of M01 plus compactified M12: 2 incidences per grid edge in each level.
CUBICAL_INCIDENCE_SUPPORT = 4 * K1  # 880

# ============================================================
# Structured rectangle generator
# ============================================================

def even_parts(total, parts):
    q, rem = divmod(total, parts)
    return [q + (i < rem) for i in range(parts)]


def layout_tiles(k, size=SIZE):
    """Candidate near-grid layouts for k root rectangles."""
    layouts = []
    for rows in range(1, k + 1):
        cols = math.ceil(k / rows)
        if rows > size or cols > size:
            continue
        hs = even_parts(size, rows)
        ws = even_parts(size, cols)
        tiles = []
        idx = 0
        y = 0
        for i, h in enumerate(hs):
            x = 0
            for j, w in enumerate(ws):
                if idx < k:
                    tiles.append((y, x, h, w, i, j))
                    idx += 1
                x += w
            y += h
        # A strict child needs a one-pixel margin on every side.
        capacities = [max(0, (min(h, w) - 1) // 2) for y,x,h,w,i,j in tiles]
        layouts.append((sum(capacities), rows, cols, tiles, capacities))
    return layouts


def max_nested_capacity(root_count, size=SIZE):
    if root_count < 1:
        return -1
    layouts = layout_tiles(root_count, size)
    return max((x[0] for x in layouts), default=-1)


def feasible_pair(r, n, size=SIZE):
    if r < 1 or n < 0 or n >= r:
        return False
    return n <= max_nested_capacity(r - n, size)


def choose_layout_and_depths(r, n, rng, size=SIZE):
    roots = r - n
    feasible = []
    for item in layout_tiles(roots, size):
        capsum, rows, cols, tiles, caps = item
        if capsum >= n:
            feasible.append(item)
    if not feasible:
        raise ValueError(
            f"(r={r}, n={n}) is not feasible on a {size}x{size} grid with strict one-pixel nesting margins."
        )
    # Prefer layouts with good nesting capacity but keep some variation.
    feasible.sort(key=lambda z: z[0], reverse=True)
    best = feasible[0][0]
    candidates = [z for z in feasible if z[0] >= max(n, best - 1)]
    capsum, rows, cols, tiles, caps = candidates[int(rng.integers(len(candidates)))]

    depths = [0] * roots
    remaining = n
    while remaining:
        available = [i for i, c in enumerate(caps) if depths[i] < c]
        if not available:
            raise RuntimeError("Internal error allocating nesting depths.")
        i = int(rng.choice(available))
        depths[i] += 1
        remaining -= 1
    return tiles, depths, (rows, cols)


def random_rect_within(tile, min_side, rng):
    y, x, h, w, _, _ = tile
    if min(h, w) < min_side:
        raise ValueError("Tile too small for requested nesting depth.")
    rh = int(rng.integers(min_side, h + 1))
    rw = int(rng.integers(min_side, w + 1))
    y0 = int(rng.integers(y, y + h - rh + 1))
    x0 = int(rng.integers(x, x + w - rw + 1))
    return (y0, x0, rh, rw)


def nested_child_rect(parent, remaining_depth_after_child, rng):
    y, x, h, w = parent
    min_side = 2 * remaining_depth_after_child + 1
    max_h = h - 2
    max_w = w - 2
    if max_h < min_side or max_w < min_side:
        raise ValueError("Parent rectangle is too small for the requested nested chain.")
    rh = int(rng.integers(min_side, max_h + 1))
    rw = int(rng.integers(min_side, max_w + 1))
    y0 = int(rng.integers(y + 1, y + h - rh)) if (y + h - rh) > (y + 1) else y + 1
    x0 = int(rng.integers(x + 1, x + w - rw)) if (x + w - rw) > (x + 1) else x + 1
    return (y0, x0, rh, rw)


def generate_rectangular_segmentation(r, n, seed, size=SIZE):
    """Generate one 10x10 five-color image with exactly r drawn rectangles, n nested."""
    if not feasible_pair(r, n, size):
        raise ValueError(
            f"Invalid/infeasible parameters r={r}, n={n}. "
            f"Require 0 <= n < r and sufficient space for strict nesting."
        )

    rng = np.random.default_rng(int(seed))
    A = np.zeros((size, size), dtype=np.int8)  # background = red / label 0
    tiles, depths, layout = choose_layout_and_depths(r, n, rng, size)
    primitives = []

    # Root colors: a randomized 4-color pattern that prevents neighboring tile roots
    # from having the same color when they touch.
    root_palette = list(rng.permutation(np.arange(1, len(COLOR_RGB))))

    for root_id, (tile, depth) in enumerate(zip(tiles, depths)):
        min_side = 2 * depth + 1
        root = random_rect_within(tile, min_side, rng)
        tile_row, tile_col = tile[4], tile[5]
        color = int(root_palette[(tile_row + tile_col) % len(root_palette)])

        y, x, h, w = root
        A[y:y+h, x:x+w] = color
        primitives.append({
            "primitive": len(primitives), "root": root_id, "depth": 0,
            "parent": None, "y": y, "x": x, "h": h, "w": w, "color": color
        })
        parent_idx = len(primitives) - 1
        parent = root
        parent_color = color

        for level in range(1, depth + 1):
            remaining = depth - level
            child = nested_child_rect(parent, remaining, rng)
            choices = [c for c in range(len(COLOR_RGB)) if c != parent_color]
            child_color = int(rng.choice(choices))
            yy, xx, hh, ww = child
            A[yy:yy+hh, xx:xx+ww] = child_color
            primitives.append({
                "primitive": len(primitives), "root": root_id, "depth": level,
                "parent": parent_idx, "y": yy, "x": xx, "h": hh, "w": ww,
                "color": child_color
            })
            parent_idx = len(primitives) - 1
            parent = child
            parent_color = child_color

    assert len(primitives) == r
    assert sum(p["parent"] is not None for p in primitives) == n
    return A, primitives, depths, layout

# ============================================================
# Intrinsic HIG implementation
# ============================================================
