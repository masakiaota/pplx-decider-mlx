"""Tile rule for the s4 kernel (W4A8 int8 x int4b).

data/tile-table.json: the fastest measured tile per (N, K) and row count on M5 for the 27B shapes. At run time the
nearest measured row count (log scale) for the same (N, K) is used; unmeasured (N, K) fall back to a fixed rule.
Integer accumulation: the tile never changes the values.
"""
import json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent
TABLE = {}
for key, tile_ in json.loads((HERE.parent / 'data' / 'tile-table.json').read_text()).items():
    nk, m = key.split('@'); n, k = nk.split('x'); TABLE[(int(n), int(k), int(m))] = (0.0, tuple(tile_))


def tile(n, k, m):
    ms = [mm for (nn, kk, mm) in TABLE if nn == n and kk == k]
    if not ms:
        return (128, 64, 8, 4) if (k > 8192 and m > 640) else (64, 64, 8, 4)
    best = min(ms, key=lambda mm: abs(math.log(mm) - math.log(max(m, 1))))
    return TABLE[(n, k, best)][1]
