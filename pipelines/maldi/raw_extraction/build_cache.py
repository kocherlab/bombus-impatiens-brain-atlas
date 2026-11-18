#!/usr/bin/env python
"""Parse the huge imzML ONCE into a compact offset table (.npz) so all later
operations are instant. Uses C-level regex findall (single pass)."""

import re
import sys
import time

import numpy as np

imzml, out = sys.argv[1], sys.argv[2]
t0 = time.time()
txt = open(imzml, "r", encoding="utf-8", errors="replace").read()
print(f"read {len(txt) / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)

offs = np.array(re.findall(r'external offset" value="(\d+)"', txt), dtype=np.int64)
lens = np.array(
    re.findall(r'external array length" value="(\d+)"', txt), dtype=np.int64
)
px = np.array(re.findall(r'position x" value="(\d+)"', txt), dtype=np.int32)
py = np.array(re.findall(r'position y" value="(\d+)"', txt), dtype=np.int32)
print(
    f"regex done in {time.time() - t0:.0f}s  offs={len(offs)} "
    f"lens={len(lens)} px={len(px)} py={len(py)}",
    flush=True,
)

n = len(px)
assert len(offs) == 2 * n and len(lens) == 2 * n and len(py) == n, (
    "field-count mismatch"
)
np.savez(
    out,
    mz_off=offs[0::2],
    int_off=offs[1::2],
    mz_len=lens[0::2],
    int_len=lens[1::2],
    px=px,
    py=py,
)
print(f"cached {n:,} pixels -> {out}  ({time.time() - t0:.0f}s total)", flush=True)
