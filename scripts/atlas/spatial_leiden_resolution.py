"""Sweep Leiden resolution on the existing combined_clustered.h5ad neighbor
graph to find the value that reproduces the canonical 31-cluster spatial
partition. PCA/neighbors are already stored in the file from the previous
``combined_analysis.py cluster`` run — so this just varies ``sc.tl.leiden``
at a handful of resolutions and reports the cluster count.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/atlas/spatial_leiden_resolution.py

Prints a table of (resolution → n_clusters) and does not modify any file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import scanpy as sc

H5AD = Path("pipelines/combined_atlas/data/combined_clustered.h5ad")
RESOLUTIONS = [2.0, 2.25, 2.5, 2.65, 2.75, 2.85, 2.95, 3.0, 3.15, 3.3]
TARGET = 31


def main():
    print(f"Loading {H5AD}...", flush=True)
    ad = sc.read_h5ad(H5AD)
    print(f"  {ad.n_obs:,} cells", flush=True)

    if "neighbors" not in ad.uns and "connectivities" not in ad.obsp:
        print("  ERROR: no precomputed neighbor graph in this file.")
        sys.exit(1)

    rows = []
    for res in RESOLUTIONS:
        sc.tl.leiden(
            ad,
            resolution=res,
            flavor="igraph",
            n_iterations=2,
            directed=False,
            key_added=f"leiden_r{res}",
        )
        n = ad.obs[f"leiden_r{res}"].nunique()
        rows.append((res, n))
        print(f"  res={res:4.2f}  →  {n:3d} clusters", flush=True)

    best = min(rows, key=lambda r: abs(r[1] - TARGET))
    print(f"\nClosest to target {TARGET}: resolution={best[0]} → {best[1]} clusters")


if __name__ == "__main__":
    main()
