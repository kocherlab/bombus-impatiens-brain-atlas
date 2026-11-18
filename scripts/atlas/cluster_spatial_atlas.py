"""Rebuild ``combined_clustered.h5ad`` and pick the Leiden resolution that
yields the canonical **31 spatial clusters** on the current neighbor graph.

Use this instead of a plain ``combined_analysis.py cluster`` call whenever the
upstream ``combined.h5ad`` is regenerated: a fixed resolution (the default is
``1.0``) does not reliably land on 31 clusters. HVG / PCA / neighbors are
computed **once**; Leiden is then swept across a grid of resolutions, the
closest match to the target cluster count is kept, UMAP is computed, and
the result is written to ``combined_clustered.h5ad``.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/atlas/cluster_spatial_atlas.py
"""

from __future__ import annotations

import os
from pathlib import Path

import scanpy as sc

DATA_DIR = Path("pipelines/combined_atlas/data")
IN_H5AD = DATA_DIR / "combined.h5ad"
OUT_H5AD = DATA_DIR / "combined_clustered.h5ad"

N_HVG = 2000
N_PCS = 50
N_NEIGHBORS = 15
MIN_DIST = 0.1

# Sweep grid — widen if you find the answer outside this range.
RESOLUTIONS = [1.5, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5]
TARGET_N = 31


def main():
    n_cpus = int(os.environ.get("NUMBA_NUM_THREADS", os.cpu_count() or 8))
    sc.settings.n_jobs = n_cpus
    print(f"n_jobs={n_cpus}", flush=True)

    print(f"Loading {IN_H5AD}...", flush=True)
    ad = sc.read_h5ad(IN_H5AD)
    print(f"  {ad.n_obs:,} cells x {ad.n_vars} genes", flush=True)

    if "log1p" not in ad.uns:
        print("Normalizing + log1p...", flush=True)
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)

    print(f"HVGs (n_top_genes={N_HVG}, batch_key='brain')...", flush=True)
    sc.pp.highly_variable_genes(ad, n_top_genes=N_HVG, batch_key="brain")

    print(f"PCA (n_comps={N_PCS})...", flush=True)
    sc.tl.pca(ad, n_comps=N_PCS)

    print(f"Neighbors (n_neighbors={N_NEIGHBORS}, n_pcs={N_PCS})...", flush=True)
    sc.pp.neighbors(ad, n_neighbors=N_NEIGHBORS, n_pcs=N_PCS)

    # ── Sweep Leiden resolution ───────────────────────────────────────────
    print("\nSweeping Leiden resolution:", flush=True)
    rows = []
    for res in RESOLUTIONS:
        key = f"leiden_r{res}"
        sc.tl.leiden(
            ad,
            resolution=res,
            flavor="igraph",
            n_iterations=2,
            directed=False,
            key_added=key,
        )
        n = int(ad.obs[key].nunique())
        rows.append((res, n))
        print(f"  res={res:4.2f}  →  {n:3d} clusters", flush=True)

    # Pick closest to target; tiebreak by picking the lower resolution.
    rows_sorted = sorted(rows, key=lambda r: (abs(r[1] - TARGET_N), r[0]))
    best_res, best_n = rows_sorted[0]
    print(
        f"\nPicked resolution={best_res} → {best_n} clusters (target was {TARGET_N})",
        flush=True,
    )

    ad.obs["leiden"] = ad.obs[f"leiden_r{best_res}"].copy()
    ad.uns["leiden_sweep"] = {
        "target_n": TARGET_N,
        "picked_resolution": float(best_res),
        "picked_n_clusters": int(best_n),
        "grid": [(float(r), int(n)) for r, n in rows],
    }
    # Drop the per-resolution scratch columns so the output stays clean.
    for res in RESOLUTIONS:
        del ad.obs[f"leiden_r{res}"]

    print(f"UMAP (min_dist={MIN_DIST})...", flush=True)
    sc.tl.umap(ad, min_dist=MIN_DIST)

    print(f"Saving {OUT_H5AD}...", flush=True)
    ad.write_h5ad(OUT_H5AD)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
