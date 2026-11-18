#!/usr/bin/env python3
"""Build the canonical per-cell cell-type lookup used by bento_differential.py.

The ``tangram_cell_type`` column inside ``results/bento/*/*/lp.parquet`` is not
a cell-type name: it holds the numeric spatial Leiden cluster index (18 levels,
'0'..'17') that ``results/tangram/combined_clustered_g2_R2_s42_tangram.h5ad``
carried under that name when the parquets were written, and the parquet's
``cell_type`` column (``KC-l / KC-ic / Glia-ens / N-oct / N-rare``) was derived
from it. Neither maps onto the canonical cell-type vocabulary.

This script re-derives the label from ``combined_clustered_g2_R2_s42_tangram.h5ad``
obs with ``cluster_vote`` (the expression-only spatial labelling used in the
paper; raw per-cell ``tangram_cell_type`` cannot resolve the KC subtypes from
the panel) and keys it by ``cell_id`` per brain, which is the key the LP
parquets join on (``cell_boundaries`` -> ``cell_id``).

Writes: results/bento/differential/lp_cell_labels.parquet
        (columns: brain, cell_id, cluster_vote)

Usage:
    PYTHONPATH=src .venv/bin/python pipelines/subcellular/build_lp_labels.py
"""

import sys

sys.path.insert(0, "src")
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import anndata as ad
import pandas as pd

from bimp_st.cluster_labels import get_pooled, hierarchical_vote

REPO = Path(__file__).resolve().parents[2]
TANGRAM_H5AD = REPO / "results/tangram/combined_clustered_g2_R2_s42_tangram.h5ad"
OUT = REPO / "results/bento/differential/lp_cell_labels.parquet"

FLAGS = ["in_lamina", "in_medulla", "in_lobula", "in_kc_ic", "in_kc_nc", "in_kc_oc"]


class _ObsShim:
    """hierarchical_vote() only touches .obs and .n_obs."""

    def __init__(self, obs):
        self.obs, self.n_obs = obs, len(obs)


def main():
    print(f"Reading {TANGRAM_H5AD}...", flush=True)
    a = ad.read_h5ad(TANGRAM_H5AD, backed="r")
    cols = ["cell_id", "leiden", "brain", "tangram_cell_type", *FLAGS]
    obs = a.obs[cols].copy()
    a.file.close()

    for c in ("cell_id", "leiden", "brain", "tangram_cell_type"):
        obs[c] = obs[c].astype(str)

    lv = sorted(obs["tangram_cell_type"].unique())
    print(
        f"  n_obs={len(obs):,}  tangram_cell_type levels ({len(lv)}): {lv}", flush=True
    )
    print(f"  leiden levels: {obs['leiden'].nunique()}", flush=True)

    shim = _ObsShim(obs)
    # The canonical vote rule (as caste_comparison_sccoda), pooled to the display
    # groups.
    tab = hierarchical_vote(
        shim, cluster_col="leiden", label_col="tangram_cell_type", dst="cluster_vote"
    )
    voted = tab["label"].map(lambda v: v if v == "Mixed" else get_pooled(v))
    obs["cluster_vote"] = (
        obs["cluster_vote"]
        .astype(str)
        .map(lambda v: v if v == "Mixed" else get_pooled(v))
    )
    print("\ncluster -> voted label:", flush=True)
    for cl, lab in voted.sort_index(key=lambda i: i.astype(int)).items():
        print(f"  {cl:>3s} -> {lab}", flush=True)

    print("\nper-cell vote counts:", flush=True)
    print(obs["cluster_vote"].value_counts(), flush=True)

    # Per brain, one row per cell_id (obs names are non-unique; the brain split
    # is load-bearing -- barcodes collide across the two runs).
    out = []
    for brain, g in obs.groupby("brain", observed=True):
        g = g[["cell_id", "cluster_vote"]].drop_duplicates(subset="cell_id").copy()
        g["brain"] = brain
        out.append(g)
        print(f"  {brain}: {len(g):,} unique cell_ids", flush=True)
    res = pd.concat(out, ignore_index=True)
    res["cluster_vote"] = res["cluster_vote"].astype(str)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_parquet(OUT, index=False)
    print(f"\nSaved {len(res):,} rows -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
