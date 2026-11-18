"""Composition tables for fig:caste_comparison, at two levels, by caste.

Writes two CSVs that the composition panels can read without touching
expression:

1. ``sn_composition_by_caste.csv`` -- the snRNA-seq reference (n=3 per caste),
   nuclei per cell type per caste, with the optic-lobe nuclei count per caste
   alongside (``obs.cell_type`` normalised by ``_norm_ct``, all 29 types of
   ``CELL_TYPE_ORDER`` kept, zeros included).

2. ``atlas_vote_composition_by_caste.csv`` -- the spatial atlas, cells per
   cell-type vote label per brain (queen = caste queen, worker = caste worker),
   where the label is ``hierarchical_vote`` on the corrected combined Tangram run.
   The vote is the same call ``caste_comparison_sccoda.py`` makes
   (``dst="cluster_vote"``,
   default cap), run on a lightweight AnnData holding only ``leiden`` and
   ``tangram_cell_type``, read positionally.

Inputs (obs only, no expression; read with h5py + anndata.io.read_elem):
    bimp_st.paths.CANONICAL_TANGRAM  (atlas labels, leiden, brain)
    bimp_st.paths.SN_COMBINED        (snRNA reference: cell_type, caste, region)

Outputs (under REPO/results/figures/fig2/):
    sn_composition_by_caste.csv
    atlas_vote_composition_by_caste.csv

REPO is ``$BIMP_REPO_ROOT`` if set, else the repository containing this file.
Set BIMP_REPO_ROOT to another checkout to write into its results directory.

Run order: after ``caste_comparison_sccoda.py`` (needs its tangram vote definition,
which
this script reproduces, not reads), before ``fig_caste_comparison.py``. Independent of
the other caste-comparison steps otherwise.

Run:
    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src:scripts \\
        .venv/bin/python scripts/regions/caste_comparison_tables.py
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import pandas as pd

from bimp_st.cluster_labels import (
    CELL_TYPE_ORDER,
    MIXED_DISPLAY,
    _norm_ct,
    get_pooled,
    hierarchical_vote,
)
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.paths import CANONICAL_TANGRAM, SN_COMBINED
from bimp_st.paths import REPO as _DEFAULT_REPO  # noqa: E402

REPO = Path(os.environ.get("BIMP_REPO_ROOT", _DEFAULT_REPO))
OUT = REPO / "results/figures/fig2"

# Expected totals and the Mixed/Unannotated share of the canonical vote.
ATLAS_TOTAL = {"queen": 686_105, "worker": 614_087}
REF_TOTAL = 73_463
MIXED_FRAC = {"queen": 0.0984, "worker": 0.0859}
MIXED_TOL = 0.0005
CASTES = ("queen", "worker")


def _read_obs(path: Path) -> pd.DataFrame:
    return read_obs(path)


def sn_composition(sn_obs: pd.DataFrame) -> pd.DataFrame:
    ct = sn_obs["cell_type"].astype(str).map(_norm_ct).values
    caste = sn_obs["caste"].astype(str).str.lower().values
    opt = sn_obs["region"].astype(str).values == "OpticLobe"

    df = pd.DataFrame({"cell_type": ct, "caste": caste, "opt": opt})
    caste_total = df.groupby("caste").size()
    caste_opt = df[df["opt"]].groupby("caste").size()
    counts = df.groupby(["cell_type", "caste"]).size()

    rows = []
    for c in CASTES:
        for t in CELL_TYPE_ORDER:
            n = int(counts.get((t, c), 0))
            tot = int(caste_total.get(c, 0))
            rows.append(
                {
                    "cell_type": t,
                    "caste": c,
                    "n_nuclei": n,
                    "frac": n / tot if tot else 0.0,
                    "n_caste_total": tot,
                    "n_caste_opticlobe": int(caste_opt.get(c, 0)),
                }
            )
    out = pd.DataFrame(rows)

    # Every reference label must land in the 29-type vocabulary.
    unmapped = set(ct) - set(CELL_TYPE_ORDER)
    assert not unmapped, f"reference labels outside CELL_TYPE_ORDER: {unmapped}"
    tot_all = int(out.groupby("caste")["n_nuclei"].sum().sum())
    assert tot_all == REF_TOTAL, f"reference total {tot_all} != {REF_TOTAL}"
    return out


def atlas_vote_composition(tg_obs: pd.DataFrame) -> pd.DataFrame:
    # Positional only: a lightweight AnnData, index replaced by row number.
    light = pd.DataFrame(
        {
            "leiden": tg_obs["leiden"].astype(str).values,
            "tangram_cell_type": tg_obs["tangram_cell_type"].astype(str).values,
            "brain": tg_obs["brain"].astype(str).values,
        }
    )
    light.index = pd.Index(light.index.astype(str))
    adata = ad.AnnData(obs=light)
    hierarchical_vote(
        adata, cluster_col="leiden", label_col="tangram_cell_type", dst="cluster_vote"
    )
    light["cluster_vote"] = adata.obs["cluster_vote"].astype(str).values

    rows = []
    for b in CASTES:
        sub = light[light["brain"] == b]
        n_tot = len(sub)
        assert n_tot == ATLAS_TOTAL[b], f"atlas {b} total {n_tot} != {ATLAS_TOTAL[b]}"
        vc = sub["cluster_vote"].value_counts()
        for lab, n in vc.items():
            rows.append(
                {
                    "cluster": lab,
                    "brain": b,
                    "n_cells": int(n),
                    "frac": n / n_tot,
                    "n_brain_total": n_tot,
                }
            )
        mixed = float(vc.get(MIXED_DISPLAY, vc.get("Mixed", 0))) / n_tot
        assert abs(mixed - MIXED_FRAC[b]) <= MIXED_TOL, (
            f"{b} Mixed/Unannotated fraction {mixed:.4f} "
            f"not within {MIXED_TOL} of {MIXED_FRAC[b]}"
        )
    return pd.DataFrame(rows).sort_values(["brain", "n_cells"], ascending=[True, False])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"REPO={REPO}\nOUT={OUT}", flush=True)

    print("Reading snRNA reference obs ...", flush=True)
    sn = sn_composition(_read_obs(SN_COMBINED))
    p1 = OUT / "sn_composition_by_caste.csv"
    sn.to_csv(p1, index=False)
    print(f"wrote {p1} ({len(sn)} rows)", flush=True)

    print("Reading atlas obs ...", flush=True)
    atlas = atlas_vote_composition(_read_obs(CANONICAL_TANGRAM))
    p2 = OUT / "atlas_vote_composition_by_caste.csv"
    atlas.to_csv(p2, index=False)
    print(f"wrote {p2} ({len(atlas)} rows)", flush=True)

    for c in CASTES:
        sub = atlas[atlas["brain"] == c].head(6)
        print(f"Top-6 vote fractions, {c}:", flush=True)
        for _, r in sub.iterrows():
            print(f"  {r['cluster']:<32s} {r['frac']:.4f}", flush=True)

    ref_kc = sn[sn["cell_type"].map(get_pooled) == "Kenyon cells"]
    for c in CASTES:
        f = ref_kc.loc[ref_kc["caste"] == c, "frac"].sum()
        print(f"Reference pooled Kenyon fraction, {c}: {f:.4f}", flush=True)


if __name__ == "__main__":
    main()
