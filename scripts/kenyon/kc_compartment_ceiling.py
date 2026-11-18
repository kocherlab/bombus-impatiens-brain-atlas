#!/usr/bin/env python
"""In-sample Bayes ceiling for recovering a Kenyon cell's calyx compartment from
its atlas label, against the majority-compartment baseline.

The quantity is the in-sample Bayes ceiling for recovering a Kenyon cell's calyx
compartment (KC-IC / KC-NC / KC-OC) from its own per-cell canonical-17 atlas label
alone: the sum over atlas types of the largest compartment count, divided by the cells,
against the majority compartment's share. It is computed over the cells with exactly one
calyx compartment flag (331,717 of the 331,816 compartment cells; 99 sit in two). The
labels are the per-cell ``tangram_cell_type`` of the canonical run
(``bimp_st.paths.CANONICAL_TANGRAM``), mapped to the canonical 17 by ``add_canonical``,
not the cluster vote and not the Kenyon-restricted five-population transfer.

It is not the held-out accuracy for the Large-type snRNA subclusters, nor the
region prediction from Leiden clusters; those are different questions.

The crosstab is computed from the objects; a cached copy, when present, is read
for comparison only. Adds a held-out version (section-grouped 70/30 splits) to show
how little the in-sample ceiling overstates. Writes to
``bimp_st.paths.MB_OL_CALLS_DIR``.

    PYTHONPATH=src:scripts .venv/bin/python scripts/kenyon/kc_compartment_ceiling.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import anndata as ad
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from bimp_st.cluster_labels import add_canonical  # noqa: E402
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.paths import (  # noqa: E402
    CANONICAL_TANGRAM,
    MB_OL_CALLS_DIR,
    SPATIAL_COMBINED,
)

OUT = MB_OL_CALLS_DIR
FLAGS = ["in_kc_ic", "in_kc_nc", "in_kc_oc"]
NICE = {"in_kc_ic": "KC-IC", "in_kc_nc": "KC-NC", "in_kc_oc": "KC-OC"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(SPATIAL_COMBINED, backed="r")
    obs = a.obs[FLAGS + ["slice_name", "brain"]].copy()
    lab = read_obs(CANONICAL_TANGRAM)["tangram_cell_type"].astype(str)
    if len(lab) != len(obs):
        raise SystemExit(
            f"labels {len(lab):,} != cells {len(obs):,}; refusing to fall back to the "
            f"stale column"
        )
    obs["tangram_cell_type"] = lab.values
    shim = ad.AnnData(obs=obs)  # add_canonical wants an AnnData
    src = add_canonical(shim, src="tangram_cell_type")
    obs = shim.obs
    print("add_canonical source column:", src, flush=True)

    flags = obs[FLAGS].astype(bool).values
    keep = flags.sum(1) == 1
    sub = obs.loc[keep].copy()
    sub["compartment"] = [NICE[FLAGS[i]] for i in flags[keep].argmax(1)]
    n = len(sub)
    ct = sub["cell_type_canonical"].astype(str)
    tab = pd.crosstab(ct, sub["compartment"])
    majority = sub["compartment"].value_counts(normalize=True).max()
    ceiling = tab.max(axis=1).sum() / n
    rows = [
        dict(
            quantity="cells with exactly one calyx flag",
            value=n,
            note=f"of {int(flags.any(1).sum()):,} compartment cells",
        ),
        dict(
            quantity="majority baseline",
            value=majority,
            note=sub["compartment"].value_counts().to_dict(),
        ),
        dict(
            quantity="Bayes ceiling, own canonical-17 type -> compartment (in sample)",
            value=ceiling,
            note="",
        ),
    ]
    print(
        f"{n:,} cells; majority {majority:.4f}; ceiling {ceiling:.4f}; increment "
        f"{ceiling - majority:+.4f}",
        flush=True,
    )

    # held-out: majority compartment per type learned on training sections only
    y = sub["compartment"].values
    g_ = sub["slice_name"].astype(str).values
    for seed in (0, 1, 2):
        tr, te = next(
            GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(ct, y, g_)
        )
        maj = pd.crosstab(ct.values[tr], y[tr]).idxmax(axis=1)
        pred = (
            pd.Series(ct.values[te])
            .map(maj)
            .fillna(pd.Series(y[tr]).value_counts().idxmax())
            .values
        )
        base = (y[te] == pd.Series(y[tr]).value_counts().idxmax()).mean()
        rows.append(
            dict(
                quantity=f"held-out accuracy, section-grouped split seed {seed}",
                value=(pred == y[te]).mean(),
                note=f"majority-class baseline on the same test sections {base:.4f}",
            )
        )
        print(rows[-1], flush=True)

    cached = (
        REPO
        / "results/reframe/snrna_region_feasibility/celltype_x_compartment_crosstab.csv"
    )
    if cached.exists():
        c = pd.read_csv(cached, index_col=0)
        same = c.reindex(tab.index).fillna(0).astype(int).equals(
            tab.astype(int)
        ) and set(c.index) == set(tab.index)
        rows.append(
            dict(
                quantity="cached crosstab identical to recomputed",
                value=bool(same),
                note=(
                    f"cached ceiling {c.max(1).sum() / c.values.sum():.4f}, majority "
                    f"{c.sum(0).max() / c.values.sum():.4f}"
                ),
            )
        )
        print(rows[-1], flush=True)
    tab.to_csv(OUT / "celltype_x_compartment_crosstab_recomputed.csv")
    pd.DataFrame(rows).to_csv(OUT / "kc_compartment_ceiling.csv", index=False)


if __name__ == "__main__":
    main()
