"""Cell-type x region composition at the cell level, all 17 canonical types.

``cluster_vote`` assigns one label per spatial Leiden cluster, so a type that
never wins a cluster contributes no cells even when Tangram assigned it many,
and several canonical types win no cluster. The question here is also
compositional (e.g. what fraction of lamina cells are subperineurial glia),
and voting would label a cluster that is partly cortex glia as entirely
cortex glia.

This table therefore uses no vote: per-cell canonical-17 argmax against
hand-drawn region, so every type contributes in proportion to its assigned
cells.

A single per-cell argmax on a 101-gene panel is unreliable (it changes
across Tangram seeds), but independent errors should average out over a
composition of hundreds of thousands of cells. The script checks this by
recomputing the table on each available seed replicate and reporting how far
the composition moves.

Writes results/reframe/celltype_anatomy/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from bimp_st.cluster_labels import CANONICAL_ORDER, get_canonical  # noqa: E402
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.nine_region_common import (  # noqa: E402
    STATISTICAL_REGIONS,
    attach_region_flags,
    display,
    region_of,
)
from bimp_st.paths import CANONICAL_TANGRAM, SPATIAL_COMBINED  # noqa: E402

OUT = REPO / "results" / "reframe" / "celltype_anatomy"
TG = REPO / "results" / "tangram"
SEED_TAGS = ("s1", "s7", "s13", "s23", "s31")


def region_vector(a):
    reg = region_of(a.obs, STATISTICAL_REGIONS)
    one = np.isin(reg, STATISTICAL_REGIONS)
    return one, np.array([display(r) for r in reg[one]])


def table_for(fine, one, region):
    canon = np.array([get_canonical(x) for x in fine])
    t = pd.crosstab(
        pd.Series(canon[one], name="cell type"), pd.Series(region, name="region")
    )
    return t.reindex(
        index=[c for c in CANONICAL_ORDER if c in t.index],
        columns=sorted(t.columns),
        fill_value=0,
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading...", flush=True)
    a = ad.read_h5ad(SPATIAL_COMBINED)
    fine = read_obs(CANONICAL_TANGRAM)["tangram_cell_type"].astype(str).values
    attach_region_flags(a, regions=STATISTICAL_REGIONS)
    one, region = region_vector(a)

    tab = table_for(fine, one, region)
    tab.to_csv(OUT / "counts_percell_canonical17.csv")
    pd.set_option("display.width", 200)
    print(
        f"\n=== per-cell canonical-17 x region, {int(one.sum()):,} cells "
        f"in exactly one traced region ==="
    )
    print(tab.to_string())
    print(
        f"\ntypes contributing at least one cell: {(tab.sum(1) > 0).sum()} "
        f"of {len(CANONICAL_ORDER)}   (the cluster vote managed 9, and "
        f"only 7 above a 1% display floor)"
    )

    rown = tab.div(tab.sum(1), axis=0).fillna(0.0)
    print("\n=== what dominates each region (per-cell, top 3) ===")
    for r in tab.columns:
        s = (tab[r] / tab[r].sum()).sort_values(ascending=False).head(3)
        print(f"  {r:15s} " + ",  ".join(f"{k} {v:.1%}" for k, v in s.items()))

    # ---- does the composition survive the per-cell seed churn? ------------
    print("\n=== stability of the COMPOSITION across independent Tangram seeds ===")
    print("    per-cell labels agree only ~71% across seeds; the question is")
    print("    whether the aggregate moves, not whether cells do.", flush=True)
    frames = {}
    for tag in SEED_TAGS:
        p = TG / f"combined_clustered_seedens_{tag}_tangram.h5ad"
        if not p.exists():
            continue
        f2 = read_obs(p)["tangram_cell_type"].astype(str).values
        frames[tag] = table_for(f2, one, region)
        print(f"    {tag} done", flush=True)
    if frames:
        base = rown
        devs = []
        for tag, t2 in frames.items():
            r2 = t2.div(t2.sum(1), axis=0).fillna(0.0)
            common = base.index.intersection(r2.index)
            d = (base.loc[common] - r2.loc[common]).abs().values
            devs.append(d.max())
            print(
                f"    {tag}: largest shift in any type-x-region fraction {d.max():.4f}"
            )
        print(
            f"\n  worst-case movement of any cell in the table across "
            f"{len(frames)} independent runs: {max(devs):.4f}"
        )
        print(
            "  (compare with the ~29% of individual cells that change "
            "label between the same runs)"
        )
    else:
        print("    no seed replicates on disk; skipped")
    print(f"\nwrote {OUT}/counts_percell_canonical17.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
