#!/usr/bin/env python3
"""Per-region cell counts and cell-type composition, queen and worker.

One producer for every per-region count or share of the caste-composition
analysis: panel C's bars, the counts per region, the regional-against-caste
contrast, and the pooled Kenyon/optic shares. Regions come from
``bimp_st.composition.atlas_regions`` (fresh nine-region flags; a cell counts
only if it lies in exactly one of the seven statistical regions), labels
from the canonical expression-only ``hierarchical_vote``
(``caste_comparison_sccoda._display_ct_series``).

Outputs (results/region_composition/):
    composition_by_region.csv          display labels (panel C's six classes)
    composition_by_region_pooled.csv   the same cells, labels pooled to the
                                       four groups (get_pooled)
    region_cell_counts.csv             cells per brain x region (incl. outside,
                                       overlap and untraced)
    composition_contrast.json          the regional-against-caste contrast

Descriptive throughout (one brain per caste); no test is run.

    PYTHONPATH=src:scripts .venv/bin/python scripts/regions/region_composition.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from bimp_st.cluster_labels import get_pooled
from bimp_st.composition import (
    COMPOSITION_REGIONS,
    PANEL_REGIONS,
    atlas_regions,
    composition_contrast,
    composition_table,
)
from bimp_st.paths import CANONICAL_TANGRAM

REPO = Path(__file__).resolve().parents[2]
OUT = Path(
    os.environ.get("BIMP_REGION_COMPOSITION_OUT", REPO / "results/region_composition")
)


def load_obs():
    """Obs of the canonical Tangram run with fresh region flags and the vote."""
    from regions.caste_comparison_sccoda import _display_ct_series

    a = ad.read_h5ad(CANONICAL_TANGRAM, backed="r")
    obs = a.obs[["slice_name", "cell_id", "brain", "leiden", "tangram_cell_type"]]
    obs = obs.copy().reset_index(drop=True)
    obs.index = obs.index.astype(str)
    b = ad.AnnData(obs=obs)
    region = atlas_regions(b)
    label = _display_ct_series(b).to_numpy().astype(str)
    return b.obs, region, label


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obs, region, label = load_obs()
    brain = obs["brain"].astype(str).to_numpy()

    disp = composition_table(brain, region, label, regions=COMPOSITION_REGIONS)
    disp.to_csv(OUT / "composition_by_region.csv", index=False)
    pooled_label = np.array([get_pooled(v) for v in label], dtype=object)
    pooled = composition_table(brain, region, pooled_label, regions=COMPOSITION_REGIONS)
    pooled.to_csv(OUT / "composition_by_region_pooled.csv", index=False)

    reg = pd.Series(region, dtype=object).fillna("untraced")
    counts = (
        pd.crosstab(reg, pd.Series(brain, name="brain"))
        .stack()
        .rename("n_cells")
        .reset_index()
        .rename(columns={"row_0": "region"})
    )
    counts.columns = ["region", "brain", "n_cells"]
    counts = counts[["brain", "region", "n_cells"]].sort_values(["brain", "region"])
    counts.to_csv(OUT / "region_cell_counts.csv", index=False)

    c = composition_contrast(disp, regions=PANEL_REGIONS)
    summ = dict(
        regions=PANEL_REGIONS,
        classes=sorted(disp.loc[disp.region.isin(PANEL_REGIONS), "cluster"].unique()),
        floor=c.floor,
        caste_median=c.caste_median,
        regional_median=c.regional_median,
        ratio=c.ratio,
        n_caste=c.n_caste,
        n_regional=c.n_regional,
        n_caste_over_regional_median=c.n_caste_over_regional_median,
        caste_max=c.caste_max,
    )
    (OUT / "composition_contrast.json").write_text(json.dumps(summ, indent=2) + "\n")
    print(counts.to_string(index=False), flush=True)
    print(json.dumps(summ, indent=2), flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
