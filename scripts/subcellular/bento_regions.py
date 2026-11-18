"""Per-section bento localization fractions by ANATOMICAL region (nine-region flags),
on the same LP-call denominator as bento_regroup.py. Output:
results/bento/by_section/region_composition_by_section.csv
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regroup_common import REPO, aggregate_lp, fractions, region_folders  # noqa: E402

from bimp_st.nine_region_common import region_of  # noqa: E402

OPTIC = ["lamina", "medulla", "lobula", "al"]
FLAGS = ["in_kc_ic", "in_kc_nc", "in_kc_oc"] + [f"in_{r}" for r in OPTIC]

rows, grows, cellrows = [], [], []
for brain in ("queen", "worker"):
    a = ad.read_h5ad(
        REPO / f"results/nine_region/atlas/{brain}_aligned_clustered_9region.h5ad",
        backed="r",
    )
    o = a.obs[["cell_id", "slice_name", *FLAGS]].copy()
    a.file.close()
    labs, conds = [], []
    # a cell in several regions is dropped to "" (kept only if in exactly one)
    regions = [c.removeprefix("in_") for c in FLAGS]
    names = dict(zip(regions, ["KC-IC", "KC-NC", "KC-OC"] + OPTIC, strict=True))
    comp = np.array([names.get(r, "") for r in region_of(o, regions)])
    obs = pd.DataFrame(
        {
            "brain": brain,
            "cell_id": o.cell_id.astype(str).to_numpy(),
            "slice_name": o.slice_name.astype(str).to_numpy(),
            "comp": comp,
            "dist": np.nan,
        }
    )
    obs["region"] = ""
    for r in region_folders(brain):
        obs.loc[obs.slice_name.str.startswith(r), "region"] = r
    obs = obs[obs.region != ""]
    amb = obs.duplicated(["region", "cell_id"], keep=False)
    obs = obs[~amb].reset_index(drop=True)
    sec, cell, gene = aggregate_lp(brain, obs)
    grows.append(fractions(gene))
    cell = cell[cell.comp != ""]
    cell.drop(columns="dist").to_parquet(
        REPO / f"results/bento/by_section/region_cells_{brain}.parquet"
    )
    cellrows.append(cell.groupby(["brain", "slice_name", "comp"], as_index=False).agg(n_cells=("cell_id", "size"), calls_per_cell=("n_calls", "mean")))
    sec = sec[sec.comp != ""]
    rows.append(fractions(sec))
t = pd.concat(rows, ignore_index=True)
t["caste"] = t.brain
out = REPO / "results/bento/by_section/region_composition_by_section.csv"
t.to_csv(out, index=False)
print(t.groupby(["caste", "comp"]).size(), flush=True)
pd.concat(grows, ignore_index=True).to_csv(
    REPO / "results/bento/by_section/region_gene_pattern_by_section.csv", index=False
)
pd.concat(cellrows, ignore_index=True).to_csv(
    REPO / "results/bento/by_section/region_cell_depth_by_section.csv", index=False
)
