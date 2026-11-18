"""Why do many optic-neuropil cells vote 'Glia' with no anatomy?

The anatomy-free sankey compares KC<->calyx (precision and recall) with the
optic lobe, where much of the medulla/lamina/lobula votes Glia. Two
candidate explanations:

  (a) Biology -- optic neuropils are glia-rich at Xenium's cell
      segmentation, and the OL_* neuron block is a minority of segmented
      cells there.
  (b) Vote failure -- one or two large Leiden clusters inside the optic
      neuropils vote 'Glia' by a thin margin. cluster_vote is winner-take-all
      per cluster, so a near-even split flips all of a cluster's cells.

(b) is tested via the per-cluster vote margins for the clusters holding the
optic cells and the per-cell argmax composition of the same cells: if the
per-cell argmax says OL but the cluster vote says Glia, the vote rather than
the mapping is responsible.

Also prints a crosstab of what the cells that change label between
anatomy_guided_vote and cluster_vote become, i.e. whether dropping the
override loses optic neurons or stops forcing glia clusters to be neurons.

Read-only.
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from bimp_st.cluster_labels import (  # noqa: E402
    anatomy_guided_vote,
    get_pooled,
    hierarchical_vote,
)
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.nine_region_common import kc_mask, optic_mask, traced_mask  # noqa: E402
from bimp_st.paths import CANONICAL_TANGRAM, SPATIAL_COMBINED  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 80)

print("loading...", flush=True)
a = ad.read_h5ad(SPATIAL_COMBINED)
fine = read_obs(CANONICAL_TANGRAM)["tangram_cell_type"].astype(str).values
a.obs["tangram_cell_type"] = fine

n = a.n_obs
optic = traced_mask(optic_mask(a.obs))
calyx = traced_mask(kc_mask(a.obs))

# The expression-only vote (pooled), and the same vote plus the anatomy
# override, so the override is the only difference between the two.
hierarchical_vote(a, cluster_col="leiden", label_col="tangram_cell_type", dst="_cv")
a.obs["_cv"] = [v if v == "Mixed" else get_pooled(v) for v in a.obs["_cv"].astype(str)]
anatomy_guided_vote(
    a,
    cluster_col="leiden",
    label_col="tangram_cell_type",
    dst="_av",
    base="hierarchical",
)
cv = a.obs["_cv"].astype(str).values
av = a.obs["_av"].astype(str).values
pooled_percell = np.array([get_pooled(x) for x in fine])

print(f"\noptic-neuropil cells: {optic.sum():,}")

print("\n=== 1. per-CELL argmax (pooled) inside the optic neuropils ===")
print("    if this says Optic lobe while the cluster vote says Glia, the")
print("    VOTE is what loses them, not the Tangram mapping.")
pc = pd.Series(pooled_percell[optic]).value_counts(normalize=True)
print((pc * 100).round(1).to_string())

print("\n=== 2. cluster_vote label of the same cells ===")
print((pd.Series(cv[optic]).value_counts(normalize=True) * 100).round(1).to_string())

print("\n=== 3. the Leiden clusters that own the optic cells ===")
lei = a.obs["leiden"].astype(str).values
rows = []
for cl, cnt in pd.Series(lei[optic]).value_counts().head(12).items():
    m = lei == cl
    vc = pd.Series(pooled_percell[m]).value_counts(normalize=True)
    rows.append(
        {
            "leiden": cl,
            "cells_in_optic": int(cnt),
            "frac_of_cluster_in_optic": float(optic[m].mean()),
            "cluster_size": int(m.sum()),
            "vote": cv[m][0],
            "top1": vc.index[0],
            "top1_frac": float(vc.iloc[0]),
            "top2": vc.index[1] if len(vc) > 1 else "",
            "top2_frac": float(vc.iloc[1]) if len(vc) > 1 else np.nan,
            "margin": float(vc.iloc[0] - (vc.iloc[1] if len(vc) > 1 else 0)),
        }
    )
d = pd.DataFrame(rows)
print(d.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(
    "\n  -> a small 'margin' on a big cluster is the failure mode: "
    "winner-take-all flips every cell in it."
)

print("\n=== 4. blast radius: what the changed cells become ===")
changed = av != cv
print(
    f"cells changing label when the override is dropped: "
    f"{changed.sum():,} ({changed.mean():.1%})"
)
ct = pd.crosstab(
    pd.Series(av[changed], name="anatomy_guided_vote"),
    pd.Series(cv[changed], name="cluster_vote"),
)
print(ct.to_string())

print("\n=== 5. same, restricted to the optic neuropils ===")
mo = changed & optic
if mo.any():
    print(
        pd.crosstab(
            pd.Series(av[mo], name="anatomy_guided_vote"),
            pd.Series(cv[mo], name="cluster_vote"),
        ).to_string()
    )

print("\n=== 6. fine (unpooled) argmax inside the optic neuropils, top 12 ===")
print(
    (pd.Series(fine[optic]).value_counts(normalize=True).head(12) * 100)
    .round(1)
    .to_string()
)

out = REPO / "results" / "reframe" / "celltype_anatomy"
out.mkdir(parents=True, exist_ok=True)
d.to_csv(out / "optic_cluster_vote_margins.csv", index=False)
ct.to_csv(out / "blast_radius_crosstab.csv")
print(f"\nwrote {out}/optic_cluster_vote_margins.csv and blast_radius_crosstab.csv")
