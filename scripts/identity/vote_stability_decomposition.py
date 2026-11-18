"""Decompose cross-seed label stability into voting and anatomy contributions.

Per-cell Tangram labels are only ~71% reproducible across random seeds (see
seed_churn_detail). Two candidate explanations for why the voted labels are
nonetheless stable:

  (i)  Voting -- aggregating per-cluster majorities averages out per-cell
       churn, independent of anatomy.
  (ii) Anatomy -- the hard override pins the two big blocks.

If (i), the override affects accuracy rather than stability, and an
anatomy-free cluster vote is equally reproducible. If (ii), the override is
also needed for reproducibility.

Measures cross-seed agreement at three levels of aggregation:
    per-cell argmax  ->  expression-only vote (hierarchical_vote, no anatomy)  ->  the
    same plus the anatomy override

Anatomy (both the F1 targets and the anatomy-guided override) is the
nine-region flag set from ``attach_region_flags``, joined on slice_name +
cell_id; the ``in_*`` columns stored in combined_clustered.h5ad predate it.
Untraced sections count as outside a region.

Writes results/handoff/kc_flags_nine_region/vote_stability_decomposition.csv.

    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/identity/vote_stability_decomposition.py
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
from bimp_st.cluster_labels import anatomy_guided_vote, get_pooled, hierarchical_vote
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.nine_region_common import (
    STATISTICAL_REGIONS,
    attach_region_flags,
    kc_mask,
    optic_mask,
    traced_mask,
)
from bimp_st.paths import CANONICAL_TANGRAM, RESULTS, SPATIAL_COMBINED

REPO = Path(".")
SPATIAL_H5 = SPATIAL_COMBINED
CANON = CANONICAL_TANGRAM
SEED1 = REPO / "results/tangram/combined_clustered_seedens_s1_tangram.h5ad"

print("loading spatial object...", flush=True)
a = ad.read_h5ad(SPATIAL_H5)
# anatomy_guided_vote casts in_* with astype(bool), which a nullable NA breaks
for c in attach_region_flags(a, regions=STATISTICAL_REGIONS):
    a.obs[c] = traced_mask(a.obs[c])  # raises on an untraced (NA) cell
n = a.n_obs
obs = a.obs

calyx = traced_mask(kc_mask(obs))
optic = traced_mask(optic_mask(obs))


def labels_of(path):
    return read_obs(path)["tangram_cell_type"].astype(str).values


runs = {"canonical(seed42)": labels_of(CANON), "seed1": labels_of(SEED1)}

out = {}
for tag, lab in runs.items():
    a.obs["tangram_cell_type"] = lab
    percell = np.array([get_pooled(x) for x in lab])

    hierarchical_vote(
        a, cluster_col="leiden", label_col="tangram_cell_type", dst=f"_cv_{tag}"
    )
    a.obs[f"_cv_{tag}"] = [
        v if v == "Mixed" else get_pooled(v) for v in a.obs[f"_cv_{tag}"].astype(str)
    ]
    cv = a.obs[f"_cv_{tag}"].astype(str).values

    anatomy_guided_vote(
        a,
        cluster_col="leiden",
        label_col="tangram_cell_type",
        dst=f"_av_{tag}",
        base="hierarchical",
    )
    av = a.obs[f"_av_{tag}"].astype(str).values

    out[tag] = {
        "per-cell argmax (pooled)": percell,
        "expression-only vote (NO anatomy)": cv,
        "anatomy_guided_vote": av,
    }
    print(f"  {tag}: computed all three", flush=True)

A, B = out["canonical(seed42)"], out["seed1"]


def score(voted):
    kc = voted == "Kenyon cells"
    ol = np.isin(voted, ["Optic lobe (lamina)", "Optic lobe (medulla)", "Optic lobe"])
    kr = calyx[kc].mean() if kc.any() else 0.0
    ck = kc[calyx].mean() if calyx.any() else 0.0
    orc = optic[ol].mean() if ol.any() else 0.0
    oo = ol[optic].mean() if optic.any() else 0.0
    return (2 * kr * ck / (kr + ck + 1e-9), 2 * orc * oo / (orc + oo + 1e-9))


print("\n" + "=" * 78)
print("CROSS-SEED REPRODUCIBILITY vs ACCURACY, by level of aggregation")
print("=" * 78)
rows = []
for level in (
    "per-cell argmax (pooled)",
    "expression-only vote (NO anatomy)",
    "anatomy_guided_vote",
):
    agree = (A[level] == B[level]).mean()
    agree_calyx = (A[level][calyx] == B[level][calyx]).mean()
    kc_f1_a, ol_f1_a = score(A[level])
    kc_f1_b, ol_f1_b = score(B[level])
    rows.append(
        {
            "level": level,
            "cross_seed_agreement": agree,
            "agreement_within_calyx": agree_calyx,
            "KC_F1_seed42": kc_f1_a,
            "KC_F1_seed1": kc_f1_b,
            "OL_F1_seed42": ol_f1_a,
            "OL_F1_seed1": ol_f1_b,
        }
    )
df = pd.DataFrame(rows).set_index("level")
pd.set_option("display.width", 250)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(df.to_string())

o = RESULTS / "handoff" / "kc_flags_nine_region"
o.mkdir(parents=True, exist_ok=True)
df.to_csv(o / "vote_stability_decomposition.csv")
print(f"\nwrote {o}/vote_stability_decomposition.csv")
