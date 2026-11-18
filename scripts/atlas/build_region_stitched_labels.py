"""Build a region-STITCHED canonical label set and compare it to the combined-run
labels.

Motivation: the single combined-ref Tangram run (29 types competing on ~100
panel genes) can assign optic types to central-brain cells (e.g. OL_8 to central
and unflagged cells), and its agreement with the region-specific runs is what
this script measures. The region-specific runs (CB ref on non-optic, OL ref on
optic) exactly partition the atlas cells. This stitches them:

    tangram_cell_type_stitched = OL-run label   if cell is optic-flagged
                                 CB-run label    otherwise

Adds the column to the canonical + spatial objects, writes a stitched canonical
h5ad, and prints the stitched-vs-combined agreement and the caste-composition
shift.

Run: PYTHONPATH=src .venv/bin/python scripts/atlas/build_region_stitched_labels.py
"""

import os
import sys

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from bimp_st.paths import (  # noqa: E402
    CANONICAL_CENTRAL,
    CANONICAL_OPTIC,
    CANONICAL_STITCHED,
    CANONICAL_TANGRAM,
    SPATIAL_COMBINED,
)

# Canonical inputs, whose cell-type columns are keyed by name, not position.
# Objects under results/tangram/_superseded/ are not, and are never read.
CANON = str(CANONICAL_TANGRAM)
CENTRAL = str(CANONICAL_CENTRAL)
OPTIC = str(CANONICAL_OPTIC)
STITCHED_OUT = str(CANONICAL_STITCHED)
# Writing the stitched column back into the canonical object mutates the file
# every other script reads. Opt in explicitly.
WRITE_INTO_CANONICAL = os.environ.get("BIMP_STITCH_WRITE_CANONICAL") == "1"
SPATIAL = str(SPATIAL_COMBINED)


def key(a):
    sl = (
        a.obs["slice_name"].astype(str).values
        if "slice_name" in a.obs
        else np.array([""] * a.n_obs)
    )
    cid = a.obs["cell_id"].astype(str).values
    return pd.Index([f"{s}||{c}" for s, c in zip(sl, cid, strict=False)])


canon = ad.read_h5ad(CANON)
central = ad.read_h5ad(CENTRAL)
optic = ad.read_h5ad(OPTIC)

canon_k = key(canon)
cen_map = pd.Series(
    central.obs["tangram_cell_type"].astype(str).values, index=key(central)
)
opt_map = pd.Series(optic.obs["tangram_cell_type"].astype(str).values, index=key(optic))
assert canon_k.is_unique and cen_map.index.is_unique and opt_map.index.is_unique

optic_flag = (
    canon.obs[["in_lamina", "in_medulla", "in_lobula"]].astype(bool).to_numpy().any(1)
)
print(f"optic-flagged: {optic_flag.sum():,} / {canon.n_obs:,}")

stitched = np.where(
    optic_flag, opt_map.reindex(canon_k).values, cen_map.reindex(canon_k).values
)
n_na = pd.isna(stitched).sum()
print(f"stitched NaN (unmatched): {n_na}")
canon.obs["tangram_cell_type_stitched"] = pd.Categorical([str(x) for x in stitched])

combined = canon.obs["tangram_cell_type"].astype(str).values
agree = (combined == canon.obs["tangram_cell_type_stitched"].astype(str).values).mean()
print(f"\nstitched vs combined agreement: {agree:.1%}")
print(
    "cross-region leakage removed: OL_8 in stitched non-optic cells =",
    int(
        (
            (canon.obs["tangram_cell_type_stitched"].astype(str) == "OL_8")
            & ~optic_flag
        ).sum()
    ),
    "(was ~110k+ under combined)",
)

# caste composition under both label sets
brain = canon.obs["brain"].astype(str).values
print(
    "\n=== caste composition shift (worker frac - queen frac), top movers by label "
    "set ==="
)
for col in ["tangram_cell_type", "tangram_cell_type_stitched"]:
    ct = canon.obs[col].astype(str).values
    df = pd.DataFrame({"ct": ct, "brain": brain})
    tab = pd.crosstab(df["ct"], df["brain"])
    fr = tab.div(tab.sum(0), axis=1)
    if "queen" in fr and "worker" in fr:
        diff = (fr["worker"] - fr["queen"]).sort_values()
        print(
            f"\n[{col}] most worker-DEPLETED:",
            {k: round(v * 100, 2) for k, v in diff.head(4).items()},
        )
        print(
            f"[{col}] most worker-ENRICHED:",
            {k: round(v * 100, 2) for k, v in diff.tail(4).items()},
        )

if WRITE_INTO_CANONICAL:
    canon.write_h5ad(CANON)
    print(f"WROTE tangram_cell_type_stitched into {CANON}")
else:
    print(f"NOT modifying {CANON} (set BIMP_STITCH_WRITE_CANONICAL=1 to)")
# write standalone stitched canonical
canon2 = canon.copy()
canon2.obs["tangram_cell_type"] = canon2.obs["tangram_cell_type_stitched"]
canon2.write_h5ad(STITCHED_OUT)
# add to spatial object too
sp = ad.read_h5ad(SPATIAL)
if WRITE_INTO_CANONICAL and key(sp).equals(canon_k):
    sp.obs["tangram_cell_type_stitched"] = canon.obs[
        "tangram_cell_type_stitched"
    ].values
    sp.write_h5ad(SPATIAL)
    print("\nadded tangram_cell_type_stitched to spatial object")
print(f"\nwrote stitched canonical to {STITCHED_OUT}")
