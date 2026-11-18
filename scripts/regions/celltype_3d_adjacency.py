"""Which cell types are 3D neighbours, and does 3D change the picture?

Asks, at the level of individual cell types rather than anatomical regions,
which pairs of types are enriched or depleted as each other's neighbours, and
whether adding the cross-section dimension changes that from what a single
section shows.

Reuses the neighbour-composition builders from the region-prediction scripts:
  - neighbourhood_composition()    (region_prediction)   -- 2D,
    strictly within one section.
  - neighbourhood_composition_3d() (region_prediction_3d)   -- 3D,
    (x, y, z_level * Z_SCALE) within one brain, brains never linked. Z_SCALE
    (4.705882 = 1/0.2125 um/px) and z_level (10 um nominal section spacing;
    the per-crop order set at conversion time, not a re-derived sort) are the
    values validated in region_prediction_3d.py.

Method: every cell has a canonical-17 type (its "source") and a k=20
neighbour-composition vector (its "target" distribution, row-summing to 1).
Averaging that vector within each source type gives a type x type adjacency
matrix, in 2D and separately in 3D. Comparing each entry to the type's
global frequency gives a log2 enrichment relative to a well-mixed null;
comparing the 3D and 2D enrichment for the same pair isolates what the third
dimension changes.

Restricted to cells with a valid slice_name/z_level (crop in the ordered z
stack); no region restriction.

Writes results/reframe/celltype_3d_adjacency/.
"""

from __future__ import annotations

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports

import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bimp_st.cluster_labels import CANONICAL_ORDER, add_canonical  # noqa: E402
from bimp_st.paths import CANONICAL_TANGRAM, SPATIAL_COMBINED  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regions.region_prediction import neighbourhood_composition  # noqa: E402
from regions.region_prediction_3d import (  # noqa: E402
    Z_SCALE,
    neighbourhood_composition_3d,
)

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "reframe" / "celltype_3d_adjacency"
OUT.mkdir(parents=True, exist_ok=True)

K = 20


def load():
    print(f"loading {SPATIAL_COMBINED.name} ...", flush=True)
    a = ad.read_h5ad(SPATIAL_COMBINED)
    from bimp_st.paths import keyed_obs_column

    # joined on (slice_name, cell_id), never by position
    a.obs["tangram_cell_type"] = keyed_obs_column(
        a, "tangram_cell_type", CANONICAL_TANGRAM
    ).astype(str)
    add_canonical(a, src="tangram_cell_type")

    obs = a.obs
    has_z = (
        obs["z_level"].notna().to_numpy()
        if "z_level" in obs
        else np.zeros(a.n_obs, bool)
    )
    has_slice = obs["slice_name"].notna().to_numpy()
    keep = has_z & has_slice
    print(
        f"  {keep.sum():,} / {a.n_obs:,} cells have a valid z_level and slice_name",
        flush=True,
    )

    sub = obs.loc[keep].reset_index(drop=True)
    xy = np.asarray(a.obsm["spatial"], float)[keep]
    types = sub["cell_type_canonical"].astype(str).values
    present = [t for t in CANONICAL_ORDER if t in set(types)]
    present += sorted(set(types) - set(present))
    code_of = {t: i for i, t in enumerate(present)}
    codes = np.array([code_of[t] for t in types])
    sections = sub["slice_name"].astype(str).values
    brains = sub["brain"].astype(str).values
    z = sub["z_level"].to_numpy(dtype=float) * Z_SCALE
    xyz = np.column_stack([xy, z])

    counts = pd.Series(types).value_counts()
    print("  cell-type counts:\n" + counts.to_string(), flush=True)

    return dict(
        types=types,
        present=present,
        codes=codes,
        sections=sections,
        brains=brains,
        xy=xy,
        xyz=xyz,
    )


def adjacency_matrix(comp, codes, present):
    """Row i = mean neighbour-composition vector of cells of type i."""
    n = len(present)
    M = np.zeros((n, n))
    for i in range(n):
        mask = codes == i
        if mask.sum() == 0:
            continue
        M[i] = comp[mask].mean(0)
    return pd.DataFrame(M, index=present, columns=present)


def enrichment(M, global_freq):
    """log2(observed neighbour frequency / global frequency), per target column."""
    return np.log2(np.maximum(M, 1e-6) / global_freq.values[None, :])


def main():
    d = load()
    n_cat = len(d["present"])
    global_freq = pd.Series(d["codes"]).value_counts(normalize=True).sort_index()
    global_freq.index = d["present"]

    print("\nbuilding 2D (in-section) neighbourhoods ...", flush=True)
    comp_2d = neighbourhood_composition(d["xy"], d["codes"], n_cat, d["sections"], K)
    print(
        "building 3D (cross-section, register-crossing) neighbourhoods ...", flush=True
    )
    comp_3d, cross_z, mean_dz = neighbourhood_composition_3d(
        d["xyz"], d["codes"], n_cat, d["brains"], K
    )

    M_2d = adjacency_matrix(comp_2d, d["codes"], d["present"])
    M_3d = adjacency_matrix(comp_3d, d["codes"], d["present"])
    E_2d = pd.DataFrame(
        enrichment(M_2d.values, global_freq), index=d["present"], columns=d["present"]
    )
    E_3d = pd.DataFrame(
        enrichment(M_3d.values, global_freq), index=d["present"], columns=d["present"]
    )
    diff = E_3d - E_2d

    M_2d.to_csv(OUT / "adjacency_2d.csv")
    M_3d.to_csv(OUT / "adjacency_3d.csv")
    E_2d.to_csv(OUT / "enrichment_2d.csv")
    E_3d.to_csv(OUT / "enrichment_3d.csv")
    diff.to_csv(OUT / "enrichment_diff_3d_minus_2d.csv")
    global_freq.to_csv(OUT / "global_freq.csv")

    print(f"\ncross-section neighbour fraction: {np.nanmean(cross_z):.3f}", flush=True)
    print(f"mean |dz| (z_level units): {np.nanmean(mean_dz):.3f}", flush=True)

    print("\nself-adjacency (diagonal), 2D vs 3D enrichment:", flush=True)
    diag = pd.DataFrame(
        {"2D": np.diag(E_2d.values), "3D": np.diag(E_3d.values)}, index=d["present"]
    )
    diag["3D - 2D"] = diag["3D"] - diag["2D"]
    print(diag.sort_values("2D", ascending=False).to_string(), flush=True)
    diag.to_csv(OUT / "self_adjacency.csv")

    print("\nlargest |3D - 2D| off-diagonal shifts:", flush=True)
    off = diff.copy()
    np.fill_diagonal(off.values, np.nan)
    flat = off.stack().rename("shift").reset_index()
    flat.columns = ["source", "target", "shift"]
    flat["abs_shift"] = flat["shift"].abs()
    print(
        flat.sort_values("abs_shift", ascending=False).head(15).to_string(index=False),
        flush=True,
    )
    flat.to_csv(OUT / "largest_shifts.csv", index=False)

    print(f"\nDone. Outputs in {OUT}", flush=True)


if __name__ == "__main__":
    main()
