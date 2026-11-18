"""Are the transferred cell-type labels spatially coherent, or salt-and-pepper?

Cross-modal comparisons are partly circular, because Tangram assigns the
Xenium labels from the snRNA reference they are compared against. This test
uses something neither the reference nor Tangram sees: the physical
arrangement of the tissue. Cells of one type sit together (Kenyon somata pack
the calyx cup, glia sheath the neuropil), so labels that recover biology
should be shared by neighbouring cells more often than chance.

Statistic: within each section, build a spatial kNN graph (k=10, the
Goltsev/Schurch default used elsewhere in this project) and compute the
**join-count**: the fraction of graph edges joining two cells with the same
label. This is a categorical Moran's I -- no ordering assumed on cell types.

Null: labels permuted within each section, which destroys spatial
arrangement while preserving the cell count, label composition and cell
positions (the same control as in the region-prediction analysis). A section
with 60% Kenyon cells has a high join-count by composition alone, and the null
absorbs that.

Reported per label set:

  cluster_vote / anatomy_guided_vote  -- circular for anatomy (the vote uses
        anatomical membership to assign calyx and optic clusters); reported
        for completeness only, not as validation.
  tangram_cell_type (per-cell argmax, pooled to canonical-17)  -- the main
        test. No anatomy enters it; known to be noisy per cell.
  leiden  -- the panel's own expression clusters, as an upper reference: what
        spatial coherence looks like for labels derived from this data alone.

Outputs to results/reframe/spatial_coherence_c17/:
    join_counts_by_section.csv   per section x label set, observed + null
    summary.csv                  pooled, with z and empirical p
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bimp_st.cluster_labels import add_canonical  # noqa: E402
from bimp_st.paths import CANONICAL_TANGRAM  # noqa: E402

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[2]
H5 = REPO / "pipelines" / "combined_atlas" / "data" / "combined_clustered.h5ad"
# H5's own tangram_cell_type is not the canonical labelling, and the per-cell
# Tangram argmax is the column tested here, so labels come from the canonical
# object.
TANGRAM_LABELS = CANONICAL_TANGRAM
OUT = REPO / "results" / "reframe" / "spatial_coherence_c17"
# Suffixed so the directory names its vocabulary; the canonical-15 run of this
# analysis is in results/reframe/spatial_coherence_c15/.
OUT.mkdir(parents=True, exist_ok=True)

K = 10
N_PERM = 200
SEED = 0


def join_count(nbr: np.ndarray, codes: np.ndarray) -> float:
    """Fraction of kNN edges joining two cells with the same label."""
    return float((codes[nbr] == codes[:, None]).mean())


def main() -> None:
    print(f"loading {H5.name} ...", flush=True)
    a = ad.read_h5ad(H5)
    from bimp_st.paths import keyed_obs_column

    # joined on (slice_name, cell_id), never by position
    a.obs["tangram_cell_type"] = keyed_obs_column(
        a, "tangram_cell_type", TANGRAM_LABELS
    ).astype(str)
    add_canonical(a, src="tangram_cell_type", dst="ct17")

    obs = a.obs
    xy = np.asarray(a.obsm["spatial"], float)
    sections = obs["slice_name"].astype(str).values

    label_sets = {
        "tangram argmax (canonical-17)": obs["ct17"].astype(str).values,
        "leiden (panel expression)": obs["leiden"].astype(str).values,
    }
    if "cluster_vote" in obs.columns:
        label_sets["anatomy-guided vote (circular)"] = (
            obs["cluster_vote"].astype(str).values
        )

    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(SEED)
    rows = []
    uniq = [s for s in pd.unique(sections)]
    print(f"  {len(uniq)} sections, k={K}, {N_PERM} permutations each", flush=True)

    for si, s in enumerate(uniq):
        idx = np.flatnonzero(sections == s)
        if len(idx) <= K + 1:
            continue
        nn = NearestNeighbors(n_neighbors=K + 1).fit(xy[idx])
        nbr = nn.kneighbors(xy[idx], return_distance=False)[:, 1:]

        for name, lab in label_sets.items():
            codes = pd.Categorical(lab[idx]).codes
            obs_jc = join_count(nbr, codes)
            null = np.empty(N_PERM)
            for p in range(N_PERM):
                null[p] = join_count(nbr, rng.permutation(codes))
            z = (obs_jc - null.mean()) / (null.std() + 1e-12)
            rows.append(
                dict(
                    section=s,
                    label_set=name,
                    n_cells=len(idx),
                    join_count=obs_jc,
                    null_mean=float(null.mean()),
                    null_sd=float(null.std()),
                    z=float(z),
                    p_emp=float((null >= obs_jc).mean()),
                )
            )
        if (si + 1) % 10 == 0:
            print(f"    {si + 1}/{len(uniq)} sections", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "join_counts_by_section.csv", index=False)

    print("\n=== pooled over sections (mean +/- sd across sections) ===", flush=True)
    summ = []
    for name, g in df.groupby("label_set"):
        # weight by cells so a 3-cell section does not count like a 50k one
        w = g["n_cells"].values
        obs_w = float(np.average(g["join_count"], weights=w))
        null_w = float(np.average(g["null_mean"], weights=w))
        summ.append(
            dict(
                label_set=name,
                n_sections=len(g),
                join_count_weighted=obs_w,
                null_weighted=null_w,
                excess=obs_w - null_w,
                ratio=obs_w / null_w if null_w else np.nan,
                median_z=float(g["z"].median()),
                sections_p_lt_0p05=int((g["p_emp"] < 0.05).sum()),
            )
        )
        print(
            f"  {name:34s} join={obs_w:.3f}  null={null_w:.3f}  "
            f"excess={obs_w - null_w:+.3f}  ratio={obs_w / null_w:.2f}x  "
            f"median z={g['z'].median():.1f}  "
            f"{int((g['p_emp'] < 0.05).sum())}/{len(g)} sections p<0.05",
            flush=True,
        )

    pd.DataFrame(summ).to_csv(OUT / "summary.csv", index=False)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
