"""Is the canonical Tangram mapping reproducible across random seeds?

A same-seed re-run on the GPU differs only through CUDA non-determinism
(low-order digits of the mapping matrix). A different random seed is a
strictly larger perturbation,
so measuring it bounds that effect from above.

The three runs compared are the production configuration at three seeds:

    combined_clustered_g2_R2_s42   (canonical, the one every result uses)
    combined_clustered_g2_R2_s43
    combined_clustered_g2_R2_s44

all made with the canonical mapping's recipe.

results/reframe/spatial_relabel_test/ (vote_stability_decomposition.csv,
seed_churn_by_group.csv) holds results from the
``combined_clustered_seedens_*`` runs, a different mapping recipe; those
numbers describe that mapping, not this one.

What this measures, at two levels of aggregation:

  per-cell argmax, pooled   -- what a single cell's own label does
  hierarchical_vote         -- the canonical labelling every figure uses

and, against hand-drawn anatomy that neither the reference nor the mapping can
see, the Kenyon-cell and optic-lobe F1 of each seed. A stable vote with an
unstable per-cell argmax supports labelling at the cluster level.

Anatomy is the nine-region flag set (``nine_region_common.attach_region_flags``,
joined on slice_name + cell_id); the ``in_*`` columns stored inside
combined_clustered.h5ad predate it and are not read. Untraced sections count as
outside a region, as in every other F1 against anatomy here.

Writes results/handoff/kc_flags_nine_region/tangram_seed_stability/.
"""

import itertools
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from bimp_st.cluster_labels import get_pooled, hierarchical_vote  # noqa: E402
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.nine_region_common import (  # noqa: E402
    STATISTICAL_REGIONS,
    attach_region_flags,
    kc_mask,
    optic_mask,
    traced_mask,
)
from bimp_st.paths import RESULTS, SPATIAL_COMBINED  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SPATIAL_H5 = SPATIAL_COMBINED
TG = REPO / "results/tangram"
OUT = RESULTS / "handoff" / "kc_flags_nine_region" / "tangram_seed_stability"

SEEDS = {
    "s42": TG / "combined_clustered_g2_R2_s42_tangram.h5ad",
    "s43": TG / "combined_clustered_g2_R2_s43_tangram.h5ad",
    "s44": TG / "combined_clustered_g2_R2_s44_tangram.h5ad",
}



def labels_of(path):
    """Read only obs/tangram_cell_type; these objects are 2.4 GB each."""
    return read_obs(path)["tangram_cell_type"].astype(str).values


def main():
    for _tag, p in SEEDS.items():
        if not p.exists():
            raise SystemExit(f"missing seed run: {p}")

    print("loading spatial object (leiden + anatomy flags)...", flush=True)
    a = ad.read_h5ad(SPATIAL_H5)
    attach_region_flags(a, regions=STATISTICAL_REGIONS)
    n = a.n_obs
    calyx = traced_mask(kc_mask(a.obs))
    optic = traced_mask(optic_mask(a.obs))
    print(
        f"  {n:,} cells; {calyx.sum():,} in calyx, {optic.sum():,} in optic", flush=True
    )

    levels = {}
    for tag, p in SEEDS.items():
        lab = labels_of(p)
        if len(lab) != n:
            raise SystemExit(f"{tag}: {len(lab)} labels for {n} cells")
        # per-cell argmax, pooled to the display groups
        percell = np.array([get_pooled(x) for x in lab])
        # the canonical labelling
        a.obs["tangram_cell_type"] = lab
        hierarchical_vote(
            a, cluster_col="leiden", label_col="tangram_cell_type", dst=f"_hv_{tag}"
        )
        voted = a.obs[f"_hv_{tag}"].astype(str).to_numpy()
        levels[tag] = {
            "per-cell argmax (pooled)": percell,
            "hierarchical_vote (shipped)": voted,
        }
        print(f"  {tag}: labelled", flush=True)

    def f1_vs(mask_pred, mask_true):
        if not mask_pred.any() or not mask_true.any():
            return 0.0
        prec = mask_true[mask_pred].mean()
        rec = mask_pred[mask_true].mean()
        return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

    def score(v):
        kc = v == "Kenyon cells"
        ol = np.char.startswith(v.astype(str), "Optic lobe")
        return f1_vs(kc, calyx), f1_vs(ol, optic)

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- agreement, every seed pair --------------------------------------
    rows = []
    for level in ("per-cell argmax (pooled)", "hierarchical_vote (shipped)"):
        for x, y in itertools.combinations(SEEDS, 2):
            A, B = levels[x][level], levels[y][level]
            rows.append(
                {
                    "level": level,
                    "pair": f"{x} vs {y}",
                    "agreement": float((A == B).mean()),
                    "agreement_within_calyx": float((A[calyx] == B[calyx]).mean()),
                    "agreement_within_optic": float((A[optic] == B[optic]).mean()),
                }
            )
    pairs = pd.DataFrame(rows)
    pairs.to_csv(OUT / "seed_pair_agreement.csv", index=False)

    # ---- per-seed accuracy against anatomy -------------------------------
    rows = []
    for level in ("per-cell argmax (pooled)", "hierarchical_vote (shipped)"):
        for tag in SEEDS:
            kc, ol = score(levels[tag][level])
            rows.append(
                {
                    "level": level,
                    "seed": tag,
                    "KC_F1_vs_calyx": kc,
                    "OL_F1_vs_optic": ol,
                }
            )
    acc = pd.DataFrame(rows)
    acc.to_csv(OUT / "seed_accuracy.csv", index=False)

    # ---- summary: cross-seed agreement per level -------------------------
    summ = (
        pairs.groupby("level")
        .agg(
            mean_agreement=("agreement", "mean"),
            min_agreement=("agreement", "min"),
            mean_agreement_calyx=("agreement_within_calyx", "mean"),
        )
        .reset_index()
    )
    spread = (
        acc.groupby("level")
        .agg(
            KC_F1_min=("KC_F1_vs_calyx", "min"),
            KC_F1_max=("KC_F1_vs_calyx", "max"),
            OL_F1_min=("OL_F1_vs_optic", "min"),
            OL_F1_max=("OL_F1_vs_optic", "max"),
        )
        .reset_index()
    )
    summary = summ.merge(spread, on="level")
    summary["KC_F1_range"] = summary["KC_F1_max"] - summary["KC_F1_min"]
    summary.to_csv(OUT / "summary.csv", index=False)

    # ---- which groups churn, per cell ------------------------------------
    base = levels["s42"]["per-cell argmax (pooled)"]
    rows = []
    for grp in pd.unique(base):
        m = base == grp
        kept = [
            float((levels[o]["per-cell argmax (pooled)"][m] == grp).mean())
            for o in ("s43", "s44")
        ]
        rows.append(
            {
                "group": grp,
                "n_cells_s42": int(m.sum()),
                "mean_kept_in_other_seeds": float(np.mean(kept)),
            }
        )
    churn = pd.DataFrame(rows).sort_values("mean_kept_in_other_seeds")
    churn.to_csv(OUT / "percell_churn_by_group.csv", index=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== agreement by seed pair ===")
    print(pairs.to_string(index=False))
    print("\n=== accuracy against hand-drawn anatomy ===")
    print(acc.to_string(index=False))
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print("\n=== per-cell churn by group (s42 -> other seeds) ===")
    print(churn.to_string(index=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
