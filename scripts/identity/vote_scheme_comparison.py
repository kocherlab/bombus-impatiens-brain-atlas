"""Pool-then-vote vs vote-over-the-17, with a purity-threshold sweep.

Compares labelling spatial figures with pooled groups against labelling them
with names drawn straight from the canonical 17.

Measures, on all 1,300,192 spatial cells:
  (A) anatomy_guided_vote(pool=True)   -- pool 17 -> 7 POOLED_ORDER, then vote
  (B) anatomy_guided_vote(pool=False)  -- vote over the canonical 17 directly
  (C) (B) swept over min_purity 0.50 .. 0.25
  (B') cluster_vote(pool=False)        -- (B) without the anatomy prior

For each: resolved categories + share, Mixed fraction, count above 1% of cells,
and calyx recovery (strict = labelled exactly "Kenyon cells"; family = any
Kenyon-cell category of the 17).

What the comparison checks (the caste_comparison_sccoda.py vocabulary comment
refers to it):
  * whether a scheme's resolved names are plausible or merely coarse, e.g.
    whether a whole glial or projection-neuron block votes for one subtype the
    panel cannot distinguish;
  * whether lowering min_purity rescues (B), and whether the categories that
    appear when it does are those the anatomical-plausibility audit flags;
  * calyx recovery, which the anatomy prior constrains in (A) and (B) (it
    forces calyx clusters to "Kenyon cells"), so only (B') tests it.

Run with the pinned venv:
    PYTHONPATH=src .venv/bin/python scripts/identity/vote_scheme_comparison.py

Outputs to results/reframe/vote_over_17/:
    vote_scheme_comparison.csv   one row per scheme
    per_cluster_labels.csv       per-Leiden-cluster label + size, so a
                                 category winning only a tiny cluster is
                                 visible rather than counted as resolved
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

W = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(W / "src"))
from bimp_st.cluster_labels import (  # noqa: E402
    CANONICAL_ORDER,
    add_canonical,
    anatomy_guided_vote,
    cluster_vote,
)

warnings.filterwarnings("ignore")
H5 = W / "pipelines" / "combined_atlas" / "data" / "combined_clustered.h5ad"
OUT = W / "results" / "reframe" / "vote_over_17"
OUT.mkdir(parents=True, exist_ok=True)

CALYX_FLAGS = ("in_kc_ic", "in_kc_nc", "in_kc_oc")
KC_FAMILY = {
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Other Kenyon cells",
    "Kenyon cells",
}


def summarize(tag: str, labels: pd.Series, calyx: np.ndarray, n: int) -> dict:
    vc = labels.value_counts()
    share = vc / n
    mixed = float(share.get("Mixed", 0.0))
    resolved = share.drop("Mixed", errors="ignore")
    above1 = resolved[resolved >= 0.01]
    lab = labels.values
    strict = float((lab[calyx] == "Kenyon cells").mean())
    family = float(pd.Series(lab[calyx]).isin(KC_FAMILY).mean())
    print(f"\n=== {tag} ===")
    print(
        f"  distinct labels           : {len(vc)}  (resolved {len(resolved)}, Mixed "
        f"present={'Mixed' in vc.index})"
    )
    print(f"  resolved >= 1% of cells   : {len(above1)}")
    print(f"  Mixed fraction            : {mixed:.4f}")
    print(f"  calyx recovery strict/fam : {strict:.4f} / {family:.4f}")
    for k, v in share.sort_values(ascending=False).items():
        print(f"      {k:34s} {v * 100:7.3f}%   ({int(vc[k]):,} cells)")
    return dict(
        scheme=tag,
        n_distinct=len(vc),
        n_resolved=len(resolved),
        n_resolved_ge_1pct=len(above1),
        mixed_frac=mixed,
        calyx_recovery_strict=strict,
        calyx_recovery_family=family,
    )


def main() -> None:
    print(f"loading {H5.name} ...", flush=True)
    a = ad.read_h5ad(H5)
    n = a.n_obs
    print(
        f"  {n:,} cells; region flags present: "
        f"{[f for f in CALYX_FLAGS if f in a.obs.columns]}",
        flush=True,
    )
    add_canonical(a, src="tangram_cell_type", dst="ct17")
    print(
        f"  canonical-17 categories present: "
        f"{sorted(set(a.obs['ct17'].astype(str)) & set(CANONICAL_ORDER))}",
        flush=True,
    )

    calyx = np.zeros(n, bool)
    for f in CALYX_FLAGS:
        if f in a.obs.columns:
            calyx |= a.obs[f].astype(bool).values
    print(f"  calyx cells: {calyx.sum():,}", flush=True)

    rows = []

    # (A) pool then vote -- the labelling the figures use
    anatomy_guided_vote(
        a,
        cluster_col="leiden",
        label_col="tangram_cell_type",
        pool=True,
        min_purity=0.50,
        dst="voteA",
    )
    rows.append(
        summarize(
            "A: pool-then-vote (7 groups), purity 0.50",
            a.obs["voteA"].astype(str),
            calyx,
            n,
        )
    )

    # (B)+(C) vote over the 17 directly, purity sweep
    for thr in (0.50, 0.45, 0.40, 0.35, 0.30, 0.25):
        anatomy_guided_vote(
            a,
            cluster_col="leiden",
            label_col="ct17",
            pool=False,
            min_purity=thr,
            dst="voteB",
        )
        rows.append(
            summarize(
                f"B: vote-over-17, purity {thr:.2f}",
                a.obs["voteB"].astype(str),
                calyx,
                n,
            )
        )

    # plain (non-anatomy) vote over 17 at 0.50, for reference: shows how much
    # of B's calyx recovery is the anatomy prior rather than expression
    cluster_vote(
        a,
        cluster_col="leiden",
        label_col="ct17",
        pool=False,
        min_purity=0.50,
        dst="voteB_noanat",
    )
    rows.append(
        summarize(
            "B': vote-over-17 WITHOUT anatomy guidance, purity 0.50",
            a.obs["voteB_noanat"].astype(str),
            calyx,
            n,
        )
    )

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "vote_scheme_comparison.csv", index=False)

    # per-cluster detail at the two headline settings, so a tiny-cluster win is visible
    det = []
    for tag, dst in (("A_pool_0.50", "voteA"), ("B_17_0.50", "voteB_noanat")):
        g = (
            pd.DataFrame(
                {
                    "cl": a.obs["leiden"].astype(str).values,
                    "lab": a.obs[dst].astype(str).values,
                }
            )
            .groupby("cl")
            .agg(label=("lab", "first"), n_cells=("lab", "size"))
        )
        g["scheme"] = tag
        det.append(g.reset_index())
    pd.concat(det).to_csv(OUT / "per_cluster_labels.csv", index=False)

    print(f"\n\n=== SUMMARY TABLE ===\n{df.to_string(index=False)}")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
