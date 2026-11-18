"""Does the cluster vote's stability come from the Leiden clusters, or from
the mere act of aggregating tens of thousands of cells?

The cluster vote agrees across seeds more often than the per-cell labels do,
but majority-voting over large groups suppresses per-cell noise however
the groups are drawn. The size-matched null partitions the same cells into
random groups with the Leiden cluster sizes, votes inside those, and measures
the same quantities:

  cross-seed agreement   how reproducible the labelling is
  Kenyon F1 vs calyx     whether the labelling is still informative

A random grouping is expected to score well on the first (every large random
group carries the global majority label in every seed) and poorly on the
second, in which case accuracy rather than stability is what supports the
vote.

Calyx anatomy is the nine-region flag set (``attach_region_flags``, joined on
slice_name + cell_id), not the ``in_kc_*`` columns stored in
combined_clustered.h5ad, which predate it.

Writes results/handoff/kc_flags_nine_region/cluster_vote_null/.
"""

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")

from bimp_st.cluster_labels import get_pooled  # noqa: E402
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.nine_region_common import STATISTICAL_REGIONS, attach_region_flags  # noqa: E402
from bimp_st.paths import RESULTS, SPATIAL_COMBINED  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SPATIAL_H5 = SPATIAL_COMBINED
TG = REPO / "results/tangram"
OUT = RESULTS / "handoff" / "kc_flags_nine_region" / "cluster_vote_null"

SEEDS = {
    s: TG / f"combined_clustered_g2_R2_{s}_tangram.h5ad" for s in ("s42", "s43", "s44")
}
N_REP = 20
RNG = np.random.default_rng(0)


def percell_pooled(path):
    lab = read_obs(path)["tangram_cell_type"].astype(str).values
    return np.array([get_pooled(x) for x in lab])


def majority_by_group(codes, labels, n_groups):
    """Most common label within each group, returned per cell."""
    out = np.empty(len(codes), dtype=object)
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(n_groups))
    ends = np.searchsorted(codes[order], np.arange(n_groups), side="right")
    for g in range(n_groups):
        idx = order[starts[g] : ends[g]]
        if len(idx) == 0:
            continue
        vals, cnt = np.unique(labels[idx], return_counts=True)
        out[idx] = vals[cnt.argmax()]
    return out


def main():
    print("loading spatial object...", flush=True)
    a = ad.read_h5ad(SPATIAL_H5)
    attach_region_flags(a, regions=STATISTICAL_REGIONS)
    n = a.n_obs
    leiden = a.obs["leiden"].astype(str).to_numpy()
    uniq, codes = np.unique(leiden, return_inverse=True)
    sizes = np.bincount(codes)
    n_groups = len(uniq)
    from bimp_st.nine_region_common import kc_mask, traced_mask

    calyx = traced_mask(kc_mask(a.obs))
    print(
        f"  {n:,} cells, {n_groups} Leiden clusters "
        f"(sizes {sizes.min():,}-{sizes.max():,}), {calyx.sum():,} calyx",
        flush=True,
    )

    pooled = {s: percell_pooled(p) for s, p in SEEDS.items()}
    print("  per-cell pooled labels read for 3 seeds", flush=True)

    def kc_f1(v):
        kc = v == "Kenyon cells"
        if not kc.any():
            return 0.0
        prec = calyx[kc].mean()
        rec = kc[calyx].mean()
        return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)

    def score(assign_codes):
        voted = {s: majority_by_group(assign_codes, pooled[s], n_groups) for s in SEEDS}
        pairs = [("s42", "s43"), ("s42", "s44"), ("s43", "s44")]
        agree = float(np.mean([(voted[x] == voted[y]).mean() for x, y in pairs]))
        f1 = float(np.mean([kc_f1(voted[s]) for s in SEEDS]))
        n_lab = int(np.mean([len(set(voted[s])) for s in SEEDS]))
        return agree, f1, n_lab

    real = score(codes)
    print(
        f"\nREAL Leiden grouping : agreement {real[0]:.4f}  "
        f"Kenyon F1 {real[1]:.4f}  distinct labels {real[2]}",
        flush=True,
    )

    rows = []
    for r in range(N_REP):
        perm = RNG.permutation(n)
        rand_codes = np.empty(n, dtype=np.int64)
        pos = 0
        for g, sz in enumerate(sizes):
            rand_codes[perm[pos : pos + sz]] = g
            pos += sz
        a_, f_, k_ = score(rand_codes)
        rows.append({"rep": r, "agreement": a_, "kenyon_f1": f_, "distinct_labels": k_})
        if r < 3 or r == N_REP - 1:
            print(
                f"  null rep {r:2d}: agreement {a_:.4f}  Kenyon F1 {f_:.4f}  "
                f"labels {k_}",
                flush=True,
            )

    null = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    null.to_csv(OUT / "size_matched_null.csv", index=False)

    summ = pd.DataFrame(
        [
            {
                "arm": "Leiden clusters (shipped)",
                "agreement": real[0],
                "kenyon_f1": real[1],
                "distinct_labels": real[2],
            },
            {
                "arm": f"size-matched random groups (mean of {N_REP})",
                "agreement": null.agreement.mean(),
                "kenyon_f1": null.kenyon_f1.mean(),
                "distinct_labels": null.distinct_labels.mean(),
            },
        ]
    )
    summ.to_csv(OUT / "summary.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n=== summary ===")
    print(summ.to_string(index=False))
    a_sd = null.agreement.std(ddof=1)
    if a_sd > 0:
        print(f"\nagreement z vs null: {(real[0] - null.agreement.mean()) / a_sd:+.2f}")
    print(f"Kenyon F1: real {real[1]:.4f} vs null {null.kenyon_f1.mean():.4f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
