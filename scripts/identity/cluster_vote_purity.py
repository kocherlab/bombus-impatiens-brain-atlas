"""Per-cluster decisions and purity for the canonical cluster vote.

Reports the purity distribution and the count of clusters the vote leaves
undecided, so it runs the rule the figures use: ``hierarchical_vote``. A
cluster takes a fine canonical name at
purity >= 0.5 or a 1.5x lead over the runner-up, else the pooled group at
>= 0.5, else ``Mixed``.

Both tables are written. ``hierarchical_vote_clusters.csv`` is the figure
labelling. ``purity_by_cluster_clustervote.csv`` is ``cluster_vote`` (a single
purity floor at 0.41, no lead test, no pooled fallback), kept for comparison
only; its per-group tallies differ from the figure labelling's.

The purity summary is the same either way: both compute the majority fraction
over the pooled display groups, so it describes the atlas rather than the
rule. Summaries restrict to clusters of at least 1000 cells: the few tiny
clusters (<= 20 cells) are noise and can reach purity 1 trivially, so filter
hierarchical_vote_clusters.csv on n >= 1000.

Writes results/reframe/cluster_vote_purity/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata as ad

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bimp_st.cluster_labels import cluster_vote, hierarchical_vote  # noqa: E402
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.paths import CANONICAL_TANGRAM, SPATIAL_COMBINED  # noqa: E402

OUT = REPO / "results" / "reframe" / "cluster_vote_purity"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading...", flush=True)
    a = ad.read_h5ad(SPATIAL_COMBINED)
    a.obs["tangram_cell_type"] = (
        read_obs(CANONICAL_TANGRAM)["tangram_cell_type"].astype(str).values
    )

    # ---- the canonical rule -----------------------------------------------
    table = hierarchical_vote(
        a, cluster_col="leiden", label_col="tangram_cell_type", dst="_hv"
    )
    table.to_csv(OUT / "hierarchical_vote_clusters.csv")

    conf = table["pooled_purity"].astype(float)
    summary = {
        "n_clusters": int(len(conf)),
        "min": float(conf.min()),
        "max": float(conf.max()),
        "median": float(conf.median()),
        "n_undecided": int((table["label"] == "Mixed").sum()),
        "vote": "hierarchical_vote (min_purity=0.5, lead_ratio=1.5)",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\ntally under the shipped vote:")
    for lab, n in table["label"].value_counts().items():
        print(f"  {lab:24s} {n}")
    print(
        f"\npooled purity across {summary['n_clusters']} clusters: "
        f"min {summary['min']:.3f}  max {summary['max']:.3f}  "
        f"median {summary['median']:.3f}; "
        f"{summary['n_undecided']} undecided at both levels"
    )

    # ---- the single-floor variant, for comparison only --------------------
    cv = cluster_vote(
        a, cluster_col="leiden", label_col="tangram_cell_type", pool=True, dst="_cv"
    )
    cv.to_csv(OUT / "purity_by_cluster_clustervote.csv")
    print("\nsingle-floor cluster_vote, for comparison only:")
    for lab, n in cv["voted"].value_counts().items():
        print(f"  {lab:24s} {n}")

    print(f"\nwrote {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
