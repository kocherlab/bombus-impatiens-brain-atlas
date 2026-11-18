"""Caste difference relative to within-brain section-to-section variation.

Computes, per pooled cell class, the ratio of the queen-worker difference in
section proportions to the mean within-brain SD.

Uses the per-section proportions fig:caste_comparison plots
(results/figures/fig2/slice_composition.csv, from
scripts/regions/caste_comparison_sccoda.py):
per section, each pooled class's share of that section's cells, with
Unannotated retained, matching fig_caste_comparison.py's panel-B transform.

ratio(class) = |mean(queen) - mean(worker)| / mean(within-brain SD),
where within-brain SD is the std across that brain's own sections and the two
brains' SDs are averaged. There is no pooled SD across brains, because each
brain is one caste.

Read-only; writes results/reframe/caste_sd_ratio/sd_ratios.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "results" / "figures" / "fig2" / "slice_composition.csv"
OUT = REPO / "results" / "reframe" / "caste_sd_ratio"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(SRC)
    classes = sorted(d["cluster"].unique())
    # Panel B's transform: renormalise each section's proportions over the
    # classes plotted (all of them here, Unannotated included).
    d = d[d["cluster"].isin(classes)].copy()
    d["prop"] = d.groupby("slice_name")["prop"].transform(lambda s: s / s.sum())

    rows = []
    for c in classes:
        sub = d[d["cluster"] == c]
        by_brain = sub.groupby("brain")["prop"]
        means = by_brain.mean()
        sds = by_brain.std(ddof=1)
        if not {"queen", "worker"} <= set(means.index):
            continue
        diff = abs(means["queen"] - means["worker"])
        mean_sd = sds.reindex(["queen", "worker"]).mean()
        rows.append(
            {
                "class": c,
                "n_queen_sections": int((sub.brain == "queen").sum()),
                "n_worker_sections": int((sub.brain == "worker").sum()),
                "mean_queen": means["queen"],
                "mean_worker": means["worker"],
                "caste_diff": diff,
                "sd_queen": sds.get("queen", np.nan),
                "sd_worker": sds.get("worker", np.nan),
                "mean_within_brain_sd": mean_sd,
                "ratio": diff / mean_sd if mean_sd > 0 else np.nan,
            }
        )
    out = pd.DataFrame(rows).sort_values("ratio")
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    out.to_csv(OUT / "sd_ratios.csv", index=False)
    print(f"\nwrote {OUT}/sd_ratios.csv")
    print(
        f"\ncredible-per-scCODA class (projection neurons) ratio: "
        f"{out.set_index('class').loc['Projection neurons', 'ratio']:.3f}"
        if "Projection neurons" in out["class"].values
        else ""
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
