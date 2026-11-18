#!/usr/bin/env python
"""What the cluster vote labels "Other", which types it never names, and how often it
differs from the per-cell label.

From the canonical Tangram object's per-cell ``tangram_cell_type`` and the expression
Leiden clusters:

1. The expression-only ``hierarchical_vote`` (fine name at purity >= 0.5 or a 1.5x lead,
   else the pooled group at purity >= 0.5, else "Mixed", displayed as *Other*). "Other"
   is therefore the share of all cells in clusters decided at neither the fine nor the
   pooled tier. Also the labels the vote ever emits and, for the groups it never names
   (projection neurons, other neurons, photoreceptors, hemocytes), their per-cell
   counts.
2. Per-cell vs vote agreement: per-cell argmax pooled to the display groups against the
   vote pooled the same way (a cell in an "Other" cluster always counts as differing),
   on the representative section of the SI figure and over every section.

    PYTHONPATH=src:scripts .venv/bin/python scripts/identity/cluster_vote_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from bimp_st.cluster_labels import get_canonical, get_pooled, hierarchical_vote  # noqa: E402
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.paths import CANONICAL_TANGRAM  # noqa: E402

OUT = REPO / "results/round6/vote_stats"
SECTION = "output-XETG00151__0040243__Region_6__20251204__195015_2_6B"
MIXED = "Mixed"
NEVER_VOTED = [
    "Anterodorsal projection neurons",
    "Lateral projection neurons",
    "Other neurons",
    "Photoreceptors",
    "Hemocytes",
]


class _Shim:
    def __init__(self, obs: pd.DataFrame):
        self.obs, self.n_obs = obs, len(obs)


def vote_table(obs: pd.DataFrame, label_col: str = "tangram_cell_type"):
    tab = hierarchical_vote(
        _Shim(obs), cluster_col="leiden", label_col=label_col, dst="cluster_vote"
    )
    return tab


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    obs = read_obs(CANONICAL_TANGRAM)
    obs = obs[["leiden", "tangram_cell_type", "slice_name", "brain"]].copy()
    n = len(obs)
    tab = vote_table(obs)
    obs["vote_fine"] = obs["cluster_vote"].astype(str)
    obs["vote"] = [v if v == MIXED else get_pooled(v) for v in obs["vote_fine"]]
    obs["raw_canonical"] = [
        get_canonical(v) for v in obs["tangram_cell_type"].astype(str)
    ]
    obs["raw_pooled"] = [get_pooled(v) for v in obs["raw_canonical"]]

    rows = []
    # ---- 1. Other
    vc = obs["vote_fine"].value_counts()
    other = int((obs["vote_fine"] == MIXED).sum())
    rows += [
        ("cells", n),
        ("leiden clusters", obs["leiden"].nunique()),
        ("Other (Mixed) cells", other),
        ("Other share of all cells, %", 100 * other / n),
        ("clusters decided Other", int((tab["label"] == MIXED).sum())),
    ]
    tab = tab.reset_index()
    print(tab.round(3).to_string(index=False), flush=True)
    tab.to_csv(OUT / "cluster_decision_table.csv", index=False)
    vote_counts = vc.rename_axis("vote_label").reset_index(name="n_cells")
    vote_counts["pct"] = 100 * vote_counts.n_cells / n
    vote_counts["tier"] = [
        "Other (neither tier)"
        if v == MIXED
        else (
            "pooled group"
            if v in ("Kenyon cells", "Glia", "Optic lobe")
            else "fine type"
        )
        for v in vote_counts.vote_label
    ]
    vote_counts.to_csv(OUT / "vote_label_counts.csv", index=False)
    print(vote_counts.round(2).to_string(index=False), flush=True)
    pooled = (
        obs["vote"].value_counts().rename_axis("vote_group").reset_index(name="n_cells")
    )
    pooled["pct"] = 100 * pooled.n_cells / n
    pooled.to_csv(OUT / "vote_group_counts.csv", index=False)
    print(pooled.round(2).to_string(index=False), flush=True)
    named = set(obs["vote_fine"].unique()) | set(obs["vote"].unique())
    pc = obs["raw_canonical"].value_counts()
    nv = pd.DataFrame(
        {
            "per_cell_label": NEVER_VOTED,
            "per_cell_cells": [int(pc.get(k, 0)) for k in NEVER_VOTED],
            "named_by_any_cluster_vote": [
                (k in named) or (get_pooled(k) in named) for k in NEVER_VOTED
            ],
        }
    )
    nv["pct_of_all_cells"] = 100 * nv.per_cell_cells / n
    pooled_pc = obs["raw_pooled"].value_counts()
    for g in ("Projection neurons", "Other neurons", "Photoreceptors", "Hemocytes"):
        rows.append((f"per-cell cells, pooled group {g}", int(pooled_pc.get(g, 0))))
        rows.append((f"group {g} named by any cluster vote", g in named))
    nv.to_csv(OUT / "never_voted_groups.csv", index=False)
    obs["raw_canonical"].value_counts().rename_axis("canonical_type").reset_index(
        name="n_cells"
    ).assign(pct=lambda d: 100 * d.n_cells / n).to_csv(
        OUT / "per_cell_canonical17_counts.csv", index=False
    )
    print(nv.round(3).to_string(index=False), flush=True)

    # ---- 2. agreement
    differs = (obs["raw_pooled"] != obs["vote"]).to_numpy()
    sec = (obs["slice_name"].astype(str) == SECTION).to_numpy()
    rows += [
        ("representative section", SECTION),
        ("section cells", int(sec.sum())),
        ("section agreement, %", 100 * (1 - differs[sec].mean())),
        ("section differing, %", 100 * differs[sec].mean()),
        ("all-cell agreement, %", 100 * (1 - differs.mean())),
        ("all-cell differing, %", 100 * differs.mean()),
    ]
    called = (obs["vote"] != MIXED).to_numpy()
    rows += [
        (
            "all-cell differing where the vote names a type, %",
            100 * differs[called].mean(),
        ),
        (
            "section differing where the vote names a type, %",
            100 * differs[sec & called].mean(),
        ),
    ]
    per = (
        pd.DataFrame(
            {
                "section": obs["slice_name"].astype(str),
                "brain": obs["brain"].astype(str),
                "agree": ~differs,
            }
        )
        .groupby(["brain", "section"])
        .agg(n_cells=("agree", "size"), agreement=("agree", "mean"))
        .reset_index()
    )
    per.to_csv(OUT / "per_section_agreement.csv", index=False)
    s = per.agreement
    rows += [
        ("per-section agreement median, %", 100 * s.median()),
        ("per-section agreement IQR low, %", 100 * s.quantile(0.25)),
        ("per-section agreement IQR high, %", 100 * s.quantile(0.75)),
        ("per-section agreement min, %", 100 * s.min()),
        ("per-section agreement max, %", 100 * s.max()),
        ("sections", len(per)),
    ]
    for b, g in per.groupby("brain"):
        rows.append(
            (
                f"agreement over {b} cells, %",
                100 * (g.agreement * g.n_cells).sum() / g.n_cells.sum(),
            )
        )
    df = pd.DataFrame(rows, columns=["quantity", "value"])
    df.to_csv(OUT / "vote_stats.csv", index=False)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False), flush=True)
    obs[
        [
            "slice_name",
            "brain",
            "leiden",
            "raw_canonical",
            "raw_pooled",
            "vote_fine",
            "vote",
        ]
    ].to_parquet(OUT / "per_cell_labels_canonical.parquet")


if __name__ == "__main__":
    main()
