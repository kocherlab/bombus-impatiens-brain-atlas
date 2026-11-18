"""Label spatial clusters at canonical-17 without a giant Mixed block.

A flat 0.5 purity floor is a much stricter bar for a 17-way vocabulary
(chance 1/17) than for the 7-way pool (chance 1/7), and it cannot tell a
clear winner under half from a two-way tie: a cluster whose top label is just
under half with a distant runner-up, and one whose top two labels are nearly
level, are both Mixed under the flat floor. The rule here is hierarchical:

  1. Fine label if decided: purity >= 0.50, or a 1.5x lead over the
     runner-up (the multiplier separates the two cases above).
  2. Otherwise the pooled label, if decided at >= 0.50 (e.g. a cluster split
     between cortex and subperineurial glia is labelled glia).
  3. Otherwise Mixed.

Writes the per-cluster decision table and the leiden x fine x region
contingency, so other rules can be evaluated without reloading the data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from bimp_st.cluster_labels import (  # noqa: E402
    add_canonical,
    get_pooled,
)
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.nine_region_common import (  # noqa: E402
    STATISTICAL_REGIONS,
    attach_region_flags,
    display,
    region_of,
)
from bimp_st.paths import CANONICAL_TANGRAM, SPATIAL_COMBINED  # noqa: E402

OUT = REPO / "results" / "reframe" / "celltype_anatomy"
PURITY = 0.50
LEAD = 1.5


def decide(vc_fine: pd.Series, vc_pool: pd.Series) -> tuple[str, str]:
    """Return (label, why) for one cluster."""
    top, top_f = vc_fine.index[0], float(vc_fine.iloc[0])
    run_f = float(vc_fine.iloc[1]) if len(vc_fine) > 1 else 0.0
    if top_f >= PURITY:
        return top, f"fine purity {top_f:.3f}"
    if run_f <= 0 or top_f >= LEAD * run_f:
        return top, f"fine lead {top_f / max(run_f, 1e-9):.2f}x"
    ptop, ptop_f = vc_pool.index[0], float(vc_pool.iloc[0])
    if ptop_f >= PURITY:
        return ptop, f"pooled purity {ptop_f:.3f} (fine was a tie)"
    return "Mixed", (
        f"fine {top_f:.3f}/{run_f:.3f} and pooled {ptop_f:.3f} both undecided"
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    print("loading...", flush=True)
    a = ad.read_h5ad(SPATIAL_COMBINED)
    a.obs["tangram_cell_type"] = (
        read_obs(CANONICAL_TANGRAM)["tangram_cell_type"].astype(str).values
    )
    attach_region_flags(a, regions=STATISTICAL_REGIONS)
    add_canonical(a, src="tangram_cell_type")

    lei = a.obs["leiden"].astype(str).values
    fine = a.obs["cell_type_canonical"].astype(str).values
    pool = np.array([get_pooled(x) for x in fine])

    label_of, rows = {}, []
    for c in np.unique(lei):
        m = lei == c
        vf = pd.Series(fine[m]).value_counts(normalize=True)
        vp = pd.Series(pool[m]).value_counts(normalize=True)
        lab, why = decide(vf, vp)
        label_of[c] = lab
        rows.append(
            {
                "leiden": c,
                "n": int(m.sum()),
                "label": lab,
                "fine_top": vf.index[0],
                "fine_purity": float(vf.iloc[0]),
                "fine_runner": vf.index[1] if len(vf) > 1 else "",
                "fine_runner_frac": float(vf.iloc[1]) if len(vf) > 1 else 0.0,
                "why": why,
            }
        )
    d = pd.DataFrame(rows).sort_values("n", ascending=False)
    pd.set_option("display.width", 210)
    print("\n=== per-cluster decision ===")
    print(d.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    d.to_csv(OUT / "fine_vote_decisions.csv", index=False)

    lab = np.array([label_of[c] for c in lei])
    a.obs["_fine"] = lab

    reg = region_of(a.obs, STATISTICAL_REGIONS)
    one = np.isin(reg, STATISTICAL_REGIONS)
    region = np.array([display(r) for r in reg[one]])
    tab = pd.crosstab(
        pd.Series(lab[one], name="cell type"), pd.Series(region, name="region")
    )
    tab.to_csv(OUT / "counts_fine_vote_hierarchical.csv")

    # full contingency, so any future rule costs no reload
    pd.crosstab(
        [pd.Series(lei[one], name="leiden"), pd.Series(fine[one], name="fine")],
        pd.Series(region, name="region"),
    ).to_csv(OUT / "contingency_leiden_fine_region.csv")

    n = tab.sum(1)
    print(f"\n=== labels inside traced tissue ({int(n.sum()):,} cells) ===")
    print(
        pd.DataFrame({"cells": n, "pct": (n / n.sum() * 100).round(2)})
        .sort_values("cells", ascending=False)
        .to_string()
    )
    mixed = n.get("Mixed", 0) / n.sum()
    print(
        f"\nMixed is {mixed:.1%} of traced cells "
        f"(was 38.8% under the flat 50% floor, 3.3% pooled)"
    )
    print(f"distinct labels: {(n > 0).sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
