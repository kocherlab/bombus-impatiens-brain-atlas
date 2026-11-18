#!/usr/bin/env python
"""SI figure and table: five-population Kenyon labels in the mushroom-body compartments
and the optic neuropils.

Reads ``mb_ol_calls.parquet`` in ``bimp_st.paths.MB_OL_CALLS_DIR``
(``project_kc_populations_with_optic.py``: the restricted five-population Tangram
reclassification with the query widened to the Kenyon compartments plus lamina, lobula
and medulla). Reports cells and label shares per region, median top
posterior, and Cramer's V of region x label for the Kenyon compartments, the optic
neuropils and all six, with a cell-level bootstrap interval and the value per caste.

Cramer's V = sqrt(chi2 / (n * min(r - 1, c - 1))) from the region x label contingency
table, chi2 without continuity correction (``scipy.stats.contingency.association(...,
method="cramer")``); r = 3 regions, c = 5 labels, so min(r-1, c-1) = 2 for each family
and 5 for the six-region table.

Writes ``results/round6/mb_ol/``: ``mb_ol_label_shares.csv``, ``mb_ol_cramers_v.csv``,
``fig_s_mb_ol_labels.{png,pdf}``, ``mb_ol_labels.tex`` (the registered
table ``tables/mb_ol_labels.tex``).

    PYTHONPATH=src:scripts .venv/bin/python scripts/kenyon/table_mb_ol_kc_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats.contingency import association  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from bimp_st.cluster_labels import get_color  # noqa: E402
from bimp_st.paths import MB_OL_CALLS_DIR  # noqa: E402
from bimp_st.si_common import set_style  # noqa: E402

IN = MB_OL_CALLS_DIR / "mb_ol_calls.parquet"
OUT = REPO / "results/round6/mb_ol"
REGIONS = ["KC-IC", "KC-NC", "KC-OC", "Lamina", "Lobula", "Medulla"]
# Row labels for the figure and the .tex table. REGIONS (above) are the
# parquet's own region values and the `fam` keys below land in
# mb_ol_cramers_v.csv, so those stay as they are; only the display replaces
# the KC-IC/KC-NC/KC-OC codes with compartment names.
DISP = {
    "KC-IC": "Small-type compartment",
    "KC-NC": "Large-type compartment",
    "KC-OC": "Class II compartment",
    "Lamina": "Lamina",
    "Lobula": "Lobula",
    "Medulla": "Medulla",
}
LABELS = ["Large", "Middle", "Small", "ClassII", "Other"]
NAMES = {
    "Large": "Large-type",
    "Middle": "Middle-type",
    "Small": "Small-type",
    "ClassII": "Class II",
    "Other": "Other KC",
}
FULL = {
    "Large": "Large-type Kenyon cells",
    "Middle": "Middle-type Kenyon cells",
    "Small": "Small-type Kenyon cells",
    "ClassII": "Class II Kenyon cells",
    "Other": "Other Kenyon cells",
}


def cramers_v(df: pd.DataFrame, regions: list[str]) -> float:
    m = df.region.isin(regions).to_numpy()
    ct = pd.crosstab(
        df.region.to_numpy()[m], df.call.to_numpy()[m]
    )  # arrays, so a resampled frame's repeated index cannot misalign
    return float(association(ct.values, method="cramer"))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(IN).reset_index(
        drop=True
    )  # cell names repeat across brains; align by position
    n = len(d)
    ct = pd.crosstab(d.region, d.call).reindex(REGIONS)[LABELS]
    pct = ct.div(ct.sum(1), axis=0) * 100
    conf = (
        d.groupby("region")
        .maxprob.agg(median="median", frac_below_0p5=lambda x: (x < 0.5).mean())
        .reindex(REGIONS)
    )
    share = pct.round(2).join(conf).assign(n_cells=ct.sum(1).astype(int))
    share.to_csv(OUT / "mb_ol_label_shares.csv")
    print(share.round(3).to_string(), flush=True)

    fam = {
        "Kenyon compartments (KC-IC, KC-NC, KC-OC)": REGIONS[:3],
        "optic neuropils (lamina, lobula, medulla)": REGIONS[3:],
        "all six regions": REGIONS,
    }
    rng = np.random.default_rng(0)
    rows = []
    for name, regs in fam.items():
        sub = d[d.region.isin(regs)].reset_index(drop=True)
        boots = []
        for _ in range(100):
            idx = rng.integers(0, len(sub), len(sub))
            boots.append(cramers_v(sub.iloc[idx], regs))
        r = dict(
            family=name,
            n_cells=len(sub),
            cramers_v=cramers_v(d, regs),
            boot_lo=np.percentile(boots, 2.5),
            boot_hi=np.percentile(boots, 97.5),
            median_top_posterior=float(sub.maxprob.median()),
        )
        for b in ("queen", "worker"):
            r[f"cramers_v_{b}"] = cramers_v(d[d.brain == b], regs)
        rows.append(r)
    V = pd.DataFrame(rows)
    V.to_csv(OUT / "mb_ol_cramers_v.csv", index=False)
    print(V.round(4).to_string(index=False), flush=True)

    # ---- figure
    set_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    y = np.arange(len(REGIONS))[::-1].astype(float)
    y[3:] -= 0.6  # a gap between the Kenyon compartments and the optic neuropils
    left = np.zeros(len(REGIONS))
    for lab in LABELS:
        ax.barh(
            y,
            pct[lab].values,
            left=left,
            color=get_color(FULL[lab]),
            edgecolor="white",
            linewidth=0.5,
            label=NAMES[lab],
            height=0.72,
        )
        for yi, lft, v in zip(y, left, pct[lab].values, strict=False):
            if v >= 8:
                ax.text(
                    lft + v / 2,
                    yi,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white",
                    fontweight="bold",
                )
        left += pct[lab].values
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{DISP[r]}\n(n = {int(ct.loc[r].sum()):,})" for r in REGIONS], fontsize=7.5
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Cells called each Kenyon population (%)")
    vk, vo = V.cramers_v.iloc[0], V.cramers_v.iloc[1]
    ax.text(101.5, y[1], f"Cramér's V\n{vk:.3f}", ha="left", va="center", fontsize=7.5)
    ax.text(101.5, y[4], f"Cramér's V\n{vo:.3f}", ha="left", va="center", fontsize=7.5)
    ax.legend(
        ncol=5,
        frameon=False,
        fontsize=7.5,
        loc="upper center",
        bbox_to_anchor=(0.45, 1.13),
        handlelength=1.0,
        columnspacing=1.0,
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.30, right=0.85, top=0.88, bottom=0.14)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig_s_mb_ol_labels.{ext}", dpi=300, facecolor="white")

    # ---- LaTeX table body
    def f(v: float) -> str:
        return f"{v:.1f}"

    body = [
        "% Table body. Needs \\usepackage{booktabs,array} in the preamble.",
        "% Generated by scripts/kenyon/table_mb_ol_kc_labels.py -- do not edit by "
        "hand.",
        "\\begin{table}[!h]",
        "\\centering",
        "{\\footnotesize\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{@{}l r r r r r r r@{}}",
        "\\toprule",
        "Region & Cells & Large-type & Middle-type & Small-type & Class II & Other & "
        "Median top posterior\\\\",
        "\\midrule",
    ]
    for i, r in enumerate(REGIONS):
        if i == 3:
            body.append("\\midrule")
        body.append(
            f"{DISP[r].replace('(', '(').replace(')', ')')} & {int(ct.loc[r].sum()):,} "
            f"& "
            + " & ".join(f(pct.loc[r, lab]) for lab in LABELS)
            + f" & {conf.loc[r, 'median']:.2f}\\\\"
        )
    body += [
        "\\bottomrule",
        "\\end{tabular}}",
        "\\caption{\\textbf{Five-population Kenyon cell labels in the Kenyon "
        "compartments and the optic neuropils.} "
        "The restricted reclassification (five central brain Kenyon populations as the "
        "reference, 32{,}672 nuclei) was run with the "
        f"query widened to the three Kenyon compartments plus lamina, lobula and "
        f"medulla ({n:,} cells). Entries are the percentage of "
        "each region's cells called each population. Cram\\'er's $V$ of region by "
        "label is "
        f"{vk:.3f} for the Kenyon compartments and {vo:.3f} for the optic neuropils "
        f"(cell-level bootstrap 95\\% intervals "
        f"{V.boot_lo.iloc[0]:.3f}--{V.boot_hi.iloc[0]:.3f} and "
        f"{V.boot_lo.iloc[1]:.3f}--{V.boot_hi.iloc[1]:.3f}). "
        "One brain per caste: descriptive.}",
        "\\label{tab:mb_ol_labels}",
        "\\end{table}",
    ]
    (OUT / "mb_ol_labels.tex").write_text("\n".join(body) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
