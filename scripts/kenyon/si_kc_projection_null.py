"""SI figure — restricting the projection to the Kenyon compartments recovers
coarse Kenyon-cell identity cleanly, but not fine morphological subtype.

Supports building the spatial panels of `fig:kc_subtypes` on unsupervised
transcriptomic clusters rather than on projected KC subtype labels: restricting
the projection to cells already inside KC-IC / KC-NC / KC-OC does not by itself
make the per-cell subtype calls anatomically specific.

Provenance: reads the tables scripts/kenyon/restricted_projection_numbers.py writes
from the canonical Tangram run (`bimp_st.paths.CANONICAL_TANGRAM`), whose
posterior columns are named by cell-type identity; the column lookups here are
by name.

Three panels, in the order the argument runs:
  A  Share of each cell's Tangram probability mass on Kenyon-cell reference
     columns, inside the three KC compartments vs optic / unannotated tissue
     (class-level identity).
  B  Four-way restricted composition for the three KC compartments and the
     negative controls. The generic "Kenyon cells" + "Other Kenyon cells"
     columns are excluded by construction, and control rows renormalise a
     small share of the mass (panel A), so they are context, not a precise
     composition.
  C  Literature expectation: KC-NC large-type, KC-IC small-/middle-type, KC-OC
     class II (Kubo-lab size names). The within-KC test asks whether each type
     peaks in its expected compartment among the three; it never looks at the
     controls. The control-aware test asks whether the expected compartment
     also beats the best negative control, by argmax share and by mean
     probability.

Inputs (written by scripts/kenyon/restricted_projection_numbers.py into
results/fig5_kc/):
  kc_prob_mass_by_region.csv, morph_comp_by_region.csv,
  morph_meanprob_by_region.csv, morph_expectation_check.csv,
  morph_share_of_kc_mass.csv

Run:
  PYTHONPATH=src:scripts .venv/bin/python scripts/kenyon/si_kc_projection_null.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bimp_st.nine_region_common import KC_POPULATION_COLORS
from bimp_st.paths import REPO  # noqa: E402
from bimp_st.print_scale import PRINTED_WIDTH_FRAC
from bimp_st.si_common import add_panel_headers
from bimp_st.style import set_pub_style  # noqa: E402

DATA = REPO / "results" / "fig5_kc"
OUT = Path("results/figures/si")
OUT.mkdir(parents=True, exist_ok=True)

# Kenyon compartments vs everything used as a negative control.  Displayed
# names; the CSVs are keyed on the lowercase region ids.
REGIONS = ["kc_ic", "kc_nc", "kc_oc", "medulla", "lamina", "lobula", "none"]
RNAME = {
    "kc_ic": "Small-type",
    "kc_nc": "Large-type",
    "kc_oc": "Class II",
    "medulla": "Medulla",
    "lamina": "Lamina",
    "lobula": "Lobula",
    "none": "Unannotated",
}
IS_KC = {r: r.startswith("kc_") for r in REGIONS}
CONTROLS = [r for r in REGIONS if not IS_KC[r]]
# Kenyon compartments in the slate-violet of `fig:kc_subtypes`; controls in a
# neutral grey so the eye separates "the tissue in question" from "the tissue
# that should not contain Kenyon subtypes" before reading any number.
KC_C, CTRL_C = "#6E5A85", "#B0AAA2"

# The Kenyon population palette of Figs 1 and 5 (fig_kc_subtypes.TYPE_COLORS
# is this same dict), so a colour means one population across figures:
# Large-type orange, Small-type mauve, Class II yellow, Middle-type brown.
TYPES = ["Large", "Middle", "Small", "Class II"]
TCOL = {
    t: KC_POPULATION_COLORS[f"{t}{'' if t == 'Class II' else '-type'} Kenyon cells"]
    for t in TYPES
}

# Shared house style (not a local rcParams copy), so sizes and bold subpanel
# titles match the neighbouring figures.
set_pub_style()


def panel_a(ax):
    m = pd.read_csv(DATA / "kc_prob_mass_by_region.csv", index_col=0).reindex(REGIONS)
    x = np.arange(len(REGIONS))
    cols = [KC_C if IS_KC[r] else CTRL_C for r in REGIONS]
    ax.bar(x, m["mean"], color=cols, width=0.66, zorder=3)
    # Interquartile range as an error bar: the distribution is wide and skewed,
    # so a bare mean would overstate how cleanly the two groups separate.
    ax.vlines(x, m["25%"], m["75%"], color="#333333", lw=1.0, zorder=4)
    ax.set_xticks(x, [RNAME[r] for r in REGIONS], rotation=30, ha="right")
    ax.set_ylabel("Kenyon cell share of Tangram\nprobability mass")
    ax.set_ylim(0, max(m["75%"]) * 1.15)
    ax.axvline(2.5, color="#999999", lw=0.9, ls="--", zorder=1)
    # No group headers: the dashed divider separates the Kenyon compartments
    # (left) from the negative controls (right); the caption (A) says so.
    # No panel title: caption (A) gives the masses and the control
    # comparison this restated.
    return m


def panel_b(ax):
    c = pd.read_csv(DATA / "morph_comp_by_region.csv", index_col=0).reindex(REGIONS)
    kcmass = pd.read_csv(DATA / "kc_prob_mass_by_region.csv", index_col=0).reindex(
        REGIONS
    )
    y = np.arange(len(REGIONS))[::-1]
    left = np.zeros(len(REGIONS))
    for t in TYPES:
        ax.barh(
            y, c[t].values, left=left, color=TCOL[t], height=0.68, label=t, zorder=3
        )
        left += c[t].values
    ax.set_yticks(y, [RNAME[r] for r in REGIONS])
    ax.set_xlabel("% of region, after restricting to the 4 named types")
    ax.set_xlim(0, 100)
    ax.axhline(y[2] - 0.5, color="#666666", lw=0.9, ls="--", zorder=5)
    # Annotate each negative-control row with its panel-A Kenyon-cell mass,
    # centred in the Class II segment (the widest one in every control row,
    # so there is room) rather than outside the axes: these rows are built by
    # renormalising a small share of each cell's probability mass, so the
    # composition is context, not a precise reading -- unlike the KC
    # compartments, where much more of the mass is on Kenyon columns.
    for r in ("medulla", "lamina", "lobula", "none"):
        i = REGIONS.index(r)
        x0 = c.loc[r, "Large"] + c.loc[r, "Middle"] + c.loc[r, "Small"]
        ax.text(
            x0 + c.loc[r, "Class II"] / 2,
            y[i],
            f"KC mass\n{kcmass.loc[r, 'mean'] * 100:.1f}%",
            va="center",
            # Dark text: the Class II segment is the light yellow of the
            # shared palette, where white would not read.
            ha="center",
            fontsize=6.0,
            color="#222222",
            fontweight="bold",
            linespacing=0.95,
            zorder=6,
        )
    # One legend for B and C (same four types, same colours).
    ax.legend(
        ncol=4,
        fontsize=6.5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        handlelength=1.2,
        columnspacing=1.1,
        handletextpad=0.4,
    )
    # No panel title: caption (B) says this almost word for word.
    return c


def panel_c(ax):
    c = pd.read_csv(DATA / "morph_comp_by_region.csv", index_col=0)
    ex = pd.read_csv(DATA / "morph_expectation_check.csv").set_index("type")
    kc3 = ["kc_ic", "kc_nc", "kc_oc"]
    x = np.arange(len(kc3))
    w = 0.2
    for i, t in enumerate(TYPES):
        v = c.loc[kc3, t].values
        ax.bar(x + (i - 1.5) * w, v, w, color=TCOL[t], label=t, zorder=3)
        j = kc3.index(ex.loc[t, "expected_peak"])
        # Ring the bar the literature predicts should be tallest. That
        # 3-compartment test never looks outside the mushroom body, so the
        # check mark below comes from the control-aware test (does the expected
        # bar also beat the best negative control, as in panel A).
        ax.bar(
            x[j] + (i - 1.5) * w,
            v[j],
            w,
            fill=False,
            edgecolor="#111111",
            lw=1.1,
            zorder=5,
        )
        # Highest of ALL four negative controls (medulla, lamina, lobula,
        # unannotated), as panel B shows them and the caption says.
        # morph_expectation_check.csv adjudicates against medulla and
        # unannotated only, which labelled Middle-type's ceiling "Medulla 41%"
        # while panel B's lobula row holds 44%.
        neg_v = float(c.loc[CONTROLS, t].max())
        neg_r = c.loc[CONTROLS, t].idxmax()
        ok = bool(v[j] > neg_v)
        ax.text(
            x[j] + (i - 1.5) * w,
            v[j] + 0.9,
            "✓" if ok else "✗",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1A7F37" if ok else "#B3261E",
            fontweight="bold",
        )
        # Best-negative-control level for this type, as a dashed ceiling
        # spanning its three bars -- makes the control-aware comparison
        # visible instead of implicit.
        # Only across the expected-peak bar itself: a ceiling spanning all
        # three compartments ran through the neighbouring Large bars and the
        # 41% / 38% lines read as one.
        x0, x1 = x[j] + (i - 1.5) * w - w / 2, x[j] + (i - 1.5) * w + w / 2
        ax.hlines(neg_v, x0, x1, color="#555555", lw=0.9, ls=(0, (3, 2)), zorder=6)
        if not ok:
            # Failing case (Class II): the ceiling sits above the expected
            # bar -- label which negative-control region beats it, since an
            # X alone does not say by how much or against what.
            # Right of the short line, level with it: never above the bar
            # (where the check/cross mark sits) and never over the tall Large
            # bar on its left. For the last compartment this runs past the
            # axis end, into the margin.
            ax.text(
                x1 + 0.03,
                neg_v,
                f"{RNAME[neg_r]}\n{neg_v:.0f}%",
                ha="left",
                va="center",
                fontsize=6.0,
                color="#B3261E",
                linespacing=0.95,
                clip_on=False,
            )
    ax.set_xticks(x, [RNAME[r] for r in kc3])
    ax.set_ylabel("% of compartment")
    top = max(c.loc[kc3, TYPES].values.max(), c.loc[CONTROLS, TYPES].values.max())
    ax.set_ylim(0, top * 1.30)
    n_pass = int(ex["control_argmax_pass"].sum())
    # No panel title: a legend gloss, and caption (C) carries the
    # comparison and its numbers.
    _ = n_pass  # still reported to the run log below
    # No legend: panel B's, directly beside it, keys the same four colours.
    return ex


def main():
    # 7.2 in wide (the printed width), so text is authored near printed size.
    plt.rcParams.update(
        {
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )
    fig = plt.figure(figsize=(7.2, 2.7))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.2, 1.2],
        wspace=0.45,
        left=0.08,
        right=0.985,
        top=0.88,
        bottom=0.24,
    )
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    m = panel_a(axA)
    panel_b(axB)
    ex = panel_c(axC)

    add_panel_headers(
        fig,
        [
            (axA, "A", "Kenyon cell share"),
            (axB, "B", "Type composition"),
            (axC, "C", "Expected peaks"),
        ],
        width_frac=PRINTED_WIDTH_FRAC["figS_kc_projection_null"],
    )

    out = OUT / "figS_kc_projection_null.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(OUT / "figS_kc_projection_null.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    print(
        f"  KC mass: compartments {m.loc['kc_oc', 'mean']:.2f}-"
        f"{m.loc['kc_nc', 'mean']:.2f} vs controls "
        f"{m.loc['lamina', 'mean']:.3f}-{m.loc['medulla', 'mean']:.3f}"
    )
    print(
        f"  3-compartment-only peak check: {int(ex['argmax_pass'].sum())}/4 argmax, "
        f"{int(ex['meanprob_pass'].sum())}/4 mean-probability"
    )
    print(
        f"  control-aware check (beats best negative control too): "
        f"{int(ex['control_argmax_pass'].sum())}/4 argmax, "
        f"{int(ex['control_meanprob_pass'].sum())}/4 mean-probability"
    )


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
