"""Does the snRNA -> spatial cell-type transfer work? Show that transferred
cell types land in the anatomy they should.

The comparison is non-circular only because the labels carry no anatomy.
Labels come from ``cluster_vote`` (per-spatial-cluster majority of the Tangram
call, expression only); regions come from the hand-drawn GeoJSON tracings.

Do not redraw this with ``anatomy_guided_vote``: it forces clusters >50%
inside calyx or optic anatomy to "Kenyon cells" / "Optic lobe" using the same
``in_*`` flags that form the region axis, so the two biggest ribbons would be
true by construction. That version is computed below only as a contrast.

Two views of the same table:

  left  -- row-normalised: of each cell type's cells, the fraction in each
           region ("87% of Kenyon cells are in a Kenyon compartment"); a
           widely spread type still looks concentrated in the biggest region.
  right -- enrichment: observed / expected, where expected is the type's
           share of all region-assigned cells times the region's size
           (1.0 = base rate).

Only cells inside exactly one traced region are counted; cells in no region
(most of the atlas) and the rare multi-region cells are dropped, and the
counts are printed.

Writes results/reframe/celltype_anatomy/.
"""

from __future__ import annotations

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports

import json
import sys
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bimp_st.cluster_labels import (  # noqa: E402
    add_canonical,
    anatomy_guided_vote,
    cluster_vote,
    get_canonical_color,
    get_pooled,
    hierarchical_vote,
)
from bimp_st.paths import CANONICAL_TANGRAM, SPATIAL_COMBINED  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from identity.celltype_region_sankey import _ribbon  # noqa: E402

from bimp_st.io import read_obs  # noqa: E402
from bimp_st.nine_region_common import (  # noqa: E402
    REGION_COLORS,
    STATISTICAL_REGIONS,
    attach_region_flags,
    display,
    display_region,
    region_of,
)
from bimp_st.si_common import set_style  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "reframe" / "celltype_anatomy"

# Where each pooled type is expected to land, for reading the figure. Not used
# in any computation.
EXPECTED = {
    "Kenyon cells": ("Small-type", "Large-type", "Class II"),
    "Optic lobe": ("Medulla", "Lamina", "Lobula"),
    # Projection neurons are not listed: the antennal lobe is their top-enriched
    # region, but this figure sees only a small fraction of their cells,
    # selected by falling inside a neuropil polygon, which selects against somata.
    # The heatmap still shows the value.
}

TYPE_ORDER = [
    "Kenyon cells",
    "Optic lobe",
    "Projection neurons",
    "Lateral projection neurons",
    "Anterodorsal projection neurons",
    "Other neurons",
    "Glia",
    "Astrocyte-like glia",
    "Cortex glia",
    "Perineurial glia",
    "Subperineurial glia",
    "Other glia",
    "Hemocytes",
    "Photoreceptors",
    "Mixed",
]
# Whole-atlas count of each pooled label under the vote, so the figure can state
# what share of each type it draws. Only cells inside a traced polygon are
# drawn, and that subset is not random: the polygons enclose neuropil, which
# selects against neuronal somata. Filled from the vote on every full run and
# cached in atlas_vote_counts.csv for --replot.
ATLAS_N: dict[str, int] = {}

REGION_ORDER = [
    "Small-type",
    "Large-type",
    "Class II",
    "Medulla",
    "Lamina",
    "Lobula",
    "Antennal lobe",
]


def _color(name):
    try:
        return get_canonical_color(name)
    except Exception:  # noqa: BLE001
        return "#9E9E9E"


def build_table(a, labels):
    reg = region_of(a.obs, STATISTICAL_REGIONS)
    one = np.isin(reg, STATISTICAL_REGIONS)
    region = np.array([display(r) for r in reg[one]])
    # regions a cell is known to be inside (an untraced flag is not "inside")
    n_in = sum(
        pd.array(a.obs[f"in_{r}"], dtype="boolean").to_numpy(
            dtype=float, na_value=np.nan
        )
        == 1
        for r in STATISTICAL_REGIONS
    )
    tab = pd.crosstab(
        pd.Series(labels[one], name="cell type"), pd.Series(region, name="region")
    )
    rows = [t for t in TYPE_ORDER if t in tab.index]
    rows += [t for t in tab.index if t not in rows]
    cols = [r for r in REGION_ORDER if r in tab.columns]
    tab = tab.reindex(index=rows, columns=cols, fill_value=0)
    return tab, int(one.sum()), int(n_in.max())


def size_bars(ax, counts, title, subtitle):
    """Size of each type on a log axis.

    Sankey ribbons cannot be log-scaled (a node's height is the sum of its
    ribbons), so the 699x range between Kenyon cells and projection neurons
    is shown here instead. Each row shows the whole-atlas size (pale) with
    the portion this figure draws (solid) on top.
    """
    names = [n for n in counts.index if n in ATLAS_N]
    y = np.arange(len(names))
    tot = np.array([ATLAS_N[n] for n in names], float)
    got = np.array([counts[n] for n in names], float)
    ax.barh(
        y,
        tot,
        color=[_color(n) for n in names],
        alpha=0.28,
        height=0.68,
        label="in the atlas",
    )
    ax.barh(
        y,
        got,
        color=[_color(n) for n in names],
        alpha=0.95,
        height=0.68,
        label="drawn here",
    )
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    for i, _n in enumerate(names):
        ax.text(
            tot[i] * 1.25,
            i,
            f"{got[i] / tot[i]:.0%} of {int(tot[i]):,}",
            va="center",
            fontsize=8.5,
            color="#333333",
        )
    ax.set_xlim(200, tot.max() * 9)
    ax.set_xlabel("cells (log scale)", fontsize=9)
    ax.grid(axis="x", alpha=0.25, which="both")
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=62)
    ax.text(
        0.0,
        1.005,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        va="bottom",
    )


def heatmap(ax, enr, title, subtitle):
    """Enrichment heatmap. Enrichment is a ratio, so it is not drawn as sankey
    ribbons, whose widths must be a conserved quantity."""
    v = enr.values
    im = ax.imshow(
        np.log2(np.maximum(v, 1e-3)), cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto"
    )
    ax.set_xticks(range(enr.shape[1]))
    ax.set_xticklabels(enr.columns, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(enr.shape[0]))
    ax.set_yticklabels(enr.index, fontsize=9)
    for i in range(enr.shape[0]):
        for j in range(enr.shape[1]):
            hit = enr.columns[j] in EXPECTED.get(enr.index[i], ())
            ax.text(
                j,
                i,
                f"{v[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold" if hit else "normal",
                color="white" if abs(np.log2(max(v[i, j], 1e-3))) > 1.8 else "#222222",
            )
            if hit:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#111111",
                        lw=2.0,
                    )
                )
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("log2 enrichment (0 = region's base rate)", fontsize=9)
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=62)
    ax.text(
        0.0,
        1.005,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        va="bottom",
    )


def draw(ax, mat, title, subtitle, counts=None, coverage=True):
    left, right = list(mat.index), list(mat.columns)
    total = mat.values.sum()
    if total <= 0:
        ax.axis("off")
        return
    gap = 0.02 * total / max(len(left), 1)
    lh, rh = mat.sum(1), mat.sum(0)
    ly, ry, y = {}, {}, 0.0
    for n in left:
        ly[n] = y
        y += lh[n] + gap
    lH = y - gap
    y = 0.0
    for n in right:
        ry[n] = y
        y += rh[n] + gap
    rH = y - gap
    for n in right:
        ry[n] += (lH - rH) / 2

    lcur, rcur = dict(ly), dict(ry)
    for s in left:
        exp = EXPECTED.get(s, ())
        for t in right:
            v = mat.loc[s, t]
            if v <= 0:
                continue
            hit = t in exp
            col = _color(s) if (hit or not exp) else "#BDBDBD"
            alpha = 0.85 if hit else (0.45 if not exp else 0.22)
            ax.add_patch(
                PathPatch(
                    _ribbon(0.02, lcur[s], lcur[s] + v, 0.98, rcur[t], rcur[t] + v),
                    facecolor=col,
                    alpha=alpha,
                    lw=0,
                )
            )
            lcur[s] += v
            rcur[t] += v

    for n in left:
        ax.add_patch(
            plt.Rectangle(
                (0.0, ly[n]),
                0.02,
                lh[n],
                facecolor=_color(n),
                edgecolor="white",
                lw=0.6,
            )
        )
        # Coverage is printed beside its node: only cells inside a traced polygon
        # are drawn, which is 91% of the Kenyon cells but 1% of the projection neurons.
        exp = EXPECTED.get(n, ())
        cols = [c for c in exp if c in mat.columns]
        lbl = n
        if lh[n] > 0:
            n_true = int(counts[n]) if counts is not None else int(lh[n])
            lbl += f"\n{n_true:,} cells"
            tot = ATLAS_N.get(n) if coverage else None
            if tot:
                cov = n_true / tot
                lbl += (
                    f"  ({cov:.0%} of this type"
                    + ("" if cov > 0.5 else ", a biased slice")
                    + ")"
                )
            if cols and tot and n_true / tot > 0.5:
                lbl += (
                    f"\n{mat.loc[n, cols].sum() / lh[n]:.1%} in "
                    f"{'/'.join(exp) if len(exp) < 2 else 'its compartments'}"
                )
        ax.text(
            -0.012,
            ly[n] + lh[n] / 2,
            lbl,
            ha="right",
            va="center",
            fontsize=8.5,
            linespacing=1.45,
        )
    for n in right:
        key = n.lower().replace(" ", "_").replace("-", "_")
        ax.add_patch(
            plt.Rectangle(
                (0.98, ry[n]),
                0.02,
                rh[n],
                facecolor=REGION_COLORS.get(key, "#999999"),
                edgecolor="white",
                lw=0.6,
            )
        )
        ax.text(
            1.012,
            ry[n] + rh[n] / 2,
            n,
            ha="left",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_xlim(-0.30, 1.30)
    ax.set_ylim(-total * 0.02, max(lH, rH) + total * 0.02)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=62)
    ax.text(
        0.0,
        1.005,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
        va="bottom",
    )


def _footer(rown, enr, colshare, n_one, counts, tab, atlas_n):
    """Footer text with both precision and recall.

    The sankey shows precision (where the cells carrying a label end up);
    recall is the fraction of each compartment's cells that carry the label.
    """
    out = []
    for name, expect in EXPECTED.items():
        if name not in rown.index:
            continue
        cols = [c for c in expect if c in rown.columns]
        if not cols:
            continue
        cov = counts[name] / atlas_n.get(name, counts[name])
        if cov <= 0.5:  # too small and too biased a slice to headline
            continue
        frac = rown.loc[name, cols].sum()
        share = colshare[cols].sum()
        rec = tab.loc[name, cols].sum() / tab[cols].values.sum()
        out.append(
            f"{name} -> {'/'.join(expect)}:   {frac:.1%} of its cells land "
            f"there ({frac / share:.2f}x enriched, of a possible "
            f"{1 / share:.2f}x)   |   and {rec:.1%} of those compartments' "
            f"cells carry the label."
        )
    out.append(
        f"n = {n_one:,} cells inside exactly one traced region. Labels "
        f"never saw a polygon; regions never saw expression."
    )
    out.append(
        "Only traced cells are drawn, and the polygons enclose "
        "neuropil, so a low-coverage row is a biased slice, not a "
        "small sample -- see each node's own percentage."
    )
    return "\n".join(out)


def main() -> int:
    set_style()
    OUT.mkdir(parents=True, exist_ok=True)

    # --replot re-renders from the cached tables (the compute is a 1.3M-cell
    # load + two cluster votes).
    # --fine draws the canonical-17 table as a separate figure. The fine vote
    # shows glial structure the pool hides (subperineurial glia are 88% of the
    # lamina, astrocyte-like glia 56% of the lobula) but sends 38% of traced cells
    # to Mixed, which weakens the Kenyon result; the two views answer different
    # questions.
    fine = OUT / "counts_fine_vote_hierarchical.csv"
    if "--fine" in sys.argv and fine.exists():
        tab17 = pd.read_csv(fine, index_col=0)
        tab17.columns = tab17.columns.map(display_region)  # cached 'KC-NC' names
        # This CSV comes from fine_vote_rule.py, so impose this figure's
        # orderings: regions in anatomical order, types in TYPE_ORDER.
        tab17 = tab17.reindex(
            index=(
                [t for t in TYPE_ORDER if t in tab17.index]
                + [t for t in tab17.index if t not in TYPE_ORDER]
            ),
            columns=[c for c in REGION_ORDER if c in tab17.columns],
            fill_value=0,
        )
        # Read Mixed's share before the 1% floor removes it.
        mixed_share = tab17.sum(1).get("Mixed", 0) / tab17.values.sum()
        # Same 1%-of-drawn-cells floor as the pooled panel: below it a node
        # is a hairline and its label collides with its neighbours'.
        keep17 = tab17.sum(1) / tab17.values.sum() >= 0.01
        drop17 = [n for n in tab17.index if not keep17[n]]
        tab17 = tab17[keep17.values].loc[keep17[keep17].index]
        expd = np.outer(tab17.sum(1), tab17.sum(0)) / tab17.values.sum()
        enr = pd.DataFrame(
            tab17.values / np.maximum(expd, 1e-9),
            index=tab17.index,
            columns=tab17.columns,
        )
        fig, axes = plt.subplots(
            1, 2, figsize=(21, 10), gridspec_kw={"width_ratios": [1.3, 1.0]}
        )
        draw(
            axes[0],
            tab17,
            "Same transfer, at the canonical-17 vocabulary",
            "cluster_vote over the fine labels, no pooling and no anatomy.\n"
            "Fine label where the vote is decided; the pooled label where\n"
            "the fine one ties; Mixed only where both are undecided\n"
            "(scripts/identity/fine_vote_rule.py). Ribbon width = cells.",
            coverage=False,
        )
        heatmap(
            axes[1],
            enr,
            "Enrichment at the fine vocabulary",
            "observed / expected. The pooled view's single Glia block\n"
            "splits four ways here, which is the point of drawing it.",
        )
        fig.subplots_adjust(top=0.86, bottom=0.13, wspace=0.42)
        mixed = mixed_share
        fig.text(
            0.012,
            0.015,
            "Subperineurial glia are 88% of lamina cells and "
            "astrocyte-like glia 56% of lobula cells; the medulla is "
            "the one optic neuropil the Optic lobe label wins.\n"
            f"Mixed is {mixed:.1%} of traced cells here, against 38.8% "
            f"under a flat 50% purity floor -- that floor was discarding "
            f"decided clusters, not cautious ones\n(cluster 13 is "
            f"Other neurons at 0.484 against Kenyon cells at 0.127, a "
            f'3.8x lead). A bare "Glia" node is the deliberate '
            f"back-off where a cluster is glial but its subtype ties."
            + (
                f"\nOmitted from this panel at under 1% of drawn cells: "
                f"{', '.join(drop17)}."
                if drop17
                else ""
            ),
            fontsize=9.5,
            va="bottom",
            linespacing=1.7,
        )
        pp = OUT / "celltype_anatomy_sankey_canonical17.png"
        fig.savefig(pp, dpi=190, facecolor="white", bbox_inches="tight")
        print(f"wrote {pp}")
        return 0

    cached = OUT / "counts_cluster_vote.csv"
    if "--replot" in sys.argv and cached.exists():
        tab = pd.read_csv(cached, index_col=0)
        tab_av = pd.read_csv(
            OUT / "counts_anatomy_guided_vote_CIRCULAR.csv", index_col=0
        )
        # cached tables may carry 'KC-NC'-style region column names
        tab.columns = tab.columns.map(display_region)
        tab_av.columns = tab_av.columns.map(display_region)
        n_one, max_multi = int(tab.values.sum()), 1
        ATLAS_N.update(pd.read_csv(OUT / "atlas_vote_counts.csv", index_col=0)["n"])
        return _finish(tab, tab_av, n_one, max_multi, n_total=None)

    print("loading...", flush=True)
    a = ad.read_h5ad(SPATIAL_COMBINED)
    a.obs["tangram_cell_type"] = (
        read_obs(CANONICAL_TANGRAM)["tangram_cell_type"].astype(str).values
    )
    attach_region_flags(a, regions=STATISTICAL_REGIONS)

    # The expression-only vote (pooled), and the same vote plus the
    # anatomy override, so the override is the only difference between the two.
    hierarchical_vote(a, cluster_col="leiden", label_col="tangram_cell_type", dst="_cv")
    a.obs["_cv"] = [
        v if v == "Mixed" else get_pooled(v) for v in a.obs["_cv"].astype(str)
    ]
    anatomy_guided_vote(
        a,
        cluster_col="leiden",
        label_col="tangram_cell_type",
        dst="_av",
        base="hierarchical",
    )
    cv = a.obs["_cv"].astype(str).values
    av = a.obs["_av"].astype(str).values
    ATLAS_N.update(pd.Series(cv).value_counts())
    pd.Series(ATLAS_N, name="n").to_csv(OUT / "atlas_vote_counts.csv")

    # Also vote at canonical-17. At the 0.5 purity floor fewer of the 17 types
    # resolve and more clusters fall below the floor, so Mixed grows. Both
    # tables are written so either can be drawn without reloading.
    add_canonical(a, src="tangram_cell_type")
    cluster_vote(
        a,
        cluster_col="leiden",
        label_col="cell_type_canonical",
        pool=False,
        min_purity=0.5,
        dst="_cv17",
    )
    cv17 = a.obs["_cv17"].astype(str).values

    tab, n_one, max_multi = build_table(a, cv)
    tab_av, _, _ = build_table(a, av)
    tab17, _, _ = build_table(a, cv17)
    tab17.to_csv(OUT / "counts_cluster_vote_canonical17.csv")
    print(
        f"\ncanonical-17 vote resolves {(tab17.sum(1) > 0).sum()} labels "
        f"inside traced tissue: {list(tab17.index[tab17.sum(1) > 0])}"
    )
    return _finish(tab, tab_av, n_one, max_multi, n_total=a.n_obs)


def _finish(tab, tab_av, n_one, max_multi, n_total):
    OUT.mkdir(parents=True, exist_ok=True)
    if n_total:
        print(
            f"  cells in exactly one traced region: {n_one:,} of {n_total:,} "
            f"({n_one / n_total:.1%}); max regions per cell = {max_multi}",
            flush=True,
        )

    rown = tab.div(tab.sum(1), axis=0).fillna(0.0)
    exp = np.outer(tab.sum(1), tab.sum(0)) / tab.values.sum()
    enr = pd.DataFrame(
        tab.values / np.maximum(exp, 1e-9), index=tab.index, columns=tab.columns
    )

    print("\n=== row-normalised (fraction of each type in each region) ===")
    pd.set_option("display.width", 220)
    print((rown * 100).round(1).to_string())
    print("\n=== enrichment (observed / expected; 1.0 = base rate) ===")
    print(enr.round(2).to_string())

    # Enrichment is capped by how much of the traced-region pool the compartment
    # group already is: a type 100% inside a group that is 72% of all
    # region-assigned cells cannot exceed 1/0.72. The ceiling is reported alongside
    # the value.
    colshare = tab.sum(0) / tab.values.sum()
    for name, expect in EXPECTED.items():
        if name not in rown.index:
            continue
        cols = [c for c in expect if c in rown.columns]
        frac = rown.loc[name, cols].sum()
        share = colshare[cols].sum()
        # Group-level enrichment, read against its ceiling (see above).
        print(
            f"\n  {name}: {frac:.1%} of its cells sit in "
            f"{'/'.join(expect)} (that group is {share:.1%} of all "
            f"region-assigned cells)"
        )
        print(
            f"      group enrichment {frac / share:.2f}x of a possible "
            f"{1 / share:.2f}x  =  {frac:.1%} of the attainable maximum"
        )
        print(
            f"      best single compartment: "
            f"{enr.loc[name, cols].idxmax()} at "
            f"{enr.loc[name, cols].max():.2f}x"
        )

    tab.to_csv(OUT / "counts_cluster_vote.csv")
    tab_av.to_csv(OUT / "counts_anatomy_guided_vote_CIRCULAR.csv")

    fig, axes = plt.subplots(
        1, 3, figsize=(25, 9), gridspec_kw={"width_ratios": [1.35, 1.0, 0.85]}
    )
    # Node height is the true cell count; row-normalised equal-height nodes would
    # hide how much larger the Kenyon population is than the projection neurons.
    # Projection neurons (below SANKEY_MIN of drawn cells) would render as a
    # hairline, so the sankey omits that row. PNs remain in the heatmap and the
    # size panel.
    SANKEY_MIN = 0.01  # share of drawn cells needed to be drawable
    keep = tab.sum(1) / tab.values.sum() >= SANKEY_MIN
    dropped = [n for n in tab.index if not keep[n]]
    note = (
        f"  {', '.join(dropped)} omitted here at "
        f"{tab.sum(1)[dropped].sum() / tab.values.sum():.2%} of cells;\n"
        f"see the other two panels."
        if dropped
        else ""
    )
    draw(
        axes[0],
        tab[keep.values].loc[keep[keep].index],
        "Transferred cell types land in the expected anatomy",
        "cluster_vote labels (expression only, NO anatomy) -> hand-drawn\n"
        "regions.  Ribbon width = cells.  Coloured = the expected\n"
        f"compartment; grey = everything else.{note}",
    )

    heatmap(
        axes[1],
        enr,
        "Corrected for how big each region is",
        "observed / expected. The base-rate control the earlier\n"
        "whole-atlas sankeys lacked; boxed cells are the\n"
        "expectations being tested.",
    )
    size_bars(
        axes[2],
        tab.sum(1),
        "How big each type actually is",
        "log scale, because the range is 699x. Pale = the whole\n"
        "atlas, solid = what the sankey draws. Only cells inside a\n"
        "traced polygon can be drawn, and coverage is very uneven.",
    )
    fig.subplots_adjust(top=0.86, bottom=0.13, wspace=0.42)
    fig.text(
        0.012,
        0.015,
        _footer(rown, enr, colshare, n_one, tab.sum(1), tab, ATLAS_N),
        fontsize=9,
        va="bottom",
        color="#222222",
        linespacing=1.7,
    )
    p = OUT / "celltype_anatomy_sankey.png"
    fig.savefig(p, dpi=190, facecolor="white", bbox_inches="tight")
    print(f"\nwrote {p}")

    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "n_cells_one_region": n_one,
                "kenyon_frac_in_calyx": float(
                    rown.loc[
                        "Kenyon cells",
                        [c for c in EXPECTED["Kenyon cells"] if c in rown.columns],
                    ].sum()
                )
                if "Kenyon cells" in rown.index
                else None,
                "optic_frac_in_optic": float(
                    rown.loc[
                        "Optic lobe",
                        [c for c in EXPECTED["Optic lobe"] if c in rown.columns],
                    ].sum()
                )
                if "Optic lobe" in rown.index
                else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
