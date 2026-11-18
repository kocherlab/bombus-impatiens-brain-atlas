#!/usr/bin/env python3
"""SI figure: per-cell Tangram argmax vs. the expression-only cluster vote.

Per-cell argmax labels are weakly resolved individually (top-1 posterior over
29 classes, top1-top2 margin, and the share of cells passing top-1 >= 0.3 and
margin >= 0.1 are summarised here), so cell-type calls are made on
clusters. This figure shows one spatial section, the same cells throughout,
painted four ways plus a summary bar chart.

  A  hand-drawn anatomical regions (the reference map a reader needs in
     order to judge whether a cell-type call is anatomically possible),
     labelled directly on the tissue
  B  raw per-cell ``tangram_cell_type`` argmax, at full canonical-17
     resolution -- the noisy, unpooled per-cell call
  C  the expression-only cluster vote (``cluster_vote``, no anatomy) --
     the label every main figure uses. This is only
     ever computed at display-group resolution here (a flat per-cluster
     vote over the 17-way vocabulary sends ~39% of traced cells to Mixed;
     a hierarchical fine/pooled backoff recovers the fine vocabulary
     cleanly, but that is a separate figure -- fine_vote_rule.py),
     so C is drawn coarser than B on purpose, not for consistency with it.
  D  per-cell Tangram top-1 posterior, painted spatially (where the mapping
     is uncertain)
  E  for every canonical-17 label present in the section, the anatomical
     distribution of its per-cell calls (bar chart). The two empirically
     worst-placed labels are bolded in its tick labels rather than getting
     their own spatial panel.

Nothing is filtered, smoothed or cleaned: panel B is meant to look noisy,
because that is the evidence.

Note: the cluster vote is computed over the full atlas and only then
subset to the section. Voting inside the section would re-score each Leiden
cluster over a fraction of its members and produce labels that differ from
the atlas-wide labels every other figure uses.

Run:
    BIMP_FIG_SUBSAMPLE_FRACTION=1 PYTHONPATH=src:scripts \
        .venv/bin/python scripts/identity/si_percell_vs_vote.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import h5py
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from anndata.io import read_elem
from matplotlib.lines import Line2D

_HALO = [pe.withStroke(linewidth=2.4, foreground="white")]

from bimp_st.cluster_labels import (
    CANONICAL_ORDER,
    MIXED_DISPLAY,
    POOLED_ORDER,
    get_canonical,
    get_canonical_color,
    get_canonical_display,
    get_pooled,
    get_pooled_color,
    get_pooled_display,
    hierarchical_vote,
)
from bimp_st.nine_region_common import DESCRIPTIVE_REGIONS, attach_region_flags
from bimp_st.nine_region_common import REGION_COLORS as _SHARED_COLORS  # noqa: E402
from bimp_st.nine_region_common import display as region_display
from bimp_st.paths import CANONICAL_TANGRAM
from bimp_st.si_common import (
    LEGEND_SIZE,
    PIXEL_SCALE,
    TICK_SIZE,
    set_style,
    strip_spines,
)

warnings.filterwarnings("ignore", message="Observation names are not unique")

# ── Configuration ────────────────────────────────────────────────────────────
from bimp_st.paths import REPO as REPO_ROOT  # noqa: E402

# The canonical mapping. Cell-type column names must be keyed by name, not
# position: tangram2 orders posterior rows by descending frequency, not by the
# alphabetical category list. This object names its columns at source.
TANGRAM_H5AD = CANONICAL_TANGRAM
# Own output directory: results/ is a symlink shared across worktrees with no
# conflict detection, so this figure never writes into another figure's dir.
OUTPUT_DIR = REPO_ROOT / "results/figures/si_percell_vs_vote"
FIG_NAME = "fig_s_percell_vs_vote"  # stem of the companion tables in OUTPUT_DIR
FIG_DIR = REPO_ROOT / "results/figures/si"  # the registered figure

# Queen section picked as the one maximising min(#cells in a Kenyon-cell
# compartment, #cells in an optic-lobe region) over all 32 sections: both the
# mushroom-body calyx and the optic lobe are well represented, so an
# anatomically impossible assignment is visible rather than merely tabulated.
SECTION = "output-XETG00151__0040243__Region_6__20251204__195015_2_6B"
SECTION_LABEL = "queen, section 2_6B"

MIXED_LABEL = "Mixed"
MIXED_COLOR = "#CBCBCB"
UNASSIGNED_REGION = "unassigned"
UNASSIGNED_COLOR = "#BBBBBB"

# Anatomical-region vocabulary: all nine hand-drawn regions -- this figure is
# purely descriptive (see nine_region_common.DESCRIPTIVE_REGIONS), unlike
# si_regional_degs.py's statistics-only seven, which excludes the two sparsest
# new regions from every fit. Display strings come from
# nine_region_common.display() so they can never retype-drift from the
# vocabulary every other figure uses; never bare IC/NC/OC,
# never "KC nurse". Palette for the six original regions matches
# fig_atlas_overview.REGION_PALETTE.
REGION_FLAGS = {f"in_{r}": region_display(r) for r in DESCRIPTIVE_REGIONS}
# Display order is a deliberate choice (calyx, then optic, then the rest);
# the set of regions is not -- it follows DESCRIPTIVE_REGIONS, so pooling the
# central complex or drawing a new region needs no edit here. The palette comes
# from nine_region_common.REGION_COLORS, the single source every figure shares.
_PREFERRED = [region_display(r) for r in ("kc_ic", "kc_nc", "kc_oc")] + [
    "Lamina",
    "Medulla",
    "Lobula",
]
_LIVE = [region_display(r) for r in DESCRIPTIVE_REGIONS]
REGION_ORDER = (
    [r for r in _PREFERRED if r in _LIVE]
    + [r for r in _LIVE if r not in _PREFERRED]
    + [UNASSIGNED_REGION]
)
REGION_COLORS = {region_display(r): _SHARED_COLORS[r] for r in DESCRIPTIVE_REGIONS}
REGION_COLORS[UNASSIGNED_REGION] = UNASSIGNED_COLOR
ASSIGNED_REGIONS = [r for r in REGION_COLORS if r != UNASSIGNED_REGION]
# REGION_FLAGS is derived from nine_region_common; REGION_ORDER/REGION_COLORS
# are typed literals (kept separate because they encode a deliberate display
# order/palette, not the data source's own order). panel_region_bars()
# bracket-indexes REGION_COLORS, so a region missing from either literal would
# raise a KeyError at plot time; check at import time instead.
assert (
    set(REGION_FLAGS.values())
    == set(ASSIGNED_REGIONS)
    == (set(REGION_ORDER) - {UNASSIGNED_REGION})
), (
    "region display strings have drifted apart (REGION_FLAGS vs. "
    "REGION_ORDER/REGION_COLORS)"
)
KC_REGIONS = tuple(region_display(r) for r in ("kc_ic", "kc_nc", "kc_oc"))
OPTIC_REGIONS = ("Lamina", "Medulla", "Lobula")

# Literature-anchored expectation used to rank per-cell labels by anatomical
# contradiction (the anatomical sanity test): Kenyon
# cells belong to the mushroom-body calyx, photoreceptors and optic neurons to
# the optic lobe. A label's "contradiction rate" is the share of its
# region-assigned per-cell calls that land in the opposite territory.
EXPECTED_TERRITORY = {
    "Large-type Kenyon cells": KC_REGIONS,
    "Middle-type Kenyon cells": KC_REGIONS,
    "Small-type Kenyon cells": KC_REGIONS,
    "Class II Kenyon cells": KC_REGIONS,
    "Other Kenyon cells": KC_REGIONS,
    "Kenyon cells": KC_REGIONS,
    "Photoreceptors": OPTIC_REGIONS,
    "Optic lobe": OPTIC_REGIONS,
}
CONTRADICTION_MIN_N = 200  # per-cell calls needed before ranking a label
N_OFFENDERS = 2  # how many worst labels get bolded in panel E's bars

# Per-cell top-1 posterior display range. The QC rule is
# top-1 >= 0.3 and margin >= 0.1; 0.3 is marked on the colourbar. The range is
# clipped to 0.10-0.60 rather than 0-1 because most cells sit low on a 0-1
# ramp, which renders the whole section one flat dark colour -- true, but
# unreadable.
CONF_VMIN, CONF_VMAX = 0.10, 0.60
CONF_THRESHOLD = 0.3
CONF_CMAP = "magma"

# Close to the printed width (6.7 in), so authored sizes print as authored
# and the print-scale hook's single text factor stays near 1.
FIG_W = 7.2
DOT_SIZE = 0.9

# Legends, colourbars and notes sit below the spatial axes, never on the
# section, so no text is drawn over tissue. Explanatory prose is in the
# caption; its numbers are written to the sidecar JSON.
LEGEND_NCOL = 3


class _ObsWrapper:
    """adata-like shim so cluster_labels helpers work on a bare obs frame."""

    def __init__(self, obs: pd.DataFrame):
        self.obs = obs

    @property
    def n_obs(self):
        return len(self.obs)


# ── Data ─────────────────────────────────────────────────────────────────────


def load():
    print(f"Loading {TANGRAM_H5AD} (obs + obsm/spatial only)...", flush=True)
    with h5py.File(TANGRAM_H5AD, "r") as h:
        obs = read_elem(h["obs"])
        spatial = read_elem(h["obsm/spatial"])
    print(f"  {len(obs):,} cells", flush=True)
    return obs, spatial


def label(obs: pd.DataFrame) -> pd.DataFrame:
    """Add the vote + raw label columns. Vote is computed over the full atlas.

    Uses cluster_vote (expression only, no anatomy). Beware persisted
    "cluster_vote" columns (e.g. old parquet caches) that hold the
    anatomy-guided result under the anatomy-free name.
    """
    print(
        "Computing the expression-only cluster vote over the full atlas "
        "(no anatomy)...",
        flush=True,
    )
    hierarchical_vote(
        _ObsWrapper(obs),
        cluster_col="leiden",
        label_col="tangram_cell_type",
        # Same rule as every other figure: fine name at purity >= 0.5 or a
        # 1.5x lead over the runner-up, else the pooled group at >= 0.5, else
        # Mixed.
        mixed_label=MIXED_LABEL,
        dst="cluster_vote",
    )
    raw = obs["tangram_cell_type"].astype(str)
    # hierarchical_vote returns fine names where a cluster decides one.
    # This figure compares the vote against the per-cell argmax pooled to
    # the display groups, so the vote is pooled to the same vocabulary --
    # otherwise "Astrocyte-like glia" vs "Glia" scores as a disagreement
    # when the two sides agree at the level being compared.
    obs["cluster_vote_fine"] = obs["cluster_vote"].astype(str)
    obs["cluster_vote"] = [
        v if v == MIXED_LABEL else get_pooled(v) for v in obs["cluster_vote_fine"]
    ]
    obs["raw_pooled"] = [get_pooled(v) for v in raw]
    obs["raw_canonical"] = [get_canonical(v) for v in raw]

    # Refresh the region flags from the crop sidecars before painting: the
    # copy baked into the tangram object predates them (e.g. four polygons on
    # worker 0068401 R1 move 119 cells from lobula to antennal lobe).
    # DESCRIPTIVE_REGIONS, not BASE_REGIONS: colours, order and legend are
    # built from the eight descriptive regions, and attaching only the six
    # base ones would leave in_al / in_central_complex unpainted (329 and 183
    # cells in this section).
    attach_region_flags(_ObsWrapper(obs), regions=DESCRIPTIVE_REGIONS)

    reg = np.full(len(obs), UNASSIGNED_REGION, dtype=object)
    overlap = np.zeros(len(obs), dtype=int)
    for flag, name in REGION_FLAGS.items():
        if flag not in obs.columns:
            # Not a skippable condition: REGION_FLAGS comes from
            # DESCRIPTIVE_REGIONS, so a flag missing here means the attach call
            # above asked for a different set, and the region would be
            # unpainted while still listed in the legend.
            raise SystemExit(
                f"{flag} missing from obs: REGION_FLAGS and the "
                "attach_region_flags() call have drifted apart"
            )
        # in_al / in_ellipsoid_body / in_fan_shaped_body are nullable
        # booleans -- pd.NA means "no tracing for this region yet", not
        # "outside it" (nine_region_common.py). ``.astype(bool)`` raises on
        # the NA entries; na_value=False here only controls which colour an
        # untraced cell is painted in this purely descriptive figure (it
        # falls through to "unassigned" grey), never a caste comparison.
        m = obs[flag].to_numpy(dtype=bool, na_value=False)
        reg[m] = name
        overlap += m
    if overlap.max() > 1:
        print(
            f"  WARNING: {(overlap > 1).sum():,} cells carry >1 region flag; "
            "the last flag in REGION_FLAGS wins",
            flush=True,
        )
    obs["anat_region"] = reg
    return obs


# ── Statistics ───────────────────────────────────────────────────────────────


def region_table(sec: pd.DataFrame) -> pd.DataFrame:
    """canonical-17 label x anatomical region counts for the section."""
    tab = pd.crosstab(sec["raw_canonical"].astype(str), sec["anat_region"])
    tab = tab.reindex(
        index=[c for c in CANONICAL_ORDER if c in tab.index],
        columns=REGION_ORDER,
        fill_value=0,
    )
    return tab


def contradiction_ranking(tab: pd.DataFrame) -> pd.DataFrame:
    """Rank canonical-17 labels by anatomical contradiction rate."""
    rows = []
    for name, expected in EXPECTED_TERRITORY.items():
        if name not in tab.index:
            continue
        counts = tab.loc[name]
        n_total = int(counts.sum())
        n_assigned = int(counts[ASSIGNED_REGIONS].sum())
        wrong = KC_REGIONS if expected == OPTIC_REGIONS else OPTIC_REGIONS
        n_right = int(counts[list(expected)].sum())
        n_wrong = int(counts[list(wrong)].sum())
        rows.append(
            {
                "label": name,
                "n_calls": n_total,
                "n_region_assigned": n_assigned,
                "expected_territory": "Kenyon-cell compartments"
                if expected == KC_REGIONS
                else "optic lobe",
                "frac_in_expected_of_all": n_right / n_total if n_total else np.nan,
                "frac_in_contradicting_of_all": n_wrong / n_total
                if n_total
                else np.nan,
                "contradiction_rate_of_region_assigned": n_wrong / n_assigned
                if n_assigned
                else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df["rankable"] = df["n_calls"] >= CONTRADICTION_MIN_N
    return df.sort_values(
        ["rankable", "contradiction_rate_of_region_assigned"], ascending=False
    ).reset_index(drop=True)


def compute_stats(
    obs: pd.DataFrame, sec: pd.DataFrame, tab: pd.DataFrame, ranking: pd.DataFrame
) -> dict:
    def disagreement(frame):
        raw = frame["raw_pooled"].astype(str)
        vote = frame["cluster_vote"].astype(str)
        differs = raw != vote
        called = vote != MIXED_LABEL
        return {
            "n_cells": int(len(frame)),
            "frac_label_changes": float(differs.mean()),
            "frac_vote_is_mixed": float((~called).mean()),
            "frac_label_changes_where_vote_calls_a_type": float(differs[called].mean())
            if called.any()
            else float("nan"),
        }

    qc = obs["tangram_label_qc"].astype(str)
    sec_qc = sec["tangram_label_qc"].astype(str)
    return {
        "section": SECTION,
        "section_label": SECTION_LABEL,
        "atlas": disagreement(obs),
        "section_stats": disagreement(sec),
        "atlas_confidence": {
            "median_top1_posterior": float(obs["tangram_confidence"].median()),
            "median_top1_minus_top2_margin": float(obs["tangram_margin"].median()),
            "frac_confident": float((qc == "confident").mean()),
            "n_classes": int(obs["tangram_cell_type"].nunique()),
        },
        "section_confidence": {
            "median_top1_posterior": float(sec["tangram_confidence"].median()),
            "median_top1_minus_top2_margin": float(sec["tangram_margin"].median()),
            "frac_confident": float((sec_qc == "confident").mean()),
        },
        "section_region_assignment": {
            "frac_cells_in_a_hand_drawn_region": float(
                (sec["anat_region"] != UNASSIGNED_REGION).mean()
            ),
        },
        "contradiction_ranking": ranking.to_dict(orient="records"),
        "section_label_by_region_counts": tab.to_dict(),
    }


# ── Drawing helpers ──────────────────────────────────────────────────────────


def _categorical_panel(ax, xy, cats, order, color_fn, rng, *, base=None):
    """Scatter categorical labels, abundant categories first so rare ones show."""
    counts = pd.Series(cats).value_counts()
    if base is not None:
        m = ~np.isin(cats, order)
        if m.any():
            ax.scatter(
                xy[m, 0],
                xy[m, 1],
                c=base,
                s=DOT_SIZE,
                alpha=0.9,
                edgecolors="none",
                rasterized=True,
            )
    for cat in sorted(order, key=lambda c: -counts.get(c, 0)):
        m = cats == cat
        if not m.any():
            continue
        idx = rng.permutation(int(m.sum()))
        ax.scatter(
            xy[m][idx, 0],
            xy[m][idx, 1],
            c=color_fn(cat),
            s=DOT_SIZE,
            alpha=0.9,
            edgecolors="none",
            rasterized=True,
        )


def _legend(
    ax, entries, *, title, loc, bbox, ncol=1, fontsize=LEGEND_SIZE, columnspacing=0.9
):
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=col,
            markeredgecolor="none",
            markersize=3.5,
            label=name,
        )
        for name, col in entries
    ]
    lg = ax.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=bbox,
        frameon=False,
        ncol=ncol,
        fontsize=fontsize,
        title=title,
        title_fontsize=fontsize,
        handletextpad=0.1,
        columnspacing=columnspacing,
        labelspacing=0.25,
        borderpad=0.0,
    )
    lg._legend_box.align = "left"
    return lg


def _below_legend(
    ax,
    entries,
    *,
    title,
    ncol=LEGEND_NCOL,
    y=-0.015,
    fontsize=LEGEND_SIZE,
    columnspacing=0.9,
):
    """Legend in the margin strip below a spatial axes.

    The axes are anchored "N" inside a gridspec cell that is deliberately
    taller than the (equal-aspect) section, so this strip is empty page rather
    than tissue. No white halo is needed here -- nothing is drawn behind it.
    """
    return _legend(
        ax,
        entries,
        title=title,
        loc="upper left",
        bbox=(0.0, y),
        ncol=ncol,
        fontsize=fontsize,
        columnspacing=columnspacing,
    )


def _margin_scalebar(ax, *, bar_um=1000, label="1 mm", x_right=0.99, y=-0.018):
    """Scale bar in the margin strip, right-aligned under the section.

    The C-shaped section leaves no in-axes position for a 1 mm bar that is
    clear of tissue, so the bar sits in the strip below, beside the section it
    measures. Length is computed in data units, so it stays a true 1 mm.
    """
    xlo, xhi = ax.get_xlim()
    frac = (bar_um / PIXEL_SCALE) / (xhi - xlo)
    x0 = x_right - frac
    ax.plot(
        [x0, x_right],
        [y, y],
        transform=ax.transAxes,
        lw=2.2,
        color="black",
        solid_capstyle="butt",
        clip_on=False,
    )
    ax.text(
        x0 + frac / 2,
        y - 0.013,
        label,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=TICK_SIZE,
        clip_on=False,
    )


def _wrap_pooled(name: str) -> str:
    """Legend label for a pooled group.

    The display strings themselves are fixed by cluster_labels (they are what
    make the pooled and fine vocabularies distinguishable). The "(all
    subtypes)" marker stays on the same line: below the axes two columns of
    full-width labels fit, and one line per entry keeps the margin strip short.
    """
    return get_pooled_display(name)


def _finish_spatial(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_facecolor("none")  # neighbouring axes must not overpaint a legend
    strip_spines(ax)


# ── Panels ───────────────────────────────────────────────────────────────────


def panel_regions(ax, xy, sec, rng, xlim, ylim, *, legend=False):
    cats = sec["anat_region"].astype(str).to_numpy()
    present = [r for r in REGION_ORDER if r in set(cats)]
    _categorical_panel(ax, xy, cats, present, lambda c: REGION_COLORS[c], rng)
    _finish_spatial(ax, xlim, ylim)

    # Region names are a margin legend below the section, like panels B and C:
    # on a section this densely packed, on-tissue labels end up over cells or
    # crossing each other's leader lines. Colours match panel E.
    _below_legend(
        ax,
        [(r, REGION_COLORS[r]) for r in present],
        title="Anatomical region (A, E)",
        ncol=2,
        y=-0.085,
        columnspacing=0.5,
    )
    _margin_scalebar(ax)


def panel_pooled(ax, xy, cats, rng, xlim, ylim, *, legend, title="Cell type"):
    present = [c for c in POOLED_ORDER if c in set(cats)]
    if MIXED_LABEL in set(cats):
        present.append(MIXED_LABEL)

    def color(c):
        return MIXED_COLOR if c == MIXED_LABEL else get_pooled_color(c)

    _categorical_panel(ax, xy, cats, present, color, rng)
    _finish_spatial(ax, xlim, ylim)
    if legend:
        entries = [
            (MIXED_DISPLAY if c == MIXED_LABEL else _wrap_pooled(c), color(c))
            for c in present
        ]
        # ncol=1, reduced fontsize. Legend columns are laid out at creation;
        # print_scale's auto-grower later enlarges the Text artists in place
        # without moving column boundaries, and its collision detector does
        # not see cross-column or cross-legend overlaps. ncol=2 overflows into
        # its own second column, and ncol=1 at the default size reaches panel
        # B's legend, so this is sized for the full uncapped growth factor
        # (as printed in the render log). Checked against the render.
        _below_legend(ax, entries, title=title, ncol=1)


def panel_canonical(ax, xy, cats, rng, xlim, ylim):
    """Full canonical-17 resolution -- same vocabulary/colours as panel E's bars.

    No legend under the section: 17 full names cannot fit one quarter-width
    column at a legible size, and panel E already lists exactly these labels
    as its rows. Each of E's row labels carries this panel's colour as a dot,
    so E's y axis is B's key (checked in compose()).
    """
    present = [c for c in CANONICAL_ORDER if c in set(cats)]
    _categorical_panel(ax, xy, cats, present, get_canonical_color, rng)
    _finish_spatial(ax, xlim, ylim)
    ax.text(
        0.0,
        -0.03,
        "Colours: dots at\nE's row labels",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=LEGEND_SIZE,
    )
    return present


def panel_confidence(ax, fig, xy, sec, rng, xlim, ylim):
    conf = sec["tangram_confidence"].astype(float).to_numpy()
    idx = rng.permutation(len(conf))
    sc = ax.scatter(
        xy[idx, 0],
        xy[idx, 1],
        c=conf[idx],
        s=DOT_SIZE,
        cmap=CONF_CMAP,
        vmin=CONF_VMIN,
        vmax=CONF_VMAX,
        alpha=0.9,
        edgecolors="none",
        rasterized=True,
    )
    _finish_spatial(ax, xlim, ylim)

    # Below the axes, in the same margin strip the other panels' legends use --
    # but inset from the left: panel C's two-column legend spills into the
    # strip under D, and an inset of 0.34 clears it while the bar stays inside D.
    cax = ax.inset_axes([0.15, -0.035, 0.70, 0.025])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(
        "Tangram top-1 posterior\n(29 candidate classes)",
        fontsize=LEGEND_SIZE,
        labelpad=3,
    )
    cb.ax.tick_params(labelsize=TICK_SIZE, length=2, pad=1.5)
    cb.outline.set_visible(False)
    cb.set_ticks([0.2, 0.4, 0.6])
    cax.plot(
        [CONF_THRESHOLD, CONF_THRESHOLD],
        [0, 1],
        color="white",
        lw=1.2,
        transform=cb.ax.get_xaxis_transform(),
        clip_on=False,
    )
    return cax


def panel_region_bars(ax, tab, offenders, ranking):
    keep = tab[tab.sum(axis=1) >= 30]
    keep = keep.loc[keep.sum(axis=1).sort_values().index]
    frac = keep.div(keep.sum(axis=1), axis=0) * 100
    y = np.arange(len(keep))
    left = np.zeros(len(keep))
    for reg in REGION_ORDER:
        vals = frac[reg].to_numpy()
        ax.barh(
            y,
            vals,
            left=left,
            height=0.72,
            color=REGION_COLORS[reg],
            edgecolor="white",
            linewidth=0.35,
            label=reg,
        )
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([get_canonical_display(n) for n in keep.index])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(keep) - 0.3)
    ax.set_xlabel("Per-cell calls with that label (%)")
    # Wide pad: a dot in panel B's colour sits between each label and the
    # bars, making this axis B's colour key.
    ax.tick_params(axis="y", length=0, pad=9)
    ax.spines["left"].set_visible(False)
    dot_tr = mtransforms.offset_copy(
        ax.get_yaxis_transform(), fig=ax.figure, x=-4.5, units="points"
    )
    ax.scatter(
        np.zeros(len(keep)),
        y,
        s=14,
        c=[get_canonical_color(n) for n in keep.index],
        edgecolors="none",
        transform=dot_tr,
        clip_on=False,
        zorder=5,
    )

    # Right-hand columns: calls per label, and the contradiction rate (the
    # term the caption defines) for the two worst offenders only. Two short
    # columns instead of one long run-on string that reached the image edge.
    rank = ranking.set_index("label")
    # Columns are offset from the right spine in points, so their spacing
    # does not depend on the axes width.
    base = ax.get_yaxis_transform()
    tr_n = mtransforms.offset_copy(base, fig=ax.figure, x=3, units="points")
    tr_c = mtransforms.offset_copy(base, fig=ax.figure, x=31, units="points")
    for tr, head in ((tr_n, "calls"), (tr_c, "contra-\ndiction")):
        ax.text(
            1.0,
            len(keep) - 0.3,
            head,
            transform=tr,
            ha="left",
            va="bottom",
            fontsize=TICK_SIZE,
            color="#555555",
            fontstyle="italic",
        )
    for n, yi in zip(keep.index, y, strict=False):
        ax.text(
            1.0,
            yi,
            f"{int(keep.loc[n].sum()):,}",
            transform=tr_n,
            va="center",
            ha="left",
            fontsize=TICK_SIZE,
            color="#555555",
        )
        if n in offenders and n in rank.index:
            rate = rank.loc[n, "contradiction_rate_of_region_assigned"] * 100
            ax.text(
                1.0,
                yi,
                f"{rate:.0f}%",
                transform=tr_c,
                va="center",
                ha="left",
                fontsize=TICK_SIZE,
                color="#555555",
            )
    return list(keep.index)


# ── Composition ──────────────────────────────────────────────────────────────


def compose(
    sec: pd.DataFrame,
    xy: np.ndarray,
    tab: pd.DataFrame,
    offenders: list[str],
    ranking: pd.DataFrame,
    stats: dict,
) -> Path:
    set_style()
    rng = np.random.default_rng(0)

    pad_x = np.ptp(xy[:, 0]) * 0.02
    pad_y = np.ptp(xy[:, 1]) * 0.02
    xlim = (xy[:, 0].min() - pad_x, xy[:, 0].max() + pad_x)
    ylim = (xy[:, 1].min() - pad_y, xy[:, 1].max() + pad_y)
    span = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])

    # Absolute-inch layout on a ~7.2 in canvas. Row 1: four equal-aspect
    # spatial panels (A anatomy / B raw Tangram / C vote / D confidence),
    # each cut exactly to the section's aspect, with a margin strip below for
    # A's region legend, C's legend and D's colourbar. Row 2: panel E's bars,
    # whose own axes leaves a left gutter for its row labels and a right one
    # for the two annotation columns, so nothing reaches the image edge.
    top_in, gap_in, hdr2_in = 0.30, 0.10, 0.34
    col_gap = 0.07
    panel_w = (FIG_W - 3 * col_gap) / 4
    panel_h = panel_w * span
    strip_in = 0.84  # tallest occupant: A's two-column region legend
    n_bars = int((tab.sum(axis=1) >= 30).sum())
    bars_in = max(n_bars * 0.135, 1.3)
    xlab_in = 0.38
    fig_h = top_in + panel_h + strip_in + gap_in + hdr2_in + bars_in + xlab_in

    fig = plt.figure(figsize=(FIG_W, fig_h))

    def _axes_in(x, y_top, w, h):
        return fig.add_axes(
            [x / FIG_W, (fig_h - y_top - h) / fig_h, w / FIG_W, h / fig_h]
        )

    axes = {}
    for i, key in enumerate("ABCD"):
        axes[key] = _axes_in(i * (panel_w + col_gap), top_in, panel_w, panel_h)
    e_left, e_right = 1.62, 0.95
    axes["E"] = _axes_in(
        e_left,
        top_in + panel_h + strip_in + gap_in + hdr2_in,
        FIG_W - e_left - e_right,
        bars_in,
    )

    panel_regions(axes["A"], xy, sec, rng, xlim, ylim, legend=False)
    b_present = panel_canonical(
        axes["B"],
        xy,
        sec["raw_canonical"].astype(str).to_numpy(),
        rng,
        xlim,
        ylim,
    )
    # C stays at display-group resolution -- the expression-only vote is
    # only ever computed there (see the module docstring); it is
    # intentionally coarser than B, not a copy of B's palette.
    panel_pooled(
        axes["C"],
        xy,
        sec["cluster_vote"].astype(str).to_numpy(),
        rng,
        xlim,
        ylim,
        legend=True,
        title="Cell type (display groups)",
    )
    panel_confidence(axes["D"], fig, xy, sec, rng, xlim, ylim)
    e_rows = panel_region_bars(axes["E"], tab, offenders, ranking)
    # B's only colour key is E's row labels, so every label painted in B must
    # be one of E's rows.
    missing = sorted(set(b_present) - set(e_rows))
    assert not missing, f"B labels without a key row in E: {missing}"

    from bimp_st.si_common import add_panel_headers

    # Letters off the axes; A-D share a top edge so they share a baseline.
    # E's letter sits over the start of its row labels, not the bars.
    add_panel_headers(
        fig,
        [
            (axes["A"], "A", "Anatomical regions"),
            (axes["B"], "B", "Argmax labels"),
            (axes["C"], "C", "Cluster vote"),
            (axes["D"], "D", "Mapping confidence"),
        ],
        width_frac=1.0,
    )
    e_hdr = fig.add_axes([0.0, axes["E"].get_position().y1, 0.01, 0.0])
    e_hdr.axis("off")
    e_hdr.set_xticks([])  # no phantom tick labels for print_scale
    e_hdr.set_yticks([])
    add_panel_headers(fig, [(e_hdr, "E", "Labels by region")], width_frac=1.0)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "figS_percell_vs_vote.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {out}", flush=True)
    return out


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    obs, spatial = load()
    obs = label(obs)

    mask = (obs["slice_name"].astype(str) == SECTION).to_numpy()
    if not mask.any():
        raise SystemExit(f"section {SECTION} not found")
    sec = obs[mask].copy()
    xy = spatial[mask].astype(float)
    print(f"Section {SECTION_LABEL}: {len(sec):,} cells", flush=True)

    tab = region_table(sec)
    ranking = contradiction_ranking(tab)
    offenders = ranking[ranking["rankable"]]["label"].tolist()[:N_OFFENDERS]
    stats = compute_stats(obs, sec, tab, ranking)
    stats["offenders_bolded_in_panel_E"] = offenders

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUTPUT_DIR / f"{FIG_NAME}_label_by_region.csv")
    ranking.to_csv(OUTPUT_DIR / f"{FIG_NAME}_contradiction_ranking.csv", index=False)
    with open(OUTPUT_DIR / f"{FIG_NAME}_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2, default=str)

    out = compose(sec, xy, tab, offenders, ranking, stats)

    # ── printed summary (numbers for the caption / report) ──
    a, s = stats["atlas"], stats["section_stats"]
    print("\n=== per-cell argmax vs. expression-only cluster vote ===")
    print(f"section: {SECTION} ({SECTION_LABEL}), {s['n_cells']:,} cells")
    print(
        f"atlas   ({a['n_cells']:,} cells): "
        f"{a['frac_label_changes'] * 100:.1f}% change label; "
        f"{a['frac_label_changes_where_vote_calls_a_type'] * 100:.1f}% among "
        f"non-{MIXED_DISPLAY}; {a['frac_vote_is_mixed'] * 100:.1f}% "
        f"of cells sit in a {MIXED_DISPLAY} cluster"
    )
    print(
        f"section ({s['n_cells']:,} cells): "
        f"{s['frac_label_changes'] * 100:.1f}% change label; "
        f"{s['frac_label_changes_where_vote_calls_a_type'] * 100:.1f}% among "
        f"non-{MIXED_DISPLAY}; {s['frac_vote_is_mixed'] * 100:.1f}% "
        f"of cells sit in a {MIXED_DISPLAY} cluster"
    )
    ac, scf = stats["atlas_confidence"], stats["section_confidence"]
    print(
        f"atlas confidence: median top-1 {ac['median_top1_posterior']:.3f}, "
        f"median margin {ac['median_top1_minus_top2_margin']:.3f}, "
        f"{ac['frac_confident'] * 100:.1f}% confident"
    )
    print(
        f"section confidence: median top-1 {scf['median_top1_posterior']:.3f}, "
        f"median margin {scf['median_top1_minus_top2_margin']:.3f}, "
        f"{scf['frac_confident'] * 100:.1f}% confident"
    )
    frac_drawn = stats["section_region_assignment"]["frac_cells_in_a_hand_drawn_region"]
    print(f"cells inside a hand-drawn region: {frac_drawn * 100:.1f}%")

    print("\nanatomical contradiction ranking (section):")
    for r in ranking.itertuples():
        flag = "" if r.rankable else "   [n < threshold]"
        print(
            f"  {r.label:<32} n={r.n_calls:>6,}  "
            f"in expected {r.frac_in_expected_of_all * 100:5.1f}% of all calls  "
            f"in contradicting {r.frac_in_contradicting_of_all * 100:5.1f}% of all "
            f"calls  contradiction rate (region-assigned only) "
            f"{r.contradiction_rate_of_region_assigned * 100:5.1f}%{flag}"
        )
    print(f"\noffenders bolded in panel E: {offenders}")
    print(f"figure: {out}")


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        main()
