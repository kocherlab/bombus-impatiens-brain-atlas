#!/usr/bin/env python
"""Caste differences in cellular composition (fig:caste_comparison).

Layout: A across the top row; B, C and D across the bottom row.
  A  queen and worker composition bars for the spatial atlas after label
     transfer, at the expression-only cluster-vote categories (+ Mixed)
  B  per-section spread of the same class proportions
  C  per-region composition, queen vs worker (horizontal stacked bars)
  D  per-region Hill-number diversity profile (q=0-3), queen vs worker
     (rarefied)

Labels.  Every panel uses one label assignment: the expression-only
``hierarchical_vote`` (defaults, no anatomy override; see _display_ct_series
in caste_comparison_sccoda.py).  C (bars, counts and the
regional-against-caste contrast, all from
results/region_composition/composition_by_region.csv) uses the same display
labels as A and B; D's diversity (region_diversity.py) recomputes the same vote
and pools its labels to POOLED_ORDER (+ Mixed).  C and D condition on anatomical
region, so an anatomy-guided vote would be circular there: it assigns Kenyon
and optic identity from the very region membership being conditioned on.
Cluster vote resolves only the categories in CLUSTER_VOTE_RESOLVED plus
"Mixed"; photoreceptors and hemocytes never win a whole cluster and are not
shown.  The finer canonical-17 vocabulary is not usable spatially: under
cluster vote it resolves 6 of 17 categories, several under the wrong name,
and raw per-cell argmax places "Small-type Kenyon cells" where the anatomy
rules them out.  Per-category evidence:
results/fig2_vocabulary_ceiling/per_category_table.csv and
results/reframe/vote_over_17/ (scripts/identity/vote_scheme_comparison.py).

Names.  Categories are named identically here and in the single-nucleus
figures (``get_pooled_display``): an exact canonical-17 name, or a pool
marked "(all subtypes)".  No short codes, because "KC" and "OL" are also
anatomical region codes in this paper.  "Mixed" (clusters with no confident
majority label, ~9% of cells) is its own neutral-grey category at its true,
un-renormalized share; see MIXED_DISPLAY in cluster_labels.py.

Panel D.  Numbers are read, not recomputed, from
results/region_diversity/hill_diversity_profile.csv (scheme="expr"), the rows
behind SI fig_s_hill_diversity panel A; REGION_ORDER and REGION_GROUP_COLOR
come from the analysis script so the two cannot drift.  The SI panel
pools castes by design; D is per caste, since queen vs worker is this
figure's point.  All regions share one axes (colour = region, linestyle =
caste), with direct end-of-line region callouts.  q=0 is close to a ceiling
under this scheme (most of the categories are present in every
region/caste).  Mixed is
included, as in A-C; excluding it ("expr_nomixed" in
region_diversity.py) leaves the robust structure unchanged and only
swaps the two least-diverse regions (Lobula/KC-IC), which already differ by
caste, so the caption reports them as a tied pair (full comparison in
fig_s_hill_diversity.py).  D puts no cell-type category name on the figure,
so it cannot be misread as a second naming of the categories; keep it that
way.

All prose lives in the caption.  Data: results/figures/fig2/
{abundance_tangram_ct.csv, slice_composition.csv} from
``caste_comparison_sccoda.py``, plus atlas_vote_composition_by_caste.csv there and
results/region_composition/composition_by_region.csv
(``region_composition.py``).  Caste = one brain per
caste -> descriptive; scCODA credibility is a posterior inclusion
probability, not a p-value.

Draws figure fig_caste_comparison (figures/main/), written to
results/figures/main/caste_comparison/. caste_comparison_sccoda.py is the analysis step
(scCODA and the CSVs above, in results/figures/fig2/).
"""

import os
import sys
from pathlib import Path
from pathlib import Path as _Path

# Repo-root idiom used across scripts/.
sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / "scripts"))
import textwrap  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Panel D reuses the SI analysis's own region order and colour ramp rather
# than restating either here, so the panel cannot drift from the
# analysis it plots (that script only computes on import of __main__; the
# module-level cost is its imports).
from regions.region_diversity import (  # noqa: E402
    REGION_GROUP_COLOR,
    REGION_ORDER,
)

from bimp_st.cluster_labels import (  # noqa: E402
    get_pooled_color,
    get_pooled_display,
)
from bimp_st.nine_region_common import display_region  # noqa: E402
from bimp_st.style import (  # noqa: E402
    QUEEN_COLOR,
    WORKER_COLOR,
    panel_header,
    set_pub_style,
)

set_pub_style()

from bimp_st.paths import REPO, require_input  # noqa: E402

ROOT = REPO / "results/figures/fig2"  # input tables (caste_comparison_tables.py)
# Every input this module reads, checked before anything is drawn or written.
for _inp in (
    ROOT / "abundance_tangram_ct.csv",
    ROOT / "atlas_vote_composition_by_caste.csv",
    ROOT / "slice_composition.csv",
    REPO / "results/region_composition/composition_by_region.csv",
    REPO / "results/region_diversity/hill_diversity_profile.csv",
):
    require_input(_inp)
# The registered figure, figures/main/fig_caste_comparison.
FIGDIR = Path(
    os.environ.get(
        "BIMP_FIG_CASTE_OUT", REPO / "results/figures/main/caste_comparison"
    )
)
FIGDIR.mkdir(parents=True, exist_ok=True)
OUT = FIGDIR / "fig_caste_comparison.png"

# Cluster-vote's resolvable categories + Mixed.  Photoreceptors and Hemocytes
# are in POOLED_ORDER but cluster-vote structurally never assigns them
# (rare/scattered types don't win a whole cluster); see
# results/fig2_vocabulary_ceiling/ for the per-category resolvability analysis.
# Glial subtypes are listed individually because hierarchical_vote names them
# where the cluster decides one, and "Glia" is the residual.  Omitting them
# would drop ~27% of the atlas from panel A while its axis says the bars sum
# to 1.0.
CLUSTER_VOTE_RESOLVED = [
    "Kenyon cells",
    "Projection neurons",
    "Other neurons",
    "Astrocyte-like glia",
    "Perineurial glia",
    "Glia",
    "Optic lobe",
]
MIXED_LABEL = "Mixed"
CLASSES = [*CLUSTER_VOTE_RESOLVED, MIXED_LABEL]
# DISPLAY_COLOR_OVERRIDES first: pooled "Glia" is the same hex as
# Perineurial glia, so without it two categories draw identically.
from bimp_st.cluster_labels import (  # noqa: E402
    DISPLAY_COLOR_OVERRIDES,
    DISPLAY_NAME_OVERRIDES,
)

CLASS_COL = {
    c: DISPLAY_COLOR_OVERRIDES.get(c, get_pooled_color(c)) for c in CLASSES
}  # "Mixed" -> neutral grey (#CBCBCB)

# Panel A: below this fraction of cells, an inline "0.0XX" label on a stacked
# bar is unreadable clutter, not information. The color + shared legend still
# identify every category regardless; exact values are in
# abundance_tangram_ct.csv.
LABEL_MIN_FRAC = 0.018


def _lbl(c):
    """Tick/axis label for category *c*.

    The full display name, wrapped rather than abbreviated.  Short codes are
    not used: "KC" and "OL" also serve as anatomical region codes in the
    other figures, so they would mean a cell type in one figure and a piece of
    tissue in another.  The "(all subtypes)" marker -- identical on every
    pooled category -- tells the reader this is a pool without a key.
    Wrapping keeps it inside the axis; see get_pooled_display().
    """
    if c == MIXED_LABEL:
        return "Other"
    name = DISPLAY_NAME_OVERRIDES.get(str(c), get_pooled_display(c))
    return (
        name.replace(" (all subtypes)", "\n(all subtypes)")
        .replace(" (unassigned)", "\n(unassigned)")
        .replace(" (unresolved)", "\n(unresolved)")
    )


def _lbl_narrow(c, width=14):
    """``_lbl`` with every line hard-wrapped to *width* characters.

    Panel B's x-axis is narrow, and rotated tick labels overlap on their
    longest line, not their average.  Wrapping (not abbreviating) is the fix:
    "KC"/"OL" are anatomical region codes in this paper, and panel C puts
    real region names on the same row.  Panel C keeps the wider ``_lbl``;
    its labels are horizontal and have room.
    """
    return "\n".join(textwrap.fill(line, width) for line in _lbl(c).split("\n"))


def _lum(hexc):
    r, g, b = (int(hexc[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return 0.299 * r + 0.587 * g + 0.114 * b


def _txt_on(hexc):
    return "white" if _lum(hexc) < 0.6 else "#1a1a1a"


ab = (
    pd.read_csv(ROOT / "abundance_tangram_ct.csv").set_index("cluster").reindex(CLASSES)
)
n_queen = int(ab["n_queen"].iloc[0])
n_worker = int(ab["n_worker"].iloc[0])

missing = ab.index[ab["prop_queen"].isna()].tolist()
if missing:
    # Drop, don't zero-fill.  A category the cluster vote never assigns has
    # no cells at all, so a bar/marker at 0 would read as a measured null
    # effect rather than as an absence.  Under the corrected Tangram column
    # naming "Other neurons" wins 0/32 clusters and lands here; photoreceptors
    # and hemocytes never reach this list because they are already excluded
    # from CLUSTER_VOTE_RESOLVED.
    print(
        f"WARNING: {len(missing)} of the {len(CLASSES)} cluster-vote "
        f"categories absent from abundance_tangram_ct.csv (no cells in "
        f"either caste) -- dropped from the figure: {missing}"
    )
    ab = ab.drop(index=missing)
    CLASSES = [c for c in CLASSES if c not in missing]

# The reverse is the dangerous direction: a category with cells that CLASSES
# does not list would be silently dropped and the remaining bars renormalised,
# so panel A would stop summing to 1 while still saying it does.
_extra = sorted(set(ab.index.astype(str)) - set(CLASSES))
if _extra:
    raise AssertionError(
        "categories with cells in abundance_tangram_ct.csv that CLASSES does "
        "not list, so panel A would drop them and renormalise: %s" % _extra
    )


# Panel A's bars take their fractions from the atlas vote composition table,
# not the abundance table's prop_* columns; the category list, order and
# colours still come from CLASSES above.  A category with no atlas cells is an
# error here, not a zero: same rule as the drop above, but loud, because the
# table is supposed to carry every CLASSES entry.
_atlas = pd.read_csv(ROOT / "atlas_vote_composition_by_caste.csv")
_unk_at = sorted(set(_atlas["cluster"].astype(str)) - set(CLASSES))
assert not _unk_at, (
    "atlas_vote_composition_by_caste.csv has categories CLASSES does not "
    "list: %s" % _unk_at
)
_at = _atlas.pivot(index="cluster", columns="brain", values="frac").reindex(CLASSES)
_at_missing = _at.index[_at.isna().any(axis=1)].tolist()
assert not _at_missing, (
    "CLASSES absent from atlas_vote_composition_by_caste.csv: %s" % _at_missing
)
_n_brain = _atlas.groupby("brain")["n_brain_total"].first()


_HDR_DY = 6  # points above the axes' top-left corner (shared baseline)


def _letter(ax, letter, tag):
    """House panel header (bimp_st.style): bold letter + bold title, one baseline.

    Identical placement for every panel: offsets are in points from the axes'
    top-left corner, so no per-panel x/y tuning.
    """
    panel_header(fig, ax, letter, tag, letter_size=16, tag_size=10.5, dy_pts=_HDR_DY)


# ── Figure scaffold: A on top, B|C|D row below. ───────────────────────────────
# LaTeX includes this at \includegraphics[width=\linewidth] (fixed
# print width, auto height): growing the canvas width shrinks every absolute
# font size once scaled to the page column, while growing height is free.
# Two rows; D shares the bottom row with B and C as one shared axes.
fig = plt.figure(figsize=(13.2, 8.0))
gs = GridSpec(
    2,
    1,
    figure=fig,
    height_ratios=[0.30, 0.70],
    hspace=0.38,
    left=0.075,
    right=0.975,
    top=0.975,
    bottom=0.07,
)
# Top row: A.  Bottom row: B | C | D.  Letters are placed above each axes' top
# edge, so they cannot land on any y tick label (verify on a rendered crop).
axA_st = fig.add_subplot(gs[0])  # spatial atlas after label transfer

# ============================================================================
# PANEL A — spatial atlas after label transfer (expression-only vote)
# ============================================================================
# No shared legend and no figure-width title (house rule): every bar segment
# names itself directly.
axA = axA_st

bars = [
    (f"Queen\n{int(_n_brain['queen']):,} cells", _at["queen"].values, 1.0),
    (f"Worker\n{int(_n_brain['worker']):,} cells", _at["worker"].values, 0.0),
]
bh = 0.70
# Callouts for segments too narrow to hold their own text.  The full category
# names are wide and the two narrow categories sit ~0.02 apart on a 0-1 axis,
# so naive placement under each segment renders them on top of each other.
# Each row's callouts are spread to a minimum horizontal spacing and joined to
# their segment by a leader line; the queen row calls out above its bar and
# the worker row below, so the two rows cannot collide with each other either.
_MIN_CALLOUT_GAP = 0.125


def _ink(col, target=0.42):
    """Darken a fill colour just enough to read as text on white.

    The narrow-segment callouts sit outside the bar, on the page.  In the
    segment's own fill colour the pale ones are unreadable -- "Unannotated"
    is near-white grey (luminance 0.89).  Scaling RGB down to a target
    luminance keeps the hue, so the label still reads as belonging to its
    segment, and the leader line keeps the true fill colour so the tie to the
    bar is exact.
    """
    import matplotlib.colors as _mc

    r, g, b = _mc.to_rgb(col)
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    if lum <= target:
        return col
    f = target / max(lum, 1e-6)
    return (r * f, g * f, b * f)


def _spread(centres, gap=_MIN_CALLOUT_GAP):
    """Nudge label x-positions apart, preserving order and centroid."""
    out = list(centres)
    for _ in range(60):
        moved = False
        for i in range(len(out) - 1):
            short = gap - (out[i + 1] - out[i])
            if short > 1e-9:
                out[i] -= short / 2
                out[i + 1] += short / 2
                moved = True
        if not moved:
            break
    return [min(max(x, 0.09), 0.91) for x in out]


_IN_FS = 8.0  # in-bar label size (both bars)
_CALL_FS = 8.0  # callout size
_TICK_FS = 9.0  # y tick label size on A


def _fits(ax, text, fs, x0, x1):
    """True when *text* (bold, size fs) fits between data x0 and x1 on *ax*."""
    t = ax.text(0, 0, text, fontsize=fs, fontweight="bold")
    w = t.get_window_extent(fig.canvas.get_renderer()).width
    t.remove()
    seg = ax.transData.transform((x1, 0))[0] - ax.transData.transform((x0, 0))[0]
    return w + 30 <= seg


def _short(c):
    """Single-line display name for in-bar / callout text."""
    if c == MIXED_LABEL:
        return "Other"
    n = DISPLAY_NAME_OVERRIDES.get(str(c), get_pooled_display(c))
    n = n.replace(" (all subtypes)", "").replace("Optic lobe cells", "Optic lobe")
    n = n.replace("Astrocyte-like glia", "Astrocyte-like")
    return n


axA.set_xlim(0, 1.0)
axA.set_ylim(-1.12, 1.95)
fig.canvas.draw()


def _stack(ax, rows):
    """Draw queen/worker stacked bars with width-aware labels and callouts."""
    for _label, vals, y in rows:
        up = y > 0.5  # queen row labels above, worker below
        narrow = []
        left = 0.0
        for cls, p in zip(CLASSES, vals, strict=False):
            col = CLASS_COL[cls]
            ax.barh(
                y, p, left=left, height=bh, color=col, edgecolor="white", linewidth=1.1
            )
            # Each segment names itself; a shared colour key would be a
            # second block of text doing the same job less directly.
            if p >= LABEL_MIN_FRAC:
                name = _short(cls)
                # Width-aware: measured text extent against the segment's pixel width.
                if _fits(axA, f"{name}\n{p:.3f}", _IN_FS, left, left + p):
                    ax.text(
                        left + p / 2,
                        y,
                        f"{name}\n{p:.3f}",
                        ha="center",
                        va="center",
                        fontsize=_IN_FS,
                        color=_txt_on(col),
                        fontweight="bold",
                        linespacing=1.3,
                    )
                else:
                    narrow.append((left + p / 2, name, p, col))
            left += p

        edge = y + bh / 2 if up else y - bh / 2
        tip = edge + (0.14 if up else -0.14)
        for (anchor, name, p, col), tx in zip(
            narrow, _spread([n[0] for n in narrow]), strict=False
        ):
            ax.plot(
                [anchor, anchor, tx],
                [edge, edge + (0.06 if up else -0.06), tip],
                color=col,
                lw=0.8,
                zorder=1,
                solid_capstyle="round",
            )
            ax.text(
                tx,
                tip + (0.03 if up else -0.03),
                f"{name}\n{p:.3f}",
                ha="center",
                va="bottom" if up else "top",
                fontsize=_CALL_FS,
                color=_ink(col),
                fontweight="bold",
                linespacing=1.25,
            )


_stack(axA, bars)
axA.set_yticks([1.0, 0.0])
axA.set_yticklabels([b[0] for b in bars], fontsize=_TICK_FS)
# The y-limits leave room for the narrow-segment callouts: the worker row's
# hang below its bar and must clear the x-axis, the queen row's sit above its
# bar (re-check on a rendered crop).
axA.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
axA.set_xlabel(
    "Fraction of cells (sums to 1 within each caste)", fontsize=9, labelpad=4
)
axA.tick_params(axis="x", labelsize=_TICK_FS)
axA.tick_params(axis="y", length=0)
for sp in ("top", "right", "left"):
    axA.spines[sp].set_visible(False)
_letter(axA_st, "A", "Atlas composition")

# What the pooled categories contain, and what "Mixed" means, is caption
# material, not small grey italic on the figure.  Panel B's
# x-axis names the categories again in their own colours, so a separate key
# would be redundant.

# ── bottom row: B (per section) | C (per region) | D (regional diversity) ──
gsB = gs[1].subgridspec(1, 3, width_ratios=[1.0, 1.1, 1.25], wspace=0.38)
axC = fig.add_subplot(gsB[1])  # per-region composition

# ============================================================================
# PANEL B — per-section spread of the same class proportions
# ============================================================================
# Panel A already gives the proportions; what it cannot show is the spread.
# Each point is one tissue section, so the caste difference can be read
# against section-to-section variability inside a single brain -- the
# comparison that decides whether a difference in A means anything.  These
# are sections within one brain per caste, not biological replicates, so the
# spread is within-brain variation and no amount of it licenses a caste test.
axB = fig.add_subplot(gsB[0])
_slice_comp = pd.read_csv(ROOT / "slice_composition.csv")
_slice_comp = _slice_comp[_slice_comp.cluster.isin(CLASSES)].copy()
_slice_comp["prop"] = _slice_comp.groupby("slice_name")["prop"].transform(
    lambda s: s / s.sum()
)
x = np.arange(len(CLASSES))
_rng = np.random.default_rng(0)
_jit = 0.085
_n_zero = 0
for _i, _c in enumerate(CLASSES):
    for _brain, _col, _off in (
        ("queen", QUEEN_COLOR, -0.19),
        ("worker", WORKER_COLOR, 0.19),
    ):
        _v = _slice_comp[(_slice_comp.cluster == _c) & (_slice_comp.brain == _brain)][
            "prop"
        ].values
        if not len(_v):
            continue
        # The categories span enough dynamic range to warrant a log y-axis
        # (linear would flatten the rare ones near zero).  Log can't display
        # exact zero (a category entirely absent from a section); those
        # points are dropped from the scatter (the mean line still uses the
        # true value, including zeros, so it isn't biased up).
        _pos = _v[_v > 0]
        _n_zero += int((_v <= 0).sum())
        if len(_pos):
            axB.scatter(
                _i + _off + _rng.uniform(-_jit, _jit, len(_pos)),
                _pos,
                s=7,
                color=_col,
                alpha=0.55,
                linewidths=0,
                zorder=2,
            )
        _mean = _v.mean()
        if _mean > 0:
            axB.plot(
                [_i + _off - 0.13, _i + _off + 0.13],
                [_mean] * 2,
                color=_col,
                lw=2.0,
                solid_capstyle="butt",
                zorder=3,
            )
if _n_zero:
    print(
        f"    Panel B: {_n_zero} section x category points are exact zero "
        f"(category absent that section) -- dropped from the log-scale scatter",
        flush=True,
    )
axB.set_xticks(x)
# 45 deg rotation plus the narrow wrap keep the labels from overlapping at
# this panel width.  Check on a native-resolution crop: the whole-figure view
# downsamples 8 pt type into illegibility either way.
axB.set_xticklabels(
    [_lbl_narrow(c) for c in CLASSES],
    fontsize=7.2,
    rotation=45,
    ha="right",
    rotation_mode="anchor",
)
for _t, _c in zip(axB.get_xticklabels(), CLASSES, strict=False):
    _t.set_color(CLASS_COL[_c])
axB.set_yscale("log")
# The pseudoreplication caveat lives in the ylabel, physically part of the
# panel, rather than in a free text() below the axes.
axB.set_ylabel(
    "Proportion of cells (log scale)\n(one point = one section; "
    "one brain per caste — not replicates)",
    fontsize=8.6,
)
axB.set_xlim(-0.55, len(CLASSES) - 0.45)
# No caste key and no title: A already carries both colours and both
# section counts, and the caption owns the title (house rule).
axB.patch.set_visible(False)
_letter(axB, "B", "Per section")

# ============================================================================
# scCODA compositional test — computed and reported, not plotted
# ============================================================================
# The result is null (0 of 6 categories credible, p_adj 0.5 for every one),
# and scCODA emits an identical placeholder log2FC for every non-credible
# category, so a plot would only show the observed ratio of the means that
# panels A and B already show.  A formal caste verdict is not licensed on one
# brain per caste (see panel B).  The test is printed per category at the end
# of this script so the null stays recorded and auditable (perineurial glia
# closest, inclusion probability 0.768, not credible); SI
# fig:label_sensitivity panel C shows the full
# canonical-versus-stitched scCODA comparison with credibility markers.
if "is_reference" in ab.columns and ab["is_reference"].any():
    ref_ct = ab.index[ab["is_reference"]][0]
else:
    ref_ct = ab.index[np.isclose(ab["inclusion_prob"], 0.0)][0]


def _plot_log2fc(row):
    """Observed log2 ratio for a non-credible category.

    scCODA's stored ``log2fc`` for a category the spike-and-slab did not
    select is a shared placeholder, not a per-category estimate (checked
    against this run and a 5-class run).  This lets the console provenance
    below print each category's real position rather than six copies of the
    placeholder.
    """
    if bool(row["credible"]):
        return float(row["log2fc"])
    q, w = float(row["prop_queen"]), float(row["prop_worker"])
    return float(np.log2(w / q)) if q > 0 and w > 0 else 0.0


ab["_plot_log2fc"] = ab.apply(_plot_log2fc, axis=1)

# ============================================================================
# PANEL C — per-region composition, queen vs worker
# ============================================================================
# One table feeds the bars, the counts and the contrast
# below: results/region_composition/composition_by_region.csv
# (scripts/regions/region_composition.py). Its cells lie inside exactly one
# region (bimp_st.composition.atlas_regions) and carry the vote's display
# labels, the categories of panels A and B, so the colours carry across.
#
# Also computed here (printed, not plotted): the caste effect on
# composition against the regional one, by
# bimp_st.composition.composition_contrast, the function the numbers checker
# calls. Both arms are |log2 ratio| of one class's share between two groups of
# cells, a class entering only when its share is at least
# composition.SHARE_FLOOR in both:
#
#   caste   worker vs queen, within one region
#   region  region vs region, within one brain
#
# Descriptive on both axes (n=1 brain per caste); no p-value is attached. The
# regional arm compares regions defined by their different populations, so it
# is a scale, not a null. axC is created in the figure scaffold.
from bimp_st.composition import PANEL_REGIONS, composition_contrast  # noqa: E402

_comp = pd.read_csv(REPO / "results/region_composition/composition_by_region.csv")
_comp = _comp[_comp["region"].isin(PANEL_REGIONS)]
_contrast = composition_contrast(_comp)
_med_c, _med_r = _contrast.caste_median, _contrast.regional_median
_n_over = _contrast.n_caste_over_regional_median

# ---- the panel itself: per-region composition, queen beside worker -------
_RA_REGION = {
    "lamina": "Lamina",
    "medulla": "Medulla",
    "lobula": "Lobula",
    "kc_ic": "Small-type",
    "kc_nc": "Large-type",
    "kc_oc": "Class II",
}
_ra = _comp.pivot_table(
    index=["region", "cluster"], columns="brain", values="proportion", fill_value=0.0
).reset_index()
_ra = _ra.rename(columns={"queen": "mean_queen", "worker": "mean_worker"})
_ra["region_name"] = _ra["region"].map(_RA_REGION)
assert _ra["region_name"].notna().all(), (
    f"unmapped region in composition_by_region.csv: {sorted(set(_ra['region']))}"
)
_unlisted = sorted(set(_ra["cluster"]) - set(CLASSES))
assert not _unlisted, f"panel C classes missing from CLASSES: {_unlisted}"
_ra_regions = [r for r in REGION_ORDER if r in set(_ra["region_name"])]

_BARH, _PAIRGAP, _ROWGAP = 0.38, 0.06, 0.46
_y, _yticks, _ylabels = 0.0, [], []
for _rg in _ra_regions:
    _sub = _ra[_ra["region_name"] == _rg].set_index("cluster")
    _pair_top = _y
    # Queen first: this axes is inverted, so the row drawn first is the top
    # one, matching panel A (queen on top) and the "Q = queen, W = worker"
    # xlabel.
    for _who, _col_name in (("Q", "mean_queen"), ("W", "mean_worker")):
        _left = 0.0
        for _cls in CLASSES:
            if _cls not in _sub.index:
                continue
            _v = float(_sub.loc[_cls, _col_name])
            if _v <= 0:
                continue
            axC.barh(
                _y,
                _v,
                height=_BARH,
                left=_left,
                color=CLASS_COL[_cls],
                edgecolor="white",
                linewidth=0.35,
                zorder=2,
            )
            _left += _v
        _ann = axC.annotate(
            "$\\mathbf{" + _who + "}$",
            xy=(0, _y),
            xycoords=("axes fraction", "data"),
            xytext=(-3, 0),
            textcoords="offset points",
            fontsize=7.0,
            ha="right",
            va="center",
            color="#777777",
            fontweight="bold",
            annotation_clip=False,
        )
        _y += _BARH + _PAIRGAP
    _yticks.append((_pair_top + _y - _BARH - _PAIRGAP) / 2)
    _ylabels.append(_rg)
    _y += _ROWGAP

axC.set_yticks(_yticks)
axC.set_yticklabels(_ylabels, fontsize=8.0, fontweight="bold")
for _t, _rg in zip(axC.get_yticklabels(), _ra_regions, strict=False):
    _t.set_color(REGION_GROUP_COLOR.get(_rg, "#333333"))
axC.tick_params(axis="y", length=0, pad=12)
axC.set_ylim(-0.45, _y - _ROWGAP - _PAIRGAP + 0.07)
axC.invert_yaxis()
axC.set_xlim(0, 1)
axC.set_xticks([0, 0.5, 1.0])
axC.set_xticklabels(["0", "0.5", "1"], fontsize=7.6)
axC.set_xlabel("Fraction of the region's cells\n(Q = queen, W = worker)", fontsize=8.2)
for _s in ("top", "right", "left"):
    axC.spines[_s].set_visible(False)
# Colours are panel A's; no second legend. The caption says so.

_letter(axC, "C", "Per region")

# ============================================================================
# PANEL D — per-region Hill-number diversity profile (expression-only labels)
# ============================================================================
# Read, never recomputed: results/region_diversity/hill_diversity_profile.csv,
# scheme="expr", the rows behind SI fig_s_hill_diversity panel A (castes
# pooled there by design; per caste here -- see the docstring).  One shared
# axes, all REGION_ORDER regions as separate lines (ellipsoid_body and
# fan_shaped_body are excluded upstream, in region_diversity.py):
# colour = REGION_GROUP_COLOR (the ramp the SI panel uses for these regions),
# linestyle = caste (solid = queen, dashed = worker).  Region names are direct
# end-of-line callouts rather than legend entries, as in panel A's house
# style, which lets one axes carry all of them inside 1/3 of the row.
axD = fig.add_subplot(gsB[2])
_hill = pd.read_csv(REPO / "results/region_diversity/hill_diversity_profile.csv")
_hill["region"] = _hill["region"].map(display_region)  # cached CSV keeps the old names
_hill = _hill[(_hill["scheme"] == "expr") & (_hill["caste"] != "both")].copy()
# Fail loudly rather than plot the wrong rows: results/ is a shared symlink
# with no conflict detection, so a re-run of the analysis under a different
# region set or rarefaction depth must break this script, not quietly change
# the figure.
assert set(_hill["region"]) == set(REGION_ORDER), (
    f"hill CSV regions {sorted(set(_hill['region']))} != {REGION_ORDER}"
)
assert set(_hill["caste"]) == {"queen", "worker"}, "expected queen/worker rows only"
_n_rare = sorted(set(_hill["n_rarefied"].astype(int)))
assert len(_n_rare) == 1, f"rarefaction depth not constant across castes: {_n_rare}"
_n_rare = _n_rare[0]

_CASTE_LS = {"queen": "-", "worker": "--"}
_end_anchors = []  # (q=3 anchor y, region, color), for the end-of-line labels
for _region in REGION_ORDER:
    _rcol = REGION_GROUP_COLOR[_region]
    _q3_vals = []
    for _caste, _ls in _CASTE_LS.items():
        _d = _hill[(_hill.region == _region) & (_hill.caste == _caste)].sort_values("q")
        axD.plot(_d["q"], _d["hill_mean"], color=_rcol, ls=_ls, lw=1.7, zorder=3)
        # Bootstrap resampling interval, not biological variance: one brain
        # per caste.  Degenerates to a zero-width ribbon for a saturated
        # region/caste (e.g. worker Lobula at q=0), which is correct -- that is
        # the point-mass interval.  Low alpha: two bands per region on one
        # axes would be an opaque smear at panel A's fill alpha (0.18).
        axD.fill_between(
            _d["q"],
            _d["hill_lo"],
            _d["hill_hi"],
            color=_rcol,
            alpha=0.10,
            lw=0,
            zorder=1,
        )
        _q3_vals.append(float(_d.loc[_d["q"] == 3.0, "hill_mean"].iloc[0]))
    _end_anchors.append((sum(_q3_vals) / len(_q3_vals), _region, _rcol))

for _qm in (0, 1, 2, 3):
    axD.axvline(_qm, ls=":", lw=0.5, color="#DADADA", zorder=0)

# ---- inset: the plateau, magnified ------------------------------------
# Five of the seven regions land between 1.0 and 1.25 at q>=2 (at q=1 the
# Large-type compartment and the lamina sit at ~1.33-1.37) and cannot be
# told apart on the main axes. They are genuinely that close (an effective
# count near 1 means one type dominates the region), so the honest fix is to
# magnify that band rather than rescale the axis and imply more separation
# than exists. q is taken from 1 rather than 0 because the q=0 richness of
# every region is 4-5, and including it would force the inset back to the
# range it exists to escape.
_axDi = axD.inset_axes([0.455, 0.50, 0.52, 0.44])
_ins_lo, _ins_hi = 1.0, 1.0
for _region in REGION_ORDER:
    _rcol = REGION_GROUP_COLOR[_region]
    for _caste, _ls in _CASTE_LS.items():
        _d = _hill[(_hill.region == _region) & (_hill.caste == _caste)]
        _d = _d[_d.q >= 1.0].sort_values("q")
        if _d.empty:
            continue
        _axDi.plot(_d["q"], _d["hill_mean"], color=_rcol, ls=_ls, lw=1.5, zorder=3)
        _ins_hi = max(_ins_hi, float(_d["hill_mean"].max()))
        _ins_lo = min(_ins_lo, float(_d["hill_mean"].min()))
# Clip to the crowded band: the medulla and KC-OC are already legible on the
# main axes, so letting them set the inset's ceiling would re-compress the
# five regions the inset is for.
_axDi.set_ylim(_ins_lo - 0.02, 1.34)
# xlim runs past q=3 only to open a label gutter; the curves still stop at 3
# and the ticks still mark 1/2/3, so nothing implies data beyond q=3.
_axDi.set_xlim(1, 4.35)
_axDi.set_xticks([1, 2, 3])
_axDi.set_yticks([1.0, 1.1, 1.2, 1.3])
_axDi.tick_params(labelsize=6.4, length=2.4, pad=1.5)
_axDi.set_title("$q \\geq 1$, magnified", fontsize=7.0, pad=2.5)
for _sp in ("top", "right"):
    _axDi.spines[_sp].set_visible(False)
for _sp in ("left", "bottom"):
    _axDi.spines[_sp].set_linewidth(0.7)
_axDi.patch.set_alpha(0.0)

# Region labels on the magnified inset as well as the main axes. Without them
# the inset is seven same-shaped lines whose identity can only be recovered by
# matching colour back to the main panel, which is the one thing the
# magnification exists to make unnecessary. Only the regions the inset
# actually shows are labelled: the medulla sits at 1.92 at q=3, far above the
# 1.34 ceiling, so a label for it here would point at a curve that is not in
# the box.
#
# Spread is linear here, unlike _spread_y on the main axes: that one works in
# log space because axD is log-scaled, and reusing it on this linear inset
# would bunch the labels toward the bottom.
_ins_top = 1.34
_ins_bot = _ins_lo - 0.02
_ins_anchors = [(a, r, c) for (a, r, c) in _end_anchors if a < _ins_top]
_ins_anchors.sort(key=lambda t_: t_[0])


def _spread_linear(centres, gap, lo, hi):
    out = [float(c) for c in centres]
    for _ in range(80):
        moved = False
        for i in range(len(out) - 1):
            short = gap - (out[i + 1] - out[i])
            if short > 1e-9:
                out[i] -= short / 2
                out[i + 1] += short / 2
                moved = True
        if not moved:
            break
    return [min(max(y, lo), hi) for y in out]


_ins_span = _ins_top - _ins_bot
_ins_ys = _spread_linear(
    [a[0] for a in _ins_anchors],
    gap=_ins_span / (len(_ins_anchors) + 1.15),
    lo=_ins_bot + _ins_span * 0.04,
    hi=_ins_top - _ins_span * 0.04,
)
for (_anchor_y, _region, _rcol), _ly in zip(_ins_anchors, _ins_ys, strict=False):
    _axDi.plot(
        [3.0, 3.06, 3.16],
        [_anchor_y, _anchor_y, _ly],
        color=_rcol,
        lw=0.55,
        zorder=2,
        solid_capstyle="round",
        clip_on=False,
    )
    _axDi.text(
        3.22,
        _ly,
        _region,
        fontsize=5.6,
        color=_rcol,
        fontweight="bold",
        ha="left",
        va="center",
        clip_on=False,
    )

# Direct end-of-line region labels, spread apart with the same leader-line
# technique panel A uses for its narrow segments (_spread), applied to y:
# most regions land within 1.0-1.4 effective types at q=3, so labels at the
# literal endpoints would stack.  y_lo sits well below the lowest curve
# (~0.85-1.0) so the spread has room; clamping a spread-apart set back
# against a hard floor re-collides the labels near it.  _spread_y's gap and
# clamp constants were tuned on a six-region render; with more regions,
# check the rendered PNG for collisions.
_end_anchors.sort(key=lambda t: t[0])
_y_lo, _y_hi = 0.92, float(_hill["hill_hi"].max()) * 1.12


def _spread_y(centres, gap=0.030):
    """Push overlapping end labels apart in log space.

    The axis is logarithmic, so a constant gap in data units is a large
    visual gap at the top of the panel and an invisible one at the
    bottom, which is exactly where the labels collide.
    """
    import numpy as _np

    out = [float(_np.log10(c)) for c in centres]
    for _ in range(80):
        moved = False
        for i in range(len(out) - 1):
            short = gap - (out[i + 1] - out[i])
            if short > 1e-9:
                out[i] -= short / 2
                out[i + 1] += short / 2
                moved = True
        if not moved:
            break
    lo, hi = float(_np.log10(_y_lo)), float(_np.log10(_y_hi))
    return [10 ** min(max(y, lo + 0.004), hi - 0.004) for y in out]


_label_x = 3.10
for (_anchor_y, _region, _rcol), _ly in zip(
    _end_anchors, _spread_y([a[0] for a in _end_anchors]), strict=False
):
    axD.plot(
        [3.0, 3.02, _label_x - 0.03],
        [_anchor_y, _anchor_y, _ly],
        color=_rcol,
        lw=0.7,
        zorder=2,
        solid_capstyle="round",
    )
    axD.text(
        _label_x,
        _ly,
        _region,
        fontsize=8.0,
        color=_rcol,
        fontweight="bold",
        ha="left",
        va="center",
    )

axD.set_xlim(0, 3.85)
axD.set_yscale("log")
axD.set_ylim(_y_lo, _y_hi)
# Hill numbers bunch between 1 and 2 for every region once q >= 1, so a
# linear axis spends most of its height on the sparse 2-5 range and
# overplots the part being read. Explicit ticks because a log axis
# defaults to decades, and there is only one here.
axD.set_yticks([1, 1.5, 2, 3, 4, 5])
axD.set_yticklabels(["1", "1.5", "2", "3", "4", "5"])
axD.minorticks_off()
axD.set_xticks([0, 1, 2, 3])
axD.set_xlabel("Hill order $q$", fontsize=9, labelpad=15)
# Name the integer orders under their tick numbers (small regular text).
for _qx, _qn in ((0, "Richness"), (1, "Shannon"), (2, "Simpson")):
    axD.annotate(
        _qn,
        xy=(_qx, 0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -17),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=6.8,
        color="#555555",
        annotation_clip=False,
    )
axD.set_ylabel("$^qD$ (effective # types)", fontsize=9)
axD.spines[["top", "right"]].set_visible(False)
_letter(axD, "D", "Hill diversity")

# Caste is carried by linestyle, not color (color is already spent on region
# identity), so it needs its own small key -- unlike the region names, solid
# vs. dashed cannot be read directly off a line end.
_hqD = Line2D([], [], color="#444444", ls="-", lw=1.7, label="Queen")
_hwD = Line2D([], [], color="#444444", ls="--", lw=1.7, label="Worker")
axD.legend(
    handles=[_hqD, _hwD],
    loc="upper left",
    frameon=False,
    fontsize=7.6,
    handlelength=1.8,
    borderaxespad=0.3,
)

# On-panel text budget: the rarefaction depth, bootstrap-vs-biological
# variance, Mixed handling and the expression-only labelling are all stated
# in the figure caption, so none is repeated here.

fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
print(f"saved {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
print(f"  reference category: {ref_ct}")
print(
    f"  panel C: caste |log2| median {_med_c:.3f} (n={_contrast.n_caste}), "
    f"region {_med_r:.3f} (n={_contrast.n_regional}), "
    f"{_contrast.ratio:.1f}x; {_n_over} caste contrasts reach the "
    f"regional median"
)
for cls, row in ab.iterrows():
    tag = "REF" if cls == ref_ct else ("cred" if row["credible"] else "not-cred")
    print(
        f"  {cls:20s} propQ={row['prop_queen']:.4f} propW={row['prop_worker']:.4f} "
        f"scCODA_log2fc={row['log2fc']:+.3f} plotted_log2fc={row['_plot_log2fc']:+.3f} "
        f"[{tag}]"
    )
# Panel D's Hill profile, printed at q=0/1/2 so a run can be checked against
# hill_diversity_profile.csv (and against the audited values) without
# opening the PNG.
print(f"  panel D: Hill diversity, scheme=expr, rarefied to N={_n_rare:,}/caste")
for _r in REGION_ORDER:
    for _qcheck in (0.0, 1.0, 2.0):
        _q = _hill[
            (_hill.region == _r) & (_hill.caste == "queen") & (_hill.q == _qcheck)
        ].iloc[0]
        _w = _hill[
            (_hill.region == _r) & (_hill.caste == "worker") & (_hill.q == _qcheck)
        ].iloc[0]
        print(
            f"    {_r:8s} q={_qcheck:.0f} queen={_q['hill_mean']:.4f} "
            f"[{_q['hill_lo']:.4f},{_q['hill_hi']:.4f}]  "
            f"worker={_w['hill_mean']:.4f} "
            f"[{_w['hill_lo']:.4f},{_w['hill_hi']:.4f}]"
        )
