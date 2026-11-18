#!/usr/bin/env python3
"""Supplementary figure: Tangram label-transfer confidence.

The mapping stores a full 29-class posterior per cell (``obsm/tangram_ct_pred``)
and derived per-cell statistics (top-1 probability, top1-top2 margin, Shannon
entropy, and the confident/ambiguous QC flag). This figure reports all of
it (panels A-E of figS_tangram_confidence), and the stats JSON also carries
the Tangram gene-level cross-validation score from the eval-metrics JSON and
the per-cluster modal-label consensus (not drawn).

    PYTHONPATH=src .venv/bin/python scripts/identity/si_tangram_confidence.py
"""

import json
import warnings
from types import SimpleNamespace

import h5py
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from anndata.io import read_elem
from matplotlib.gridspec import GridSpec

from bimp_st.nine_region_common import DESCRIPTIVE_REGIONS, attach_region_flags
from bimp_st.nine_region_common import display as region_display
from bimp_st.paths import CANONICAL_TANGRAM
from bimp_st.si_common import (
    LEGEND_KW,
    OUTPUT_DIR,
    PIXEL_SCALE,
    QUEEN_COLOR,
    TICK_SIZE,
    WORKER_COLOR,
    add_panel_headers,
    set_style,
    strip_spines,
)

warnings.filterwarnings("ignore", message="Observation names are not unique")

# ── Inputs ───────────────────────────────────────────────────────────────────

## Must be the canonical object: panels D/E/F break confidence down by
## cell-type name, so cell-type columns must be keyed by name, not position.
## Panels A-C (aggregate per-cell stats) do not depend on the names.
TANGRAM_H5AD = str(CANONICAL_TANGRAM)
EVAL_METRICS = "results/tangram/combined_clustered_g2_R2_s42_eval_metrics.json"

# Thresholds hard-coded in pipelines/combined_atlas/tangram_mapping.py
# (LABEL_QC_TOP1_MIN / LABEL_QC_MARGIN_MIN) and used to set tangram_label_qc.
TOP1_MIN = 0.30
MARGIN_MIN = 0.10

CONF_COLOR = "#2C7A6B"
AMB_COLOR = "#C97B3C"
CASTE_COLOR = {"queen": QUEEN_COLOR, "worker": WORKER_COLOR}


# ── Loading ──────────────────────────────────────────────────────────────────


def load_data():
    with h5py.File(TANGRAM_H5AD, "r") as h:
        obs = read_elem(h["obs"])
        umap = read_elem(h["obsm/X_umap"])
        spatial = read_elem(h["obsm/spatial"])
        ct_pred = read_elem(h["obsm/tangram_ct_pred"])

    # This object predates the antennal lobe and central complex, so it carries
    # only the original six in_* columns. Join the rest on by
    # (slice_name, cell_id); attach_region_flags takes anything exposing .obs.
    attach_region_flags(SimpleNamespace(obs=obs), regions=DESCRIPTIVE_REGIONS)
    with open(EVAL_METRICS) as f:
        metrics = json.load(f)
    print(f"  {len(obs):,} cells; posterior matrix {np.shape(ct_pred)}", flush=True)
    print(f"  eval metrics: {metrics}", flush=True)
    return obs, umap, spatial, ct_pred, metrics


# ── Figure ───────────────────────────────────────────────────────────────────


# One 7.2 in canvas, saved straight to the registered stem
# (figS_tangram_confidence). Panels A-C (per-cell posteriors) sit above D-E
# (per-type confidence) on a canvas short enough for one page with its caption.
# No cluster-consensus panel: Fig. percell_vs_vote makes that point (per-cell
# argmax labels are weakly resolved, hence the cluster vote). Its numbers are
# computed and written to the stats JSON.
FIG_W = 7.2
H_AC = 5.6  # inches given to panels A-C
ROW_IN = 0.115  # inches per reference type in D/E
HIST_COLOR = "#4A6C8C"
QC_COLOR = "#D95F02"


def _hdr_anchor(fig, ax):
    """Zero-height invisible axes at the left edge of ``ax``'s gridspec cell.

    Equal-aspect axes shrink inside their cell by different amounts; a letter
    anchored on one is indented relative to its neighbours'. Anchoring on the
    cell instead keeps every letter on the same left edge and baseline.
    """
    ss = ax.get_subplotspec()
    pos = ss.get_position(fig)
    h = fig.add_axes([pos.x0, pos.y1, 1e-3, 1e-3])
    h.axis("off")
    h.set_xticks([])  # no phantom tick labels for print_scale
    h.set_yticks([])
    return h


def _left_anchor(fig, ax):
    """Header anchor at the figure's left edge, level with ``ax``'s top.

    D's axes start right of a wide band of row labels; its letter belongs at
    the page edge with F's, not indented by the label band.
    """
    h = fig.add_axes([0.004, ax.get_position().y1, 1e-3, 1e-3])
    h.axis("off")
    h.set_xticks([])  # no phantom tick labels for print_scale
    h.set_yticks([])
    return h


def build_figure_a(fig, obs, umap, spatial, n_classes, top, bottom):
    """Panels A-C: per-cell posteriors, drawn into the top of ``fig``."""
    chance = 1.0 / n_classes
    conf = obs["tangram_confidence"].to_numpy(dtype=float)
    margin = obs["tangram_margin"].to_numpy(dtype=float)
    entropy = obs["tangram_entropy"].to_numpy(dtype=float)
    is_conf = obs["tangram_label_qc"].astype(str).to_numpy() == "confident"

    gs = GridSpec(
        3,
        1,
        figure=fig,
        hspace=0.80,
        top=top,
        bottom=bottom,
        left=0.09,
        right=0.985,
        height_ratios=[0.9, 1.0, 1.4],
    )

    # ── A: distributions ─────────────────────────────────────────────────────
    gsa = gs[0].subgridspec(1, 3, wspace=0.42)
    leg_kw = dict(LEGEND_KW, handlelength=1.2, handletextpad=0.4, borderaxespad=0.2)

    ax = fig.add_subplot(gsa[0])
    hdr = {"A": ax}
    ax.hist(conf, bins=120, color=HIST_COLOR, range=(0, 1))
    ax.axvline(TOP1_MIN, color=QC_COLOR, ls="--", lw=1.0, label=f"QC = {TOP1_MIN}")
    ax.axvline(
        chance,
        color="0.35",
        ls=":",
        lw=1.0,
        label=f"chance = {chance:.3f}",
    )
    ax.axvline(
        np.median(conf), color="black", lw=1.0, label=f"median = {np.median(conf):.3f}"
    )
    ax.set_xlabel("Top-1 posterior probability")
    ax.set_ylabel("Cells (k)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.04, 1.02), **leg_kw)

    ax = fig.add_subplot(gsa[1])
    ax.hist(margin, bins=120, color=HIST_COLOR, range=(0, 1))
    ax.axvline(
        MARGIN_MIN,
        color=QC_COLOR,
        ls="--",
        lw=1.0,
        label=f"QC = {MARGIN_MIN}",
    )
    ax.axvline(
        np.median(margin),
        color="black",
        lw=1.0,
        label=f"median = {np.median(margin):.3f}",
    )
    ax.set_xlabel("Margin (top-1 − top-2)")
    ax.set_ylabel("Cells (k)")
    ax.legend(loc="upper right", **leg_kw)

    ax = fig.add_subplot(gsa[2])
    ax.hist(entropy, bins=120, color=HIST_COLOR)
    ax.axvline(
        np.log(n_classes),
        color="0.35",
        ls=":",
        lw=1.0,
        label=f"uniform = {np.log(n_classes):.2f}",
    )
    ax.axvline(
        np.median(entropy),
        color="black",
        lw=1.0,
        label=f"median = {np.median(entropy):.3f}",
    )
    ax.set_xlabel("Shannon entropy (nats)")
    ax.set_ylabel("Cells (k)")
    ax.legend(loc="upper left", **leg_kw)
    # Counts in thousands: integer ticks, and short enough not to crowd the
    # narrow columns. Axis only -- the histograms are unchanged.
    for a in fig.axes:
        a.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _p: f"{v / 1000:.0f}")
        )

    # ── B: confidence on the shared UMAP ─────────────────────────────────────
    gsb = gs[1].subgridspec(1, 3, wspace=0.42)
    x, y = umap[:, 0], umap[:, 1]

    def _hex(
        ax,
        vals,
        title,
        cb_label,
        cmap="viridis",
        reduce=np.median,
        vmin=None,
        vmax=None,
    ):
        hb = ax.hexbin(
            x,
            y,
            C=np.asarray(vals, dtype=float),
            gridsize=120,
            reduce_C_function=reduce,
            cmap=cmap,
            linewidths=0.0,
            mincnt=3,
            vmin=vmin,
            vmax=vmax,
        )
        cb = plt.colorbar(hb, ax=ax, fraction=0.06, pad=0.03, aspect=18)
        cb.ax.tick_params(labelsize=TICK_SIZE, length=2)
        cb.set_label(cb_label, fontsize=TICK_SIZE)
        cb.outline.set_visible(False)
        ax.set_title(title, fontsize=TICK_SIZE, fontweight="bold", pad=3)
        strip_spines(ax)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    ax = fig.add_subplot(gsb[0])
    hdr["B"] = ax
    _hex(ax, conf, "Top-1 probability", "Median top-1")
    ax = fig.add_subplot(gsb[1])
    _hex(ax, margin, "Margin (top-1 − top-2)", "Median margin")
    ax = fig.add_subplot(gsb[2])
    _hex(
        ax,
        is_conf.astype(float),
        "Confident-label fraction",
        "Fraction passing QC",
        cmap="cividis",
        reduce=np.mean,
        vmin=0,
        vmax=1,
    )

    # ── C: confidence in anatomical space ────────────────────────────────────
    gsc_outer = gs[2].subgridspec(1, 2, wspace=0.62, width_ratios=[2.15, 1.25])
    gsc = gsc_outer[0].subgridspec(1, 2, wspace=0.04)
    # Both sections are shown in one shared data window, so both panels have
    # the same µm-per-inch, the scale bars are comparable, and the real size
    # difference between sections stays visible.
    sec_masks = {}
    for brain in ["queen", "worker"]:
        # obs_names are not unique -> index with a boolean mask, never .loc
        bm = (obs["brain"] == brain).to_numpy()
        counts = obs["slice_name"][bm].value_counts()
        sl = counts.index[0]
        sec_masks[brain] = (obs["slice_name"] == sl).to_numpy() & bm
    half_span = 0.53 * max(
        spatial[m, k].max() - spatial[m, k].min()
        for m in sec_masks.values()
        for k in (0, 1)
    )

    c_axes = []
    c_vmin, c_vmax = np.percentile(conf, 1), np.percentile(conf, 99)
    sc = None
    for i, brain in enumerate(["queen", "worker"]):
        m = sec_masks[brain]
        ax = fig.add_subplot(gsc[i])
        c_axes.append(ax)
        hdr.setdefault("C", ax)
        sc = ax.scatter(
            spatial[m, 0],
            spatial[m, 1],
            c=conf[m],
            s=0.12,
            cmap="viridis",
            vmin=c_vmin,
            vmax=c_vmax,
            linewidths=0,
            rasterized=True,
        )
        cx = 0.5 * (spatial[m, 0].min() + spatial[m, 0].max())
        cy = 0.5 * (spatial[m, 1].min() + spatial[m, 1].max())
        ax.set_xlim(cx - half_span, cx + half_span)
        ax.set_ylim(cy + half_span, cy - half_span)  # y inverted in place of
        ax.set_aspect("equal")  # invert_yaxis()
        ax.set_anchor("SW")
        strip_spines(ax)
        ax.set_title(
            brain.capitalize(),
            color=CASTE_COLOR[brain],
            fontsize=TICK_SIZE,
            fontweight="bold",
            pad=3,
        )
        print(
            f"  C: {brain} largest section {m.sum():,} cells, "
            f"{100 * is_conf[m].mean():.1f}% confident",
            flush=True,
        )
        # obsm['spatial'] is in Xenium pixels (PIXEL_SCALE µm/px)
        bar = 1000.0 / PIXEL_SCALE
        # Anchored to the shared window, not to this section's own extent, so
        # the two bars sit at the same place and are the same length.
        x0 = cx - half_span + 0.06 * 2 * half_span
        y0 = cy + half_span - 0.06 * 2 * half_span
        ax.plot(
            [x0, x0 + bar],
            [y0, y0],
            color="black",
            lw=1.6,
            solid_capstyle="butt",
            clip_on=False,
        )
        # Label below the bar (y is inverted, so a negative point offset
        # moves it down the page) so it never sits on the bar itself.
        ax.annotate(
            "1 mm",
            (x0 + bar / 2, y0),
            xytext=(0, -3),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=TICK_SIZE,
            annotation_clip=False,
        )
    # Both sections share one colour range, so they share one colourbar.
    # Explicit inset to the right of the worker section, so it neither steals
    # width from one section only nor lands on the bar chart's labels.
    cax = c_axes[1].inset_axes([1.02, 0.12, 0.045, 0.72])
    cb = plt.colorbar(sc, cax=cax)
    cb.ax.tick_params(labelsize=TICK_SIZE, length=2)
    cb.set_label("Top-1 probability", fontsize=TICK_SIZE)
    cb.outline.set_visible(False)

    # Confidence by anatomical annotation (GeoJSON flags carried in obs)
    ax = fig.add_subplot(gsc_outer[1])
    c_axes.append(ax)
    anat = _anat_table(obs, conf, is_conf)
    yy = np.arange(len(anat))[::-1]
    ax.barh(yy, anat["median_conf"], color=HIST_COLOR, height=0.68)
    ax.axvline(np.median(conf), color=QC_COLOR, ls="--", lw=1.0)
    # n labels sit in their own right-hand column, clear of both the bars and
    # the dashed all-cells median; the median's value goes in the x label
    # instead of a legend that had to share a corner with the bottom n label.
    x_hi = np.nanmax(anat["median_conf"]) * 2.05
    for yi, (_, r) in zip(yy, anat.iterrows(), strict=False):
        ax.text(
            x_hi * 0.99,
            yi,
            f"n = {int(r['n']):,}",
            va="center",
            ha="right",
            fontsize=TICK_SIZE,
            color="0.3",
        )
    ax.set_yticks(yy)
    ax.set_yticklabels(anat["label"])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel(f"Median top-1 (dashed: all cells, {np.median(conf):.3f})")
    ax.set_xlim(0, x_hi)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.grid(axis="x", lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)

    tags = {
        "A": "Confidence distributions",
        "B": "Confidence on UMAP",
        "C": "Confidence in anatomy",
    }
    # Every letter on the same left edge: anchored on the gridspec cell of
    # its first axes, lifted clear of the B/C sub-titles.
    add_panel_headers(
        fig, [(_hdr_anchor(fig, hdr["A"]), "A", tags["A"])], width_frac=1.0, dy_pts=6
    )
    add_panel_headers(
        fig,
        [(_hdr_anchor(fig, hdr[k]), k, tags[k]) for k in "BC"],
        width_frac=1.0,
        dy_pts=17,
    )
    return anat


def _anat_table(obs, conf, is_conf):
    anat_cols = [f"in_{r}" for r in DESCRIPTIVE_REGIONS]
    # Hyphenated KC-OC/KC-NC/KC-IC is the canonical short form used in every
    # figure. Display strings come from nine_region_common.display(),
    # the single source for all nine regions.
    anat_lab = [region_display(r) for r in DESCRIPTIVE_REGIONS]

    # The GeoJSON region flags are not mutually exclusive (e.g. 86 cells in
    # KC-NC∩KC-IC, 13 in KC-OC∩KC-NC), so raw per-flag counts double-count.
    # Each cell is counted once: in its single region, in an "in >=2 regions"
    # row, or in "Unannotated". The {n_flags==1} / {n_flags>1} / {n_flags==0}
    # split partitions the atlas exactly (asserted below) and matches the
    # exclusive counts of fig_s_regional_diversity.
    #
    # in_al / in_ellipsoid_body / in_fan_shaped_body are nullable booleans --
    # pd.NA means "no tracing for this region yet", not "outside it" (see
    # nine_region_common.py); ``.to_numpy(dtype=bool)`` alone would raise.
    # na_value=False is a display/count simplification (this panel pools both
    # castes, so no caste comparison is biased): untraced cells are not
    # counted for that region, so "Unannotated" includes a few cells that are
    # only untraced (same as fig_atlas_overview.get_region_labels).
    # np.median([]) returns nan rather than raising, and the small regions
    # (ellipsoid_body 975 cells, fan_shaped_body 1,004) can have zero
    # single-flag cells; an unguarded nan would corrupt the x-axis limit via
    # `max(anat["median_conf"])`, so guard explicitly.
    def _median_or_nan(vals):
        return float(np.median(vals)) if vals.size else float("nan")

    def _mean_or_nan(vals):
        return float(vals.mean()) if vals.size else float("nan")

    flags = obs[anat_cols].to_numpy(dtype=bool, na_value=False)
    n_flags = flags.sum(axis=1)
    rows = []
    for i, (_col, lab) in enumerate(zip(anat_cols, anat_lab, strict=False)):
        m = flags[:, i] & (n_flags == 1)
        rows.append((lab, m.sum(), _median_or_nan(conf[m]), _mean_or_nan(is_conf[m])))
    both = n_flags > 1
    if both.any():
        rows.append(
            (
                "In ≥2 regions",
                both.sum(),
                _median_or_nan(conf[both]),
                _mean_or_nan(is_conf[both]),
            )
        )
    other = n_flags == 0
    rows.append(
        (
            "Unannotated",
            other.sum(),
            _median_or_nan(conf[other]),
            _mean_or_nan(is_conf[other]),
        )
    )
    anat = pd.DataFrame(rows, columns=["label", "n", "median_conf", "frac_conf"])
    assert int(anat["n"].sum()) == len(obs), (
        f"anatomical-region partition sums to {int(anat['n'].sum()):,}, "
        f"not the atlas total {len(obs):,}"
    )
    return anat


def type_summary(obs):
    """Per reference type: mean/median/IQR of top-1, n, confident fraction."""
    conf = obs["tangram_confidence"].to_numpy(dtype=float)
    is_conf = obs["tangram_label_qc"].astype(str).to_numpy() == "confident"
    ct = obs["tangram_cell_type"].astype(str)
    df = pd.DataFrame({"ct": ct.to_numpy(), "conf": conf, "is_conf": is_conf})
    g = df.groupby("ct", observed=True)
    return pd.DataFrame(
        {
            "mean": g["conf"].mean(),
            "median": g["conf"].median(),
            "q25": g["conf"].quantile(0.25),
            "q75": g["conf"].quantile(0.75),
            "n": g.size(),
            "frac_conf": g["is_conf"].mean(),
        }
    ).sort_values("mean", ascending=False)


def cluster_consensus(obs):
    """Share of each Leiden cluster's cells at its modal fine label (not drawn)."""
    conf = obs["tangram_confidence"].to_numpy(dtype=float)
    ct = obs["tangram_cell_type"].astype(str)
    lei = obs["leiden"].astype(str)
    cdf = pd.DataFrame({"leiden": lei.to_numpy(), "ct": ct.to_numpy(), "conf": conf})
    agree = cdf.groupby("leiden", observed=True).apply(
        lambda d: (d["ct"] == d["ct"].value_counts().idxmax()).mean(),
        include_groups=False,
    )
    cl_conf = cdf.groupby("leiden", observed=True)["conf"].mean()
    cl_n = cdf.groupby("leiden", observed=True).size()
    return agree, cl_conf, cl_n


def build_figure_b(fig, obs, summ, n_classes, y0, fig_h):
    """Panels D-E: per-type confidence, drawn ``y0`` inches below the top."""
    chance = 1.0 / n_classes
    is_conf = obs["tangram_label_qc"].astype(str).to_numpy() == "confident"
    frac_conf = float(is_conf.mean())

    # D and E are horizontal and side by side on one shared row axis: the
    # cell-type names are written once, horizontally (no rotated labels),
    # and both panels read off the same rows.
    n_ct = len(summ)
    rows_in = n_ct * ROW_IN
    top_in = y0 + 0.30

    def _axes_in(x, y_top, w, h):
        return fig.add_axes(
            [x / FIG_W, (fig_h - y_top - h) / fig_h, w / FIG_W, h / fig_h]
        )

    lab_in = 2.12
    de_gap = 0.22
    de_w = (FIG_W - lab_in - de_gap - 0.06) / 2
    ax_d = _axes_in(lab_in, top_in, de_w, rows_in)
    ax_e = _axes_in(lab_in + de_w + de_gap, top_in, de_w, rows_in)
    ys = np.arange(n_ct)[::-1]

    # ── D ──
    ax = ax_d
    ax.barh(ys, summ["mean"], color=HIST_COLOR, height=0.72, label="mean (bar)")
    ax.hlines(
        ys,
        summ["q25"],
        summ["q75"],
        color="black",
        lw=0.9,
        label="IQR",
    )
    ax.scatter(
        summ["median"],
        ys,
        color="white",
        edgecolor="black",
        s=7,
        zorder=4,
        linewidths=0.6,
        label="median",
    )
    ax.axvline(TOP1_MIN, color=QC_COLOR, ls="--", lw=1.0, label=f"QC = {TOP1_MIN}")
    ax.axvline(chance, color="0.35", ls=":", lw=1.0, label=f"chance = 1/{n_classes}")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c} ({int(summ.loc[c, 'n']):,})" for c in summ.index])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_ylabel("Reference cell type (cells)")
    ax.set_xlabel("Top-1 posterior probability")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(-0.6, n_ct - 0.4)
    ax.grid(axis="x", lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.40 / rows_in),
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.4,
        columnspacing=0.8,
        **{k: v for k, v in LEGEND_KW.items()},
    )

    # ── E ──
    ax = ax_e
    fc = summ["frac_conf"].to_numpy() * 100
    ax.barh(ys, fc, color=CONF_COLOR, height=0.72, label="confident")
    ax.barh(ys, 100 - fc, left=fc, color=AMB_COLOR, height=0.72, label="ambiguous")
    ax.axvline(
        100 * frac_conf,
        color="black",
        ls="--",
        lw=1.0,
        label=f"all cells: {100 * frac_conf:.1f}%",
    )
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Cells (%)")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, n_ct - 0.4)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, -0.40 / rows_in),
        ncol=2,
        borderaxespad=0.0,
        handlelength=1.4,
        columnspacing=0.8,
        **LEGEND_KW,
    )

    add_panel_headers(
        fig,
        [
            (_left_anchor(fig, ax_d), "D", "Confidence by cell type"),
            (ax_e, "E", "Confident fraction"),
        ],
        width_frac=1.0,
        dy_pts=6,
    )


def build_figure(obs, umap, spatial, ct_pred, metrics, output_dir):
    n_classes = int(np.shape(ct_pred)[1])
    summ = type_summary(obs)
    # D/E block: header, one row per type, then the two legends underneath.
    h_de = 0.30 + len(summ) * ROW_IN + 0.80
    fig_h = H_AC + h_de
    fig = plt.figure(figsize=(FIG_W, fig_h))
    anat = build_figure_a(
        fig,
        obs,
        umap,
        spatial,
        n_classes,
        top=1 - 0.24 / fig_h,
        bottom=(h_de + 0.42) / fig_h,
    )
    build_figure_b(fig, obs, summ, n_classes, H_AC, fig_h)
    out = output_dir / "figS_tangram_confidence.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"    Saved {out} ({FIG_W} x {fig_h:.2f} in)", flush=True)
    agree, cl_conf, cl_n = cluster_consensus(obs)
    return summ, anat, agree, cl_conf, cl_n


def report(obs, summ, anat, agree, metrics, n_classes):
    conf = obs["tangram_confidence"].to_numpy(dtype=float)
    margin = obs["tangram_margin"].to_numpy(dtype=float)
    entropy = obs["tangram_entropy"].to_numpy(dtype=float)
    is_conf = (obs["tangram_label_qc"].astype(str) == "confident").to_numpy()
    print("\n── Per-cell label confidence ──", flush=True)
    for name, v in [("top-1", conf), ("margin", margin), ("entropy (nats)", entropy)]:
        print(
            f"  {name}: median {np.median(v):.4f}  "
            f"mean {v.mean():.4f}  "
            f"IQR {np.percentile(v, 25):.4f}–{np.percentile(v, 75):.4f}  "
            f"range {v.min():.4g}–{v.max():.4g}",
            flush=True,
        )
    print(f"  chance (1/{n_classes}) = {1 / n_classes:.4f}", flush=True)
    print(
        f"  confident: {int(is_conf.sum()):,} / {len(obs):,} "
        f"({100 * is_conf.mean():.2f}%)",
        flush=True,
    )
    for b in ["queen", "worker"]:
        m = (obs["brain"] == b).to_numpy()
        print(
            f"    {b}: median top-1 {np.median(conf[m]):.4f}, "
            f"{100 * is_conf[m].mean():.2f}% confident "
            f"(n = {m.sum():,})",
            flush=True,
        )

    print("\n── Per-cell-type (top / bottom 5 by mean top-1) ──", flush=True)
    print(summ.head(5).to_string(), flush=True)
    print(summ.tail(5).to_string(), flush=True)

    print("\n── Confidence by anatomical annotation ──", flush=True)
    print(anat.to_string(index=False), flush=True)

    a = agree * 100
    print(
        "\n── Cluster-level consensus (% cells at cluster modal label) ──", flush=True
    )
    print(f"  median {a.median():.1f}%  range {a.min():.1f}–{a.max():.1f}%", flush=True)

    # Machine-readable copy of the summary statistics, so they can be
    # recomputed from data. Largest section per brain (the spatial panels
    # of panel C): cell count and confident fraction.
    largest_section_stats = {}
    for b in ["queen", "worker"]:
        bm = (obs["brain"] == b).to_numpy()
        sl = obs["slice_name"][bm].value_counts().index[0]
        m = (obs["slice_name"] == sl).to_numpy() & bm
        largest_section_stats[f"largest_section_{b}_n_cells"] = int(m.sum())
        largest_section_stats[f"largest_section_{b}_pct_confident"] = float(
            100 * is_conf[m].mean()
        )
    stats = {
        "median_top1": float(np.median(conf)),
        "median_margin": float(np.median(margin)),
        "median_entropy_nats": float(np.median(entropy)),
        "pct_confident": float(100 * is_conf.mean()),
        "cluster_consensus_median_pct": float(a.median()),
        "cluster_consensus_min_pct": float(a.min()),
        "cluster_consensus_max_pct": float(a.max()),
        "n_leiden_clusters": int(len(a)),
        **largest_section_stats,
        "cv_avg_test_score": float(metrics["avg_test_score"]),
        "cv_auc_score": (
            None
            if not np.isfinite(metrics["auc_score"])
            else float(metrics["auc_score"])
        ),
        "tangram_object": TANGRAM_H5AD,
    }
    with open(OUTPUT_DIR / "fig_s_tangram_confidence_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n── Tangram cross-validation metrics (unreported until now) ──", flush=True)
    for k, v in metrics.items():
        print(f"  {k} = {v:.4f}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        set_style()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Loading {TANGRAM_H5AD}...", flush=True)
        obs, umap, spatial, ct_pred, metrics = load_data()

        summ, anat, agree, cl_conf, cl_n = build_figure(
            obs, umap, spatial, ct_pred, metrics, OUTPUT_DIR
        )
        report(obs, summ, anat, agree, metrics, int(np.shape(ct_pred)[1]))
        print(f"\nDone. Output in {OUTPUT_DIR}/", flush=True)
