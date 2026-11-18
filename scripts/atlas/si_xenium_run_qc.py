#!/usr/bin/env python3
"""Supplementary figure: Xenium run-level QC and negative-control performance.

Reports the per-capture-region ``metrics_summary.csv`` written by Xenium Ranger
for all 32 imaged capture regions (16 queen + 16 worker) plus the per-cell
negative-control counts carried in the combined AnnData. The 101-plex panel is
accompanied by 20 negative-control probes, 41 negative-control codewords and
379 unassigned codewords. The last panel additionally names the run/caste
confound: queen and worker were imaged in two separate Xenium runs ~6 weeks
apart, so caste and batch are the same variable in this design.

Only the lower three rows form the registered figure (figS_xenium_qc, cut by
scripts/paper/split_tall_si_figures.py), so only they carry panel letters, A-C.
The upper four rows (the per-region run metrics, given as Supplementary Data
xenium_qc_metrics.csv instead) keep their titles without letters.

    PYTHONPATH=src .venv/bin/python scripts/atlas/si_xenium_run_qc.py
"""

import glob
import warnings

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata.io import read_elem
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from scipy import sparse
from scipy.stats import binomtest

from bimp_st.paths import CANONICAL_TANGRAM
from bimp_st.print_scale import PRINTED_WIDTH_FRAC
from bimp_st.si_common import (
    LEGEND_KW,
    OUTPUT_DIR,
    QUEEN_COLOR,
    WORKER_COLOR,
    add_panel_headers,
    gene_display,
    save,
    set_style,
    strip_spines,
)

warnings.filterwarnings("ignore", message="Observation names are not unique")

# ── Inputs ───────────────────────────────────────────────────────────────────

RUNS = {
    "queen": "data/20251204__194847___20251204/output-XETG00151__*",
    "worker": "data/20260115__193720__20260115/output-XETG00151__*",
}
TANGRAM_H5AD = str(CANONICAL_TANGRAM)

CASTE_COLOR = {"queen": QUEEN_COLOR, "worker": WORKER_COLOR}

# The canvas is ~7.2 in wide, about the printed width, so text is authored
# near its printed size (the print-scale hook then only trims it).
TS = 6.5  # ticks, in-panel numbers
LS = 7.0  # axis labels
LKW = {**LEGEND_KW, "fontsize": 6.5, "frameon": True}
CASTE_ORDER = ["queen", "worker"]

# Focal genes, marked in panel C so a reader can see where the focal caste
# calls sit relative to the panel-wide run offset.
# IDs from scripts/neuropeptides/fig_neuropeptide_spatial.py and _si_common.FOCAL_GENES.
PANEL_G_FOCAL = {
    "sNPF": "BIMP_vNP3.1_01359",
    "Tk": "BIMP_vNP3.1_07029",
    "Kr-h1": "BIMP_vNP3.1_01378",
    "Vg": "BIMP_vNP3.1_05833",
}

# sc.pp.normalize_total(target_sum=1e4) + log1p, verified by expm1-summing the
# rows of pipelines/combined_atlas/data/combined_clustered.h5ad.
UPSTREAM_NORM = "normalize_total(target_sum=1e4) + log1p, per cell"

# Requested QC fields that Xenium Ranger did not emit for these runs
# (all-NaN in every one of the 32 CSVs) — reported, never imputed.
EXPECTED_BUT_ABSENT = [
    "adjusted_genomic_control_probe_rate",
    "genomic_control_probe_counts_per_control_per_cell",
    "fraction_of_nuclei_without_cell",
    "fraction_of_cell_polygons_removed",
    "fraction_of_nucleus_polygons_removed",
]


# ── Loading ──────────────────────────────────────────────────────────────────


def load_metrics():
    """Concatenate the per-region metrics_summary.csv files (n = 32)."""
    rows = []
    for caste, pattern in RUNS.items():
        files = sorted(glob.glob(f"{pattern}/metrics_summary.csv"))
        if not files:
            raise FileNotFoundError(f"No metrics_summary.csv under {pattern}")
        for f in files:
            d = pd.read_csv(f)
            d["caste"] = caste
            # e.g. output-XETG00151__0040240__Region_1__20251204__195014
            d["section"] = f.split("/")[-2]
            d["slide"] = d["section"].str.split("__").str[1]
            rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    print(
        f"  Loaded metrics for {len(df)} capture regions "
        f"({(df.caste == 'queen').sum()} queen / "
        f"{(df.caste == 'worker').sum()} worker)",
        flush=True,
    )

    absent = [
        c for c in EXPECTED_BUT_ABSENT if c in df.columns and df[c].notna().sum() == 0
    ]
    if absent:
        print(
            "  NOT EMITTED by Xenium Ranger for these runs (all-NaN, not plotted):",
            flush=True,
        )
        for c in absent:
            print(f"    - {c}", flush=True)
    return df


def load_panel_composition():
    """Count feature types in one cell_feature_matrix.h5 (identical panel)."""
    f = sorted(glob.glob(f"{RUNS['queen']}/cell_feature_matrix.h5"))[0]
    with h5py.File(f) as h:
        ftype = np.array(h["matrix/features/feature_type"]).astype(str)
    counts = pd.Series(ftype).value_counts().to_dict()
    print(f"  Panel composition: {counts}", flush=True)
    return counts


def load_cell_qc():
    """Read only obs + X_umap from the combined AnnData (no expression)."""
    with h5py.File(TANGRAM_H5AD, "r") as h:
        obs = read_elem(h["obs"])
        umap = read_elem(h["obsm/X_umap"])
    print(f"  Loaded per-cell QC for {len(obs):,} cells", flush=True)
    return obs, umap


def load_pseudobulk(obs, sections):
    """Per-capture-region pseudobulk raw counts (n_sections x n_genes) from raw/X.

    ``raw/X`` holds integer Xenium counts for exactly the cells in the figure,
    so the log2FC below is computed on counts, not on the log-normalized X
    that downstream differential analysis uses.
    """
    with h5py.File(TANGRAM_H5AD, "r") as h:
        raw = read_elem(h["raw/X"])
        genes = read_elem(h["raw/var"]).index.to_numpy()

    sl = obs["slice_name"].astype(str).to_numpy().astype("U")
    sec = np.full(len(sl), -1)
    for i, b in enumerate(sections):
        sec[np.char.startswith(sl, b)] = i
    if (sec < 0).any():
        raise ValueError(f"{(sec < 0).sum()} cells not matched to a capture region")

    S = sparse.csr_matrix(
        (np.ones(len(sec)), (sec, np.arange(len(sec)))),
        shape=(len(sections), len(sec)),
    )
    pb = np.asarray((S @ raw).todense())
    ncell = np.asarray(S.sum(1)).ravel()
    print(
        f"  Pseudobulk: {pb.shape[0]} capture regions x {pb.shape[1]} genes, "
        f"{pb.sum():,.0f} counts",
        flush=True,
    )
    return pb, ncell, genes


def batch_log2fc(pb, ncell, is_queen):
    """log2FC(worker/queen) per gene, before and after per-capture-region CPM."""
    # (i) uncorrected: mean raw counts per cell, pooled within caste
    mq = pb[is_queen].sum(0) / ncell[is_queen].sum()
    mw = pb[~is_queen].sum(0) / ncell[~is_queen].sum()
    lfc_raw = np.log2(mw / mq)

    # (ii) per-capture-region CPM, then averaged across that caste's regions
    p = pb / pb.sum(1, keepdims=True)
    lfc_cpm = np.log2(p[~is_queen].mean(0) / p[is_queen].mean(0))
    return lfc_raw, lfc_cpm


# ── Plot helpers ─────────────────────────────────────────────────────────────


def strip_by_caste(
    ax, df, col, ylabel, *, scale=1.0, fmt="{:.3g}", annotate_median=True
):
    """Jittered per-capture-region strip plot, queen vs worker, with median bars."""
    rng = np.random.default_rng(0)
    for i, caste in enumerate(CASTE_ORDER):
        v = df.loc[df.caste == caste, col].to_numpy(dtype=float) * scale
        x = i + rng.uniform(-0.12, 0.12, size=v.size)
        ax.scatter(
            x,
            v,
            s=9,
            color=CASTE_COLOR[caste],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.25,
            zorder=3,
        )
        med = float(np.median(v))
        ax.plot(
            [i - 0.2, i + 0.2],
            [med, med],
            color="black",
            lw=1.1,
            zorder=4,
            solid_capstyle="butt",
        )
        if annotate_median:
            ax.text(
                i + 0.23,
                med,
                fmt.format(med),
                fontsize=TS,
                va="center",
                ha="left",
                color="black",
            )
    ax.set_xticks(range(len(CASTE_ORDER)))
    ax.set_xticklabels([f"{c}\n(n = {(df.caste == c).sum()})" for c in CASTE_ORDER])
    ax.set_xlim(-0.6, len(CASTE_ORDER) - 0.15)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)


SUB = dict(
    fontsize=7, fontweight="bold", pad=6
)  # small sub-headers kept on data panels
# Header lift for rows whose axes carry SUB titles (see build_figure).
HEADER_LIFT_PTS = 16


def hexbin_metric(ax, x, y, values, title, cmap="viridis", cb_label=""):
    hb = ax.hexbin(
        x,
        y,
        C=np.asarray(values, dtype=float),
        gridsize=180,
        reduce_C_function=np.mean,
        cmap=cmap,
        linewidths=0.0,
        mincnt=3,
    )
    # Nearly every hexagon sits near 0, so a full-range colour scale is one
    # flat colour; clip it at the 99th percentile of the hexagon means.
    hb.set_clim(0, float(np.percentile(hb.get_array(), 99)))
    cb = plt.colorbar(hb, ax=ax, shrink=0.8, pad=0.03, extend="max")
    cb.ax.tick_params(labelsize=TS)
    cb.set_label(cb_label, fontsize=TS)
    ax.set_title(title, **SUB)
    strip_spines(ax)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")


# ── Figure ───────────────────────────────────────────────────────────────────


def build_figure(df, obs, umap, panel_counts, batch, output_dir):
    n_probe = panel_counts.get("Negative Control Probe", 0)
    n_codeword = panel_counts.get("Negative Control Codeword", 0)
    n_unassigned = panel_counts.get("Unassigned Codeword", 0)
    n_genes = panel_counts.get("Gene Expression", 0)

    plt.rcParams.update(
        {
            "xtick.labelsize": TS,
            "ytick.labelsize": TS,
            "axes.labelsize": LS,
            "legend.fontsize": LKW["fontsize"],
            "savefig.pad_inches": 0.06,
        }
    )
    fig = plt.figure(figsize=(7.2, 13.4))
    gs = GridSpec(
        7,
        1,
        figure=fig,
        hspace=0.55,
        top=0.975,
        bottom=0.03,
        left=0.085,
        right=0.985,
        height_ratios=[1, 1, 1, 1, 1, 1.05, 1.0],
    )

    # ── A: sensitivity ───────────────────────────────────────────────────────
    gsa = gs[0].subgridspec(1, 3, wspace=0.55)
    ax = fig.add_subplot(gsa[0])
    strip_by_caste(
        ax,
        df,
        "median_transcripts_per_cell",
        "Median transcripts per cell",
        fmt="{:.0f}",
    )
    ax = fig.add_subplot(gsa[1])
    strip_by_caste(
        ax,
        df,
        "median_genes_per_cell",
        f"Median genes per cell\n(of {n_genes})",
        fmt="{:.0f}",
    )
    ax = fig.add_subplot(gsa[2])
    strip_by_caste(
        ax,
        df,
        "num_cells_detected",
        "Cells detected per capture region",
        scale=1e-3,
        fmt="{:.0f}k",
    )
    ax.set_ylabel("Cells detected per\ncapture region (×10³)")

    # ── B: negative controls ─────────────────────────────────────────────────
    gsb = gs[1].subgridspec(1, 3, wspace=0.55)
    ax = fig.add_subplot(gsb[0])
    strip_by_caste(
        ax,
        df,
        "adjusted_negative_control_probe_rate",
        "Negative control\nprobe rate (%)",
        scale=100,
        fmt="{:.3f}%",
    )
    ax = fig.add_subplot(gsb[1])
    strip_by_caste(
        ax,
        df,
        "adjusted_negative_control_codeword_rate",
        "Negative control\ncodeword rate (%)",
        scale=100,
        fmt="{:.3f}%",
    )
    ax = fig.add_subplot(gsb[2])
    strip_by_caste(
        ax,
        df,
        "negative_control_probe_counts_per_control_per_cell",
        "Counts per control\nprobe per cell",
        fmt="{:.4f}",
    )

    # ── C: false positives ───────────────────────────────────────────────────
    gsc = gs[2].subgridspec(1, 3, wspace=0.55)
    ax = fig.add_subplot(gsc[0])
    strip_by_caste(
        ax,
        df,
        "estimated_number_of_false_positive_transcripts_per_cell",
        "Est. false positive\ntranscripts per cell",
        fmt="{:.3f}",
    )

    ax = fig.add_subplot(gsc[1])
    df = df.copy()
    df["fp_pct"] = (
        100
        * df["estimated_number_of_false_positive_transcripts_per_cell"]
        / df["median_transcripts_per_cell"]
    )
    strip_by_caste(
        ax,
        df,
        "fp_pct",
        "False positives (% of\nmedian transcripts/cell)",
        fmt="{:.3f}%",
    )

    # Per-cell negative-control burden from the AnnData
    ax = fig.add_subplot(gsc[2])
    ctrl_cols = [
        "control_probe_counts",
        "control_codeword_counts",
        "unassigned_codeword_counts",
        "genomic_control_counts",
        "deprecated_codeword_counts",
    ]
    # Horizontal bars: five category names fit as single-line y labels,
    # where under vertical bars they overran the panel and the row below.
    labels = [
        "Neg. ctrl probe",
        "Neg. ctrl codeword",
        "Unassigned",
        "Genomic ctrl",
        "Deprecated",
    ]
    total = float(obs["transcript_counts"].sum())
    rates = [100 * float(obs[c].sum()) / total for c in ctrl_cols]
    ys = np.arange(len(rates))[::-1]
    bars = ax.barh(ys, rates, color="#4A6C8C", height=0.62)
    for b, r in zip(bars, rates, strict=False):
        ax.text(
            r + max(rates) * 0.03,
            b.get_y() + b.get_height() / 2,
            "0" if r == 0 else f"{r:.3f}%",
            ha="left",
            va="center",
            fontsize=TS,
        )
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=TS)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel("Control counts\n(% of gene transcripts)")
    ax.set_xlim(0, max(rates) * 1.45)
    ax.grid(axis="x", lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)

    # ── D: decoding quality ──────────────────────────────────────────────────
    gsd = gs[3].subgridspec(1, 3, wspace=0.55)
    ax = fig.add_subplot(gsd[0])
    strip_by_caste(
        ax,
        df,
        "fraction_transcripts_decoded_q20",
        "Transcripts decoded\nat Q20 (%)",
        scale=100,
        fmt="{:.1f}%",
    )
    ax = fig.add_subplot(gsd[1])
    strip_by_caste(
        ax,
        df,
        "fraction_transcripts_assigned",
        "Transcripts assigned\nto cells (%)",
        scale=100,
        fmt="{:.1f}%",
    )
    ax = fig.add_subplot(gsd[2])
    strip_by_caste(
        ax,
        df,
        "decoded_transcripts_per_100um2",
        "Decoded transcripts\nper 100 µm²",
        fmt="{:.0f}",
    )

    # ── E: segmentation ──────────────────────────────────────────────────────
    gse = gs[4].subgridspec(1, 2, wspace=0.30, width_ratios=[1.6, 1.0])
    ax = fig.add_subplot(gse[0])
    # Interior stain dominates (>96% everywhere), so plot the two fallback
    # methods stacked; interior stain is the remainder up to 100%.
    seg_cols = ["segmented_cell_nuc_expansion_frac", "segmented_cell_boundary_frac"]
    seg_lab = ["Nucleus expansion (5 µm)", "Boundary stain"]
    seg_cmap = plt.get_cmap("cividis")
    seg_colors = [seg_cmap(v) for v in (0.35, 0.85)]
    order = df.sort_values(["caste", "section"]).index
    xs = np.arange(len(order))
    bottom = np.zeros(len(order))
    for col, lab, c in zip(seg_cols, seg_lab, seg_colors, strict=False):
        vals = 100 * df.loc[order, col].to_numpy(dtype=float)
        ax.bar(xs, vals, bottom=bottom, color=c, label=lab, width=0.85)
        bottom += vals
    interior = 100 * df.loc[order, "segmented_cell_interior_frac"]
    ax.set_ylim(0, max(6.0, bottom.max() * 1.45))
    ax.set_ylabel("Fallback-segmented cells (%)")
    ax.set_xlabel("Capture region (16 queen, then 16 worker)")
    ax.set_xticks([7.5, 23.5])
    ax.set_xticklabels(["queen", "worker"])
    ax.axvline(15.5, ymax=0.68, color="black", lw=0.8, ls="--")
    # NOTE: the legends in this file use frameon=True, unlike the frameless
    # house style elsewhere; switching them is a separate cosmetic change.
    # Re-rendering is safe: scripts/paper/split_tall_si_figures.py regenerates the
    # _a/_b crops deterministically from this script's output.
    ax.legend(loc="upper left", ncol=2, **LKW)
    print(
        f"  E: interior stain (18S) {interior.min():.1f}-{interior.max():.1f}%",
        flush=True,
    )
    ax.grid(axis="y", lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)

    ax = fig.add_subplot(gse[1])
    strip_by_caste(
        ax,
        df,
        "fraction_empty_cells",
        "Empty cells\n(% of cells detected)",
        scale=100,
        fmt="{:.3f}%",
    )

    # The third E cell is left empty: the absent vendor QC fields, the
    # detected-nucleus fraction and the panel design are in the caption.
    nuc_frac = 100 * float(obs["nucleus_area"].notna().mean())
    print(
        f"  E: detected nucleus {nuc_frac:.2f}% "
        f"({int(obs['nucleus_area'].notna().sum()):,}/{len(obs):,}); panel "
        f"{n_genes} targets / {n_probe} probes / {n_codeword} codewords / "
        f"{n_unassigned} unassigned; absent fields: " + ", ".join(EXPECTED_BUT_ABSENT),
        flush=True,
    )

    # ── F: control signal on the shared UMAP ─────────────────────────────────
    gsf = gs[5].subgridspec(1, 3, wspace=0.62, width_ratios=[1, 1, 1.1])
    x, y = umap[:, 0], umap[:, 1]
    ax = fig.add_subplot(gsf[0])
    hexbin_metric(
        ax,
        x,
        y,
        obs["control_probe_counts"],
        "Neg. control probe counts",
        cmap="viridis",
        cb_label="Mean counts per cell",
    )
    ax = fig.add_subplot(gsf[1])
    hexbin_metric(
        ax,
        x,
        y,
        obs["unassigned_codeword_counts"],
        "Unassigned codeword counts",
        cmap="viridis",
        cb_label="Mean counts per cell",
    )

    # Per-cluster negative-control rate: flat => control signal is not
    # cluster-structured (i.e. not driving any cluster).
    ax = fig.add_subplot(gsf[2])
    grp = obs.groupby("leiden", observed=True)
    rate = (
        100
        * (grp["control_probe_counts"].sum() + grp["control_codeword_counts"].sum())
        / grp["transcript_counts"].sum()
    )
    rate = rate.sort_values()
    global_rate = (
        100
        * (obs["control_probe_counts"].sum() + obs["control_codeword_counts"].sum())
        / obs["transcript_counts"].sum()
    )
    ax.bar(np.arange(len(rate)), rate.to_numpy(), color="#4A6C8C", width=0.85)
    ax.axhline(
        global_rate,
        color="#D95F02",
        lw=1.2,
        ls="--",
        label=f"all cells: {global_rate:.4f}%",
    )
    ax.set_xlabel(f"Leiden cluster\n(n = {len(rate)}, sorted by rate)")
    ax.set_ylabel("Neg. control counts\n(% of gene transcripts)")
    ax.legend(loc="upper left", **LKW)
    ax.grid(axis="y", lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)
    n_zero = int((rate == 0).sum())
    nz = rate[rate > 0]
    ax.set_ylim(0, rate.max() * 1.32)
    # Range and zero-count clusters are given in the caption.
    print(
        f"  F: non-zero cluster rate {nz.min():.4f}-{nz.max():.4f}% "
        f"({nz.max() / nz.min():.1f}x); {n_zero} clusters at zero",
        flush=True,
    )

    # ── G: caste is confounded with Xenium run ───────────────────────────────
    lfc_raw, lfc_cpm, genes, depth_q, depth_w = batch
    del depth_q, depth_w  # the depth caveat is in the caption, not the figure
    gsg = gs[6].subgridspec(1, 3, wspace=0.34)
    lo = min(lfc_raw.min(), lfc_cpm.min()) - 0.2
    hi = max(lfc_raw.max(), lfc_cpm.max()) + 0.2
    bins = np.linspace(lo, hi, 46)

    def _lfc_hist(ax, v, title):
        med = float(np.median(v))
        ax.hist(v, bins=bins, color="#4A6C8C")
        ax.axvline(0, ymax=0.68, color="0.35", ls=":", lw=1.2, label="no change")
        ax.axvline(
            med,
            ymax=0.68,
            color="#D95F02",
            ls="--",
            lw=1.4,
            label=f"median = {med:+.3f} ({2**med:.2f}×)",
        )
        ax.set_xlabel("log₂ fold change (worker / queen)")
        ax.set_ylabel(f"Genes (of {len(v)})")
        ax.set_title(title, **SUB)
        ax.legend(loc="upper left", **LKW)
        # The caption gives the tilt for both histograms (median, genes
        # positive, sign-test P), so it is not repeated in-panel; the median
        # line keeps its legend.
        ax.grid(axis="y", lw=0.4, alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.45)
        return med

    ax = fig.add_subplot(gsg[0, 0])
    _lfc_hist(ax, lfc_raw, "Raw counts")
    ax = fig.add_subplot(gsg[0, 1])
    med_cpm = _lfc_hist(ax, lfc_cpm, "CPM-centered")

    ax = fig.add_subplot(gsg[0, 2])
    ax.scatter(
        lfc_raw,
        lfc_cpm,
        s=8,
        color="#4A6C8C",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
    )
    lim = [lo, hi]
    ax.plot(lim, lim, color="0.5", ls=":", lw=1.0)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.axvline(0, color="0.8", lw=0.6)
    ax.axhline(med_cpm, color="#D95F02", ls="--", lw=1.0)
    # Focal genes sit within ~10 pt of each other on the diagonal, so labels
    # are spread to both sides of it (axes fractions) with short leaders that
    # point in different directions and do not cross.
    label_pos = {
        "sNPF": (0.46, 0.20, "left"),
        "Vg": (0.66, 0.36, "left"),
        "Kr-h1": (0.30, 0.70, "right"),
        "Tk": (0.53, 0.80, "center"),
    }
    for label, prefix in PANEL_G_FOCAL.items():
        idx = [i for i, g in enumerate(genes) if g.startswith(prefix)]
        if not idx:
            print(
                f"  WARNING: focal gene {label} ({prefix}) not on the panel", flush=True
            )
            continue
        i = idx[0]
        ax.scatter(
            lfc_raw[i],
            lfc_cpm[i],
            s=22,
            color=QUEEN_COLOR,
            edgecolor="black",
            linewidth=0.5,
            zorder=4,
            label="focal caste gene (all queen-up)" if label == "sNPF" else None,
        )
        ax.annotate(
            gene_display(label),
            (lfc_raw[i], lfc_cpm[i]),
            textcoords="axes fraction",
            xytext=label_pos[label][:2],
            ha=label_pos[label][2],
            va="center",
            fontsize=TS,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", lw=0.5, color="0.45", shrinkA=2, shrinkB=4),
        )
    ax.legend(loc="upper left", **LKW)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("log₂FC, mean counts per cell")
    ax.set_ylabel("log₂FC, per-capture-region CPM")
    ax.set_title("Raw vs centered", **SUB)
    ax.grid(lw=0.4, alpha=0.3)
    ax.set_axisbelow(True)

    # The run-confound explanation (batch = caste in this design; the
    # per-gene medians and the downstream-normalization caveat) is in the
    # LaTeX caption for this panel, not on the artwork.

    add_panel_headers(
        fig,
        [
            # Upper four rows: not part of the registered figure, so titles only.
            (gs[0], "", "Sensitivity"),
            (gs[1], "", "Negative controls"),
            (gs[2], "", "False positives"),
            (gs[3], "", "Decoding quality"),
            (gs[4], "A", "Segmentation"),
        ],
        width_frac=PRINTED_WIDTH_FRAC["fig_s_xenium_qc"],
    )
    # Rows B and C carry small per-axes sub-titles (SUB, pad 15 pt) that sit
    # exactly where the default header lands; lift these two headers clear
    # of them instead of letting the letter/tag print over the sub-titles.
    add_panel_headers(
        fig,
        [
            (gs[5], "B", "Control signal"),
            (gs[6], "C", "Caste versus run"),
        ],
        width_frac=PRINTED_WIDTH_FRAC["fig_s_xenium_qc"],
        dy_pts=HEADER_LIFT_PTS,
    )

    # Caste key on the header line of the top row, flush with the right-hand axes.
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            ls="none",
            markersize=4,
            markerfacecolor=CASTE_COLOR[c],
            markeredgecolor="white",
            markeredgewidth=0.3,
            label=c,
        )
        for c in CASTE_ORDER
    ]
    handles.append(Line2D([0], [0], color="black", lw=1.1, label="median"))
    row_a = gs[0].get_position(fig)
    fig.legend(
        handles=handles,
        loc="lower right",
        ncol=3,
        bbox_to_anchor=(row_a.x1, row_a.y1 + 0.002),
        borderaxespad=0,
        borderpad=0,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=1.2,
        **LKW,
    )
    # No suptitle banner -- it repeated the caption's bold opening line
    # verbatim (house style: the caption owns the title).
    save(fig, output_dir, "fig_s_xenium_qc")


# ── Report ───────────────────────────────────────────────────────────────────


def report(df, obs):
    """Print every number the caption needs (nothing is invented)."""
    print(
        "\n── Negative-control summary (per-capture-region, n = 16 per caste) ──",
        flush=True,
    )
    for col, scale, unit in [
        ("adjusted_negative_control_probe_rate", 100, "%"),
        ("adjusted_negative_control_codeword_rate", 100, "%"),
        ("negative_control_probe_counts_per_control_per_cell", 1, ""),
        ("estimated_number_of_false_positive_transcripts_per_cell", 1, ""),
        ("fraction_transcripts_decoded_q20", 100, "%"),
        ("fraction_transcripts_assigned", 100, "%"),
        ("median_transcripts_per_cell", 1, ""),
        ("median_genes_per_cell", 1, ""),
        ("fraction_empty_cells", 100, "%"),
    ]:
        v = df[col] * scale
        q = df.loc[df.caste == "queen", col] * scale
        w = df.loc[df.caste == "worker", col] * scale
        print(
            f"  {col}\n"
            f"    all    median {v.median():.4g}{unit}  "
            f"range {v.min():.4g}–{v.max():.4g}{unit}\n"
            f"    queen  median {q.median():.4g}{unit}  "
            f"range {q.min():.4g}–{q.max():.4g}{unit}\n"
            f"    worker median {w.median():.4g}{unit}  "
            f"range {w.min():.4g}–{w.max():.4g}{unit}",
            flush=True,
        )

    total = float(obs["transcript_counts"].sum())
    print(
        f"\n  Pooled cells: {len(obs):,}; gene transcripts in cells: {total:,.0f}",
        flush=True,
    )
    for c in [
        "control_probe_counts",
        "control_codeword_counts",
        "unassigned_codeword_counts",
        "genomic_control_counts",
        "deprecated_codeword_counts",
    ]:
        s = float(obs[c].sum())
        print(
            f"    {c}: {s:,.0f} ({100 * s / total:.4f}% of gene "
            f"transcripts; mean {s / len(obs):.5f} per cell)",
            flush=True,
        )


def report_batch(lfc_raw, lfc_cpm, genes, depth_q, depth_w):
    print("\n── Caste == Xenium run: whole-panel log2FC(worker/queen) ──", flush=True)
    print(
        f"  mean gene counts per cell: queen {depth_q:.2f}, "
        f"worker {depth_w:.2f} "
        f"(log2 {np.log2(depth_w / depth_q):+.4f})",
        flush=True,
    )
    for name, v in [
        ("uncorrected (mean counts/cell)", lfc_raw),
        ("per-capture-region CPM", lfc_cpm),
    ]:
        up = int((v > 0).sum())
        pv = binomtest(up, len(v), 0.5).pvalue
        print(
            f"  {name}:\n"
            f"    median {np.median(v):+.4f} ({2 ** np.median(v):.3f}x)  "
            f"mean {v.mean():+.4f}\n"
            f"    IQR {np.percentile(v, 25):+.3f}..{np.percentile(v, 75):+.3f}"
            f"  range {v.min():+.3f}..{v.max():+.3f}\n"
            f"    worker-up {up}/{len(v)} genes (sign test p = {pv:.2e}); "
            f"|log2FC|>0.5: {int((np.abs(v) > 0.5).sum())}, "
            f">1: {int((np.abs(v) > 1).sum())}",
            flush=True,
        )
    print("  focal genes (negative = queen-up):", flush=True)
    for label, prefix in PANEL_G_FOCAL.items():
        idx = [i for i, g in enumerate(genes) if g.startswith(prefix)]
        if not idx:
            print(f"    {label}: NOT on the panel", flush=True)
            continue
        i = idx[0]
        print(
            f"    {label:6s} uncorrected {lfc_raw[i]:+.3f}  CPM {lfc_cpm[i]:+.3f}",
            flush=True,
        )


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        set_style()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print("Loading Xenium metrics_summary.csv files...", flush=True)
        metrics = load_metrics()
        panel_counts = load_panel_composition()

        print(f"Loading per-cell QC from {TANGRAM_H5AD}...", flush=True)
        obs, umap = load_cell_qc()

        print("Building per-capture-region pseudobulk from raw counts...", flush=True)
        sections = metrics["section"].tolist()
        pb, ncell, genes = load_pseudobulk(obs, sections)
        is_queen = (metrics["caste"] == "queen").to_numpy()
        lfc_raw, lfc_cpm = batch_log2fc(pb, ncell, is_queen)
        depth_q = pb[is_queen].sum() / ncell[is_queen].sum()
        depth_w = pb[~is_queen].sum() / ncell[~is_queen].sum()
        batch = (lfc_raw, lfc_cpm, genes, depth_q, depth_w)

        build_figure(metrics, obs, umap, panel_counts, batch, OUTPUT_DIR)
        report(metrics, obs)
        report_batch(lfc_raw, lfc_cpm, genes, depth_q, depth_w)
        print(f"\nDone. Output in {OUTPUT_DIR}/", flush=True)
