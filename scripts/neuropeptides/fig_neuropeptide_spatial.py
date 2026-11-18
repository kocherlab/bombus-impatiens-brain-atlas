#!/usr/bin/env python3
"""Neuropeptide figure (fig:neuropeptide_spatial):
sNPF/Tk transcript and MALDI peptide maps, prevalence by region, caste effects.

Layout (letters as printed):
  A  2 x 4 map grid, rows sNPF / Tk, columns queen Xenium, queen MALDI, worker
     Xenium, worker MALDI. Xenium maps show the fraction of cells expressing;
     MALDI maps come from bimp_st.maldi (single-peak readouts, qualitative).
  B  % of cells expressing each peptide in the calyx and the optic neuropils.
  C  snRNA-seq caste log2FC per central brain cell class (n = 3 per caste).
  D  optic lobe neurons expressing each peptide (one dissection per caste).

save_individual_panels also writes Supplementary figS_ap_gradient (Xenium
anterior-posterior profiles) and unshipped standalone panels.

Run: PYTHONPATH=src:scripts .venv/bin/python \
    scripts/neuropeptides/fig_neuropeptide_spatial.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from neuropeptides.neuropeptide_spatial_data import (  # noqa: E402
    DEFAULT_OUTPUT_DIR,
    OUTPUT_DIR,
    _bbox,
    _compute_shared,
    _density_map,
    _kc_enrichment,  # noqa: F401 (re-exported)
    _load_maldi_obs,
    _maldi_available,
    resolve_gene,  # noqa: F401 (re-exported)
)
from neuropeptides.neuropeptide_spatial_panels import (  # noqa: E402
    _FS_LAB,
    _FS_TICK,
    _FS_TITLE,
    MALDI_CMAP,
    SOLID_VIRIDIS,
    SPATIAL_GAMMA,
    _draw_ap_xenium,
    _draw_forest,
    _draw_optic_prev,
    _panel_scalebar,
    _plot_density,
    _plot_maldi_panel,
    _PowerNorm,
    _spatial_scatter,
)
from scipy.stats import mannwhitneyu

from bimp_st import maldi
from bimp_st.nine_region_common import (
    KC_REGIONS,
    OPTIC_REGIONS,
    attach_region_flags,
    kc_mask,
    optic_mask,
)
from bimp_st.si_common import (
    EXPR_CMAP,
    PIXEL_SCALE,
    SPATIAL_H5AD,
    format_pval,
    save,
)
from bimp_st.si_common import OUTPUT_DIR as SI_OUTPUT_DIR
from bimp_st.si_common import (
    add_scalebar as _add_scalebar,
)
from bimp_st.si_common import (
    get_expr as _get_expr,
)
from bimp_st.si_common import (
    strip_spines as _strip_spines,
)
from bimp_st.style import (
    QUEEN_COLOR,
    WORKER_COLOR,
    panel_header,
    set_pub_style,
)

SPATIAL_VMAX_PCTL = 99  # conservative clip (sqrt norm handles the range)


# -- Standalone panel save functions ----------------------------------------


def _save_spatial_panel(
    sp,
    expr,
    mask,
    shared_vmax,
    shared_span,
    gene_name,
    brain_label,
    output_dir,
    filename,
):
    """Save a single spatial scatter panel as a standalone figure."""
    fig, ax = plt.subplots(figsize=(4.5, 5.1))
    sc_obj = _spatial_scatter(
        ax,
        sp,
        expr,
        mask,
        EXPR_CMAP,
        shared_vmax,
        brain_label,
        shared_span,
        show_mean=False,
    )
    _add_scalebar(ax, shared_span)
    ax.text(
        0.02,
        0.98,
        gene_name,
        transform=ax.transAxes,
        fontsize=10,
        fontstyle="italic",
        fontweight="bold",
        va="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.5),
    )
    cbar = fig.colorbar(
        sc_obj, ax=ax, fraction=0.03, pad=0.02, location="bottom", shrink=0.6
    )
    cbar.set_label("Expression (log1p)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    save(fig, output_dir, filename)


def save_panel_c_bars(adata, expr_tk, expr_snpf, queen_mask, output_dir):
    """Save panel C (expression bars + MWU) as standalone figure."""
    worker_mask = ~queen_mask
    fig, ax = plt.subplots(figsize=(4.0, 5.1))
    gene_names = ["Tk", "sNPF"]
    gene_exprs = [expr_tk, expr_snpf]
    x = np.arange(len(gene_names))
    bar_w = 0.35
    for gi, (_gn, gx) in enumerate(zip(gene_names, gene_exprs, strict=False)):
        q_mean = gx[queen_mask].mean()
        w_mean = gx[worker_mask].mean()
        q_pct = (gx[queen_mask] > 0).mean() * 100
        w_pct = (gx[worker_mask] > 0).mean() * 100
        ax.bar(gi - bar_w / 2, q_mean, bar_w, color=QUEEN_COLOR, alpha=0.85)
        ax.bar(gi + bar_w / 2, w_mean, bar_w, color=WORKER_COLOR, alpha=0.85)
        ax.text(
            gi - bar_w / 2,
            q_mean / 2,
            f"{q_pct:.0f}%",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="white",
        )
        ax.text(
            gi + bar_w / 2,
            w_mean / 2,
            f"{w_pct:.0f}%",
            ha="center",
            va="center",
            fontsize=7,
            fontweight="bold",
            color="white",
        )
        slices = adata.obs["slice_name"].values.astype(str)
        q_slice_means = (
            pd.Series(gx[queen_mask]).groupby(slices[queen_mask]).mean().values
        )
        w_slice_means = (
            pd.Series(gx[worker_mask]).groupby(slices[worker_mask]).mean().values
        )
        _, pval = mannwhitneyu(q_slice_means, w_slice_means, alternative="two-sided")
        y_top = max(q_mean, w_mean) * 1.08
        ax.plot(
            [gi - bar_w / 2, gi - bar_w / 2, gi + bar_w / 2, gi + bar_w / 2],
            [y_top, y_top * 1.03, y_top * 1.03, y_top],
            lw=0.8,
            color="k",
        )
        pstr = format_pval(pval)
        ax.text(gi, y_top * 1.05, pstr, ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(gene_names, fontsize=9, fontstyle="italic")
    ax.set_ylabel("Mean expression", fontsize=9)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save(fig, output_dir, "panel_c_bars")


def save_panel_f_gradient(expr_tk, expr_snpf, queen_mask, z_levels, output_dir):
    """Save Supplementary figS_ap_gradient: Xenium A-P profiles, one panel.

    MALDI has no A-P trace: the aligned-panel section order is not tied to the
    anterior-posterior axis anywhere upstream, and a per-plane MALDI mean tracks
    tissue area, not peptide.
    """
    fig, ax1 = plt.subplots(figsize=(7.2, 3.1))
    _draw_ap_xenium(
        ax1,
        expr_tk,
        expr_snpf,
        queen_mask,
        z_levels,
        fs=9,
        headroom=1.42,
        legend=True,
    )
    ax1.set_title("")  # the caption carries the title
    fig.tight_layout()
    si_dir = SI_OUTPUT_DIR if Path(output_dir) == DEFAULT_OUTPUT_DIR else output_dir
    si_dir.mkdir(parents=True, exist_ok=True)
    save(fig, si_dir, "figS_ap_gradient")


def _draw_maldi_standalone(sample_key, gene_label, output_dir, filename):
    """Save a single MALDI panel as standalone figure."""
    if not _maldi_available(sample_key):
        print(f"  MALDI {sample_key} not found, skipping {filename}", flush=True)
        return
    mip, ext, vmin, vmax, channel_name = _load_maldi_obs(sample_key, gene_label)
    fig, ax = plt.subplots(figsize=(4.5, 5.1))
    _plot_maldi_panel(
        ax,
        mip,
        ext,
        vmin,
        vmax,
        f"MALDI {sample_key.capitalize()} {gene_label}",
        window=None,
    )
    sb_len = 1000.0
    sx = ext[0] + 0.05 * (ext[1] - ext[0])
    sy = ext[3] + 0.05 * (ext[2] - ext[3])
    ax.plot([sx, sx + sb_len], [sy, sy], color="black", lw=3, solid_capstyle="butt")
    ax.text(
        sx + sb_len / 2,
        sy + 0.03 * (ext[2] - ext[3]),
        "1 mm",
        color="black",
        fontsize=7,
        ha="center",
        va="bottom",
    )
    ax.text(
        0.02,
        0.98,
        gene_label,
        transform=ax.transAxes,
        fontsize=10,
        fontstyle="italic",
        fontweight="bold",
        va="top",
        color="white",
    )
    cb = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=MALDI_CMAP
        ),
        ax=ax,
        fraction=0.035,
        pad=0.02,
        orientation="horizontal",
    )
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Intensity (a.u.)", fontsize=8)
    save(fig, output_dir, filename)


def save_individual_panels(adata, sp, queen_mask, z_levels, output_dir, shared):
    """Save standalone panels (unshipped) and Supplementary figS_ap_gradient."""
    print("  Saving individual panels...", flush=True)
    et, es = shared["expr_tk"], shared["expr_snpf"]
    sv, ss = shared["shared_vmax"], shared["shared_span"]

    # A: Queen Tk
    _save_spatial_panel(
        sp, et, queen_mask, sv, ss, "Tk", "Queen", output_dir, "panel_a_queen_tk"
    )
    # B: Queen sNPF
    _save_spatial_panel(
        sp, es, queen_mask, sv, ss, "sNPF", "Queen", output_dir, "panel_b_queen_snpf"
    )
    # C: Expression bars
    save_panel_c_bars(adata, et, es, queen_mask, output_dir)
    # D: Worker Tk
    _save_spatial_panel(
        sp, et, ~queen_mask, sv, ss, "Tk", "Worker", output_dir, "panel_d_worker_tk"
    )
    # E: Worker sNPF
    _save_spatial_panel(
        sp, es, ~queen_mask, sv, ss, "sNPF", "Worker", output_dir, "panel_e_worker_snpf"
    )
    # F: A-P gradient
    save_panel_f_gradient(et, es, queen_mask, z_levels, output_dir)
    # MALDI maps of panel A, standalone
    _draw_maldi_standalone("queen", "Tk", output_dir, "panel_row1_queen_maldi_tk")
    _draw_maldi_standalone("queen", "sNPF", output_dir, "panel_row1_queen_maldi_snpf")
    _draw_maldi_standalone("worker", "Tk", output_dir, "panel_g_maldi_tk")
    _draw_maldi_standalone("worker", "sNPF", output_dir, "panel_h_maldi_snpf")


# -- Composite figure -------------------------------------------------------

_FS_LETTER = 9
_LET_DY = 16  # A/B sit above the Xenium/MALDI column headers


def compose_figure(
    adata, sp, queen_mask, z_levels, output_dir, out_name="fig_neuropeptide_spatial.png"
):
    """The composite figure, laid out in inches.

    Top block: 2x4 map grid A (row 0 = sNPF, row 1 = Tk; columns queen Xenium,
    queen MALDI, worker Xenium, worker MALDI), with B (prevalence by region) to
    its right spanning both rows. Below: C (two forest plots sharing y labels)
    and D (optic-lobe recruitment).
    """
    print("  Composing final figure...", flush=True)
    worker_mask = ~queen_mask

    genes = {}
    for name in ["sNPF", "Tk"]:
        var = resolve_gene(name, adata.var_names)
        if var is None:
            raise SystemExit(f"ERROR: {name} not found in data")
        genes[name] = _get_expr(adata, var)
        print(
            f"  {name}: {var} "
            f"(mean Q={genes[name][queen_mask].mean():.2f}, "
            f"W={genes[name][worker_mask].mean():.2f})",
            flush=True,
        )
    expr_snpf, expr_tk = genes["sNPF"], genes["Tk"]

    def _pctspan(idx):
        xlo, xhi = np.percentile(sp[idx, 0], [0.5, 99.5])
        ylo, yhi = np.percentile(sp[idx, 1], [0.5, 99.5])
        return max(xhi - xlo, yhi - ylo)

    q_i, w_i = np.where(queen_mask)[0], np.where(worker_mask)[0]
    shared_span = max(_pctspan(q_i), _pctspan(w_i)) * 1.06

    # MALDI union bbox (queen+worker, both peptides; shared aligned frame).
    _mb = []
    for _pep in ("sNPF_v1", "Tk2"):
        _qd, _wd, _, _ext = maldi.princeton_maps(_pep)
        _union = np.where(np.isfinite(_qd) | np.isfinite(_wd), 1.0, np.nan)
        _mb.append(_bbox(_union, _ext))
    m_x0 = min(b[0] for b in _mb)
    m_x1 = max(b[1] for b in _mb)
    m_y0 = min(b[2] for b in _mb)
    m_y1 = max(b[3] for b in _mb)
    maldi_c = ((m_x0 + m_x1) / 2, (m_y0 + m_y1) / 2)

    # Xenium density maps (square canvas big enough for any brain; the visible
    # window is cropped below).
    spatial_span_um = shared_span * PIXEL_SCALE
    canvas_um = max(spatial_span_um, m_x1 - m_x0, m_y1 - m_y0) * 1.05
    spum_all = sp * PIXEL_SCALE
    dmaps = {
        "qs": _density_map(spum_all[q_i], expr_snpf[q_i], canvas_um),
        "qt": _density_map(spum_all[q_i], expr_tk[q_i], canvas_um),
        "ws": _density_map(spum_all[w_i], expr_snpf[w_i], canvas_um),
        "wt": _density_map(spum_all[w_i], expr_tk[w_i], canvas_um),
    }
    xb = {k: _bbox(*dmaps[k]) for k in dmaps}

    # One common window size (um) for all eight image panels, so every 1 mm
    # bar has the same length. Plus a strip below the tissue for the bar.
    PAD, PADB = 1.04, 0.07
    win_w = max(max(b[1] - b[0] for b in xb.values()), m_x1 - m_x0) * PAD
    win_h = max(max(b[3] - b[2] for b in xb.values()), m_y1 - m_y0) * PAD
    print(
        f"  image window {win_w:.0f} x {win_h:.0f} um (+{PADB:.0%} bar strip)",
        flush=True,
    )

    def _lims_xen(k):
        x0, x1, y0, y1 = xb[k]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return (
            (cx - win_w / 2, cx + win_w / 2),
            (cy - win_h / 2 - PADB * win_h, cy + win_h / 2),
        )

    def _lims_maldi():
        cx, cy = maldi_c
        return (
            (cx - win_w / 2, cx + win_w / 2),
            (cy + win_h / 2 + PADB * win_h, cy - win_h / 2),
        )  # inverted y

    def _pep_vmax(*keys):
        vals = np.concatenate([dmaps[k][0][np.isfinite(dmaps[k][0])] for k in keys])
        return float(np.percentile(vals, SPATIAL_VMAX_PCTL)) if vals.size else 1.0

    vmax_snpf = _pep_vmax("qs", "ws")
    vmax_tk = _pep_vmax("qt", "wt")
    vmx = {"qs": vmax_snpf, "ws": vmax_snpf, "qt": vmax_tk, "wt": vmax_tk}

    # Nine-region flags joined on (slice_name, cell_id), not the in_* columns
    # stored in the h5ad. Nullable: untraced cells are neither inside nor outside.
    attach_region_flags(adata, regions=[*KC_REGIONS, *OPTIC_REGIONS])
    kcm, olm = kc_mask(adata.obs), optic_mask(adata.obs)
    kc_flags = kcm.to_numpy(dtype=bool, na_value=False)
    kc_traced = ~np.asarray(kcm.isna())
    print(
        f"  Tk KC-vs-rest enrichment: queen "
        f"{_kc_enrichment(expr_tk, queen_mask & kc_traced, kc_flags):.2f}x  "
        f"worker {_kc_enrichment(expr_tk, worker_mask & kc_traced, kc_flags):.2f}x "
        f"({int((~kc_traced).sum()):,} cells untraced)",
        flush=True,
    )

    # ---- geometry (inches; y measured down from the top) -------------------
    W = 7.4
    LM = 0.30  # left margin (row labels)
    IW = 1.02  # image panel width
    GAP = 0.07
    IH = IW * (win_h * (1 + PADB)) / win_w
    HDR = 0.17  # caste header row
    TTL = 0.38  # title row (letters + modality titles)
    ROWGAP = 0.17
    top0 = HDR + TTL
    top1 = top0 + IH + ROWGAP
    maps_bot = top1 + IH
    CB_Y = maps_bot + 0.03
    J_TOP = maps_bot + 0.62
    J_H = 1.05
    K_TOP = J_TOP + J_H + 0.70
    K_H = 0.58
    H = K_TOP + K_H + 0.40
    fig = plt.figure(figsize=(W, H))

    def _ax(x, ytop, w, h):
        return fig.add_axes([x / W, 1 - (ytop + h) / H, w / W, h / H])

    colx = [LM + i * (IW + GAP) for i in range(4)]
    img_right = colx[3] + IW

    def _header(anchor, s, tag, dy=0):
        """House panel header (bimp_st.style): bold letter + bold title."""
        panel_header(
            fig,
            anchor,
            s,
            tag,
            letter_size=_FS_LETTER,
            tag_size=_FS_TITLE,
            dy_pts=3 + dy,
        )

    def _title(ax, s, color="black"):
        ax.annotate(
            s,
            xy=(0.5, 1),
            xycoords="axes fraction",
            xytext=(0, 3),
            textcoords="offset points",
            fontsize=_FS_TITLE,
            fontweight="bold",
            va="bottom",
            ha="center",
            color=color,
            annotation_clip=False,
        )

    axes = {}
    for r, (pep, kq, kw) in enumerate([("sNPF", "qs", "ws"), ("Tk", "qt", "wt")]):
        ytop = top0 if r == 0 else top1
        for c in range(4):
            ax = _ax(colx[c], ytop, IW, IH)
            axes[(r, c)] = ax
            if c in (0, 2):
                k = kq if c == 0 else kw
                _plot_density(ax, dmaps[k][0], dmaps[k][1], 0.0, vmx[k], "")
                xl, yl = _lims_xen(k)
            else:
                key = "queen" if c == 1 else "worker"
                if not _maldi_available(key):
                    raise SystemExit(f"ERROR: MALDI data missing for {key}")
                mip, ext, vmin, vmaxm, _ = _load_maldi_obs(key, pep)
                _plot_maldi_panel(ax, mip, ext, vmin, vmaxm, "")
                xl, yl = _lims_maldi()
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            ax.set_aspect("equal", adjustable="box")
            _strip_spines(ax)
            if (r, c) == (0, 0):
                # one letter for the whole 2x4 map grid
                _header(ax, "A", "Spatial maps", _LET_DY)
            if (r, c) == (0, 0):
                _panel_scalebar(ax, label="1 mm")
        axes[(r, 0)].annotate(
            pep,
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-9, 0),
            textcoords="offset points",
            rotation=90,
            ha="center",
            va="center",
            fontsize=_FS_TITLE,
            fontweight="bold",
            annotation_clip=False,
        )
    for c, t in enumerate(["Xenium", "MALDI", "Xenium", "MALDI"]):
        _title(axes[(0, c)], t)
    # caste headers over column pairs (house colours), with a thin rule
    for c0, t, col in [(0, "Queen", QUEEN_COLOR), (2, "Worker", WORKER_COLOR)]:
        x0, x1 = colx[c0], colx[c0 + 1] + IW
        yy = 1 - (HDR - 0.02) / H
        fig.text(
            (x0 + x1) / 2 / W,
            1.0,
            t,
            ha="center",
            va="top",
            fontsize=_FS_TITLE + 0.5,
            fontweight="bold",
            color=col,
        )
        fig.add_artist(plt.Line2D([x0 / W, x1 / W], [yy, yy], color=col, lw=1.0))

    # colorbars (under cols 0 and 1)
    spatial_sm = plt.cm.ScalarMappable(
        norm=_PowerNorm(gamma=SPATIAL_GAMMA, vmin=0, vmax=1), cmap=SOLID_VIRIDIS
    )
    maldi_sm = plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=0, vmax=1), cmap=MALDI_CMAP
    )
    for c, sm, lab in [(2, spatial_sm, "cells expressing"), (3, maldi_sm, "intensity")]:
        cax = _ax(colx[c] + 0.14, CB_Y, IW - 0.28, 0.07)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 1])
        cb.set_ticklabels(["lo", "hi"])
        cb.ax.tick_params(labelsize=_FS_TICK - 0.5, length=1.5, pad=1)
        cb.set_label(lab, fontsize=_FS_TICK, labelpad=1)
        cb.outline.set_linewidth(0.4)

    # ---- B: prevalence by region ------------------------------------------
    I_LEFT = img_right + 0.55
    ax_c = _ax(I_LEFT, top0, W - 0.06 - I_LEFT, maps_bot - top0)
    # inside only: prevalence among cells in each region
    _kc_j = kc_flags
    _ol_j = olm.to_numpy(dtype=bool, na_value=False)
    _jg = [
        ("sNPF", "Calyx", expr_snpf, _kc_j),
        ("sNPF", "Optic", expr_snpf, _ol_j),
        ("Tk", "Calyx", expr_tk, _kc_j),
        ("Tk", "Optic", expr_tk, _ol_j),
    ]
    _bwj = 0.38
    for _i, (_g, _r, _ex, _rm) in enumerate(_jg):
        _q = (_ex[_rm & queen_mask] > 0).mean() * 100 if (_rm & queen_mask).any() else 0
        _w = (
            (_ex[_rm & worker_mask] > 0).mean() * 100
            if (_rm & worker_mask).any()
            else 0
        )
        ax_c.bar(_i - _bwj / 2, _q, _bwj, color=QUEEN_COLOR, alpha=0.9)
        ax_c.bar(_i + _bwj / 2, _w, _bwj, color=WORKER_COLOR, alpha=0.9)
        _dz_r = int(round(_q - _w))
        print(f"  B {_g} {_r}: queen {_q:.2f}% worker {_w:.2f}%", flush=True)
        _dz_lbl = f"{_dz_r:+d} pp" if _dz_r != 0 else "0 pp"
        ax_c.annotate(
            _dz_lbl,
            xy=(_i, max(_q, _w)),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=_FS_TICK,
            fontweight="bold",
            clip_on=False,
            color=QUEEN_COLOR if _dz_r >= 2 else "#999999",
        )
    ax_c.set_xticks(range(len(_jg)))
    ax_c.set_xticklabels([f"{_g}\n{_r}" for _g, _r, _, _ in _jg], fontsize=_FS_TICK)
    ax_c.set_ylabel("% cells expressing", fontsize=_FS_LAB, labelpad=2)
    ax_c.set_ylim(0, 112)
    ax_c.set_yticks([0, 50, 100])
    ax_c.tick_params(axis="y", labelsize=_FS_TICK, length=2, pad=1.5)
    ax_c.tick_params(axis="x", length=0, pad=2)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)
    _header(ax_c, "B", "Prevalence", _LET_DY)

    # ---- C: two forests sharing y labels -----------------------------------
    JL = LM + 0.92
    JGAP = 0.12
    jw = (W - 0.08 - JL - JGAP) / 2
    axJ1 = _ax(JL, J_TOP, jw, J_H)
    axJ2 = _ax(JL + jw + JGAP, J_TOP, jw, J_H)
    _draw_forest(axJ1, "sNPF", True)
    _draw_forest(axJ2, "Tk", False)

    # ---- D: optic-lobe recruitment -----------------------------------------
    axK = _ax(JL, K_TOP, W - 0.08 - JL, K_H)
    _draw_optic_prev(axK)

    # C / D letters share A's and E's left x and sit just above their panels
    for s, tg, ytop in (
        ("C", "snRNA-seq caste effect", J_TOP),
        ("D", "Optic lobe recruitment", K_TOP),
    ):
        _header((LM / W, 1 - (ytop - 0.20) / H), s, tg, dy=-3)

    out = output_dir / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"    Saved {out}", flush=True)


# -- Main --------------------------------------------------------------------

if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        set_pub_style()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Loading {SPATIAL_H5AD}...", flush=True)
        adata = sc.read_h5ad(SPATIAL_H5AD)
        print(f"  {adata.n_obs:,} cells, {adata.n_vars} genes", flush=True)
        queen_mask = adata.obs["brain"].values == "queen"
        sp = adata.obsm["spatial_aligned"]
        z_levels = adata.obs["z_level"].values.astype(float)

        shared = _compute_shared(adata, sp, queen_mask)
        save_individual_panels(adata, sp, queen_mask, z_levels, OUTPUT_DIR, shared)
        compose_figure(adata, sp, queen_mask, z_levels, OUTPUT_DIR)

        print(f"\nDone. Panels + composite in {OUTPUT_DIR}/", flush=True)
