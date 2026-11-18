"""Data for fig_neuropeptide_spatial.py:
loading, computation and the constants they use.

See fig_neuropeptide_spatial.py (the composer) for the figure and how to render it.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

from bimp_st import maldi
from bimp_st.paths import CASTE_DE_DIR  # noqa: E402
from bimp_st.si_common import (
    PIXEL_SCALE,
)
from bimp_st.si_common import (
    get_expr as _get_expr,
)

# -- Configuration -----------------------------------------------------------
# Registered render (manuscript_figure_registry.FIGURES): panel C reads the
# trial-blocked caste DE. BIMP_FIG4_OUT renders somewhere else (and then
# Supplementary figS_ap_gradient goes there too instead of the shared si/ dir).
DEFAULT_OUTPUT_DIR = Path("results/figures/main/neuropeptide_spatial")
OUTPUT_DIR = Path(os.environ.get("BIMP_FIG4_OUT", DEFAULT_OUTPUT_DIR))
# Per-panel MALDI contrast: the runs are separate and median-normalized, so each
# map keeps its own scale, clipped at this percentile of its tissue bins.
MALDI_PANEL_VMAX_PCTL = 96

# Spatial density rendering:
#   DENSITY_MODE = "fracpos" (fraction of cells expressing; density-normalized,
#                  so the MB's high cell packing doesn't inflate it) |
#                  "mean" (per cell) | "total" (sum/area, density-amplified)
DENSITY_MODE = "fracpos"
DENSITY_BIN_UM = 35.0
DENSITY_SMOOTH = 1.3

# Focal genes: NP3.1 prefix -> (short name, description)
FOCAL_GENES = {
    "sNPF": "BIMP_vNP3.1_01359",  # short neuropeptide F (queen-enriched)
    "Tk": "BIMP_vNP3.1_07029",  # tachykinin (queen-enriched)
}


def resolve_gene(name, var_names):
    prefix = FOCAL_GENES.get(name)
    if prefix:
        matches = [v for v in var_names if v.startswith(prefix)]
        return matches[0] if matches else None
    return name if name in var_names else None


def _normalize_z(z_levels, queen_mask):
    z_norm = np.zeros_like(z_levels, dtype=float)
    for mask in [queen_mask, ~queen_mask]:
        z = z_levels[mask]
        zmin, zmax = z.min(), z.max()
        if zmax > zmin:
            z_norm[mask] = (z - zmin) / (zmax - zmin)
    return z_norm


def _density_map(
    sp_um, v, span_um, bin_um=None, smooth=None, tissue_frac=0.05, mode=None
):
    """Smoothed expression heatmap over a square span_um window centered on the
    brain bbox. mode='total' -> sum / area (matches MALDI's total signal per
    area); mode='mean' -> per-cell average. Returns (img[row=y, col=x] with NaN
    off-tissue, extent=(x0,x1,y0,y1))."""
    from scipy import ndimage

    bin_um = DENSITY_BIN_UM if bin_um is None else bin_um
    smooth = DENSITY_SMOOTH if smooth is None else smooth
    mode = DENSITY_MODE if mode is None else mode
    xlo, xhi = np.percentile(sp_um[:, 0], [0.5, 99.5])
    ylo, yhi = np.percentile(sp_um[:, 1], [0.5, 99.5])
    cx, cy = (xlo + xhi) / 2, (ylo + yhi) / 2
    half = span_um / 2
    x0, x1, y0, y1 = cx - half, cx + half, cy - half, cy + half
    ex = np.arange(x0, x1 + bin_um, bin_um)
    ey = np.arange(y0, y1 + bin_um, bin_um)
    wsum = np.histogram2d(sp_um[:, 0], sp_um[:, 1], bins=[ex, ey], weights=v)[0]
    cnt = np.histogram2d(sp_um[:, 0], sp_um[:, 1], bins=[ex, ey])[0]
    pos = np.histogram2d(
        sp_um[:, 0], sp_um[:, 1], bins=[ex, ey], weights=(v > 0).astype(float)
    )[0]
    ws = ndimage.gaussian_filter(wsum, smooth)  # smoothed sum of expr
    cs = ndimage.gaussian_filter(cnt, smooth)  # smoothed cell count
    ps = ndimage.gaussian_filter(pos, smooth)  # smoothed expressing-cell count
    tissue = cs > tissue_frac * cs.max() if cs.max() > 0 else cs > 0
    eps = 1e-9
    metric = {
        "mean": ws / np.maximum(cs, eps),  # mean expr / cell
        "meanpos": ws / np.maximum(ps, eps),  # mean among expressing
        "fracpos": ps / np.maximum(cs, eps),  # fraction of cells "on"
        "posdens": ps,  # expressing cells / area
        "total": ws,  # total expr / area
        "logtotal": np.log1p(ws),  # compressed total
        "celldens": cs,  # cell density (control)
    }.get(mode, ws / np.maximum(cs, eps))
    img = np.where(tissue, metric, np.nan)
    return img.T, (x0, x1, y0, y1)


# -- MALDI panels (bimp_st.maldi) ---------------------------------------------


def _maldi_available(sample_key):
    path = (
        maldi.PRINCETON_DIR
        / f"aligned_panel_princeton_{maldi.SAMPLES[sample_key]}.h5ad"
    )
    return path.exists()


def _image_window(img, extent, pad=0.06):
    """Square (cx, cy, span) tight to the finite (brain) region of img, in the
    physical coords given by extent=(xmin, xmax, ymin, ymax)."""
    finite = np.isfinite(img)
    ys, xs = np.where(finite)
    if xs.size == 0:
        cx = (extent[0] + extent[1]) / 2
        cy = (extent[2] + extent[3]) / 2
        return cx, cy, (extent[1] - extent[0])
    H, W = img.shape
    sx = (extent[1] - extent[0]) / W
    sy = (extent[3] - extent[2]) / H
    x0 = extent[0] + xs.min() * sx
    x1 = extent[0] + (xs.max() + 1) * sx
    y0 = extent[2] + ys.min() * sy
    y1 = extent[2] + (ys.max() + 1) * sy
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    span = max(x1 - x0, y1 - y0) * (1 + pad)
    return cx, cy, span


def _kc_enrichment(expr, mask, kc_flags):
    """KC-vs-rest fold enrichment of *expr* for cells in *mask*.

    kc_flags = boolean array (cell is in any Kenyon-cell anatomical region).
    Returns the ratio of the fraction of cells expressing Tk inside KCs to the
    fraction outside KCs. Fraction-expressing (prevalence), not mean amount, is
    the specificity metric, because the fraction expressing inside KCs is
    near saturation in both castes: a higher ratio means Tk is confined to
    KCs, a lower one that it extends into non-KC neurons.
    """
    in_kc = mask & kc_flags
    out_kc = mask & ~kc_flags
    frac_in = (expr[in_kc] > 0).mean()
    frac_out = (expr[out_kc] > 0).mean()
    return float(frac_in / frac_out) if frac_out > 0 else float("nan")


def _load_maldi_obs(sample_key, peptide):
    """Upright median-normalized MALDI map of one brain for 'Tk' or 'sNPF'
    (bimp_st.maldi). Returns (img, extent, vmin, vmax, readout column)."""
    img, ext, column = maldi.peptide_map(sample_key, peptide)
    img = img.copy()
    finite = img[np.isfinite(img)]
    panel_vmax = float(np.percentile(finite, MALDI_PANEL_VMAX_PCTL))
    return img, ext, 0.0, panel_vmax, column


# -- Shared state for panels ------------------------------------------------


def _compute_shared(adata, sp, queen_mask):
    """Compute expression arrays and shared axis limits for all panels."""
    worker_mask = ~queen_mask
    genes = {}
    for name in ["sNPF", "Tk"]:
        var = resolve_gene(name, adata.var_names)
        if var is None:
            raise SystemExit(f"ERROR: {name} not found in data")
        genes[name] = _get_expr(adata, var)

    expr_snpf, expr_tk = genes["sNPF"], genes["Tk"]
    all_pos = np.concatenate(
        [
            expr_snpf[expr_snpf > 0],
            expr_tk[expr_tk > 0],
        ]
    )
    shared_vmax = np.percentile(all_pos, 95) if len(all_pos) > 0 else 1

    q_idx, w_idx = np.where(queen_mask)[0], np.where(worker_mask)[0]
    q_ext = max(np.ptp(sp[q_idx, 0]), np.ptp(sp[q_idx, 1]))
    w_ext = max(np.ptp(sp[w_idx, 0]), np.ptp(sp[w_idx, 1]))
    shared_span = max(q_ext, w_ext) * 1.05
    shared_span_um = shared_span * PIXEL_SCALE

    return {
        "expr_tk": expr_tk,
        "expr_snpf": expr_snpf,
        "shared_vmax": shared_vmax,
        "shared_span": shared_span,
        "shared_span_um": shared_span_um,
    }


# -- Panels C and D: snRNA-seq caste forest, optic-lobe prevalence -----------

# Central-brain cell-type rows: (label, log2FC, lfcSE, padj), DESeq2 n=3, +=queen-up,
# read from the trial-blocked fit (~ trial + caste;
# scripts/caste_de/caste_de_blocked_all.py).
_PANEL_K_DE = CASTE_DE_DIR / "de_coarse_blocked.csv"
_PANEL_K_GENES = {"sNPF": "BIMP_v4.3_005730", "Tk": "BIMP_v4.3_009257"}
_PANEL_K_ROWS = [
    ("Whole brain", "all"),
    ("Kenyon cells", "Kenyon_cells"),
    ("Other neurons", "Other_neurons"),
    ("Projection neurons", "Projection_neurons"),
]


def _panel_k_cb() -> dict:
    de = pd.read_csv(_PANEL_K_DE)
    out = {}
    for gene, gid in _PANEL_K_GENES.items():
        g = de[de.gene == gid].set_index("cell_type")
        out[gene] = [
            (
                lab,
                float(g.loc[ct, "log2FoldChange"]),
                float(g.loc[ct, "lfcSE"]),
                float(g.loc[ct, "padj"]),
            )
            for lab, ct in _PANEL_K_ROWS
        ]
    return out


def _bbox(img, extent):
    """(x0, x1, y0, y1) of the finite region of img, in extent coordinates."""
    ys, xs = np.where(np.isfinite(img))
    H, W = img.shape
    sx = (extent[1] - extent[0]) / W
    sy = (extent[3] - extent[2]) / H
    return (
        extent[0] + xs.min() * sx,
        extent[0] + (xs.max() + 1) * sx,
        extent[2] + ys.min() * sy,
        extent[2] + (ys.max() + 1) * sy,
    )
