"""Shared utilities for all figure scripts (main + supplementary).

Provides consistent styling, save helpers, gene resolution, and
cell type column detection used across fig*_panels.py and figS_*.py.

``set_style()`` is a thin wrapper around
``bimp_st.style.set_pub_style()`` for backwards compatibility — new
code should import from the package directly.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from bimp_st.cluster_labels import get_short, get_spatial_short
from bimp_st.style import (  # noqa: F401  (re-exported for older imports)
    CASTE_COLORS,
    PANEL_LABEL_PRINTED_PT,
    QUEEN_COLOR,
    WORKER_COLOR,
    add_panel_headers,
    add_panel_labels,
    panel_header,
    panel_label_size,
    set_pub_style,
)

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("results/figures/si")

# ── Shared palettes (caste colours live in bimp_st.style) ────────────────────

EXPR_CMAP = LinearSegmentedColormap.from_list(
    "expr",
    [
        "#F5F5F5",
        "#FEE08B",
        "#F46D43",
        "#D53E4F",
        "#9E0142",
    ],
    N=256,
)

# Diverging colormap: queen-enriched (gold) ← white → worker-enriched (blue)
# Matches QUEEN_COLOR / WORKER_COLOR used in bars and volcano plots
CASTE_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "caste_div",
    [
        "#B8860B",
        QUEEN_COLOR,
        "#F5E6C0",
        "#FFFFFF",
        "#C0D8F0",
        WORKER_COLOR,
        "#2A5FAA",
    ],
    N=256,
)

# ── Standardized font sizes ─────────────────────────────────────────────────
PANEL_LABEL_SIZE = 16
TITLE_SIZE = 11
LABEL_SIZE = 10
TICK_SIZE = 8
LEGEND_SIZE = 8

# \linewidth of the SI in inches (481.9 pt, measured off a build).
LINEWIDTH_IN = 481.9 / 72.27

# Install the print-scale hook for scripts that import this module without
# going through bimp_st.style.set_pub_style(). Both routes are idempotent.
try:  # pragma: no cover - styling must never break a render
    from bimp_st.print_scale import install as _install_print_scale

    _install_print_scale()
except Exception:
    pass

# Legends are frameless by default. Pass frameon=True explicitly where a box is
# needed for legibility (e.g. a legend over a dense dark scatter).
LEGEND_KW = dict(fontsize=LEGEND_SIZE, frameon=False, framealpha=0.9, edgecolor="none")
SCALEBAR_KW = dict(lw=2.5, color="black", clip_on=False, solid_capstyle="butt")

# Xenium pixel scale (µm per pixel)
PIXEL_SCALE = 0.2125

# Random subsample of *large* AnnData objects for faster figure iteration.
# Default 10%; full data: ``export BIMP_FIG_SUBSAMPLE_FRACTION=1``
# Objects with n_obs < FIG_SUBSAMPLE_MIN_OBS are never subsampled (e.g. snRNA).
FIG_SUBSAMPLE_FRACTION = float(os.environ.get("BIMP_FIG_SUBSAMPLE_FRACTION", "0.1"))
FIG_SUBSAMPLE_MIN_OBS = 100_000

# ── Focal genes ──────────────────────────────────────────────────────────────
FOCAL_GENES = {
    "Vg": {
        "bimp": "BIMP_vNP3.1_05833",
        # Not canonical vitellogenin: this probe targets BIMP_v4.3_001050, a
        # 319-residue paralog with only the N-terminal lipid-binding domain,
        # ~22% identical to Vgn (BIMP_v4.3_005500, 1772 aa), which the panel
        # does not target.
        "full": "vitellogenin-family paralog",
    },
    "Kr-h1": {
        "bimp": "BIMP_vNP3.1_01378",
        "full": "Krüppel homolog 1",
    },
    "yl": {
        "bimp": "BIMP_vNP3.1_05619",
        "full": "Vitellogenin receptor",
    },
    "GAD1": {
        "bimp": "BIMP_vNP3.1_10812",
        "full": "Glutamic acid decarboxylase 1",
    },
    "apolpp": {
        "bimp": "BIMP_vNP3.1_08000",
        "full": "Apolipophorin",
    },
}

# Reader-facing gene names for figure art. FOCAL_GENES keys stay as the code
# identifiers (fig_neuropeptide_spatial.py and others index by "Vg"); the probe
# targets a vitellogenin-family paralog, not canonical Vgn, so figures call it
# Vg-like.
GENE_DISPLAY = {"Vg": "Vg-like"}


def gene_display(name: str) -> str:
    """Figure label for a focal-gene key (``"Vg"`` -> ``"Vg-like"``)."""
    return GENE_DISPLAY.get(name, name)


# ── Data paths ───────────────────────────────────────────────────────────────
SPATIAL_H5AD = "pipelines/combined_atlas/data/combined_clustered.h5ad"


# ── Style ────────────────────────────────────────────────────────────────────


def set_style():
    """Configure matplotlib for publication-quality output.

    Thin wrapper around ``bimp_st.style.set_pub_style()`` that wires in
    the shared ``TITLE_SIZE`` / ``LABEL_SIZE`` / ``TICK_SIZE`` constants
    defined at the top of this module.
    """
    set_pub_style(
        title_size=TITLE_SIZE,
        label_size=LABEL_SIZE,
        tick_size=TICK_SIZE,
        legend_size=LEGEND_SIZE,
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


def maybe_subsample_adata(
    adata, *, fraction: float | None = None, random_state: int = 42
):
    """Return a random subset of cells when fraction < 1 (speed for big objects).

    No-op for small AnnData (below ``FIG_SUBSAMPLE_MIN_OBS``) so e.g. snRNA
    reference UMAPs stay dense enough to read.
    """
    frac = FIG_SUBSAMPLE_FRACTION if fraction is None else fraction
    if frac >= 1.0 or adata.n_obs == 0:
        return adata
    if adata.n_obs < FIG_SUBSAMPLE_MIN_OBS:
        return adata
    n_keep = max(1, int(np.floor(adata.n_obs * frac)))
    rng = np.random.default_rng(random_state)
    idx = rng.permutation(adata.n_obs)[:n_keep]
    out = adata[idx].copy()
    print(
        f"  Subsampled {adata.n_obs:,} → {out.n_obs:,} cells "
        f"({frac:.0%}; BIMP_FIG_SUBSAMPLE_FRACTION=1 for full data)",
        flush=True,
    )
    return out


def save(fig, output_dir, name):
    """Save figure as PNG and close."""
    out = output_dir / f"{name}.png"
    fig.savefig(out)
    if os.environ.get("BIMP_ALSO_PDF") == "1":  # opt-in vector copy next to the PNG
        # no CreationDate: re-renders of the same artwork are byte-identical
        fig.savefig(out.with_suffix(".pdf"), metadata={"CreationDate": None})
    plt.close(fig)
    print(f"    Saved {out}", flush=True)


def add_scalebar(ax, shared_span, bar_um=1000, label="1 mm", *, center_x=None):
    """Add a scale bar below a spatial axes (coords in pixels, PIXEL_SCALE µm/px).

    Uses data coordinates so it's identical across all panels sharing the
    same axis limits, regardless of aspect-ratio reshaping.
    ``center_x`` places the bar under that x (data coords); default is axis midpoint.
    """
    bar_len_px = bar_um / PIXEL_SCALE
    xlo, xhi = ax.get_xlim()
    ylo, _yhi = ax.get_ylim()
    span = xhi - xlo
    cx = (xlo + xhi) / 2 if center_x is None else float(center_x)
    x0 = cx - bar_len_px / 2
    y0 = ylo - span * 0.04
    ax.plot([x0, x0 + bar_len_px], [y0, y0], **SCALEBAR_KW)
    ax.text(
        cx,
        y0 - span * 0.02,
        label,
        ha="center",
        va="top",
        fontsize=TICK_SIZE,
        clip_on=False,
    )


def resolve_gene(name, var_names):
    """Resolve a display gene name to its var_names entry."""
    if name in FOCAL_GENES:
        prefix = FOCAL_GENES[name]["bimp"]
        matches = [v for v in var_names if v.startswith(prefix)]
        return matches[0] if matches else None
    if name in var_names:
        return name
    return None


def get_expr(adata, gene):
    """Extract dense expression vector for a single gene."""
    X = adata[:, gene].X
    if hasattr(X, "todense"):
        return np.asarray(X.todense()).flatten()
    return np.asarray(X).flatten()


# ── Cell type naming ─────────────────────────────────────────────────────────


def _is_spatial(adata):
    """Heuristic: does this AnnData look like spatial (vs snRNA) data?"""
    cols = set(adata.obs.columns)
    return bool(cols & {"brain", "slice_name"})


def add_cell_type_names(adata, source="spatial"):
    """Add a ``cell_type`` column with short cluster names.

    Parameters
    ----------
    adata : AnnData
        Spatial (leiden → spatial labels) or snRNA (cell_type → short labels).
    source : ``"spatial"`` or ``"snrna"``
        Which naming scheme to use.

    For spatial data, maps leiden → cell_type via get_spatial_short().
    For snRNA data, uses cell_type directly (or maps via get_short()).
    """
    if "cell_type" in adata.obs.columns:
        return "cell_type"

    if source == "spatial":
        adata.obs["cell_type"] = (
            adata.obs["leiden"].astype(int).map(get_spatial_short).astype(str)
        )
    else:
        adata.obs["cell_type"] = (
            adata.obs["leiden"].astype(str).map(get_short).astype(str)
        )

    n = adata.obs["cell_type"].notna().sum()
    print(f"  Added cell_type names ({n:,}/{adata.n_obs:,} cells)", flush=True)
    return "cell_type"


def detect_ct_col(adata):
    """Get the cell_type column, creating it if needed.

    Auto-detects spatial vs snRNA data and routes to the appropriate
    naming scheme.
    """
    if "cell_type" in adata.obs.columns:
        return "cell_type"
    if "leiden" not in adata.obs.columns:
        return "leiden"
    source = "spatial" if _is_spatial(adata) else "snrna"
    return add_cell_type_names(adata, source=source)


def sort_cts(cts):
    """Sort cell type labels numerically."""
    try:
        return sorted(cts, key=lambda s: int(s))
    except (ValueError, TypeError):
        return sorted(cts)


def format_pval(pval, threshold=1e-300):
    """Format p-value consistently across all figures."""
    if pval < threshold:
        return "p < 1e-300"
    if pval < 1e-20:
        return "p < 1e-20"
    return f"p = {pval:.1e}"


def strip_spines(ax):
    """Remove spines and ticks from an axis."""
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
