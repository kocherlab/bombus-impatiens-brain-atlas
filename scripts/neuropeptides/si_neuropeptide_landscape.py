"""Neuropeptide caste-bias landscape (companion to `fig:neuropeptide_spatial`).

Heatmap: rows = neuropeptide precursor genes (BIMPv4.3 MALDI neuropeptidome),
cols = major snRNA cell classes, value = (queen% - worker%) of cells expressing
each precursor (percentage-point difference in prevalence). Blue = worker-lean,
gold = queen-lean. sNPF & Tk are boxed to place the two focal peptides in the
broader caste pattern.

Source: data/sn/BIMP_atlas_qw_combined.h5ad (caste + major_cell_type).
Descriptive prevalence; central-brain classes n=3 bees/caste, optic-lobe column
n=1 dissection/caste. Run:
    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/neuropeptides/si_neuropeptide_landscape.py
"""

import os
import sys
from pathlib import Path

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
from scipy.sparse import issparse

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bimp_st.si_common import (  # noqa: E402
    CASTE_DIVERGING_CMAP,
    set_style,
)

ATLAS = "data/sn/BIMP_atlas_qw_combined.h5ad"
# BIMP_NP_OUT redirects the figure (and its matrix / n tables) so a check never
# overwrites the registered file.
OUT = Path(
    os.environ.get("BIMP_NP_OUT", "results/figures/si/figS_neuropeptide_landscape.png")
)
OUT.parent.mkdir(parents=True, exist_ok=True)
# The reference carries a generic cell_type "Kenyon cells" entry (662 optic-lobe
# nuclei, 335 queen + 327 worker) that is not one of the five central-brain Kenyon
# types. It is excluded from the Kenyon column here, so that column is the 32,672
# central-brain nuclei (n = 3 bees per caste);
# BIMP_NP_INCLUDE_GENERIC=1 includes it.
INCLUDE_GENERIC = os.environ.get("BIMP_NP_INCLUDE_GENERIC", "0") == "1"

# base gene_id -> display name (named precursors from BIMPv4.3_neuropeptides_MALDI.tsv)
NP = {
    "BIMP_v4.3_009257": "Tachykinin",
    "BIMP_v4.3_005730": "sNPF",
    "BIMP_v4.3_007506": "Neuropeptide F",
    "BIMP_v4.3_009248": "Allatostatin-A",
    "BIMP_v4.3_000471": "Allatotropin",
    "BIMP_v4.3_002239": "Corazonin",
    "BIMP_v4.3_007299": "SIFamide",
    "BIMP_v4.3_003387": "Myosuppressin",
    "BIMP_v4.3_010316": "FMRFamide",
    "BIMP_v4.3_000907": "Ion transport peptide",
    "BIMP_v4.3_000407": "Diuretic hormone 44",
    "BIMP_v4.3_008248": "Diuretic hormone 31",
    "BIMP_v4.3_009546": "CAPA",
    "BIMP_v4.3_003354": "Orcokinin",
    "BIMP_v4.3_003584": "PDF",
    "BIMP_v4.3_006216": "Insulin-like peptide 2",
    "BIMP_v4.3_006404": "Nplp1",
    "BIMP_v4.3_010795": "Bursicon-alpha",
    "BIMP_v4.3_010796": "Bursicon-beta",
    "BIMP_v4.3_009293": "CCHamide-1",
    "BIMP_v4.3_008883": "PBAN",
    "BIMP_v4.3_005801": "Eclosion hormone",
    "BIMP_v4.3_007298": "Ecdysis-trig. hormone",
    "BIMP_v4.3_007504": "Space blanket",
    "BIMP_v4.3_008745": "Hypertrehalosaemic h.",
    "BIMP_v4.3_004066": "Prohormone-1",
    "BIMP_v4.3_003417": "Prohormone-2",
    "BIMP_v4.3_003431": "Prohormone-4",
    "BIMP_v4.3_007977": "Queen brain-sel. prot.-1",
}
# Display-only row names. NP's values are the matrix CSV's row index, so the
# abbreviated forms there stay as they are; the figure spells them out.
DISPLAY = {
    "Ecdysis-trig. hormone": "Ecdysis-triggering hormone",
    "Hypertrehalosaemic h.": "Hypertrehalosaemic hormone",
    "Queen brain-sel. prot.-1": "Queen brain-selective protein 1",
}
COLS = [
    "Kenyon cells",
    "Projection neurons",
    "Other neurons",
    "Optic lobe neurons",
    "Photoreceptors",
    "Glia",
    "Hemocytes",
]
NLAB = {}  # filled from the data in compute_matrix (nuclei per class, rounded)
CAP = 20.0


def compute_matrix():
    A = ad.read_h5ad(ATLAS, backed="r")
    present = [g for g in NP if g in A.var_names]
    sub = A[:, present].to_memory()
    X = sub.X
    X = X.toarray() if issparse(X) else np.asarray(X)
    expr = X > 0
    caste = A.obs["caste"].astype(str).values
    mct = A.obs["major_cell_type"].astype(str).values
    generic = A.obs["cell_type"].astype(str).values == "Kenyon cells"
    mat = pd.DataFrame(index=[NP[g] for g in present], columns=COLS, dtype=float)
    for ci, ct in enumerate(COLS):
        ctm = mct == ct
        if ct == "Kenyon cells" and not INCLUDE_GENERIC:
            ctm = ctm & ~generic
        n = int(ctm.sum())
        NLAB[ct] = (
            f"{n / 1000:.1f}k"
            if n >= 10000
            else (f"{n / 1000:.1f}k" if n >= 1000 else str(n))
        )
        for gi in range(len(present)):
            qm = ctm & (caste == "Queen")
            wm = ctm & (caste == "Worker")
            qp = expr[qm, gi].mean() * 100 if qm.sum() else np.nan
            wp = expr[wm, gi].mean() * 100 if wm.sum() else np.nan
            mat.iloc[gi, ci] = qp - wp
    return mat


def render(mat):
    set_style()
    # drop precursors flat across all classes (no cell reaches the +/-1
    # percentage-point floor)
    mat = mat.loc[mat.abs().max(axis=1) >= 1.0]
    mat = mat.reindex(mat.abs().max(axis=1).sort_values(ascending=False).index)
    M = mat.values.astype(float)
    nr, nc = M.shape
    cmap = CASTE_DIVERGING_CMAP.reversed()  # low(-)->blue worker, high(+)->gold queen
    fig, ax = plt.subplots(figsize=(7.8, 8.6))
    im = ax.imshow(M, cmap=cmap, vmin=-CAP, vmax=CAP, aspect="auto")
    for i in range(nr):
        for j in range(nc):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "·", ha="center", va="center", color="#BBB", fontsize=7)
                continue
            txt = f"{v:+.0f}".replace("-", "\u2212") if abs(v) >= 0.5 else "0"
            tc = "white" if abs(v) > 12 else ("#333333" if abs(v) >= 0.5 else "#BFBFBF")
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=6.6,
                color=tc,
                fontweight="bold" if abs(v) >= 8 else "normal",
            )
    ax.set_xticks(np.arange(nc))
    ax.set_xticklabels(
        [f"{c}\n(n≈{NLAB[c]})" for c in mat.columns],
        fontsize=7.5,
        rotation=40,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_yticks(np.arange(nr))
    ax.set_yticklabels([DISPLAY.get(g, g) for g in mat.index], fontsize=8)
    for gi, gene in enumerate(mat.index):
        if gene in ("sNPF", "Tachykinin"):
            ax.get_yticklabels()[gi].set_fontweight("bold")
            ax.add_patch(
                Rectangle(
                    (-0.5, gi - 0.5),
                    nc,
                    1,
                    fill=False,
                    edgecolor="#111111",
                    lw=1.6,
                    zorder=5,
                )
            )
    ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # No figure title and no restating footnote here — both are redundant
    # with the LaTeX caption (fig:neuropeptide_landscape), which already
    # states the n=3/n=1 caveat, the colour-scale cap, and the omission rule.
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, extend="both")
    cb.set_label("Queen \u2212 worker prevalence (percentage points)", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    # End labels: the reversed caste colormap puts +20 (gold) at the top.
    for y, va, word in ((1.07, "bottom", "Queen"), (-0.07, "top", "Worker")):
        cb.ax.text(
            0.5,
            y,
            word,
            transform=cb.ax.transAxes,
            ha="center",
            va=va,
            fontsize=8,
            fontweight="bold",
        )
    fig.subplots_adjust(left=0.27, right=0.99, top=0.97, bottom=0.14)
    save_kw = dict(facecolor="white", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT, dpi=300, **save_kw)
    if os.environ.get("BIMP_ALSO_PDF") == "1":  # opt-in vector copy next to the PNG
        fig.savefig(OUT.with_suffix(".pdf"), **save_kw)
    print("saved", OUT)


if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        _mat = compute_matrix()
        _mat.to_csv(OUT.with_suffix(".matrix.csv"))
        pd.Series(NLAB, name="n_nuclei_label").to_csv(OUT.with_suffix(".n.csv"))
        render(_mat)
