#!/usr/bin/env python
"""Figure ``fig:caste_de``: snRNA-seq caste DE and joint cNMF programs.

Analysis design: I. Traniello. Writes
results/figures/main/caste_de/fig_caste_de.{png,pdf}.

Content follows the caste-DE analysis notebook
(``<snrna_source>/src/scripts/03_analysis/
01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb``, cell 30); styling is the
shared house style from ``bimp_st.style`` (7.09 in wide; panel letter and
title via ``add_panel_headers``; queen gold / worker blue).

Inputs (no h5ad is opened):
- ``bimp_st.paths.CASTE_DE_FINE``: the repo's fine-type caste DE
  (``caste_de_blocked_all.py``, recipe ``bimp_st.caste_de``: PyDESeq2
  ``~ trial + caste``, whole trial pairs, BH, padj < 0.05; parity with the
  saved table in ``parity_saved_de.py``).
- ``results/caste_programs/pipeline/fig3_inputs/``: the joint cNMF (k = 50)
  usage and gene scores of the five figure programs, the per-program caste table
  and the GSEA program labels (panel d), from the repository port of the
  analysis notebook code (``scripts/caste_de/cnmf_gsea/``, ``run_pipeline.sh
  all``; parity in ``results/caste_programs/pipeline/parity/``).
  ``--saved-inputs`` reads the analysis's SAVED outputs instead
  (``bimp_st.paths.CASTE_DE_SAVED_INPUTS``), for comparison.
- ``BIMP_lookup.csv`` in ``bimp_st.paths.CASTE_DE_SAVED_INPUTS``: the gene
  annotation (``caste_de_freeze_saved_outputs.py``, PROVENANCE.md).

Panels
  A  Spearman of the Wald statistic between the 11 glial / neuronal types, over
     genes tested in both whose baseMean is >= that type's median in both
     (>= 500 genes per pair).
  B  DEGs per 1,000 tested genes, worker-up left, queen-up right; the number is
     the TOTAL (queen-up + worker-up). Rows grouped Glia / Neurons / Unassigned /
     Hemocytes, sorted by total within the group.
  C  For the whole brain (glia + neurons, 10 types, Other KC omitted), neurons
     and glia: the five annotated genes or lncRNAs significant (same sign) in the
     most types, ties broken by mean |Wald| over those types. Colour = log2FC;
     star = FDR < 0.05 in either direction; grey = not tested; n / types in the
     block (the analysis's denominator).
  D  Joint cNMF programs 18, 20, 6, 7, 12, titled by GO label only.
     Strip = Spearman rho between program gene scores and the type's Wald
     statistic (all genes with a statistic; colour only, no printed values);
     boxes = per-nucleus usage (%), whiskers 5th-95th percentile; dots = the
     three trial means; grey = replicated caste effect (same sign in all three
     trials, |mean| >= 0.5 SD, weakest trial >= 0.25 SD, mean usage >= 2%).

    PYTHONPATH=src:scripts .venv/bin/python scripts/caste_de/fig_caste_de.py

Writes ``<out-dir>/<out-name>.{png,pdf}`` (default
``results/figures/main/caste_de/fig_caste_de``) and the plotted numbers to
``--tables-dir`` (default ``results/caste_programs/pipeline/fig3_tables/``;
``fig3_tables/`` in ``bimp_st.paths.CASTE_DE_SAVED_DIR`` with
``--saved-inputs``). The compute
functions (``load_de``, ``panel_a``, ``panel_b``, ``recurrence``,
``panel_d_data``) import without drawing anything.
"""

import argparse
import textwrap
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

from bimp_st.caste_de import FDR  # noqa: E402
from bimp_st.paths import (  # noqa: E402
    CASTE_DE_FINE,
    CASTE_DE_SAVED_DIR,
    CASTE_DE_SAVED_INPUTS,
    CASTE_PROGRAMS_DIR,
    REPO,
    explain_missing_inputs,
)
from bimp_st.style import (  # noqa: E402
    QUEEN_COLOR,
    WORKER_COLOR,
    panel_header,
    set_pub_style,
)

RES = REPO / "results"
SAVED = CASTE_DE_SAVED_INPUTS
R9 = CASTE_PROGRAMS_DIR / "pipeline"  # cnmf_gsea/run_pipeline.sh outputs
DE = CASTE_DE_FINE
OUT = RES / "figures/main/caste_de"
TOP = 5
A_MIN_GENES = 500
A_LIM, D_LIM = 0.6, 0.4  # colour limits
PROGS = [18, 20, 6, 7, 12]
USE_FLOOR = 2.0

GLIA = ["Astrocyte-like glia", "Cortex glia", "Perineurial glia", "Subperineurial glia"]
KCS = [
    "Class II Kenyon cells",
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Other Kenyon cells",
]
PNS = ["Anterodorsal projection neurons", "Lateral projection neurons"]
A_ORDER = GLIA + KCS + PNS
B_GROUPS = {
    "Glia": GLIA,
    "Neurons": KCS + PNS,
    # Other glia is not tested (two trial pairs only; bimp_st.caste_de.MIN_PER_CASTE)
    "Unassigned": ["Other neurons"],
    "Hemocytes": ["Hemocytes"],
}
# panel c: 10 types, glia then neurons, each alphabetical (Other KC omitted)
C_NEURON = sorted(KCS[:-1] + PNS)
C_BLOCKS = {"Whole brain": GLIA + C_NEURON, "Neurons": C_NEURON, "Glia": GLIA}
D_GROUPS = {"Glia": GLIA, "Kenyon cells": KCS, "Other": PNS + ["Hemocytes"]}
NAME = {
    "Astrocyte-like glia": "Astrocyte-like",
    "Cortex glia": "Cortex glia",
    "Perineurial glia": "Perineurial",
    "Subperineurial glia": "Subperineurial",
    "Class II Kenyon cells": "Class II KC",
    "Large-type Kenyon cells": "Large-type KC",
    "Middle-type Kenyon cells": "Middle-type KC",
    "Small-type Kenyon cells": "Small-type KC",
    "Other Kenyon cells": "Other KC",
    "Anterodorsal projection neurons": "Anterodorsal PN",
    "Lateral projection neurons": "Lateral PN",
    "Hemocytes": "Hemocytes",
    "Other neurons": "Other neurons",
    "Other glia": "Other glia",
}
# The annotation name alone misleads for these, so the ID is added: 004137 is
# annotated heat shock protein 90 but is the panel's Hsp83; 001975 is a separate
# heat shock protein 83; 009252 matches futsch only by domain.
ALWAYS_ID = {"BIMP_v4.3_004137", "BIMP_v4.3_001975", "BIMP_v4.3_009252"}
TICK_ROT = 50


# ── data ────────────────────────────────────────────────────────────────────
def load_de(path=DE) -> pd.DataFrame:
    de = pd.read_csv(path)
    de["tested"] = de.tested.astype(bool)
    de["sig"] = de.tested & (de.padj < FDR)
    return de


def load_lookup() -> pd.DataFrame:
    lk = pd.read_csv(SAVED / "BIMP_lookup.csv", sep="\t").drop_duplicates("gene_id")
    return lk.set_index("gene_id")


def label_text(lk, g):
    """The analysis's label_text: 'name: description', or whichever exists."""
    if g not in lk.index:
        return None
    r = lk.loc[g]
    d = (
        r.description
        if isinstance(r.description, str) and r.description.strip()
        else None
    )
    n = r.gene if isinstance(r.gene, str) and r.gene.strip() else None
    return f"{n}: {d}" if (d and n) else (d or n)


def is_lnc(lk, g):
    return g in lk.index and lk.at[g, "gene_biotype"] == "lncRNA"


def keep_set(lk) -> set:
    """The analysis's keep_set: genes with an annotation, or lncRNAs."""
    return {g for g in lk.index if label_text(lk, g) or is_lnc(lk, g)}


def gene_label(lk, g, width=26, max_lines=2):
    sid = g.replace("BIMP_v4.3_", "")
    text = label_text(lk, g)
    if text is None:
        text = f"lncRNA [{sid}]" if is_lnc(lk, g) else f"unannotated [{sid}]"
    elif g in ALWAYS_ID:
        text += f" [{sid}]"
    lines = textwrap.wrap(text, width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(" ,;:") + "…"
    return "\n".join(lines)


def panel_a(de):
    d = de[de.tested]
    W = d.pivot_table(index="gene", columns="cell_type", values="stat")[A_ORDER]
    B = d.pivot_table(index="gene", columns="cell_type", values="baseMean")[A_ORDER]
    thr = B.quantile(0.5)
    M = pd.DataFrame(np.nan, index=A_ORDER, columns=A_ORDER)
    N = pd.DataFrame(0, index=A_ORDER, columns=A_ORDER)
    for a in A_ORDER:
        for b in A_ORDER:
            if a == b:
                continue
            keep = (B[a] >= thr[a]) & (B[b] >= thr[b])
            sub = W.loc[keep, [a, b]].dropna()
            N.loc[a, b] = len(sub)
            if len(sub) >= A_MIN_GENES:
                M.loc[a, b] = stats.spearmanr(sub[a], sub[b])[0]
    return M, N


def panel_b(de):
    t = de[de.tested].groupby("cell_type").size()
    q = de[de.sig & (de.log2FoldChange > 0)].groupby("cell_type").size()
    w = de[de.sig & (de.log2FoldChange < 0)].groupby("cell_type").size()
    tab = pd.DataFrame({"tested": t, "queen_up": q, "worker_up": w}).fillna(0)
    tab = tab.astype(int)
    tab["total"] = tab.queen_up + tab.worker_up
    tab["queen_per1000"] = tab.queen_up / tab.tested * 1000
    tab["worker_per1000"] = tab.worker_up / tab.tested * 1000
    return tab


def recurrence(de, keep, types, direction):
    """The analysis's top_recurrent: n types significant with that sign (annotated
    genes and lncRNAs only, ``keep``), ties broken by mean |Wald| over those types."""
    sgn = 1 if direction == "queen" else -1
    d = de[de.cell_type.isin(types)]
    s = d[d.sig & (np.sign(d.log2FoldChange) == sgn) & d.gene.isin(keep)]
    agg = s.groupby("gene").agg(
        n=("cell_type", "nunique"), m=("stat", lambda x: x.abs().mean())
    )
    rank = agg.sort_values(["n", "m"], ascending=False).head(TOP)
    lfc = d.pivot(index="gene", columns="cell_type", values="log2FoldChange")
    pad = d.pivot(index="gene", columns="cell_type", values="padj")
    tst = d.pivot(index="gene", columns="cell_type", values="tested")
    lfc, pad, tst = (
        x.reindex(index=rank.index, columns=types) for x in (lfc, pad, tst)
    )
    tst = tst.eq(True)
    rank["n_tested"] = tst.sum(axis=1)
    return rank, lfc, pad, tst


def panel_d_data(de, cnmf_dir):
    use = pd.read_csv(cnmf_dir / "cnmf_k50_fig_usage.csv.gz", index_col=0)
    gs = pd.read_csv(cnmf_dir / "cnmf_k50_fig_gene_scores.csv.gz", index_col=0)
    jc = pd.read_csv(cnmf_dir / "joint_k50_caste.csv")
    jc["cell_type"] = jc.cell_type.str.replace(r"\s+", " ", regex=True)
    labels = pd.read_csv(cnmf_dir / "program_labels.csv").set_index("program")
    rows = []
    for p in PROGS:
        for ct in [c for v in D_GROUPS.values() for c in v]:
            wv = de[de.cell_type == ct].set_index("gene").stat.dropna()
            g = gs[f"p{p}"].reindex(wv.index).dropna()
            rho = (
                stats.spearmanr(g, wv.reindex(g.index))[0] if len(wv) >= 200 else np.nan
            )
            r = jc[(jc.cell_type == ct) & (jc.program == p)]
            r = r.iloc[0] if len(r) else None
            flag = bool(
                r is not None
                and r["same"]
                and abs(r["mean"]) >= 0.5
                and r["min_abs"] >= 0.25
                and r["usage_pct"] >= USE_FLOOR
            )
            u = use[use.cell_type == ct]
            rows.append(
                dict(
                    program=p,
                    cell_type=ct,
                    rho=rho,
                    n_genes_rho=len(g),
                    mean_effect_sd=None if r is None else r["mean"],
                    min_abs_sd=None if r is None else r["min_abs"],
                    same_sign=None if r is None else bool(r["same"]),
                    usage_pct=float(u[f"p{p}"].mean()),
                    usage_pct_saved=None if r is None else r["usage_pct"],
                    replicated=flag,
                )
            )
    return use, pd.DataFrame(rows), labels, jc


# ── figure ──────────────────────────────────────────────────────────────────
W = 7.09
FS, FSS = 6.0, 5.5
A_CELL = 0.145
C_CELL_W, C_ROW = 0.125, 0.155
LX = 0.02
HDR_BASE = 0.1  # in: header top to its baseline
ROW1_HDR = -0.08  # row-1 headers sit above the canvas top (tight bbox keeps them)
D_TITLE, D_STRIP, D_PLOT, D_GAP = 0.09, 0.06, 0.17, 0.03


def draw(de, lk, cnmf_dir, tab_dir, out_dir, out_name):
    """Render the figure and write the plotted tables; returns them for the
    printed comparison."""
    set_pub_style(
        font_size=FS,
        title_size=FS + 0.5,
        label_size=FS,
        tick_size=FSS,
        legend_size=FSS,
        axes_linewidth=0.5,
    )
    QC, WC = QUEEN_COLOR, WORKER_COLOR
    n = len(A_ORDER)
    row1_h = 0.25 + n * A_CELL + 0.6
    c_h = 0.32 + 2 * (0.16 + TOP * C_ROW) + 0.08 + 0.54
    d_rows_h = len(PROGS) * (D_TITLE + D_STRIP + 0.02 + D_PLOT + D_GAP)
    d_h = 0.22 + 0.18 + d_rows_h + 0.5
    H = 0.04 + row1_h + 0.04 + c_h + 0.04 + d_h
    fig = plt.figure(figsize=(W, H))

    def ax_in(x, y_top, w, h):
        return fig.add_axes([x / W, 1 - (y_top + h) / H, w / W, h / H])

    def ftext(x, y, s, **kw):
        return fig.text(x / W, 1 - y / H, s, **kw)

    def title(x, y, letter, text):
        """House panel header (bold letter + bold title) with its top at y in."""
        panel_header(
            fig,
            (x / W, 1 - (y + HDR_BASE) / H),
            letter,
            text,
            letter_size=8.5,
            dy_pts=0,
        )

    def hcolorbar(x, y, w, lim, cmap, label, label_left=True):
        cax = ax_in(x, y, w, 0.06)
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(-lim, lim), cmap=cmap)
        cb = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[-lim, 0, lim])
        cb.ax.tick_params(labelsize=FSS, length=2, width=0.5, pad=1)
        cb.outline.set_visible(False)
        cb.ax.set_xticklabels([f"{-lim:g}".replace("-", "−"), "0", f"{lim:g}"])
        if label_left:
            ftext(x - 0.06, y + 0.03, label, fontsize=FS, ha="right", va="center")
        return cb

    tab_dir.mkdir(parents=True, exist_ok=True)

    # ---- a
    M, N = panel_a(de)
    M.to_csv(tab_dir / "panel_a_spearman.csv")
    N.to_csv(tab_dir / "panel_a_n_genes.csv")
    ax_x = 0.92
    ax = ax_in(ax_x, 0.25, n * A_CELL, n * A_CELL)
    ax.imshow(
        np.ma.masked_invalid(M.to_numpy()),
        cmap="RdBu_r",
        vmin=-A_LIM,
        vmax=A_LIM,
        aspect="auto",
    )
    for i in range(n):
        for j in range(n):
            v = M.iloc[i, j]
            if np.isfinite(v):
                s = f"{v:.2f}".replace("0.", ".").replace("-", "−")
                ax.text(
                    j,
                    i,
                    s,
                    ha="center",
                    va="center",
                    fontsize=FSS - 1.0,
                    color="white" if abs(v) > 0.42 else "black",
                )
    ax.set_xticks(range(n))
    ax.set_xticklabels(
        [NAME[c] for c in A_ORDER],
        rotation=TICK_ROT,
        ha="right",
        rotation_mode="anchor",
        fontsize=FS,
    )
    ax.set_yticks(range(n))
    ax.set_yticklabels([NAME[c] for c in A_ORDER], fontsize=FS)
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    title(LX, ROW1_HDR, "A", "Similarity of the caste effect")
    cax_a = ax_in(ax_x + n * A_CELL + 0.08, 0.25, 0.06, 0.7)
    sm_a = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(-A_LIM, A_LIM), cmap="RdBu_r"
    )
    cb_a = fig.colorbar(sm_a, cax=cax_a, ticks=[-A_LIM, 0, A_LIM])
    cb_a.ax.set_yticklabels([f"−{A_LIM:g}", "0", f"{A_LIM:g}"])
    cb_a.ax.tick_params(labelsize=FSS, length=2, width=0.5, pad=1)
    cb_a.outline.set_visible(False)
    cb_a.set_label("Spearman ρ", fontsize=FS, labelpad=2)

    # ---- b
    tab = panel_b(de)
    tab.to_csv(tab_dir / "panel_b_counts.csv")
    rows = []
    for grp, types in B_GROUPS.items():
        rows.append(("hdr", grp))
        present = [t for t in types if t in tab.index]
        rows += [("row", t) for t in sorted(present, key=lambda t: -tab.at[t, "total"])]
    b_lab = 3.12
    bx0 = b_lab + 1.02
    b_h = row1_h - 0.25 - 0.42
    bax = ax_in(bx0, 0.25, W - bx0 - 0.16, b_h)
    y = len(rows)
    ytick, ylab = [], []
    for kind, t in rows:
        y -= 1
        if kind == "hdr":
            bax.text(
                (b_lab + 0.3 - bx0) / (W - bx0 - 0.16),
                y,
                t,
                fontsize=FS,
                color="0.45",
                style="italic",
                va="center",
                ha="left",
                transform=bax.get_yaxis_transform(),
            )
            continue
        q, w = tab.at[t, "queen_per1000"], tab.at[t, "worker_per1000"]
        bax.barh(y, q, color=QC, height=0.72)
        bax.barh(y, -w, color=WC, height=0.72)
        bax.text(
            q + 1.0,
            y,
            f"{tab.at[t, 'total']:,}",
            fontsize=FSS,
            color="0.3",
            va="center",
        )
        ytick.append(y)
        ylab.append(NAME[t] + ("†" if t == "Hemocytes" else ""))
    bax.axvline(0, color="0.3", lw=0.5)
    bax.set_ylim(-0.6, len(rows) - 0.4)
    bax.set_xlim(-36, 72)
    bax.xaxis.set_major_formatter(
        mpl.ticker.FuncFormatter(lambda v, _: f"{abs(v):.0f}")
    )
    bax.set_yticks(ytick)
    bax.set_yticklabels(ylab, fontsize=FS)
    bax.tick_params(axis="y", length=0)
    bax.tick_params(axis="x", labelsize=FSS, length=2, width=0.5)
    bax.set_xlabel("DEGs per 1,000 tested genes", fontsize=FS, labelpad=2)
    tr = mpl.transforms.blended_transform_factory(bax.transData, bax.transAxes)
    bax.text(
        -18,
        1.01,
        "worker-up",
        color=WC,
        fontsize=FS,
        fontweight="bold",
        ha="center",
        va="bottom",
        transform=tr,
    )
    bax.text(
        36,
        1.01,
        "queen-up",
        color=QC,
        fontsize=FS,
        fontweight="bold",
        ha="center",
        va="bottom",
        transform=tr,
    )
    title(b_lab, ROW1_HDR, "B", "Caste DEGs per cell type")

    # ---- c
    keep = keep_set(lk)
    c_top = 0.04 + row1_h + 0.04
    all_ranks = {
        (b, d_): recurrence(de, keep, t, d_)
        for b, t in C_BLOCKS.items()
        for d_ in ("queen", "worker")
    }
    crow = []
    for (b, d_), (rank, *_rest) in all_ranks.items():
        for g, r in rank.iterrows():
            crow.append(
                dict(
                    block=b,
                    direction=d_,
                    gene=g,
                    n_sig=int(r.n),
                    n_tested=int(r.n_tested),
                    n_types=len(C_BLOCKS[b]),
                    mean_abs_wald=r.m,
                    label=gene_label(lk, g).replace("\n", " "),
                )
            )
    pd.DataFrame(crow).to_csv(tab_dir / "panel_c_genes.csv", index=False)
    cmap_c = LinearSegmentedColormap.from_list("qw", [WC, "#f7f7f7", QC])
    cmap_c.set_bad("0.85")
    LAB_W, NN = 1.12, 0.36
    x = 0.04
    im = None
    title(LX, c_top, "C", "Genes recurring across cell types")
    for bname, types in C_BLOCKS.items():
        axx, axw = x + LAB_W, len(types) * C_CELL_W
        for ri, d_ in enumerate(("queen", "worker")):
            rank, lfc, pad, tst = all_ranks[(bname, d_)]
            yy = c_top + 0.32 + ri * (0.16 + TOP * C_ROW + 0.08) + 0.16
            ax = ax_in(axx, yy, axw, TOP * C_ROW)
            shown = np.ma.masked_where(
                ~tst.to_numpy() | lfc.isna().to_numpy(), lfc.to_numpy()
            )
            im = ax.imshow(shown, cmap=cmap_c, vmin=-3, vmax=3, aspect="auto")
            ax.set_xticks(np.arange(-0.5, len(types), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(rank), 1), minor=True)
            ax.grid(which="minor", color="white", lw=0.8)
            ax.tick_params(which="both", length=0)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            ax.set_yticks(range(len(rank)))
            ax.set_yticklabels(
                [gene_label(lk, g) for g in rank.index],
                fontsize=FSS,
                linespacing=0.85,
            )
            ax.set_xticks(range(len(types)))
            if ri == 1:
                ax.set_xticklabels(
                    [NAME[t] for t in types],
                    rotation=TICK_ROT,
                    ha="right",
                    rotation_mode="anchor",
                    fontsize=FS,
                )
            else:
                ax.set_xticklabels([])
            for i, g in enumerate(rank.index):
                for j, t in enumerate(types):
                    if pad.loc[g, t] < FDR and tst.loc[g, t]:
                        ax.text(
                            j, i + 0.12, "*", ha="center", va="center", fontsize=FS + 1
                        )
                ax.text(
                    1.03,
                    i,
                    f"{int(rank.n.iloc[i])}/{len(types)}",
                    va="center",
                    fontsize=FSS,
                    color="0.3",
                    transform=ax.get_yaxis_transform(),
                )
            ftext(
                axx,
                yy - 0.02,
                f"{bname}: {d_}-biased",
                fontsize=FS,
                color=QC if d_ == "queen" else WC,
                va="bottom",
                ha="left",
            )
        x = axx + axw + NN
    kx = 5.15
    cb_ax = ax_in(kx, c_top + 0.05, 1.0, 0.07)
    cb = fig.colorbar(im, cax=cb_ax, orientation="horizontal", ticks=[-3, 0, 3])
    cb.ax.tick_params(labelsize=FSS, length=2, width=0.5)
    cb.ax.set_xticklabels(["−3", "0", "3"])
    cb.outline.set_visible(False)
    ftext(
        kx - 0.06,
        c_top + 0.085,
        "log2FC (queen / worker)",
        fontsize=FS,
        ha="right",
        va="center",
    )
    ng = ax_in(kx + 1.12, c_top + 0.04, 0.09, 0.09)
    ng.set_facecolor("0.85")
    ng.set_xticks([])
    ng.set_yticks([])
    for s_ in ng.spines.values():
        s_.set_visible(False)
    ftext(kx + 1.24, c_top + 0.085, "not tested", fontsize=FS, ha="left", va="center")

    # ---- d
    use, drho, labels, jc = panel_d_data(de, cnmf_dir)
    drho.to_csv(tab_dir / "panel_d_programs.csv", index=False)
    d_top = c_top + c_h + 0.04
    title(LX, d_top + 0.02, "D", "Joint cNMF activity programs")
    hcolorbar(5.15, d_top + 0.0, 0.75, D_LIM, "RdBu_r", "ρ (program weights, Wald)")
    ftext(
        6.20,
        d_top + 0.035,
        "queen",
        color=QC,
        fontsize=FS,
        fontweight="bold",
        ha="left",
        va="center",
    )
    ftext(
        6.58,
        d_top + 0.035,
        "worker",
        color=WC,
        fontsize=FS,
        fontweight="bold",
        ha="left",
        va="center",
    )
    left, gap, right = 0.42, 0.14, 0.06
    ntypes = sum(len(v) for v in D_GROUPS.values())
    tw = (W - left - right - gap * (len(D_GROUPS) - 1)) / ntypes
    xs, xpos = {}, left
    for _g, types in D_GROUPS.items():
        for t in types:
            xs[t] = xpos
            xpos += tw
        xpos += gap
    y0 = d_top + 0.25
    for g, types in D_GROUPS.items():
        xa, xb = xs[types[0]], xs[types[-1]] + tw
        ftext((xa + xb) / 2, y0 + 0.02, g, ha="center", va="center", fontsize=FS + 0.5)
        fig.add_artist(
            Line2D(
                [xa / W + 0.003, xb / W - 0.003],
                [1 - (y0 + 0.11) / H] * 2,
                color="0.3",
                lw=0.6,
            )
        )
    y0 += 0.14
    cmap_rho = plt.get_cmap("RdBu_r").copy()
    cmap_rho.set_bad("0.85")
    for k, p in enumerate(PROGS):
        lab = labels.loc[p]
        if lab.manual:
            text = lab.label
        else:
            f = "<0.001" if lab.fdr < 0.001 else f"{lab.fdr:.3f}"
            text = f"{lab.label} ({lab.go_id}, FDR {f})"
        ftext(left, y0 + 0.01, text, ha="left", va="top", fontsize=FS)
        ys, yp = y0 + D_TITLE, y0 + D_TITLE + D_STRIP + 0.02
        pa_list = []
        for types in D_GROUPS.values():
            xa = xs[types[0]]
            wg = len(types) * tw
            sa = ax_in(xa, ys, wg, D_STRIP)
            rr = drho[(drho.program == p)].set_index("cell_type").reindex(types)
            sa.imshow(
                np.ma.masked_invalid(rr.rho.to_numpy()[None, :]),
                cmap=cmap_rho,
                vmin=-D_LIM,
                vmax=D_LIM,
                aspect="auto",
            )
            sa.set_xticks([])
            sa.set_yticks([])
            for s_ in sa.spines.values():
                s_.set_visible(False)
            pa = ax_in(xa, yp, wg, D_PLOT)
            pa_list.append(pa)
            top = 0.0
            for j, t in enumerate(types):
                if rr.replicated[t]:
                    pa.axvspan(j - 0.47, j + 0.47, color="0.9", lw=0, zorder=0)
                sub = use[use.cell_type == t]
                for c, colr, dx in (("Queen", QC, -0.2), ("Worker", WC, 0.2)):
                    m = sub[sub.caste == c]
                    v = m[f"p{p}"].to_numpy()
                    pa.boxplot(
                        [v],
                        positions=[j + dx],
                        widths=0.3,
                        whis=(5, 95),
                        patch_artist=True,
                        showfliers=False,
                        manage_ticks=False,
                        boxprops=dict(fc=colr, alpha=0.55, lw=0.4, ec=colr),
                        medianprops=dict(color="black", lw=0.7),
                        whiskerprops=dict(color=colr, lw=0.5),
                        capprops=dict(color=colr, lw=0.5),
                    )
                    tm = m.groupby("trial")[f"p{p}"].mean().sort_values().to_numpy()
                    pa.scatter(
                        j + dx + np.array([-0.05, 0.0, 0.05])[: len(tm)],
                        tm,
                        s=3,
                        color="black",
                        lw=0,
                        zorder=5,
                    )
                    top = max(top, np.percentile(v, 95), tm.max())
            pa.set_xlim(-0.5, len(types) - 0.5)
            pa._top = top
            pa.set_xticks(range(len(types)))
            pa.tick_params(axis="x", length=0)
            if k == len(PROGS) - 1:
                pa.set_xticklabels(
                    [NAME[t] for t in types],
                    rotation=TICK_ROT,
                    ha="right",
                    rotation_mode="anchor",
                    fontsize=FS,
                )
            else:
                pa.set_xticklabels([])
        # one y scale per program row, shown on the first group only
        ymax = max(a._top for a in pa_list) * 1.08
        for i_, pa in enumerate(pa_list):
            pa.set_ylim(0, ymax)
            pa.yaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
            pa.tick_params(axis="y", labelsize=FSS, length=2, width=0.5, pad=1)
            if i_ > 0:
                pa.tick_params(axis="y", labelleft=False)
        y0 += D_TITLE + D_STRIP + 0.02 + D_PLOT + D_GAP
    ya = d_top + 0.36 + D_TITLE
    yb = y0 - D_GAP
    ftext(
        0.1,
        (ya + yb) / 2,
        "Program usage (%)",
        rotation=90,
        ha="center",
        va="center",
        fontsize=FS,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{out_name}.png", dpi=300)
    # no CreationDate, so a re-render of the same artwork is byte-identical
    fig.savefig(out_dir / f"{out_name}.pdf", metadata={"CreationDate": None})
    print(f"figure {W:.2f} x {H:.2f} in -> {out_dir / out_name}.png")
    return tab, all_ranks, drho, labels


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-name", default="fig_caste_de")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=OUT,
        help="render elsewhere (e.g. a scratch reproducibility check)",
    )
    ap.add_argument(
        "--tables-dir",
        type=Path,
        default=None,
        help="where the plotted tables go (default: the pipeline's fig3_tables)",
    )
    ap.add_argument(
        "--saved-inputs",
        action="store_true",
        help="panel d from the analysis's SAVED outputs "
        "(bimp_st.paths.CASTE_DE_SAVED_INPUTS), "
        "for comparison; default is the repo's run (caste_programs)",
    )
    args = ap.parse_args(argv)
    cnmf_dir = SAVED if args.saved_inputs else R9 / "fig3_inputs"  # panel d inputs
    tab_dir = args.tables_dir or (
        CASTE_DE_SAVED_DIR / "fig3_tables" if args.saved_inputs else R9 / "fig3_tables"
    )
    de = load_de()
    lk = load_lookup()
    tab, all_ranks, drho, labels = draw(
        de, lk, cnmf_dir, tab_dir, args.out_dir, args.out_name
    )

    # ── printed summary of the plotted values ───────────────────────────────────
    print("\npanel b (DEGs total; tested):")
    print(tab[["tested", "queen_up", "worker_up", "total"]].to_string())
    print("\npanel c:")
    for (b, d_), (rank, *_r) in all_ranks.items():
        print(f"  {b}, {d_}-biased:")
        for g, r in rank.iterrows():
            print(
                f"    {int(r.n)}/{int(r.n_tested)} (of {len(C_BLOCKS[b])})  "
                f"|W| {r.m:.2f}  {g[-6:]}  "
                f"{gene_label(lk, g).replace(chr(10), ' ')}"
            )
    print("\npanel d replicated (shaded) cells:")
    print(
        drho[drho.replicated][
            ["program", "cell_type", "rho", "mean_effect_sd", "usage_pct"]
        ]
    )
    print("\npanel d labels:")
    print(labels.to_string())


if __name__ == "__main__":
    with explain_missing_inputs():
        main()
