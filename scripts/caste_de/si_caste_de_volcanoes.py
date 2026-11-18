#!/usr/bin/env python
"""Supplementary figure ``fig:caste_de_volcanoes``: caste DE by class and subtype.

Queen vs worker, n=3 vs n=3; results/figures/si/figS_caste_de_volcanoes.{png,pdf}.

Layout is the approved scaffold, driven by real tables:
  A  whole-brain volcano              B  glia: all glia + four subtypes
  C  non-neural: hemocytes, PRs       D  Kenyon cells: all + five subtypes
  E  projection and other neurons     F  top-2 caste DEGs per subtype, dot plot
                                         over 14 cell types (queen | worker)

Both DE tables are the trial-blocked fit (``~ trial + caste``,
``caste_de_blocked_all.py``).

Inputs (repo-relative, or under --root):
  bimp_st.paths.CASTE_DE_COARSE   coarse. cell_type:
      all, Glia, Kenyon_cells, Projection_neurons, Other_neurons, Hemocytes,
      Photoreceptors. Columns: baseMean, log2FoldChange, lfcSE, stat, pvalue,
      padj, tested, cell_type, gene, gene_name.
  bimp_st.paths.CASTE_DE_FINE  fine. Other glia (two trial pairs)
      is fitted; it is not drawn but counts as a glial sibling in panel F.
      cell_type: Astrocyte-like glia, Cortex glia, Perineurial glia,
      Subperineurial glia, Large/Middle/Small-type Kenyon cells, Class II Kenyon
      cells, Other Kenyon cells, Anterodorsal/Lateral projection neurons,
      Other neurons, Hemocytes, Photoreceptors. gene_name is quoted CSV (it can
      contain commas) and equals the BIMP id when unannotated.
  results/round5/glia_labels/v43_dmel_rbh.csv.gz   fly ortholog symbols, used
      for display names (dmel_symbol, only where call is a reciprocal best hit).
  data/sn/BIMP_atlas_qw_CentralBrain.h5ad          only for --stage expr.
      cell_type for the subtype rows and Other neurons; major_cell_type for
      Hemocytes and Photoreceptors.

Outputs (new files only):
  results/caste_de/fig_s_caste_de_volcanoes/expression_frac.csv  (--stage expr)
  results/figures/si/figS_caste_de_volcanoes.{png,pdf}

Stages:
  PYTHONPATH=src:scripts .venv/bin/python scripts/caste_de/si_caste_de_volcanoes.py \\
      --stage expr          # needs the atlas h5ad
  PYTHONPATH=src:scripts .venv/bin/python scripts/caste_de/si_caste_de_volcanoes.py \\
      --stage plot          # default; reads CSVs only (matplotlib, numpy, pandas)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from bimp_st.caste_de import focal_gene_ids  # noqa: E402
from bimp_st.genes import GENE_SYMBOLS  # noqa: E402
from bimp_st.paths import (
    CASTE_DE_COARSE,
    CASTE_DE_DIR,  # noqa: E402
    CASTE_DE_FINE,
    REPO,
    explain_missing_inputs,  # noqa: E402
)
from bimp_st.site import get as site_path  # noqa: E402

COARSE_REL = CASTE_DE_COARSE.relative_to(REPO)
FINE_REL = CASTE_DE_FINE.relative_to(REPO)
ORTH_REL = Path("results/round5/glia_labels/v43_dmel_rbh.csv.gz")
OUT_REL = CASTE_DE_DIR.relative_to(REPO) / "fig_s_caste_de_volcanoes"
EXPR_NAME = "expression_frac.csv"
ATLAS_REL = Path("data/sn/BIMP_atlas_qw_CentralBrain.h5ad")

SAMPLES = [f"BIMP_{c}_t{i}" for c in ("Queen", "Worker") for i in (1, 2, 3)]
CASTE_OF = {s: ("Queen" if "Queen" in s else "Worker") for s in SAMPLES}
CASTES = ("Queen", "Worker")
FDR = 0.05
XCAP = 4.0
YCAP = 15.0
YLIM = (-1.2, 18.5)  # headroom: counts sit above points capped at 15
SHORT_MAX = 12  # characters of a description-derived name, before the ellipsis

# Focal genes (the `fig:caste_de` markers): id -> (display name, marker).
_FOCAL = focal_gene_ids()
FOCAL = {
    _FOCAL["Kr-h1"]: ("Kr-h1", "^"),
    _FOCAL["sNPF"]: ("sNPF", "s"),
    _FOCAL["Tk"]: ("Tk", "D"),
}

Q, W, G = "#DAA520", "#4A90D9", "#D3D3D3"  # queen-up, worker-up, n.s.

C = {
    "Whole brain": "#222222", "Glia": "#2A8A8A", "Glia-astro": "#5BA05B",
    "Glia-cortex": "#8FBF86", "Glia-peri": "#2A8A8A", "Glia-sub": "#1F5F55",
    "Kenyon cells": "#E07B10", "KC-l": "#F0922E", "KC-m": "#9C6B4E",
    "KC-s": "#B87AA8", "KC-II": "#E2B624", "KC-o": "#8C8C8C",
    "Proj. neurons": "#2A4A9A", "PN-ad": "#2A4A9A", "PN-lat": "#6F8FD8",
    "N-other": "#3FA9D9", "Hem": "#6B4A3A", "PR": "#8B1E3A",
}  # fmt: skip
LABE = {
    "Glia-astro": "Astrocyte-like glia", "Glia-cortex": "Cortex glia",
    "Glia-peri": "Perineurial glia", "Glia-sub": "Subperineurial glia",
    "KC-l": "Large-type KCs", "KC-m": "Middle-type KCs", "KC-s": "Small-type KCs",
    "KC-II": "Class II KCs", "KC-o": "Other KCs", "PN-ad": "Anterodorsal PNs",
    "PN-lat": "Lateral PNs", "N-other": "Other neurons", "Hem": "Hemocytes",
    "PR": "Photoreceptors",
}  # fmt: skip
LAB = {
    "Glia-astro": "Astrocyte-like", "Glia-cortex": "Cortex",
    "Glia-peri": "Perineurial", "Glia-sub": "Subperineurial",
    "KC-l": "Large-type", "KC-m": "Middle-type", "KC-s": "Small-type",
    "KC-II": "Class II", "KC-o": "Other", "Proj. neurons": "All PNs",
    "PN-ad": "Anterodorsal", "PN-lat": "Lateral", "N-other": "Other neurons",
    "Hem": "Hemocytes", "PR": "Photoreceptors",
}  # fmt: skip

# Where each display key's DE rows come from: (table, cell_type value).
SOURCE = {
    "Whole brain": ("coarse", "all"),
    "Glia": ("coarse", "Glia"),
    "Kenyon cells": ("coarse", "Kenyon_cells"),
    "Proj. neurons": ("coarse", "Projection_neurons"),
    "N-other": ("coarse", "Other_neurons"),
    "Hem": ("coarse", "Hemocytes"),
    "PR": ("coarse", "Photoreceptors"),
    "Glia-astro": ("fine", "Astrocyte-like glia"),
    "Glia-cortex": ("fine", "Cortex glia"),
    "Glia-peri": ("fine", "Perineurial glia"),
    "Glia-sub": ("fine", "Subperineurial glia"),
    "Glia-other": ("fine", "Other glia"),  # panel F siblings only, not drawn
    "KC-l": ("fine", "Large-type Kenyon cells"),
    "KC-m": ("fine", "Middle-type Kenyon cells"),
    "KC-s": ("fine", "Small-type Kenyon cells"),
    "KC-II": ("fine", "Class II Kenyon cells"),
    "KC-o": ("fine", "Other Kenyon cells"),
    "PN-ad": ("fine", "Anterodorsal projection neurons"),
    "PN-lat": ("fine", "Lateral projection neurons"),
}
# Atlas (column, value) holding each expression row's nuclei. Subtypes and Other
# neurons are cell_type values; Hemocytes and Photoreceptors are major_cell_type.
ATLAS_SOURCE = {
    "Hem": ("major_cell_type", "Hemocytes"),
    "PR": ("major_cell_type", "Photoreceptors"),
    "N-other": ("cell_type", "Other neurons"),
}

SUBTYPES = [
    "Glia-astro", "Glia-cortex", "Glia-peri", "Glia-sub", "KC-l", "KC-m",
    "KC-s", "KC-II", "KC-o", "PN-ad", "PN-lat",
]  # fmt: skip
ROWS = [
    "Glia-astro", "Glia-cortex", "Glia-peri", "Glia-sub", "KC-l", "KC-m",
    "KC-s", "KC-II", "KC-o", "PN-ad", "PN-lat", "N-other", "Hem", "PR",
]  # fmt: skip
HEAD = {
    "Glia-astro": "Astrocyte-\nlike", "Glia-cortex": "Cortex",
    "Glia-peri": "Perineurial", "Glia-sub": "Subperi-\nneurial",
    "KC-l": "Large-\ntype", "KC-m": "Middle-\ntype", "KC-s": "Small-\ntype",
    "KC-II": "Class II", "KC-o": "Other\nKCs", "PN-ad": "Antero-\ndorsal PNs",
    "PN-lat": "Lateral\nPNs",
}  # fmt: skip


# ---------------------------------------------------------------- data helpers
def load_tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    coarse = pd.read_csv(root / COARSE_REL)
    fine = pd.read_csv(root / FINE_REL)
    return coarse, fine


def load_orthologs(root: Path) -> dict[str, dict[str, str | None]]:
    """gene id -> {"symbol": fly symbol or None, "description": text or None}.

    The symbol is used only where the call is a reciprocal best hit; a leading
    ``Dmel\\`` prefix is stripped and CG-numbered placeholders are skipped."""
    path = root / ORTH_REL
    if not path.exists():
        print(f"WARNING: ortholog table missing ({path}); names fall back")
        return {}
    df = pd.read_csv(path)
    out: dict[str, dict[str, str | None]] = {}
    for _, r in df.iterrows():
        sym = None
        if str(r.get("call", "")).startswith("reciprocal best hit"):
            cand = str(r.get("dmel_symbol", "")).strip()
            cand = re.sub(r"^Dmel[\\`]", "", cand).strip()
            if cand and cand.lower() != "nan" and not PLACEHOLDER.match(cand):
                sym = cand
        hit = None  # best fly hit that is not a reciprocal ortholog call
        if sym is None:
            for col in ("dmel_symbol", "relaxed_dmel_symbol"):
                c = re.sub(r"^Dmel[\\`]", "", str(r.get(col, "")).strip()).strip()
                if c and c.lower() != "nan" and not PLACEHOLDER.match(c):
                    hit = c
                    break
        desc = str(r.get("description", "")).strip()
        out[str(r["gene"])] = {
            "symbol": sym,
            "hit": hit,
            "description": desc if desc and desc.lower() != "nan" else None,
        }
    return out


def table_for(key: str, coarse: pd.DataFrame, fine: pd.DataFrame) -> pd.DataFrame:
    kind, ct = SOURCE[key]
    t = coarse if kind == "coarse" else fine
    return t[t.cell_type == ct]


def named(d: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a real annotation (drop empty names and BIMP-id fallbacks)."""
    name = d.gene_name.fillna("").astype(str).str.strip()
    ok = (name != "") & (name.str.lower() != d.gene.astype(str).str.lower())
    ok &= ~name.str.upper().str.startswith("BIMP_V4.3")
    return d[ok]


# Fly "symbols" that are database ids, not names (CG/CT annotation ids, FBgn).
PLACEHOLDER = re.compile(r"^(C[GT]|FBgn)\d+")

# Genes with no fly ortholog or best hit: the standard abbreviation of their NCBI
# product name (HGNC-style), so labels stay short. DPEP1a/b are two paralogs
# that NCBI both names "dipeptidase 1".
DESC_ABBREV = {
    "bimp_v4.3_001295": "CPD",  # carboxypeptidase D
    "bimp_v4.3_001615": "DPEP1a",  # dipeptidase 1
    "bimp_v4.3_006293": "DPEP1b",  # dipeptidase 1
    "bimp_v4.3_001860": "P3H1",  # prolyl 3-hydroxylase 1
    "bimp_v4.3_003840": "SLC26",  # sodium-independent sulfate anion transporter
    "bimp_v4.3_003928": "Or63a",  # odorant receptor 63a
    "bimp_v4.3_006317": "QRFPR",  # QRFP-like peptide receptor
    "bimp_v4.3_007269": "FBXL3",  # F-box/LRR-repeat protein 3
    "bimp_v4.3_008660": "CFAP206",  # cilia- and flagella-associated protein 206
    "bimp_v4.3_008764": "IMPI",  # inducible metalloproteinase inhibitor protein
    "bimp_v4.3_009985": "CFAP47",  # cilia and flagella-associated protein 47
    "bimp_v4.3_009568": "unchar. 9568",
    "bimp_v4.3_007096": "CAPON",  # carboxyl-terminal PDZ ligand of nNOS
    "bimp_v4.3_004392": "CRISP2",  # cysteine-rich secretory protein 2
    "bimp_v4.3_008229": "Myo81F",  # unconventional myosin 81F
    "bimp_v4.3_004327": "SMPD4",  # sphingomyelin phosphodiesterase 4
    "bimp_v4.3_001938": "TIA1",  # cytotoxic granule associated RNA binding protein
    "bimp_v4.3_010031": "Myr1",  # myrosinase 1  # uncharacterized protein DDB_G0272420
}


def _cut_words(s: str, width: int = SHORT_MAX) -> str:
    """Cut at the last word boundary within width; add an ellipsis if cut."""
    if len(s) <= width:
        return s
    words, out = s.split(), ""
    for w in words:
        trial = (out + " " + w).strip()
        if len(trial) > width:
            break
        out = trial
    if not out:
        out = s[:width]
    return out.rstrip(" ,;-") + "…"


def short_symbol(gene_id: str, gene_name: str, orth: dict) -> str:
    """Human-readable display name for a gene.

    Priority: hand-curated GENE_SYMBOLS, then the fly ortholog symbol (reciprocal
    best hits only), then the description cut to <= SHORT_MAX characters."""
    key = str(gene_id)
    sym = GENE_SYMBOLS.get(key.lower())
    if sym:
        return sym
    info = orth.get(key, {})
    if info.get("symbol"):
        # Greek letters as in the fly gene nomenclature (nAChRalpha4 -> nAChRα4).
        return re.sub(r"(?<=[A-Za-z])alpha(?=\d)", "α", info["symbol"])
    if key.lower() in DESC_ABBREV:
        return DESC_ABBREV[key.lower()]
    if info.get("hit"):
        return info["hit"] + "†"  # best fly hit, not a reciprocal ortholog
    s = re.sub(r"\s*isoform\s+[\w-]+$", "", str(gene_name))
    s = re.sub(r"\s*[_,]\s*$", "", s).strip()
    if not s or s.upper().startswith("BIMP_V4.3"):
        s = info.get("description") or ""
        s = re.sub(r"\s*isoform\s+[\w-]+$", "", s).strip()
    return _cut_words(s) if s else "unnamed"


# Each subtype's parent class (pooled coarse test) and its sibling subtypes.
PARENT = {
    **dict.fromkeys(
        ["Glia-astro", "Glia-cortex", "Glia-peri", "Glia-sub", "Glia-other"], "Glia"
    ),
    **dict.fromkeys(["KC-l", "KC-m", "KC-s", "KC-II", "KC-o"], "Kenyon cells"),
    **dict.fromkeys(["PN-ad", "PN-lat"], "Proj. neurons"),
}


def _sig_genes(tbl: pd.DataFrame) -> set:
    return set(tbl.loc[tbl.padj < FDR, "gene"])


def select_panel_f(
    coarse: pd.DataFrame, fine: pd.DataFrame
) -> list[tuple[str, str, str]]:
    """Subtype-specific caste DEGs: for each subtype, the two lowest-padj annotated
    genes that are significant in that subtype but in neither its pooled class
    nor any sibling subtype (so each column shows what the subtype adds beyond
    its class). Genes already chosen for an earlier subtype are skipped.
    Returns (subtype, gene, gene_name)."""
    picks, taken = [], set()
    for key in SUBTYPES:
        parent = PARENT[key]
        sibs = [k for k, v in PARENT.items() if v == parent and k != key]
        exclude = _sig_genes(table_for(parent, coarse, fine))
        for k in sibs:
            exclude |= _sig_genes(table_for(k, coarse, fine))
        d = table_for(key, coarse, fine).dropna(subset=["padj"])
        d = named(d[(d.padj > 0) & (d.padj < FDR)]).sort_values("padj")
        d = d[~d.gene.isin(exclude)]
        print(f"{key}: {len(d)} subtype-specific DEGs")
        got = 0
        for _, r in d.iterrows():
            if r.gene in taken:
                continue
            taken.add(r.gene)
            picks.append((key, r.gene, str(r.gene_name)))
            got += 1
            if got == 2:
                break
        if got < 2:
            print(f"WARNING: {key} has only {got} subtype-specific DEGs")
    return picks


# ----------------------------------------------------------- stage: expression
def find_atlas(root: Path) -> Path:
    for cand in (
        root / ATLAS_REL,
        site_path("data_repo") / ATLAS_REL,
    ):
        if cand.exists():
            return cand
    raise SystemExit(f"ERROR: atlas not found at {root / ATLAS_REL}")


def atlas_selector(row: str) -> tuple[str, str]:
    if row in ATLAS_SOURCE:
        return ATLAS_SOURCE[row]
    return ("cell_type", SOURCE[row][1])


def stage_expr(root: Path, picks: list[tuple[str, str, str]]) -> None:
    """Fraction of nuclei with count > 0 for each picked gene x row x caste.

    Reads the atlas in backed mode, in row chunks, keeping only the picked gene
    columns, so memory stays small."""
    import anndata as ad

    atlas = find_atlas(root)
    a = ad.read_h5ad(atlas, backed="r")
    obs = a.obs[["sample", "cell_type", "major_cell_type"]].astype(str)
    gpos = {g: i for i, g in enumerate(a.var_names)}
    genes = [g for _, g, _ in picks]
    missing = [g for g in genes if g not in gpos]
    if missing:
        raise SystemExit(f"ERROR: genes not in atlas: {missing}")
    idx = [gpos[g] for g in genes]
    M = a.layers["counts"] if "counts" in a.layers else a.X

    samp = obs["sample"].to_numpy()
    caste = np.array([CASTE_OF.get(s, "") for s in samp])
    sel = {}
    for row in ROWS:
        col, val = atlas_selector(row)
        sel[row] = obs[col].to_numpy() == val
        print(f"{row:12s} {col}=={val!r}: {int(sel[row].sum())} nuclei")
        if not sel[row].any():
            raise SystemExit(f"ERROR: no atlas nuclei for row {row} ({col}=={val!r})")

    pos = np.zeros((len(ROWS), len(CASTES), len(idx)))
    n = np.zeros((len(ROWS), len(CASTES)))
    step = 20000
    for start in range(0, a.n_obs, step):
        stop = min(start + step, a.n_obs)
        blk = M[start:stop][:, idx]
        X = (blk.toarray() if hasattr(blk, "toarray") else np.asarray(blk)) > 0
        for ri, row in enumerate(ROWS):
            for ci, cst in enumerate(CASTES):
                m = sel[row][start:stop] & (caste[start:stop] == cst)
                n[ri, ci] += m.sum()
                if m.any():
                    pos[ri, ci] += X[m].sum(axis=0)
    a.file.close()

    out = []
    for gi, (sub, gene, gname) in enumerate(picks):
        for ri, row in enumerate(ROWS):
            for ci, cst in enumerate(CASTES):
                frac = pos[ri, ci, gi] / n[ri, ci] if n[ri, ci] else np.nan
                out.append(
                    dict(
                        subtype=sub, gene=gene, gene_name=gname, row=row,
                        caste=cst, n_nuclei=int(n[ri, ci]), frac_expressing=frac,
                    )
                )  # fmt: skip
    d = out_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(d / EXPR_NAME, index=False)
    print("WROTE", d / EXPR_NAME, len(out), "rows")


def out_dir(root: Path) -> Path:
    return root / OUT_REL


# ------------------------------------------------------------------- plotting
def clean(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _overlaps(a, b) -> bool:
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


def _inside(a, box) -> bool:
    return a.x0 >= box.x0 and a.x1 <= box.x1 and a.y0 >= box.y0 and a.y1 <= box.y1


def place_labels(fig, ax, items, counts) -> None:
    """Place gene labels inside a big volcano, clear of the counts and each other.

    Each label starts at y=13 and steps down by 1.8 until its window extent lies
    inside the axes and touches no corner count or earlier label. Items are
    dicts with label, right, x, y (the point). Labels that find no clear slot are
    dropped with a warning rather than drawn over the data."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axbb = ax.get_window_extent(rend)
    fixed = [t.get_window_extent(rend) for t in counts]
    placed = []
    for it in items:
        right = it["right"]
        x_t = 3.3 if right else -3.3
        slot = None
        for y in np.arange(13.0, 0.0, -1.8):
            t = ax.text(
                x_t, y, it["label"], fontsize=5.8, style="italic", zorder=8,
                path_effects=[patheffects.withStroke(linewidth=2, foreground="w")],
                ha="right" if right else "left", va="center",
            )  # fmt: skip
            bb = t.get_window_extent(rend)
            if _inside(bb, axbb) and not any(_overlaps(bb, o) for o in fixed + placed):
                slot = (t, bb, y)
                break
            t.remove()
        if slot is None:
            print(f"WARNING: no clear slot for label {it['label']!r}; dropped")
            continue
        t, bb, y = slot
        placed.append(bb)
        # Leader from the label edge nearest the point, so it never crosses text.
        (x0, y0), (x1, y1) = ax.transData.inverted().transform(
            [[bb.x0, bb.y0], [bb.x1, bb.y1]]
        )
        px, py = it["x"], it["y"]
        if x0 <= px <= x1:
            sx, sy = (px, y1 + 0.2) if py > y1 else (px, y0 - 0.2)
        else:
            sx, sy = (x1 + 0.05 if px > x1 else x0 - 0.05), (y0 + y1) / 2
        ax.plot([sx, px], [sy, py], color="#666", lw=0.35, zorder=7,
                clip_on=False)  # fmt: skip


def volcano(ax, key, tbl, orth, big=False, title=None, xt=False, yt=False):
    """Draw one volcano. Returns pending label items and count texts for big
    panels (labels are placed after all axes exist), else None."""
    clean(ax)
    d = tbl.dropna(subset=["padj"])
    d = d[d.padj > 0]
    lf = d.log2FoldChange.to_numpy(float)
    pa = d.padj.to_numpy(float)
    nl = np.minimum(-np.log10(pa), YCAP)
    sig = pa < FDR
    col = np.where(~sig, G, np.where(lf > 0, Q, W))
    ax.scatter(np.clip(lf, -XCAP, XCAP), nl, s=2 if big else 0.9, c=col, lw=0,
               rasterized=True)  # fmt: skip
    ax.axhline(-np.log10(FDR), ls="--", lw=0.4, color="#888")
    ax.axvline(0, lw=0.4, color="#c8c8c8")
    ax.set_xlim(-XCAP - 0.2, XCAP + 0.2)
    ax.set_ylim(*YLIM)
    ax.set_xticks([-4, -2, 0, 2, 4] if big else [-4, 0, 4])
    ax.set_yticks([0, 5, 10, 15])
    if big:
        ax.set_xlabel("log$_2$ fold change (queen / worker)", labelpad=1)
        ax.set_ylabel("−log$_{10}$ $\\mathit{P}_{adj}$", labelpad=1)
    else:
        # Tick labels only on the outer small multiples (bottom row / left column);
        # every small panel shares the axes of the big ones.
        ax.tick_params(labelleft=yt, labelbottom=xt, length=1.5, labelsize=5.2,
                       pad=1)  # fmt: skip
    fs = 7 if big else 6
    nw = int((sig & (lf < 0)).sum())
    nq = int((sig & (lf > 0)).sum())
    counts = [
        ax.text(
            0.03,
            0.97,
            str(nw),
            color=W,
            transform=ax.transAxes,
            va="top",
            fontweight="bold",
            fontsize=fs,
        ),  # fmt: skip
        ax.text(
            0.97,
            0.97,
            str(nq),
            color=Q,
            transform=ax.transAxes,
            va="top",
            ha="right",
            fontweight="bold",
            fontsize=fs,
        ),  # fmt: skip
    ]

    full = tbl.drop_duplicates("gene").set_index("gene", drop=False)
    for gid, (_name, mk) in FOCAL.items():
        if gid not in full.index:
            continue
        r = full.loc[gid]
        if pd.isna(r.log2FoldChange):
            continue
        y = 0.0 if pd.isna(r.padj) else min(-np.log10(max(r.padj, 1e-300)), YCAP)
        ax.scatter(float(np.clip(r.log2FoldChange, -XCAP, XCAP)), y, marker=mk,
                   s=26 if big else 12, facecolor="white", edgecolor="k", lw=0.6,
                   zorder=10, clip_on=False)  # fmt: skip

    # Dark title text for legibility; the colour code lives in the bar above it.
    ax.set_title(title or LAB.get(key, key), fontsize=7.5 if big else 6.3,
                 fontweight="bold", loc="left", pad=2.5, color="#1a1a1a")  # fmt: skip
    ax.plot([0, 1], [1, 1], transform=ax.transAxes, color=C[key],
            lw=2.4 if big else 1.8, clip_on=False, solid_capstyle="butt")  # fmt: skip
    if not big:
        return None
    # Two strongest significant annotated genes per side, by padj, non-focal.
    top = named(d[sig])
    top = top[~top.gene.isin(FOCAL)].sort_values("padj")
    items, per_side = [], {True: 0, False: 0}
    for _, r in top.iterrows():
        right = bool(r.log2FoldChange > 0)
        if per_side[right] >= 2:
            continue
        per_side[right] += 1
        items.append(dict(
            label=short_symbol(r.gene, r.gene_name, orth), right=right,
            x=float(np.clip(r.log2FoldChange, -XCAP, XCAP)),
            y=min(-np.log10(r.padj), YCAP),
        ))  # fmt: skip
    return items, counts


def make_figure(fig_dir: Path, coarse, fine, expr: pd.DataFrame, orth: dict) -> None:
    plt.rcParams.update({
        # Arial-metric sans (Liberation Sans) for print; editable text in the PDF.
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "mathtext.default": "regular", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 6.5, "axes.labelsize": 6.5,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5,
    })  # fmt: skip
    fig = plt.figure(figsize=(7.2, 6.9))
    hr = [0.22, 1, 1, 0.5, 1, 1, 0.62, 2.6]
    gs = fig.add_gridspec(len(hr), 7, height_ratios=hr, hspace=0.32, wspace=0.22,
                          left=0.07, right=0.985, top=0.95, bottom=0.085)  # fmt: skip

    def tbl(key):
        return table_for(key, coarse, fine)

    def header(axs, text, c):
        b0 = axs[0].get_position()
        b1 = axs[-1].get_position()
        y = max(a.get_position().y1 for a in axs) + 0.027
        fig.add_artist(Line2D([b0.x0, b1.x1], [y, y], color=c, lw=0.9,
                              transform=fig.transFigure))  # fmt: skip
        fig.text(b0.x0, y + 0.003, text, fontsize=7.5, fontweight="bold", color=c,
                 va="bottom")  # fmt: skip

    pending = []  # (ax, items, counts) for big panels, labelled at the end

    # band 1: whole brain | glia | non-neural
    axW = fig.add_subplot(gs[1:3, 0:2])
    pending.append(
        (
            axW,
            *volcano(
                axW,
                "Whole brain",
                tbl("Whole brain"),
                orth,
                big=True,
                title="Whole brain",
            ),
        )
    )
    axG = fig.add_subplot(gs[1:3, 2:4])
    pending.append(
        (axG, *volcano(axG, "Glia", tbl("Glia"), orth, big=True, title="All glia"))
    )
    axG.set_ylabel("")
    gl = [fig.add_subplot(gs[1 + i // 2, 4 + i % 2]) for i in range(4)]
    for i, (a, k) in enumerate(
        zip(gl, ["Glia-astro", "Glia-cortex", "Glia-peri", "Glia-sub"], strict=False)
    ):
        volcano(a, k, tbl(k), orth, xt=i >= 2, yt=i % 2 == 0)
    nn = [fig.add_subplot(gs[1, 6]), fig.add_subplot(gs[2, 6])]
    for i, (a, k) in enumerate(zip(nn, ["Hem", "PR"], strict=False)):
        volcano(a, k, tbl(k), orth, xt=i == 1, yt=True)
    header([axG, gl[1]], "Glia", "#2A8A8A")
    header([nn[0]], "Non-neural", "#6B4A3A")

    # band 2: Kenyon cells | projection & other neurons
    axK = fig.add_subplot(gs[4:6, 0:2])
    kc_items = volcano(
        axK, "Kenyon cells", tbl("Kenyon cells"), orth,
        big=True, title="All Kenyon cells (KCs)",
    )  # fmt: skip
    pending.append((axK, *kc_items))
    kc = [
        fig.add_subplot(gs[4 + r, 2 + c])
        for r, c in [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    ]
    for i, (a, k) in enumerate(
        zip(kc, ["KC-l", "KC-m", "KC-s", "KC-II", "KC-o"], strict=False)
    ):
        volcano(a, k, tbl(k), orth, xt=i >= 3, yt=i in (0, 3))
    key_ax = fig.add_subplot(gs[5, 4])
    key_ax.axis("off")
    key_handles = [
        Line2D([], [], marker=m, ls="", mfc="w", mec="k", ms=4.5, label=lab)
        for m, lab in zip("^sD", ["Kr-h1", "sNPF", "Tk"], strict=False)
    ] + [
        Line2D([], [], marker="o", ls="", mfc=c, mec="none", ms=3.5, label=lab)
        for c, lab in ((Q, "queen-up"), (W, "worker-up"), (G, "n.s."))
    ]
    key_ax.legend(handles=key_handles, loc="upper center", bbox_to_anchor=(0.5, 1.12),
                  frameon=False, fontsize=6, handletextpad=0.3, labelspacing=0.3,
                  borderaxespad=0)  # fmt: skip
    key_ax.text(0.5, -0.27,
                "Dashed line, FDR 0.05. Axes clipped at\n"
                "|log$_2$FC| 4 and −log$_{10}$ $\\mathit{P}_{adj}$ 15.",
                transform=key_ax.transAxes, ha="center", va="top", fontsize=5,
                color="#555", linespacing=1.15)  # fmt: skip
    pn = [fig.add_subplot(gs[4 + i // 2, 5 + i % 2]) for i in range(4)]
    for i, (a, k) in enumerate(
        zip(pn, ["Proj. neurons", "PN-ad", "PN-lat", "N-other"], strict=False)
    ):
        volcano(a, k, tbl(k), orth, xt=i >= 2, yt=i % 2 == 0)
    header([axK, kc[2]], "Kenyon cells", "#C2610A")
    header([pn[0], pn[1]], "Projection & other neurons", "#2A4A9A")
    for ax_, L in ((axW, "A"), (axG, "B"), (nn[0], "C"), (axK, "D"), (pn[0], "E")):
        b = ax_.get_position()
        big_left = ax_ in (axW, axK)
        fig.text(b.x0 - (0.055 if big_left else 0.012), b.y1 + 0.045, L,
                 fontsize=8, fontweight="bold", va="bottom",
                 ha="left" if big_left else "right")  # fmt: skip

    for ax_, items, counts in pending:
        if items:
            place_labels(fig, ax_, items, counts)

    # panel F: dot plot
    axE = fig.add_subplot(gs[7, :])
    _p = axE.get_position()
    axE.set_position([_p.x0 + 0.07, _p.y0, _p.width - 0.07, _p.height])
    picks = [
        tuple(x)
        for x in expr[["subtype", "gene", "gene_name"]].drop_duplicates().to_numpy()
    ]
    frac = {(r.gene, r.row, r.caste): r.frac_expressing for r in expr.itertuples()}
    gx, go, x = [], [], 0.0
    prev = None
    for sub, _gene, _gname in picks:
        if prev is not None and sub != prev:
            x += 0.55
        gx.append(x)
        go.append(sub)
        x += 1.0
        prev = sub
    for gi, (_sub, gene, _gname) in enumerate(picks):
        for ri, row in enumerate(ROWS):
            for dx, cst, cc in ((-0.18, "Queen", Q), (0.18, "Worker", W)):
                f = frac.get((gene, row, cst), np.nan)
                if not np.isfinite(f):
                    f = 0.0
                axE.scatter(gx[gi] + dx, ri, s=0.4 + 15 * f, color=cc,
                            alpha=0.3 + 0.7 * f, lw=0)  # fmt: skip
    axE.set_xticks(
        gx,
        [short_symbol(g, n_, orth) for _, g, n_ in picks],
        rotation=90,
        fontsize=5.8,
        style="italic",
    )
    axE.set_ylim(len(ROWS) - 0.5, -2.6)
    axE.set_yticks(range(len(ROWS)), [LABE.get(r, LAB.get(r, r)) for r in ROWS],
                   fontsize=6)  # fmt: skip
    axE.tick_params(axis="y", pad=7)
    for tl, r in zip(axE.get_yticklabels(), ROWS, strict=False):
        tl.set_color("#1a1a1a")
        # Colour chip just left of the axes keeps the class colour code.
        axE.plot([-0.62], [ROWS.index(r)], marker="s", ms=3.4, mfc=C[r], mec="none",
                 clip_on=False, ls="")  # fmt: skip
    axE.set_xlim(-0.6, gx[-1] + 0.6)
    axE.tick_params(length=0)
    for sp in axE.spines.values():
        sp.set_visible(False)
    for y in (3.5, 8.5, 10.5):
        axE.axhline(y, color="#e4e4e4", lw=0.5)
    # one coloured bar and name per subtype
    for sub in SUBTYPES:
        xs = [gx[i] for i, s in enumerate(go) if s == sub]
        if not xs:
            continue
        a_, b_ = xs[0], xs[-1]
        axE.plot([a_ - 0.35, b_ + 0.35], [-0.9, -0.9], color=C[sub], lw=1.8,
                 clip_on=False, solid_capstyle="butt")  # fmt: skip
        axE.text((a_ + b_) / 2, -1.12, HEAD[sub], ha="center", va="bottom",
                 fontsize=5.8, fontweight="bold", color="#1a1a1a",
                 linespacing=0.95)  # fmt: skip
    b = axE.get_position()
    fig.text(0.008, b.y1 + 0.03, "F", fontsize=8, fontweight="bold", va="bottom")
    fig.text(0.04, b.y1 + 0.031,
             "Subtype-specific caste DEGs, and where they are expressed",
             fontsize=7.5, fontweight="bold", va="bottom")  # fmt: skip
    fig.text(0.985, b.y0 - 0.052, "† best fly hit, not a reciprocal ortholog",
             fontsize=5.2, color="#555", ha="right", va="top")  # fmt: skip
    legend_handles = [
        Line2D([], [], marker="o", ls="", mfc=Q, mec="none", ms=4, label="queen"),
        Line2D([], [], marker="o", ls="", mfc=W, mec="none", ms=4, label="worker"),
    ] + [
        Line2D([], [], marker="o", ls="", mfc="#777", mec="none",
               ms=np.sqrt(0.4 + 15 * f),
               label=(f"{int(f * 100)}% nuclei" if f == 0.1 else f"{int(f * 100)}%"))
        for f in (0.1, 0.5, 0.9)
    ]  # fmt: skip
    fig.legend(
        handles=legend_handles,
        loc="lower right", bbox_to_anchor=(0.985, b.y1 + 0.022), ncol=5,
        frameon=False, fontsize=6, handletextpad=0.25, columnspacing=0.9,
    )  # fmt: skip

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "figS_caste_de_volcanoes.png", dpi=300)
    fig.savefig(fig_dir / "figS_caste_de_volcanoes.pdf")
    plt.close(fig)
    print("WROTE", fig_dir / "figS_caste_de_volcanoes.png")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--stage", choices=["expr", "plot", "all"], default="plot")
    p.add_argument(
        "--root",
        type=Path,
        default=REPO,
        help="repo root holding results/ and data/ (default: this checkout)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory for the rendered figure (default: <root>/results/figures/si)",
    )
    args = p.parse_args()
    root = args.root
    coarse, fine = load_tables(root)
    orth = load_orthologs(root)
    picks = select_panel_f(coarse, fine)
    print("panel F genes:", picks)
    if args.stage in ("expr", "all"):
        stage_expr(root, picks)
    if args.stage in ("plot", "all"):
        path = out_dir(root) / EXPR_NAME
        if not path.exists():
            raise SystemExit(f"ERROR: {path} missing; run --stage expr first")
        expr = pd.read_csv(path)
        make_figure(args.out or root / "results/figures/si", coarse, fine, expr, orth)


if __name__ == "__main__":
    with explain_missing_inputs():
        main()
