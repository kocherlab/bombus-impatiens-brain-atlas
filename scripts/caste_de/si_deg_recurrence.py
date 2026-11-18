#!/usr/bin/env python
"""Which caste DEGs recur across cell-type classifications, and which are on the Xenium
panel.

For every classification with a replicated snRNA caste DE (the caste_de fit,
``bimp_st.caste_de``: DESeq2 ``~ trial + caste``, queen vs worker, n = 3 bees per caste,
FDR < 0.05, positive log2FC = queen-up) take the DEG list, then count for each gene the
lists in which it is significant with the SAME direction.

Lists (distinct cell populations only):

* coarse, ``bimp_st.paths.CASTE_DE_COARSE`` (``caste_de_blocked_all.py``):
  whole brain, Kenyon cells, Glia, Hemocytes, Projection neurons, Other neurons,
  Photoreceptors;
* fine, ``bimp_st.paths.CASTE_DE_FINE`` (same fit):
  the central-brain ``cell_type`` values that are NOT the same cells as a coarse list:
  Astrocyte-like / Cortex / Perineurial / Subperineurial glia, Large- / Middle- /
  Small-type / Class II / Other Kenyon cells, Anterodorsal and Lateral projection
  neurons. Hemocytes, Photoreceptors and Other neurons are one population at both levels
  and are counted once. Other glia (two trial pairs) is left
  out, and the 13
  optic-lobe types and the optic-lobe Kenyon entry exist in one library per caste, so
  they have no replicated test.

The lists are nested (Glia contains its four subtypes, Kenyon cells contain theirs,
whole brain contains everything), so a count is a recurrence count, not an
independent-replication count; ``n_coarse`` and ``n_fine`` are reported separately.

Ranking: most lists first, ties broken by the smaller geometric-mean padj over the lists
where it is significant. Writes ``bimp_st.paths.CASTE_DE_RECURRENCE``; no figure
reads it; ``snrna_controls.py`` reads its counts and top-20 tables.

    PYTHONPATH=src:scripts .venv/bin/python scripts/caste_de/si_deg_recurrence.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from bimp_st.caste_de import focal_gene_ids  # noqa: E402
from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import (  # noqa: E402
    CASTE_DE_COARSE,
    CASTE_DE_FINE,
    CASTE_DE_RECURRENCE,
    PANEL_CROSSWALK,
)
from bimp_st.si_common import QUEEN_COLOR, WORKER_COLOR, set_style  # noqa: E402

COARSE = CASTE_DE_COARSE
FINE = CASTE_DE_FINE
RBH = REPO / "results/round5/glia_labels/v43_dmel_rbh.csv.gz"
MAP = PANEL_CROSSWALK
OUT = CASTE_DE_RECURRENCE
FDR = 0.05
TOP = 10

COARSE_LISTS = {
    "all": "Whole brain",
    "Kenyon_cells": "Kenyon cells",
    "Glia": "Glia",
    "Hemocytes": "Hemocytes",
    "Projection_neurons": "Projection neurons",
    "Other_neurons": "Other neurons",
    "Photoreceptors": "Photoreceptors",
}
FINE_LISTS = [
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Other Kenyon cells",
    "Astrocyte-like glia",
    "Cortex glia",
    "Perineurial glia",
    "Subperineurial glia",
    "Anterodorsal projection neurons",
    "Lateral projection neurons",
]
SYMBOLS = {
    g: s for s, g in focal_gene_ids().items() if s in ("Kr-h1", "sNPF", "Tk")
}


def load_lists() -> pd.DataFrame:
    c = pd.read_csv(COARSE, index_col=0)
    c = c[c.cell_type.isin(COARSE_LISTS)].assign(level="coarse")
    c["list"] = c.cell_type.map(COARSE_LISTS)
    f = pd.read_csv(FINE, index_col=0)
    # sanity: a fine type that is the same cells as a coarse list must reproduce its DEG
    # counts
    for fine, coarse in (
        ("Hemocytes", "Hemocytes"),
        ("Photoreceptors", "Photoreceptors"),
        ("Other neurons", "Other_neurons"),
    ):
        a, b = f[f.cell_type == fine], c[c.cell_type == coarse]
        sa, sb = (a.padj < FDR).sum(), (b.padj < FDR).sum()
        assert sa == sb, f"{fine}: fine {sa} vs coarse {sb} DEGs"
    missing = [x for x in FINE_LISTS if x not in set(f.cell_type)]
    assert not missing, missing
    f = f[f.cell_type.isin(FINE_LISTS)].assign(level="fine")
    f["list"] = f.cell_type
    d = pd.concat([c, f], ignore_index=True)
    return d.dropna(subset=["padj"])


def short_name(
    gene: str, rbh: pd.DataFrame, mp: pd.DataFrame, desc: str, maxlen: int | None = 30
) -> str:
    """Readable gene name. ``maxlen`` truncates long descriptions (with an
    ellipsis); the CSVs keep the historical 30-character form, the figure
    passes ``maxlen=None`` and wraps instead (see ``figure_label``)."""
    if gene in SYMBOLS:
        return SYMBOLS[gene]
    if not gene.startswith("BIMP_v4.3_"):  # e.g. the mitochondrial MT-COX1 entry
        return gene
    if gene in mp.index:
        m = re.search(r"\[protein=([^\]]+)\]", str(mp.loc[gene, "description"]))
        if m:
            return re.sub(r"\s*isoform\s+\S+$", "", m.group(1))
    if gene in rbh.index:
        r = rbh.loc[gene]
        if (
            r["call"] == "reciprocal best hit"
            and pd.notna(r["dmel_symbol"])
            and not re.fullmatch(r"CG\d+", str(r["dmel_symbol"]).replace("Dmel\\", ""))
        ):
            return str(r["dmel_symbol"]).replace("Dmel\\", "")
    d = (
        str(rbh.loc[gene, "description"])
        if gene in rbh.index and pd.notna(rbh.loc[gene, "description"])
        else ""
    )
    d = re.sub(r"\s*isoform\s+\S+$", "", d).strip()
    if d:
        if maxlen is None or len(d) <= maxlen:
            return d
        return d[:maxlen].rsplit(" ", 1)[0] + "…"
    bt = str(rbh.loc[gene, "gene_biotype"]) if gene in rbh.index else ""
    return gene[-6:] + (" (lncRNA)" if bt == "lncRNA" else " (unannotated)")


# Figure-only short forms for NCBI descriptions too long for the row labels
# (the CSVs keep the full names).
FIGURE_ABBREV = {
    "leucine-rich repeat, immunoglobulin-like domain-containing kekkon 5 protein": (
        "kekkon 5"
    ),
    "deleted in lung and esophageal cancer protein 1 homolog": "DLEC1 homolog",
    "protein lethal(2)essential for life-like": "l(2)efl-like",
    "cilia- and flagella-associated protein 44": "CFAP44",
}


def figure_label(
    gene: str, rbh: pd.DataFrame, mp: pd.DataFrame, desc: str, width: int = 34
) -> str:
    """Untruncated name for the dot matrix, wrapped onto at most two lines."""
    import textwrap

    name = short_name(gene, rbh, mp, desc, maxlen=None)
    name = FIGURE_ABBREV.get(name, name)
    # no breaks at hyphens: rejoining a hyphen break left "domain- containing"
    lines = textwrap.wrap(name, width, break_on_hyphens=False)
    if len(lines) > 2:
        lines = [lines[0], " ".join(lines[1:])]
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_lists()
    lists = list(COARSE_LISTS.values()) + FINE_LISTS
    d["sig"] = d.padj < FDR
    d["dir"] = np.where(d.log2FoldChange > 0, "queen", "worker")
    s = d[d.sig]
    rows = []
    desc = d.drop_duplicates("gene").set_index("gene").gene_name
    rbh = pd.read_csv(RBH).set_index("gene")
    mp = read_panel_crosswalk().set_index("v43_id")
    for g, gg in s.groupby("gene"):
        q, w = gg[gg.dir == "queen"], gg[gg.dir == "worker"]
        rows.append(
            dict(
                gene=g,
                n_lists_sig=len(gg),
                n_queen_up=len(q),
                n_worker_up=len(w),
                n_coarse_queen_up=int((q.level == "coarse").sum()),
                n_fine_queen_up=int((q.level == "fine").sum()),
                n_coarse_worker_up=int((w.level == "coarse").sum()),
                n_fine_worker_up=int((w.level == "fine").sum()),
                lists_queen_up="; ".join(q.sort_values("padj")["list"]),
                lists_worker_up="; ".join(w.sort_values("padj")["list"]),
                gmean_padj_queen=float(np.exp(np.log(q.padj.clip(lower=1e-300)).mean()))
                if len(q)
                else np.nan,
                gmean_padj_worker=float(
                    np.exp(np.log(w.padj.clip(lower=1e-300)).mean())
                )
                if len(w)
                else np.nan,
            )
        )
    t = pd.DataFrame(rows)
    ntest = d.groupby("gene").size().rename("n_lists_tested")
    t = t.merge(ntest, left_on="gene", right_index=True)
    t["name"] = [short_name(g, rbh, mp, desc.get(g, "")) for g in t.gene]
    t["description"] = [
        str(desc.get(g, "")) if str(desc.get(g, "")) != "nan" else "" for g in t.gene
    ]
    t["on_101_gene_panel"] = np.where(t.gene.isin(mp.index), "yes", "no")
    t = t.sort_values(["n_queen_up", "gmean_padj_queen"], ascending=[False, True])
    cols = [
        "gene",
        "name",
        "description",
        "on_101_gene_panel",
        "n_lists_tested",
        "n_lists_sig",
        "n_queen_up",
        "n_worker_up",
        "n_coarse_queen_up",
        "n_fine_queen_up",
        "n_coarse_worker_up",
        "n_fine_worker_up",
        "lists_queen_up",
        "lists_worker_up",
        "gmean_padj_queen",
        "gmean_padj_worker",
    ]
    t[cols].to_csv(OUT / "cross_list_counts.csv", index=False)

    topq = t.sort_values(
        ["n_queen_up", "gmean_padj_queen"], ascending=[False, True]
    ).head(TOP)
    topw = t.sort_values(
        ["n_worker_up", "gmean_padj_worker"], ascending=[False, True]
    ).head(TOP)
    top = pd.concat([topq.assign(side="queen-up"), topw.assign(side="worker-up")])
    top[["side"] + cols].to_csv(OUT / "cross_list_top20.csv", index=False)

    # panel genes only: these are the ones that can be drawn as Xenium maps
    pg = t[t.on_101_gene_panel == "yes"]
    pq = (
        pg.sort_values(["n_queen_up", "gmean_padj_queen"], ascending=[False, True])
        .head(TOP)
        .assign(side="queen-up")
    )
    pw = (
        pg.sort_values(["n_worker_up", "gmean_padj_worker"], ascending=[False, True])
        .head(TOP)
        .assign(side="worker-up")
    )
    pd.concat([pq, pw])[["side"] + cols].to_csv(
        OUT / "cross_list_panel_genes_top.csv", index=False
    )

    # per-list numbers behind the dot matrix
    dm = d[d.gene.isin(top.gene)][
        ["gene", "list", "level", "log2FoldChange", "padj", "sig", "dir"]
    ]
    dm.to_csv(OUT / "cross_list_top20_per_list.csv", index=False)

    # ---------------- figure
    set_style()
    genes = list(topq.gene) + list(topw.gene)
    label = {g: figure_label(g, rbh, mp, desc.get(g, "")) for g in genes}
    fig, ax = plt.subplots(figsize=(7.2, 5.9))  # printed at full text width (6.7 in)
    ng = len(genes)
    gap = 0.9
    ypos = {g: (ng - 1 - i) + (gap if i < TOP else 0) for i, g in enumerate(genes)}
    ypos = {g: (ng - 1 - i) + (0.0 if i >= TOP else 0.9) for i, g in enumerate(genes)}
    xpos = {
        lst: i + (0.6 if i >= len(COARSE_LISTS) else 0.0) for i, lst in enumerate(lists)
    }
    piv = dm.set_index(["gene", "list"])
    for g in genes:
        for lst in lists:
            if (g, lst) not in piv.index:
                continue
            r = piv.loc[(g, lst)]
            if r.sig:
                ax.scatter(
                    xpos[lst],
                    ypos[g],
                    s=18 + 38 * min(abs(r.log2FoldChange), 6) ** 1.0,
                    color=QUEEN_COLOR if r.log2FoldChange > 0 else WORKER_COLOR,
                    edgecolor="#222222",
                    lw=0.4,
                    zorder=3,
                )
            else:
                ax.scatter(xpos[lst], ypos[g], s=5, color="#CFCFCF", zorder=2)
    for i, g in enumerate(genes):
        n = int(top.iloc[i].n_queen_up if i < TOP else top.iloc[i].n_worker_up)
        panel = g in mp.index
        ax.text(
            -0.9,
            ypos[g],
            label[g] + (" ◆" if panel else ""),
            ha="right",
            va="center",
            fontsize=7.4 if "\n" not in label[g] else 6.6,
            linespacing=0.95,
            fontweight="bold" if panel else "normal",
        )
        ax.text(
            max(xpos.values()) + 0.8,
            ypos[g],
            f"{n}",
            ha="left",
            va="center",
            fontsize=7.4,
            color="#333333",
        )
    ax.text(
        max(xpos.values()) + 0.8,
        ng + 0.7,
        "# lists",
        ha="left",
        va="bottom",
        fontsize=7.4,
        color="#333333",
    )
    ax.set_xticks([xpos[lst] for lst in lists])
    ax.set_xticklabels(
        lists, rotation=55, ha="right", rotation_mode="anchor", fontsize=7.2
    )
    ax.set_yticks([])
    ax.set_xlim(-0.6, max(xpos.values()) + 1.6)
    ax.set_ylim(-0.8, ng + 1.6)  # headroom: "queen-up" / "# lists" clear the edge
    ax.axhline(ypos[genes[TOP - 1]] - 0.55, color="#888888", lw=0.5)
    ax.axvline(len(COARSE_LISTS) - 0.5 + 0.3, color="#888888", lw=0.5, ls=":")
    ax.text(
        (len(COARSE_LISTS) - 1) / 2,
        ng + 0.55,
        "coarse",
        ha="center",
        fontsize=7.6,
        fontweight="bold",
    )
    ax.text(
        len(COARSE_LISTS) + 0.6 + (len(FINE_LISTS) - 1) / 2,
        ng + 0.55,
        "fine",
        ha="center",
        fontsize=7.6,
        fontweight="bold",
    )
    ax.text(
        -0.9,
        ypos[genes[0]] + 0.9,
        "queen-up",
        ha="right",
        fontsize=7.6,
        fontweight="bold",
        color=QUEEN_COLOR,
    )
    ax.text(
        -0.9,
        ypos[genes[TOP]] + 0.9,
        "worker-up",
        ha="right",
        fontsize=7.6,
        fontweight="bold",
        color=WORKER_COLOR,
    )
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    # key: colour = direction, size = |log2FC|
    hx = [
        ax.scatter([], [], s=18 + 38 * v, color="#999999", edgecolor="#222222", lw=0.4)
        for v in (1, 2, 4)
    ]
    ax.legend(
        hx,
        ["1", "2", "4"],
        # under the gene names, in the corner the rotated column labels leave
        loc="upper right",
        bbox_to_anchor=(-0.03, -0.02),
        ncol=3,
        frameon=False,
        fontsize=7.2,
        title=(
            "gold, queen-up; blue, worker-up\n"
            "grey, tested, FDR ≥ 0.05\n"
            "dot size = |log$_2$FC|"
        ),
        title_fontsize=7.2,
        alignment="right",
        handletextpad=0.3,
        columnspacing=1.2,
    )
    fig.savefig(
        OUT / "cross_list_dotmatrix.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUT / "cross_list_dotmatrix.pdf", bbox_inches="tight", facecolor="white"
    )
    pd.set_option("display.width", 250)
    print(
        top[
            [
                "side",
                "gene",
                "name",
                "on_101_gene_panel",
                "n_queen_up",
                "n_worker_up",
                "n_lists_tested",
            ]
        ].to_string(index=False)
    )
    print("lists:", len(lists))
    print("\nTop panel genes:")
    print(
        pd.concat([pq, pw])[
            [
                "side",
                "gene",
                "name",
                "n_queen_up",
                "n_worker_up",
                "n_lists_tested",
                "lists_queen_up",
                "lists_worker_up",
            ]
        ].to_string(index=False, max_colwidth=70)
    )


if __name__ == "__main__":
    main()
