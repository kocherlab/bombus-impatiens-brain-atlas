"""Shared setup of the caste-DE / joint-cNMF notebook, as functions.

Notebook author: I. Traniello.

Source: <snrna_source>/src/scripts/03_analysis/
01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb
(md5 184efd9af8192fd09f7329f1eeaa9f17), setup cells 2, 4, 6, 9, 11, 13 and 14.
In the notebook these cells run once and leave globals behind; here each is a
function that returns what the later cells read, so a step script loads only
what it uses. The computations are the notebook's, unchanged.

Not ported: the notebook's DESeq2 functions in cell 6 (``pseudobulk``, ``run_deseq``,
``run_pseudobulk_de``): the repo has one DESeq2 implementation,
``scripts/caste_de/caste_de_blocked_all.py``, whose table is read instead;
cells 7, 8 (QC table and p-value histograms, display only); the ``fitz`` and
IPython display imports.
"""

import random
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
import scanpy as sc
from cnmf_gsea_paths import EMAPPER, GO_OBO, LOOKUP

mpl.use("Agg")

CT, CASTE, TRIAL = "cell_type", "caste", "trial"  # cell 4
COUNTS_LAYER = "counts"  # cell 6: raw counts layer
PADJ = 0.05  # cell 9
DROP = {"Other neurons", "Other glia", "Photoreceptors"}  # cell 11


def setup() -> None:
    """Cell 2: global seeds and scanpy settings, set before anything else runs."""
    random.seed(42)
    np.random.seed(42)
    sc.settings.verbosity = 3  # errors (0), warnings (1), info (2), hints (3)
    sc.settings.set_figure_params()


def load_atlas(path) -> tuple[sc.AnnData, str, str]:
    """Cell 4: the central-brain snRNA atlas and the queen / worker labels."""
    adata = sc.read_h5ad(path)
    castes = adata.obs[CASTE].astype(str).unique()
    ql = [c for c in castes if c.lower().startswith("q")][0]  # "Queen"
    wl = [c for c in castes if c.lower().startswith("w")][0]  # "Worker"
    print(adata)
    print(adata.obs.groupby([CASTE, TRIAL]).size().unstack())
    return adata, ql, wl


def load_lookup() -> pd.DataFrame:
    """Cell 4: the gene annotation table, ``gene_id`` renamed ``gene_ids``."""
    lookup = pd.read_csv(LOOKUP, sep="\t")
    return lookup.rename(columns={"gene_id": "gene_ids"})


def load_de(de_csv) -> pd.DataFrame:
    """Cells 6 and 9: the cached DE table with gene name / description / biotype.

    Positive log2FoldChange and Wald ``stat`` are queen-up.
    """
    de = pd.read_csv(de_csv)
    de["tested"] = de["tested"].astype(bool)
    print(
        de.shape,
        de.cell_type.nunique(),
        "cell types | fraction tested:",
        de.tested.mean().round(3),
    )

    # cell 9: attach gene name / description / biotype from the reference lookup
    lookup = (
        load_lookup()[["gene_ids", "gene", "description", "gene_biotype"]]
        # 'gene' becomes the ID column to match `de`
        .rename(columns={"gene_ids": "gene", "gene": "gene_name"})
        .replace({"gene_name": {"": np.nan}, "description": {"": np.nan}})
        .drop_duplicates("gene")
    )
    stale = ("gene_name", "description", "gene_biotype", "label", "label_unique")
    de = de.drop(columns=[c for c in stale if c in de.columns])
    de = de.merge(lookup, on="gene", how="left")
    de["label"] = de["gene_name"].fillna(de["gene"])  # name if any, else the ID
    print(
        "fraction of rows with a name / description:",
        de["gene_name"].notna().mean().round(3),
        de["description"].notna().mean().round(3),
    )
    print(de.gene_biotype.value_counts(dropna=False).head())

    # DEGs per cell type and direction (padj < 0.05; positive log2FC = queen-up)
    deg_counts = (
        de[de.padj < PADJ]
        .assign(dir=lambda x: np.where(x.log2FoldChange > 0, "queen", "worker"))
        .groupby(["cell_type", "dir"])
        .size()
        .unstack(fill_value=0)
    )
    print(deg_counts)
    return de


def lineage(ct: str) -> str | None:
    """Cell 11: glia / neurons for the annotated types, None for the rest."""
    if ct in DROP:
        return None
    if ct.endswith("glia"):
        return "glia"
    if "Kenyon" in ct or "projection neurons" in ct:
        return "neurons"
    return None


def annotated_de(de: pd.DataFrame) -> pd.DataFrame:
    """Cell 11: DE rows of the 11 annotated glial / neuronal types (``d0``)."""
    d0 = de[de.cell_type.map(lineage).notna()].copy()
    d0["lin"] = d0.cell_type.map(lineage)
    print({k: sorted(v) for k, v in d0.groupby("lin").cell_type.unique().items()})
    return d0


def parse_obo(path=GO_OBO) -> dict[str, tuple[str, str]]:
    """Cell 13: GO id and alt_id -> (name, namespace)."""
    meta, ids, name, ns = {}, [], None, None

    def flush():
        if ids and name:
            for i in ids:
                meta[i] = (name, ns)

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("["):
                flush()
                ids, name, ns = [], None, None
            elif line.startswith("id: "):
                ids.append(line[4:])
            elif line.startswith("alt_id: "):
                ids.append(line[8:])
            elif line.startswith("name: "):
                name = line[6:]
            elif line.startswith("namespace: "):
                ns = line[11:]
    flush()
    print("go_meta entries:", len(meta))
    return meta


def go_library(path: Path = EMAPPER) -> dict[str, dict[str, list[str]]]:
    """Cell 14: GO gene sets from the eggNOG-mapper annotation.

    Returns ``{"GO_BP": {term: [genes]}}``. Despite the key, the library holds
    every GO namespace (the notebook's note in cell 14); BP is selected after GSEA.
    """
    # skip the ## preamble; the header is the line starting "#query"
    with open(path) as fh:
        hdr = next(n for n, line in enumerate(fh) if line.startswith("#query"))
    ann = pd.read_csv(path, sep="\t", skiprows=hdr, dtype=str)
    ann = ann.rename(columns={"#query": "query"})
    ann = ann[~ann["query"].str.startswith("#")].set_index("query")  # ## footer
    print(f"{len(ann)} genes in emapper file")

    s = ann["GOs"].dropna()
    s = s[s != "-"]
    go_df = (
        s.str.split(",")
        .explode()
        .loc[lambda x: x.str.startswith("GO:")]
        .reset_index()
        .rename(columns={"query": "gene", "GOs": "term"})
        .drop_duplicates()
    )
    libs = {"GO_BP": {t: g.gene.tolist() for t, g in go_df.groupby("term")}}

    sets = libs["GO_BP"].values()
    n_in = sum(10 <= len(g) <= 500 for g in sets)
    print(
        f"GO library: {len(libs['GO_BP'])} terms, {n_in} in the 10-500 size "
        f"window, {len(set().union(*sets))} genes covered"
    )
    return libs


def go_lookup(meta: dict, go_id, field: int):
    """``go_meta.get(go_id, (None, None))[field]`` (cells 15 and 27)."""
    return meta.get(go_id, (None, None))[field]
