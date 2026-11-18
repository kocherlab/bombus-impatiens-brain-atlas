#!/usr/bin/env python
"""Supplementary Data CSVs: data tables for the reader to look up.

Data the reader looks up rather than reads as part of an argument go out as CSV files
(``supplementary_data/<slug>.csv`` beside the LaTeX sources), with a one-line entry each
in the Supplementary Information's Supplementary Data list. File names carry no
number: LaTeX numbers the items (with \\suppdata) in order of first citation, and
``manuscript_figure_registry.SUPPLEMENTARY_DATA_ITEMS`` holds that order, the slugs
and the labels. ``deposit/export_numbered_si.py`` writes numbered copies for a
journal from the .aux files of a LaTeX build.

Nothing is recomputed here. Each file is a renamed, LaTeX-free view of a table an
analysis script already wrote, plus Drosophila ortholog columns:

  panel_key                  results/handoff/D_tables/S1_panel_key.csv
                             (gene_annotation/tables_s1_s2.py)
  xenium_qc_metrics          the per-capture-region Xenium Ranger metrics_summary.csv
                             files, read by atlas/si_xenium_run_qc.py::load_metrics
  snrna_populations_top50    results/handoff/D_tables/S2_snrna_annotations.csv and
                             snrna_markers_top50.csv (both gene_annotation/tables_s1_s2.py)
  cnmf_program_caste         results/caste_programs/pipeline/fig3_inputs/
                             joint_k50_caste.csv and program_labels.csv
                             (caste_de/cnmf_gsea/fig3_inputs.py)
  regional_caste_de          results/round4/regional_degs/caste_effects/
                             supp_table_caste_effects_by_region.csv
                             (caste_de/regional_deg_table.py)
  glia_caste_de_top10        results/caste_de/glia_top10/glia_top10_degs.csv
                             (caste_de/table_glia_top10_degs.py)
  snrna_markers_top500       results/handoff/D_tables/snrna_markers_top500_canonical17.csv
                             (gene_annotation/snrna_markers_top500.py)
  spatial_markers_top500     results/handoff/D_tables/spatial_markers_top500.csv
  spatial_measured_gene_scores  results/handoff/D_tables/spatial_markers_panel_scores.csv
                             (both gene_annotation/spatial_markers_top500.py)

Orthologs: the panel key and the regional table (panel genes only) carry the panel
key's curated calls; the other gene tables carry the genome-wide reciprocal best hit
of ``results/round5/glia_labels/v43_dmel_rbh.csv.gz`` (gene_annotation/
genome_dmel_orthologs.py), only where the call is a reciprocal best hit.

Writes only results/supplementary_data/ (the files named in
manuscript_figure_registry.SUPPLEMENTARY_DATA plus SUPPLEMENTARY_DATA.md). Copy the
directory into supplementary_data/ of the LaTeX sources;
verify_manuscript_figures.py byte-compares the two.

    PYTHONPATH=src:scripts .venv/bin/python scripts/deposit/supplementary_data_csvs.py
"""

# ruff: noqa: E501  (README text strings)
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from paper.manuscript_figure_registry import SUPPLEMENTARY_DATA_ITEMS  # noqa: E402

from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import (  # noqa: E402
    CASTE_DE_DIR,
    CASTE_PROGRAMS_DIR,
    PANEL_CROSSWALK,
    explain_missing_inputs,
)

D = ROOT / "results" / "handoff" / "D_tables"
REGDEG = (
    ROOT
    / "results"
    / "round4"
    / "regional_degs"
    / "caste_effects"
    / "supp_table_caste_effects_by_region.csv"
)
CNMF = CASTE_PROGRAMS_DIR / "pipeline" / "fig3_inputs"
GLIA10 = CASTE_DE_DIR / "glia_top10" / "glia_top10_degs.csv"
RBH = ROOT / "results" / "round5" / "glia_labels" / "v43_dmel_rbh.csv.gz"
MAP = PANEL_CROSSWALK
OUT = Path(os.environ.get("SD_OUT", ROOT / "results" / "supplementary_data"))
FLOAT = "%.6g"
# Replicated program x cell type call (label_swap_null.py; the stars in panel D of
# fig:caste_de): one sign in all three trials, |mean| >= 0.5 SD, weakest trial
# >= 0.25 SD, usage >= 2%.
CNMF_MEAN_MIN, CNMF_WEAKEST_MIN, CNMF_USE_MIN = 0.5, 0.25, 2.0


# Published file name and LaTeX label of each Supplementary Data item, by slug, in
# first-citation order.
NAME = {d.slug: f"{d.slug}.csv" for d in SUPPLEMENTARY_DATA_ITEMS}
LABEL = {d.slug: d.label for d in SUPPLEMENTARY_DATA_ITEMS}

# Regional DE region keys -> display names, in the order of the registered table.
REGIONS = {
    "kc_ic": "Small-type compartment",
    "kc_nc": "Large-type compartment",
    "kc_oc": "Class II compartment",
    "lamina": "Lamina",
    "medulla": "Medulla",
    "lobula": "Lobula",
}
# spatial_markers_top500.py region names -> the same paper names.
SPATIAL_REGIONS = {
    "Small-type": "Small-type compartment",
    "Large-type": "Large-type compartment",
    "Class II": "Class II compartment",
}
TYPO = {"serotinergic": "serotonergic", "seritonergic": "serotonergic"}

# Xenium Ranger metrics_summary.csv fields, one row per capture region.
QC_FIELDS = [
    "median_transcripts_per_cell",
    "median_genes_per_cell",
    "num_cells_detected",
    "adjusted_negative_control_probe_rate",
    "adjusted_negative_control_codeword_rate",
    "negative_control_probe_counts_per_control_per_cell",
    "estimated_number_of_false_positive_transcripts_per_cell",
    "fraction_transcripts_decoded_q20",
    "fraction_transcripts_assigned",
    "decoded_transcripts_per_100um2",
]


def rbh_orthologs() -> pd.DataFrame:
    """v4.3 gene -> Drosophila ortholog, reciprocal best hits only."""
    r = pd.read_csv(RBH, dtype=str, keep_default_na=False)
    ok = r.call.str.startswith("reciprocal best hit")
    r = r[ok].set_index("gene")
    return pd.DataFrame(
        {
            "dmel_ortholog_symbol": r.dmel_symbol.str.replace(
                "Dmel\\", "", regex=False
            ),
            "dmel_flybase_id": r.dmel_FBgn,
            "dmel_ortholog_call": r.call,
        }
    )


def add_rbh(df: pd.DataFrame, orth: pd.DataFrame, after: str) -> pd.DataFrame:
    o = orth.reindex(df.gene_id).fillna("").reset_index(drop=True)
    df = df.reset_index(drop=True)
    i = df.columns.get_loc(after) + 1
    for k, c in enumerate(o.columns):
        df.insert(i + k, c, o[c])
    return df


def write(
    slug: str, df: pd.DataFrame, float_format: str | None = FLOAT
) -> pd.DataFrame:
    df.to_csv(
        OUT / NAME[slug], index=False, float_format=float_format, encoding="utf-8"
    )
    return df


# snRNA caste tables. The builders (ported from the
# analysis notebook and out/fig4_supp scripts) keep the saved tables' columns and
# full float precision. The two tables that depend on the DE fit read the repo's
# fit (``source="repo"``: pyDESeq2 0.5.4, ~ trial + caste, caste_de_blocked_all.py,
# bimp_st.paths.CASTE_DE_FINE), the one fit behind fig:caste_de and every DE
# number. The saved pyDESeq2 0.5.2 fit (``source="saved"``) is kept for
# comparison only (bimp_st.paths.CASTE_DE_PARITY_SAVED).
CNMF_GSEA_DIR = ROOT / "scripts" / "caste_de" / "cnmf_gsea"
SNRNA_SOURCE = "repo"
# EGAPx inputs: rebuilt from the run's own parameter file
# (data/genome/egapx_run_params.yaml) by deposit/sd_egapx_inputs.py. The short-read
# table reproduces the author-supplied sheet byte for byte (checked there).


def snrna_caste_tables() -> dict[str, pd.DataFrame]:
    sys.path.insert(0, str(CNMF_GSEA_DIR))
    sys.path.insert(0, str(ROOT / "scripts" / "deposit"))
    import sd_cellranger_qc
    import sd_celltype_go
    import sd_program_go
    import sd_pseudobulk_degs

    return {
        "snrna_caste_degs": sd_pseudobulk_degs.build(SNRNA_SOURCE),
        "snrna_caste_gsea_celltypes": sd_celltype_go.build(SNRNA_SOURCE),
        "cnmf_program_go_terms": sd_program_go.build("repo"),
        "snrna_library_qc": sd_cellranger_qc.build(),
    }


def blank(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str)


def xenium_qc() -> pd.DataFrame:
    """One row per capture region plus the median over the 32 (the values the
    Results quote), from the Xenium Ranger metrics files."""
    from atlas.si_xenium_run_qc import load_metrics

    cwd = Path.cwd()
    os.chdir(ROOT)  # load_metrics globs data/... relative to the repo root
    try:
        df = load_metrics()
    finally:
        os.chdir(cwd)
    q = df[["caste", "slide", "section"] + QC_FIELDS].rename(
        columns={"section": "capture_region"}
    )
    q["false_positives_pct_of_median_transcripts"] = (
        100
        * q["estimated_number_of_false_positive_transcripts_per_cell"]
        / q["median_transcripts_per_cell"]
    )
    q = q.sort_values(["caste", "capture_region"]).reset_index(drop=True)
    med = q[QC_FIELDS + ["false_positives_pct_of_median_transcripts"]].median()
    pooled = pd.DataFrame(
        [{"caste": "all", "slide": "", "capture_region": f"median of {len(q)}", **med}]
    )
    return pd.concat([q, pooled], ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    orth = rbh_orthologs()
    s1 = pd.read_csv(D / "S1_panel_key.csv", dtype=str, keep_default_na=False)
    out: dict[str, pd.DataFrame] = {}

    # ---- Xenium panel key ----------------------------------------------------------
    pk = pd.DataFrame(
        {
            "symbol": s1.symbol,
            "symbol_source": s1.symbol_source,
            "gene_id": s1.v43_id,
            "xenium_probe_id": s1.BIMP_NP_Symbol,
            "ncbi_gene": s1.loc_id,
            "ncbi_protein_name": s1.protein,
            "dmel_ortholog_symbol": s1.dmel_symbol,
            "dmel_flybase_id": s1.dmel_flybase_id,
            "dmel_ortholog_method": s1.dmel_method,
            "category": s1.marker_class,
            "relevance": s1.selection_category,
            "anatomical_marker": s1.anatomical_marker,
            "molecular_class": s1.molecular_class,
            "circuit_sender": s1.circuit_sender.replace(TYPO),
            "circuit_receiver": s1.circuit_receiver.replace(TYPO),
            "in_snrna_reference": s1.in_snrna_reference.map(
                {"True": "yes", "False": "no"}
            ),
            "probe_note": s1.probe_note,
        }
    )
    assert len(pk) == 101 and pk.in_snrna_reference.isin(["yes", "no"]).all()
    out["xenium_panel_key"] = write("xenium_panel_key", pk)

    # ---- Xenium run metrics per capture region ---------------------------------------
    qc = xenium_qc()
    assert len(qc) == 33 and set(qc.caste) == {"queen", "worker", "all"}
    out["xenium_qc_metrics"] = write("xenium_qc_metrics", qc)

    # ---- snRNA populations, top 50 markers each -----------------------------------
    s2 = pd.read_csv(D / "S2_snrna_annotations.csv")
    top = pd.read_csv(D / "snrna_markers_top50.csv")
    assert set(top.cell_type) == set(s2.annotation), "population names differ"
    for r in s2.itertuples():  # S2's top-10 column is this file's ranks 1-10
        t10 = top[top.cell_type == r.annotation].sort_values("rank").gene_id.head(10)
        assert list(t10) == r.top10_gene_ids.split("; "), r.annotation
    pop = s2.set_index("annotation")
    top = top.assign(pid=top.cell_type.map(pop["id"])).sort_values(["pid", "rank"])
    p5 = pd.DataFrame(
        {
            "population_id": top.pid,
            "population": top.cell_type,
            "pooled_group": top.cell_type.map(pop.pooled_group),
            "nuclei_queen": top.cell_type.map(pop.n_queen),
            "nuclei_worker": top.cell_type.map(pop.n_worker),
            "panel_genes_in_top50": top.cell_type.map(pop.n_panel_in_top50),
            "rank": top["rank"],
            "gene_id": top.gene_id,
            "symbol": blank(top.symbol),
            "description": blank(top.description),
            "wilcoxon_score": top.score,
            "log2_fold_change": top.logfoldchange,
            "adjusted_p_value": top.pval_adj,
            "on_xenium_panel": top.on_panel.astype(bool),
        }
    )
    chk = p5[p5.on_xenium_panel].groupby("population").size()
    assert (chk.reindex(pop.index).fillna(0).astype(int) == pop.n_panel_in_top50).all()
    out["snrna_populations_top50"] = write(
        "snrna_populations_top50", add_rbh(p5, orth, "description")
    )

    # ---- cNMF program usage by caste, per cell type (snRNA-seq) ---------------------
    jc = pd.read_csv(CNMF / "joint_k50_caste.csv")
    lab = pd.read_csv(CNMF / "program_labels.csv").set_index("program")
    assert len(jc) == 650 and (jc.k == 50).all()
    eff = jc[["t1", "t2", "t3"]]
    rep_call = (
        jc.same.astype(bool)
        & (jc["mean"].abs() >= CNMF_MEAN_MIN)
        & (eff.abs().min(1) >= CNMF_WEAKEST_MIN)
        & (jc.usage_pct >= CNMF_USE_MIN)
    )
    assert int(rep_call.sum()) == 9, int(rep_call.sum())
    cn = pd.DataFrame(
        {
            "cell_type": jc.cell_type,
            "program": jc.program,
            "effect_trial_1": jc.t1,
            "effect_trial_2": jc.t2,
            "effect_trial_3": jc.t3,
            "mean_effect": jc["mean"],
            "weakest_trial_abs_effect": jc.min_abs,
            "same_sign_all_trials": jc.same.astype(bool),
            "usage_pct": jc.usage_pct,
            "replicated": rep_call,
            "program_home_cell_type": jc.home,
            "program_home_fraction": jc.home_frac,
            "program_label": jc.program.map(lab.label).fillna(""),
            "program_label_go_id": jc.program.map(lab.go_id).fillna(""),
            "program_label_go_fdr": jc.program.map(lab.fdr),
        }
    ).sort_values(["program", "cell_type"])
    out["cnmf_program_caste"] = write("cnmf_program_caste", cn)

    # ---- regional queen vs worker panel DE (spatial arm) ----------------------------
    rd = pd.read_csv(REGDEG)
    m = read_panel_crosswalk()
    np2v = dict(zip(m.np31_id, m.v43_id, strict=True))
    rd["xenium_probe_id"] = rd.gene.str.split("|").str[0]
    rd["gene_id"] = rd.xenium_probe_id.map(np2v)
    assert rd.gene_id.notna().all()
    k = s1.set_index("v43_id")
    rd["order"] = rd.region.map({r: i for i, r in enumerate(REGIONS)})
    rd["absfc"] = rd.log2FC_worker_over_queen.abs()
    rd["abscons"] = rd.slice_consistency.abs()
    rd = rd.sort_values(
        ["order", "abscons", "absfc"], ascending=[True, False, False], kind="mergesort"
    )
    rd["rank_in_region"] = rd.groupby("region").cumcount() + 1
    r6 = pd.DataFrame(
        {
            "region": rd.region.map(REGIONS),
            "rank_in_region": rd.rank_in_region,
            "gene_id": rd.gene_id,
            "xenium_probe_id": rd.xenium_probe_id,
            "symbol": rd.gene_id.map(k.symbol),
            "description": rd.gene_id.map(k.protein),
            "dmel_ortholog_symbol": rd.gene_id.map(k.dmel_symbol),
            "dmel_flybase_id": rd.gene_id.map(k.dmel_flybase_id),
            "dmel_ortholog_method": rd.gene_id.map(k.dmel_method),
            "log2_fold_change_worker_over_queen": rd.log2FC_worker_over_queen,
            "slice_consistency": rd.slice_consistency,
            "mean_expression_queen": rd.mean_queen,
            "mean_expression_worker": rd.mean_worker,
            "sections_queen": rd.n_sections_queen,
            "sections_worker": rd.n_sections_worker,
            "cells_queen": rd.n_cells_queen,
            "cells_worker": rd.n_cells_worker,
        }
    )
    assert r6.region.notna().all() and r6.symbol.notna().all()
    out["regional_caste_de"] = write("regional_caste_de", r6)

    # ---- ten most significant pooled-glia caste DEGs (snRNA-seq) --------------------
    g = pd.read_csv(GLIA10, dtype={"apis_geneid": str}, keep_default_na=False)
    assert list(g["rank"]) == list(range(1, 11)), "expected the ten top glia DEGs"
    gd = g.short_description.where(~g.short_description.str.endswith("no annotation"))
    g10 = pd.DataFrame(
        {
            "rank": g["rank"],
            "gene_id": g.bimp_v43_id,
            "description": blank(gd),
            "gene_biotype": g.gene_biotype,
            "higher_in": g.direction.map({"queen-up": "queen", "worker-up": "worker"}),
            "log2_fold_change_queen_over_worker": g.log2FC_queen_over_worker,
            "adjusted_p_value": g.padj.astype(float),
            "on_xenium_panel": g.on_101_gene_panel.map({"yes": True, "no": False}),
        }
    )
    assert g10.higher_in.notna().all() and g10.on_xenium_panel.notna().all()
    out["glia_caste_de_top10"] = write(
        "glia_caste_de_top10", add_rbh(g10, orth, "gene_biotype")
    )

    # ---- snRNA markers, 17 atlas types x 500 ---------------------------------------
    sn = pd.read_csv(D / "snrna_markers_top500_canonical17.csv")
    sn = pd.DataFrame(
        {
            "cell_type": sn.cell_type,
            "pooled_group": sn.pooled_group,
            "rank": sn["rank"],
            "gene_id": sn.gene_id,
            "symbol": blank(sn.symbol),
            "description": blank(sn.description),
            "wilcoxon_score": sn.score,
            "log2_fold_change": sn.logfoldchange,
            "adjusted_p_value": sn.pval_adj,
            "cohens_d": sn.cohens_d,
            "on_xenium_panel": sn.on_panel.astype(bool),
        }
    )
    out["snrna_markers_top500"] = write(
        "snrna_markers_top500", add_rbh(sn, orth, "description")
    )

    # ---- regional top 500, measured and inferred -----------------------------------
    sp = pd.read_csv(D / "spatial_markers_top500.csv")
    sp = pd.DataFrame(
        {
            "region": sp.region.replace(SPATIAL_REGIONS),
            "rank": sp["rank"],
            "gene_id": sp.gene_id,
            "symbol": blank(sp.symbol),
            "description": blank(sp.description),
            "basis": sp.basis,
            "cohens_d": sp.score,
            "log2_fold_change": sp.log2fc,
            "mean_expression_in_region": sp.mean_in,
            "mean_expression_elsewhere": sp.mean_out,
        }
    )
    out["spatial_markers_top500"] = write(
        "spatial_markers_top500", add_rbh(sp, orth, "description")
    )

    # ---- the 100 measured genes in every region ------------------------------------
    pn = pd.read_csv(D / "spatial_markers_panel_scores.csv")
    regs = [c for c in pn.columns if c not in ("gene_id", "symbol", "description")]
    pl = pn.melt(
        id_vars=["gene_id", "symbol", "description"],
        value_vars=regs,
        var_name="region",
        value_name="cohens_d",
    )
    pl["region_order"] = pl.region.map({r: i for i, r in enumerate(regs)})
    pl = pl.sort_values(["region_order", "cohens_d"], ascending=[True, False])
    pl["rank_among_measured"] = pl.groupby("region").cumcount() + 1
    pl = pd.DataFrame(
        {
            "region": pl.region.replace(SPATIAL_REGIONS),
            "rank_among_measured": pl.rank_among_measured,
            "gene_id": pl.gene_id,
            "symbol": blank(pl.symbol),
            "description": blank(pl.description),
            "cohens_d": pl.cohens_d,
        }
    )
    out["spatial_measured_gene_scores"] = write(
        "spatial_measured_gene_scores", add_rbh(pl, orth, "description")
    )

    for slug, df in snrna_caste_tables().items():
        out[slug] = write(slug, df, float_format=None)
    sys.path.insert(0, str(ROOT / "scripts" / "deposit"))
    import sd_egapx_inputs

    sr = sd_egapx_inputs.short_reads()
    (OUT / NAME["egapx_rnaseq_samples"]).write_bytes(
        ("\ufeff" + sr.to_csv(index=False, lineterminator="\r\n")).encode("utf-8")
    )
    out["egapx_rnaseq_samples"] = sr
    out["egapx_isoseq_flnc"] = write(
        "egapx_isoseq_flnc", sd_egapx_inputs.long_reads(), float_format=None
    )

    assert set(out) == set(NAME), set(NAME) ^ set(out)
    (OUT / "SUPPLEMENTARY_DATA.md").write_text(readme(out, rd), encoding="utf-8")
    for s in NAME:
        p = OUT / NAME[s]
        print(f"{p.name}  {len(out[s])} rows  {p.stat().st_size / 1024:.0f} KiB")


# ---------------------------------------------------------------- README
def sd(slug: str) -> str:
    return f"`{NAME[slug]}`"


ORTH_RBH = (
    "`dmel_ortholog_symbol`, `dmel_flybase_id`, `dmel_ortholog_call`: *Drosophila "
    "melanogaster* ortholog by reciprocal best hit of the BIMP v4.3 proteome against the "
    "*D. melanogaster* UniProt proteome (DIAMOND blastp, e-value <= 1e-10, >= 30% query "
    "coverage; `reciprocal best hit (one-to-many)` when the second-best fly gene scores "
    "0.8-0.9 of the best). Blank: no reciprocal call (non-reciprocal and ambiguous hits are "
    "not listed)."
)
GENE = (
    "`gene_id`: *B. impatiens* gene, BIMP_v4.3 annotation (the snRNA-seq reference). "
    "`symbol`: paper or annotation symbol, blank where none. `description`: NCBI product "
    "description, blank where none."
)
WORDS = "no one two three four five six seven eight nine ten".split()

# One line per item, mirrored by the \suppdata list in 02_Article_Supplementary.tex
# (same order, file names, descriptions and citing locations). Figures are named by
# LaTeX label, never by printed number, which LaTeX owns.
TITLE = {
    "xenium_panel_key": "The 101 Xenium probes (BIMP vNP3.1) with gene IDs, Drosophila orthologs, panel-design category and presence in the snRNA-seq reference",
    "xenium_qc_metrics": "Xenium run metrics of each of the 32 capture regions (transcripts and genes per cell, cells, negative-control and decoding rates), with their median",
    "snrna_populations_top50": "The 29 populations of the snRNA-seq reference with nuclei per caste and each population's top 50 marker genes",
    "cnmf_program_caste": "Queen versus worker usage of each of the 50 cNMF expression programs in each of 13 cell types, per trial, with the replicated program x cell type calls (snRNA-seq)",
    "regional_caste_de": "Queen versus worker expression of the 101 panel genes in each of the six compared regions (spatial arm; log2 fold change is worker/queen; one brain per caste, so effect sizes only, no P values)",
    "glia_caste_de_top10": "The ten most significant caste-differentially expressed genes in pooled glia (snRNA-seq, n = 3 pooled samples per caste), with annotation and Drosophila orthologs",
    "snrna_markers_top500": "The 500 top-ranked marker genes of each of the 17 atlas types in the snRNA-seq reference",
    "spatial_markers_top500": "The 500 top-ranked genes of each traced region by Cohen's d, each marked measured (panel gene) or inferred (Tangram projection of the reference)",
    "spatial_measured_gene_scores": "Cohen's d of each of the 100 measured panel genes in every traced region",
    "snrna_caste_degs": "Caste-differentially expressed genes (FDR < 0.05) in each of 13 central brain cell types of the snRNA-seq reference (pseudobulk DESeq2, ~ trial + caste, n = 3 samples per caste), with gene annotation",
    "snrna_caste_gsea_celltypes": "Preranked GSEA of the caste Wald statistic against GO gene sets in each of 11 glial and neuronal cell types (snRNA-seq): every term tested, unfiltered by FDR, with leading-edge genes",
    "egapx_rnaseq_samples": "The 38 paired-end short-read RNA-seq samples used for EGAPx genome annotation (24 from the NCBI SRA, 14 generated in-house), with read file names",
    "egapx_isoseq_flnc": "The two PacBio Iso-Seq FLNC read sets used for EGAPx genome annotation, with barcode, run and read counts",
    "snrna_library_qc": "Cell Ranger mapping and cell metrics and CellBender settings of the eight snRNA-seq libraries",
    "cnmf_program_go_terms": "GO terms of the joint cNMF programs from preranked GSEA on program gene scores (terms at FDR < 0.25, tiered at 0.05), with leading-edge genes, whether the program is shown in panel D of `fig:caste_de` and its replicated caste effect",
}
CITED = {
    "xenium_panel_key": "Results; figure `fig:kc_subtypes`, panel D",
    "xenium_qc_metrics": "Results; Supplementary figure `fig:xenium_qc_confound`",
    "snrna_populations_top50": "Results",
    "cnmf_program_caste": "Results; figure `fig:caste_de`, panel D",
    "regional_caste_de": "Results",
    "glia_caste_de_top10": (
        "Supplementary figure `fig:caste_de_volcanoes`, panel B; Supplementary Methods"
    ),
    "snrna_markers_top500": "Supplementary Methods",
    "spatial_markers_top500": "Supplementary Methods",
    "spatial_measured_gene_scores": "Supplementary Methods",
    "snrna_caste_degs": "Results",
    "snrna_caste_gsea_celltypes": "Results; Methods",
    "egapx_rnaseq_samples": "Methods",
    "egapx_isoseq_flnc": "Methods",
    "snrna_library_qc": "Methods",
    "cnmf_program_go_terms": "Methods",
}


def readme(out: dict[str, pd.DataFrame], rd: pd.DataFrame) -> str:
    n = {k: len(v) for k, v in out.items()}
    sdeg = out["snrna_caste_degs"]
    n_deg_genes_text = sdeg[sdeg.cell_type != "Photoreceptors"].gene_id.nunique()
    pk = out["xenium_panel_key"]
    in_ref = int((pk.in_snrna_reference == "yes").sum())
    n_names = pk.symbol.str.replace(r" \(probe \d\)$", "", regex=True).nunique()
    npop = out["snrna_populations_top50"].population.nunique()
    qc = out["xenium_qc_metrics"]
    nreg = int((qc.caste != "all").sum())
    g10 = out["glia_caste_de_top10"]
    n_unann = int((g10.description == "").sum())
    n_panel = int(g10.on_xenium_panel.sum())
    glia_panel = (
        "None is on the 101-gene Xenium panel."
        if n_panel == 0
        else f"{WORDS[n_panel].capitalize()} of the ten are on the 101-gene panel."
    )
    orth_panel = (
        "`dmel_ortholog_symbol`, `dmel_flybase_id`, `dmel_ortholog_method`: the panel key's "
        f"curated call ({sd('xenium_panel_key')}). `NCBI ortholog`: NCBI Gene ortholog "
        "assignment; `reciprocal best hit`: DIAMOND blastp reciprocal best hit, e-value "
        "<= 1e-10, >= 30% query coverage, not tied with a paralog, B. impatiens RefSeq "
        "BIMP_2.2 against the D. melanogaster UniProt proteome; `two-source consensus`: at "
        "least two independent lines (Ensembl Metazoa Compara; OrthoDB v11 single-member "
        "group; DIAMOND --ultra-sensitive reciprocal best hit at e-value <= 1e-5; matching "
        "gene name plus significant alignment) name the same gene and none contradicts it. "
        "Blank: no ortholog call, including vitellogenin and *Dnmt1a*, which *Drosophila* "
        "lacks."
    )
    detail = {
        "xenium_panel_key": (
            [
                f"Each of the 101 probes. {in_ref} of 101 probes match a gene in the snRNA-seq "
                "reference; corazonin does not. Two probes are annotated as fatty acid "
                "synthase and share one Xenium `gene_name`; they are different genes "
                f"(distinct BIMP_v4.3 IDs) and are listed separately, so the panel has 101 "
                f"probes but {n_names} distinct gene names.",
            ],
            [
                "`symbol`: paper or NCBI annotation symbol, otherwise the NCBI protein name "
                "(`symbol_source` says which).",
                "`gene_id`: BIMP_v4.3 gene. `xenium_probe_id`: BIMP_vNP3.1 panel ID. "
                "`ncbi_gene`: RefSeq LOC ID. `ncbi_protein_name`: NCBI protein name.",
                orth_panel,
                "`category`, `relevance`: panel-design annotation recorded for each probe "
                "(`category` joins `anatomical_marker` and `molecular_class`); "
                "`circuit_sender`, `circuit_receiver`: the same design sheet's circuit "
                "annotation. Blank where none was recorded.",
                "`in_snrna_reference`: yes when the gene is in the snRNA-seq reference.",
                "`probe_note`: marks the two fatty acid synthase probes.",
            ],
        ),
        "xenium_qc_metrics": (
            [
                f"One row per capture region ({nreg}: 16 per brain, two slides per brain, "
                "each brain in its own instrument run), with the fields of the Xenium Ranger "
                "`metrics_summary.csv` written for that region, sorted by caste and region; "
                f"the last row (`caste` = `all`) is the median of each field over the {nreg} "
                "regions. Caste is confounded with instrument run (Supplementary figure "
                "`fig:xenium_qc_confound`, panel C).",
            ],
            [
                "`caste`: queen, worker, or `all` for the median row. `slide`: Xenium slide "
                "serial. `capture_region`: the Xenium Ranger output directory of the region.",
                "`median_transcripts_per_cell`, `median_genes_per_cell`, "
                "`num_cells_detected`, `adjusted_negative_control_probe_rate`, "
                "`adjusted_negative_control_codeword_rate`, "
                "`negative_control_probe_counts_per_control_per_cell`, "
                "`estimated_number_of_false_positive_transcripts_per_cell`, "
                "`fraction_transcripts_decoded_q20`, `fraction_transcripts_assigned`, "
                "`decoded_transcripts_per_100um2`: as defined by Xenium Ranger (rates and "
                "fractions are proportions, not percentages).",
                "`false_positives_pct_of_median_transcripts`: 100 x estimated false-positive "
                "transcripts per cell / median transcripts per cell.",
            ],
        ),
        "snrna_populations_top50": (
            [
                f"The {npop} transcriptionally distinct populations of the combined "
                "*B. impatiens* snRNA-seq reference (central brain and optic lobe), one row per "
                "population and marker rank. Markers: Scanpy `rank_genes_groups`, Wilcoxon, one "
                "population versus the rest, ranked by score. OL_1 to OL_13 are unannotated "
                "optic-lobe subclusters.",
            ],
            [
                "`population_id`, `population`, `pooled_group`: population, numbered in the "
                "paper's order, and its seven-group pooling.",
                "`nuclei_queen`, `nuclei_worker`: nuclei per caste. `panel_genes_in_top50`: "
                "Xenium panel genes among the population's 50 top markers. These three repeat "
                "on each of a population's rows.",
                "`rank`: 1 = highest Wilcoxon score.",
                GENE,
                ORTH_RBH,
                "`wilcoxon_score`, `log2_fold_change`, `adjusted_p_value`, `on_xenium_panel`: "
                f"as in {sd('snrna_markers_top500')}.",
            ],
        ),
        "cnmf_program_caste": (
            [
                "Joint consensus NMF (k = 50) of the central brain snRNA-seq nuclei. For each "
                "cell type and trial, the effect is the queen minus worker difference in mean "
                "program usage (cell type x trial x caste groups of at least 30 nuclei), "
                "divided by the standard deviation of usage within the cell type, so positive "
                "means higher in queens. With three within-trial queen-worker pairs, caste "
                "can be relabeled within trials in only eight ways, so the smallest "
                "attainable permutation P is 0.25 and the calls below are descriptive. "
                "A program x cell type is a replicated call when the "
                "effect has one sign in all three trials, |mean| >= 0.5 SD, the weakest trial "
                f">= 0.25 SD and mean usage >= 2%: {int(out['cnmf_program_caste'].replicated.sum())} "
                "calls (stars in panel D of `fig:caste_de`). One row per program and "
                "cell type.",
            ],
            [
                "`cell_type`, `program`: cell type and cNMF program (numbered as in panel D "
                "of `fig:caste_de`).",
                "`effect_trial_1`, `effect_trial_2`, `effect_trial_3`: queen - worker usage "
                "difference in SD units, per trial. `mean_effect`: their mean. "
                "`weakest_trial_abs_effect`: smallest absolute trial effect. "
                "`same_sign_all_trials`: the three trials agree in sign.",
                "`usage_pct`: mean usage of the program in the cell type, percent.",
                "`replicated`: meets the replication rule above.",
                "`program_home_cell_type`, `program_home_fraction`: cell type holding the "
                "largest share of the program's total usage, and that share.",
                "`program_label`, `program_label_go_id`, `program_label_go_fdr`: row label in "
                "panel D of `fig:caste_de`, the top GO biological-process term by GSEA at "
                "FDR < 0.05; a descriptive label (marked with a dagger) where no term passes. "
                "Blank for programs not in that panel.",
            ],
        ),
        "regional_caste_de": (
            [
                f"All {n['regional_caste_de']} rows (6 compared regions x 101 genes). "
                "Expression is the mean log-normalised value over a region's cells in each "
                "section (sections with at least five cells in the region). The log2 fold "
                "change is log2((worker + 0.01)/(queen + 0.01)), so positive means higher in "
                "workers, the reverse of the snRNA-seq DESeq2 contrasts (queen/worker). The "
                "slice consistency is the rank-biserial correlation of the queen and worker "
                "section means (from the Mann-Whitney U statistic): +1 when every worker "
                "section is above every queen section, -1 for the reverse, so genes whose "
                "sections separate completely tie at +1 or -1. There is one brain per caste, "
                "so sections are pseudoreplicates of one animal: these are effect sizes, and "
                "no P value is computed. Rows are ranked by absolute slice consistency and "
                "then by absolute log2 fold change. The antennal lobe and central complex "
                "are not compared.",
            ],
            [
                "`region`, `rank_in_region`: region and the gene's rank in it (absolute "
                "slice consistency, then absolute log2 fold change).",
                "`gene_id`: BIMP_v4.3 gene. `xenium_probe_id`: BIMP_vNP3.1 panel ID. "
                f"`symbol`, `description`: as in {sd('xenium_panel_key')}.",
                orth_panel,
                "`log2_fold_change_worker_over_queen`: see above (positive = higher in "
                "workers). `slice_consistency`: rank-biserial r of the section means, "
                "positive = worker sections higher.",
                "`mean_expression_queen`, `mean_expression_worker`: mean over sections of "
                "the per-section mean log-normalised expression.",
                "`sections_queen`, `sections_worker`, `cells_queen`, `cells_worker`: "
                "sections and cells contributing per caste.",
            ],
        ),
        "glia_caste_de_top10": (
            [
                "Pooled glia of the central brain snRNA-seq reference, pseudobulk DESeq2 on "
                "the three queen and three worker pooled samples (design ~ trial + caste, "
                "Wald test, Benjamini-Hochberg FDR < 0.05): the ten genes with the smallest "
                "adjusted P. Positive log2 fold change is higher in queens. "
                f"{WORDS[n_unann].capitalize()} of the ten have no product annotation. "
                f"{glia_panel}",
            ],
            [
                "`rank`: 1 = smallest adjusted P.",
                "`gene_id`: *B. impatiens* gene, BIMP_v4.3 annotation. `description`: "
                "product description of the v4.3 annotation, blank where none. "
                "`gene_biotype`: v4.3 gene biotype.",
                ORTH_RBH,
                "`higher_in`: caste with the higher expression. "
                "`log2_fold_change_queen_over_worker`: DESeq2 log2 fold change, queen/worker "
                "(three decimals). `adjusted_p_value`: Benjamini-Hochberg.",
                "`on_xenium_panel`: gene is one of the 101 Xenium panel genes.",
            ],
        ),
        "snrna_markers_top500": (
            [
                "Scanpy `rank_genes_groups`, Wilcoxon rank-sum test of each type against all "
                "other nuclei of the combined central-brain and optic-lobe reference (queen and "
                "worker pooled; *Optic lobe* pools the 13 unannotated subclusters OL_1 to "
                "OL_13). Ranked by Wilcoxon score, which grows with the number of nuclei; "
                "Cohen's d is the comparable effect size. Most top markers are not on the "
                "Xenium panel.",
            ],
            [
                "`cell_type`, `pooled_group`: atlas type and its seven-group pooling.",
                "`rank`: 1 = highest Wilcoxon score.",
                GENE,
                ORTH_RBH,
                "`wilcoxon_score`: Scanpy Wilcoxon z-score. `log2_fold_change`: Scanpy "
                "log2 fold change, type versus rest. `adjusted_p_value`: Benjamini-Hochberg "
                "(0 means below double precision). `cohens_d`: difference in mean "
                "normalised expression in units of the pooled standard deviation.",
                "`on_xenium_panel`: gene is one of the 101 Xenium panel genes.",
            ],
        ),
        "spatial_markers_top500": (
            [
                "Each region's cells against the other cells traced for that region (queen and "
                "worker pooled; 1,300,192 cells). Only the 100 panel genes present in the "
                "projected object are measured; every other gene is inferred by projecting "
                "the snRNA-seq reference onto each cell through its Tangram cell type mapping, "
                "so an inferred gene reports what the cell types of a region express in the "
                "reference and is not an independent spatial measurement. No P values: cells "
                "are not independent and each caste is one brain. The central complex is "
                "descriptive (its traced cells are almost entirely glia).",
            ],
            [
                "`region`: traced region (the three Kenyon cell compartments are named by "
                "the Kenyon cell type they hold).",
                "`rank`: 1 = largest Cohen's d.",
                GENE,
                ORTH_RBH,
                "`basis`: `Measured` (Xenium panel gene) or `Inferred` (Tangram projection).",
                "`cohens_d`: region versus elsewhere, pooled standard deviation. "
                "`log2_fold_change`: log2((mean in region + 0.001)/(mean elsewhere + 0.001)). "
                "`mean_expression_in_region`, `mean_expression_elsewhere`: mean normalised "
                "expression per cell.",
            ],
        ),
        "spatial_measured_gene_scores": (
            [
                f"The measured genes of {sd('spatial_markers_top500')} without the top-500 "
                "cut: every region by every panel gene present in the projected object, same "
                "Cohen's d.",
            ],
            [
                f"`region`: as in {sd('spatial_markers_top500')}.",
                "`rank_among_measured`: rank of the gene among the 100 measured genes in "
                "that region (1 = largest Cohen's d).",
                GENE,
                ORTH_RBH,
                f"`cohens_d`: as in {sd('spatial_markers_top500')}.",
            ],
        ),
    }
    saved_cols = "Column names and order as written by the analysis; numbers at full precision."
    detail |= {
        "snrna_caste_degs": (
            [
                f"{n['snrna_caste_degs']} gene x cell type rows "
                f"({out['snrna_caste_degs'].gene_id.nunique()} genes): every gene with "
                "Benjamini-Hochberg adjusted P < 0.05 in the pseudobulk DESeq2 fit "
                "(design ~ trial + caste, Wald test) of a cell type, sorted by cell type "
                "and adjusted P. Positive log2 fold change is higher in queens. Hemocytes "
                "have none. From the PyDESeq2 0.5.4 fit used for `fig:caste_de` and the "
                f"text. The gene total quoted in the text ({n_deg_genes_text:,}) leaves out "
                "photoreceptors, whose rows are flagged.",
            ],
            [
                "`cell_type`, `gene_id`: cell type and BIMP_v4.3 gene.",
                "`gene_name`, `description`, `display_name`, `gene_biotype`, "
                "`chromosome`: BIMP_v4.3 annotation (`display_name`: name, else "
                "description, else ID).",
                "`direction`: queen-up or worker-up. `baseMean`, `log2FoldChange` "
                "(queen/worker), `lfcSE`, `stat`, `pvalue`, `padj`: DESeq2.",
                "`flag`: `photoreceptors` for the photoreceptor rows (residual optic lobe "
                "contamination; not drawn in panel B of `fig:caste_de` and left out of the "
                "gene total in the text); `Vgn, no P value reported` for vitellogenin "
                "(BIMP_v4.3_005500), whose reads come mostly from a few fat-body nuclei "
                "in the queen samples, so the text gives no P value for it (Supplementary "
                "Methods); blank otherwise.",
                saved_cols,
            ],
        ),
        "snrna_caste_gsea_celltypes": (
            [
                f"{n['snrna_caste_gsea_celltypes']} rows: for each of 11 glial and "
                "neuronal types, preranked GSEA (GSEApy, 1,000 permutations, seed 0) of "
                "genes with baseMean > 10 ranked by caste Wald statistic, against GO gene "
                "sets of 10-500 genes (all three namespaces), every term tested. Terms "
                "whose GO ID is secondary in the ontology file have a blank term and "
                "namespace.",
            ],
            [
                "`cell_type`; `direction`: queen-up (NES > 0) or worker-up.",
                "`go_id`, `term`, `namespace` (BP, MF, CC).",
                "`NES`, `nominal_p`, `FDR_q`, `FWER_p`, `tag_pct`, `gene_pct`: GSEApy. "
                "`n_ranked`: genes ranked. `leading_edge_genes`: BIMP_v4.3 IDs.",
                saved_cols,
            ],
        ),
        "egapx_rnaseq_samples": (
            [
                "The RNA-seq libraries given to EGAPx for the BIMP_v4.3 annotation, in "
                "the order of the run's `short_reads` input.",
            ],
            [
                "`#`: row number. `Source`: NCBI SRA or in-house. `Sample`: SRA run or "
                "library name. `R1 file`, `R2 file`: read files.",
            ],
        ),
        "egapx_isoseq_flnc": (
            [
                "The PacBio Iso-Seq full-length non-concatemer (FLNC) read sets given "
                "to EGAPx (its `long_reads` input): barcodes 1 and 2 of one Revio SMRT "
                "cell (library Pool001_IT_KN), demultiplexed and refined by SMRT Link "
                "25.1 and converted to FASTA for EGAPx. The cell's other five barcodes "
                "were not used.",
            ],
            [
                "`egapx_set`, `egapx_input_fasta`: name and file name in the EGAPx input. "
                "`smrt_link_flnc_bam`: the SMRT Link output it was converted from.",
                "`pacbio_bio_sample`, `barcode`, `library`, `movie`, `run_id`, "
                "`smrt_cell_id`: from the BAM read group and the Iso-Seq primer report.",
                "`segmented_reads_with_primers`, `flnc_reads`, `flnc_polya_reads`: "
                "reads with both primers, non-concatemer reads, and those with a "
                "poly(A) tail (the FLNC set), from the primer report.",
            ],
        ),
        "snrna_library_qc": (
            [
                "One row per snRNA-seq library (three trials x two castes of central "
                "brain, one optic lobe library per caste): the Cell Ranger multi "
                "`metrics_summary.csv` cell metrics and the CellBender remove-background "
                "run used for the atlas.",
            ],
            [
                "`Tissue`, `Run`, `Sample ID`, `Caste`.",
                "`Cells` to `Total genes detected`: Cell Ranger metrics (fractions as "
                "proportions).",
                "`Cellbender_Notes`: CellBender settings (half learning rate, forced "
                "cell number, or default).",
            ],
        ),
        "cnmf_program_go_terms": (
            [
                f"{n['cnmf_program_go_terms']} rows: for each joint cNMF program "
                "(k = 50), preranked GSEA of the program gene scores against all GO gene "
                "sets (1,000 permutations, seed 0), positive-NES terms at FDR < 0.25, "
                "obsolete and unnamed terms removed.",
            ],
            [
                "`program`; `in_figure`: shown in panel D of `fig:caste_de` (program 26 is "
                "also flagged); `replicated_caste_effect`: cell types where the program "
                "meets the replicated caste call (see `cnmf_program_caste.csv`).",
                "`namespace`, `go_id`, `term`, `NES`, `nominal_p`.",
                "`FDR_all_GO_library`: FDR across all GO terms; `FDR_BP_only_library`: "
                "the same term against biological-process sets only (figure programs "
                "only). `tier`: FDR < 0.05 or 0.05-0.25.",
                "`tag_pct`, `gene_pct`, `n_leading_edge`, `leading_edge_genes` (gene "
                "names, else BIMP_v4.3 IDs without prefix).",
            ],
        ),
    }
    assert set(detail) == set(NAME) == set(TITLE) == set(CITED)
    L = [
        "# Supplementary Data",
        "",
        "Comma-separated, UTF-8, one header row, no LaTeX markup. Generated by "
        "`scripts/deposit/supplementary_data_csvs.py` in "
        "https://github.com/kocherlab/bombus-impatiens-brain-atlas "
        "from the outputs of the analysis scripts named there; do not edit by hand. "
        "Files are listed in order of first citation, the order of the Supplementary "
        "Information's Supplementary Data list. File names carry no number: LaTeX "
        "numbers each item from its label, and `scripts/deposit/export_numbered_si.py` "
        "writes numbered copies from a built manuscript.",
        "",
        "| File | Label | Description | Cited in |",
        "|---|---|---|---|",
        *[f"| `{NAME[s]}` | `{LABEL[s]}` | {TITLE[s]}. | {CITED[s]} |" for s in NAME],
        "",
    ]
    for s in NAME:
        about, cols = detail[s]
        L += [
            f"## `{NAME[s]}` (`{LABEL[s]}`)",
            "",
            f"{TITLE[s]}. {n[s]} rows. Cited in: {CITED[s]}.",
            "",
            *about,
            "",
            "Columns:",
            "",
            *[f"- {c}" for c in cols],
            "",
        ]
    return "\n".join(L)


if __name__ == "__main__":
    with explain_missing_inputs():
        main()
