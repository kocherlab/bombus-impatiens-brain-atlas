"""Single source of truth: every shipped manuscript item, its generator and paths.

Pure data + pure functions, no side effects at import time. Imported by
`verify_manuscript_figures.py` (byte/registry checks), `sync_manuscript_figures.py`
(copies results/ -> manuscript), `deposit/export_numbered_si.py` (journal-ready
numbered copies), `deposit/supplementary_data_csvs.py` and the figure Snakefile
(`scripts/paper/Snakefile`).

Names carry content slugs, never printed numbers: LaTeX numbers every item
(figures and tables by their float counters, Supplementary Data by \\suppdata),
so a reordering in the paper renames nothing here.

FIGURES  -- one Figure per shipped figure file: slug, LaTeX label, kind
            ("main" or "si"), generator, the results/ stem it renders to, and the
            formats that ship. The manuscript path follows from kind + slug:
              main  figures/main/fig_<slug>.<ext>   results/figures/main/<slug>/
              si    figures/si/figS_<slug>.<ext>    results/figures/si/
            Format policy: a figure ships as PDF + PNG when its generator writes
            a PDF of the same artwork (LaTeX's extension-less \\includegraphics
            picks the PDF; the PNG is the preview), otherwise as PNG. PNG_ONLY
            lists the figures whose generator also writes a PDF that does not
            match the shipped PNG.
TABLES   -- Supplementary tables (tables/<slug>.tex): label and generator, or
            None where the table is written by hand in the manuscript.
SUPPLEMENTARY_DATA_ITEMS -- Supplementary Data CSVs (supplementary_data/<slug>.csv),
            in order of first citation (the order of the SI's \\suppdata list,
            which is what numbers them).
SCRIPTS  -- derived view (manuscript path -> generator) for the Snakefile and
            the figure check.
NOT_REPRODUCIBLE -- documented exceptions.
"""

from __future__ import annotations

import ast
import os
import pathlib
import sys
from typing import NamedTuple

REPO = pathlib.Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.append(str(REPO / "src"))
from bimp_st.paths import CASTE_DE_SAVED_INPUTS  # noqa: E402

MANUSCRIPT = pathlib.Path(
    os.environ.get(
        "BIMP_MANUSCRIPT", REPO / "submodules" / "bombus_impatiens_atlas_manuscript"
    )
)


class Figure(NamedTuple):
    slug: str
    label: str  # LaTeX \label of the float that includes it
    kind: str  # "main" or "si"
    generator: str  # repo-relative script that renders it
    formats: tuple[str, ...] = ("png",)
    results_stem: str | None = None  # override; default from kind + slug

    @property
    def manuscript_stem(self) -> str:
        if self.kind == "main":
            return f"figures/main/fig_{self.slug}"
        return f"figures/si/figS_{self.slug}"

    @property
    def results(self) -> str:
        if self.results_stem:
            return self.results_stem
        if self.kind == "main":
            return f"results/figures/main/{self.slug}/fig_{self.slug}"
        return f"results/figures/si/figS_{self.slug}"

    def files(self) -> list[tuple[str, str]]:
        """(manuscript path, results path) for every shipped format."""
        return [
            (f"{self.manuscript_stem}.{e}", f"{self.results}.{e}") for e in self.formats
        ]


PDF = ("pdf", "png")
# The generator writes a PDF too, but the PDF beside the shipped PNG is not the same
# artwork (text weight and label placement differ when the page is rendered),
# so only the PNG ships; re-render both and compare before adding it.
PNG_ONLY = ("png",)

# Main figures, in printed order (LaTeX numbers them 1-6).
_MAIN = [
    Figure(
        "atlas_overview",
        "fig:atlas_overview",
        "main",
        "scripts/atlas/fig_atlas_overview.py",
    ),
    Figure(
        "xenium_celltype_overlay",
        "fig:xenium_celltype_overlay",
        "main",
        "scripts/atlas/fig_xenium_celltype_overlay.py",
        PNG_ONLY,
    ),
    # caste_comparison_sccoda.py is the analysis step (ANALYSIS_PREDECESSOR); the
    # composer renders the shipped figure.
    Figure(
        "caste_comparison",
        "fig:caste_comparison",
        "main",
        "scripts/regions/fig_caste_comparison.py",
        PDF,
    ),
    Figure("caste_de", "fig:caste_de", "main", "scripts/caste_de/fig_caste_de.py", PDF),
    Figure(
        "neuropeptide_spatial",
        "fig:neuropeptide_spatial",
        "main",
        "scripts/neuropeptides/fig_neuropeptide_spatial.py",
    ),
    # Rendered by fig_kc_subtypes.py.
    Figure(
        "kc_subtypes",
        "fig:kc_subtypes",
        "main",
        "scripts/kenyon/fig_kc_subtypes.py",
        PDF,
    ),
]

# Supplementary figures, in printed order.
_SI = [
    # Lower half (panels A-C) of si_xenium_run_qc.py's tall fig_s_xenium_qc.png,
    # cut by split_tall_si_figures.py; the unlettered upper half is Supplementary Data
    # xenium_qc_metrics. A re-split is pixel-identical, not byte-identical (PIL).
    Figure(
        "xenium_qc",
        "fig:xenium_qc_confound",
        "si",
        "scripts/paper/split_tall_si_figures.py",
    ),
    Figure(
        "exploded_regions",
        "fig:exploded_regions",
        "si",
        "scripts/regions/si_exploded_regions.py",
    ),
    # `figure` mode redraws from leiden_stability.csv
    # (pipelines/analyses/atlas_analyses.smk).
    Figure(
        "leiden_stability",
        "fig:leiden_stability",
        "si",
        "scripts/atlas/si_leiden_stability.py",
        PNG_ONLY,
    ),
    # fig:panel_gene_recovery holds two files.
    Figure(
        "probe_clusters",
        "fig:panel_gene_recovery",
        "si",
        "scripts/identity/si_probe_clusters.py",
    ),
    Figure(
        "sn_panel_umap",
        "fig:panel_gene_recovery",
        "si",
        "scripts/atlas/si_snrna_panel_umap.py",
    ),
    Figure(
        "tangram_validation",
        "fig:tangram_qc",
        "si",
        "scripts/identity/si_tangram_validation.py",
    ),
    # The channel strip (fig_s_xenium_channels, not shipped) is panels A-B of this.
    Figure(
        "xenium_kc_populations",
        "fig:xenium_kc_populations",
        "si",
        "scripts/atlas/fig_xenium_celltype_overlay.py",
    ),
    Figure(
        "tangram_confidence",
        "fig:tangram_confidence",
        "si",
        "scripts/identity/si_tangram_confidence.py",
    ),
    Figure(
        "percell_vs_vote",
        "fig:percell_vs_vote",
        "si",
        "scripts/identity/si_percell_vs_vote.py",
    ),
    Figure(
        "atlas_quality",
        "fig:atlas_quality",
        "si",
        "scripts/atlas/si_atlas_quality.py",
        PDF,
    ),
    Figure(
        "panel_concordance",
        "fig:panel_concordance",
        "si",
        "scripts/atlas/si_panel_concordance.py",
    ),
    Figure(
        "region_prediction",
        "fig:region_prediction",
        "si",
        "scripts/regions/si_region_prediction.py",
        PDF,
    ),
    Figure(
        "reference_composition",
        "fig:reference_composition",
        "si",
        "scripts/identity/si_reference_composition.py",
        PNG_ONLY,
    ),
    Figure(
        "caste_de_volcanoes",
        "fig:caste_de_volcanoes",
        "si",
        "scripts/caste_de/si_caste_de_volcanoes.py",
        PDF,
    ),
    # Only panels A-C ship; the 29 x 29 confusion matrix (_a) does not.
    Figure(
        "label_sensitivity",
        "fig:label_sensitivity",
        "si",
        "scripts/identity/si_label_sensitivity.py",
    ),
    Figure(
        "regional_volcanoes",
        "fig:regional_volcanoes",
        "si",
        "scripts/caste_de/si_regional_degs.py",
    ),
    # Panel F of the neuropeptide figure, rendered standalone by its generator.
    Figure(
        "ap_gradient",
        "fig:ap_gradient",
        "si",
        "scripts/neuropeptides/fig_neuropeptide_spatial.py",
    ),
    Figure(
        "multimodal_tk",
        "fig:multimodal_tk",
        "si",
        "scripts/neuropeptides/si_tk_multimodal.py",
    ),
    Figure(
        "focal_vg_krh1",
        "fig:focal_genes",
        "si",
        "scripts/subcellular/si_focal_genes.py",
    ),
    Figure(
        "subcellular_regions",
        "fig:subcellular",
        "si",
        "scripts/subcellular/si_subcellular_summary.py",
        PDF,
    ),
    Figure(
        "deg_concordance",
        "fig:deg_concordance",
        "si",
        "scripts/caste_de/si_deg_concordance.py",
        PDF,
    ),
    Figure(
        "tk_3d_spillover",
        "fig:tk_3d_spillover",
        "si",
        "scripts/neuropeptides/si_tk_3d_spillover.py",
    ),
    # Pooled glia vote, generic optic-lobe Kenyon entry excluded (the default).
    Figure(
        "neuropeptide_landscape",
        "fig:neuropeptide_landscape",
        "si",
        "scripts/neuropeptides/si_neuropeptide_landscape.py",
        PDF,
    ),
    Figure(
        "snuc_kc_subclusters",
        "fig:snuc_kc_subclusters",
        "si",
        "scripts/kenyon/snrna_kc_subclusters.py",
        PNG_ONLY,
    ),
    # BIMP_ALSO_PDF=1 writes the PDF beside the PNG (si_common.save).
    Figure(
        "mosaicfield",
        "fig:mosaicfield_si",
        "si",
        "scripts/atlas/si_mosaicfield.py",
        PDF,
    ),
    Figure(
        "kc_projection_null",
        "fig:kc_projection_null",
        "si",
        "scripts/kenyon/si_kc_projection_null.py",
        PDF,
    ),
    # Region x region Bray-Curtis beta diversity (si_hill_diversity.py).
    Figure(
        "beta_diversity",
        "fig:beta_diversity",
        "si",
        "scripts/regions/si_hill_diversity.py",
        PDF,
    ),
]

FIGURES: list[Figure] = _MAIN + _SI


class Table(NamedTuple):
    slug: str
    label: str
    generator: str | None  # None: written by hand in the manuscript
    results: str | None = None  # the generator's .tex output, byte-compared


TABLES: list[Table] = [
    Table(
        "mb_ol_labels",
        "tab:mb_ol_labels",
        "scripts/kenyon/table_mb_ol_kc_labels.py",
        "results/round6/mb_ol/mb_ol_labels.tex",
    ),
    Table("label_sources", "tab:label_sources", None),
]


class DataItem(NamedTuple):
    slug: str  # file supplementary_data/<slug>.csv
    label: str  # the \suppdata label that numbers it


# Supplementary Data, in order of first citation (the article incl. Methods, then the
# Supplementary Information, then Supplementary Methods); the SI's \suppdata list
# must follow this order, and verify_manuscript_figures.py checks both against the
# first \ref of each label. One generator writes them all.
SUPPLEMENTARY_DATA_SCRIPT = "scripts/deposit/supplementary_data_csvs.py"
SUPPLEMENTARY_DATA_ITEMS: list[DataItem] = [
    DataItem("xenium_panel_key", "sd:panel_key"),
    DataItem("xenium_qc_metrics", "sd:xenium_qc"),
    DataItem("snrna_populations_top50", "sd:snrna_annotations"),
    DataItem("regional_caste_de", "sd:regional_caste_degs"),
    DataItem("snrna_caste_degs", "sd:snrna_caste_degs"),
    DataItem("cnmf_program_caste", "sd:cnmf_program_caste"),
    DataItem("snrna_caste_gsea_celltypes", "sd:caste_gsea_celltypes"),
    DataItem("snrna_library_qc", "sd:snrna_library_qc"),
    DataItem("cnmf_program_go_terms", "sd:cnmf_program_go"),
    DataItem("egapx_rnaseq_samples", "sd:egapx_samples"),
    DataItem("egapx_isoseq_flnc", "sd:egapx_isoseq"),
    DataItem("glia_caste_de_top10", "sd:glia_caste_degs"),
    DataItem("snrna_markers_top500", "sd:markers_snrna"),
    DataItem("spatial_markers_top500", "sd:markers_spatial"),
    DataItem("spatial_measured_gene_scores", "sd:markers_spatial_measured"),
]
SUPPLEMENTARY_DATA_FILES = tuple(f"{d.slug}.csv" for d in SUPPLEMENTARY_DATA_ITEMS)
SUPPLEMENTARY_DATA = {
    f"supplementary_data/{name}": f"results/supplementary_data/{name}"
    for name in (*SUPPLEMENTARY_DATA_FILES, "SUPPLEMENTARY_DATA.md")
}

# ── Derived views ────────────────────────────────────────────────────────────
# manuscript path (relative to the manuscript root) -> generator, one entry per
# shipped file (PNG and, where it ships, PDF).
SCRIPTS: dict[str, str] = {m: f.generator for f in FIGURES for m, _ in f.files()}

# Figures whose render is known not to be byte-reproducible, with the reason.
NOT_REPRODUCIBLE: dict[str, str] = {
    "figures/si/figS_xenium_qc.png": (
        "a crop written by PIL; a re-split of results/figures/si/fig_s_xenium_qc.png "
        "is pixel-identical to the shipped file (checked 2026-11-17), not "
        "byte-identical"
    ),
}

# Analysis + compose pairs: caste_comparison_sccoda.py writes the results/ CSVs that
# fig_caste_comparison.py plots; si_xenium_run_qc.py renders the tall QC figure that
# split_tall_si_figures.py crops. `declared_inputs()` excludes results/ paths, so this
# script -> script edge is declared explicitly. Other results-to-results edges
# are not traced.
ANALYSIS_PREDECESSOR = {
    "scripts/regions/fig_caste_comparison.py": (
        "scripts/regions/caste_comparison_sccoda.py"
    ),
    "scripts/paper/split_tall_si_figures.py": "scripts/atlas/si_xenium_run_qc.py",
}

# Further analysis steps a generator needs, run in this order after its
# ANALYSIS_PREDECESSOR: fig_caste_comparison.py also plots the composition tables
# caste_comparison_tables.py writes from caste_comparison_sccoda.py's output, and
# panel C's per-region table that region_composition.py writes.
PRE_STEPS: dict[str, tuple[str, ...]] = {
    "scripts/regions/fig_caste_comparison.py": (
        "scripts/regions/caste_comparison_tables.py",
        "scripts/regions/region_composition.py",
    ),
}


def predecessors(script: str) -> list[str]:
    """The analysis steps that must run before ``script``, in run order."""
    first = [ANALYSIS_PREDECESSOR[script]] if script in ANALYSIS_PREDECESSOR else []
    return first + list(PRE_STEPS.get(script, ()))

# Generators whose real input is a results/ file produced by another Snakefile
# (not by a figure generator). scripts/paper/Snakefile adds these repo-relative paths
# to the figure rule's `input:`, so a missing file fails the dry-run. The
# producing pipeline is named per entry.
ANALYSIS_RESULTS_INPUT = {
    # pipelines/analyses/atlas_analyses.smk: leiden_ref + leiden_seed x20
    # -> leiden_aggregate -> leiden_figure. The generator (no argument) redraws
    # the PNG/PDF from the aggregated CSV.
    "scripts/atlas/si_leiden_stability.py": [
        "results/handoff/A_leiden_stability/leiden_stability.csv",
    ],
    # Caste DE by class and subtype, the ~ trial + caste fit:
    # caste_de_blocked_all.py (.venv-deseq2); genome_dmel_orthologs.py (symbols).
    "scripts/caste_de/si_caste_de_volcanoes.py": [
        "results/caste_de/three_trial_pairs/de_coarse_blocked.csv",
        "results/caste_de/three_trial_pairs/de_fine_blocked.csv",
        "results/round5/glia_labels/v43_dmel_rbh.csv.gz",
    ],
    # The same fit feeds the neuropeptide figure's panel C and the DEG
    # concordance figure (tables in results/caste_de/deg_concordance).
    "scripts/neuropeptides/fig_neuropeptide_spatial.py": [
        "results/caste_de/de_coarse_blocked.csv",
    ],
    "scripts/caste_de/si_deg_concordance.py": [
        "results/caste_de/three_trial_pairs/de_coarse_blocked.csv",
    ],
    # fig:caste_de: trial-blocked fine-type DE from caste_de_blocked_all.py
    # (.venv-deseq2); panel d (joint cNMF k = 50, caste table, GSEA program
    # labels) from the in-repo port of the cNMF / GSEA analysis,
    # `bash scripts/caste_de/cnmf_gsea/run_pipeline.sh all` (freeze ->
    # inputs -> prepare -> factorize -> caste -> map -> null/go -> fig3inputs;
    # parity in pipeline/parity/); gene annotation frozen by
    # caste_de_freeze_saved_outputs.py
    "scripts/caste_de/fig_caste_de.py": [
        "results/caste_de/three_trial_pairs/de_fine_blocked.csv",
        str((CASTE_DE_SAVED_INPUTS / "BIMP_lookup.csv").relative_to(REPO)),
        "results/caste_programs/pipeline/fig3_inputs/cnmf_k50_fig_usage.csv.gz",
        (
            "results/caste_programs/pipeline/fig3_inputs/"
            "cnmf_k50_fig_gene_scores.csv.gz"
        ),
        "results/caste_programs/pipeline/fig3_inputs/joint_k50_caste.csv",
        "results/caste_programs/pipeline/fig3_inputs/program_labels.csv",
    ],
    # Subcellular localization across regions: ELLA summaries, bento region
    # profiles and the adjusted odds ratios, written by scripts/subcellular/.
    "scripts/subcellular/si_subcellular_summary.py": [
        "results/bento/regions_summary/region_profiles.csv",
        "results/bento/regions_summary/size_control_glm.csv",
        "results/bento/differential/pergene_caste_stats.csv",
        "results/bento/differential/region_gene_means.csv",
        "results/ella/summary/group_summary.csv",
        "results/ella_regions/summary/group_summary.csv",
    ],
    # The composition figure's two levels, written by caste_comparison_tables.py
    # (run after caste_comparison_sccoda.py, before fig_caste_comparison.py) into the
    # analysis
    # directory results/figures/fig2/.
    "scripts/regions/fig_caste_comparison.py": [
        "results/figures/fig2/sn_composition_by_caste.csv",
        "results/figures/fig2/atlas_vote_composition_by_caste.csv",
    ],
    # MOSAICField (panels E-F): the chunked nonlinear alignment
    # (pipelines/wholebrain/queen/run_mosaicfield.py --slices all --chunk-size 6
    # --output-dir results/mosaicfield_chunked), stitched into one 3D frame by
    # scripts/atlas/mosaicfield_stitch.py (run order in its docstring).
    "scripts/atlas/si_mosaicfield.py": [
        "results/mosaicfield_stitched/stitched_coords_3d.npy",
        "results/mosaicfield_chunked/3B_cells_final.npy",
    ],
    # The tall Xenium QC render it crops (si_xenium_run_qc.py).
    "scripts/paper/split_tall_si_figures.py": [
        "results/figures/si/fig_s_xenium_qc.png",
    ],
}

# ── Resolving a generator's real inputs ──────────────────────────────────────
# A generator's declared inputs come from literal `"...h5ad"`-style strings in
# its source and from `bimp_st.paths` constants (the preferred way), which have
# no literal in the generator's text. Both are scanned.
DATA_EXT = (".h5ad", ".csv", ".tsv", ".parquet", ".geojson", ".json", ".npy", ".tif")

try:
    import sys as _sys

    _sys.path.insert(0, str(REPO / "src"))
    import bimp_st.paths as _paths_mod
except ImportError:
    _paths_mod = None


def literal_inputs(tree: ast.AST) -> list[str]:
    """Repo-relative literal path strings a script's source names."""
    found = []
    for node in ast.walk(tree):
        v = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
        if not v or not v.endswith(DATA_EXT):
            continue
        if v.startswith(("results/", "/")) or "{" in v or "*" in v:
            continue
        if "/" not in v:
            continue
        found.append(v)
    return found


def paths_module_inputs(tree: ast.AST) -> list[str]:
    """Resolve bimp_st.paths.X / (imported) X usages to real, absolute paths."""
    if _paths_mod is None:
        return []
    module_aliases: set[str] = set()
    imported_names: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "bimp_st.paths":
                for alias in node.names:
                    imported_names[alias.asname or alias.name] = alias.name
            elif node.module == "bimp_st":
                for alias in node.names:
                    if alias.name == "paths":
                        module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "bimp_st.paths":
                    module_aliases.add(alias.asname or "paths")

    found = []
    for node in ast.walk(tree):
        attr_name = None
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in module_aliases
        ):
            attr_name = node.attr
        elif isinstance(node, ast.Name) and node.id in imported_names:
            attr_name = imported_names[node.id]
        if attr_name is None:
            continue
        val = getattr(_paths_mod, attr_name, None)
        # Files only: directory constants (REPO, RESULTS, ...) are not inputs.
        if isinstance(val, pathlib.Path) and val.suffix:
            found.append(str(val))
    return found


def declared_inputs(script_rel: str) -> list[str]:
    """Every real-looking data path a generator script names, resolved to an
    absolute path where possible. Best-effort: literal repo-relative strings
    plus resolved `bimp_st.paths` constants. Does not know about inputs a
    script computes at runtime (a glob, an argparse flag)."""
    f = REPO / script_rel
    if not f.exists():
        return []
    try:
        tree = ast.parse(f.read_text(errors="replace"))
    except (SyntaxError, ValueError):
        return []
    out = []
    for v in literal_inputs(tree):
        out.append(str(REPO / v))
    out.extend(paths_module_inputs(tree))
    return out
