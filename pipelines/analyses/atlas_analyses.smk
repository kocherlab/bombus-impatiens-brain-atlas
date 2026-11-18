"""Atlas analyses.

The supplementary analyses of the atlas: the canonical 17-type vocabulary,
regional diversity and region predictability, cross-modal concordance, the
Kenyon-cell threshold sweep, the snRNA KC subclustering, and the snRNA/Xenium
joint embedding.

Running them as a pipeline makes the outputs declared, the dependencies
explicit, and a failure visible (a script that dies partway can otherwise leave
an output directory that looks complete).

Usage (from this directory):
    uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake -s atlas_analyses.smk --profile ../../config/profiles/slurm    # SLURM
    uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake -s atlas_analyses.smk --profile ../../config/profiles/local    # local
    uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake -s atlas_analyses.smk -n                                         # dry run

NOTE ON PATHS. `SCRIPTS` below resolves relative to this Snakefile, so the
pipeline runs the scripts of the checkout it lives in; pointing at another
checkout's scripts/ would silently run a different branch's version.

harmonypy is installed in a job-local prefix rather than the shared .venv,
because it pulls a numpy that breaks scipy and numba there.
"""

import sys
from pathlib import Path

REPO = Path(workflow.basedir).parents[1]
SCRIPTS = REPO / "scripts"
SRC = REPO / "src"
sys.path.insert(0, str(SRC))
from bimp_st import paths as _paths  # noqa: E402
RESULTS = f"{REPO}/results/reframe"
PYLIBS = f"{REPO}/vendor/pylibs"
PY = f"{REPO}/.venv/bin/python"
# The HANDOFF_TABLE analyses (see HANDOFF_SCRIPT) write under REPO/results/handoff,
# where REPO is whichever checkout runs this Snakefile (results/ is the shared
# symlink in a worktree). That matches the scripts' own `parents[2]` resolution.
HANDOFF = f"{REPO}/results/handoff"
NINE_ATLAS = {
    b: f"{REPO}/results/nine_region/atlas/{b}_aligned_clustered_9region.h5ad"
    for b in ("queen", "worker")
}
# The snRNA caste DE: one ~ trial + caste fit (caste_de_blocked_all.py).
R9 = f"{REPO}/results/caste_de"
CASTE_FIT = f"{R9}/three_trial_pairs"  # bimp_st.paths.CASTE_DE_FIT_DIR
DE_COARSE = f"{CASTE_FIT}/de_coarse_blocked.csv"
DE_FINE = f"{CASTE_FIT}/de_fine_blocked.csv"
PANEL_CSV = f"{REPO}/data/panel/XeniumBrainPanel_BIMP+MGEN_vers2.csv"
PANEL_CLASSIFIERS = f"{REPO}/data/panel/BIMP_Xenium_Classifiers.xlsx"
PANEL_MAPPING = f"{REPO}/data/panel/np31_to_v43_mapping.csv"
V43_LOOKUP = str(_paths.SPATIAL_COMBINED.parent / "BIMP_v4.3_lookup.tsv")
# Leiden-stability design (must match si_leiden_stability.py; the script
# imports numpy at top level, which the uvx snakemake env lacks, so it is
# restated here; the dry-run lists the 27 label files per seed).
LEIDEN_SEEDS = [f"{i:02d}" for i in range(20)]
LEIDEN_RES = [f"{r / 10:.1f}" for r in range(1, 21)] + ["2.2", "2.4", "2.6", "2.8", "3.0", "3.2", "3.5"]
LEIDEN_DIR = f"{HANDOFF}/A_leiden_stability"

# `uv run` re-syncs the shared venv, so call the venv interpreter directly.
ENV = (
    f'PYTHONPATH="{PYLIBS}:{SRC}:{SCRIPTS}" '
    "OMP_NUM_THREADS={threads} OPENBLAS_NUM_THREADS={threads} "
    "MKL_NUM_THREADS={threads}"
)


rule all:
    input:
        f"{RESULTS}/region_structure/region_prediction.csv",
        f"{RESULTS}/region_structure/diversity_by_region.csv",
        f"{RESULTS}/panel_concordance/per_gene.csv",
        f"{RESULTS}/kc_threshold/threshold_sweep.csv",
        f"{RESULTS}/fig5_snuc_kc/marker_panel_coverage.csv",
        f"{RESULTS}/snuc_vs_xenium/mixing_by_cell_type.csv",
        f"{RESULTS}/slice_celltypes/slice_composition.csv",
        # HANDOFF_TABLE analyses (scripts/<topic>/, see HANDOFF_SCRIPT)
        f"{REPO}/results/figures/si/figS_leiden_stability.png",  # leiden_figure
        f"{R9}/B_caste_benchmark/caste_benchmark_counts.csv",
        f"{HANDOFF}/C_antennal_lobe/step3_classifier_metrics.csv",
        f"{HANDOFF}/E_orthologs/panel_dmel_orthologs.csv",
        f"{HANDOFF}/D_tables/S1_panel_key.csv",
        f"{HANDOFF}/middle_type_control/snpf_ratio_exclusions.csv",
        f"{R9}/glia_composition/survival.csv",
        f"{R9}/no_fatbody/comparison.csv",
        f"{R9}/sn_controls/pathway_kc.csv",
        f"{R9}/B_caste_benchmark_offset/direction_agreement.csv",


rule region_structure:
    """Cell-type diversity per region, and region predicted from local composition."""
    input:
        str(_paths.SPATIAL_COMBINED),
    output:
        pred=f"{RESULTS}/region_structure/region_prediction.csv",
        div=f"{RESULTS}/region_structure/diversity_by_region.csv",
        conf=f"{RESULTS}/region_structure/region_confusion.csv",
    threads: 16
    resources:
        mem_mb=192000,
        cpus_per_task=16,
        runtime=240,
    shell:
        ENV + f" {PY} -u {SCRIPTS}/regions/region_structure.py"


rule panel_concordance:
    """Does the panel measure what snRNA measures -- per gene and per cell type."""
    input:
        spatial=str(_paths.SPATIAL_COMBINED),
        tangram=str(_paths.CANONICAL_TANGRAM),
        cb=str(_paths.SN_CB),
        ol=str(_paths.SN_OL),
    output:
        gene=f"{RESULTS}/panel_concordance/per_gene.csv",
        ct=f"{RESULTS}/panel_concordance/per_cell_type.csv",
        bulk=f"{RESULTS}/panel_concordance/bulk.csv",
    threads: 16
    resources:
        mem_mb=192000,
        cpus_per_task=16,
        runtime=240,
    shell:
        ENV + f" {PY} -u {SCRIPTS}/atlas/si_panel_concordance.py"


rule kc_threshold:
    """What posterior does a cell need before it is called a Kenyon cell."""
    input:
        str(_paths.CANONICAL_TANGRAM),
    output:
        sweep=f"{RESULTS}/kc_threshold/threshold_sweep.csv",
        overlap=f"{RESULTS}/kc_threshold/definition_overlap.csv",
    threads: 8
    resources:
        mem_mb=128000,
        cpus_per_task=8,
        runtime=180,
    shell:
        ENV + f" {PY} -u {SCRIPTS}/kenyon/si_kc_threshold.py"


rule snuc_kc_subclusters:
    """snRNA Kenyon-cell subclustering + how much of it the panel can see."""
    input:
        # Built by scripts/kenyon/large_type_kc_subclusters.py, not part
        # of this DAG (not itself re-derivable without the raw multiome
        # checkpoint); declared so a re-run of that prep step invalidates this
        # rule.
        f"{REPO}/results/fig5_kc/CB_LargeType_subclustered.h5ad",
    output:
        cov=f"{RESULTS}/fig5_snuc_kc/marker_panel_coverage.csv",
        mk=f"{RESULTS}/fig5_snuc_kc/kc_subcluster_markers.csv",
    threads: 8
    resources:
        mem_mb=128000,
        cpus_per_task=8,
        runtime=120,
    shell:
        ENV + f" {PY} -u {SCRIPTS}/kenyon/snrna_kc_subclusters.py"


rule snuc_vs_xenium:
    """Joint embedding of both modalities on the shared panel genes.

    The heaviest rule and the one that died as a background process. Needs the
    larger allocation for the Harmony step on the concatenated object.
    """
    input:
        spatial=str(_paths.SPATIAL_COMBINED),
        cb=str(_paths.SN_CB),
        ol=str(_paths.SN_OL),
    output:
        mix=f"{RESULTS}/snuc_vs_xenium/mixing_by_cell_type.csv",
        cells=f"{RESULTS}/snuc_vs_xenium/joint_embedding_cells.csv.gz",
    threads: 16
    resources:
        mem_mb=256000,
        cpus_per_task=16,
        runtime=360,
    shell:
        ENV + f" {PY} -u {SCRIPTS}/atlas/snrna_xenium_joint_embedding.py"


rule slice_celltypes:
    """One queen section coloured by the raw Tangram call, before anatomy tidies it."""
    input:
        str(_paths.SPATIAL_COMBINED),
    output:
        comp=f"{RESULTS}/slice_celltypes/slice_composition.csv",
    threads: 8
    resources:
        mem_mb=128000,
        cpus_per_task=8,
        runtime=120,
    shell:
        ENV + f" {PY} -u {SCRIPTS}/identity/section_raw_celltypes.py"


# ---------------------------------------------------------------------------
# HANDOFF_TABLE analyses (scripts/<topic>/, see HANDOFF_SCRIPT). Outputs land in results/handoff/<A..E>_*/.
# ---------------------------------------------------------------------------

LEIDEN_ENV = (
    # BLAS/OMP capped at 8 (OpenBLAS segfaults when it fans out over this
    # host's cores); numba and scanpy at 16.
    f'PYTHONPATH="{SRC}" OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 '
    "NUMBA_NUM_THREADS=16 N_JOBS=16"
)


rule leiden_ref:
    """Leiden at every resolution on the STORED full-data graph (reference labels)."""
    input:
        h5ad=str(_paths.SPATIAL_COMBINED),
    output:
        labels=expand(f"{LEIDEN_DIR}/ref/labels_r{{res}}.npy", res=LEIDEN_RES),
        validation=f"{LEIDEN_DIR}/ref/validation.txt",
    threads: 16
    resources:
        mem_mb=128000,
        cpus_per_task=16,
        runtime=1440,
    shell:
        LEIDEN_ENV + f" {PY} -u {SCRIPTS}/atlas/si_leiden_stability.py ref"


rule leiden_seed:
    """One 80% subsample (seed 00..19): rebuild PCA + kNN, Leiden at every resolution."""
    input:
        h5ad=str(_paths.SPATIAL_COMBINED),
    output:
        idx=f"{LEIDEN_DIR}/seed{{seed}}/idx.npy",
        labels=expand(f"{LEIDEN_DIR}/seed{{{{seed}}}}/labels_r{{res}}.npy", res=LEIDEN_RES),
    wildcard_constraints:
        seed=r"\d\d",
    threads: 16
    resources:
        mem_mb=128000,
        cpus_per_task=16,
        runtime=1440,
    shell:
        LEIDEN_ENV + f" {PY} -u {SCRIPTS}/atlas/si_leiden_stability.py seed {{wildcards.seed}}"


rule leiden_aggregate:
    """ARI / cluster count / region purity per resolution over the 20 seeds."""
    input:
        h5ad=str(_paths.SPATIAL_COMBINED),
        ref=rules.leiden_ref.output.labels,
        seeds=expand(f"{LEIDEN_DIR}/seed{{seed}}/idx.npy", seed=LEIDEN_SEEDS),
        labels=expand(
            f"{LEIDEN_DIR}/seed{{seed}}/labels_r{{res}}.npy",
            seed=LEIDEN_SEEDS, res=LEIDEN_RES,
        ),
    output:
        per_seed=f"{LEIDEN_DIR}/leiden_stability_per_seed.csv",
        summary=f"{LEIDEN_DIR}/leiden_stability.csv",
    threads: 4
    resources:
        mem_mb=64000,
        cpus_per_task=4,
        runtime=120,
    shell:
        # `aggregate` also draws the figure as a side effect; leiden_figure
        # below owns the declared png/pdf (it redraws them from the csv).
        LEIDEN_ENV + f" {PY} -u {SCRIPTS}/atlas/si_leiden_stability.py aggregate"


rule leiden_figure:
    """Supplementary figure figS_leiden_stability, redrawn from leiden_stability.csv."""
    input:
        csv=rules.leiden_aggregate.output.summary,
    output:
        png=f"{REPO}/results/figures/si/figS_leiden_stability.png",
        pdf=f"{REPO}/results/figures/si/figS_leiden_stability.pdf",
    threads: 1
    resources:
        mem_mb=8000,
        cpus_per_task=1,
        runtime=15,
    shell:
        LEIDEN_ENV + f" {PY} -u {SCRIPTS}/atlas/si_leiden_stability.py figure"


rule caste_benchmark:
    """Per-region caste panel-gene contrasts vs within-brain split-half null and snRNA direction."""
    input:
        queen=NINE_ATLAS["queen"],
        worker=NINE_ATLAS["worker"],
        deseq=DE_COARSE,
        mapping=PANEL_MAPPING,
    output:
        table=f"{R9}/B_caste_benchmark/caste_benchmark_table.csv",
        counts=f"{R9}/B_caste_benchmark/caste_benchmark_counts.csv",
        sections=f"{R9}/B_caste_benchmark/caste_benchmark_sections.csv",
        md=f"{R9}/B_caste_benchmark/caste_benchmark_counts.md",
    threads: 4
    resources:
        mem_mb=96000,
        cpus_per_task=4,
        runtime=60,
    shell:
        ENV + f" {PY} -u {SCRIPTS}/caste_de/regional_caste_benchmark.py"


AL_OUTPUTS = {
    "prev": "step1_prevalence_scenarios.csv",
    "aptab": "step1_ap_crop_table.csv",
    "comp": "step1_al_composition.csv",
    "agree": "step1_al_composition_agreement.csv",
    "fig_png": "fig_al_ap_profile.png",
    "fig_pdf": "fig_al_ap_profile.pdf",
    "tracing": "tracing_by_section.csv",
    "m2": "step2_classifier_metrics.csv",
    "r2": "step2_per_region_report.csv",
    "p2": "step2_worker_crop_predictions.csv",
    "n2": "step2_predicted_AL_not_traced.csv",
    "d2": "step2_traced_vs_untraced_diagnostic.csv",
    "m3": "step3_classifier_metrics.csv",
    "r3": "step3_per_region_report.csv",
    "p3": "step3_worker_crop_predictions.csv",
    "n3": "step3_predicted_AL_not_traced.csv",
    "d3": "step3_traced_vs_untraced_diagnostic.csv",
}
AL_ANNOTATIONS = sorted(
    str(f)
    for brain in ("queen", "worker")
    for d in ("v1", "v4")
    for f in (REPO / "data/annotations" / brain / d).glob("*.geojson")
)


rule al_matched_range:
    """Antennal lobe: matched anterior-posterior range, prevalence scenarios, cross-brain transfer."""
    input:
        spatial=str(_paths.SPATIAL_COMBINED),
        tangram=str(_paths.CANONICAL_TANGRAM),
        # per-cell in_* flags (attach_region_flags) and the GeoJSON tracings
        # (annotation_sources / load_regions) the "traced" call is read from
        flags=f"{REPO}/results/nine_region/region_flags.parquet",
        annotations=AL_ANNOTATIONS,
        helper=str(SCRIPTS / "regions/region_prediction.py"),
    output:
        **{k: f"{HANDOFF}/C_antennal_lobe/{v}" for k, v in AL_OUTPUTS.items()},
    threads: 16
    resources:
        mem_mb=192000,
        cpus_per_task=16,
        runtime=240,
    shell:
        ENV + f" {PY} -u {SCRIPTS}/regions/antennal_lobe_matched_range.py"


rule panel_orthologs:
    """Dmel orthologs of the 101 panel genes (NCBI Gene orthologs, else DIAMOND reciprocal best hit,
    else a >=2-independent-line consensus: Ensembl Compara, OrthoDB v11, ultra-sensitive RBH, annotation+alignment).
    Per-gene evidence lands in ortholog_evidence.md; the extra fetches (Ensembl/OrthoDB JSON, ultra-sensitive
    DIAMOND) are cached under E_orthologs/cache_ext and, like the other caches, are not declared outputs.
    `panel_ortholog_alignments.py` (needs Biopython, outside the project venv) is run by hand for alignments_named.txt.

    NEEDS NETWORK on a cold cache (NCBI FTP / Datasets API, UniProt) and a
    `diamond` binary (env DIAMOND or on PATH), so it runs on the submit node
    (`localrule`), never in a compute job. Every download and alignment is cached
    under results/handoff/E_orthologs/cache (and D_tables/cache for the NCBI
    GeneID lookups), and each step is skipped when its cache file exists: with a
    warm cache the rule is offline-reproducible. The caches are deliberately NOT
    declared as outputs, so deleting one cannot trigger a silent re-download in a
    DAG-driven rerun.
    """
    input:
        panel=PANEL_CSV,
        classifiers=PANEL_CLASSIFIERS,
        mapping=PANEL_MAPPING,
        lookup=V43_LOOKUP,
        helper=str(SCRIPTS / "gene_annotation/tables_s1_s2.py"),
        sources=str(SCRIPTS / "gene_annotation/panel_ortholog_sources.py"),
    output:
        csv=f"{HANDOFF}/E_orthologs/panel_dmel_orthologs.csv",
        summary=f"{HANDOFF}/E_orthologs/summary_by_method.txt",
        evidence=f"{HANDOFF}/E_orthologs/ortholog_evidence.md",
    localrule: True
    threads: 8
    shell:
        ENV + f" {PY} -u {SCRIPTS}/gene_annotation/panel_orthologs.py"


rule si_tables:
    """Supplementary Table S1 (panel key) and S2 (snRNA annotations) as CSV + LaTeX."""
    input:
        panel=PANEL_CSV,
        classifiers=PANEL_CLASSIFIERS,
        mapping=PANEL_MAPPING,
        lookup=V43_LOOKUP,
        sn=str(_paths.SN_COMBINED),
        orthologs=rules.panel_orthologs.output.csv,
    output:
        s1_csv=f"{HANDOFF}/D_tables/S1_panel_key.csv",
        s1_tex=f"{HANDOFF}/D_tables/S1_panel_key.tex",
        markers=f"{HANDOFF}/D_tables/snrna_markers_top50.csv",
        s2_csv=f"{HANDOFF}/D_tables/S2_snrna_annotations.csv",
        s2_tex=f"{HANDOFF}/D_tables/S2_snrna_annotations.tex",
    threads: 4
    resources:
        mem_mb=64000,
        cpus_per_task=4,
        runtime=30,
    shell:
        ENV + f" {PY} -u {SCRIPTS}/gene_annotation/tables_s1_s2.py"


rule middle_type_control:
    """sNPF zone (fig:kc_subtypes): does the queen excess survive excluding Middle-type KCs?"""
    input:
        ltkc=f"{REPO}/results/tangram_ltKC_cb/ltKC_CB_spatial_subclusters_dsource.h5ad",
        kc5pop=f"{REPO}/results/tangram_ltKC/kc5pop_spatial_allKC_dsource.h5ad",
        sn=f"{REPO}/data/sn/BIMP_atlas_qw_CentralBrain.h5ad",
        # distances also read the raw per-crop h5ads under pipelines/*/data
    output:
        ratios=f"{HANDOFF}/middle_type_control/snpf_ratio_exclusions.csv",
        bins=f"{HANDOFF}/middle_type_control/middle_by_distance_bin.csv",
        zone=f"{HANDOFF}/middle_type_control/zone_middle_caste.json",
        fig=f"{HANDOFF}/middle_type_control/middle_type_check.png",
    threads: 8
    resources:
        mem_mb=128000,
        cpus_per_task=8,
        runtime=120,
    shell:
        ENV + f" {PY} -u {SCRIPTS}/kenyon/snpf_middle_type_control.py"


# pyDESeq2 0.5.4 lives in its own environment (config/envs/deseq2.txt), never the shared .venv.
DESEQ_PY = f"{REPO}/.venv-deseq2/bin/python"
GLIA_SUBTYPES = [
    "Astrocyte_like_glia", "Cortex_glia", "Other_glia", "Perineurial_glia",
    "Subperineurial_glia", "Class_II_Kenyon_cells", "Large_type_Kenyon_cells",
    "Middle_type_Kenyon_cells", "Other_Kenyon_cells", "Small_type_Kenyon_cells",
]


rule caste_de_blocked:
    """The snRNA caste DE: whole brain, pooled classes, fine types."""
    input:
        coarse=f"{REPO}/results/figures/fig3/pseudobulk/counts_all.csv",
        fine=f"{REPO}/results/caste_de_pseudobulk/fine_de/pseudobulk/groups.json",
        nuclei=f"{REPO}/results/caste_de_pseudobulk/fine_de/nuclei_per_sample.csv",
    output:
        coarse=DE_COARSE,
        fine=DE_FINE,
        counts=f"{CASTE_FIT}/deg_counts.csv",
        focal=f"{CASTE_FIT}/focal_genes.csv",
    threads: 2
    resources:
        mem_mb=24000,
        cpus_per_task=2,
        runtime=120,
    shell:
        ENV + f" {DESEQ_PY} -u {SCRIPTS}/caste_de/caste_de_blocked_all.py"


rule de_without_fat_body:
    """Caste DE refit without the 13 fat-body nuclei."""
    input:
        atlas=f"{REPO}/data/sn/BIMP_atlas_qw_CentralBrain.h5ad",
        deseq=DE_COARSE,
    output:
        comparison=f"{R9}/no_fatbody/comparison.csv",
        core=f"{R9}/no_fatbody/core.csv",
    threads: 2
    resources:
        mem_mb=64000,
        cpus_per_task=2,
        runtime=240,
    shell:
        ENV + f" {DESEQ_PY} -u {SCRIPTS}/caste_de/de_without_fat_body_nuclei.py --stage all"


rule caste_benchmark_offset:
    """Offset, region specificity and direction-vs-chance for the caste benchmark."""
    input:
        counts=f"{R9}/B_caste_benchmark/caste_benchmark_counts.csv",
        comp=f"{REPO}/results/figures/fig2/slice_composition.csv",
        atlas=list(NINE_ATLAS.values()),
        deseq=DE_COARSE,
    output:
        f"{R9}/B_caste_benchmark_offset/offset_benchmark_counts.csv",
        f"{R9}/B_caste_benchmark_offset/region_jaccard.csv",
        f"{R9}/B_caste_benchmark_offset/direction_agreement.csv",
    threads: 8
    resources:
        mem_mb=128000,
        cpus_per_task=8,
        runtime=180,
    shell:
        ENV + f" {PY} -u {SCRIPTS}/caste_de/regional_caste_benchmark_offset.py"


rule sn_pathway:
    """JH / ecdysone pathway genes in Kenyon cells, from the trial-blocked fit."""
    input:
        deseq=DE_COARSE,
        fine=DE_FINE,
    output:
        f"{R9}/sn_controls/pathway_kc.csv",
    threads: 2
    resources:
        mem_mb=16000,
        cpus_per_task=2,
        runtime=30,
    shell:
        ENV + f" {PY} -u {SCRIPTS}/caste_de/snrna_controls.py --stage pathway"


rule glia_composition:
    """n=3 snRNA caste DE: is the pooled glia (and KC) result a subtype-composition shift?"""
    input:
        pooled=expand(
            f"{REPO}/results/figures/fig3/pseudobulk/counts_{{c}}.csv",
            c=["Glia", "Kenyon_cells"],
        ),
        subtypes=expand(
            f"{REPO}/results/caste_de_pseudobulk/fine_de/pseudobulk/counts_{{s}}.csv",
            s=GLIA_SUBTYPES,
        ),
        nuclei=f"{REPO}/results/caste_de_pseudobulk/fine_de/nuclei_per_sample.csv",
        fine_de=DE_FINE,
        deseq=DE_COARSE,
    output:
        survival=f"{R9}/glia_composition/survival.csv",
        diag=f"{R9}/glia_composition/diag_Glia.csv",
        overlap=f"{R9}/glia_composition/subtype_deg_overlap.csv",
        props=f"{R9}/glia_composition/subtype_proportions.csv",
    threads: 8
    resources:
        mem_mb=32000,
        cpus_per_task=8,
        runtime=120,
    shell:
        ENV + f" {DESEQ_PY} -u {SCRIPTS}/caste_de/glia_composition_confound.py"


# The remaining results/handoff analyses: one rule each, from this table.
# name -> (outputs under HANDOFF, inputs, threads, mem_mb, runtime_min, python)
_FSC_REF = f"{REPO}/results/fig5_kc/CB_LargeType_subclustered.h5ad"
_LTKC = f"{REPO}/results/tangram_ltKC_cb/ltKC_CB_spatial_subclusters_dsource.h5ad"
HANDOFF_TABLE = {
    "al_polygon_adequacy": (
        ["C_antennal_lobe/adequacy/halo_summary.csv",
         "C_antennal_lobe/adequacy/prevalence_under_hypotheses.csv"],
        [f"{HANDOFF}/C_antennal_lobe/step1_ap_crop_table.csv", *AL_ANNOTATIONS],
        16, 192000, 240, PY),
    "forced_simplex_controls": (["forced_simplex_controls/summary.json"],
                                [_LTKC, _FSC_REF, str(_paths.CANONICAL_TANGRAM)],
                                8, 128000, 120, PY),
    "label_placement": (["label_placement/summary.csv"],
                        [f"{REPO}/results/round6/vote_stats/per_cell_labels_canonical.parquet",
                         str(_paths.CANONICAL_TANGRAM)],
                        8, 128000, 120, PY),
    "sn_controls": (["sn_controls/sccoda_summary.csv"],
                    [f"{REPO}/results/figures/fig2/slice_composition.csv"],
                    8, 64000, 120, PY),
    "snpf_spillover": (["snpf_spillover/summary.csv"], [_LTKC, PANEL_MAPPING],
                       8, 160000, 180, PY),
    "spatial_breadth": (["G_spatial_breadth/cluster_to_vote_group.csv"],
                        [str(_paths.CANONICAL_TANGRAM), *NINE_ATLAS.values()],
                        8, 160000, 180, PY),
    "tk_battery_nine_region": (["tk_battery_nine_region/summary.csv"], [*NINE_ATLAS.values()],
                               8, 160000, 180, PY),
    "tk_detection_sensitivity": (["tk_detection_sensitivity/summary.csv"],
                                 [*NINE_ATLAS.values()], 8, 128000, 120, PY),
    "vgn_fat_body": (["vgn_fat_body/summary.csv"], [str(_paths.SN_COMBINED)],
                     8, 96000, 60, PY),
}

# Script for each HANDOFF_TABLE entry, relative to scripts/.
HANDOFF_SCRIPT = {
    "al_polygon_adequacy": "regions/antennal_lobe_polygon_adequacy.py",
    "forced_simplex_controls": "kenyon/forced_simplex_controls.py",
    "label_placement": "identity/label_anatomy_placement.py",
    "sn_controls": "caste_de/snrna_controls.py",
    "snpf_spillover": "kenyon/snpf_spillover_controls.py",
    "spatial_breadth": "regions/celltype_spatial_breadth.py",
    "tk_battery_nine_region": "neuropeptides/tk_shift_battery.py",
    "tk_detection_sensitivity": "neuropeptides/tk_detection_sensitivity.py",
    "vgn_fat_body": "caste_de/vgn_fat_body_origin.py",
}

for _name, (_outs, _ins, _cpu, _mem, _rt, _py) in HANDOFF_TABLE.items():

    rule:
        name:
            f"handoff_{_name}"
        input:
            script=str(SCRIPTS / HANDOFF_SCRIPT[_name]),
            data=_ins,
        output:
            [f"{HANDOFF}/{o}" for o in _outs],
        threads: _cpu
        resources:
            mem_mb=_mem,
            cpus_per_task=_cpu,
            runtime=_rt,
        shell:
            ENV + f" {_py} -u {{input.script}}"


rule handoff_all:
    input:
        [f"{HANDOFF}/{o}" for outs, *_ in HANDOFF_TABLE.values() for o in outs],
