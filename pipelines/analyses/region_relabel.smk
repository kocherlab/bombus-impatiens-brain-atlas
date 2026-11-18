# Nine-region relabel of the whole-brain atlas, end to end.
#
#   uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake -s region_relabel.smk --profile ../../config/profiles/slurm     # production
#   uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake -s region_relabel.smk --profile ../../config/profiles/local     # local
#   uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake -s region_relabel.smk -n --profile ../../config/profiles/slurm  # dry run
#   uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake relabel --profile ...                       # labels only, ~2 min
#
# Adds the antennal lobe and the two central-complex substructures (ellipsoid
# and fan-shaped body) to the six regions of the first annotation round, then
# re-runs every analysis and figure whose numbers depend on the region set.
#
# Why this is a separate pipeline rather than a re-run of the wholebrain one
# -------------------------------------------------------------------------
# `label` is the root of the wholebrain DAG, so re-running it there rebuilds
# `align` (PASTE3, GPU, ~6 h) and the Tangram mapping below it. That alignment
# is stochastic: it would move every downstream spatial number to add
# polygons that do not move a single cell. Alignment, clustering and Tangram
# never read the `in_*` columns, so the labels are recomputed here on their own
# and joined back by (slice_name, cell_id).
#
# Figure and analysis steps are tracked by stamp files. Those scripts do not
# declare their outputs in one place -- several write a directory of PNGs and
# CSVs through a shared helper -- so a stamp is used as the dependency edge
# rather than an enumerated output list that could drift.

from pathlib import Path

REPO = Path(workflow.basedir).resolve().parents[1]
RESULTS = REPO / "results/nine_region"
STAMPS = RESULTS / "stamps"
BRAINS = ["queen", "worker"]

ATLAS = str(RESULTS / "atlas" / "{brain}_aligned_clustered_9region.h5ad")
ATLASES = expand(ATLAS, brain=BRAINS)

# `uv run` would re-sync the venv against pyproject (which pins tangram-sc),
# so call the venv interpreter directly.
PY = f"{REPO}/.venv/bin/python"

# `verify`/`crops` (below) call scripts/regions/relabel_nine_regions.py,
# which reads these paths; declaring them as `input:` makes a changed annotation
# GeoJSON or a re-converted crop trigger a re-run. Mirrors
# nine_region_common.BRAINS / annotation_sources() and
# relabel_nine_regions.find_crop_file()'s filter, so declared and actual reads
# stay in sync.
BRAIN_DIRS = {
    "queen": REPO / "pipelines/wholebrain/queen",
    "worker": REPO / "pipelines/wholebrain/worker",
}
ANNOTATION_DIRS = {b: REPO / "data/annotations" / b for b in BRAIN_DIRS}


def _crop_h5ads(brain):
    data_dir = BRAIN_DIRS[brain] / "data"
    if not data_dir.exists():
        return []
    return sorted(
        str(f) for f in data_dir.glob("*.h5ad")
        if "_clustered" not in f.name and "_labeled" not in f.name
        and not f.name.startswith(("aligned", "test_"))
    )


def _newest_annotation_dir(brain):
    base = ANNOTATION_DIRS[brain]
    newest = base / "v2"
    for cand in ("v3", "v4"):
        if (base / cand).exists():
            newest = base / cand
    return str(newest)

# scipy's bundled OpenBLAS segfaults without a traceback when joblib fans out
# over this host's 128 cores, so thread counts are capped here.
# BIMP_FIG_SUBSAMPLE_FRACTION defaults to 0.1 (preview) -- full renders
# need 1.
shell.prefix(
    f"export PYTHONPATH={REPO}/src:{REPO}/scripts HDF5_USE_FILE_LOCKING=FALSE "
    "OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 "
    "NUMEXPR_NUM_THREADS=8 BIMP_FIG_SUBSAMPLE_FRACTION=1 && "
)

# script -> what it must run after. Analyses write the CSVs the figures read.
# caste_comparison in particular asserts that the Hill-diversity CSV carries
# exactly the region set it expects, so it has to follow its producer.
ANALYSES = {
    "region_celltype_diversity": ATLASES,
    "reframe_region_prediction_v2": ATLASES,
    "regional_degs": ATLASES,
    # Predicts region from a fold-wise Leiden
    # vocabulary, so the region set is one of its classes.
    "reframe_foldwise_leiden": ATLASES,
    # Measures whether adjacent sections agree
    # on a bin's region, so a new region adds something to agree about.
    "reframe_registration_accuracy": ATLASES,
}
FIGURES = {
    "reframe_region_prediction_v2_fig": ["reframe_region_prediction_v2"],
    "figS_hill_diversity": ["region_celltype_diversity"],
    "caste_comparison": ["region_celltype_diversity"],
    "figS_percell_vs_vote": [],
    "figS_tangram_confidence": [],
    "atlas_overview": [],
    "figS_exploded_regions": [],
}

# Not in this pipeline: region_structure.py / fig_atlas_structure.py.
# Neither feeds a registered figure, and reframe_region_structure only replots
# cached six-region CSVs while REPLOT_DIVERSITY is set. To include them, set
# REPLOT_DIVERSITY=False and add them here.

# ANALYSES/FIGURES keys are logical names (also used for stamp/log filenames
# and cross-references between the two dicts), not script locations.
# SCRIPT_PATH maps a name to its path under scripts/; a name absent from it
# resolves to scripts/{name}.py.
SCRIPT_PATH = {
    "reframe_region_prediction_v2": "regions/region_prediction.py",
    "region_celltype_diversity": "regions/region_diversity.py",
    "reframe_foldwise_leiden": "regions/region_prediction_foldwise_leiden.py",
    "reframe_registration_accuracy": "atlas/paste3_registration_accuracy.py",
    "reframe_region_prediction_v2_fig": "regions/si_region_prediction.py",
    "figS_hill_diversity": "regions/si_hill_diversity.py",
    "caste_comparison": "regions/fig_caste_comparison.py",
    "figS_percell_vs_vote": "identity/si_percell_vs_vote.py",
    "figS_tangram_confidence": "identity/si_tangram_confidence.py",
    "atlas_overview": "atlas/fig_atlas_overview.py",
    "figS_exploded_regions": "regions/si_exploded_regions.py",
    "regional_degs": "caste_de/si_regional_degs.py",
}


def script_path(name):
    """Resolve a logical ANALYSES/FIGURES name to its path under scripts/."""
    return SCRIPT_PATH.get(name, f"{name}.py")


def figure_inputs(wildcards):
    deps = FIGURES[wildcards.fig]
    return ATLASES + [str(STAMPS / f"{d}.done") for d in deps]


rule all:
    input:
        RESULTS / "coverage.json",
        ATLASES,
        expand(str(RESULTS / "verify" / "{brain}" / "_summary.csv"), brain=BRAINS),
        expand(str(STAMPS / "{a}.done"), a=list(ANALYSES)),
        expand(str(STAMPS / "{f}.done"), f=list(FIGURES)),
        STAMPS / "reframe_foldwise_leiden_res2.done",


rule relabel:
    """Labels only, without the downstream re-analysis. About two minutes."""
    input:
        RESULTS / "coverage.json",
        ATLASES,
        expand(str(RESULTS / "verify" / "{brain}" / "_summary.csv"), brain=BRAINS),


rule verify:
    """Correctness proof: recompute the SIX original regions from the ORIGINAL
    annotations and require cell-for-cell equality with the stored in_* flags.

    This is what licenses trusting the three new columns. It must reproduce
    the stored six-region flags exactly (cell count, cells in more than one
    region, per-region counts) with the polygon repair disabled, which is the
    like-for-like comparison.
    """
    input:
        annotations=lambda w: directory(str(ANNOTATION_DIRS[w.brain] / "v1")),
        crops=lambda w: _crop_h5ads(w.brain),
    output:
        RESULTS / "verify" / "{brain}" / "_summary.csv",
    log:
        "logs/verify_{brain}.log",
    resources:
        mem_mb=64000,
        runtime=60,
    shell:
        f"cd {REPO} && {PY} scripts/regions/relabel_nine_regions.py "
        "--brain {wildcards.brain} --stage verify > {log} 2>&1"


rule crops:
    """Point-in-polygon per crop, in the crop's own micron coordinates."""
    input:
        old_annotations=lambda w: directory(str(ANNOTATION_DIRS[w.brain] / "v1")),
        newest_annotations=lambda w: directory(_newest_annotation_dir(w.brain)),
        crops=lambda w: _crop_h5ads(w.brain),
    output:
        RESULTS / "crops" / "{brain}" / "_summary.csv",
    log:
        "logs/crops_{brain}.log",
    resources:
        mem_mb=64000,
        runtime=60,
    shell:
        f"cd {REPO} && {PY} scripts/regions/relabel_nine_regions.py "
        "--brain {wildcards.brain} --stage crops > {log} 2>&1"


rule join:
    """Join the flags onto the aligned atlas by (slice_name, cell_id).

    Not positional: a crop can hold more cells than it contributes to the
    atlas (crop 1A does), so position would silently shift labels within a
    slice.
    """
    input:
        RESULTS / "crops" / "{brain}" / "_summary.csv",
    output:
        ATLAS,
    log:
        "logs/join_{brain}.log",
    resources:
        mem_mb=128000,
        runtime=60,
    shell:
        f"cd {REPO} && {PY} scripts/regions/relabel_nine_regions.py "
        "--brain {wildcards.brain} --stage join > {log} 2>&1"


rule coverage:
    """Per-region coverage, each region against its own denominator.

    Regions were not all drawn on all sections, so a single figure across all
    nine would judge each region on sections that have nothing to do with it.
    """
    input:
        ATLASES,
    output:
        RESULTS / "coverage.json",
    log:
        "logs/coverage.log",
    resources:
        mem_mb=64000,
        runtime=30,
    shell:
        f"cd {REPO} && {PY} scripts/regions/region_coverage.py > {{log}} 2>&1"


rule analysis:
    """One region-dependent analysis. Writes the CSVs the figures read."""
    input:
        lambda w: ANALYSES[w.analysis],
    output:
        touch(str(STAMPS / "{analysis}.done")),
    params:
        script=lambda w: script_path(w.analysis),
    log:
        "logs/{analysis}.log",
    resources:
        mem_mb=192000,
        runtime=720,
        cpus_per_task=8,
    wildcard_constraints:
        analysis="|".join(ANALYSES),
    shell:
        f"cd {REPO} && {PY} scripts/{{params.script}} > {{log}} 2>&1"


rule figure:
    """One region-dependent figure."""
    input:
        figure_inputs,
    output:
        touch(str(STAMPS / "{fig}.done")),
    params:
        script=lambda w: script_path(w.fig),
    log:
        "logs/{fig}.log",
    resources:
        mem_mb=192000,
        runtime=360,
        cpus_per_task=8,
    wildcard_constraints:
        fig="|".join(FIGURES),
    shell:
        f"cd {REPO} && {PY} scripts/{{params.script}} > {{log}} 2>&1"


rule foldwise_leiden_res2:
    """The vocabulary-width control.

    Same fold-wise procedure at Leiden resolution 2.0, which yields 16-19
    clusters and so matches the width of the projected cell-type vocabulary.
    It answers whether the expression arm's advantage is feature count rather
    than biology, so it has to move with the region set like everything else.
    Writes to its own output directory, keyed off the env var.
    """
    input:
        ATLASES,
        STAMPS / "reframe_foldwise_leiden.done",
    output:
        touch(str(STAMPS / "reframe_foldwise_leiden_res2.done")),
    log:
        "logs/reframe_foldwise_leiden_res2.log",
    resources:
        mem_mb=192000,
        runtime=720,
        cpus_per_task=8,
    shell:
        f"cd {REPO} && BIMP_LEIDEN_RES=2.0 {PY} "
        "scripts/regions/region_prediction_foldwise_leiden.py > {log} 2>&1"
