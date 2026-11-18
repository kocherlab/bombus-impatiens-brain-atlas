# Whole-brain Xenium pipeline, shared by both brains.
#
# Each brain directory (queen/ and worker/ here) holds a two-line Snakefile
# that loads its config.yaml and includes this file, plus its own data/,
# figures/, labeling.py (ANNOTATION_MAPPING) and convert_single_region.py
# (raw-data paths and crop boxes); its region annotations are in
# data/annotations/<brain>/ at the repository root. Everything that differs between the brains is in config.yaml; the
# code is in this directory and runs with the brain directory as cwd.
#
# DAG:
#   [base_image -> convert_region, per raw region; worker only]
#       -> label -> align -> cluster -+-> visualize_3d
#                                     +-> render_video
#
# Usage, from a brain directory:
#   uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake --profile ../../../config/profiles/slurm    # SLURM production
#   uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake --profile ../../../config/profiles/local     # Local testing
#   uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake -n --profile ../../../config/profiles/slurm  # Dry run
#   uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake --config aligner=mosaicfield --profile ../../../config/profiles/slurm
#
# Do not add inputs (scripts, config) to label/align/cluster: a fresh mtime on
# a new input marks the canonical alignment stale, and `align` is a stochastic
# ~6 h GPU job whose output every spatial number depends on.

import yaml
from pathlib import Path

_defaults_path = Path(workflow.basedir) / "../../../config/snakemake_defaults.yaml"
if _defaults_path.exists():
    with open(_defaults_path) as f:
        _defaults = yaml.safe_load(f)
    # Local config wins over defaults
    _merged = {**_defaults, **config}
    config.update(_merged)

shell.prefix("export PATH=$HOME/.local/bin:$PATH UV_LINK_MODE=copy HDF5_USE_FILE_LOCKING=FALSE && ")

# workflow.basedir is the MAIN Snakefile's directory, i.e. the brain directory,
# even though these lines are parsed from the included file.
WORKDIR = str(Path(workflow.basedir).resolve())
# Repo root, for PYTHONPATH when calling .venv/bin/python directly instead of
# `uv run` (which re-syncs the shared venv and drops the --no-deps tangram2
# install). Referenced as {PROJECT_ROOT} in shell bodies, the same way
# {WORKDIR} already is.
PROJECT_ROOT = str(Path(workflow.basedir).resolve().parents[2])
DATA_DIR = f"{WORKDIR}/data"
BRAIN_ID = config["brain_id"]
ALIGNER = config.get("aligner", "paste3")

# Raw Xenium regions this pipeline converts to h5ad crops (worker only; the
# queen's crops were made by its convert_single_region.py before the pipeline
# existed, and `label` does not depend on them).
CONVERT_REGIONS = list(config.get("convert_regions") or [])
HIGH_MEM_REGIONS = set(config.get("high_mem_regions") or [])

# Targets
TARGETS = [
    f"{DATA_DIR}/aligned_clustered.h5ad",
    f"{WORKDIR}/figures/brain_3d.html",
    f"{WORKDIR}/figures/brain_rotation.mp4",
]

# `rule all: input: TARGETS` is in each brain's Snakefile: Snakemake never takes
# the default target from an included file, even with `default_target: True`.

# =============================================================================
# PARALLEL CROPPING: base_images and convert (one job per region; worker only)
# =============================================================================
if CONVERT_REGIONS:
    rule all_base_images:
        """Generate all base morphology images"""
        input:
            expand(f"{WORKDIR}/figures/crops/{{region}}_base.png", region=CONVERT_REGIONS)

    rule base_image:
        """Generate base morphology image for a single region"""
        output:
            f"{WORKDIR}/figures/crops/{{region}}_base.png"
        resources:
            mem_mb=256000,
            runtime=60
        shell:
            "cd {WORKDIR} && PYTHONPATH={PROJECT_ROOT}/src ../../../.venv/bin/python convert_single_region.py base-images '{wildcards.region}'"

    rule all_convert:
        """Convert all regions to h5ad"""
        input:
            expand(f"{WORKDIR}/.convert_{{region}}_done", region=CONVERT_REGIONS)

    rule convert_region:
        """Convert a single region to h5ad (high_mem_regions get 512GB)"""
        input:
            f"{WORKDIR}/figures/crops/{{region}}_base.png"
        output:
            touch(f"{WORKDIR}/.convert_{{region}}_done")
        resources:
            mem_mb=lambda wc: 512000 if wc.region in HIGH_MEM_REGIONS else 256000,
            runtime=lambda wc: 180 if wc.region in HIGH_MEM_REGIONS else 120
        shell:
            "cd {WORKDIR} && PYTHONPATH={PROJECT_ROOT}/src ../../../.venv/bin/python convert_single_region.py convert '{wildcards.region}'"

# =============================================================================
# SEQUENTIAL: label → align → cluster
# =============================================================================
rule label:
    """Assign anatomical region labels from the GeoJSON annotations"""
    input:
        expand(f"{WORKDIR}/.convert_{{region}}_done", region=CONVERT_REGIONS)
    output:
        touch(f"{WORKDIR}/.label_done")
    resources:
        mem_mb=config["label_mem_mb"],
        runtime=config["label_runtime"]
    shell:
        "cd {WORKDIR} && PYTHONPATH={PROJECT_ROOT}/src ../../../.venv/bin/python ../analysis_workflow.py label-all"

if ALIGNER == "mosaicfield":
    rule align:
        """MOSAICField nonlinear alignment (experimental)"""
        input:
            f"{WORKDIR}/.label_done"
        output:
            f"{DATA_DIR}/aligned.h5ad"
        resources:
            mem_mb=256000,
            runtime=480,
            gpu=1
        shell:
            "cd {WORKDIR} && PYTHONPATH={PROJECT_ROOT}/src ../../../.venv/bin/python ../run_mosaicfield_wrapper.py align --device cuda:0 --epochs 3000"

elif ALIGNER == "paste3":
    rule align:
        """PASTE3 pairwise alignment into 3D stack (flags per brain: config align_args)"""
        input:
            f"{WORKDIR}/.label_done"
        output:
            f"{DATA_DIR}/aligned.h5ad"
        params:
            args=config["align_args"]
        resources:
            mem_mb=config["align_mem_mb"],
            runtime=360,
            gpu=1
        shell:
            "cd {WORKDIR} && PYTHONPATH={PROJECT_ROOT}/src ../../../.venv/bin/python ../analysis_workflow.py align {params.args}"

rule cluster:
    """Leiden clustering"""
    input:
        f"{DATA_DIR}/aligned.h5ad"
    output:
        f"{DATA_DIR}/aligned_clustered.h5ad"
    threads: 16
    resources:
        mem_mb=128000,
        runtime=60,
        cpus_per_task=16
    shell:
        "cd {WORKDIR} && NUMBA_NUM_THREADS={threads} PYTHONPATH={PROJECT_ROOT}/src ../../../.venv/bin/python ../analysis_workflow.py cluster --n-jobs {threads}"

# =============================================================================
# PARALLEL: visualize_3d, render_video (both after cluster)
# =============================================================================
rule visualize_3d:
    """Interactive 3D plot"""
    input:
        f"{DATA_DIR}/aligned_clustered.h5ad"
    output:
        f"{WORKDIR}/figures/brain_3d.html"
    resources:
        mem_mb=8000,
        runtime=30
    shell:
        "cd {WORKDIR} && PYTHONPATH={PROJECT_ROOT}/src ../../../.venv/bin/python ../analysis_workflow.py visualize-3d --color-by leiden"

rule render_video:
    """Rotation video (colour per brain: config render_color_by)"""
    input:
        f"{DATA_DIR}/aligned_clustered.h5ad"
    output:
        f"{WORKDIR}/figures/brain_rotation.mp4"
    params:
        color_by=config["render_color_by"]
    resources:
        mem_mb=32000,
        runtime=120
    shell:
        "cd {WORKDIR} && PYTHONPATH={PROJECT_ROOT}/src ../../../.venv/bin/python ../analysis_workflow.py render-video --input-file aligned_clustered --color-by {params.color_by} --output brain_rotation"
