#!/usr/bin/env python3
"""Build the public-deposit staging tree (symlinks + manifests), read-only on sources.

Stdlib only (bimp_st.paths is stdlib too), so it runs under any Python >= 3.9
(use .venv/bin/python). Repo paths and the
canonical Tangram run come from bimp_st.paths, so the tree links whichever
checkout the script is run from. It never moves, renames or deletes a source
file: everything under the staging root is either a symlink to a source, a small
real file copied out of git (the GeoJSON annotations, which the cluster's main
checkout does not have on disk), or a manifest/README written here.

    PY=.venv/bin/python
    $PY scripts/deposit/build_staging.py  # symlinks + MANIFEST (md5 'pending')
    $PY scripts/deposit/build_staging.py --md5  # also md5 every file < 5 GB
    $PY scripts/deposit/build_staging.py --dry-run  # print the plan, write nothing

Layout (see the README.md written into each folder for what is in it and why):

    <root>/geo/      GEO series: snRNA FASTQs + per-sample matrices, Xenium output
                     bundles
    <root>/zenodo/   Zenodo records: atlas objects, Tangram outputs, annotations,
                     MALDI, micro-CT, webviz bundle, figure source data
    <root>/ncbi/     NCBI: BIMP_v4 assembly + v4.3 annotation (+ raw reads for SRA)
    <root>/MANIFEST.tsv, <root>/geo_metadata_draft.tsv

Canonical choices are encoded here, once. Superseded objects
(results/tangram/_superseded, *_imputed.h5ad of non-canonical runs, the
pre-nine-region per-brain atlas objects) are deliberately absent.
"""

import argparse
import csv
import glob
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bimp_st import paths  # noqa: E402
from bimp_st.site import get as site_path  # noqa: E402
from bimp_st.site import value as site_value  # noqa: E402

# Off-repo locations (bimp_st.site keys): the staging root, the snRNA-seq
# analysis tree, the genome assemblies and the raw MALDI exports.
ROOT = site_path("deposit_staging")
REPO = paths.REPO
IT = site_path("snrna_source")
GENOMES = site_path("genomes_dir")
MALDI_RAW = site_path("maldi_raw")
WEBVIZ_EXPORT = REPO / "viewer/pages/data"
MD5_LIMIT = 5 * 1024**3

XENIUM_RUNS = {
    "queen": REPO / "data/20251204__194847___20251204",
    "worker": REPO / "data/20260115__193720__20260115",
}
SN_RUNS = {  # cellranger multi run dir name, FASTQ dir, pool id
    "Trial1": (
        "Trial1_centralBrain_out/20260615_BIMP_QW_atlas_Trial1",
        "Trial1_cb",
        "BIMP_QW_snucAtlas_pool_A_1-4",
        "CentralBrain",
        "t1",
    ),
    "Trial2": (
        "Trial2_centralBrain_out/20260615_BIMP_QW_atlas_Trial2",
        "Trial2_cb",
        "BIMP_QW_snucAtlas_pool_A_5-8",
        "CentralBrain",
        "t2",
    ),
    "Trial3": (
        "Trial3_centralBrain_out/20260615_BIMP_QW_atlas_Trial3",
        "Trial3_cb",
        "BIMP_QW_snucAtlas_pool_A_9-12",
        "CentralBrain",
        "t3",
    ),
    "OpticLobe": (
        "opticLobe_out/20260615_BIMP_QW_atlas_OpticLobe",
        "opticLobe",
        "BIMP_QW_snucAtlas_pool_A_13-16",
        "OpticLobe",
        "OpticLobe",
    ),
}
CB_DIR = {
    "Trial1": "Trial1",
    "Trial2": "Trial2",
    "Trial3": "Trial3",
    "OpticLobe": "opticLobe",
}
OCM = {"Queen": "OB1|OB2", "Worker": "OB3|OB4"}

CANONICAL_RUN = paths.CANONICAL_TANGRAM.name.removesuffix("_tangram.h5ad")
assert CANONICAL_RUN.endswith("_s42"), CANONICAL_RUN
SEED_RUNS = [CANONICAL_RUN.removesuffix("_s42") + s for s in ("_s43", "_s44")]
SMALL_RUN_FILES = [
    "column_identity.csv",
    "eval_metrics.json",
    "gene_scores.csv",
    "gene_split.json",
    "train_genes.csv",
]

# Figure source tables (results-relative): the summary tables of the
# analyses plus the inputs of the fig:caste_comparison composer.
FIGURE_SOURCES = """
fig5_kc/kc_prob_mass_by_region.csv fig5_kc/morph_comp_by_region.csv
fig5_kc/morph_expectation_check.csv fig5_kc/morph_share_of_kc_mass.csv
fig5_kc/subzone_stats.csv figures/fig2/abundance_tangram_ct.csv
figures/fig2/atlas_vote_composition_by_caste.csv
region_composition/composition_by_region.csv
region_composition/composition_by_region_pooled.csv
region_composition/region_cell_counts.csv
figures/fig2/slice_composition.csv figures/fig2/sn_composition_by_caste.csv
figures/si/_cache_label_sensitivity_v2/label_sensitivity_obs.parquet
figures/si/_cache_label_sensitivity_v2/sccoda_combined.csv
figures/si/_cache_label_sensitivity_v2/sccoda_stitched.csv
figures/si/fig_s_kc_restricted_confidence_stats.json
figures/si/fig_s_label_sensitivity_agreement.csv
figures/si/fig_s_probe_clusters_stats.json
figures/si/fig_s_tangram_confidence_stats.json
figures/si/fig_s_vg_localization_stats.json
figures/si_percell_vs_vote/fig_s_percell_vs_vote_stats.json
handoff/A_leiden_stability/leiden_stability.csv
caste_de/B_caste_benchmark/caste_benchmark_counts.csv
handoff/C_antennal_lobe/adequacy/halo_summary.csv
handoff/C_antennal_lobe/adequacy/prevalence_under_hypotheses.csv
handoff/C_antennal_lobe/step1_al_composition_agreement.csv
handoff/C_antennal_lobe/step1_prevalence_scenarios.csv
handoff/C_antennal_lobe/step2_classifier_metrics.csv
handoff/C_antennal_lobe/step3_classifier_metrics.csv
handoff/C_antennal_lobe/tracing_by_section.csv handoff/tk_carryover/depth_bins.csv
handoff/tk_carryover/summary.csv nine_region/coverage.json
reframe/caste_sd_ratio/sd_ratios.csv
reframe/celltype_anatomy/blast_radius_crosstab.csv
reframe/celltype_anatomy/counts_cluster_vote.csv
reframe/celltype_anatomy/counts_percell_canonical17.csv
reframe/celltype_territory_check/summary.json reframe/cluster_vote_null/summary.csv
reframe/cluster_vote_purity/hierarchical_vote_clusters.csv
reframe/cluster_vote_purity/summary.json reframe/foldwise_leiden_v2/foldwise_leiden.csv
reframe/foldwise_leiden_v2_res2.0/foldwise_leiden.csv
reframe/kc5pop_negative_control/negcontrol_comparison.txt
reframe/kc_subcluster_separability/summary.json
reframe/kc_threshold/definition_overlap.csv reframe/kc_threshold/threshold_sweep.csv
reframe/loocv_null/summary.json reframe/loocv_null_0af3056_20261020/summary.json
tangram/loocv_0af3056_20261020/combined_clustered_loocv_scores.csv
tangram/loocv_0af3056_20261020/combined_clustered_loocv_run.json
reframe/mapping_ceiling/ceiling_per_type.csv reframe/mapping_ceiling/summary.json
reframe/panel_concordance/bulk.csv reframe/panel_concordance/class_agreement.csv
reframe/panel_concordance/per_cell_type.csv reframe/panel_concordance/per_gene.csv
reframe/posterior_resolution/summary.json
reframe/region_prediction_3d_v2/cross_z_diagnostics.csv
reframe/region_prediction_3d_v2/summary_2d_vs_3d.csv
reframe/region_prediction_v3/per_region_f1.csv
reframe/region_prediction_v3/scheme_comparison.csv
reframe/snrna_region_feasibility/celltype_x_compartment_crosstab.csv
reframe/spatial_coherence_c17/summary.csv
reframe/spatial_relabel_test/vote_stability_decomposition.csv
reframe/tangram_seed_stability/percell_churn_by_group.csv
reframe/tangram_seed_stability/seed_accuracy.csv
reframe/tangram_seed_stability/seed_pair_agreement.csv
reframe/tangram_seed_stability/summary.csv
reframe/tk_3d_spillover_v2/distance_gradient_summary.csv
reframe/tk_3d_spillover_v2/per_section_fold.csv
reframe/tk_3d_spillover_v2/render_optic_tk_counts.csv
reframe/xenium_pct_expressing_confound/summary.json
regional_degs/composition_by_region.csv
""".split()


class Plan:
    def __init__(self):
        self.rows = []  # (staging_rel, source, description)
        self.copies = []  # (staging_rel, bytes, description)
        self.missing = []  # (staging_rel, source)

    def link(self, rel, src, desc):
        src = Path(src)
        if not src.exists():
            self.missing.append((rel, str(src)))
            return
        self.rows.append((rel, src, desc))

    def copy_bytes(self, rel, data, desc, source):
        self.copies.append((rel, data, desc, source))


def one(pattern, exclude=()):
    hits = [
        p
        for p in glob.glob(pattern)
        if not any(x in os.path.basename(p) for x in exclude)
    ]
    if len(hits) != 1:
        raise SystemExit(
            f"expected exactly one match for {pattern} (excluding {exclude}), got "
            f"{hits}"
        )
    return hits[0]


# ---------------------------------------------------------------------------- GEO
def plan_geo(p: Plan):
    for run, (rdir, fqdir, pool, tissue, _trial) in SN_RUNS.items():
        for rn in ("1", "4"):
            f = IT / "data" / fqdir / f"3697__{pool}-read-{rn}.fastq.gz"
            p.link(
                f"geo/snrna/fastq/{run}/{f.name}",
                f,
                f"snRNA-seq FASTQ, GEM well {run} ({tissue}; queen OB1|OB2 + worker "
                "OB3|OB4 "
                f"on-chip multiplexed), read-{rn} "
                f"({'R1 28 bp barcode+UMI' if rn == '1' else 'R2 94 bp cDNA'})",
            )
        out = IT / "out/aligned" / rdir / "outs"
        p.link(
            f"geo/snrna/processed/{run}/multi_config.csv",
            out / "config.csv",
            f"cellranger multi config for {run} (OCM barcode -> sample)",
        )
        p.link(
            f"geo/snrna/processed/{run}/qc_library_metrics.csv",
            out / "qc_library_metrics.csv",
            f"cellranger multi library QC for {run}",
        )
        for caste in ("Queen", "Worker"):
            sid = f"{run}_BIMP_{caste}"
            so = out / "per_sample_outs" / sid
            base = f"geo/snrna/processed/{run}/{sid}"
            p.link(
                f"{base}/sample_filtered_feature_bc_matrix.h5",
                so / "sample_filtered_feature_bc_matrix.h5",
                f"cellranger 10.0.0 filtered matrix, {sid} (BIMP_v4 + OGS v4.3 "
                "reference)",
            )
            p.link(
                f"{base}/sample_raw_feature_bc_matrix.h5",
                so / "sample_raw_feature_bc_matrix.h5",
                f"cellranger 10.0.0 raw matrix, {sid} (CellBender input)",
            )
            p.link(
                f"{base}/metrics_summary.csv",
                so / "metrics_summary.csv",
                f"cellranger metrics, {sid}",
            )
            cbd = IT / "out/cellbender" / CB_DIR[run] / caste.lower()
            cb = one(
                str(cbd / "*CellbenderOut_filtered.h5"),
                exclude=("halfLR", "quarterLR", "force"),
            )
            p.link(
                f"{base}/cellbender_filtered.h5",
                cb,
                "CellBender remove-background (default params) filtered output, "
                f"{sid}; "
                f"source file {os.path.basename(cb)}",
            )
            p.link(
                f"{base}/cellbender_metrics.csv",
                cb.replace("_filtered.h5", "_metrics.csv"),
                f"CellBender metrics, {sid}",
            )
    for name, desc in (
        (
            "BIMP_atlas_qw_combined.h5ad",
            "snRNA reference used by the manuscript (CB+OL, 73,463 nuclei x 10,469 "
            "genes; layers['counts'] raw)",
        ),
        (
            "BIMP_atlas_qw_CentralBrain.h5ad",
            "annotated central-brain atlas (3 trials x 2 castes; n=3 DESeq2 input)",
        ),
        (
            "BIMP_atlas_qw_OpticLobe.h5ad",
            "annotated optic-lobe atlas (1 queen + 1 worker)",
        ),
    ):
        p.link(f"geo/snrna/processed/atlas/{name}", REPO / "data/sn" / name, desc)

    for brain, run in XENIUM_RUNS.items():
        for d in sorted(run.glob("output-XETG00151__*")):
            p.link(
                f"geo/xenium/{brain}/{d.name}",
                d,
                f"Xenium Onboard Analysis output bundle ({brain}; directory; tar "
                "before upload)",
            )


# ---------------------------------------------------------------------------- Zenodo
def plan_zenodo(p: Plan):
    T = paths.TANGRAM_DIR
    a = "zenodo/Z1_atlas/source_as_is"
    p.link(
        f"{a}/{CANONICAL_RUN}_tangram.h5ad",
        T / f"{CANONICAL_RUN}_tangram.h5ad",
        "CANONICAL combined atlas + Tangram mapping (paths.CANONICAL_TANGRAM). NOT "
        "release-ready: see README traps",
    )
    for f in SMALL_RUN_FILES:
        p.link(
            f"{a}/{CANONICAL_RUN}_{f}",
            T / f"{CANONICAL_RUN}_{f}",
            f"canonical Tangram run sidecar: {f}",
        )
    for run in SEED_RUNS:
        p.link(
            f"{a}/seed_replicates/{run}_tangram.h5ad",
            T / f"{run}_tangram.h5ad",
            "seed replicate of the canonical configuration (stability check)",
        )
        for f in ("eval_metrics.json", "gene_scores.csv"):
            p.link(
                f"{a}/seed_replicates/{run}_{f}",
                T / f"{run}_{f}",
                f"seed replicate sidecar: {f}",
            )
    for name, desc in (
        (
            "combined_clustered_central_colfix_tangram.h5ad",
            "paths.CANONICAL_CENTRAL (central-brain-only reference run; "
            "label-sensitivity SI)",
        ),
        (
            "combined_clustered_optic_colfix_tangram.h5ad",
            "paths.CANONICAL_OPTIC (optic-lobe-only reference run; label-sensitivity "
            "SI)",
        ),
        (
            "combined_clustered_stitched_colfix_tangram.h5ad",
            "paths.CANONICAL_STITCHED (region-stitched labels composed from the "
            "central and optic runs)",
        ),
    ):
        p.link(f"{a}/region_reference_runs/{name}", T / name, desc)
    NR = REPO / "results/nine_region"
    for b in ("queen", "worker"):
        p.link(
            f"{a}/{b}_aligned_clustered_9region.h5ad",
            NR / f"atlas/{b}_aligned_clustered_9region.h5ad",
            f"{b} per-brain atlas with eight-region in_* flags (annotations_v4). "
            "obsm['spatial'] is the PASTE3 frame",
        )
    p.link(
        f"{a}/region_flags.parquet",
        NR / "region_flags.parquet",
        "per-cell region flags keyed by (slice_name, cell_id), nullable booleans (NA "
        "= untraced)",
    )
    p.link(
        f"{a}/coverage.json",
        NR / "coverage.json",
        "region coverage counts (eight/seven-region blocks)",
    )
    p.link(
        f"{a}/region_crops_sidecars",
        NR / "crops",
        "per-crop region-flag sidecars (directory)",
    )
    # raw per-crop h5ads are deliberately NOT staged (2.8-3.5 GB each because
    # uns['image'] embeds the morphology crop); the public export carries their
    # raw-um coordinates instead.

    m = "zenodo/Z2_maldi"
    MD = REPO / "pipelines/maldi"
    for sid in ("queen_80524", "worker_80642"):
        p.link(
            f"{m}/processed/aligned_{sid}.h5ad",
            MD / f"data/aligned_{sid}.h5ad",
            f"MALDI-MSI per-spot peptide intensities, PASTE3-aligned, {sid} "
            "(fig:neuropeptide_spatial input)",
        )
        p.link(
            f"{m}/processed/volume_{sid}.zarr",
            MD / f"data/volume_{sid}.zarr",
            f"MALDI 20 um voxel volume, {sid} (directory; "
            "fig:neuropeptide_spatial input)",
        )
        n = sid.split("_")[1]
        ion = MD / "raw_extraction/ion_out"
        for f in (
            f"aligned_panel_princeton_{n}.h5ad",
            f"features_princeton_{n}_ppm15.parquet",
            f"ionimg_princeton_{n}_ppm15.h5",
        ):
            p.link(
                f"{m}/processed/princeton_pipeline/{f}",
                ion / f,
                f"imzML-derived sNPF@974.5891 / Tk2@993.4757 panel, {sid} "
                "(fig:neuropeptide_spatial caste difference)",
            )
        p.link(
            f"{m}/raw_imzml/princeton_{n}.imzML",
            MALDI_RAW / f"Princeton_imzmls/princeton_{n}.imzML",
            f"raw MALDI imzML header, {sid}",
        )
        p.link(
            f"{m}/raw_imzml/princeton_{n}.ibd",
            MALDI_RAW / f"Princeton_imzmls/princeton_{n}.ibd",
            f"raw MALDI imzML binary, {sid} (exceeds Zenodo 50 GB record limit on "
            "its own)",
        )

    r = "zenodo/Z3_resources"
    p.link(
        f"{r}/microct/Z1224-18TF_Segmentation-Interpolated.tif",
        REPO / "data/microct/Z1224-18TF_Segmentation-Interpolated.tif",
        "micro-CT worker brain volume (THIRD-PARTY, K. Pfeiffer, unpublished): "
        "deposit only with written permission",
    )
    p.link(
        f"{r}/webviz_bundle",
        WEBVIZ_EXPORT,
        "exported web-atlas data (cells, 9,710 per-gene files, histology tiles); "
        "directory",
    )
    for rel in FIGURE_SOURCES:
        p.link(
            f"{r}/figure_source_data/{rel}",
            REPO / "results" / rel,
            "manuscript figure/number source table",
        )
    for f in sorted(glob.glob(str(REPO / "results/regional_degs/degs_*.csv"))):
        p.link(
            f"{r}/figure_source_data/regional_degs/{os.path.basename(f)}",
            f,
            "regional DE table",
        )

    # GeoJSON annotations: tracked in git on origin/main, absent from the cluster's
    # (stale) main checkout on disk, so extract them as real files.
    for brain in ("queen", "worker"):
        path = f"data/annotations/{brain}/v4"
        sha = (
            subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "origin/main"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .strip()
        )
        blob = subprocess.run(
            ["git", "-C", str(REPO), "archive", sha, path],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(blob)) as tf:
            for mem in tf.getmembers():
                if mem.isfile() and mem.name.endswith(".geojson"):
                    data = tf.extractfile(mem).read()
                    p.copy_bytes(
                        f"zenodo/Z1_atlas/annotations_v4/{brain}/{os.path.basename(mem.name)}",
                        data,
                        f"hand-drawn region polygons, {brain} (copied from git)",
                        f"git:{sha}:{mem.name}",
                    )


# ---------------------------------------------------------------------------- NCBI
def plan_ncbi(p: Plan):
    G = IT / "src/reference_genome/genome"
    p.link(
        "ncbi/assembly/BIMP_genome_v4.fasta",
        G / "BIMP_genome_v4.fasta",
        "BIMP_v4 assembly, 19 scaffolds (the cellranger mkref input)",
    )
    p.link(
        "ncbi/annotation/BIMP_OGS_v4.3.gtf",
        G / "BIMP_OGS_v4.3.gtf",
        "BIMP OGS v4.3 gene models (GTF; NCBI needs GFF3/.sqn via table2asn)",
    )
    p.link(
        "ncbi/sra_candidates/hic_3346__Hi-C_BIMP_Brain_3__BIMP-HI-C-Full-read-1.fastq.gz",
        GENOMES
        / "2025_BIMPv3+CAL/hic/3346__Hi-C_BIMP_Brain_3__BIMP-HI-C-Full-read-1.fastq.gz",
        "CANDIDATE proximity-ligation reads for the assembly (to be "
        "confirmed)",
    )
    p.link(
        "ncbi/sra_candidates/hic_3346__Hi-C_BIMP_Brain_3__BIMP-HI-C-Full-read-4.fastq.gz",
        GENOMES
        / "2025_BIMPv3+CAL/hic/3346__Hi-C_BIMP_Brain_3__BIMP-HI-C-Full-read-4.fastq.gz",
        "CANDIDATE proximity-ligation reads for the assembly (to be "
        "confirmed)",
    )
    p.link(
        "ncbi/sra_candidates/pacbio_5875_hifi_reads",
        GENOMES / "2025_BIMPv3+CAL/hifi/5875__hifi_reads",
        "CANDIDATE PacBio HiFi reads (directory; to be confirmed)",
    )
    p.link(
        "ncbi/sra_candidates/pacbio_5876_hifi_reads",
        GENOMES / "2025_UMB+CAL_secondCell/5876__hifi_reads",
        "CANDIDATE PacBio HiFi reads, second cell (directory; to be confirmed)",
    )


# ---------------------------------------------------------------------------- GEO sheet
def geo_sheet():
    # Author-confirmed metadata with no default: an unset key stops here with
    # an error naming it (bimp_st.site; template config/site.example.toml).
    individuals = site_value("snrna_individuals_per_sample")
    instrument = site_value("snrna_instrument")
    rows = []
    # per-crop cell counts in the canonical atlas (slice_name -> n), written by
    # a one-off read of obs['slice_name']; optional, the column stays blank without it
    atlas_cells = {}
    f = ROOT / "_tools/atlas_cells_per_slice.csv"
    if f.exists():
        for r in csv.DictReader(open(f)):
            atlas_cells[r["slice_name"]] = int(r["n_cells"])
    hdr = [
        "sample_name",
        "title",
        "source_name",
        "organism",
        "assay",
        "caste",
        "trial_or_batch",
        "tissue",
        "brain_region",
        "individuals",
        "sex",
        "age",
        "library_strategy",
        "chemistry",
        "gem_well_or_slide",
        "multiplex_barcodes",
        "cellranger_sample_id",
        "raw_files",
        "raw_file_md5",
        "processed_files",
        "instrument",
        "read_lengths",
        "reference",
        "cells_called_by_pipeline",
        "cells_in_published_object",
        "notes",
    ]
    md5 = {}
    for line in (IT / "data/md5sum.txt").read_text().splitlines():
        h, n = line.split()
        md5[n] = h
    atlas_counts = {
        "Queen_t1": 6208,
        "Queen_t2": 8971,
        "Queen_t3": 8825,
        "Worker_t1": 9681,
        "Worker_t2": 8368,
        "Worker_t3": 11177,
        "Queen_OpticLobe": 12312,
        "Worker_OpticLobe": 7921,
    }  # BIMP_atlas_qw_combined.h5ad obs['sample']
    for run, (rdir, _fqdir, pool, tissue, trial) in SN_RUNS.items():
        out = IT / "out/aligned" / rdir / "outs"
        ocm_cells = {}
        for r in csv.reader(open(out / "qc_library_metrics.csv")):
            if len(r) == 6 and r[4] == "Cells per OCM barcode":
                ocm_cells[r[3]] = int(r[5])
        fq = [f"3697__{pool}-read-1.fastq.gz", f"3697__{pool}-read-4.fastq.gz"]
        for caste in ("Queen", "Worker"):
            sid = f"{run}_BIMP_{caste}"
            cells = None
            for r in csv.reader(
                open(out / "per_sample_outs" / sid / "metrics_summary.csv")
            ):
                if len(r) == 6 and r[0] == "Cells" and r[4] == "Cells":
                    cells = r[5]
            obs = ocm_cells
            bcs = OCM[caste]
            rows.append(
                {
                    "sample_name": f"snRNA_{caste}_{trial}",
                    "title": f"B. impatiens {caste.lower()} {tissue} snRNA-seq {trial}",
                    "source_name": f"{tissue} nuclei",
                    "organism": "Bombus impatiens",
                    "assay": "snRNA-seq",
                    "caste": caste.lower(),
                    "trial_or_batch": trial,
                    "tissue": "brain",
                    "brain_region": "central brain"
                    if tissue == "CentralBrain"
                    else "optic lobes",
                    "individuals": individuals,
                    "sex": "female",
                    "age": "worker ~10 d post-eclosion; queen = colony foundress",
                    "library_strategy": "RNA-Seq (single-nucleus)",
                    "chemistry": "10x GEM-X Single Cell 3' v4 (polyA) OCM",
                    "gem_well_or_slide": f"{run} ({pool})",
                    "multiplex_barcodes": f"{bcs} (cells/OCM: "
                    + ", ".join(f"{b}={obs.get(b)}" for b in bcs.split("|"))
                    + ")",
                    "cellranger_sample_id": sid,
                    "raw_files": "; ".join(fq)
                    + "  [SHARED with the other caste of this GEM well]",
                    "raw_file_md5": "; ".join(md5.get(f, "?") for f in fq),
                    "processed_files": (
                        f"{sid}/sample_filtered_feature_bc_matrix.h5; "
                        f"{sid}/cellbender_filtered.h5"
                    ),
                    "instrument": instrument,
                    "read_lengths": "R1 28; R2 94",
                    "reference": "BIMP_v4 + OGS v4.3 (cellranger 10.0.0 mkref)",
                    "cells_called_by_pipeline": cells,
                    "cells_in_published_object": atlas_counts[f"{caste}_{trial}"],
                    "notes": (
                        "Index-read FASTQs (I1/I2) not retained in delivery; core "
                        "demultiplexed by dual index."
                    ),
                }
            )
    for brain, run in XENIUM_RUNS.items():
        crops = {}
        datadir = REPO / (
            "pipelines/wholebrain/queen/data"
            if brain == "queen"
            else "pipelines/wholebrain/worker/data"
        )
        for h in sorted(datadir.glob("output-XETG00151__*.h5ad")):
            stem = h.name[:-5]
            region_key = tuple(stem.split("__")[1:3])  # (slide, Region_N)
            crops.setdefault(region_key, []).append(stem)
        for d in sorted(run.glob("output-XETG00151__*")):
            exp = json.loads((d / "experiment.xenium").read_text())
            key = tuple(d.name.split("__")[1:3])
            stems = crops.get(key, [])
            n_pub = sum(atlas_cells.get(st, 0) for st in stems) if atlas_cells else ""
            rows.append(
                {
                    "sample_name": (
                        f"xenium_{brain}_{exp['slide_id']}_{exp['region_name']}"
                    ),
                    "title": (
                        f"B. impatiens {brain} brain Xenium slide {exp['slide_id']} "
                        f"{exp['region_name']}"
                    ),
                    "source_name": "fresh-frozen brain, 10 um serial coronal sections",
                    "organism": "Bombus impatiens",
                    "assay": "Xenium In Situ",
                    "caste": brain,
                    "trial_or_batch": exp["run_name"].strip("_"),
                    "tissue": "whole brain",
                    "brain_region": "whole brain (serial sections)",
                    "individuals": "1",
                    "sex": "female",
                    "age": "worker ~10 d post-eclosion; queen = colony foundress",
                    "library_strategy": "OTHER (in situ imaging)",
                    "chemistry": (
                        f"Xenium chemistry {exp.get('chemistry_version')}; "
                        f"{exp.get('segmentation_stain')}"
                    ),
                    "gem_well_or_slide": (
                        f"slide {exp['slide_id']}, {exp['region_name']}"
                    ),
                    "multiplex_barcodes": "",
                    "cellranger_sample_id": "",
                    "raw_files": f"{d.name}.tar",
                    "raw_file_md5": "deferred",
                    "processed_files": (
                        "cell_feature_matrix.h5; cells.parquet; "
                        "cell_boundaries.parquet (in bundle)"
                    ),
                    "instrument": (
                        f"Xenium Analyzer {exp['instrument_sn']} (sw "
                        f"{exp['instrument_sw_version']}, "
                        f"{exp['analysis_sw_version']})"
                    ),
                    "read_lengths": "",
                    "reference": (
                        f"custom panel {exp['panel_name']} "
                        f"({exp['panel_design_id']}, "
                        f"{exp['panel_num_targets_custom']} genes; BIMP vNP3.1 IDs)"
                    ),
                    "cells_called_by_pipeline": exp.get("num_cells"),
                    "cells_in_published_object": n_pub,
                    "notes": "sections (crops) in this capture region: "
                    f"{','.join(st.rsplit('_', 1)[-1] for st in stems) or '?'}; "
                    "experiment.xenium says "
                    f"panel_organism={exp.get('panel_organism')} "
                    "(designer template, not the species)",
                }
            )
    return hdr, rows


# ---------------------------------------------------------------------------- write
def dir_size(path):
    total, n = 0, 0
    for dp, _, fs in os.walk(path, followlinks=False):
        for f in fs:
            try:
                total += os.lstat(os.path.join(dp, f)).st_size
                n += 1
            except OSError:
                pass
    return total, n


def core_md5s():
    """basename -> md5 from the md5sum.txt files the sequencing core delivered."""
    out = {}
    for f in (IT / "data/md5sum.txt", GENOMES / "2025_BIMPv3+CAL/hic/md5sum.txt"):
        if f.exists():
            for line in f.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2:
                    out[os.path.basename(parts[1])] = parts[0]
    return out


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(16 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--md5", action="store_true", help="md5 every regular file < 5 GB")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    p = Plan()
    plan_geo(p)
    plan_zenodo(p)
    plan_ncbi(p)
    if args.dry_run:
        for rel, src, _ in p.rows:
            print(f"link {rel} -> {src}")
        for rel, data, _, _ in p.copies:
            print(f"copy {rel} ({len(data)} B)")
        for rel, src in p.missing:
            print(f"MISSING {rel} <- {src}")
        return

    root = args.root
    old = {}
    man = root / "MANIFEST.tsv"
    if man.exists():  # keep md5s already computed
        for r in csv.DictReader(open(man), delimiter="\t"):
            old[r["staging_path"]] = r
    core = core_md5s()
    out_rows = []
    for rel, src, desc in p.rows:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            if os.readlink(dst) != str(src):
                raise SystemExit(
                    f"{dst} already links to {os.readlink(dst)}, not {src}; refusing "
                    "to replace"
                )
        elif dst.exists():
            raise SystemExit(f"{dst} exists and is not a symlink; refusing to touch it")
        else:
            os.symlink(src, dst)
        if src.is_dir():
            size, n = dir_size(src)
            md5 = f"deferred (directory, {n} files)"
        else:
            size = src.stat().st_size
            prev = old.get(rel, {})
            if (
                prev.get("md5", "").strip()
                and len(prev["md5"]) == 32
                and prev.get("size_bytes") == str(size)
            ):
                md5 = prev["md5"]
            elif src.name in core:
                md5 = core[src.name]  # from the sequencing core's md5sum.txt
            elif size >= MD5_LIMIT:
                md5 = "deferred"
            elif args.md5:
                md5 = md5sum(src)
            else:
                md5 = "pending"
        out_rows.append([rel, str(src), size, md5, desc])
    for rel, data, desc, source in p.copies:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_bytes(data)
        out_rows.append([rel, source, len(data), hashlib.md5(data).hexdigest(), desc])
    with open(man, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["staging_path", "source_path", "size_bytes", "md5", "description"])
        w.writerows(out_rows)
    for top in ("geo", "zenodo", "ncbi"):
        sub = [r for r in out_rows if r[0].startswith(top + "/")]
        with open(root / top / "MANIFEST.tsv", "w", newline="") as fh:
            w = csv.writer(fh, delimiter="\t", lineterminator="\n")
            w.writerow(
                ["staging_path", "source_path", "size_bytes", "md5", "description"]
            )
            w.writerows(sub)
    hdr, rows = geo_sheet()
    with open(root / "geo_metadata_draft.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(
        f"{len(p.rows)} links, {len(p.copies)} copied files, {len(p.missing)} "
        "missing sources"
    )
    for rel, src in p.missing:
        print(f"  MISSING {rel} <- {src}", file=sys.stderr)


if __name__ == "__main__":
    main()
