#!/usr/bin/env python3
"""
Combined analysis of queen and worker brains with shared embedding space.

Usage:
    python combined_analysis.py combine              # Combine aligned data from both
                                                     # brains
    python combined_analysis.py cluster              # Cluster combined data
    python combined_analysis.py differential         # Differential composition analysis

Note: the Snakefile maps with tangram_mapping.py, not with the Tangram
commands here (tangram, visualize-tangram, tangram-qc).
"""

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import typer

app = typer.Typer(help="Combined queen/worker brain analysis")

# =============================================================================
# CONFIG
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
QUEEN_DIR = REPO_ROOT / "pipelines/wholebrain/queen"
WORKER_DIR = REPO_ROOT / "pipelines/wholebrain/worker"
OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
FIGURES_DIR = OUTPUT_DIR / "figures"
RESULTS_ROOT = REPO_ROOT / "results"

SNRNA_REF = REPO_ROOT / "data/sn/BIMP_atlas_qw_combined.h5ad"
# Gene mapping helpers live in the shared package — single source of truth.
from bimp_st.utils.gene_mapping import (  # noqa: E402
    load_gene_mapping,
    map_spatial_genes_to_v43,
)

# =============================================================================
# COMMANDS
# =============================================================================


@app.command()
def combine(
    queen_file: str = typer.Option(
        "aligned_clustered", help="Queen input file (without .h5ad)"
    ),
    worker_file: str = typer.Option(
        "aligned_clustered", help="Worker input file (without .h5ad)"
    ),
    output_file: str = typer.Option(
        "combined", help="Output file name (without .h5ad)"
    ),
):
    """Combine aligned data from both brains into single AnnData."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    queen_path = QUEEN_DIR / "data" / f"{queen_file}.h5ad"
    worker_path = WORKER_DIR / "data" / f"{worker_file}.h5ad"

    # Check files exist
    if not queen_path.exists():
        typer.echo(f"Queen file not found: {queen_path}")
        raise typer.Exit(1)
    if not worker_path.exists():
        typer.echo(f"Worker file not found: {worker_path}")
        typer.echo(
            "Run the worker pipeline first: cd pipelines/wholebrain/worker "
            "&& snakemake --profile ../../config/profiles/slurm"
        )
        raise typer.Exit(1)

    typer.echo(f"Loading queen: {queen_path.name}")
    queen = sc.read_h5ad(queen_path)
    queen.obs["brain"] = "queen"
    queen.obs["brain_id"] = "20251204"
    typer.echo(f"  Cells: {queen.n_obs:,}")

    typer.echo(f"Loading worker: {worker_path.name}")
    worker = sc.read_h5ad(worker_path)
    worker.obs["brain"] = "worker"
    worker.obs["brain_id"] = "20260115"
    typer.echo(f"  Cells: {worker.n_obs:,}")

    # Find common genes
    common_genes = list(set(queen.var_names) & set(worker.var_names))
    typer.echo(f"Common genes: {len(common_genes)}")

    queen = queen[:, common_genes].copy()
    worker = worker[:, common_genes].copy()

    # Concatenate
    typer.echo("Concatenating...")
    combined = ad.concat(
        [queen, worker], join="inner", label="brain_source", keys=["queen", "worker"]
    )

    # Store original cluster assignments
    if "leiden" in queen.obs.columns:
        combined.obs["leiden_original"] = combined.obs["leiden"].copy()

    # Clear computed embeddings (will recompute on combined data)
    keys_to_remove = ["X_pca", "X_umap"]
    for key in keys_to_remove:
        if key in combined.obsm:
            del combined.obsm[key]

    # Clear neighbors graph
    if "neighbors" in combined.uns:
        del combined.uns["neighbors"]
    for key in list(combined.obsp.keys()):
        del combined.obsp[key]

    typer.echo(f"Combined cells: {combined.n_obs:,}")
    typer.echo(f"  Queen: {(combined.obs['brain'] == 'queen').sum():,}")
    typer.echo(f"  Worker: {(combined.obs['brain'] == 'worker').sum():,}")

    output_path = DATA_DIR / f"{output_file}.h5ad"
    combined.write_h5ad(output_path)
    typer.echo(f"Saved: {output_path}")


@app.command()
def cluster(
    input_file: str = typer.Option("combined", help="Input h5ad file"),
    resolution: float = typer.Option(1.0, help="Leiden resolution"),
    n_neighbors: int = typer.Option(15, help="Number of neighbors"),
    n_pcs: int = typer.Option(50, help="Number of PCs"),
    n_jobs: int = typer.Option(-1, help="Parallel jobs"),
    min_dist: float = typer.Option(0.1, help="UMAP min_dist"),
):
    """Cluster combined data with shared UMAP embedding."""
    import os

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    n_cpus = os.cpu_count() or 4
    if n_jobs == -1:
        n_jobs = n_cpus
    sc.settings.n_jobs = n_jobs

    input_path = DATA_DIR / f"{input_file}.h5ad"
    typer.echo(f"Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)
    typer.echo(f"  Cells: {adata.n_obs:,}, Genes: {adata.n_vars}")

    # Normalize if needed
    if "log1p" not in adata.uns:
        typer.echo("Normalizing...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    # HVGs
    typer.echo("Finding HVGs...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, batch_key="brain")

    # PCA. random_state=0 throughout this block is scanpy 1.11's default,
    # written out so the seed is explicit and a scanpy default change cannot
    # move the atlas clustering.
    typer.echo("Running PCA...")
    sc.tl.pca(adata, n_comps=n_pcs, random_state=0)

    # Optional: Harmony batch correction
    # typer.echo("Running Harmony batch correction...")
    # sc.external.pp.harmony_integrate(adata, "brain")

    # Neighbors
    typer.echo("Building neighbor graph...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=0)

    # Leiden
    typer.echo(f"Clustering (resolution={resolution})...")
    sc.tl.leiden(
        adata,
        resolution=resolution,
        flavor="igraph",
        n_iterations=2,
        directed=False,
        random_state=0,
    )
    typer.echo(f"  Found {adata.obs['leiden'].nunique()} clusters")

    # UMAP
    typer.echo("Computing UMAP...")
    sc.tl.umap(adata, min_dist=min_dist, random_state=0)

    # Save
    output_path = DATA_DIR / f"{input_file}_clustered.h5ad"
    adata.write_h5ad(output_path)
    typer.echo(f"Saved: {output_path}")

    # Generate plots
    _plot_combined_umap(adata, input_file)


def _plot_combined_umap(adata, prefix: str):
    """Generate UMAP plots colored by brain and cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # By brain
    sc.pl.umap(
        adata,
        color="brain",
        ax=axes[0],
        show=False,
        palette={"queen": "#DAA520", "worker": "#4A90D9"},
        title="Brain Identity",
    )

    # By cluster
    sc.pl.umap(
        adata,
        color="leiden",
        ax=axes[1],
        show=False,
        legend_loc="on data",
        legend_fontsize=6,
        title="Leiden Clusters",
    )

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / f"{prefix}_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"Saved: {prefix}_umap.png")

    # Cluster composition by brain
    _plot_cluster_composition(adata, prefix)


def _plot_cluster_composition(adata, prefix: str):
    """Plot cluster composition split by brain."""
    # Count cells per cluster per brain
    counts = adata.obs.groupby(["leiden", "brain"]).size().unstack(fill_value=0)

    # Normalize to proportions within each brain
    props = counts.div(counts.sum(axis=0), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Absolute counts
    counts.plot(kind="bar", ax=axes[0], color=["#DAA520", "#4A90D9"])
    axes[0].set_xlabel("Cluster")
    axes[0].set_ylabel("Cell Count")
    axes[0].set_title("Cells per Cluster by Brain")
    axes[0].legend(title="Brain")
    axes[0].tick_params(axis="x", rotation=45)

    # Proportions
    props.plot(kind="bar", ax=axes[1], color=["#DAA520", "#4A90D9"])
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Proportion")
    axes[1].set_title("Cluster Proportions by Brain")
    axes[1].legend(title="Brain")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"{prefix}_cluster_composition.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    typer.echo(f"Saved: {prefix}_cluster_composition.png")


@app.command()
def differential(
    input_file: str = typer.Option("combined_clustered", help="Input clustered h5ad"),
    groupby: str = typer.Option("leiden", help="Grouping for composition analysis"),
):
    """Differential composition analysis between queen and worker."""
    from scipy import stats

    input_path = DATA_DIR / f"{input_file}.h5ad"
    typer.echo(f"Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)

    # Get counts per cluster per brain
    counts = adata.obs.groupby(["brain", groupby]).size().unstack(fill_value=0)

    # Normalize to proportions
    props = counts.div(counts.sum(axis=1), axis=0)

    typer.echo("\nCluster proportions:")
    typer.echo(props.to_string())

    # Chi-square test for overall difference
    chi2, pval, dof, expected = stats.chi2_contingency(counts.values)
    typer.echo(f"\nChi-square test: chi2={chi2:.2f}, p={pval:.2e}")

    # Per-cluster fold changes
    queen_props = props.loc["queen"]
    worker_props = props.loc["worker"]

    results = pd.DataFrame(
        {
            "cluster": queen_props.index,
            "queen_prop": queen_props.values,
            "worker_prop": worker_props.values,
            "fold_change": (worker_props.values + 0.001) / (queen_props.values + 0.001),
            "diff": worker_props.values - queen_props.values,
        }
    )
    results = results.sort_values("fold_change", ascending=False)

    typer.echo("\nDifferential composition (worker vs queen):")
    typer.echo(results.to_string(index=False))

    # Save results
    results.to_csv(DATA_DIR / f"{input_file}_differential.csv", index=False)
    typer.echo(f"\nSaved: {input_file}_differential.csv")

    # Plot
    _plot_differential(results, input_file)


def _plot_differential(results: pd.DataFrame, prefix: str):
    """Plot differential composition."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Fold change bar plot
    colors = ["#DAA520" if fc < 1 else "#4A90D9" for fc in results["fold_change"]]
    axes[0].barh(
        results["cluster"].astype(str), np.log2(results["fold_change"]), color=colors
    )
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.5)
    axes[0].set_xlabel("log2(Fold Change) - Worker vs Queen")
    axes[0].set_ylabel("Cluster")
    axes[0].set_title("Differential Cluster Abundance")

    # Scatter plot: queen vs worker proportions
    axes[1].scatter(results["queen_prop"], results["worker_prop"], s=80, alpha=0.7)
    for _, row in results.iterrows():
        axes[1].annotate(
            str(row["cluster"]),
            (row["queen_prop"], row["worker_prop"]),
            fontsize=8,
            ha="center",
            va="bottom",
        )
    max_val = max(results["queen_prop"].max(), results["worker_prop"].max()) * 1.1
    axes[1].plot([0, max_val], [0, max_val], "k--", alpha=0.3)
    axes[1].set_xlabel("Queen Proportion")
    axes[1].set_ylabel("Worker Proportion")
    axes[1].set_title("Cluster Proportions: Queen vs Worker")

    plt.tight_layout()
    fig.savefig(
        FIGURES_DIR / f"{prefix}_differential.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    typer.echo(f"Saved: {prefix}_differential.png")


def map_spatial_genes(adata_sp, gene_mapping: dict):
    """Map spatial gene names (BIMP|MGEN format) to v4.3 IDs in place."""
    new_names = map_spatial_genes_to_v43(adata_sp.var_names, gene_mapping)
    adata_sp.var_names = pd.Index(new_names)
    adata_sp.var_names_make_unique()
    return adata_sp


@app.command()
def tangram(
    input_file: str = typer.Option("combined_clustered", help="Input clustered h5ad"),
    max_cells: int = typer.Option(30000, help="Max cells for training"),
    device: str = typer.Option("cuda", help="Device (cuda or cpu)"),
    train_ratio: float = typer.Option(
        0.9, help="Fraction of genes for training (rest for validation)"
    ),
    seed: int = typer.Option(42, help="Random seed"),
    n_neighbors: int = typer.Option(
        15, help="k-NN neighbors for propagation to full dataset"
    ),
    mode: str = typer.Option(
        "cells",
        help="Tangram mode: 'cells' (maps individual cells) or 'clusters' (fast, "
        "maps centroids)",
    ),
    cluster_label: str = typer.Option(
        "cell_type", help="Reference cluster label for clusters mode"
    ),
    num_epochs: int = typer.Option(500, help="Training epochs"),
    exclude_genes: str = typer.Option(
        "", help="Comma-separated v4.3 gene IDs to exclude from training"
    ),
    exclude_from_qc: bool = typer.Option(
        False, help="Auto-exclude genes flagged by previous QC (cos_sim<0.3 & AUC<0.6)"
    ),
):
    """Run Tangram cell type mapping on combined data.

    Trains Tangram on a subsample, validates with held-out genes,
    then propagates predictions to all cells via 3D spatial k-NN.

    In clusters mode, maps Leiden centroids → spots (fast).
    In cells mode (default), maps individual cells → spots (slow, needs GPU).
    """
    import json
    import random

    import tangram2.mapping as tg
    from scipy.sparse import issparse
    from scipy.stats import spearmanr
    from sklearn.neighbors import NearestNeighbors

    random.seed(seed)
    np.random.seed(seed)

    TANGRAM_DIR = RESULTS_ROOT / "tangram"
    TANGRAM_DIR.mkdir(parents=True, exist_ok=True)

    input_path = DATA_DIR / f"{input_file}.h5ad"
    typer.echo(f"Loading spatial data: {input_path.name}")
    adata_sp_full = sc.read_h5ad(input_path)
    typer.echo(
        f"  Spatial: {adata_sp_full.n_obs:,} cells, {adata_sp_full.n_vars} genes"
    )

    typer.echo(f"Loading reference: {SNRNA_REF.name}")
    adata_ref = sc.read_h5ad(SNRNA_REF)
    typer.echo(f"  Reference: {adata_ref.n_obs:,} cells, {adata_ref.n_vars} genes")

    # Map spatial genes to v4.3 IDs
    typer.echo("Mapping spatial gene names to v4.3 IDs...")
    gene_mapping = load_gene_mapping()
    typer.echo(f"  Loaded {len(gene_mapping)} gene mappings from crosswalk")
    adata_sp_full = map_spatial_genes(adata_sp_full, gene_mapping)

    # Find common genes after mapping
    common_genes = list(set(adata_sp_full.var_names) & set(adata_ref.var_names))
    typer.echo(f"Common genes after mapping: {len(common_genes)}")

    # Exclude genes (manual list or auto from previous QC)
    genes_to_exclude = set()
    if exclude_genes:
        genes_to_exclude.update(
            g.strip() for g in exclude_genes.split(",") if g.strip()
        )
    if exclude_from_qc:
        qc_path = TANGRAM_DIR / f"{input_file}_gene_qc.csv"
        if qc_path.exists():
            qc_df = pd.read_csv(qc_path, index_col=0)
            bad = qc_df[(qc_df["cos_sim"] < 0.3) & (qc_df["auc"].fillna(0) < 0.6)]
            genes_to_exclude.update(bad.index)
            typer.echo(
                f"  Auto-excluding {len(bad)} genes from previous QC "
                "(cos<0.3 & AUC<0.6)"
            )
        else:
            typer.echo(
                "  WARNING: --exclude-from-qc set but no gene_qc.csv found. "
                "Run tangram-qc first."
            )
    if genes_to_exclude:
        before = len(common_genes)
        common_genes = [g for g in common_genes if g not in genes_to_exclude]
        typer.echo(
            f"  Excluded {before - len(common_genes)} genes "
            f"→ {len(common_genes)} remaining"
        )

    if len(common_genes) < 50:
        typer.echo(
            f"ERROR: Only {len(common_genes)} common genes found. Need at least 50."
        )
        raise typer.Exit(1)

    # =========================================================================
    # SUBSAMPLE spatial for training
    # =========================================================================
    if adata_sp_full.n_obs > max_cells:
        typer.echo(f"Subsampling spatial to {max_cells} cells for training...")
        train_idx = np.random.choice(adata_sp_full.n_obs, max_cells, replace=False)
        adata_sp = adata_sp_full[train_idx].copy()
    else:
        adata_sp = adata_sp_full.copy()

    # =========================================================================
    # PREPARE DATA per Tangram tutorial:
    # - Reference: raw counts + normalize_total only (no log1p)
    # - Spatial: raw/linear counts (reverse log1p if needed)
    # tg.pp_adatas does no normalization — it only subsets genes and computes
    # density priors (rna_count_based uses X.sum(axis=1), needs linear scale).
    # =========================================================================
    # Reference: use raw counts layer if available
    if "counts" in adata_ref.layers:
        typer.echo("  Using raw counts layer for reference")
        adata_ref_sub = adata_ref[:, common_genes].copy()
        adata_ref_sub.X = adata_ref_sub.layers["counts"].copy()
        sc.pp.normalize_total(adata_ref_sub)  # library size correction only, no log1p
    else:
        typer.echo("  WARNING: No counts layer in reference, reversing log1p")
        adata_ref_sub = adata_ref[:, common_genes].copy()
        adata_ref_sub.X = np.expm1(
            adata_ref_sub.X.toarray() if issparse(adata_ref_sub.X) else adata_ref_sub.X
        )
        sc.pp.normalize_total(adata_ref_sub)

    # Spatial: reverse log1p to get back to linear/count-like scale
    typer.echo("  Reversing log1p on spatial data (Tangram needs linear scale)")
    adata_sp_sub = adata_sp[:, common_genes].copy()
    adata_sp_sub.X = np.expm1(
        adata_sp_sub.X.toarray() if issparse(adata_sp_sub.X) else adata_sp_sub.X
    ).astype(np.float32)

    ref_mean = (
        adata_ref_sub.X.toarray().mean()
        if issparse(adata_ref_sub.X)
        else adata_ref_sub.X.mean()
    )
    sp_mean = (
        adata_sp_sub.X.mean()
        if not issparse(adata_sp_sub.X)
        else adata_sp_sub.X.toarray().mean()
    )
    typer.echo(
        f"  Ref mean: {ref_mean:.3f}, Sp mean: {sp_mean:.3f}, "
        f"ratio: {sp_mean / max(ref_mean, 1e-10):.1f}x"
    )

    # =========================================================================
    # TRAIN/TEST gene split for validation
    # =========================================================================
    n_train = int(len(common_genes) * train_ratio)
    random.shuffle(common_genes)
    training_genes = common_genes[:n_train]
    test_genes = common_genes[n_train:]
    typer.echo(
        f"Gene split: {len(training_genes)} training, {len(test_genes)} test (held out)"
    )

    # =========================================================================
    # PREPARE + TRAIN TANGRAM (on training genes only)
    # =========================================================================
    typer.echo("Preparing data for Tangram...")
    tg.pp_adatas(adata_ref_sub, adata_sp_sub, genes=training_genes)

    mode_str = f"mode={mode}" + (
        f", cluster_label={cluster_label}" if mode == "clusters" else ""
    )
    typer.echo(
        f"Training Tangram on {device} ({num_epochs} "
        f"epochs, {len(training_genes)} genes, uniform "
        f"density, {mode_str})..."
    )
    map_kwargs = dict(
        device=device,
        num_epochs=num_epochs,
        density_prior="uniform",
        mode=mode,
    )
    if mode == "clusters":
        map_kwargs["cluster_label"] = cluster_label
    ad_map = tg.map_cells_to_space(adata_ref_sub, adata_sp_sub, **map_kwargs)

    # Save training gene scores for QC
    if "train_genes_df" in ad_map.uns:
        train_genes_path = TANGRAM_DIR / f"{input_file}_train_genes.csv"
        ad_map.uns["train_genes_df"].to_csv(train_genes_path)
        typer.echo(f"  Saved training gene scores: {train_genes_path.name}")

    # =========================================================================
    # PROJECT cell type annotations
    # =========================================================================
    typer.echo("Projecting cell type annotations...")
    if mode == "clusters":
        # In clusters mode, mapping matrix M is (n_clusters × n_spots).
        # M.T gives (n_spots × n_clusters) = cluster weights per spot.
        M = ad_map.X  # (n_clusters, n_spots)
        ct_vals = M.T.copy()  # (n_spots, n_clusters)
        if hasattr(ct_vals, "toarray"):
            ct_vals = ct_vals.toarray()
        ct_cols = list(ad_map.obs_names)  # cluster names
        typer.echo(f"  Clusters mode: M shape {M.shape}, {len(ct_cols)} clusters")
    else:
        # In cells mode, project_cell_annotations does M.T @ one_hot
        tg.project_cell_annotations(ad_map, adata_sp_sub, annotation="cell_type")
        ct_pred_raw = adata_sp_sub.obsm["tangram_ct_pred"]
        if hasattr(ct_pred_raw, "values"):
            ct_vals = ct_pred_raw.values.copy()
            ct_cols = list(ct_pred_raw.columns)
        else:
            ct_vals = ct_pred_raw.copy()
            ct_cols = None

    # Normalize per spot to get proper probabilities
    row_sums = ct_vals.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    ct_pred_norm = ct_vals / row_sums
    ct_pred_matrix = pd.DataFrame(
        ct_pred_norm,
        index=adata_sp_sub.obs_names,
        columns=ct_cols
        if ct_cols
        else [f"CT_{i}" for i in range(ct_pred_norm.shape[1])],
    )
    adata_sp_sub.obsm["tangram_ct_pred"] = ct_pred_matrix

    adata_sp_sub.obs["tangram_cell_type"] = ct_pred_matrix.idxmax(axis=1).values
    adata_sp_sub.obs["tangram_confidence"] = ct_pred_matrix.max(axis=1).values
    typer.echo(
        f"  Cell types: {adata_sp_sub.obs['tangram_cell_type'].nunique()} unique"
    )
    typer.echo(
        f"  Mean confidence: {adata_sp_sub.obs['tangram_confidence'].mean():.3f}"
    )
    typer.echo(f"  ct_pred row sums (pre-norm): {ct_vals.sum(axis=1).mean():.3f}")

    # =========================================================================
    # FILTER + PROJECT reference genes onto spatial
    # =========================================================================
    typer.echo("Preparing reference for gene projection...")
    adata_ref_full_norm = adata_ref.copy()
    if "counts" in adata_ref.layers:
        adata_ref_full_norm.X = adata_ref_full_norm.layers["counts"].copy()
    else:
        if issparse(adata_ref_full_norm.X):
            adata_ref_full_norm.X = adata_ref_full_norm.X.toarray()
        adata_ref_full_norm.X = np.expm1(adata_ref_full_norm.X)
    sc.pp.normalize_total(adata_ref_full_norm)

    # Filter reference genes: keep only those detected in ≥5% of ref cells
    # (genes with near-zero expression in ref can't produce meaningful projections)
    ref_X = adata_ref_full_norm.X
    if issparse(ref_X):
        ref_X = ref_X.toarray()
    frac_nz = (ref_X > 0).mean(axis=0)
    min_frac = 0.05
    keep_mask = frac_nz >= min_frac
    n_before = adata_ref_full_norm.n_vars
    adata_ref_filtered = adata_ref_full_norm[:, keep_mask].copy()
    typer.echo(
        f"  Filtered ref genes: {n_before:,} → {adata_ref_filtered.n_vars:,} "
        f"(kept genes detected in ≥{min_frac:.0%} of cells)"
    )

    typer.echo(
        f"Projecting {adata_ref_filtered.n_vars:,} filtered reference genes "
        "onto spatial cells..."
    )
    project_kwargs = dict(adata_map=ad_map, adata_sc=adata_ref_filtered)
    if mode == "clusters":
        project_kwargs["cluster_label"] = cluster_label
    ad_ge = tg.project_genes(**project_kwargs)
    typer.echo(f"  Imputed expression: {ad_ge.shape}")

    # Copy metadata from spatial subset
    for key in ("X_umap", "spatial_aligned"):
        if key in adata_sp_sub.obsm:
            ad_ge.obsm[key] = adata_sp_sub.obsm[key]
    for col in ("cell_id", "brain", "leiden", "slice_name"):
        if col in adata_sp_sub.obs.columns:
            ad_ge.obs[col] = adata_sp_sub.obs[col].values
    ad_ge.obs["tangram_cell_type"] = adata_sp_sub.obs["tangram_cell_type"].values
    ad_ge.obs["tangram_confidence"] = adata_sp_sub.obs["tangram_confidence"].values

    # NOTE: UMAP for visualization uses the snRNA reference embedding.

    # Save imputed expression
    imputed_path = TANGRAM_DIR / f"{input_file}_imputed.h5ad"
    ad_ge.write_h5ad(imputed_path)
    typer.echo(f"  Saved imputed expression: {imputed_path.name}")

    # =========================================================================
    # VALIDATE: Spearman correlations on all genes (train + held-out test)
    # =========================================================================
    typer.echo("Validating gene predictions (Spearman r)...")
    validation_results = []
    for gene in common_genes:
        gene_lower = gene.lower()
        if (
            gene_lower not in ad_ge.var_names
            or gene_lower not in adata_sp_sub.var_names
        ):
            if gene not in ad_ge.var_names or gene not in adata_sp_sub.var_names:
                continue
            gene_key_ge, gene_key_sp = gene, gene
        else:
            gene_key_ge, gene_key_sp = gene_lower, gene_lower

        predicted = ad_ge[:, gene_key_ge].X.flatten()
        actual = adata_sp_sub[:, gene_key_sp].X
        if issparse(actual):
            actual = actual.toarray().flatten()
        else:
            actual = actual.flatten()

        if np.std(predicted) > 0 and np.std(actual) > 0:
            corr, pval = spearmanr(predicted, actual)
            validation_results.append(
                {
                    "gene": gene,
                    "spearman_r": corr,
                    "pval": pval,
                    "predicted_mean": float(predicted.mean()),
                    "actual_mean": float(actual.mean()),
                    "is_training": gene in training_genes,
                }
            )

    if validation_results:
        val_df = pd.DataFrame(validation_results)
        train_df = val_df[val_df["is_training"]]
        test_df = val_df[~val_df["is_training"]]

        typer.echo(
            f"  Training genes ({len(train_df)}): mean "
            f"r={train_df['spearman_r'].mean():.3f}, "
            f"median={train_df['spearman_r'].median():.3f}"
        )
        typer.echo(
            f"  Test genes ({len(test_df)}): mean "
            f"r={test_df['spearman_r'].mean():.3f}, "
            f"median={test_df['spearman_r'].median():.3f}"
        )
        n_sig = (test_df["pval"] < 0.05).sum()
        typer.echo(f"  Test significant (p<0.05): {n_sig}/{len(test_df)}")

        val_path = TANGRAM_DIR / f"{input_file}_validation.csv"
        val_df.to_csv(val_path, index=False)
        typer.echo(f"  Saved: {val_path.name}")

    # Save gene split
    gene_split_path = TANGRAM_DIR / f"{input_file}_gene_split.json"
    with open(gene_split_path, "w") as f:
        json.dump(
            {
                "training_genes": training_genes,
                "test_genes": test_genes,
                "train_ratio": train_ratio,
                "seed": seed,
            },
            f,
            indent=2,
        )

    # =========================================================================
    # PROPAGATE to all cells via 3D spatial k-NN
    # (much better than sc.tl.ingest — uses actual spatial coordinates)
    # =========================================================================
    if adata_sp_full.n_obs > len(adata_sp_sub):
        typer.echo(
            f"\nPropagating predictions to all {adata_sp_full.n_obs:,} cells "
            "via 3D k-NN..."
        )

        # Build 3D coordinates: (x, y, z) from spatial_aligned + slice info
        train_coords_2d = adata_sp.obsm.get(
            "spatial_aligned", adata_sp.obsm.get("spatial")
        )
        full_coords_2d = adata_sp_full.obsm.get(
            "spatial_aligned", adata_sp_full.obsm.get("spatial")
        )

        # Build z from slice_name if available
        has_slices = "slice_name" in adata_sp_full.obs.columns
        if has_slices:
            slice_names = sorted(adata_sp_full.obs["slice_name"].unique())
            slice_to_idx = {s: i for i, s in enumerate(slice_names)}
            z_spacing = 10.0  # µm between slices
            train_z = np.array(
                [slice_to_idx.get(s, 0) * z_spacing for s in adata_sp.obs["slice_name"]]
            )
            full_z = np.array(
                [
                    slice_to_idx.get(s, 0) * z_spacing
                    for s in adata_sp_full.obs["slice_name"]
                ]
            )
            typer.echo(f"  3D mode: {len(slice_names)} slices, {z_spacing}µm spacing")
        else:
            train_z = np.zeros(len(adata_sp))
            full_z = np.zeros(len(adata_sp_full))
            typer.echo("  2D mode (no slice_name column)")

        train_coords = np.column_stack([train_coords_2d, train_z])
        full_coords = np.column_stack([full_coords_2d, full_z])

        typer.echo(
            f"  Fitting k-NN (k={n_neighbors}) on {len(train_coords)} training cells..."
        )
        knn = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm="ball_tree", n_jobs=-1
        )
        knn.fit(train_coords)
        distances, indices = knn.kneighbors(full_coords)

        # Propagate cell type predictions via distance-weighted voting
        train_ct_pred = ct_pred_matrix
        if hasattr(train_ct_pred, "values"):
            train_ct_pred = train_ct_pred.values
        n_cells = len(full_coords)
        n_ct = train_ct_pred.shape[1]

        weights = 1.0 / (distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Weighted average of soft predictions
        typer.echo(
            f"  Propagating {n_ct} cell type predictions to {n_cells:,} cells..."
        )
        full_ct_pred = np.zeros((n_cells, n_ct), dtype=np.float32)
        for i in range(n_neighbors):
            full_ct_pred += weights[:, i : i + 1] * train_ct_pred[indices[:, i]]

        # Store predictions
        ct_labels = (
            list(ct_pred_matrix.columns) if hasattr(ct_pred_matrix, "columns") else None
        )
        adata_sp_full.obsm["tangram_ct_pred"] = full_ct_pred

        max_ct_idx = full_ct_pred.argmax(axis=1)
        if ct_labels is not None:
            adata_sp_full.obs["tangram_cell_type"] = [ct_labels[i] for i in max_ct_idx]
        else:
            adata_sp_full.obs["tangram_cell_type"] = [f"CT_{i}" for i in max_ct_idx]
        adata_sp_full.obs["tangram_confidence"] = full_ct_pred.max(axis=1)

        typer.echo(
            "  Mean confidence (all "
            f"cells): {adata_sp_full.obs['tangram_confidence'].mean():.3f}"
        )
    else:
        # All cells were used for training — just transfer directly
        adata_sp_full.obsm["tangram_ct_pred"] = (
            ct_pred_matrix.values
            if hasattr(ct_pred_matrix, "values")
            else ct_pred_matrix
        )
        adata_sp_full.obs["tangram_cell_type"] = adata_sp_sub.obs[
            "tangram_cell_type"
        ].values
        adata_sp_full.obs["tangram_confidence"] = adata_sp_sub.obs[
            "tangram_confidence"
        ].values

    # Save
    output_path = TANGRAM_DIR / f"{input_file}_tangram.h5ad"
    adata_sp_full.write_h5ad(output_path)
    typer.echo(f"Saved: {output_path}")


@app.command()
def visualize_tangram(
    input_file: str = typer.Option("combined_clustered", help="Base input filename"),
):
    """Visualize Tangram cell type mapping results."""
    TANGRAM_DIR = RESULTS_ROOT / "tangram"
    TANGRAM_FIGS = FIGURES_DIR / "tangram"
    TANGRAM_FIGS.mkdir(parents=True, exist_ok=True)

    # Load Tangram-annotated data
    tangram_path = TANGRAM_DIR / f"{input_file}_tangram.h5ad"
    if not tangram_path.exists():
        typer.echo(f"ERROR: Tangram results not found: {tangram_path}")
        typer.echo("Run: python combined_analysis.py tangram")
        raise typer.Exit(1)

    typer.echo(f"Loading Tangram results: {tangram_path.name}")
    adata = sc.read_h5ad(tangram_path)

    # Check for cell_type annotation
    if "tangram_cell_type" not in adata.obs.columns:
        typer.echo("ERROR: No cell_type annotation found in data")
        raise typer.Exit(1)

    typer.echo(f"  Cells: {adata.n_obs:,}")
    typer.echo(f"  Cell types: {adata.obs['tangram_cell_type'].nunique()}")

    # Use original UMAP from probes (not tangram-imputed)
    cluster_key = "leiden"
    typer.echo(
        f"  Cluster key: {cluster_key} ({adata.obs[cluster_key].nunique()} clusters)"
    )

    # Plot 1: UMAP overview - brain, clusters, cell types
    typer.echo("\nGenerating UMAP visualizations...")
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))

    # Brain identity
    sc.pl.umap(
        adata,
        color="brain",
        ax=axes[0],
        show=False,
        palette={"queen": "#DAA520", "worker": "#4A90D9"},
        title="Brain Identity",
    )

    # Leiden clusters (from probes)
    sc.pl.umap(
        adata,
        color=cluster_key,
        ax=axes[1],
        show=False,
        legend_loc="none",
        title=f"Spatial Leiden Clusters (n={adata.obs[cluster_key].nunique()})",
    )

    # Cell types (from reference)
    sc.pl.umap(
        adata,
        color="tangram_cell_type",
        ax=axes[2],
        show=False,
        legend_loc="none",
        title=f"Reference Cell Types (n={adata.obs['tangram_cell_type'].nunique()})",
    )

    plt.tight_layout()
    fig.savefig(TANGRAM_FIGS / "umap_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo("  Saved: tangram/umap_overview.png")

    # Plot 2: Cell type distribution by brain
    typer.echo("\nPlotting cell type distributions...")
    _plot_celltype_distribution(adata, TANGRAM_FIGS)

    # Plot 3: UMAP split by brain
    typer.echo("\nGenerating split UMAPs...")
    _plot_split_umap(adata, TANGRAM_FIGS)

    # Plot 4: Cell type composition by cluster (tangram leiden vs reference cell types)
    typer.echo("\nPlotting cluster-cell type composition...")
    _plot_cluster_celltype_composition(adata, TANGRAM_FIGS, cluster_key=cluster_key)

    typer.echo("\nDone! Figures saved to: figures/tangram/")


def _plot_celltype_distribution(adata, output_dir):
    """Plot cell type distribution by brain."""
    # Count cells per type per brain
    counts = (
        adata.obs.groupby(["tangram_cell_type", "brain"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    props = counts.div(counts.sum(axis=0), axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Absolute counts
    counts.plot(kind="barh", ax=axes[0], color=["#DAA520", "#4A90D9"], stacked=False)
    axes[0].set_xlabel("Cell Count")
    axes[0].set_ylabel("Cell Type")
    axes[0].set_title("Cell Type Counts by Brain")
    axes[0].legend(title="Brain")

    # Proportions
    props.plot(kind="barh", ax=axes[1], color=["#DAA520", "#4A90D9"], stacked=False)
    axes[1].set_xlabel("Proportion")
    axes[1].set_ylabel("Cell Type")
    axes[1].set_title("Cell Type Proportions by Brain")
    axes[1].legend(title="Brain")

    plt.tight_layout()
    fig.savefig(output_dir / "celltype_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo("  Saved: tangram/celltype_distribution.png")


def _plot_split_umap(adata, output_dir):
    """Plot UMAPs split by brain."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Queen
    queen_data = adata[adata.obs["brain"] == "queen"]
    sc.pl.umap(
        queen_data,
        color="tangram_cell_type",
        ax=axes[0],
        show=False,
        legend_loc="right margin",
        legend_fontsize=8,
        title="Queen - Cell Types",
    )

    # Worker
    worker_data = adata[adata.obs["brain"] == "worker"]
    sc.pl.umap(
        worker_data,
        color="tangram_cell_type",
        ax=axes[1],
        show=False,
        legend_loc="right margin",
        legend_fontsize=8,
        title="Worker - Cell Types",
    )

    plt.tight_layout()
    fig.savefig(output_dir / "umap_split_by_brain.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo("  Saved: tangram/umap_split_by_brain.png")


def _plot_cluster_celltype_composition(adata, output_dir, cluster_key="leiden"):
    """Plot cell type composition within each cluster."""
    # Count cells per cluster per cell type
    counts = (
        adata.obs.groupby([cluster_key, "tangram_cell_type"], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    # Normalize to proportions within each cluster
    props = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 8))
    props.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("Leiden Cluster")
    ax.set_ylabel("Proportion")
    ax.set_title("Cell Type Composition by Cluster")
    ax.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.xticks(rotation=45)

    plt.tight_layout()
    fig.savefig(
        output_dir / "cluster_celltype_composition.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    typer.echo("  Saved: tangram/cluster_celltype_composition.png")


# =============================================================================
# TANGRAM QC
# =============================================================================

CASTE_COLORS = {"queen": "#DAA520", "worker": "#4A90D9"}


def _set_pub_style():
    """Set publication-quality matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


@app.command()
def tangram_qc(
    input_file: str = typer.Option("combined_clustered", help="Base input filename"),
):
    """Tangram QC: cell type predictions vs spatial Leiden clusters."""
    import seaborn as sns

    _set_pub_style()

    TANGRAM_DIR = RESULTS_ROOT / "tangram"
    TANGRAM_FIGS = RESULTS_ROOT / "tangram_qc"
    TANGRAM_FIGS.mkdir(parents=True, exist_ok=True)

    tangram_path = TANGRAM_DIR / f"{input_file}_tangram.h5ad"
    if not tangram_path.exists():
        typer.echo(f"ERROR: Tangram results not found: {tangram_path}")
        typer.echo("Run: python combined_analysis.py tangram")
        raise typer.Exit(1)

    typer.echo(f"Loading Tangram results: {tangram_path.name}")
    adata = sc.read_h5ad(tangram_path)
    typer.echo(f"  Cells: {adata.n_obs:,}")
    typer.echo(f"  Leiden clusters (probes): {adata.obs['leiden'].nunique()}")
    typer.echo(
        f"  Tangram cell types (reference): {adata.obs['tangram_cell_type'].nunique()}"
    )

    umap = adata.obsm["X_umap"]
    pt, alpha = 0.1, 0.2  # point params for 1.3M cells

    # =========================================================================
    # Plot 1: UMAP — Leiden clusters vs Tangram cell types
    # =========================================================================
    typer.echo("\n[1/3] UMAP: Leiden vs Tangram cell types...")

    ct_cats = sorted(adata.obs["tangram_cell_type"].unique())
    n_ct = len(ct_cats)
    ct_cmap = plt.colormaps.get_cmap("gist_ncar").resampled(n_ct)
    ct_color_map = {ct: ct_cmap(i) for i, ct in enumerate(ct_cats)}

    leiden_cats = sorted(
        adata.obs["leiden"].unique(), key=lambda x: int(x) if str(x).isdigit() else x
    )
    n_lei = len(leiden_cats)
    lei_cmap = (
        plt.colormaps.get_cmap("tab20").resampled(n_lei)
        if n_lei <= 20
        else plt.colormaps.get_cmap("gist_ncar").resampled(n_lei)
    )
    lei_color_map = {cl: lei_cmap(i) for i, cl in enumerate(leiden_cats)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Top-left: Brain identity
    ax = axes[0, 0]
    for brain, color in CASTE_COLORS.items():
        mask = adata.obs["brain"].values == brain
        ax.scatter(
            umap[mask, 0],
            umap[mask, 1],
            s=pt,
            alpha=alpha,
            c=color,
            label=brain,
            rasterized=True,
        )
    ax.set_title("Brain Identity")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(markerscale=10, fontsize=8)

    # Top-right: Leiden clusters (from probes alone)
    ax = axes[0, 1]
    lei_vals = adata.obs["leiden"].values
    for cl in leiden_cats:
        mask = lei_vals == cl
        ax.scatter(
            umap[mask, 0],
            umap[mask, 1],
            s=pt,
            alpha=alpha,
            c=[lei_color_map[cl]],
            rasterized=True,
        )
    ax.set_title(f"Spatial Leiden Clusters (n={n_lei}, probes only)")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    # Bottom-left: Tangram cell types
    ax = axes[1, 0]
    ct_vals = adata.obs["tangram_cell_type"].values
    for ct in ct_cats:
        mask = ct_vals == ct
        ax.scatter(
            umap[mask, 0],
            umap[mask, 1],
            s=pt,
            alpha=alpha,
            c=[ct_color_map[ct]],
            rasterized=True,
        )
    ax.set_title(f"Tangram Cell Types (n={n_ct}, from snRNA ref)")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    # Bottom-right: Confidence
    ax = axes[1, 1]
    if "tangram_confidence" in adata.obs.columns:
        scatter = ax.scatter(
            umap[:, 0],
            umap[:, 1],
            c=adata.obs["tangram_confidence"].values,
            s=pt,
            alpha=alpha,
            cmap="viridis",
            rasterized=True,
        )
        plt.colorbar(scatter, ax=ax, label="Confidence", shrink=0.8)
    ax.set_title(
        f"Prediction Confidence (median={adata.obs['tangram_confidence'].median():.2f})"
    )
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    plt.tight_layout()
    fig.savefig(TANGRAM_FIGS / "tangram_umap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    typer.echo("  Saved: tangram/tangram_umap.png")

    # =========================================================================
    # Plot 2: Cluster mapping heatmap — Leiden vs Tangram types
    # =========================================================================
    typer.echo("[2/3] Cluster mapping heatmap...")

    ct_tab = pd.crosstab(adata.obs["tangram_cell_type"], adata.obs["leiden"])
    # Normalize by column: for each Leiden cluster, what tangram types does it contain?
    ct_by_leiden = ct_tab.div(ct_tab.sum(axis=0), axis=1)
    ct_by_leiden = ct_by_leiden[
        sorted(ct_by_leiden.columns, key=lambda x: int(x) if str(x).isdigit() else x)
    ]
    # Sort tangram types by abundance
    ct_by_leiden = ct_by_leiden.reindex(
        adata.obs["tangram_cell_type"].value_counts().index
    )

    fig, ax = plt.subplots(figsize=(max(10, n_lei * 0.6), max(10, n_ct * 0.3)))
    sns.heatmap(
        ct_by_leiden,
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Proportion within Leiden cluster"},
    )
    ax.set_xlabel("Spatial Leiden Cluster (probes only)")
    ax.set_ylabel("Tangram Cell Type (snRNA reference)")
    ax.set_title("What cell types does each spatial cluster contain?")

    plt.tight_layout()
    fig.savefig(
        TANGRAM_FIGS / "tangram_cluster_mapping.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    typer.echo("  Saved: tangram/tangram_cluster_mapping.png")

    # =========================================================================
    # Plot 3–5: Gene-level QC (cosine similarity, AUC, sparsity)
    # Requires imputed h5ad + spatial subset (same 30k cells)
    # =========================================================================
    imputed_path = TANGRAM_DIR / f"{input_file}_imputed.h5ad"
    gene_split_path = TANGRAM_DIR / f"{input_file}_gene_split.json"

    if imputed_path.exists() and gene_split_path.exists():
        typer.echo("\n[3/6] Computing gene-level QC metrics...")
        typer.echo(f"  Loading snRNA reference: {SNRNA_REF.name}")
        adata_ref = sc.read_h5ad(SNRNA_REF)

        qc_df = _compute_gene_qc(adata, imputed_path, gene_split_path, adata_ref)

        if qc_df is not None and len(qc_df) > 0:
            qc_path = TANGRAM_DIR / f"{input_file}_gene_qc.csv"
            qc_df.to_csv(qc_path)
            typer.echo(f"  Saved: {qc_path.name}")

            typer.echo("[4/6] Gene QC diagnostic plots...")
            _plot_gene_qc(qc_df, TANGRAM_FIGS)

            typer.echo("[5/6] Sparsity analysis...")
            _plot_sparsity_from_qc(qc_df, TANGRAM_FIGS)
        else:
            typer.echo("  WARNING: Could not compute gene QC — skipping")

        # Imputed-space UMAP
        typer.echo("[6/6] Imputed expression UMAP...")
        ad_ge = sc.read_h5ad(imputed_path)
        _plot_imputed_umap(ad_ge, adata, TANGRAM_FIGS)
    else:
        typer.echo("\n[3/6] No imputed h5ad found — falling back to sparsity only")
        typer.echo(f"  Loading snRNA reference: {SNRNA_REF.name}")
        adata_ref = sc.read_h5ad(SNRNA_REF)
        _plot_sparsity(adata, adata_ref, TANGRAM_DIR, TANGRAM_FIGS, input_file)

    typer.echo(f"\nDone! Figures saved to: {TANGRAM_FIGS}")


def _compute_gene_qc(adata_sp_full, imputed_path, gene_split_path, adata_ref):
    """Compute per-gene QC: cosine similarity, AUC, sparsity mismatch.

    Compares Tangram-imputed expression (ad_ge) against actual Xenium counts
    (adata_sp) for the same 30k cell subset.
    """
    import json

    from scipy.sparse import issparse
    from sklearn.metrics import roc_auc_score

    ad_ge = sc.read_h5ad(imputed_path)
    with open(gene_split_path) as f:
        gene_split = json.load(f)
    training_genes = set(gene_split["training_genes"])

    # Match imputed cells to spatial data (same 30k subset)
    ge_ids = (
        set(ad_ge.obs["cell_id"].values) if "cell_id" in ad_ge.obs.columns else None
    )
    if ge_ids is None:
        typer.echo("  WARNING: No cell_id in imputed data, using index matching")
        common_idx = ad_ge.obs_names.intersection(adata_sp_full.obs_names)
        if len(common_idx) == 0:
            return None
        ad_ge_sub = ad_ge[common_idx]
        sp_sub = adata_sp_full[common_idx]
    else:
        sp_cell_to_idx = {}
        for i, cid in enumerate(adata_sp_full.obs["cell_id"].values):
            if cid not in sp_cell_to_idx:
                sp_cell_to_idx[cid] = i
        match_idx = [
            sp_cell_to_idx[cid]
            for cid in ad_ge.obs["cell_id"].values
            if cid in sp_cell_to_idx
        ]
        if len(match_idx) < ad_ge.n_obs * 0.5:
            typer.echo(f"  WARNING: Only {len(match_idx)}/{ad_ge.n_obs} cells matched")
            return None
        sp_sub = adata_sp_full[match_idx]
        ad_ge_sub = ad_ge

    # Match genes: spatial has BIMP_v4.3_*, imputed has bimp_v4.3_* (lowercase)
    ge_names_lower = {v.lower(): v for v in ad_ge_sub.var_names}
    ref_names = set(adata_ref.var_names)

    records = []
    for sp_gene in sp_sub.var_names:
        v43_id = sp_gene  # spatial already has v4.3 IDs
        ge_key = ge_names_lower.get(sp_gene.lower())
        if ge_key is None:
            continue

        # Actual spatial counts (reverse log1p to linear)
        actual = sp_sub[:, sp_gene].X
        if issparse(actual):
            actual = actual.toarray()
        actual = np.expm1(actual.flatten().astype(np.float64))

        # Predicted (already linear from project_genes)
        predicted = ad_ge_sub[:, ge_key].X
        if issparse(predicted):
            predicted = predicted.toarray()
        predicted = predicted.flatten().astype(np.float64)

        n = len(actual)
        frac_nz_sp = float((actual > 0).mean())
        frac_nz_ref = float(
            1.0 - (adata_ref[:, v43_id].X.toarray() == 0).mean()
            if v43_id in ref_names
            else np.nan
        )

        # Cosine similarity (Tangram's native metric)
        norm_sq = np.linalg.norm(predicted) * np.linalg.norm(actual)
        cos_sim = float(predicted @ actual / norm_sq) if norm_sq > 0 else 0.0

        # Count-normalized cosine similarity: residualize total counts
        # This measures gene-specific spatial pattern beyond cell-size effects
        sp_total = (
            np.expm1(sp_sub.X.toarray() if issparse(sp_sub.X) else sp_sub.X)
            .sum(axis=1)
            .flatten()
        )
        ge_total = (
            ad_ge_sub.X.toarray().sum(axis=1).flatten()
            if issparse(ad_ge_sub.X)
            else ad_ge_sub.X.sum(axis=1).flatten()
        )

        # Residualize: subtract projection onto total counts
        def _residualize(vals, covariate):
            cov = covariate.astype(np.float64)
            v = vals.astype(np.float64)
            if np.std(cov) < 1e-10:
                return v
            beta = np.dot(v, cov) / np.dot(cov, cov)
            return v - beta * cov

        pred_resid = _residualize(predicted, ge_total)
        act_resid = _residualize(actual, sp_total)
        norm_resid = np.linalg.norm(pred_resid) * np.linalg.norm(act_resid)
        cos_sim_resid = (
            float(pred_resid @ act_resid / norm_resid) if norm_resid > 0 else 0.0
        )

        # AUC: can predicted values discriminate detected vs undetected cells?
        binary_detected = (actual > 0).astype(int)
        n_pos = binary_detected.sum()
        n_neg = n - n_pos
        if n_pos > 10 and n_neg > 10:
            try:
                auc = float(roc_auc_score(binary_detected, predicted))
            except ValueError:
                auc = np.nan
        else:
            auc = np.nan  # too few positives/negatives

        # Spearman on non-zero cells only
        nz_mask = actual > 0
        if nz_mask.sum() > 30:
            from scipy.stats import spearmanr

            r_nz, _ = spearmanr(predicted[nz_mask], actual[nz_mask])
            r_nz = float(r_nz)
        else:
            r_nz = np.nan

        records.append(
            {
                "gene_v43": v43_id,
                "gene_spatial": sp_gene,
                "is_training": v43_id in training_genes,
                "cos_sim": cos_sim,
                "cos_sim_resid": cos_sim_resid,
                "auc": auc,
                "spearman_nz": r_nz,
                "frac_nz_spatial": frac_nz_sp,
                "frac_nz_reference": frac_nz_ref,
                "sparsity_diff": frac_nz_sp - frac_nz_ref,
                "predicted_mean": float(predicted.mean()),
                "actual_mean": float(actual.mean()),
            }
        )

    df = pd.DataFrame(records).set_index("gene_v43")
    train = df[df["is_training"]]
    test = df[~df["is_training"]]

    typer.echo(f"  Genes scored: {len(df)} ({len(train)} training, {len(test)} test)")
    typer.echo(
        f"  Cosine sim — train: {train['cos_sim'].median():.3f} (median), "
        f"test: {test['cos_sim'].median():.3f}"
    )
    typer.echo(
        "  Cosine sim (count-residualized) — "
        f"train: {train['cos_sim_resid'].median():.3f}, "
        f"test: {test['cos_sim_resid'].median():.3f}"
    )
    typer.echo(
        f"  AUC — train: {train['auc'].median():.3f} (median), "
        f"test: {test['auc'].median():.3f}"
    )
    typer.echo(
        f"  Spearman (non-zero) — train: {train['spearman_nz'].median():.3f}, "
        f"test: {test['spearman_nz'].median():.3f}"
    )

    return df


def _plot_imputed_umap(ad_ge, adata_sp, output_dir):
    """Plot UMAP computed from imputed expression, colored by cell type / brain /
    leiden."""
    pt, alpha = 0.3, 0.3

    # Check for imputed UMAP
    if "X_umap_imputed" not in ad_ge.obsm:
        typer.echo("  No X_umap_imputed in imputed h5ad — computing on the fly...")
        ad_tmp = ad_ge.copy()
        sc.pp.log1p(ad_tmp)
        sc.pp.highly_variable_genes(ad_tmp, n_top_genes=min(2000, ad_tmp.n_vars))
        sc.pp.scale(ad_tmp, max_value=10)
        sc.tl.pca(ad_tmp, n_comps=min(50, ad_tmp.n_vars - 1), random_state=0)
        sc.pp.neighbors(
            ad_tmp, n_pcs=min(30, ad_tmp.obsm["X_pca"].shape[1]), random_state=0
        )
        sc.tl.umap(ad_tmp, random_state=0)
        umap_imp = ad_tmp.obsm["X_umap"]
    else:
        umap_imp = ad_ge.obsm["X_umap_imputed"]

    # Also get the probe-based UMAP for comparison
    umap_probes = ad_ge.obsm.get("X_umap")

    # Get labels — try from imputed obs, fall back to spatial
    brain = ad_ge.obs.get("brain")
    if brain is None and "cell_id" in ad_ge.obs.columns:
        sp_ids = {cid: i for i, cid in enumerate(adata_sp.obs["cell_id"].values)}
        idx = [sp_ids.get(cid, 0) for cid in ad_ge.obs["cell_id"].values]
        brain = adata_sp.obs["brain"].values[idx]
    tangram_ct = ad_ge.obs.get("tangram_cell_type")
    if tangram_ct is None:
        sp_ids = {cid: i for i, cid in enumerate(adata_sp.obs["cell_id"].values)}
        idx = [sp_ids.get(cid, 0) for cid in ad_ge.obs["cell_id"].values]
        tangram_ct = adata_sp.obs["tangram_cell_type"].values[idx]

    n_panels = 4 if umap_probes is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 2:
        axes = list(axes)

    # Panel 1: Imputed UMAP — brain identity
    ax = axes[0]
    for b, color in CASTE_COLORS.items():
        mask = np.array(brain) == b
        ax.scatter(
            umap_imp[mask, 0],
            umap_imp[mask, 1],
            s=pt,
            alpha=alpha,
            c=color,
            label=b,
            rasterized=True,
        )
    ax.set_title("Imputed Expression UMAP — Brain")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(markerscale=10, fontsize=8)

    # Panel 2: Imputed UMAP — tangram cell types
    ax = axes[1]
    ct_cats = sorted(set(tangram_ct))
    ct_cmap = plt.colormaps.get_cmap("gist_ncar").resampled(len(ct_cats))
    for i, ct in enumerate(ct_cats):
        mask = np.array(tangram_ct) == ct
        ax.scatter(
            umap_imp[mask, 0],
            umap_imp[mask, 1],
            s=pt,
            alpha=alpha,
            c=[ct_cmap(i)],
            rasterized=True,
        )
    ax.set_title("Imputed Expression UMAP — Cell Types")
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")

    if umap_probes is not None:
        # Panel 3: Probe UMAP — brain (for comparison)
        ax = axes[2]
        for b, color in CASTE_COLORS.items():
            mask = np.array(brain) == b
            ax.scatter(
                umap_probes[mask, 0],
                umap_probes[mask, 1],
                s=pt,
                alpha=alpha,
                c=color,
                label=b,
                rasterized=True,
            )
        ax.set_title("Probe UMAP (101 genes) — Brain")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.legend(markerscale=10, fontsize=8)

        # Panel 4: Probe UMAP — cell types (for comparison)
        ax = axes[3]
        for i, ct in enumerate(ct_cats):
            mask = np.array(tangram_ct) == ct
            ax.scatter(
                umap_probes[mask, 0],
                umap_probes[mask, 1],
                s=pt,
                alpha=alpha,
                c=[ct_cmap(i)],
                rasterized=True,
            )
        ax.set_title("Probe UMAP (101 genes) — Cell Types")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")

    plt.tight_layout()
    fig.savefig(output_dir / "tangram_imputed_umap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    typer.echo("  Saved: tangram/tangram_imputed_umap.png")


def _plot_gene_qc(qc_df, output_dir):
    """Plot gene-level QC: cosine sim, AUC, Spearman (non-zero), all vs sparsity."""

    train = qc_df[qc_df["is_training"]]
    test = qc_df[~qc_df["is_training"]]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # --- Row 1: Score distributions (train vs test) ---

    # (A) Cosine similarity bar chart, sorted, colored by train/test
    ax = axes[0, 0]
    sorted_df = qc_df.sort_values("cos_sim", ascending=False).reset_index()
    colors = ["#3C5488" if t else "#E64B35" for t in sorted_df["is_training"]]
    ax.bar(
        range(len(sorted_df)), sorted_df["cos_sim"], color=colors, alpha=0.8, width=1.0
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Gene (ranked)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(f"Cosine Similarity (train median={train['cos_sim'].median():.2f})")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color="#3C5488", label=f"Train (n={len(train)})"),
            Patch(color="#E64B35", label=f"Test (n={len(test)})"),
        ],
        fontsize=8,
    )

    # (B) AUC bar chart
    ax = axes[0, 1]
    sorted_auc = (
        qc_df.dropna(subset=["auc"]).sort_values("auc", ascending=False).reset_index()
    )
    colors_auc = ["#3C5488" if t else "#E64B35" for t in sorted_auc["is_training"]]
    ax.bar(
        range(len(sorted_auc)),
        sorted_auc["auc"],
        color=colors_auc,
        alpha=0.8,
        width=1.0,
    )
    ax.axhline(
        0.5, color="red", linestyle="--", linewidth=0.5, alpha=0.5, label="Chance"
    )
    ax.set_xlabel("Gene (ranked)")
    ax.set_ylabel("AUC (detection)")
    ax.set_title(
        f"AUC: Predict Detection (train={train['auc'].median():.2f}, "
        f"test={test['auc'].median():.2f})"
    )
    ax.legend(fontsize=8)

    # (C) Cosine similarity, residualized (count-corrected)
    ax = axes[0, 2]
    if "cos_sim_resid" in qc_df.columns:
        sorted_resid = qc_df.sort_values("cos_sim_resid", ascending=False).reset_index()
        colors_r = ["#3C5488" if t else "#E64B35" for t in sorted_resid["is_training"]]
        ax.bar(
            range(len(sorted_resid)),
            sorted_resid["cos_sim_resid"],
            color=colors_r,
            alpha=0.8,
            width=1.0,
        )
        ax.axhline(0, color="black", linewidth=0.5)
        train_med = train["cos_sim_resid"].median()
        test_med = test["cos_sim_resid"].median()
        ax.set_title(
            f"Cosine Sim (count-residualized, tr={train_med:.2f}, te={test_med:.2f})"
        )
    else:
        for subset, label, color in [
            (train, "Train", "#3C5488"),
            (test, "Test", "#E64B35"),
        ]:
            vals = subset["spearman_nz"].dropna()
            if len(vals) > 0:
                ax.hist(
                    vals,
                    bins=15,
                    alpha=0.6,
                    color=color,
                    label=f"{label} (med={vals.median():.2f})",
                    edgecolor="white",
                )
        ax.set_title("Rank Correlation (excluding zeros)")
        ax.legend(fontsize=8)
    ax.set_xlabel("Gene (ranked)")
    ax.set_ylabel("Cosine Similarity (residualized)")

    # --- Row 2: Score vs sparsity (diagnostic) ---

    # (D) Cosine sim vs spatial detection rate
    ax = axes[1, 0]
    for subset, label, color, marker in [
        (train, "Train", "#3C5488", "o"),
        (test, "Test", "#E64B35", "^"),
    ]:
        ax.scatter(
            subset["frac_nz_spatial"],
            subset["cos_sim"],
            s=40,
            alpha=0.7,
            c=color,
            marker=marker,
            label=label,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.set_xlabel("Spatial Detection Rate (frac non-zero)")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Cosine Sim vs Spatial Detection")
    ax.legend(fontsize=8)

    # (E) AUC vs sparsity mismatch
    ax = axes[1, 1]
    valid = qc_df.dropna(subset=["auc"])
    ax.scatter(
        valid["sparsity_diff"],
        valid["auc"],
        s=40,
        alpha=0.7,
        c=valid["is_training"].map({True: "#3C5488", False: "#E64B35"}),
        edgecolors="white",
        linewidths=0.5,
    )
    ax.axhline(0.5, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
    ax.set_xlabel("Detection Rate Diff (spatial − reference)")
    ax.set_ylabel("AUC")
    ax.set_title("AUC vs Sparsity Mismatch")

    # (F) Summary: flag problematic genes
    ax = axes[1, 2]
    # Genes with low cosine sim AND bad sparsity mismatch
    cos_thresh = 0.3
    auc_thresh = 0.6
    bad_cos = qc_df["cos_sim"] < cos_thresh
    bad_auc = qc_df["auc"].fillna(0) < auc_thresh
    n_bad_cos = bad_cos.sum()
    n_bad_auc = bad_auc.sum()
    n_bad_both = (bad_cos & bad_auc).sum()

    summary_text = (
        f"Gene QC Summary\n"
        f"{'─' * 35}\n"
        f"Total genes:     {len(qc_df)}\n"
        f"Training:        {len(train)}\n"
        f"Held-out test:   {len(test)}\n\n"
        f"Cosine sim < {cos_thresh}:  {n_bad_cos} genes\n"
        f"AUC < {auc_thresh}:        {n_bad_auc} genes\n"
        f"Both bad:        {n_bad_both} genes\n\n"
        f"Metrics (train / test):\n"
        f"  Cos sim:    {train['cos_sim'].median():.3f} "
        f"/ {test['cos_sim'].median():.3f}\n"
        "  Cos (resid):"
        f"{train.get('cos_sim_resid', train['cos_sim']).median():.3f} "
        f"/ {test.get('cos_sim_resid', test['cos_sim']).median():.3f}\n"
        f"  AUC:        {train['auc'].median():.3f} / {test['auc'].median():.3f}\n"
        f"  Spearman:   {train['spearman_nz'].median():.3f} "
        f"/ {test['spearman_nz'].median():.3f}\n"
    )
    # Flag specific genes
    bad_genes = qc_df[bad_cos & bad_auc].sort_values("cos_sim")
    if len(bad_genes) > 0:
        summary_text += f"\nPoor genes (cos<{cos_thresh} & AUC<{auc_thresh}):\n"
        for gene, row in bad_genes.head(10).iterrows():
            sp_name = str(row.get("gene_spatial", gene))[:20]
            summary_text += (
                f"  {sp_name}: cos={row['cos_sim']:.2f}, AUC={row['auc']:.2f}\n"
            )

    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(output_dir / "tangram_gene_qc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    typer.echo("  Saved: tangram/tangram_gene_qc.png")


def _plot_sparsity_from_qc(qc_df, output_dir):
    """Sparsity analysis figure using pre-computed gene QC dataframe."""

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sparsity scatter
    ax = axes[0]
    for subset, label, color, marker in [
        (qc_df[qc_df["is_training"]], "Train", "#3C5488", "o"),
        (qc_df[~qc_df["is_training"]], "Test", "#E64B35", "^"),
    ]:
        ax.scatter(
            subset["frac_nz_reference"],
            subset["frac_nz_spatial"],
            s=40,
            alpha=0.7,
            c=color,
            marker=marker,
            label=label,
            edgecolors="white",
            linewidths=0.5,
        )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("Detection Rate (snRNA reference)")
    ax.set_ylabel("Detection Rate (spatial)")
    ax.set_title("Gene Detection: Spatial vs Reference")
    ax.legend(fontsize=8)

    # Detection rate distributions
    ax = axes[1]
    ax.hist(
        qc_df["frac_nz_spatial"],
        bins=20,
        alpha=0.6,
        color=CASTE_COLORS["worker"],
        label="Spatial",
        edgecolor="white",
    )
    ax.hist(
        qc_df["frac_nz_reference"],
        bins=20,
        alpha=0.6,
        color=CASTE_COLORS["queen"],
        label="Reference",
        edgecolor="white",
    )
    ax.set_xlabel("Detection Rate (frac non-zero)")
    ax.set_ylabel("Count")
    ax.set_title("Gene Detection Rate Distributions")
    ax.legend(fontsize=8)

    # Detection rate mismatch
    ax = axes[2]
    sorted_df = qc_df.sort_values("sparsity_diff")
    colors = [
        "#DAA520" if d > 0.2 else "#4A90D9" if d < -0.2 else "#3C5488"
        for d in sorted_df["sparsity_diff"]
    ]
    ax.bar(
        range(len(sorted_df)),
        sorted_df["sparsity_diff"].values,
        color=colors,
        alpha=0.8,
    )
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(0.2, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(-0.2, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Gene (sorted)")
    ax.set_ylabel("Detection Diff (spatial − reference)")
    ax.set_title("Per-Gene Detection Mismatch")

    plt.tight_layout()
    fig.savefig(output_dir / "tangram_sparsity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    typer.echo("  Saved: tangram/tangram_sparsity.png")


def _plot_sparsity(adata, adata_ref, tangram_dir, output_dir, input_file):
    """Sparsity analysis — per-gene sparsity in spatial vs reference."""
    from scipy.sparse import issparse

    # Try saved training gene scores first
    train_genes_path = tangram_dir / f"{input_file}_train_genes.csv"
    if train_genes_path.exists():
        typer.echo("  Loading saved training gene scores...")
        df = pd.read_csv(train_genes_path, index_col=0)
    else:
        df = None

    # Compute sparsity from raw data if needed
    if df is None:
        gene_mapping = load_gene_mapping()
        sp_to_ref = {}
        for g in adata.var_names:
            bimp_id = g.split("|")[0] if "|" in g else g
            sp_to_ref[g] = gene_mapping.get(bimp_id, g)

        ref_genes = set(adata_ref.var_names)
        common = [
            (sp_g, ref_id) for sp_g, ref_id in sp_to_ref.items() if ref_id in ref_genes
        ]

        if not common:
            typer.echo("  WARNING: No common genes — skipping sparsity")
            return

        typer.echo(f"  Computing sparsity for {len(common)} common genes...")
        records = []
        for sp_g, ref_id in common:
            sp_col = adata[:, sp_g].X
            if issparse(sp_col):
                sp_col = sp_col.toarray()
            ref_col = adata_ref[:, ref_id].X
            if issparse(ref_col):
                ref_col = ref_col.toarray()
            records.append(
                {
                    "gene": ref_id,
                    "gene_spatial": sp_g,
                    "sparsity_sp": float((sp_col == 0).mean()),
                    "sparsity_sc": float((ref_col == 0).mean()),
                    "sparsity_diff": float(
                        (sp_col == 0).mean() - (ref_col == 0).mean()
                    ),
                }
            )
        df = pd.DataFrame(records).set_index("gene")

    if df is None or len(df) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Sparsity scatter
    ax = axes[0]
    ax.scatter(
        df["sparsity_sc"],
        df["sparsity_sp"],
        s=40,
        alpha=0.7,
        c="#3C5488",
        edgecolors="white",
        linewidths=0.5,
    )
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    outliers = df[df["sparsity_diff"].abs() > 0.2]
    for gene, row in outliers.iterrows():
        label = str(row.get("gene_spatial", gene))[:15]
        ax.annotate(
            label,
            (row["sparsity_sc"], row["sparsity_sp"]),
            fontsize=6,
            ha="center",
            va="bottom",
            alpha=0.8,
        )
    ax.set_xlabel("Sparsity (scRNA-seq reference)")
    ax.set_ylabel("Sparsity (spatial)")
    ax.set_title("Gene Sparsity: Spatial vs Reference")

    # Sparsity distributions
    ax = axes[1]
    ax.hist(
        df["sparsity_sp"],
        bins=20,
        alpha=0.6,
        color=CASTE_COLORS["worker"],
        label="Spatial",
        edgecolor="white",
    )
    ax.hist(
        df["sparsity_sc"],
        bins=20,
        alpha=0.6,
        color=CASTE_COLORS["queen"],
        label="Reference",
        edgecolor="white",
    )
    ax.set_xlabel("Sparsity (fraction zeros)")
    ax.set_ylabel("Count")
    ax.set_title("Gene Sparsity Distributions")
    ax.legend(fontsize=8)

    # Sparsity difference
    ax = axes[2]
    sorted_df = df.sort_values("sparsity_diff")
    colors = [
        "#DAA520" if d > 0.2 else "#4A90D9" if d < -0.2 else "#3C5488"
        for d in sorted_df["sparsity_diff"]
    ]
    ax.bar(
        range(len(sorted_df)),
        sorted_df["sparsity_diff"].values,
        color=colors,
        alpha=0.8,
    )
    ax.axhline(0, color="black", linestyle="-", linewidth=0.5)
    ax.axhline(0.2, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axhline(-0.2, color="red", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_xlabel("Gene (sorted)")
    ax.set_ylabel("Sparsity Diff (spatial − reference)")
    ax.set_title("Per-Gene Sparsity Discrepancy")

    plt.tight_layout()
    fig.savefig(output_dir / "tangram_sparsity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    n_high = (df["sparsity_diff"].abs() > 0.2).sum()
    typer.echo("  Saved: tangram/tangram_sparsity.png")
    typer.echo(f"  Genes with |sparsity_diff| > 0.2: {n_high}/{len(df)}")
    typer.echo(
        f"  Mean spatial sparsity: {df['sparsity_sp'].mean():.3f}, "
        f"reference: {df['sparsity_sc'].mean():.3f}"
    )


if __name__ == "__main__":
    app()
