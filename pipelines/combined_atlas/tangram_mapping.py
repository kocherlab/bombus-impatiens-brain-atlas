#!/usr/bin/env python3
"""
Tangram cell type mapping and gene imputation.

Trains Tangram on marker genes from the snRNA reference, projects reference
gene expression onto spatial cells, and computes QC metrics.

Usage:
    python tangram_mapping.py train   # Train Tangram + project genes + QC
    python tangram_mapping.py qc      # Recompute QC on existing results

Design (relative to the combined_analysis.py tangram command):
    1. Training genes = all panel ∩ reference (96 train, 5 held-out test)
    2. density_prior = rna_count_based (accounts for cell density)
    3. Project only HVGs/markers from reference (not all 10k genes)
    4. QC via tg.compare_spatial_geneexp (cosine similarity + AUC)
    5. Stratified reference subsampling for GPU memory efficiency
"""

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import typer

app = typer.Typer(help="Tangram cell type mapping")

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"
TANGRAM_DIR = RESULTS_ROOT / "tangram"
RESULTS_QC = RESULTS_ROOT / "tangram_qc"
RESULTS_IMPUTED = RESULTS_ROOT / "imputed_genes"

import os as _os

# Canonical reference is the annotated CB+OL snRNA-seq atlas
# (obs['cell_type'] with 29 union types). Override with env
# var BIMP_SNRNA_REF to point at any other h5ad.
SNRNA_REF = Path(
    _os.environ.get(
        "BIMP_SNRNA_REF",
        str(
            Path(__file__).resolve().parents[2] / "data/sn/BIMP_atlas_qw_combined.h5ad"
        ),
    )
)
# Gene mapping helpers live in the shared package — single source of truth.
from bimp_st.utils.gene_mapping import (  # noqa: E402
    GENE_MAPPING_FILE,
    load_gene_mapping,
    map_spatial_genes_to_v43,
)

CASTE_COLORS = {"queen": "#DAA520", "worker": "#4A90D9"}


# =============================================================================
# UTILITIES
# =============================================================================


def map_spatial_genes(adata_sp, gene_mapping: dict):
    """Map spatial gene names (BIMP|MGEN format) to v4.3 IDs in place."""
    new_names = map_spatial_genes_to_v43(adata_sp.var_names, gene_mapping)
    adata_sp.var_names = pd.Index(new_names)
    adata_sp.var_names_make_unique()
    return adata_sp


def _get_reference_markers(adata_ref, top_n=100) -> list[str]:
    """Get marker genes from reference rank_genes_groups (top N per cluster)."""
    if "rank_genes_groups" not in adata_ref.uns:
        typer.echo("  WARNING: No rank_genes_groups in reference, using HVGs instead")
        return list(adata_ref.var_names[adata_ref.var["highly_variable"]])

    markers_df = pd.DataFrame(adata_ref.uns["rank_genes_groups"]["names"])
    top = markers_df.iloc[:top_n, :]
    markers = list(np.unique(top.melt().value.values))
    typer.echo(
        f"  Reference markers (top {top_n}/cluster): {len(markers)} unique genes"
    )
    return markers


def _set_pub_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


# Default thresholds for the categorical confidence flag stored alongside
# `tangram_cell_type`. Top1 alone is a poor filter when probabilities are
# spread across types; margin (top1 - top2) catches "two types fighting it
# out" cases that high top1 misses.
LABEL_QC_TOP1_MIN = 0.30
LABEL_QC_MARGIN_MIN = 0.10


def _compute_label_qc(
    ct_pred, ct_labels=None, top1_min=LABEL_QC_TOP1_MIN, margin_min=LABEL_QC_MARGIN_MIN
):
    """Summarize per-cell label-transfer confidence.

    Returns top1 / top2 / margin / entropy plus a categorical qc flag.
    Margin = top1 - top2 is the key filter; entropy is reported for
    diagnostics (low entropy => sharp distribution).
    """
    arr = ct_pred.values if hasattr(ct_pred, "values") else ct_pred
    arr = np.asarray(arr, dtype=np.float64)
    # Ensure rows sum to ~1 so threshold semantics are stable
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    arr = arr / row_sums

    sorted_desc = np.sort(arr, axis=1)[:, ::-1]
    top1 = sorted_desc[:, 0]
    top2 = sorted_desc[:, 1] if sorted_desc.shape[1] > 1 else np.zeros_like(top1)
    margin = top1 - top2

    safe = np.where(arr > 0, arr, 1.0)
    entropy = -(arr * np.log(safe)).sum(axis=1)

    if ct_labels is not None:
        argmax = arr.argmax(axis=1)
        top_label = np.asarray([ct_labels[i] for i in argmax])
    else:
        top_label = None

    qc_flag = np.where(
        (top1 >= top1_min) & (margin >= margin_min),
        "confident",
        "ambiguous",
    )
    return {
        "tangram_cell_type": top_label,
        "tangram_confidence": top1.astype(np.float32),
        "tangram_top2": top2.astype(np.float32),
        "tangram_margin": margin.astype(np.float32),
        "tangram_entropy": entropy.astype(np.float32),
        "tangram_label_qc": qc_flag,
    }


# Gate for the column-identity self-check.
#
# The statistic is the ratio of total profile correlation under identity
# naming to the Hungarian optimum, not a count of self-matching columns: with a
# 100-gene panel, near-degenerate types (glial classes, OL types) can swap
# harmlessly, which barely moves the ratio, while a systematic permutation
# drags it far down.
#
# Calibrated on results/prior_distortion/column_identity*.csv: runs named by
# identity score a ratio near 1, runs with position-assigned names score far
# lower, and 0.95 sits well clear of both groups.
COLUMN_IDENTITY_MIN_TRACE_RATIO = 0.95
# Below this, the per-column agreement is reported as a warning only.
COLUMN_IDENTITY_WARN_AGREE = 0.80


def _check_column_identity(
    ct_pred_matrix,
    adata_sp_sub,
    adata_ref_sub,
    cluster_label,
    min_ratio=COLUMN_IDENTITY_MIN_TRACE_RATIO,
):
    """Do the posterior columns carry the reference profiles they are named for?

    Independent of how the names were derived: for each posterior column, take
    the mean panel-gene profile of the cells that argmax to it, and correlate
    it against every reference type's mean profile over the same genes (genes
    z-scored across categories first, so shared depth structure does not swamp
    identity structure). A correctly named column's best 1-1 match should be
    its own name, and the total correlation under identity naming should be
    close to the Hungarian optimum.

    Returns ``(passed, DataFrame, summary_dict)``. Never raises on data
    problems — a check that quietly skips itself is worse than no check, so an
    unusable comparison returns ``passed=False`` with a reason.
    """
    from scipy.optimize import linear_sum_assignment
    from scipy.sparse import issparse as _isp

    cols = [str(c) for c in ct_pred_matrix.columns]
    P = np.asarray(ct_pred_matrix.values, dtype=np.float64)
    lab = np.asarray(cols)[P.argmax(axis=1)]

    sp_genes = pd.Index([str(g) for g in adata_sp_sub.var_names])
    ref_genes = pd.Index([str(g) for g in adata_ref_sub.var_names])
    lut = {g.lower(): g for g in ref_genes}
    shared_sp = [g for g in sp_genes if g.lower() in lut]
    shared_ref = [lut[g.lower()] for g in shared_sp]
    if len(shared_sp) < 20:
        return (
            False,
            pd.DataFrame(),
            {"reason": f"only {len(shared_sp)} shared genes; cannot check"},
        )

    si = [sp_genes.get_loc(g) for g in shared_sp]
    ri = [ref_genes.get_loc(g) for g in shared_ref]

    Xs = adata_sp_sub.X
    Xs = Xs.toarray() if _isp(Xs) else np.asarray(Xs)
    Xs = np.asarray(Xs[:, si], dtype=np.float64)
    n_by_col = {c: int((lab == c).sum()) for c in cols}
    SP = pd.DataFrame(
        {
            c: (Xs[lab == c].mean(0) if n_by_col[c] else np.full(len(si), np.nan))
            for c in cols
        },
        index=shared_sp,
    ).T
    del Xs

    rct = pd.Series([str(v) for v in adata_ref_sub.obs[cluster_label]])
    Xr = adata_ref_sub.X
    Xr = Xr.toarray() if _isp(Xr) else np.asarray(Xr)
    Xr = np.asarray(Xr[:, ri], dtype=np.float64)
    types = sorted(set(rct))
    RP = pd.DataFrame(
        np.vstack([Xr[(rct == t).to_numpy()].mean(0) for t in types]),
        index=types,
        columns=shared_sp,
    )
    del Xr

    # log1p tames the count tails; z-scoring per gene across categories makes
    # the two differently-normalized matrices comparable.
    def _z(D):
        A = np.log1p(D.to_numpy())
        return (A - np.nanmean(A, 0)) / (np.nanstd(A, 0) + 1e-9)

    live = [c for c in cols if n_by_col[c] > 0]
    SPz, RPz = _z(SP.loc[live]), _z(RP)
    C = pd.DataFrame(
        np.corrcoef(SPz, RPz)[: len(live), len(live) :], index=live, columns=RP.index
    )

    r, c_ = linear_sum_assignment(-np.nan_to_num(C.to_numpy(), nan=-1.0))
    assign = {C.index[i]: C.columns[j] for i, j in zip(r, c_, strict=True)}

    res = pd.DataFrame(
        [
            {
                "column_label": c,
                "n_argmax_cells": n_by_col[c],
                "corr_to_own_name": float(C.loc[c, c]) if c in C.columns else np.nan,
                "best_corr_type": C.loc[c].idxmax(),
                "best_corr": float(C.loc[c].max()),
                "hungarian_type": assign[c],
                "corr_hungarian": float(C.loc[c, assign[c]]),
                "hungarian_eq_name": assign[c] == c,
            }
            for c in live
        ]
        + [
            {
                "column_label": c,
                "n_argmax_cells": 0,
                "corr_to_own_name": np.nan,
                "best_corr_type": None,
                "best_corr": np.nan,
                "hungarian_type": None,
                "corr_hungarian": np.nan,
                "hungarian_eq_name": False,
            }
            for c in cols
            if n_by_col[c] == 0
        ]
    )

    n_ok = int(res["hungarian_eq_name"].sum())
    agree = n_ok / max(len(cols), 1)
    tr_id = float(np.nansum([C.loc[c, c] for c in live if c in C.columns]))
    tr_hu = float(np.nansum([C.loc[i, assign[i]] for i in C.index]))
    ratio = tr_id / tr_hu if tr_hu > 0 else 0.0
    n_alien = sum(1 for c in cols if c not in C.columns)
    summary = {
        "n_columns": len(cols),
        "n_columns_with_cells": len(live),
        "n_hungarian_eq_name": n_ok,
        "agreement": float(agree),
        "trace_identity": tr_id,
        "trace_hungarian": tr_hu,
        "trace_ratio": float(ratio),
        "min_trace_ratio": float(min_ratio),
        "n_shared_genes": len(shared_sp),
        # A column named for a type the reference does not contain means the
        # names came from somewhere other than the mapping matrix's own rows.
        "n_columns_named_not_in_reference": int(n_alien),
        "passed": bool(ratio >= min_ratio and n_alien == 0),
    }
    return summary["passed"], res, summary


def _plot_tangram_qc(df_scores, ad_map, out_dir):
    """Plot Tangram QC figures (replaces tg.plot_auc/training/test_scores which use
    deprecated seaborn API)."""
    from sklearn.metrics import auc as sk_auc

    # --- AUC plot: score vs sparsity with fit curve ---
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["#3C5488" if t else "#E64B35" for t in df_scores["is_training"]]
    ax.scatter(
        df_scores["score"],
        df_scores["sparsity_sp"],
        c=colors,
        s=50,
        alpha=0.6,
        edgecolors="white",
        linewidths=0.5,
    )
    # Compute AUC from sorted scores
    test_df = df_scores[~df_scores["is_training"]].sort_values("score")
    if len(test_df) > 2:
        xs = test_df["score"].values
        ys = test_df["sparsity_sp"].values
        # Fit polynomial for AUC curve
        coeffs = np.polyfit(xs, ys, deg=min(3, len(xs) - 1))
        x_smooth = np.linspace(0, 1, 100)
        y_smooth = np.clip(np.polyval(coeffs, x_smooth), 0, 1)
        ax.plot(x_smooth, y_smooth, "r-", linewidth=2, alpha=0.7)
        auc_val = sk_auc(x_smooth, y_smooth)
        ax.text(
            0.03,
            0.1,
            f"auc_score={auc_val:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Spatial Sparsity")
    ax.set_title("Prediction on test transcriptome")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#3C5488",
                markersize=8,
                label="Training",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="#E64B35",
                markersize=8,
                label="Test",
            ),
        ],
        fontsize=9,
    )
    fig.savefig(out_dir / "tangram_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Training scores (4-panel like tg.plot_training_scores) ---
    if "train_genes_df" in ad_map.uns:
        df_train = ad_map.uns["train_genes_df"]
        fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
        axes[0].hist(
            df_train["train_score"],
            bins=20,
            color="coral",
            alpha=0.7,
            edgecolor="white",
        )
        axes[0].set_ylabel("Score")
        axes[0].set_xlabel("Count")
        for i, col in enumerate(["sparsity_sc", "sparsity_sp", "sparsity_diff"], 1):
            if col in df_train.columns:
                axes[i].scatter(
                    df_train[col],
                    df_train["train_score"],
                    c="coral",
                    s=30,
                    alpha=0.6,
                    edgecolors="white",
                    linewidths=0.3,
                )
                axes[i].set_xlabel(col.replace("_", " ").title())
                axes[i].set_title(f"Score vs {col}")
        plt.suptitle("Training gene diagnostics", fontsize=12, y=1.02)
        plt.tight_layout()
        fig.savefig(
            out_dir / "tangram_training_scores.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    # --- Test scores (score distribution + score vs sparsity) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    train_s = df_scores[df_scores["is_training"]]["score"]
    test_s = df_scores[~df_scores["is_training"]]["score"]
    axes[0].hist(
        train_s,
        bins=20,
        alpha=0.6,
        color="#3C5488",
        label=f"Train (n={len(train_s)}, med={train_s.median():.2f})",
        edgecolor="white",
    )
    axes[0].hist(
        test_s,
        bins=20,
        alpha=0.6,
        color="#E64B35",
        label=f"Test (n={len(test_s)}, med={test_s.median():.2f})",
        edgecolor="white",
    )
    axes[0].set_xlabel("Cosine Similarity")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Gene Score Distribution")

    axes[1].scatter(
        df_scores["sparsity_sp"],
        df_scores["score"],
        c=colors,
        s=40,
        alpha=0.6,
        edgecolors="white",
        linewidths=0.3,
    )
    axes[1].set_xlabel("Spatial Sparsity")
    axes[1].set_ylabel("Cosine Similarity")
    axes[1].set_title("Score vs Spatial Sparsity")

    if "sparsity_diff" in df_scores.columns:
        axes[2].scatter(
            df_scores["sparsity_diff"],
            df_scores["score"],
            c=colors,
            s=40,
            alpha=0.6,
            edgecolors="white",
            linewidths=0.3,
        )
        axes[2].axvline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
        axes[2].set_xlabel("Sparsity Diff (spatial − reference)")
        axes[2].set_ylabel("Cosine Similarity")
        axes[2].set_title("Score vs Sparsity Mismatch")

    plt.tight_layout()
    fig.savefig(out_dir / "tangram_test_scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# TRAIN COMMAND
# =============================================================================


@app.command()
def train(
    input_file: str = typer.Option("combined_clustered", help="Input clustered h5ad"),
    max_cells: int = typer.Option(100000, help="Max spatial cells for training"),
    max_ref_cells: int = typer.Option(
        15000, help="Max reference cells (stratified by cluster). 0=no limit"
    ),
    ref_sampling: str = typer.Option(
        "balanced",
        help="How to split max_ref_cells across clusters: 'balanced' "
        "(equal per cluster; the historical behaviour, keeps every "
        "published result reproducible) or 'proportional' (each cluster "
        "gets its true share of the reference, with a floor). See the "
        "note at the subsampling block.",
    ),
    device: str = typer.Option("cuda", help="Device (cuda or cpu)"),
    train_ratio: float = typer.Option(
        1.0, help="Fraction of marker genes for training (rest become test)"
    ),
    seed: int = typer.Option(42, help="Random seed"),
    mode: str = typer.Option("cells", help="Tangram mode: 'cells' or 'clusters'"),
    cluster_label: str = typer.Option(
        "leiden_k35",
        help="Cluster label for clusters mode (k-sweep ref: leiden_k35 IS the "
        "cell-type set; leiden_k14 is the super-type fallback; "
        "the annotated CB+OL ref: cell_type)",
    ),
    num_epochs: int = typer.Option(500, help="Training epochs"),
    density_prior: str = typer.Option(
        "rna_count_based",
        help="Spatial-side density prior: 'rna_count_based', 'uniform', or "
        "'none'. 'none' is the ONLY way to switch the density term off -- "
        "see lambda_d.",
    ),
    lambda_d: float = typer.Option(
        0.0,
        help="Weight on the density term. NOTE: 0 does NOT disable it. "
        "map_cells_to_space (mapping_utils.py:225) reads "
        "`if density_prior is not None and lambda_d == 0: lambda_d = 1`, "
        "so 0 is a sentinel meaning 'unset' and is coerced to 1 whenever a "
        "density_prior exists -- which it does by default. Measured:"
        " lambda_d 0.0 and 1.0 give byte-identical training "
        "traces and identical labels. Every shipped run therefore had the "
        "density term ON at weight 1. To turn it off pass "
        "--density-prior none (with lambda_d 0). With source_density='none' "
        "the term uses a UNIFORM source prior (1/n_clusters), not 'no prior'.",
    ),
    source_density: str = typer.Option(
        "none",
        help="Reference-side density in cluster mode: 'none' (uniform source "
        "prior) or 'cluster_density' (each cluster weighted by its share "
        "of the reference). This -- not lambda_d -- is what is actually "
        "disabled upstream: tangram2 documents d_source as 'to be used when "
        "S corresponds to cluster-level expression', and the assignment is "
        "commented out at mapping_utils.py:346-347 (incidentally, in commit "
        "9308fd8). Without it every cluster spends one unit of softmax mass "
        "whatever its size, so small clusters can become attractors. Only "
        "meaningful when mode='clusters'.",
    ),
    block_mass: str = typer.Option(
        "",
        help="With --source-density cluster_density_within: the total prior "
        "mass each dissection block receives, as "
        "'CentralBrain=0.502,OpticLobe=0.498'. The reference can fix the "
        "weights WITHIN a block (same dissection, same prep) but says "
        "nothing about the ratio BETWEEN blocks -- its block totals are "
        "sequencing effort (CB 53,230 vs OL 20,233 nuclei). Left empty, "
        "the block masses default to the uniform prior's split "
        "(n_clusters_in_block / n_clusters), which is set by how finely "
        "the vocabulary happens to divide each block and is therefore "
        "arbitrary. Supplying masses estimated from the ATLAS removes that "
        "free parameter: the tissue is the only unbiased source for the "
        "between-block ratio. Estimate them from a run with NO source "
        "prior, so the estimate cannot be circular with respect to this "
        "choice.",
    ),
    data_scale: str = typer.Option(
        "linear",
        help="Scale fed to Tangram: 'linear' (production) or 'log1p' "
        "(report-only sensitivity). Tangram is a linear mixture model and "
        "its loss already stabilises variance via a square-root cosine, so "
        "log1p is expected to hurt; the option exists to measure that "
        "rather than assert it.",
    ),
    markers_per_cluster: int = typer.Option(
        200, help="Top N marker genes per cluster for training"
    ),
    project_hvgs_only: bool = typer.Option(
        True, help="Project only HVGs + markers (not all ref genes)"
    ),
    exclude_genes: str = typer.Option(
        "", help="Comma-separated v4.3 gene IDs to exclude"
    ),
    reference: str = typer.Option(
        "", help="Reference h5ad path (overrides BIMP_SNRNA_REF/SNRNA_REF)"
    ),
    region: str = typer.Option(
        "all",
        help="Spatial region: 'all', 'optic' (lamina|medulla|lobula), or "
        "'central' (~optic)",
    ),
    out_tag: str = typer.Option(
        "", help="Suffix appended to output filenames so region runs don't collide"
    ),
):
    """Train Tangram, project cell types + gene expression, compute QC.

    Design:
    - Training genes = reference cluster markers ∩ panel genes (not random)
    - density_prior = rna_count_based (accounts for cell density variation)
    - Projects only HVGs/markers from reference (not all 10k genes)
    - Uses tg.compare_spatial_geneexp for cosine similarity QC
    """
    import tangram2.mapping as tg
    from scipy.sparse import issparse

    random.seed(seed)
    np.random.seed(seed)

    TANGRAM_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_QC.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    input_path = DATA_DIR / f"{input_file}.h5ad"
    # output basename (region runs don't collide)
    stem = f"{input_file}{('_' + out_tag) if out_tag else ''}"
    typer.echo(f"Loading spatial data: {input_path.name}")
    adata_sp_full = sc.read_h5ad(input_path)
    typer.echo(
        f"  Spatial: {adata_sp_full.n_obs:,} cells, {adata_sp_full.n_vars} genes"
    )

    # Region-specific subset (CB/OL separate runs); 'all' = no subset (combined run).
    if region != "all":
        optic_cols = [
            c
            for c in ["in_lamina", "in_medulla", "in_lobula"]
            if c in adata_sp_full.obs.columns
        ]
        optic = (
            adata_sp_full.obs[optic_cols].astype(bool).to_numpy().any(axis=1)
            if optic_cols
            else np.zeros(adata_sp_full.n_obs, dtype=bool)
        )
        mask = optic if region == "optic" else ~optic  # 'central' = complement of optic
        typer.echo(
            f"  Region '{region}': {int(mask.sum()):,}/{adata_sp_full.n_obs:,} cells"
        )
        adata_sp_full = adata_sp_full[mask].copy()

    ref_path = Path(reference) if reference else SNRNA_REF
    typer.echo(f"Loading reference: {ref_path.name}")
    adata_ref = sc.read_h5ad(ref_path)
    typer.echo(
        f"  Reference: {adata_ref.n_obs:,} cells, {adata_ref.n_vars} genes, "
        f"{adata_ref.obs[cluster_label].nunique()} cell types"
    )

    # Compute marker genes per cell type if not already present
    if "rank_genes_groups" not in adata_ref.uns:
        typer.echo(f"  Computing rank_genes_groups on '{cluster_label}'...")
        sc.tl.rank_genes_groups(adata_ref, groupby=cluster_label, method="wilcoxon")
        n_groups = len(adata_ref.uns["rank_genes_groups"]["names"].dtype.names)
        typer.echo(f"  Done — {n_groups} groups")

    # Subsample reference for training (mapping matrix = n_ref × n_spatial × float32)
    #
    # ref_sampling sets how the quota is split across clusters. Under "balanced"
    # each cluster gets 1/n_clusters of the training signal regardless of size,
    # so an entry's influence depends on how finely the vocabulary splits it:
    # a type split into several entries gets several quotas, and pooling it into
    # one entry shrinks its share of the sample (and its recall). "proportional"
    # uses each cluster's true share, with a floor
    # for rare types. "balanced" is the default and reproduces existing results.
    if max_ref_cells > 0 and adata_ref.n_obs > max_ref_cells:
        clusters = adata_ref.obs[cluster_label]
        n_clusters = clusters.nunique()
        keep_idx = []
        if ref_sampling == "proportional":
            floor = min(50, max_ref_cells // max(n_clusters, 1))
            sizes = {cl: int(np.sum(clusters == cl)) for cl in clusters.unique()}
            total = sum(sizes.values())
            # proportional allocation, floored, then scaled back under the cap
            want = {
                cl: max(floor, int(round(max_ref_cells * n / total)))
                for cl, n in sizes.items()
            }
            over = sum(min(want[cl], sizes[cl]) for cl in want) / max_ref_cells
            if over > 1:
                want = {cl: max(floor, int(want[cl] / over)) for cl in want}
            for cl in clusters.unique():
                cl_idx = np.where(clusters == cl)[0]
                n_take = min(len(cl_idx), want[cl])
                keep_idx.extend(np.random.choice(cl_idx, n_take, replace=False))
            keep_idx = sorted(keep_idx)
            typer.echo(
                f"  Subsampling reference for training: {adata_ref.n_obs:,} → "
                f"{len(keep_idx):,} (PROPORTIONAL, floor {floor}, "
                f"{n_clusters} clusters)"
            )
        else:
            per_cluster = max(50, max_ref_cells // n_clusters)
            for cl in clusters.unique():
                cl_idx = np.where(clusters == cl)[0]
                n_take = min(len(cl_idx), per_cluster)
                keep_idx.extend(np.random.choice(cl_idx, n_take, replace=False))
            keep_idx = sorted(keep_idx)
            typer.echo(
                f"  Subsampling reference for training: {adata_ref.n_obs:,} "
                f"→ {len(keep_idx):,} "
                f"(~{per_cluster}/cluster, {n_clusters} clusters)"
            )
        adata_ref = adata_ref[keep_idx].copy()
        est_gb = max_cells * len(keep_idx) * 4 / 1e9
        typer.echo(f"  Mapping matrix estimate: {est_gb:.1f} GB")
    else:
        typer.echo(f"  Using all {adata_ref.n_obs:,} reference cells for training")

    # Map spatial genes to v4.3 IDs
    typer.echo("Mapping spatial gene names to v4.3 IDs...")
    gene_mapping = load_gene_mapping()
    typer.echo(f"  Loaded {len(gene_mapping)} gene mappings from crosswalk")
    adata_sp_full = map_spatial_genes(adata_sp_full, gene_mapping)

    # =========================================================================
    # GENE SELECTION: use reference markers ∩ panel genes
    # =========================================================================
    ref_markers = _get_reference_markers(adata_ref, top_n=markers_per_cluster)
    panel_genes = set(adata_sp_full.var_names)
    ref_genes = set(adata_ref.var_names)

    # Common genes = panel ∩ reference.
    #
    # Sorted so the held-out draw is reproducible: this list orders
    # marker_training / non_marker / all_common and hence `testable`, which
    # `random.shuffle` draws from below. Set iteration order over strings varies
    # per process (PYTHONHASHSEED is unset), so an unsorted list would hold out
    # different genes at the same --seed.
    common_genes = sorted(panel_genes & ref_genes)
    typer.echo(f"Common genes (panel ∩ reference): {len(common_genes)}")

    # Training candidates = common genes that are also reference markers
    marker_training = [g for g in common_genes if g in set(ref_markers)]
    non_marker = [g for g in common_genes if g not in set(ref_markers)]
    typer.echo(f"  Panel genes that are reference markers: {len(marker_training)}")
    typer.echo(f"  Panel genes that are NOT markers: {len(non_marker)}")

    # Exclude genes
    genes_to_exclude = set()
    if exclude_genes:
        genes_to_exclude.update(
            g.strip() for g in exclude_genes.split(",") if g.strip()
        )
    if genes_to_exclude:
        marker_training = [g for g in marker_training if g not in genes_to_exclude]
        non_marker = [g for g in non_marker if g not in genes_to_exclude]
        typer.echo(
            f"  After exclusions: {len(marker_training)} "
            f"markers, {len(non_marker)} non-markers"
        )

    if len(common_genes) < 50:
        typer.echo(f"ERROR: Only {len(common_genes)} common genes. Need at least 50.")
        raise typer.Exit(1)

    # Protect focal genes (must always be in training, never held out)
    np31_to_v43 = gene_mapping  # already loaded above
    v43_to_np31 = {v: k for k, v in np31_to_v43.items()}
    FOCAL_NP31 = {
        "BIMP_vNP3.1_05833",  # Vg (vitellogenin)
        "BIMP_vNP3.1_01378",  # Kr-h1
        "BIMP_vNP3.1_05619",  # yl (vitellogenin receptor)
        "BIMP_vNP3.1_10812",  # GAD1
        "BIMP_vNP3.1_08000",  # apolpp (apolipophorin)
    }
    protected = {g for g in common_genes if v43_to_np31.get(g, "") in FOCAL_NP31}
    if protected:
        typer.echo(f"  Protected focal genes (always train): {len(protected)}")

    # Train on all but ~15 held-out genes (85/15 split per Tangram best practices)
    # Hold-outs drawn from non-focal pool with ≥15% detection
    all_common = marker_training + non_marker
    from scipy.sparse import issparse as _issparse

    sp_X = adata_sp_full[:, all_common].X
    if _issparse(sp_X):
        det_frac = np.asarray((sp_X > 0).mean(axis=0)).flatten()
    else:
        det_frac = (sp_X > 0).mean(axis=0)
    det_map = dict(zip(all_common, det_frac, strict=True))
    testable = [
        g for g in all_common if g not in protected and det_map.get(g, 0) >= 0.15
    ]
    random.shuffle(testable)
    n_test = min(15, len(testable))
    test_genes = testable[:n_test]
    training_genes = [g for g in all_common if g not in set(test_genes)]
    typer.echo(
        f"Training on {len(training_genes)} genes, holding back {n_test} for test "
        f"(test pool: {len(testable)} genes with ≥15% detection)"
    )

    # =========================================================================
    # SUBSAMPLE spatial
    # =========================================================================
    if adata_sp_full.n_obs > max_cells:
        typer.echo(f"Subsampling spatial to {max_cells} cells for training...")
        train_idx = np.random.choice(adata_sp_full.n_obs, max_cells, replace=False)
        adata_sp = adata_sp_full[train_idx].copy()
    else:
        adata_sp = adata_sp_full.copy()

    # =========================================================================
    # PREPARE DATA (linear scale for Tangram)
    # =========================================================================
    # Reference: raw counts + normalize_total (no log1p)
    if "counts" in adata_ref.layers:
        typer.echo("  Using raw counts layer for reference")
        adata_ref_sub = adata_ref[:, all_common].copy()
        adata_ref_sub.X = adata_ref_sub.layers["counts"].copy()
        sc.pp.normalize_total(adata_ref_sub)
    else:
        typer.echo("  WARNING: No counts layer, reversing log1p")
        adata_ref_sub = adata_ref[:, all_common].copy()
        adata_ref_sub.X = np.expm1(
            adata_ref_sub.X.toarray() if issparse(adata_ref_sub.X) else adata_ref_sub.X
        )
        sc.pp.normalize_total(adata_ref_sub)

    # Spatial: reverse log1p
    typer.echo("  Reversing log1p on spatial data")
    adata_sp_sub = adata_sp[:, all_common].copy()
    adata_sp_sub.X = np.expm1(
        adata_sp_sub.X.toarray() if issparse(adata_sp_sub.X) else adata_sp_sub.X
    ).astype(np.float32)

    if data_scale == "log1p":
        # Report-only sensitivity, not a production option. Tangram's model is a
        # linear mixture, G_pred = M_probs.T @ S: a cell that is a mixture of
        # two types carries the mixture of their profiles, which log1p breaks.
        # The loss already applies its own variance stabilisation -- a
        # Bhattacharyya coefficient on square roots,
        # cosine_similarity(sqrt(G_pred), G_sqrt) -- so log1p on top is a second
        # compression. Expected to degrade the held-out cosine; run to check
        # rather than assume it.
        typer.echo("  DATA SCALE = log1p on BOTH sides (sensitivity arm)")
        adata_ref_sub.X = np.log1p(
            adata_ref_sub.X.toarray() if issparse(adata_ref_sub.X) else adata_ref_sub.X
        )
        adata_sp_sub.X = np.log1p(adata_sp_sub.X).astype(np.float32)

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
    # PREPARE + TRAIN TANGRAM
    # =========================================================================
    typer.echo("Preparing data for Tangram...")
    tg.pp_adatas(adata_ref_sub, adata_sp_sub, genes=training_genes)

    # "none" -> None is the only route to a genuinely disabled density term:
    # upstream coerces lambda_d 0 -> 1 whenever density_prior is not None
    # (mapping_utils.py:225), and refuses lambda_d > 0 with density_prior None
    # (line 228), so the two must be set together.
    density_prior = (
        None if str(density_prior).lower() in ("none", "") else density_prior
    )
    if density_prior is None and lambda_d > 0:
        raise typer.BadParameter(
            "--density-prior none requires --lambda-d 0 (tangram2 raises "
            "otherwise); together they are the only way to switch the density "
            "term off."
        )

    mode_str = f"mode={mode}" + (
        f", cluster_label={cluster_label}" if mode == "clusters" else ""
    )
    typer.echo(
        f"Training Tangram on {device} ({num_epochs} "
        f"epochs, {len(training_genes)} genes, "
        f"density={density_prior}, {mode_str})..."
    )

    map_kwargs = dict(
        device=device,
        num_epochs=num_epochs,
        density_prior=density_prior,
        mode=mode,
        # Seed the optimizer's own initialisation. random.seed/np.random.seed
        # above only cover the reference subsample and the gene holdout draw;
        # without this, re-running the same command changes the mapping (e.g.
        # the share of cells on the generic "Kenyon cells" entry).
        random_state=seed,
        lambda_d=lambda_d,
    )
    if mode == "clusters":
        map_kwargs["cluster_label"] = cluster_label
    _restore = None
    if mode == "clusters" and source_density.startswith("cluster_density"):
        # Supply the reference-side density that upstream leaves disabled.
        # `map_cells_to_space` does not take d_source (upstream's cluster-mode
        # assignment is commented out), so the vector is injected by wrapping two
        # upstream callables. adata_to_cluster_expression orders rows by
        # value_counts().index (descending frequency), not alphabetically;
        # capturing cluster_density from the object it builds keeps d_source in
        # the order the mapper sees, whereas reindexing to cat.categories would
        # permute the prior. The wrappers are removed in the finally block below.
        from tangram2.mapping import mapping_optimizer as _mo
        from tangram2.mapping import mapping_utils as _mu

        _cap = {}
        _orig_a2c = _mu.adata_to_cluster_expression
        _orig_mapper = _mo.Mapper

        def _a2c_capture(adata, label, *a, **kw):
            _out = _orig_a2c(adata, label, *a, **kw)
            _ad = _out[0] if isinstance(_out, tuple) else _out
            if "cluster_density" in _ad.obs:
                _cap["order"] = [str(c) for c in _ad.obs[label]]
                _cap["dens"] = np.asarray(_ad.obs["cluster_density"], dtype="float32")
            return _out

        class _MapperWithSource(_orig_mapper):
            def __init__(self, *a, **kw):
                # built here, after adata_to_cluster_expression has run, so the
                # vector is in the same row order as S
                kw["d_source"] = _cap["build"]()
                super().__init__(*a, **kw)

        _mu.adata_to_cluster_expression = _a2c_capture
        _mo.Mapper = _MapperWithSource
        _restore = (_orig_a2c, _orig_mapper)

        # Block membership, resolved now but applied to the captured order.
        _blk_col = "region"
        _blk_of = None
        _block_mass = None
        if source_density == "cluster_density_within":
            # Within-block variant. The reference pools two dissections
            # (region: CentralBrain, OpticLobe), so raw
            # cluster_density reflects sequencing effort per dissection, not
            # abundance in a brain. Within one dissection the size effect is
            # real (optic-only types: reference size and atlas share are
            # negatively rank-correlated), so only within-block weights are
            # corrected; the
            # between-block mass stays at the uniform prior's
            # n_clusters_in_block / n_clusters.
            if _blk_col not in adata_ref_sub.obs:
                raise typer.BadParameter(
                    f"source_density=cluster_density_within needs "
                    f"adata_ref.obs['{_blk_col}']"
                )
            _blk_map = adata_ref_sub.obs.groupby(cluster_label, observed=True)[
                _blk_col
            ].agg(lambda s: s.astype(str).value_counts().index[0])
            _blk_of = {str(k): str(v) for k, v in _blk_map.items()}
            if block_mass:
                _block_mass = {}
                for _kv in block_mass.split(","):
                    _k, _v = _kv.split("=")
                    _block_mass[_k.strip()] = float(_v)
                _missing = set(_blk_of.values()) - set(_block_mass)
                if _missing:
                    raise typer.BadParameter(
                        f"--block-mass is missing {sorted(_missing)}"
                    )
                _tot = sum(_block_mass.values())
                _block_mass = {k: v / _tot for k, v in _block_mass.items()}
                typer.echo(
                    "  block mass supplied: "
                    + ", ".join("%s=%.3f" % kv for kv in sorted(_block_mass.items()))
                )

        def _build_source_vector():
            """Final d_source, in the captured (mapper-visible) cluster order."""
            if "dens" not in _cap:
                raise RuntimeError(
                    "cluster_density was never captured -- "
                    "adata_to_cluster_expression did not run with add_density"
                )
            d = _cap["dens"].astype("float64")
            d = d / d.sum()
            if _blk_of is not None:
                blk = np.array([_blk_of[c] for c in _cap["order"]])
                out = np.zeros_like(d)
                for b in np.unique(blk):
                    m = blk == b
                    # block total: the supplied mass if given, else the uniform
                    # prior's own split (cluster count), which leaves the
                    # between-block behaviour exactly as it already is
                    share = _block_mass[b] if _block_mass else m.sum() / len(d)
                    out[m] = d[m] / d[m].sum() * share
                d = out / out.sum()
                typer.echo(
                    "  within-block d_source: "
                    + ", ".join(
                        "%s n_clusters=%d mass=%.3f"
                        % (b, (blk == b).sum(), d[blk == b].sum())
                        for b in np.unique(blk)
                    )
                )
            top = np.argsort(-d)[:3]
            typer.echo(
                "  d_source over %d clusters (min %.5f, max %.5f); "
                "heaviest: %s"
                % (
                    len(d),
                    d.min(),
                    d.max(),
                    ", ".join("%s %.4f" % (_cap["order"][i], d[i]) for i in top),
                )
            )
            return d.astype("float32")

        _cap["build"] = _build_source_vector

    typer.echo(f"  lambda_d={lambda_d}, source_density={source_density}, seed={seed}")
    try:
        ad_map = tg.map_cells_to_space(adata_ref_sub, adata_sp_sub, **map_kwargs)
    finally:
        if _restore is not None:
            _mu.adata_to_cluster_expression, _mo.Mapper = _restore

    # Save training gene scores
    if "train_genes_df" in ad_map.uns:
        train_genes_path = TANGRAM_DIR / f"{stem}_train_genes.csv"
        ad_map.uns["train_genes_df"].to_csv(train_genes_path)
        typer.echo(f"  Saved training gene scores: {train_genes_path.name}")

    # =========================================================================
    # PROJECT cell type annotations
    # =========================================================================
    typer.echo("Projecting cell type annotations...")
    # One code path for both modes. `tg.project_cell_annotations` one-hot-encodes
    # `ad_map.obs[cluster_label]` and computes M.T @ onehot, so column names come
    # from the same object and row order as the mapping matrix.
    #
    # In clusters mode `ad_map.obs_names` are positional integers ("0", "1", ...)
    # because tangram2's `adata_to_cluster_expression` (mapping_utils.py:133)
    # orders rows by `value_counts().index` (descending frequency) of the
    # subsampled reference. Naming those positions via the alphabetical
    # `cat.categories` would put another type's posterior under the wrong name.
    # See `_check_column_identity` below and
    # scripts/identity/prior_distortion_diagnostic.py --stage columns.
    #
    # In clusters mode the one-hot is a permutation matrix, so the numbers are
    # identical to `ad_map.X.T`; only the column names differ.
    tg.project_cell_annotations(ad_map, adata_sp_sub, annotation=cluster_label)
    ct_pred_raw = adata_sp_sub.obsm["tangram_ct_pred"]
    if hasattr(ct_pred_raw, "values"):
        ct_vals = np.asarray(ct_pred_raw.values, dtype=np.float64)
        ct_cols = [str(c) for c in ct_pred_raw.columns]
    else:
        ct_vals = np.asarray(ct_pred_raw, dtype=np.float64)
        ct_cols = None
    if mode == "clusters":
        row_names = [str(x) for x in ad_map.obs[cluster_label]]
        if set(row_names) != set(ct_cols or []):
            raise RuntimeError(
                "Posterior columns do not match the mapping matrix's own row "
                f"labels.\n  rows: {sorted(set(row_names))}\n"
                f"  cols: {sorted(set(ct_cols or []))}"
            )
        typer.echo(
            f"  {len(ct_cols)} cluster columns named from ad_map.obs"
            f"['{cluster_label}'] (mapping-matrix row order): {ct_cols}"
        )

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

    # =========================================================================
    # SELF-CHECK: do the columns carry the profiles they are named for?
    # =========================================================================
    # The check is cheap, independent of how the names were derived, and fails
    # the job (non-zero exit, after outputs are written so GPU time is kept).
    typer.echo("Checking posterior column identity against reference profiles...")
    ident_ok, ident_df, ident_summary = _check_column_identity(
        ct_pred_matrix, adata_sp_sub, adata_ref_sub, cluster_label
    )
    ident_path = TANGRAM_DIR / f"{stem}_column_identity.csv"
    if len(ident_df):
        ident_df.to_csv(ident_path, index=False)
        typer.echo(f"  Saved: {ident_path.name}")
    adata_sp_sub.uns["column_identity_check"] = {
        str(k): v for k, v in ident_summary.items()
    }
    typer.echo(f"  {ident_summary}")
    if ident_ok:
        typer.echo(
            f"  COLUMN IDENTITY OK: identity/Hungarian trace ratio "
            f"{ident_summary.get('trace_ratio', 0):.3f} "
            f">= {COLUMN_IDENTITY_MIN_TRACE_RATIO}; best 1-1 reference match "
            f"== column name for {ident_summary.get('n_hungarian_eq_name')}/"
            f"{ident_summary.get('n_columns')} columns"
        )
        if ident_summary.get("agreement", 1.0) < COLUMN_IDENTITY_WARN_AGREE:
            typer.echo(
                f"  WARNING: only "
                f"{100 * ident_summary['agreement']:.0f}% of columns best-match "
                f"their own name. The trace ratio says this is degenerate "
                f"reference types, not a permutation — but check "
                f"{ident_path.name} before trusting fine subtype labels."
            )
    else:
        typer.echo("\n" + "!" * 78)
        typer.echo("!!  COLUMN IDENTITY CHECK FAILED — DO NOT USE THESE LABELS")
        typer.echo(f"!!  {ident_summary}")
        typer.echo(f"!!  per-column detail: {ident_path}")
        typer.echo("!" * 78 + "\n")
        if len(ident_df):
            typer.echo(ident_df.to_string(index=False))

    label_qc = _compute_label_qc(ct_pred_matrix, ct_labels=list(ct_pred_matrix.columns))
    for col, vals in label_qc.items():
        adata_sp_sub.obs[col] = vals
    n_conf = (adata_sp_sub.obs["tangram_label_qc"] == "confident").sum()
    typer.echo(
        f"  Cell types: {adata_sp_sub.obs['tangram_cell_type'].nunique()} unique"
    )
    typer.echo(
        f"  Mean top1 / margin: "
        f"{adata_sp_sub.obs['tangram_confidence'].mean():.3f} / "
        f"{adata_sp_sub.obs['tangram_margin'].mean():.3f}"
    )
    typer.echo(
        f"  Confident (top1≥{LABEL_QC_TOP1_MIN}, margin≥{LABEL_QC_MARGIN_MIN}): "
        f"{n_conf:,}/{len(adata_sp_sub):,} ({100 * n_conf / len(adata_sp_sub):.1f}%)"
    )

    # =========================================================================
    # PROJECT gene expression (filtered to HVGs + markers)
    # =========================================================================
    typer.echo("Preparing reference for gene projection...")
    adata_ref_proj = (
        adata_ref.copy()
    )  # must match training ref cells (mapping matrix dims)
    if "counts" in adata_ref.layers:
        adata_ref_proj.X = adata_ref_proj.layers["counts"].copy()
    else:
        if issparse(adata_ref_proj.X):
            adata_ref_proj.X = adata_ref_proj.X.toarray()
        adata_ref_proj.X = np.expm1(adata_ref_proj.X)
    sc.pp.normalize_total(adata_ref_proj)

    if project_hvgs_only:
        # Keep HVGs + markers + panel genes (panel needed for QC comparison)
        ref_X = (
            adata_ref_proj.X.toarray()
            if issparse(adata_ref_proj.X)
            else adata_ref_proj.X
        )
        frac_nz = (ref_X > 0).mean(axis=0)
        is_hvg = (
            adata_ref_proj.var["highly_variable"].values
            if "highly_variable" in adata_ref_proj.var
            else np.zeros(adata_ref_proj.n_vars, dtype=bool)
        )
        is_marker = np.isin(adata_ref_proj.var_names, ref_markers)
        is_panel = np.isin(adata_ref_proj.var_names, list(common_genes))
        # Panel genes always kept (needed for QC); others require ≥5% detection
        keep = ((is_hvg | is_marker) & (frac_nz > 0.05)) | is_panel
        n_before = adata_ref_proj.n_vars
        adata_ref_proj = adata_ref_proj[:, keep].copy()
        typer.echo(
            f"  Filtered ref genes: {n_before:,} → {adata_ref_proj.n_vars:,} "
            f"(HVGs + markers, detected in ≥5% cells)"
        )
    else:
        ref_X = (
            adata_ref_proj.X.toarray()
            if issparse(adata_ref_proj.X)
            else adata_ref_proj.X
        )
        frac_nz = (ref_X > 0).mean(axis=0)
        keep = frac_nz >= 0.05
        n_before = adata_ref_proj.n_vars
        adata_ref_proj = adata_ref_proj[:, keep].copy()
        typer.echo(
            f"  Filtered ref genes: {n_before:,} → {adata_ref_proj.n_vars:,} "
            "(≥5% detection)"
        )

    typer.echo(f"Projecting {adata_ref_proj.n_vars:,} genes onto spatial cells...")
    project_kwargs = dict(adata_map=ad_map, adata_sc=adata_ref_proj)
    if mode == "clusters":
        project_kwargs["cluster_label"] = cluster_label
    ad_ge = tg.project_genes(**project_kwargs)
    typer.echo(f"  Imputed expression: {ad_ge.shape}")

    # Copy tangram uns keys from pp_adatas (needed by compare_spatial_geneexp)
    # project_genes lowercases gene names and may drop all-zero genes, so
    # restrict overlap/training genes to those actually present in ad_ge
    ge_genes = set(ad_ge.var_names)
    for key in ("training_genes", "overlap_genes"):
        if key in adata_sp_sub.uns:
            ad_ge.uns[key] = [g for g in adata_sp_sub.uns[key] if g in ge_genes]
    # Also update adata_sp_sub to match (compare_spatial_geneexp asserts equality)
    adata_sp_sub.uns["overlap_genes"] = ad_ge.uns.get("overlap_genes", [])
    adata_sp_sub.uns["training_genes"] = ad_ge.uns.get("training_genes", [])
    typer.echo(
        f"  QC overlap: {len(ad_ge.uns.get('overlap_genes', []))} genes "
        f"({len(ad_ge.uns.get('training_genes', []))} training)"
    )

    # Copy metadata
    for key in ("X_umap", "spatial_aligned"):
        if key in adata_sp_sub.obsm:
            ad_ge.obsm[key] = adata_sp_sub.obsm[key]
    for col in ("cell_id", "brain", "leiden", "slice_name"):
        if col in adata_sp_sub.obs.columns:
            ad_ge.obs[col] = adata_sp_sub.obs[col].values
    for col in (
        "tangram_cell_type",
        "tangram_confidence",
        "tangram_top2",
        "tangram_margin",
        "tangram_entropy",
        "tangram_label_qc",
    ):
        if col in adata_sp_sub.obs.columns:
            ad_ge.obs[col] = adata_sp_sub.obs[col].values

    # =========================================================================
    # QC: tg.compare_spatial_geneexp (cosine similarity)
    # =========================================================================
    typer.echo("Computing QC via tg.compare_spatial_geneexp...")
    try:
        # Sync adata_ref_sub uns to match (pp_adatas set these originally)
        adata_ref_sub.uns["overlap_genes"] = ad_ge.uns["overlap_genes"]
        adata_ref_sub.uns["training_genes"] = ad_ge.uns["training_genes"]
        df_scores = tg.compare_spatial_geneexp(ad_ge, adata_sp_sub, adata_ref_sub)
        df_scores.to_csv(TANGRAM_DIR / f"{stem}_gene_scores.csv")
        typer.echo(f"  Gene scores: {len(df_scores)} genes")

        train_scores = df_scores[df_scores["is_training"]]
        test_scores = df_scores[~df_scores["is_training"]]
        typer.echo(
            f"  Cosine sim — train: {train_scores['score'].median():.3f} (median), "
            f"test: {test_scores['score'].median():.3f}"
        )

        # Compute summary metrics (avg scores, AUC, sparsity-weighted score)
        try:
            eval_results, (auc_coords, raw_coords) = tg.eval_metric(df_scores)
            metrics_path = TANGRAM_DIR / f"{stem}_eval_metrics.json"
            with open(metrics_path, "w") as mf:
                json.dump({k: float(v) for k, v in eval_results.items()}, mf, indent=2)
            typer.echo(f"  eval_metric: {eval_results}")
        except Exception as e:
            typer.echo(f"  WARNING: tg.eval_metric failed: {e}")

        # Plot QC (tangram's built-in plots use deprecated seaborn API, so roll our own)
        _set_pub_style()
        _plot_tangram_qc(df_scores, ad_map, RESULTS_QC)
        typer.echo(f"  Saved QC plots to {RESULTS_QC}")
    except Exception as e:
        typer.echo(f"  WARNING: tg.compare_spatial_geneexp failed: {e}")
        typer.echo("  Falling back to manual cosine similarity...")
        df_scores = None

    # Save imputed expression (for per-gene spatial plots, not UMAP)
    imputed_path = TANGRAM_DIR / f"{stem}_imputed.h5ad"
    ad_ge.write_h5ad(imputed_path)
    typer.echo(f"  Saved: {imputed_path.name}")

    # Save gene split
    gene_split_path = TANGRAM_DIR / f"{stem}_gene_split.json"
    with open(gene_split_path, "w") as f:
        json.dump(
            {
                "training_genes": training_genes,
                "test_genes": test_genes,
                "marker_training": marker_training,
                "non_marker_panel": non_marker,
                "train_ratio": train_ratio,
                "seed": seed,
                "density_prior": density_prior,
                "markers_per_cluster": markers_per_cluster,
            },
            f,
            indent=2,
        )

    # Every cell in adata_sp_sub was directly mapped by Tangram. If the user
    # set --max-cells below the input size, only that subset gets a label;
    # the rest are not k-NN-propagated, since neighbor smoothing would change
    # the meaning of tangram_cell_type / tangram_margin.
    if adata_sp_sub.n_obs < adata_sp_full.n_obs:
        typer.echo(
            f"\nWARNING: --max-cells subsampled "
            f"{adata_sp_sub.n_obs:,}/{adata_sp_full.n_obs:,} cells. "
            f"Only the subset will have tangram_cell_type. "
            f"Set --max-cells >= {adata_sp_full.n_obs} to label every cell."
        )
    output_path = TANGRAM_DIR / f"{stem}_tangram.h5ad"
    adata_sp_sub.write_h5ad(output_path)
    typer.echo(f"Saved ({adata_sp_sub.n_obs:,} cells): {output_path}")
    typer.echo(f"\nDone! QC figures in: {RESULTS_QC}")

    # Raised last, so the (expensive) outputs exist for inspection but the job
    # exits non-zero and nothing downstream can quietly consume bad labels.
    # `uns['column_identity_check']` in the h5ad records the same verdict.
    if not ident_ok:
        raise RuntimeError(
            f"Posterior column identity check FAILED for '{stem}': "
            f"{ident_summary}. Outputs were written for inspection but the "
            f"labels in {output_path.name} are NOT trustworthy. "
            f"See {ident_path.name} and "
            f"scripts/identity/prior_distortion_diagnostic.py --stage other."
        )


# =============================================================================
# QC COMMAND
# =============================================================================


@app.command()
def qc(
    input_file: str = typer.Option("combined_clustered", help="Base input filename"),
    out_tag: str = typer.Option("", help="Suffix matching the train run to QC"),
):
    """Recompute QC on existing Tangram results."""
    import seaborn as sns

    _set_pub_style()
    RESULTS_QC.mkdir(parents=True, exist_ok=True)
    stem = f"{input_file}{('_' + out_tag) if out_tag else ''}"

    tangram_path = TANGRAM_DIR / f"{stem}_tangram.h5ad"

    if not tangram_path.exists():
        typer.echo(
            f"ERROR: {tangram_path} not found. Run: python tangram_mapping.py train"
        )
        raise typer.Exit(1)

    typer.echo(f"Loading Tangram results: {tangram_path.name}")
    adata = sc.read_h5ad(tangram_path)
    typer.echo(f"  Cells: {adata.n_obs:,}")
    typer.echo(f"  Leiden clusters: {adata.obs['leiden'].nunique()}")
    typer.echo(f"  Tangram cell types: {adata.obs['tangram_cell_type'].nunique()}")

    umap = adata.obsm["X_umap"]
    pt, alpha = 0.1, 0.2

    # --- Plot 1: UMAP overview ---
    typer.echo("\n[1/3] UMAP overview...")
    ct_cats = sorted(adata.obs["tangram_cell_type"].unique())
    ct_cmap = plt.colormaps.get_cmap("gist_ncar").resampled(len(ct_cats))
    ct_color_map = {ct: ct_cmap(i) for i, ct in enumerate(ct_cats)}

    lei_cats = sorted(
        adata.obs["leiden"].unique(), key=lambda x: int(x) if str(x).isdigit() else x
    )
    lei_cmap = (
        plt.colormaps.get_cmap("tab20").resampled(len(lei_cats))
        if len(lei_cats) <= 20
        else plt.colormaps.get_cmap("gist_ncar").resampled(len(lei_cats))
    )

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
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
    ax.legend(markerscale=10, fontsize=8)

    ax = axes[0, 1]
    for cl in lei_cats:
        mask = adata.obs["leiden"].values == cl
        ax.scatter(
            umap[mask, 0],
            umap[mask, 1],
            s=pt,
            alpha=alpha,
            c=[lei_cmap(lei_cats.index(cl))],
            rasterized=True,
        )
    ax.set_title(f"Spatial Leiden Clusters (n={len(lei_cats)})")

    ax = axes[1, 0]
    for ct in ct_cats:
        mask = adata.obs["tangram_cell_type"].values == ct
        ax.scatter(
            umap[mask, 0],
            umap[mask, 1],
            s=pt,
            alpha=alpha,
            c=[ct_color_map[ct]],
            rasterized=True,
        )
    ax.set_title(f"Tangram Cell Types (n={len(ct_cats)})")

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
    ax.set_title(f"Confidence (median={adata.obs['tangram_confidence'].median():.2f})")

    for ax in axes.flat:
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
    plt.tight_layout()
    fig.savefig(RESULTS_QC / "tangram_umap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    typer.echo("  Saved: tangram_umap.png")

    # --- Plot 2a: Per-celltype label-transfer confidence ---
    if {"tangram_confidence", "tangram_margin"}.issubset(adata.obs.columns):
        typer.echo("[2a/4] Per-celltype confidence + margin...")
        ct_order = (
            adata.obs.groupby("tangram_cell_type", observed=True)["tangram_confidence"]
            .median()
            .sort_values(ascending=False)
            .index.tolist()
        )
        df_qc = adata.obs[
            [
                "tangram_cell_type",
                "tangram_confidence",
                "tangram_margin",
                "tangram_label_qc",
            ]
        ].copy()
        df_qc["tangram_cell_type"] = pd.Categorical(
            df_qc["tangram_cell_type"], categories=ct_order, ordered=True
        )

        n_ct = len(ct_order)
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(max(14, 0.5 * n_ct + 8), 5),
            gridspec_kw={"width_ratios": [3, 3, 2]},
        )

        # Top1 distribution per type
        ax = axes[0]
        sns.boxplot(
            data=df_qc,
            x="tangram_cell_type",
            y="tangram_confidence",
            order=ct_order,
            ax=ax,
            showfliers=False,
            color="#3C5488",
        )
        ax.axhline(
            LABEL_QC_TOP1_MIN,
            color="red",
            linestyle="--",
            linewidth=0.8,
            label=f"min={LABEL_QC_TOP1_MIN}",
        )
        ax.axhline(
            1.0 / n_ct if n_ct else 0,
            color="grey",
            linestyle=":",
            linewidth=0.8,
            label=f"uniform=1/{n_ct}",
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("Top1 (tangram_confidence)")
        ax.set_title("Per-celltype top1 distribution")
        ax.legend(fontsize=7)

        # Margin distribution per type
        ax = axes[1]
        sns.boxplot(
            data=df_qc,
            x="tangram_cell_type",
            y="tangram_margin",
            order=ct_order,
            ax=ax,
            showfliers=False,
            color="#E64B35",
        )
        ax.axhline(
            LABEL_QC_MARGIN_MIN,
            color="red",
            linestyle="--",
            linewidth=0.8,
            label=f"min={LABEL_QC_MARGIN_MIN}",
        )
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("")
        ax.set_ylabel("Margin (top1 − top2)")
        ax.set_title("Per-celltype margin distribution")
        ax.legend(fontsize=7)

        # Confident fraction per type
        ax = axes[2]
        conf_frac = (
            df_qc.groupby("tangram_cell_type", observed=True)["tangram_label_qc"]
            .apply(lambda s: (s == "confident").mean())
            .reindex(ct_order)
        )
        ax.barh(range(len(conf_frac)), conf_frac.values, color="#4DAF4A")
        ax.set_yticks(range(len(conf_frac)))
        ax.set_yticklabels(conf_frac.index, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Fraction confident")
        ax.set_title(
            f"Confident: top1≥{LABEL_QC_TOP1_MIN} & margin≥{LABEL_QC_MARGIN_MIN}"
        )

        plt.tight_layout()
        fig.savefig(
            RESULTS_QC / "tangram_label_confidence.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        typer.echo(
            f"  Saved: tangram_label_confidence.png "
            f"(overall confident: "
            f"{100 * (df_qc['tangram_label_qc'] == 'confident').mean():.1f}%)"
        )

    # --- Plot 2: Cluster mapping heatmap ---
    typer.echo("[2/4] Cluster mapping heatmap...")
    ct_tab = pd.crosstab(adata.obs["tangram_cell_type"], adata.obs["leiden"])
    ct_by_leiden = ct_tab.div(ct_tab.sum(axis=0), axis=1)
    ct_by_leiden = ct_by_leiden[
        sorted(ct_by_leiden.columns, key=lambda x: int(x) if str(x).isdigit() else x)
    ]
    ct_by_leiden = ct_by_leiden.reindex(
        adata.obs["tangram_cell_type"].value_counts().index
    )

    fig, ax = plt.subplots(
        figsize=(max(10, len(lei_cats) * 0.6), max(10, len(ct_cats) * 0.3))
    )
    sns.heatmap(
        ct_by_leiden,
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"label": "Proportion within Leiden cluster"},
    )
    ax.set_xlabel("Spatial Leiden Cluster")
    ax.set_ylabel("Tangram Cell Type")
    ax.set_title("Cell type composition per spatial cluster")
    plt.tight_layout()
    fig.savefig(
        RESULTS_QC / "tangram_cluster_mapping.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)
    typer.echo("  Saved: tangram_cluster_mapping.png")

    # --- Plot 3: Gene-level QC ---
    scores_path = TANGRAM_DIR / f"{stem}_gene_scores.csv"
    if scores_path.exists():
        typer.echo("[3/4] Gene scores (from tg.compare_spatial_geneexp)...")
        df_scores = pd.read_csv(scores_path, index_col=0)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # Score distribution
        ax = axes[0]
        train_s = df_scores[df_scores["is_training"]]["score"]
        test_s = df_scores[~df_scores["is_training"]]["score"]
        ax.hist(
            train_s,
            bins=20,
            alpha=0.6,
            color="#3C5488",
            label=f"Train (med={train_s.median():.2f})",
            edgecolor="white",
        )
        ax.hist(
            test_s,
            bins=20,
            alpha=0.6,
            color="#E64B35",
            label=f"Test (med={test_s.median():.2f})",
            edgecolor="white",
        )
        ax.set_xlabel("Cosine Similarity")
        ax.set_ylabel("Count")
        ax.set_title("Gene-level Cosine Similarity")
        ax.legend(fontsize=8)

        # Score vs sparsity
        ax = axes[1]
        colors = ["#3C5488" if t else "#E64B35" for t in df_scores["is_training"]]
        ax.scatter(
            df_scores.get("sparsity_sp", df_scores.index),
            df_scores["score"],
            c=colors,
            s=40,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )
        ax.set_xlabel("Spatial Sparsity")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Score vs Sparsity (key diagnostic)")

        # Score vs sparsity diff
        ax = axes[2]
        if "sparsity_diff" in df_scores.columns:
            ax.scatter(
                df_scores["sparsity_diff"],
                df_scores["score"],
                c=colors,
                s=40,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
            )
            ax.axvline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.3)
            ax.set_xlabel("Sparsity Diff (spatial − reference)")
        ax.set_ylabel("Cosine Similarity")
        ax.set_title("Score vs Sparsity Mismatch")

        plt.tight_layout()
        fig.savefig(
            RESULTS_QC / "tangram_gene_scores.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        typer.echo("  Saved: tangram_gene_scores.png")

        # Summary metrics (avg scores, AUC, sparsity-weighted score)
        import tangram2.mapping as tg

        try:
            eval_results, _ = tg.eval_metric(df_scores)
            metrics_path = TANGRAM_DIR / f"{stem}_eval_metrics.json"
            with open(metrics_path, "w") as mf:
                json.dump({k: float(v) for k, v in eval_results.items()}, mf, indent=2)
            typer.echo(f"  eval_metric: {eval_results}")
        except Exception as e:
            typer.echo(f"  WARNING: tg.eval_metric failed: {e}")
    else:
        typer.echo("[3/4] No gene_scores.csv found — skipping")

    typer.echo(f"\nDone! Figures saved to: {RESULTS_QC}")


# =============================================================================
# VALIDATE: density comparison of held-out test genes (probe vs imputed)
# =============================================================================


@app.command("validate-test-genes")
def validate_test_genes(
    input_file: str = typer.Option("combined_clustered", help="Base input filename"),
    out_tag: str = typer.Option("", help="Suffix matching the train run to validate"),
):
    """Validate Tangram: probe vs imputed spatial expression.

    Shows 3 training genes (reference for what "good" looks like), then 5
    held-out test genes.
    focal_genes: log1p expression, 95th percentile vmax from probe applied
    to both columns so colors are directly comparable.
    """
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.sparse import issparse
    from scipy.stats import spearmanr

    _set_pub_style()
    RESULTS_QC.mkdir(parents=True, exist_ok=True)
    stem = f"{input_file}{('_' + out_tag) if out_tag else ''}"

    # Load gene split + scores
    gene_split_path = TANGRAM_DIR / f"{stem}_gene_split.json"
    scores_path = TANGRAM_DIR / f"{stem}_gene_scores.csv"
    if not gene_split_path.exists():
        typer.echo(f"ERROR: {gene_split_path} not found. Run train first.")
        raise typer.Exit(1)
    with open(gene_split_path) as f:
        gene_split = json.load(f)
    test_genes = gene_split["test_genes"]

    # Pick 3 training genes spanning sparsity (≥10% detection to be useful)
    train_picks = []
    if scores_path.exists():
        df_scores = pd.read_csv(scores_path, index_col=0)
        train_df = df_scores[df_scores["is_training"]].sort_values(
            "score", ascending=False
        )
        train_df = train_df[train_df["sparsity_sp"] < 0.9]
        for lo, hi in [(0.5, 0.9), (0.2, 0.5), (0.0, 0.2)]:
            subset = train_df[
                (train_df["sparsity_sp"] >= lo) & (train_df["sparsity_sp"] < hi)
            ]
            if len(subset):
                train_picks.append(subset.index[0])
        typer.echo(f"Training reference genes: {train_picks}")

    # Load gene descriptions (both upper and lower keys)
    gene_descs = {}
    if GENE_MAPPING_FILE.exists():
        df_map = pd.read_csv(GENE_MAPPING_FILE)
        for _, row in df_map.iterrows():
            v43 = str(row.get("v43_id", "")).strip()
            desc = str(row.get("description", "")).strip()
            if v43 and desc and desc != "nan":
                if "[protein=" in desc:
                    prot = desc.split("[protein=")[1].split("]")[0]
                    name = prot[:30] + ("..." if len(prot) > 30 else "")
                else:
                    name = desc[:30]
                gene_descs[v43] = name
                gene_descs[v43.lower()] = name

    # Load data
    tangram_path = TANGRAM_DIR / f"{stem}_tangram.h5ad"
    imputed_path = TANGRAM_DIR / f"{stem}_imputed.h5ad"
    for p in (tangram_path, imputed_path):
        if not p.exists():
            typer.echo(f"ERROR: {p} not found. Run train first.")
            raise typer.Exit(1)

    typer.echo(f"Loading probe data: {tangram_path.name}")
    adata_probe = sc.read_h5ad(tangram_path)
    typer.echo(f"Loading imputed data: {imputed_path.name}")
    ad_ge = sc.read_h5ad(imputed_path)
    typer.echo(f"  Probe: {adata_probe.n_obs:,} × {adata_probe.n_vars}")
    typer.echo(f"  Imputed: {ad_ge.n_obs:,} × {ad_ge.n_vars}")

    sp = adata_probe.obsm.get("spatial_aligned", adata_probe.obsm.get("spatial"))
    # Normalize gene names to lowercase for matching (probe may use BIMP_v4.3_*, imputed
    # bimp_v4.3_*)
    probe_name_map = {g.lower(): g for g in adata_probe.var_names}  # low → original
    probe_genes_set = set(probe_name_map.keys())
    imputed_genes_set = set(ad_ge.var_names)  # already lowercase

    # Build gene list: training refs first, then test
    gene_entries = []  # (gene_low, label, group)
    for g_low in train_picks:
        if g_low in probe_genes_set and g_low in imputed_genes_set:
            desc = gene_descs.get(g_low, g_low)
            gene_entries.append((g_low, desc, "train"))
    for g in test_genes:
        g_low = g.lower()
        if g_low in probe_genes_set and g_low in imputed_genes_set:
            desc = gene_descs.get(g, gene_descs.get(g_low, g_low))
            gene_entries.append((g_low, desc, "test"))

    if not gene_entries:
        typer.echo("ERROR: No genes found in both datasets")
        raise typer.Exit(1)

    n_train = sum(1 for _, _, g in gene_entries if g == "train")
    n_test = sum(1 for _, _, g in gene_entries if g == "test")
    typer.echo(f"Plotting {n_train} training + {n_test} test genes")

    # Colormap: EXPR_CMAP (white background)
    expr_cmap = LinearSegmentedColormap.from_list(
        "expr",
        [
            "#F5F5F5",
            "#FEE08B",
            "#F46D43",
            "#D53E4F",
            "#9E0142",
        ],
        N=256,
    )

    n_rows = len(gene_entries)
    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, (gene_low, gene_label, group) in enumerate(gene_entries):
        # ── Extract expression as dense arrays ──
        probe_gene = probe_name_map[gene_low]  # original case for indexing
        probe_X = adata_probe[:, probe_gene].X
        probe_raw = (
            probe_X.toarray().flatten()
            if issparse(probe_X)
            else np.asarray(probe_X).flatten()
        )
        imp_X = ad_ge[:, gene_low].X
        imp_raw = (
            imp_X.toarray().flatten()
            if issparse(imp_X)
            else np.asarray(imp_X).flatten()
        )

        frac_expr = (probe_raw > 0).mean()

        # ── Truncate to [p5, p95] of non-zero values, min-max to [0, 1] ──
        # This stretches even tiny imputed differences to fill the colormap,
        # making spatial patterns visible regardless of absolute scale.
        def _trunc_norm(v):
            nz = v[v > 0]
            if len(nz) < 20:
                return np.zeros_like(v, dtype=float)
            lo, hi = np.percentile(nz, [5, 95])
            if hi <= lo:
                hi = lo + 1e-10
            out = np.clip(v, lo, hi)
            out = (out - lo) / (hi - lo)
            # Cells with original value 0 stay at 0 (white in EXPR_CMAP)
            out[v == 0] = 0.0
            return out

        probe_n = _trunc_norm(probe_raw)
        imp_n = _trunc_norm(imp_raw)

        # Spearman correlation on raw values
        valid = (probe_raw > 0) | (imp_raw > 0)
        if valid.sum() > 10:
            rho, pval = spearmanr(probe_raw[valid], imp_raw[valid])
        else:
            rho, pval = 0.0, 1.0

        # ── Adaptive point size ──
        if frac_expr < 0.15:
            _s, _a = 1.5, 0.5
        else:
            _s, _a = 0.5, 0.3

        # Row label
        tag = "TRAIN" if group == "train" else "TEST"
        sig = (
            "***"
            if pval < 0.001
            else "**"
            if pval < 0.01
            else "*"
            if pval < 0.05
            else "ns"
        )
        ylabel = f"[{tag}] {gene_label}\n{frac_expr:.0%} det | ρ={rho:.2f}{sig}"

        # ── Col 0: Probe spatial (truncated + normalized) ──
        ax = axes[row, 0]
        order = np.argsort(probe_n)
        ax.scatter(
            sp[order, 0],
            sp[order, 1],
            c=probe_n[order],
            cmap=expr_cmap,
            s=_s,
            alpha=_a,
            edgecolors="none",
            rasterized=True,
            vmin=0,
            vmax=1,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row == 0:
            ax.set_title("Probe (real)", fontsize=13, fontweight="bold")
        ax.set_ylabel(
            ylabel, fontsize=8, color="#1565C0" if group == "train" else "#333333"
        )

        # ── Col 1: Imputed spatial (truncated + normalized) ──
        ax = axes[row, 1]
        order = np.argsort(imp_n)
        ax.scatter(
            sp[order, 0],
            sp[order, 1],
            c=imp_n[order],
            cmap=expr_cmap,
            s=_s,
            alpha=_a,
            edgecolors="none",
            rasterized=True,
            vmin=0,
            vmax=1,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if row == 0:
            ax.set_title("Tangram (imputed)", fontsize=13, fontweight="bold")

        # Separator line between training and test sections
        if group == "train" and row == n_train - 1 and n_test > 0:
            for c in range(2):
                axes[row, c].annotate(
                    "",
                    xy=(0, -0.06),
                    xycoords="axes fraction",
                    xytext=(1, -0.06),
                    textcoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="-", color="gray", lw=1.5, linestyle="--"
                    ),
                )

        typer.echo(f"  [{tag}] {gene_label}: ρ={rho:.3f}, {frac_expr:.1%} detected")

    plt.suptitle(
        "Tangram validation: Probe vs Imputed (5–95th pctl, normalized)\n"
        f"Top {n_train} training genes │ {n_test} held-out test genes",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = RESULTS_QC / "tangram_test_validation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    typer.echo(f"\nSaved: {out}")
    typer.echo(f"Done! Validation figure in: {RESULTS_QC}")


# =============================================================================
# LEAVE-ONE-OUT CROSS-VALIDATION
# =============================================================================


@app.command()
def cross_validate(
    input_file: str = typer.Option("combined_clustered", help="Input clustered h5ad"),
    max_cells: int = typer.Option(100000, help="Max spatial cells"),
    max_ref_cells: int = typer.Option(15000, help="Max reference cells (stratified)"),
    device: str = typer.Option("cuda", help="Device"),
    num_epochs: int = typer.Option(500, help="Epochs per fold"),
    density_prior: str = typer.Option("rna_count_based", help="Density prior"),
    seed: int = typer.Option(42, help="Random seed"),
    out_dir: str = typer.Option(
        "",
        help="Directory for the scores CSV, run metadata and plot. Default "
        "(empty) keeps the historical paths, results/tangram/"
        "<input>_loocv_scores.csv and results/tangram_qc/tangram_loocv.png, "
        "which hold the shipped run: pass a new directory for any re-run.",
    ),
):
    """Leave-one-out cross-validation: train N times, each holding out 1 gene.

    Produces a per-gene cosine similarity score computed when that gene was
    held out from training.  This is the gold-standard Tangram benchmark
    (Biancalani et al. 2021).  Expect ~2 min per gene on GPU (~3.5 h total).
    """
    import tangram2.mapping as tg
    from scipy.sparse import issparse
    from tqdm import tqdm

    random.seed(seed)
    np.random.seed(seed)

    _set_pub_style()
    if out_dir:
        cv_dir = plot_dir = Path(out_dir)
        if (cv_dir / f"{input_file}_loocv_scores.csv").exists():
            raise typer.BadParameter(
                f"{cv_dir} already holds a LOO-CV run; pick a new --out-dir"
            )
    else:
        cv_dir, plot_dir = TANGRAM_DIR, RESULTS_QC
    cv_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    spatial_path = DATA_DIR / f"{input_file}.h5ad"
    typer.echo(f"Loading spatial: {spatial_path.name}")
    adata_sp_full = sc.read_h5ad(spatial_path)

    typer.echo(f"Loading reference: {SNRNA_REF.name}")
    adata_ref = sc.read_h5ad(SNRNA_REF)

    # ── Gene mapping (vNP3.1 → v4.3) ──
    gene_mapping = load_gene_mapping()
    adata_sp_full = map_spatial_genes(adata_sp_full, gene_mapping)
    common_genes = sorted(set(adata_sp_full.var_names) & set(adata_ref.var_names))
    typer.echo(f"  Common genes: {len(common_genes)}")

    # ── Subsample spatial ──
    if adata_sp_full.n_obs > max_cells:
        idx = np.random.choice(adata_sp_full.n_obs, max_cells, replace=False)
        adata_sp = adata_sp_full[idx].copy()
    else:
        adata_sp = adata_sp_full.copy()

    # ── Subsample reference (stratified) ──
    if max_ref_cells > 0 and adata_ref.n_obs > max_ref_cells:
        if "cell_type" in adata_ref.obs.columns:
            clusters = adata_ref.obs["cell_type"].values
            unique_cl = np.unique(clusters)
            per_cl = max(1, max_ref_cells // len(unique_cl))
            keep = []
            for cl in unique_cl:
                cl_idx = np.where(clusters == cl)[0]
                n = min(per_cl, len(cl_idx))
                keep.extend(np.random.choice(cl_idx, n, replace=False))
            adata_ref = adata_ref[keep].copy()
        else:
            idx = np.random.choice(adata_ref.n_obs, max_ref_cells, replace=False)
            adata_ref = adata_ref[idx].copy()
        typer.echo(f"  Reference subsampled to {adata_ref.n_obs:,} cells")

    # ── Prepare data (linear scale) ──
    adata_ref_sub = adata_ref[:, common_genes].copy()
    if "counts" in adata_ref_sub.layers:
        adata_ref_sub.X = adata_ref_sub.layers["counts"].copy()
        sc.pp.normalize_total(adata_ref_sub)
    else:
        adata_ref_sub.X = np.expm1(
            adata_ref_sub.X.toarray() if issparse(adata_ref_sub.X) else adata_ref_sub.X
        )
        sc.pp.normalize_total(adata_ref_sub)

    adata_sp_sub = adata_sp[:, common_genes].copy()
    adata_sp_sub.X = np.expm1(
        adata_sp_sub.X.toarray() if issparse(adata_sp_sub.X) else adata_sp_sub.X
    ).astype(np.float32)

    typer.echo(
        f"  Spatial: {adata_sp_sub.n_obs:,} cells, "
        f"Reference: {adata_ref_sub.n_obs:,} cells"
    )
    typer.echo(f"  LOO-CV over {len(common_genes)} genes ({num_epochs} epochs each)")
    typer.echo(f"  Estimated time: ~{len(common_genes) * 2} min on GPU\n")

    # ── LOO-CV loop ──
    # Work on lowercase copies to avoid tangram's in-place lowercasing
    # tg.pp_adatas and tg.project_genes both lowercase var_names in-place,
    # which corrupts the originals across iterations.
    ref_lc = adata_ref_sub.copy()
    sp_lc = adata_sp_sub.copy()
    genes_lc = [g.lower() for g in common_genes]

    results = []
    for i, held_out in enumerate(tqdm(genes_lc, desc="LOO-CV")):
        train_genes = [g for g in genes_lc if g != held_out]

        # pp_adatas sets uns keys; it lowercases in-place but we already did
        tg.pp_adatas(ref_lc, sp_lc, genes=train_genes)

        ad_map = tg.map_cells_to_space(
            adata_sc=ref_lc,
            adata_sp=sp_lc,
            device=device,
            num_epochs=num_epochs,
            density_prior=density_prior,
            random_state=seed,
            verbose=False,
        )

        # Project the held-out gene (manual: mapping.T @ ref.X avoids project_genes
        # mutating var_names)
        ref_gene_X = ref_lc[:, held_out].X
        ref_vec = (
            ref_gene_X.toarray() if issparse(ref_gene_X) else np.asarray(ref_gene_X)
        )
        proj = (ad_map.X.T @ ref_vec).flatten()

        # Observed spatial expression
        obs_X = sp_lc[:, held_out].X
        obs = (
            obs_X.toarray().flatten()
            if issparse(obs_X)
            else np.asarray(obs_X).flatten()
        )

        norm_p = np.linalg.norm(proj)
        norm_o = np.linalg.norm(obs)
        if norm_p > 0 and norm_o > 0:
            cosine = float(np.dot(proj, obs) / (norm_p * norm_o))
        else:
            cosine = 0.0

        sparsity_sp = float((obs == 0).mean())
        ref_vec_flat = ref_vec.flatten()
        sparsity_sc = float((ref_vec_flat == 0).mean())

        # tangram2 main may store None entries in the loss history — take the
        # last real value (or NaN) rather than crashing on float(None).
        _losses = [
            x
            for x in ad_map.uns.get("training_history", {}).get("main_loss", [])
            if x is not None
        ]
        train_loss = float(_losses[-1]) if _losses else float("nan")

        results.append(
            {
                "gene": held_out,
                "cosine_sim": float(cosine),
                "sparsity_sp": float(sparsity_sp),
                "sparsity_sc": float(sparsity_sc),
                "sparsity_diff": float(sparsity_sp - sparsity_sc),
                "train_loss": train_loss,
            }
        )

        if (i + 1) % 10 == 0:
            avg_so_far = np.mean([r["cosine_sim"] for r in results])
            typer.echo(f"  [{i + 1}/{len(genes_lc)}] avg cosine: {avg_so_far:.3f}")

    # ── Save results ──
    df_cv = pd.DataFrame(results).set_index("gene")
    cv_path = cv_dir / f"{input_file}_loocv_scores.csv"
    df_cv.to_csv(cv_path)

    avg_score = df_cv["cosine_sim"].mean()
    med_score = df_cv["cosine_sim"].median()
    typer.echo(f"\nLOO-CV complete: mean={avg_score:.3f}, median={med_score:.3f}")
    typer.echo(f"  Saved: {cv_path}")

    # Provenance: record the tangram2 commit alongside the scores.
    import subprocess

    try:
        tg_commit = subprocess.run(
            [
                "git",
                "-C",
                str(Path(tg.__file__).resolve().parents[2]),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        tg_commit = "unknown"
    meta = {
        "input_file": input_file,
        "spatial": str(spatial_path),
        "reference": str(SNRNA_REF),
        "max_cells": max_cells,
        "max_ref_cells": max_ref_cells,
        "n_spatial_cells": int(adata_sp_sub.n_obs),
        "n_ref_cells": int(adata_ref_sub.n_obs),
        "n_genes": int(len(common_genes)),
        "num_epochs": num_epochs,
        "density_prior": density_prior,
        "mode": "cells",
        "seed": seed,
        "device": device,
        "tangram2_commit": tg_commit,
        "median": float(med_score),
        "mean": float(avg_score),
    }
    (cv_dir / f"{input_file}_loocv_run.json").write_text(json.dumps(meta, indent=2))

    # ── Plot ──
    _plot_loocv(df_cv, plot_dir)
    typer.echo(f"  Saved LOO-CV plots to {plot_dir}")


def _plot_loocv(df_cv, out_dir):
    """Plot LOO-CV results: score distribution, score vs sparsity, ranked genes."""
    from sklearn.metrics import auc as sk_auc

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Score distribution
    ax = axes[0]
    ax.hist(df_cv["cosine_sim"], bins=20, color="#3C5488", alpha=0.7, edgecolor="white")
    ax.axvline(
        df_cv["cosine_sim"].median(),
        color="#E64B35",
        linestyle="--",
        linewidth=2,
        label=f"median={df_cv['cosine_sim'].median():.3f}",
    )
    ax.axvline(
        df_cv["cosine_sim"].mean(),
        color="#F39B7F",
        linestyle=":",
        linewidth=2,
        label=f"mean={df_cv['cosine_sim'].mean():.3f}",
    )
    ax.set_xlabel("Cosine Similarity (held-out)")
    ax.set_ylabel("Count")
    ax.set_title("LOO-CV Score Distribution")
    ax.legend(fontsize=9)

    # Panel 2: Score vs spatial sparsity
    ax = axes[1]
    ax.scatter(
        df_cv["sparsity_sp"],
        df_cv["cosine_sim"],
        c="#3C5488",
        s=50,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.5,
    )
    ax.set_xlabel("Spatial Sparsity")
    ax.set_ylabel("Cosine Similarity (held-out)")
    ax.set_title("Score vs Sparsity")
    # Fit + AUC
    if len(df_cv) > 3:
        sorted_df = df_cv.sort_values("cosine_sim")
        xs = sorted_df["cosine_sim"].values
        ys = sorted_df["sparsity_sp"].values
        coeffs = np.polyfit(xs, ys, deg=min(3, len(xs) - 1))
        x_smooth = np.linspace(0, 1, 100)
        y_smooth = np.clip(np.polyval(coeffs, x_smooth), 0, 1)
        auc_val = sk_auc(x_smooth, y_smooth)
        ax.text(
            0.03,
            0.1,
            f"AUC={auc_val:.3f}",
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

    # Panel 3: Ranked gene scores (bar plot)
    ax = axes[2]
    df_sorted = df_cv.sort_values("cosine_sim", ascending=True)
    colors = ["#E64B35" if s < 0.5 else "#3C5488" for s in df_sorted["cosine_sim"]]
    ax.barh(range(len(df_sorted)), df_sorted["cosine_sim"], color=colors, alpha=0.8)
    ax.set_xlabel("Cosine Similarity (held-out)")
    ax.set_ylabel("Gene rank")
    ax.set_title(f"Per-Gene LOO-CV Scores (n={len(df_cv)})")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    plt.suptitle(
        f"Leave-One-Out Cross-Validation (n={len(df_cv)} genes)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    out = out_dir / "tangram_loocv.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    app()
