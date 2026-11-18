"""
Compositional Data Analysis (CoDA) for Spatial Transcriptomics

Aitchison-style analysis of cell type compositions across brain regions.
Designed for within-individual analysis with extensibility for cross-individual
comparison.

Key concepts:
- Cell type proportions are compositional (sum to 1, carry relative information)
- Uses log-ratio transformations (CLR, ALR, ILR) to handle the simplex constraint
- Aitchison distance for proper compositional geometry
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import typer
from scipy.spatial.distance import pdist, squareform

app = typer.Typer()

# ============================================================================
# Compositional Data Transformations (Aitchison Geometry)
# ============================================================================


def closure(X: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Close compositions to sum to total (default 1)."""
    X = np.atleast_2d(X)
    return X / X.sum(axis=1, keepdims=True) * total


def multiplicative_replacement(
    X: np.ndarray, delta: Optional[float] = None
) -> np.ndarray:
    """
    Replace zeros with small values using multiplicative replacement.
    This preserves the ratios between non-zero components.
    """
    X = np.atleast_2d(X).astype(float)
    if delta is None:
        # Use 65% of minimum non-zero value (Martín-Fernández et al. 2003)
        delta = 0.65 * X[X > 0].min()

    X_replaced = X.copy()
    for i in range(X.shape[0]):
        row = X[i]
        n_zeros = (row == 0).sum()
        if n_zeros > 0:
            # Add delta to zeros
            X_replaced[i, row == 0] = delta
            # Adjust non-zeros to maintain closure
            adjustment = 1 - n_zeros * delta / row.sum()
            X_replaced[i, row > 0] = row[row > 0] * adjustment

    return X_replaced


def clr(X: np.ndarray) -> np.ndarray:
    """
    Centered Log-Ratio transformation.
    CLR(x)_i = log(x_i / g(x)) where g(x) is the geometric mean.
    """
    X = np.atleast_2d(X)
    # Handle zeros
    if (X <= 0).any():
        X = multiplicative_replacement(X)

    log_X = np.log(X)
    geometric_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geometric_mean


def alr(X: np.ndarray, ref_idx: int = -1) -> np.ndarray:
    """
    Additive Log-Ratio transformation.
    ALR(x)_i = log(x_i / x_ref) for i != ref.
    """
    X = np.atleast_2d(X)
    if (X <= 0).any():
        X = multiplicative_replacement(X)

    ref = X[:, ref_idx : ref_idx + 1] if ref_idx != -1 else X[:, -1:]
    indices = [i for i in range(X.shape[1]) if i != (ref_idx % X.shape[1])]
    return np.log(X[:, indices] / ref)


def ilr(X: np.ndarray) -> np.ndarray:
    """
    Isometric Log-Ratio transformation.
    Projects onto orthonormal basis in the simplex.
    """
    X = np.atleast_2d(X)
    if (X <= 0).any():
        X = multiplicative_replacement(X)

    D = X.shape[1]
    # Helmert submatrix basis
    ilr_coords = np.zeros((X.shape[0], D - 1))
    for i in range(D - 1):
        ilr_coords[:, i] = np.sqrt((D - i - 1) / (D - i)) * np.log(
            X[:, i] / np.power(np.prod(X[:, i + 1 :], axis=1), 1 / (D - i - 1))
        )
    return ilr_coords


def aitchison_distance(X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Aitchison distance between compositions.
    d_A(x, y) = ||clr(x) - clr(y)||
    """
    clr_X = clr(X)
    if Y is None:
        # Pairwise distances
        return squareform(pdist(clr_X, metric="euclidean"))
    else:
        clr_Y = clr(Y)
        return np.sqrt(((clr_X - clr_Y) ** 2).sum(axis=1))


# ============================================================================
# Cell Type Composition Extraction
# ============================================================================


def get_region_compositions(
    adata: sc.AnnData,
    cluster_key: str,
    region_key: str = "regions",
    min_cells: int = 50,
) -> pd.DataFrame:
    """
    Compute cell type composition per region.

    Returns DataFrame with regions as rows, cell types as columns,
    values are proportions (sum to 1 per region).
    """
    # Filter to labeled cells
    mask = adata.obs[region_key] != "unknown"
    obs = adata.obs.loc[mask, [region_key, cluster_key]].copy()

    # Count cells per region-cluster combination
    counts = obs.groupby([region_key, cluster_key]).size().unstack(fill_value=0)

    # Filter regions with too few cells
    counts = counts[counts.sum(axis=1) >= min_cells]

    # Convert to proportions
    proportions = counts.div(counts.sum(axis=1), axis=0)

    return proportions


def get_slice_compositions(
    adata: sc.AnnData,
    cluster_key: str,
    slice_key: str = "slice_idx",
    min_cells: int = 100,
) -> pd.DataFrame:
    """
    Compute cell type composition per slice (z-layer).
    """
    obs = adata.obs[[slice_key, cluster_key]].copy()
    counts = obs.groupby([slice_key, cluster_key]).size().unstack(fill_value=0)
    counts = counts[counts.sum(axis=1) >= min_cells]
    proportions = counts.div(counts.sum(axis=1), axis=0)
    return proportions


def get_region_slice_compositions(
    adata: sc.AnnData,
    cluster_key: str,
    region_key: str = "regions",
    slice_key: str = "slice_idx",
    min_cells: int = 30,
) -> pd.DataFrame:
    """
    Compute cell type composition per region-slice combination.
    Useful for analyzing variation within regions across z.
    """
    mask = adata.obs[region_key] != "unknown"
    obs = adata.obs.loc[mask, [region_key, slice_key, cluster_key]].copy()
    obs["region_slice"] = obs[region_key] + "_z" + obs[slice_key].astype(str)

    counts = obs.groupby(["region_slice", cluster_key]).size().unstack(fill_value=0)
    counts = counts[counts.sum(axis=1) >= min_cells]
    proportions = counts.div(counts.sum(axis=1), axis=0)

    return proportions


# ============================================================================
# Statistical Analysis
# ============================================================================


def compositional_pca(proportions: pd.DataFrame, n_components: int = 2) -> dict:
    """
    PCA on CLR-transformed compositions.
    This is equivalent to PCA in Aitchison geometry.
    """
    from sklearn.decomposition import PCA

    clr_data = clr(proportions.values)
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(clr_data)

    return {
        "coords": pd.DataFrame(
            coords,
            index=proportions.index,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        ),
        "loadings": pd.DataFrame(
            pca.components_.T,
            index=proportions.columns,
            columns=[f"PC{i + 1}" for i in range(n_components)],
        ),
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "pca": pca,
    }


def compositional_variation(proportions: pd.DataFrame) -> pd.DataFrame:
    """
    Compute variation matrix (Aitchison 1986).
    var(i,j) = var(log(x_i/x_j))
    Measures how much the log-ratio between components varies.
    """
    X = multiplicative_replacement(proportions.values)
    n_components = X.shape[1]
    var_matrix = np.zeros((n_components, n_components))

    for i in range(n_components):
        for j in range(n_components):
            log_ratio = np.log(X[:, i] / X[:, j])
            var_matrix[i, j] = np.var(log_ratio)

    return pd.DataFrame(
        var_matrix, index=proportions.columns, columns=proportions.columns
    )


def total_variance(proportions: pd.DataFrame) -> float:
    """
    Total variance in Aitchison geometry.
    Sum of variances of CLR-transformed components / D.
    """
    clr_data = clr(proportions.values)
    return np.var(clr_data, axis=0).sum()


# ============================================================================
# Visualization
# ============================================================================

# Set up beautiful plot defaults
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Custom color palettes
CLUSTER_PALETTE = sns.color_palette("husl", 30)  # For many clusters
REGION_PALETTE = sns.color_palette("Set2", 10)  # For regions


def plot_clr_heatmap(
    proportions: pd.DataFrame,
    output_path: Path,
    title: str = "CLR-transformed Cell Type Composition",
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    figsize: tuple = (14, 10),
):
    """Plot heatmap of CLR-transformed compositions."""
    clr_data = pd.DataFrame(
        clr(proportions.values), index=proportions.index, columns=proportions.columns
    )

    g = sns.clustermap(
        clr_data,
        cmap="RdBu_r",
        center=0,
        figsize=figsize,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "CLR value", "shrink": 0.6},
        dendrogram_ratio=(0.1, 0.15),
    )
    g.fig.suptitle(title, y=1.02, fontweight="bold")
    g.ax_heatmap.set_xlabel("Leiden Cluster", fontweight="bold")
    g.ax_heatmap.set_ylabel("Brain Region", fontweight="bold")
    g.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_composition_pca(
    pca_result: dict,
    output_path: Path,
    color_by: Optional[pd.Series] = None,
    title: str = "Compositional PCA (Aitchison geometry)",
    figsize: tuple = (12, 10),
):
    """Plot PCA of compositions with loadings biplot."""
    fig, ax = plt.subplots(figsize=figsize)

    coords = pca_result["coords"]
    loadings = pca_result["loadings"]
    var_explained = pca_result["explained_variance_ratio"]

    # Plot samples with labels
    ax.scatter(
        coords["PC1"],
        coords["PC2"],
        c=range(len(coords)),
        cmap="viridis",
        s=200,
        alpha=0.8,
        edgecolors="white",
        linewidth=2,
        zorder=3,
    )

    # Add region labels
    for idx, row in coords.iterrows():
        ax.annotate(
            str(idx),
            (row["PC1"], row["PC2"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
            alpha=0.9,
        )

    # Plot loadings as arrows (cluster contributions)
    scale = max(coords["PC1"].abs().max(), coords["PC2"].abs().max()) * 0.7
    for idx, row in loadings.iterrows():
        ax.arrow(
            0,
            0,
            row["PC1"] * scale,
            row["PC2"] * scale,
            head_width=0.04 * scale,
            head_length=0.025 * scale,
            fc="#E74C3C",
            ec="#C0392B",
            alpha=0.7,
            linewidth=1.5,
            zorder=2,
        )
        ax.text(
            row["PC1"] * scale * 1.15,
            row["PC2"] * scale * 1.15,
            f"C{idx}",
            fontsize=8,
            color="#E74C3C",
            ha="center",
            fontweight="bold",
        )

    ax.set_xlabel(f"PC1 ({var_explained[0] * 100:.1f}% variance)", fontweight="bold")
    ax.set_ylabel(f"PC2 ({var_explained[1] * 100:.1f}% variance)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3, zorder=1)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3, zorder=1)
    ax.set_facecolor("#FAFAFA")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_variation_matrix(
    proportions: pd.DataFrame,
    output_path: Path,
    title: str = "Log-ratio Variation Matrix",
    figsize: tuple = (12, 10),
):
    """Plot the variation matrix heatmap."""
    var_matrix = compositional_variation(proportions)

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.triu(np.ones_like(var_matrix, dtype=bool), k=1)

    sns.heatmap(
        var_matrix,
        mask=mask,
        cmap="YlOrRd",
        annot=True if var_matrix.shape[0] <= 15 else False,
        fmt=".2f",
        square=True,
        ax=ax,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Variance of log-ratio", "shrink": 0.7},
    )
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlabel("Leiden Cluster", fontweight="bold")
    ax.set_ylabel("Leiden Cluster", fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_region_compositions(
    proportions: pd.DataFrame,
    output_path: Path,
    title: str = "Cell Type Composition by Region",
    figsize: tuple = (16, 8),
):
    """Stacked bar plot of compositions."""
    fig, ax = plt.subplots(figsize=figsize)

    # Use a nice color palette
    n_clusters = len(proportions.columns)
    colors = sns.color_palette("husl", n_clusters)

    proportions.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors,
        width=0.85,
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_ylabel("Proportion", fontweight="bold")
    ax.set_xlabel("Brain Region", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_ylim(0, 1)

    # Better legend
    ax.legend(
        title="Leiden\nCluster",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
        fancybox=True,
        shadow=True,
        ncol=1 if n_clusters <= 15 else 2,
    )
    plt.xticks(rotation=45, ha="right")

    # Add gridlines
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_aitchison_distance_heatmap(
    proportions: pd.DataFrame,
    output_path: Path,
    title: str = "Aitchison Distance Between Regions",
    figsize: tuple = (12, 10),
):
    """Plot heatmap of pairwise Aitchison distances."""
    dist_matrix = aitchison_distance(proportions.values)
    dist_df = pd.DataFrame(
        dist_matrix, index=proportions.index, columns=proportions.index
    )

    g = sns.clustermap(
        dist_df,
        cmap="mako_r",
        figsize=figsize,
        xticklabels=True,
        yticklabels=True,
        linewidths=0.5,
        linecolor="white",
        annot=True if len(dist_df) <= 10 else False,
        fmt=".2f",
        cbar_kws={"label": "Aitchison distance", "shrink": 0.6},
        dendrogram_ratio=(0.1, 0.1),
    )
    g.fig.suptitle(title, y=1.02, fontweight="bold")
    g.ax_heatmap.set_xlabel("Brain Region", fontweight="bold")
    g.ax_heatmap.set_ylabel("Brain Region", fontweight="bold")
    g.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# Main Analysis Commands
# ============================================================================


@app.command()
def analyze(
    input_path: Path = typer.Option(
        "tangram_results/aligned_clustered_tangram_full.h5ad",
        help="Input h5ad with clustered + tangram results",
    ),
    output_dir: Path = typer.Option(
        "coda_results", help="Output directory for results"
    ),
    cluster_key: str = typer.Option(
        "leiden", help="Column for cluster/cell type labels"
    ),
    tangram_key: str = typer.Option(
        "tangram_ct_pred", help="Column for Tangram cell type predictions"
    ),
    region_key: str = typer.Option(
        "regions",
        help="Column for anatomical region labels (or 'slice_name' for by-slice "
        "analysis)",
    ),
    min_cells: int = typer.Option(50, help="Minimum cells per region for analysis"),
    individual_id: str = typer.Option(
        "brain1",
        help="Identifier for this individual (for future cross-individual analysis)",
    ),
):
    """
    Run compositional data analysis on aligned spatial transcriptomics data.

    Analyzes cell type compositions using Aitchison geometry:
    1. Extracts compositions per region (from both Leiden clusters and Tangram
       predictions)
    2. Applies CLR transformation
    3. Performs compositional PCA
    4. Computes variation matrices and Aitchison distances
    5. Generates visualization plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\n{'=' * 60}")
    print(f"Compositional Data Analysis - {individual_id}")
    print(f"{'=' * 60}\n")

    # Load data
    print(f"Loading data from {input_path}...")
    if not input_path.exists():
        # Try alternate path
        alt_path = Path("data/aligned_clustered.h5ad")
        if alt_path.exists():
            print(f"  Using alternate path: {alt_path}")
            input_path = alt_path
        else:
            raise FileNotFoundError(f"Could not find {input_path} or {alt_path}")

    adata = sc.read_h5ad(input_path)
    print(f"  Loaded: {adata.n_obs} cells, {adata.n_vars} genes")

    # Check available keys
    has_tangram = tangram_key in adata.obs.columns
    has_clusters = cluster_key in adata.obs.columns
    has_regions = region_key in adata.obs.columns

    # Check for boolean region columns (in_lamina, in_modula, etc.)
    bool_region_cols = [c for c in adata.obs.columns if c.startswith("in_")]
    if not has_regions and bool_region_cols:
        print(f"\n  Found boolean region columns: {bool_region_cols}")
        print("  Creating 'regions' column from boolean columns...")

        # Assign each cell to its region (or 'unknown' if none)
        def get_region(row):
            for col in bool_region_cols:
                if row[col]:
                    return col.replace("in_", "")
            return "unknown"

        adata.obs[region_key] = adata.obs[bool_region_cols].apply(get_region, axis=1)
        has_regions = True
        print(f"  Region counts: {adata.obs[region_key].value_counts().to_dict()}")

    print("\n  Available annotations:")
    print(f"    - Regions ({region_key}): {has_regions}")
    print(f"    - Clusters ({cluster_key}): {has_clusters}")
    print(f"    - Tangram predictions ({tangram_key}): {has_tangram}")

    if not has_regions:
        raise ValueError(f"Region column '{region_key}' not found in adata.obs")

    # Results storage for cross-individual comparison
    results = {"individual_id": individual_id, "n_cells": adata.n_obs, "analyses": {}}

    # ========================================================================
    # Analysis 1: Leiden cluster composition by region
    # ========================================================================
    if has_clusters:
        print(f"\n{'=' * 60}")
        print(f"Analysis 1: {cluster_key} composition by region")
        print(f"{'=' * 60}")

        leiden_props = get_region_compositions(
            adata, cluster_key, region_key, min_cells
        )
        print(f"  Regions with sufficient cells: {len(leiden_props)}")
        print(f"  Cell types: {len(leiden_props.columns)}")

        if len(leiden_props) < 2:
            print(
                "  WARNING: Need at least 2 regions for compositional "
                "analysis. Skipping."
            )
        else:
            # Save proportions
            leiden_props.to_csv(output_dir / f"{individual_id}_leiden_proportions.csv")

            # CLR transformation
            leiden_clr = pd.DataFrame(
                clr(leiden_props.values),
                index=leiden_props.index,
                columns=leiden_props.columns,
            )
            leiden_clr.to_csv(output_dir / f"{individual_id}_leiden_clr.csv")

            # Total variance
            tot_var = total_variance(leiden_props)
            print(f"  Total variance (Aitchison): {tot_var:.4f}")

            # Compositional PCA
            n_components = min(2, len(leiden_props) - 1, len(leiden_props.columns) - 1)
            if n_components >= 1:
                pca_result = compositional_pca(leiden_props, n_components=n_components)
                pca_result["coords"].to_csv(
                    output_dir / f"{individual_id}_leiden_pca_coords.csv"
                )
            else:
                pca_result = None

            # Store results
            results["analyses"]["leiden"] = {
                "n_regions": len(leiden_props),
                "n_celltypes": len(leiden_props.columns),
                "total_variance": tot_var,
                "var_explained": pca_result["explained_variance_ratio"].tolist()
                if pca_result
                else [],
            }

            # Plots
            print("\n  Generating plots...")
            plot_region_compositions(
                leiden_props,
                output_dir / f"{individual_id}_leiden_stacked_bar.png",
                title=f"{individual_id}: Leiden Cluster Composition by Region",
            )
            plot_clr_heatmap(
                leiden_props,
                output_dir / f"{individual_id}_leiden_clr_heatmap.png",
                title=f"{individual_id}: CLR-transformed Leiden Composition",
            )
            if pca_result and n_components >= 2:
                plot_composition_pca(
                    pca_result,
                    output_dir / f"{individual_id}_leiden_pca.png",
                    title=f"{individual_id}: Compositional PCA (Leiden)",
                )
            plot_variation_matrix(
                leiden_props,
                output_dir / f"{individual_id}_leiden_variation.png",
                title=f"{individual_id}: Leiden Log-ratio Variation",
            )
            plot_aitchison_distance_heatmap(
                leiden_props,
                output_dir / f"{individual_id}_leiden_aitchison_dist.png",
                title=f"{individual_id}: Aitchison Distance (Leiden)",
            )

    # ========================================================================
    # Analysis 2: Tangram cell type composition by region
    # ========================================================================
    if has_tangram:
        print(f"\n{'=' * 60}")
        print(f"Analysis 2: {tangram_key} composition by region")
        print(f"{'=' * 60}")

        tangram_props = get_region_compositions(
            adata, tangram_key, region_key, min_cells
        )
        print(f"  Regions with sufficient cells: {len(tangram_props)}")
        print(f"  Cell types: {len(tangram_props.columns)}")

        # Save proportions
        tangram_props.to_csv(output_dir / f"{individual_id}_tangram_proportions.csv")

        # CLR transformation
        tangram_clr = pd.DataFrame(
            clr(tangram_props.values),
            index=tangram_props.index,
            columns=tangram_props.columns,
        )
        tangram_clr.to_csv(output_dir / f"{individual_id}_tangram_clr.csv")

        # Total variance
        tot_var = total_variance(tangram_props)
        print(f"  Total variance (Aitchison): {tot_var:.4f}")

        # Compositional PCA
        pca_result = compositional_pca(tangram_props)
        pca_result["coords"].to_csv(
            output_dir / f"{individual_id}_tangram_pca_coords.csv"
        )

        # Store results
        results["analyses"]["tangram"] = {
            "n_regions": len(tangram_props),
            "n_celltypes": len(tangram_props.columns),
            "total_variance": tot_var,
            "var_explained": pca_result["explained_variance_ratio"].tolist(),
        }

        # Plots
        print("\n  Generating plots...")
        plot_region_compositions(
            tangram_props,
            output_dir / f"{individual_id}_tangram_stacked_bar.png",
            title=f"{individual_id}: Tangram Cell Type Composition by Region",
        )
        plot_clr_heatmap(
            tangram_props,
            output_dir / f"{individual_id}_tangram_clr_heatmap.png",
            title=f"{individual_id}: CLR-transformed Tangram Composition",
        )
        plot_composition_pca(
            pca_result,
            output_dir / f"{individual_id}_tangram_pca.png",
            title=f"{individual_id}: Compositional PCA (Tangram)",
        )
        plot_variation_matrix(
            tangram_props,
            output_dir / f"{individual_id}_tangram_variation.png",
            title=f"{individual_id}: Tangram Log-ratio Variation",
        )
        plot_aitchison_distance_heatmap(
            tangram_props,
            output_dir / f"{individual_id}_tangram_aitchison_dist.png",
            title=f"{individual_id}: Aitchison Distance (Tangram)",
        )

    # ========================================================================
    # Analysis 3: Composition variation along Z-axis
    # ========================================================================
    if "slice_idx" in adata.obs.columns:
        print(f"\n{'=' * 60}")
        print("Analysis 3: Composition variation along Z-axis")
        print(f"{'=' * 60}")

        if has_clusters:
            slice_props = get_slice_compositions(adata, cluster_key, min_cells=100)
            slice_props.to_csv(output_dir / f"{individual_id}_leiden_by_slice.csv")

            fig, ax = plt.subplots(figsize=(14, 6))
            slice_props.plot(
                kind="area", stacked=True, ax=ax, colormap="tab20", alpha=0.8
            )
            ax.set_xlabel("Slice Index (Z)")
            ax.set_ylabel("Proportion")
            ax.set_title(f"{individual_id}: Leiden Cluster Composition Along Z-axis")
            ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            fig.savefig(
                output_dir / f"{individual_id}_leiden_z_composition.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
            print(
                f"  Saved: {output_dir / f'{individual_id}_leiden_z_composition.png'}"
            )

    # ========================================================================
    # Save summary results (for cross-individual comparison)
    # ========================================================================
    import json

    with open(output_dir / f"{individual_id}_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'=' * 60}\n")

    return results


@app.command()
def compare(
    individual1: Path = typer.Option(..., help="Path to individual 1 summary JSON"),
    individual2: Path = typer.Option(..., help="Path to individual 2 summary JSON"),
    output_dir: Path = typer.Option("coda_results/comparison", help="Output directory"),
):
    """
    Compare cell type compositions between two individuals.

    [Placeholder for future cross-individual analysis]

    Will implement:
    - PERMANOVA on Aitchison distances
    - Differential composition analysis
    - Effect size estimation for each cell type
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(individual1) as f:
        results1 = json.load(f)
    with open(individual2) as f:
        results2 = json.load(f)

    print(f"\nComparing {results1['individual_id']} vs {results2['individual_id']}")
    print("  [Cross-individual comparison not yet implemented]")
    print("  This will include:")
    print("    - PERMANOVA on Aitchison distances")
    print("    - Differential abundance analysis")
    print("    - Region-specific composition differences")


if __name__ == "__main__":
    app()
