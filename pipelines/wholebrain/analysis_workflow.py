#!/usr/bin/env python3
"""
Analysis workflow for the whole-brain Xenium pipelines (queen and worker).

Shared by both brains: run it from the brain's directory (see brain_dir.py),
which is what every Snakemake rule does.

Usage (from pipelines/wholebrain/queen or pipelines/wholebrain/worker):
    python ../analysis_workflow.py cluster [--resolution 1.0]
    python ../analysis_workflow.py align [--output aligned]
    python ../analysis_workflow.py visualize-3d [--color-by leiden]
"""

from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import plotly.graph_objects as go
import scanpy as sc
import typer
from brain_dir import BRAIN_DIR, annotations_dir
from labeling_core import (
    add_labels_to_adata,
    discover_annotations,
    load_geojson_regions,
    parse_annotation_filename,
    plot_labels,
    propagate_labels_to_neighbors,
)

app = typer.Typer(help="Analysis workflow for whole-brain Xenium data")

# =============================================================================
# CONFIG
# =============================================================================

DATA_DIR = BRAIN_DIR / "data"
FIGURES_DIR = BRAIN_DIR / "figures"


def get_slice_sort_key(filepath: Path) -> tuple:
    """
    Sort slices in stacking order: 1A, 1B, 2A, ..., 3C, 3D, ..., 8A, 2_1A, ..., 2_8D.

    Returns tuple: (sample_order, region_num, suffix_order)
    """
    import re

    name = filepath.stem

    # Extract crop suffix (e.g., "1A", "1B", "2_1A", "2_1B")
    # Region numbers are single digits 1-8, use \d not \d+ to avoid matching timestamps
    match = re.search(r"_(2_)?(\d)([A-D])(?:_clustered)?$", name)
    if not match:
        raise ValueError(f"cannot parse crop suffix from {name!r}")

    sample_prefix = match.group(1)  # "2_" or None
    region_num = int(match.group(2))  # 1-8
    suffix = match.group(3)  # "A" to "D"

    sample_order = 1 if sample_prefix else 0  # Sample 1 (no prefix) first
    suffix_order = "ABCD".index(suffix)

    return (sample_order, region_num, suffix_order)


# =============================================================================
# CLUSTERING
# =============================================================================


@app.command()
def cluster(
    input_file: str = typer.Option("aligned", help="Input h5ad file (without .h5ad)"),
    resolution: float = typer.Option(1.0, help="Leiden clustering resolution"),
    n_neighbors: int = typer.Option(15, help="Number of neighbors for graph"),
    n_pcs: int = typer.Option(50, help="Number of PCs to use"),
    n_jobs: int = typer.Option(-1, help="Number of parallel jobs (-1 for all cores)"),
    min_dist: float = typer.Option(
        0.1, help="UMAP min_dist (lower=tighter clusters, 0.0-1.0)"
    ),
    spread: float = typer.Option(1.0, help="UMAP spread (higher=more spread)"),
):
    """Run Leiden clustering on aligned data."""
    import os

    # Set parallelization
    n_cpus = os.cpu_count() or 4
    if n_jobs == -1:
        n_jobs = n_cpus
    sc.settings.n_jobs = n_jobs
    typer.echo(f"⚡ Using {n_jobs} threads")

    input_path = DATA_DIR / f"{input_file}.h5ad"
    if not input_path.exists():
        typer.echo(f"❌ File not found: {input_path}")
        typer.echo("   Run 'align' first to create aligned.h5ad")
        raise typer.Exit(1)

    typer.echo(f"📂 Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)
    typer.echo(f"   Cells: {adata.n_obs}, Genes: {adata.n_vars}")

    # Preprocessing (if not already done)
    if "log1p" not in adata.uns:
        typer.echo("   Normalizing...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    typer.echo("   Finding HVGs...")
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)

    # PCA + neighbors
    typer.echo("   Running PCA...")
    sc.tl.pca(adata, n_comps=n_pcs)
    typer.echo("   Building neighbor graph...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    # Leiden clustering
    typer.echo(f"   Clustering (resolution={resolution})...")
    sc.tl.leiden(
        adata, resolution=resolution, flavor="igraph", n_iterations=2, directed=False
    )
    typer.echo(f"   ✅ Found {adata.obs['leiden'].nunique()} clusters")

    # UMAP
    typer.echo(f"   Computing UMAP (min_dist={min_dist}, spread={spread})...")
    sc.tl.umap(adata, min_dist=min_dist, spread=spread)

    # Save
    output_path = DATA_DIR / f"{input_file}_clustered.h5ad"
    adata.write_h5ad(output_path)
    typer.echo(f"💾 Saved: {output_path}")

    # Generate UMAP plot
    typer.echo("   Generating UMAP plot...")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # UMAP colored by leiden
    sc.pl.umap(
        adata,
        color="leiden",
        ax=axes[0],
        show=False,
        legend_loc="on data",
        legend_fontsize=8,
        title="Leiden Clusters",
    )

    # UMAP colored by slice (if available)
    if "slice_name" in adata.obs:
        sc.pl.umap(
            adata, color="slice_name", ax=axes[1], show=False, title="Slice Origin"
        )
    else:
        axes[1].axis("off")

    plt.tight_layout()
    umap_path = FIGURES_DIR / f"{input_file}_umap.png"
    plt.savefig(umap_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    typer.echo(f"📊 Saved UMAP: {umap_path}")


@app.command()
def recompute_umap(
    input_file: str = typer.Option(
        ..., help="Input h5ad file path (Tangram output with metadata)"
    ),
    projected_genes: str = typer.Option(
        None, help="Projected genes h5ad (if None, derives from input_file)"
    ),
    output_file: str = typer.Option(
        None, help="Output h5ad file path (default: overwrites input)"
    ),
    n_pcs: int = typer.Option(50, help="Number of PCs for PCA"),
    n_neighbors: int = typer.Option(15, help="Number of neighbors for graph"),
    min_dist: float = typer.Option(0.1, help="UMAP min_dist parameter"),
    spread: float = typer.Option(1.0, help="UMAP spread parameter"),
    n_jobs: int = typer.Option(16, help="Number of threads"),
):
    """
    Re-compute UMAP on Tangram output using the full imputed gene set.

    The UMAP is computed on all genes (the ~100 panel genes plus the
    Tangram-imputed ones) rather than on the panel alone.

    Loads metadata (leiden, spatial, slice_name) from input_file, and
    gene expression from projected_genes file (the imputed ~12k genes).
    """
    from pathlib import Path

    import matplotlib.pyplot as plt
    import scanpy as sc

    sc.settings.n_jobs = n_jobs
    typer.echo(f"⚡ Using {n_jobs} threads")

    # Load Tangram output (metadata)
    input_path = Path(input_file)
    if not input_path.exists():
        typer.echo(f"❌ File not found: {input_path}")
        raise typer.Exit(1)

    typer.echo(f"📂 Loading metadata from {input_path.name}...")
    adata = sc.read_h5ad(input_path)
    typer.echo(f"   Cells: {adata.n_obs:,}, Original genes: {adata.n_vars:,}")

    # Load projected genes (imputed expression)
    if projected_genes is None:
        # Derive from input file name: *_tangram.h5ad -> *_projected_genes.h5ad
        projected_path = input_path.parent / input_path.name.replace(
            "_tangram.h5ad", "_projected_genes.h5ad"
        )
    else:
        projected_path = Path(projected_genes)

    if projected_path.exists():
        typer.echo(f"📂 Loading imputed genes from {projected_path.name}...")
        adata_ge = sc.read_h5ad(projected_path)
        typer.echo(f"   Imputed genes: {adata_ge.n_vars:,}")

        # Match cells between the two files
        common_cells = adata.obs_names.intersection(adata_ge.obs_names)
        typer.echo(f"   Common cells: {len(common_cells):,}")

        if len(common_cells) > 0:
            # Subset and reorder both to match
            adata = adata[common_cells].copy()
            adata_ge = adata_ge[common_cells].copy()

            # Replace .X with imputed gene expression
            adata_combined = sc.AnnData(
                X=adata_ge.X,
                obs=adata.obs,
                var=adata_ge.var,
                obsm=adata.obsm,
                uns=adata.uns,
            )
            adata = adata_combined
            typer.echo(f"   ✅ Using {adata.n_vars:,} imputed genes for UMAP")
        else:
            typer.echo("   ⚠️ No common cells, using original genes")
    else:
        typer.echo(
            f"   ⚠️ Projected genes not found at {projected_path}, using "
            f"original {adata.n_vars} genes"
        )

    # Re-run PCA on gene set (cap n_pcs at n_features - 1)
    actual_n_pcs = min(n_pcs, adata.n_vars - 1)
    typer.echo(f"   Running PCA on {adata.n_vars} genes (n_pcs={actual_n_pcs})...")
    sc.tl.pca(adata, n_comps=actual_n_pcs)

    # Re-build neighbor graph
    typer.echo(f"   Building neighbor graph (n_neighbors={n_neighbors})...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=actual_n_pcs)

    # Re-compute UMAP
    typer.echo(f"   Computing UMAP (min_dist={min_dist}, spread={spread})...")
    sc.tl.umap(adata, min_dist=min_dist, spread=spread)

    # Save
    output_path = Path(output_file) if output_file else input_path
    adata.write_h5ad(output_path)
    typer.echo(f"💾 Saved: {output_path}")

    # Generate comprehensive UMAP visualizations
    typer.echo("   Generating UMAP visualizations...")
    umap_dir = FIGURES_DIR / "umap_tangram"
    umap_dir.mkdir(parents=True, exist_ok=True)

    umap_coords = adata.obsm["X_umap"]

    # 1. Main overview figure (clusters + slices)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    if "leiden" in adata.obs:
        sc.pl.umap(
            adata,
            color="leiden",
            ax=axes[0],
            show=False,
            legend_loc="right margin",
            legend_fontsize=9,
            title="Leiden Clusters (Tangram genes)",
            frameon=False,
        )

    if "slice_name" in adata.obs:
        sc.pl.umap(
            adata,
            color="slice_name",
            ax=axes[1],
            show=False,
            legend_loc="right margin",
            title="Slice Origin",
            frameon=False,
        )

    plt.tight_layout()
    plt.savefig(
        umap_dir / "umap_overview.png", dpi=200, bbox_inches="tight", facecolor="white"
    )
    plt.close()

    # 2. Per-slice UMAP panels (highlight each slice)
    if "slice_name" in adata.obs:
        slices = sorted(adata.obs["slice_name"].unique())
        n_slices = len(slices)

        fig, axes = plt.subplots(1, n_slices, figsize=(5 * n_slices, 5))
        if n_slices == 1:
            axes = [axes]

        for i, slice_name in enumerate(slices):
            ax = axes[i]
            # Plot all cells in gray
            ax.scatter(
                umap_coords[:, 0],
                umap_coords[:, 1],
                c="#e0e0e0",
                s=1,
                alpha=0.3,
                rasterized=True,
            )
            # Highlight this slice
            mask = adata.obs["slice_name"] == slice_name
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c="#e41a1c",
                s=3,
                alpha=0.7,
                rasterized=True,
            )

            short_name = (
                slice_name.split("__")[-1] if "__" in slice_name else slice_name
            )
            ax.set_title(f"{short_name}", fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        plt.suptitle(
            "UMAP by Slice (Tangram genes)", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        plt.savefig(
            umap_dir / "umap_per_slice.png",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

    # 3. Spatial projection - show UMAP colors on physical coordinates
    if "slice_name" in adata.obs and "spatial" in adata.obsm and "leiden" in adata.obs:
        slices = sorted(adata.obs["slice_name"].unique())
        n_slices = len(slices)

        # Get leiden colors
        n_clusters = adata.obs["leiden"].nunique()
        cmap = plt.cm.get_cmap("tab20", n_clusters)
        cluster_colors = {str(i): cmap(i) for i in range(n_clusters)}

        fig, axes = plt.subplots(2, n_slices, figsize=(5 * n_slices, 10))
        if n_slices == 1:
            axes = axes.reshape(2, 1)

        for i, slice_name in enumerate(slices):
            mask = adata.obs["slice_name"] == slice_name
            slice_data = adata[mask]
            coords = slice_data.obsm["spatial"]
            clusters = slice_data.obs["leiden"].astype(str)
            colors = [cluster_colors.get(c, "#999999") for c in clusters]

            # Top row: spatial with cluster colors
            ax_spatial = axes[0, i]
            ax_spatial.scatter(
                coords[:, 0], coords[:, 1], c=colors, s=1, alpha=0.7, rasterized=True
            )
            ax_spatial.set_aspect("equal")
            ax_spatial.invert_yaxis()
            short_name = (
                slice_name.split("__")[-1] if "__" in slice_name else slice_name
            )
            ax_spatial.set_title(f"{short_name} - Spatial", fontsize=10)
            ax_spatial.axis("off")

            # Bottom row: UMAP for this slice with cluster colors
            ax_umap = axes[1, i]
            slice_umap = slice_data.obsm["X_umap"]
            ax_umap.scatter(
                slice_umap[:, 0],
                slice_umap[:, 1],
                c=colors,
                s=1,
                alpha=0.7,
                rasterized=True,
            )
            ax_umap.set_title(f"{short_name} - UMAP", fontsize=10)
            ax_umap.set_xticks([])
            ax_umap.set_yticks([])
            for spine in ax_umap.spines.values():
                spine.set_visible(False)

        plt.suptitle(
            "Spatial vs UMAP by Slice (colored by Leiden)",
            fontsize=14,
            fontweight="bold",
            y=1.01,
        )
        plt.tight_layout()
        plt.savefig(
            umap_dir / "spatial_umap_comparison.png",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

    # 4. Large 2D spatial plots per slice (publication quality)
    if "slice_name" in adata.obs and "spatial" in adata.obsm and "leiden" in adata.obs:
        slices = sorted(adata.obs["slice_name"].unique())
        n_slices = len(slices)

        n_clusters = adata.obs["leiden"].nunique()
        cmap = plt.cm.get_cmap("tab20", n_clusters)
        cluster_colors = {str(i): cmap(i) for i in range(n_clusters)}

        # Individual large spatial plots
        for slice_name in slices:
            mask = adata.obs["slice_name"] == slice_name
            slice_data = adata[mask]
            coords = slice_data.obsm["spatial"]
            clusters = slice_data.obs["leiden"].astype(str)
            colors = [cluster_colors.get(c, "#999999") for c in clusters]

            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(
                coords[:, 0], coords[:, 1], c=colors, s=2, alpha=0.8, rasterized=True
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.axis("off")

            short_name = (
                slice_name.split("__")[-1] if "__" in slice_name else slice_name
            )
            ax.set_title(
                f"{short_name} - Spatial Clusters", fontsize=14, fontweight="bold"
            )

            plt.tight_layout()
            plt.savefig(
                umap_dir / f"spatial_{short_name}.png",
                dpi=300,
                bbox_inches="tight",
                facecolor="white",
            )
            plt.close()

        # Combined grid of all slices
        fig, axes = plt.subplots(1, n_slices, figsize=(6 * n_slices, 6))
        if n_slices == 1:
            axes = [axes]

        for i, slice_name in enumerate(slices):
            mask = adata.obs["slice_name"] == slice_name
            slice_data = adata[mask]
            coords = slice_data.obsm["spatial"]
            clusters = slice_data.obs["leiden"].astype(str)
            colors = [cluster_colors.get(c, "#999999") for c in clusters]

            ax = axes[i]
            ax.scatter(
                coords[:, 0], coords[:, 1], c=colors, s=1, alpha=0.8, rasterized=True
            )
            ax.set_aspect("equal")
            ax.invert_yaxis()
            ax.axis("off")

            short_name = (
                slice_name.split("__")[-1] if "__" in slice_name else slice_name
            )
            ax.set_title(f"{short_name}", fontsize=12, fontweight="bold")

        plt.suptitle(
            "Spatial Clusters by Slice", fontsize=14, fontweight="bold", y=1.02
        )
        plt.tight_layout()
        plt.savefig(
            umap_dir / "spatial_all_slices.png",
            dpi=200,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close()

    # 5. Interactive 3D visualization
    if "slice_name" in adata.obs and "spatial" in adata.obsm and "leiden" in adata.obs:
        typer.echo("   Generating 3D visualization...")
        try:
            import plotly.graph_objects as go

            slices = sorted(adata.obs["slice_name"].unique())
            z_spacing = 20  # µm between slices

            # Subsample for performance
            n_sample = min(30000, adata.n_obs)
            idx = np.random.choice(adata.n_obs, n_sample, replace=False)
            adata_sub = adata[idx]

            # Build 3D coordinates
            coords_2d = adata_sub.obsm["spatial"]
            slice_names = adata_sub.obs["slice_name"].values
            z_coords = np.array([slices.index(s) * z_spacing for s in slice_names])

            coords_3d = np.column_stack([coords_2d, z_coords])

            # Colors by cluster
            n_clusters = adata.obs["leiden"].nunique()
            cmap = plt.cm.get_cmap("tab20", n_clusters)
            clusters = adata_sub.obs["leiden"].astype(int).values
            colors = [
                f"rgb({int(cmap(c)[0] * 255)},{int(cmap(c)[1] * 255)},"
                f"{int(cmap(c)[2] * 255)})"
                for c in clusters
            ]

            fig = go.Figure(
                data=[
                    go.Scatter3d(
                        x=coords_3d[:, 0],
                        y=coords_3d[:, 1],
                        z=coords_3d[:, 2],
                        mode="markers",
                        marker=dict(size=1.5, color=colors, opacity=0.7),
                        text=[f"Cluster {c}" for c in clusters],
                        hoverinfo="text",
                    )
                ]
            )

            fig.update_layout(
                title="3D Spatial Clusters (Tangram)",
                scene=dict(
                    xaxis_title="X (pixels)",
                    yaxis_title="Y (pixels)",
                    zaxis_title="Z (slice)",
                    aspectmode="data",
                ),
                width=1000,
                height=800,
            )

            fig.write_html(umap_dir / "spatial_3d_clusters.html")
            typer.echo(f"   Saved 3D plot: {umap_dir / 'spatial_3d_clusters.html'}")

        except ImportError:
            typer.echo("   ⚠️ Plotly not available, skipping 3D visualization")

    typer.echo(f"📊 Saved figures to: {umap_dir}/")


# =============================================================================
# IMAGE ALIGNMENT HELPERS
# =============================================================================


def transform_image_with_full_extent(
    img, rotation_rad, center_of_rotation, additional_translation, output_shape
):
    """
    Apply transformation to image with full extent preservation.

    Args:
        img: Image tensor of shape (C, H, W)
        rotation_rad: Rotation angle in radians
        center_of_rotation: (2,) array giving center to rotate around
        additional_translation: (2,) array giving translation to apply after rotation
        output_shape: Output shape (H, W)

    Returns:
        Transformed image
    """
    from skimage.transform import AffineTransform, warp

    # Convert PyTorch tensors to numpy if needed
    if hasattr(rotation_rad, "cpu"):
        rotation_rad = rotation_rad.cpu().numpy()
    if isinstance(rotation_rad, np.ndarray):
        rotation_rad = float(rotation_rad)

    if hasattr(center_of_rotation, "cpu"):
        center_of_rotation = center_of_rotation.cpu().numpy()
    center_of_rotation = np.array(center_of_rotation).flatten()

    if hasattr(additional_translation, "cpu"):
        additional_translation = additional_translation.cpu().numpy()
    additional_translation = np.array(additional_translation).flatten()

    tx, ty = center_of_rotation
    add_tx, add_ty = additional_translation

    # Build Affine Transformation: rotate around center, then translate
    transform = (
        AffineTransform(translation=(-tx, -ty))
        + AffineTransform(rotation=rotation_rad)
        + AffineTransform(translation=(tx, ty))
        + AffineTransform(translation=(add_tx, add_ty))
    )

    # Apply the transformation to each channel
    C, H, W = img.shape
    transformed_channels = []

    for c in range(C):
        transformed = warp(
            img[c],
            inverse_map=transform.inverse,
            output_shape=output_shape,
            order=1,
            preserve_range=True,
            mode="constant",
            cval=0,
        )
        transformed_channels.append(transformed)

    return np.stack(transformed_channels, axis=0)


def align_images_with_paste(
    aligned_slices, original_slices, rotation_angles, translations
):
    """
    Apply PASTE3 transformations to images.

    Slice 0 is the reference (padded only). Each later slice gets its single
    pairwise transform (i-1), the same as apply_transforms_to_coordinates:
    stack_slices_pairwise aligns each slice to the already-aligned previous one,
    so the transforms are not compounded.
    """
    import torch
    from skimage.transform import AffineTransform, warp

    typer.echo(f"   Processing {len(original_slices)} images...")

    if not all("image" in s.uns for s in original_slices):
        typer.echo("   ⚠️ Not all slices have images in uns['image']")
        return None

    def get_val(v):
        if isinstance(v, torch.Tensor):
            return v.cpu().numpy()
        return np.array(v).flatten()

    def get_rot(r):
        if isinstance(r, torch.Tensor):
            return (
                float(r.cpu().item()) if r.numel() == 1 else float(r.cpu().numpy()[0])
            )
        return float(r)

    def transform_image(img, rotation_rad, center, translation, output_shape=None):
        """Apply rotation around center, then translate.

        If output_shape provided, use that. Otherwise keep same size as input.
        """
        tx, ty = float(center[0]), float(center[1])
        add_tx, add_ty = float(translation[0]), float(translation[1])

        H, W = img.shape[1], img.shape[2]
        out_shape = output_shape if output_shape else (H, W)

        # Rotate CCW around center, then translate
        transform = (
            AffineTransform(translation=(-tx, -ty))
            + AffineTransform(rotation=float(rotation_rad))
            + AffineTransform(translation=(tx, ty))
            + AffineTransform(translation=(add_tx, add_ty))
        )

        C = img.shape[0]
        channels = []
        for c in range(C):
            warped = warp(
                img[c],
                inverse_map=transform.inverse,
                order=1,
                preserve_range=True,
                output_shape=out_shape,
            )
            channels.append(warped)
        return np.stack(channels, axis=0)

    transformed_images = []
    n_slices = len(original_slices)

    # Get real_center (slice 0's centroid) - used for final positioning
    real_center = (
        get_val(translations[0][0]) if len(translations) > 0 else np.array([0.0, 0.0])
    )

    # Compute common output shape (max H, W across all images)
    max_H = max(s.uns["image"].shape[1] for s in original_slices)
    max_W = max(s.uns["image"].shape[2] for s in original_slices)
    common_shape = (max_H, max_W)
    typer.echo(f"   Common output shape: {common_shape}")

    for i in range(n_slices):
        img = original_slices[i].uns["image"]

        if i == 0:
            # Slice 0: reference - no rotation, but pad to common size
            typer.echo("   Slice 1: reference (no transform)")
            # Pad to common shape
            C = img.shape[0]
            padded = np.zeros((C, max_H, max_W), dtype=img.dtype)
            padded[:, : img.shape[1], : img.shape[2]] = img
            transformed_images.append(padded)
        else:
            # Single transform per slice - matches apply_transforms_to_coordinates
            rotation_rad = get_rot(rotation_angles[i - 1])
            trans_target = get_val(translations[i - 1][1])  # slice i's centroid

            # Coord transform: new = (coord - trans_target) @ R
            # Coord overlay:   displayed = new + real_center
            # So:              displayed = (coord - trans_target) @ R + real_center
            #
            # Image transform must match:
            # new_pos = (pos - trans_target) @ R + real_center
            #
            # This is: rotate around trans_target, then translate by (real_center -
            # trans_target)
            translation = real_center - trans_target

            typer.echo(
                f"   Slice {i + 1}: rot={np.degrees(rotation_rad):.2f}°, "
                f"center=({trans_target[0]:.0f},{trans_target[1]:.0f}), "
                f"translate=({translation[0]:.0f},{translation[1]:.0f})"
            )

            transformed = transform_image(
                img, rotation_rad, trans_target, translation, output_shape=common_shape
            )
            transformed_images.append(transformed.astype(img.dtype))

    return transformed_images, 0


def save_aligned_images(images, output_path):
    """Save aligned images as a multi-page TIFF (DAPI channel only).

    Matches spatial_dataification_3slices.py output format:
    - Only saves channel 0 (DAPI)
    - Output shape: (N_slices, H, W)
    """
    from tifffile import imwrite

    # Find max dimensions across all images
    max_H = max(img.shape[1] for img in images)
    max_W = max(img.shape[2] for img in images)

    padded_images = []
    for img in images:
        H, W = img.shape[1], img.shape[2]
        # Only use channel 0 (DAPI), pad to common size
        padded = np.zeros((max_H, max_W), dtype=img[0].dtype)
        padded[:H, :W] = img[0]  # Channel 0 only
        padded_images.append(padded)

    # Stack into (N_slices, H, W)
    tiff_stack = np.stack(padded_images, axis=0)
    typer.echo(f"   Saving TIFF stack with shape {tiff_stack.shape}")

    imwrite(output_path, tiff_stack)
    typer.echo(f"   ✅ Saved to {output_path}")


def extract_paste3_transforms(rotation_angles, translations, n_slices):
    """
    Extract rotation and translation values from PASTE3 output.

    PASTE3 returns:
    - rotation_angles: list of angles (one per pairwise alignment)
    - translations: list of (tensor([x1,y1]), tensor([x2,y2])) tuples

    Returns list of dicts with rotation_deg, trans_x, trans_y for each slice.
    """
    import torch

    transform_data = []

    for i in range(n_slices):
        # First slice is reference (no transformation)
        if i == 0:
            transform_data.append(
                {
                    "rotation_deg": 0.0,
                    "trans_x": 0.0,
                    "trans_y": 0.0,
                }
            )
            continue

        # For subsequent slices, use the pairwise alignment (i-1)
        pair_idx = i - 1

        # Extract rotation angle
        if pair_idx < len(rotation_angles):
            theta = rotation_angles[pair_idx]
            if isinstance(theta, torch.Tensor):
                if theta.numel() == 1:
                    theta = float(theta.cpu().item())
                else:
                    theta = float(theta.cpu().numpy()[0])
            elif hasattr(theta, "__float__"):
                theta = float(theta)
            else:
                theta = 0.0
        else:
            theta = 0.0

        # Extract translation - PASTE3 returns (trans_i, trans_j) tuples of 2D tensors
        tx, ty = 0.0, 0.0
        if pair_idx < len(translations):
            trans = translations[pair_idx]
            if isinstance(trans, tuple) and len(trans) >= 2:
                # trans = (tensor([x1,y1]), tensor([x2,y2]))
                # Use the second element (the "to" slice translation)
                t = trans[1]
                if isinstance(t, torch.Tensor):
                    t_np = t.cpu().numpy()
                    tx, ty = float(t_np[0]), float(t_np[1])
                elif hasattr(t, "__iter__"):
                    tx, ty = float(t[0]), float(t[1])
            elif isinstance(trans, torch.Tensor):
                t_np = trans.cpu().numpy()
                if t_np.ndim == 1 and len(t_np) >= 2:
                    tx, ty = float(t_np[0]), float(t_np[1])

        transform_data.append(
            {
                "rotation_deg": np.degrees(theta),
                "trans_x": tx,
                "trans_y": ty,
            }
        )

    return transform_data


def save_image_alignment_validation(
    original_slices,
    transformed_images,
    rotation_angles,
    translations,
    slice_names: list[str],
    aligned_slices=None,
    padding=0,
):
    """
    Save before/after comparison of image alignment.

    Shows transformed images with aligned coordinates overlaid (like
    spatial_dataification_3slices.py).
    NOTE: Coordinates are already in pixel space (converted before PASTE).
    """
    import matplotlib.pyplot as plt
    import torch

    output_dir = FIGURES_DIR / "alignment_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_slices = len(original_slices)
    typer.echo(f"   Creating before/after image comparison for {n_slices} slices...")

    def get_val(v):
        if isinstance(v, torch.Tensor):
            return v.cpu().numpy()
        return np.array(v).flatten()

    # Get real_center for coordinate shifting (like spatial_dataification_3slices.py)
    # Already in pixels (converted before PASTE)
    real_center = (
        get_val(translations[0][0]) if len(translations) > 0 else np.array([0, 0])
    )

    # Extract transformation values for display
    transform_data = extract_paste3_transforms(rotation_angles, translations, n_slices)

    # Create figure: 2 columns (before/after) x n_slices rows
    fig, axes = plt.subplots(n_slices, 2, figsize=(14, 5 * n_slices))
    if n_slices == 1:
        axes = axes.reshape(1, -1)

    for i in range(n_slices):
        short_name = (
            slice_names[i].split("__")[-1] if "__" in slice_names[i] else slice_names[i]
        )
        td = transform_data[i]

        # BEFORE: original image with ORIGINAL coordinates
        orig_img = original_slices[i].uns["image"]
        orig_display = orig_img[0] if orig_img.ndim == 3 else orig_img
        orig_vmax = (
            np.percentile(orig_display[orig_display > 0], 99.5)
            if np.any(orig_display > 0)
            else 1
        )

        ax_before = axes[i, 0]
        ax_before.imshow(orig_display, cmap="gray", vmin=0, vmax=orig_vmax)

        # Overlay ORIGINAL coordinates on BEFORE image (already in pixels)
        orig_coords = original_slices[i].obsm.get("spatial", None)
        if orig_coords is not None:
            if isinstance(orig_coords, torch.Tensor):
                orig_coords = orig_coords.cpu().numpy()
            n_pts = min(3000, len(orig_coords))
            idx = np.random.choice(len(orig_coords), n_pts, replace=False)
            ax_before.scatter(
                orig_coords[idx, 0], orig_coords[idx, 1], s=1, c="cyan", alpha=0.4
            )

        ax_before.set_title(
            f"{short_name} - BEFORE\n(cyan = original coords)", fontsize=10
        )
        ax_before.axis("off")

        # AFTER: transformed image with ALIGNED coords overlay
        trans_img = transformed_images[i]
        trans_display = trans_img[0] if trans_img.ndim == 3 else trans_img
        trans_vmax = (
            np.percentile(trans_display[trans_display > 0], 99.5)
            if np.any(trans_display > 0)
            else 1
        )

        ax_after = axes[i, 1]
        ax_after.imshow(trans_display, cmap="gray", vmin=0, vmax=trans_vmax)

        # Overlay ALIGNED coordinates - shift to match transformed image
        # stack_slices_pairwise returns coords in a common frame centered at origin
        # ALL slices need the same shift (real_center) to put them in image coordinates
        # (matches spatial_dataification_3slices.py: shifted_coords = coords +
        # real_center)
        if aligned_slices is not None and i < len(aligned_slices):
            coords = aligned_slices[i].obsm.get("spatial", None)
            if coords is not None:
                if isinstance(coords, torch.Tensor):
                    coords = coords.cpu().numpy()
                # ALL slices use real_center (translations[0][0]) - same reference frame
                shifted = coords + real_center
                n_pts = min(3000, len(shifted))
                idx = np.random.choice(len(shifted), n_pts, replace=False)
                ax_after.scatter(
                    shifted[idx, 0], shifted[idx, 1], s=1, c="red", alpha=0.4
                )

        ax_after.set_title(
            f"{short_name} - AFTER (rot: {td['rotation_deg']:.1f}°)\n(red = "
            "aligned coords)",
            fontsize=10,
        )
        ax_after.axis("off")

    plt.suptitle(
        "Image Alignment: Before vs After (red = aligned coords)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = output_dir / "alignment_images_before_after.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    typer.echo(f"   Saved before/after comparison: {output_path}")


def save_image_alignment_composite(transformed_images, slice_names: list[str]):
    """Create color composite showing image alignment (red/cyan for consecutive
    pairs)."""
    import matplotlib.pyplot as plt

    output_dir = FIGURES_DIR / "alignment_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_slices = len(transformed_images)
    typer.echo(f"   Creating color composite for {n_slices} aligned images...")

    def norm(img):
        img_2d = img[0] if img.ndim == 3 else img
        if np.any(img_2d > 0):
            vmax = np.percentile(img_2d[img_2d > 0], 99.5)
            return np.clip(img_2d / (vmax + 1e-8), 0, 1)
        return np.zeros_like(img_2d)

    def pad_to_common_size(img1, img2):
        """Pad both images to the same size."""
        h1, w1 = img1.shape
        h2, w2 = img2.shape
        H, W = max(h1, h2), max(w1, w2)
        padded1 = np.zeros((H, W), dtype=img1.dtype)
        padded2 = np.zeros((H, W), dtype=img2.dtype)
        padded1[:h1, :w1] = img1
        padded2[:h2, :w2] = img2
        return padded1, padded2

    # Pairwise composites
    n_pairs = n_slices - 1
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 6))
    if n_pairs == 1:
        axes = [axes]

    for i in range(n_pairs):
        img1, img2 = norm(transformed_images[i]), norm(transformed_images[i + 1])
        img1, img2 = pad_to_common_size(img1, img2)
        H, W = img1.shape
        composite = np.zeros((H, W, 3), dtype=np.float32)
        composite[:, :, 0] = img1  # Red
        composite[:, :, 1] = img2  # Cyan (G+B)
        composite[:, :, 2] = img2

        # Crop to content
        mask = (img1 > 0.01) | (img2 > 0.01)
        if np.any(mask):
            rows, cols = np.any(mask, axis=1), np.any(mask, axis=0)
            r0, r1 = np.where(rows)[0][[0, -1]]
            c0, c1 = np.where(cols)[0][[0, -1]]
            composite = composite[max(0, r0 - 50) : r1 + 50, max(0, c0 - 50) : c1 + 50]

        n1 = (
            slice_names[i].split("__")[-1] if "__" in slice_names[i] else slice_names[i]
        )
        n2 = (
            slice_names[i + 1].split("__")[-1]
            if "__" in slice_names[i + 1]
            else slice_names[i + 1]
        )
        axes[i].imshow(composite)
        axes[i].set_title(f"{n1} (red) vs {n2} (cyan)")
        axes[i].axis("off")

    plt.suptitle(
        "Image Alignment Composite (white=overlap)", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "alignment_images_composite.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()
    typer.echo(f"   Saved: {output_dir / 'alignment_images_composite.png'}")


def save_coordinate_alignment_composite(
    aligned_slices, slice_names: list[str], translations
):
    """Create pairwise coordinate composite showing alignment (red/cyan for consecutive
    pairs)."""
    import matplotlib.pyplot as plt
    import torch

    output_dir = FIGURES_DIR / "alignment_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_slices = len(aligned_slices)
    typer.echo(f"   Creating coordinate composite for {n_slices} aligned slices...")

    def get_val(v):
        if isinstance(v, torch.Tensor):
            return v.cpu().numpy()
        return np.array(v).flatten()

    # Get real_center for shifting coords to image space
    real_center = (
        get_val(translations[0][0]) if len(translations) > 0 else np.array([0, 0])
    )

    # Pairwise composites
    n_pairs = n_slices - 1
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 6))
    if n_pairs == 1:
        axes = [axes]

    for i in range(n_pairs):
        ax = axes[i]
        ax.set_facecolor("black")

        # Get aligned coords for both slices
        coords1 = aligned_slices[i].obsm.get("spatial", None)
        coords2 = aligned_slices[i + 1].obsm.get("spatial", None)

        if coords1 is None or coords2 is None:
            continue

        if isinstance(coords1, torch.Tensor):
            coords1 = coords1.cpu().numpy()
        if isinstance(coords2, torch.Tensor):
            coords2 = coords2.cpu().numpy()

        # Shift to image space
        coords1_shifted = coords1 + real_center
        coords2_shifted = coords2 + real_center

        # Subsample for plotting
        n1 = min(5000, len(coords1_shifted))
        n2 = min(5000, len(coords2_shifted))
        idx1 = np.random.choice(len(coords1_shifted), n1, replace=False)
        idx2 = np.random.choice(len(coords2_shifted), n2, replace=False)

        # Plot slice i in red, slice i+1 in cyan
        ax.scatter(
            coords1_shifted[idx1, 0],
            coords1_shifted[idx1, 1],
            s=1,
            c="red",
            alpha=0.5,
            label=f"Slice {i + 1}",
        )
        ax.scatter(
            coords2_shifted[idx2, 0],
            coords2_shifted[idx2, 1],
            s=1,
            c="cyan",
            alpha=0.5,
            label=f"Slice {i + 2}",
        )

        n1_name = (
            slice_names[i].split("__")[-1] if "__" in slice_names[i] else slice_names[i]
        )
        n2_name = (
            slice_names[i + 1].split("__")[-1]
            if "__" in slice_names[i + 1]
            else slice_names[i + 1]
        )
        ax.set_title(f"{n1_name} (red) vs {n2_name} (cyan)", fontsize=10)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # Match image coordinate system
        ax.axis("off")

    plt.suptitle(
        "Coordinate Alignment Composite (pairwise)", fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(
        output_dir / "alignment_coords_composite.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="black",
    )
    plt.close()
    typer.echo(f"   Saved: {output_dir / 'alignment_coords_composite.png'}")


def save_alignment_validation(
    aligned_slices,
    rotation_angles,
    translations,
    output_dir: Path,
    slice_names: list[str],
    original_slices: list = None,
):
    """
    Save validation figures showing alignment transformations.

    Creates:
    1. alignment_summary.png - Table with rotation angles and translations
    2. alignment_overlay.png - Overlaid slice centroids to verify alignment
    3. alignment_per_slice.png - Individual slice visualizations
    4. alignment_images.png - Raw images with cell points overlaid (if images exist)

    Args:
        original_slices: Original unaligned slices with uns['image'] data
    """
    import matplotlib.pyplot as plt
    import torch

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = FIGURES_DIR / "alignment_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract transformation values using helper
    n_slices = len(aligned_slices)
    transform_data = extract_paste3_transforms(rotation_angles, translations, n_slices)

    # Add slice names to transform_data
    for i, td in enumerate(transform_data):
        td["slice"] = slice_names[i] if i < len(slice_names) else f"slice_{i}"

    # 1. Create summary table figure
    fig, ax = plt.subplots(figsize=(12, max(4, len(transform_data) * 0.5)))
    ax.axis("off")

    table_data = [["Slice", "Rotation (°)", "Trans X", "Trans Y"]]
    for td in transform_data:
        table_data.append(
            [
                td["slice"].split("__")[-1]
                if "__" in td["slice"]
                else td["slice"],  # Shorten name
                f"{td['rotation_deg']:.2f}",
                f"{td['trans_x']:.1f}",
                f"{td['trans_y']:.1f}",
            ]
        )

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
        colWidths=[0.4, 0.2, 0.2, 0.2],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Header styling
    for j in range(4):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(4):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#D9E2F3")

    ax.set_title(
        "PASTE3 Alignment Transformations", fontsize=14, fontweight="bold", pad=20
    )
    plt.tight_layout()
    summary_path = output_dir / "alignment_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    typer.echo(f"   Saved alignment summary: {summary_path}")

    # 2. Create overlay figure showing aligned slice centroids/boundaries
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: Overlaid centroids (each slice as a different color)
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(aligned_slices)))

    for i, s in enumerate(aligned_slices):
        coords = s.obsm["spatial"]
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()

        # Subsample for plotting
        n_plot = min(2000, len(coords))
        idx = np.random.choice(len(coords), n_plot, replace=False)

        ax.scatter(
            coords[idx, 0],
            coords[idx, 1],
            c=[colors[i]],
            alpha=0.3,
            s=1,
            label=slice_names[i].split("__")[-1]
            if "__" in slice_names[i]
            else slice_names[i],
        )

    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title("Aligned Slice Overlay (subsampled)")
    ax.legend(loc="upper right", fontsize=8, markerscale=5)
    ax.set_aspect("equal")

    # Right: Centroid trajectory showing alignment quality
    ax2 = axes[1]
    centroids = []
    for _i, s in enumerate(aligned_slices):
        coords = s.obsm["spatial"]
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
        centroids.append((cx, cy))

    centroids = np.array(centroids)
    ax2.plot(
        centroids[:, 0],
        centroids[:, 1],
        "o-",
        color="#2ecc71",
        markersize=10,
        linewidth=2,
    )

    for i, (cx, cy) in enumerate(centroids):
        label = (
            slice_names[i].split("__")[-1] if "__" in slice_names[i] else slice_names[i]
        )
        ax2.annotate(
            label, (cx, cy), xytext=(5, 5), textcoords="offset points", fontsize=8
        )

    ax2.set_xlabel("X centroid (µm)")
    ax2.set_ylabel("Y centroid (µm)")
    ax2.set_title("Slice Centroid Trajectory\n(should be roughly aligned)")
    ax2.set_aspect("equal")

    # Add centroid drift stats
    x_drift = centroids[:, 0].max() - centroids[:, 0].min()
    y_drift = centroids[:, 1].max() - centroids[:, 1].min()
    ax2.text(
        0.02,
        0.98,
        f"X drift: {x_drift:.1f} µm\nY drift: {y_drift:.1f} µm",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    overlay_path = output_dir / "alignment_overlay.png"
    plt.savefig(overlay_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    typer.echo(f"   Saved alignment overlay: {overlay_path}")

    # 3. Create per-slice rotation visualization
    n_slices = len(aligned_slices)
    n_cols = min(4, n_slices)
    n_rows = (n_slices + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, s in enumerate(aligned_slices):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]

        coords = s.obsm["spatial"]
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()

        # Subsample for faster plotting
        n_plot = min(5000, len(coords))
        idx = np.random.choice(len(coords), n_plot, replace=False)

        ax.scatter(coords[idx, 0], coords[idx, 1], s=0.5, alpha=0.5, c="steelblue")

        # Add rotation info
        td = transform_data[i]
        short_name = (
            slice_names[i].split("__")[-1] if "__" in slice_names[i] else slice_names[i]
        )
        ax.set_title(
            f"{short_name}\nRot: {td['rotation_deg']:.1f}°, Trans: "
            f"({td['trans_x']:.0f}, {td['trans_y']:.0f})"
        )
        ax.set_aspect("equal")
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")

    # Hide empty subplots
    for i in range(n_slices, n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis("off")

    plt.suptitle("Individual Slice Alignments", fontsize=14, fontweight="bold")
    plt.tight_layout()
    slices_path = output_dir / "alignment_per_slice.png"
    plt.savefig(slices_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    typer.echo(f"   Saved per-slice alignment: {slices_path}")

    # 4. Create image visualization showing raw DAPI images
    # Use original_slices if provided (they have uns['image']), otherwise check
    # aligned_slices
    source_slices = original_slices if original_slices else aligned_slices
    has_images = all("image" in s.uns for s in source_slices)

    if has_images:
        typer.echo("   Creating raw image visualization...")
        n_slices = len(source_slices)

        # Create grid figure showing raw images (no coordinate overlay - different
        # spaces)
        n_cols = min(4, n_slices)
        n_rows = (n_slices + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)

        for i in range(n_slices):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]

            # Get image (use channel 0, typically DAPI) - display in native pixel space
            img = source_slices[i].uns["image"]
            if img.ndim == 3:
                img_display = img[0]  # Channel 0 = DAPI
            else:
                img_display = img

            td = transform_data[i]
            short_name = (
                slice_names[i].split("__")[-1]
                if "__" in slice_names[i]
                else slice_names[i]
            )

            # Display image in native pixel coordinates (no extent transformation)
            vmax = np.percentile(img_display, 99.5)
            ax.imshow(img_display, cmap="gray", vmin=0, vmax=vmax, origin="upper")
            ax.set_title(f"{short_name}\nShape: {img_display.shape}", fontsize=10)
            ax.axis("off")

        # Hide empty subplots
        for i in range(n_slices, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis("off")

        plt.suptitle("Raw DAPI Images (Channel 0)", fontsize=14, fontweight="bold")
        plt.tight_layout()
        images_path = output_dir / "alignment_images.png"
        plt.savefig(images_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close()
        typer.echo(f"   Saved raw images: {images_path}")
    else:
        typer.echo("   No images in uns['image'] - skipping image overlay")

    return output_dir


# =============================================================================
# PASTE3 ALIGNMENT
# =============================================================================


def apply_paste_transforms_to_full_slices(
    full_slices, subsampled_before, aligned_subsampled, rotation_angles=None
):
    """
    Apply PASTE3-learned transformations to full (non-subsampled) slices.

    Computes the cumulative rigid transformation by comparing subsampled
    coordinates before and after PASTE3 alignment (which already compounds
    all pairwise transforms), then applies the same transformation to full data.

    Args:
        full_slices: List of full AnnData objects (all cells)
        subsampled_before: List of subsampled AnnData objects BEFORE alignment
        aligned_subsampled: List of subsampled AnnData objects AFTER alignment
        rotation_angles: Unused (kept for API compatibility)

    Returns:
        List of transformed AnnData objects with aligned spatial coordinates
    """

    aligned_full = []

    for i, adata in enumerate(full_slices):
        adata = adata.copy()
        full_coords = adata.obsm["spatial"].copy()

        # Get before/after coordinates from subsample (same cells, different coords)
        coords_before = subsampled_before[i].obsm["spatial"]
        coords_after = aligned_subsampled[i].obsm["spatial"]

        # Convert torch tensors if needed
        if hasattr(coords_before, "cpu"):
            coords_before = coords_before.cpu().numpy()
        if hasattr(coords_after, "cpu"):
            coords_after = coords_after.cpu().numpy()

        # Sanity check: same number of cells
        if len(coords_before) != len(coords_after):
            typer.echo(
                "   ⚠️  ERROR: Cell count mismatch! "
                f"before={len(coords_before)}, after={len(coords_after)}"
            )
            continue

        # Debug: Check point-by-point displacement to verify cells are matched
        displacements = coords_after - coords_before
        disp_std = np.std(displacements, axis=0)
        disp_mean = np.mean(displacements, axis=0)
        typer.echo(
            f"   Slice {i + 1} debug: disp_mean={disp_mean}, disp_std={disp_std}"
        )

        # Compute cumulative rigid transformation using Procrustes/Kabsch
        # This finds R and t such that: after ≈ (before - c_before) @ R + c_after
        centroid_before = coords_before.mean(axis=0)
        centroid_after = coords_after.mean(axis=0)

        before_centered = coords_before - centroid_before
        after_centered = coords_after - centroid_after

        # SVD to find optimal rotation
        H = before_centered.T @ after_centered
        U, S, Vt = np.linalg.svd(H)

        # Check determinant before correction
        d = np.linalg.det(Vt.T @ U.T)

        # Ensure proper rotation (det = +1, no reflection)
        R = Vt.T @ np.diag([1.0, np.sign(d)]) @ U.T

        # Extract rotation angle for logging
        theta = np.arctan2(R[1, 0], R[0, 0])

        # Apply transformation: p' = (p - centroid_before) @ R + centroid_after
        full_centered = full_coords - centroid_before
        full_rotated = full_centered @ R
        full_transformed = full_rotated + centroid_after

        # Validate reconstruction on subsampled data
        sub_transformed = before_centered @ R + centroid_after
        recon_error = np.sqrt(np.mean((sub_transformed - coords_after) ** 2))

        # Log with Kabsch correction indicator
        correction_str = " [Kabsch corrected]" if d < 0 else ""
        typer.echo(
            f"   Slice {i + 1}: rot={np.degrees(theta):.2f}°, "
            f"err={recon_error:.2f}{correction_str}"
        )

        if recon_error > 10.0:
            typer.echo(f"   ⚠️  WARNING: High error! det={d:.4f}, singular values={S}")

        adata.obsm["spatial"] = full_transformed
        adata.obsm["spatial_aligned"] = full_transformed.copy()
        aligned_full.append(adata)

    return aligned_full


def apply_transforms_to_coordinates(full_slices, rotation_angles, translations):
    """
    Apply PASTE3 transforms to full (non-subsampled) coordinates.

    Transforms are not compounded. stack_slices_pairwise computes each R_i
    relative to the already-aligned source slice, so R_1 already accounts for R_0;
    each slice gets only its single pairwise transform.

    PASTE3's GPA transform: coords_aligned = (coords - weighted_target) @ R.T
    Result is centered at origin.

    Args:
        full_slices: List of AnnData objects with spatial coordinates
        rotation_angles: List of rotation angles from PASTE3 (len = n_slices - 1)
        translations: List of (source_centroid, target_centroid) tuples from PASTE3

    Returns:
        List of transformed AnnData objects
    """
    import torch

    def get_value(v):
        if isinstance(v, torch.Tensor):
            return v.cpu().numpy()
        return np.array(v).flatten()

    def get_rotation(rot):
        if isinstance(rot, torch.Tensor):
            if rot.numel() == 1:
                return float(rot.cpu().item())
            return float(rot.cpu().numpy()[0])
        return float(rot)

    aligned_full = []

    for i, adata in enumerate(full_slices):
        adata = adata.copy()
        coords = adata.obsm["spatial"].copy()

        if i == 0:
            # First slice: center it using the weighted source centroid from pair 0
            # This matches what PASTE3 does
            if len(translations) > 0:
                centroid = get_value(translations[0][0])
            else:
                centroid = coords.mean(axis=0)
            coords_aligned = coords - centroid
            typer.echo(f"   Slice {i + 1}: reference (centered)")
        else:
            # Slice i > 0: apply ONLY the single pairwise transform (i-1)
            # PASTE3's GPA: (target - weighted_target) @ R.T
            pair_idx = i - 1
            theta = get_rotation(rotation_angles[pair_idx])
            trans_target = get_value(translations[pair_idx][1])

            # Build rotation matrix
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

            # Apply PASTE3's transform exactly: (coords - weighted_target) @ R.T
            coords_centered = coords - trans_target
            coords_aligned = coords_centered @ R.T

            typer.echo(f"   Slice {i + 1}: rot={np.degrees(theta):.2f}°")

        adata.obsm["spatial"] = coords_aligned
        adata.obsm["spatial_aligned"] = coords_aligned.copy()
        aligned_full.append(adata)

    return aligned_full


def constrained_stack_slices_pairwise(
    slices, pis, max_rotation_deg=20.0, allow_90_rotation=False
):
    """
    Constrained version of stack_slices_pairwise that rejects bad alignments.

    Rotation handling:
    - |rotation| <= max_rotation_deg: Apply (valid small rotation)
    - If allow_90_rotation: Also allow rotations in 70-110° range (and negatives)
    - Otherwise: Reject and skip rotation (use translation only)

    Args:
        slices: List of AnnData objects
        pis: List of pi matrices from pairwise_align()
        max_rotation_deg: Maximum allowed rotation in degrees (default: 20.0)
        allow_90_rotation: Also allow rotations around ±90° (70-110°)

    Returns:
        Tuple of (aligned_slices, rotation_angles, translations)
    """
    import torch
    from paste3.visualization import generalized_procrustes_analysis

    def is_rotation_valid(angle_deg):
        """Check if rotation is within allowed ranges."""
        abs_angle = abs(angle_deg)
        # Small rotations always allowed
        if abs_angle <= max_rotation_deg:
            return True, "small rotation"
        # ~90° rotations if enabled
        if allow_90_rotation and 70 <= abs_angle <= 110:
            return True, "~90° rotation"
        return False, None

    assert len(slices) == len(pis) + 1, (
        "'slices' should have length one more than 'pis'"
    )
    assert len(slices) > 1, "You should have at least 2 layers"

    if allow_90_rotation:
        typer.echo(
            f"Constrained alignment with max rotation: ±{max_rotation_deg}° OR ±70-110°"
        )
    else:
        typer.echo(f"Constrained alignment with max rotation: ±{max_rotation_deg}°")

    aligned_coordinates = []
    rotation_angles = []
    translations = []

    # Process first pair (slices 0 and 1)
    typer.echo("   Processing slice pair 0-1...")
    (
        source_coordinates,
        target_coordinates,
        rotation_angle,
        x_translation,
        y_translation,
    ) = generalized_procrustes_analysis(
        torch.Tensor(slices[0].obsm["spatial"]).to(pis[0].dtype).to(pis[0].device),
        torch.Tensor(slices[1].obsm["spatial"]).to(pis[0].dtype).to(pis[0].device),
        pis[0],
        is_partial=False,
        allow_reflections=False,
    )

    # Check if rotation is reasonable
    rotation_angle_rad = float(
        rotation_angle.cpu().item()
        if hasattr(rotation_angle, "item")
        else rotation_angle
    )
    original_angle_deg = np.degrees(rotation_angle_rad)

    valid, reason = is_rotation_valid(original_angle_deg)
    if valid:
        typer.echo(f"      ✅ Rotation: {original_angle_deg:.2f}° ({reason})")
    else:
        typer.echo(
            f"      ❌ ALIGNMENT FAILURE: Rotation {original_angle_deg:.2f}° "
            "outside allowed ranges"
        )
        typer.echo("      → Skipping rotation, using translation only")
        rotation_angle = torch.tensor(
            0.0, dtype=rotation_angle.dtype, device=rotation_angle.device
        )
        target_coordinates = (
            torch.Tensor(slices[1].obsm["spatial"]).to(pis[0].dtype).to(pis[0].device)
        )
        weighted_target_skip = pis[0].sum(axis=0).matmul(target_coordinates)
        target_coordinates = target_coordinates - weighted_target_skip

    rotation_angles.append(rotation_angle)
    translations.append((x_translation, y_translation))
    aligned_coordinates.append(source_coordinates)
    aligned_coordinates.append(target_coordinates)

    # Process remaining slices
    for i in range(1, len(slices) - 1):
        typer.echo(f"   Processing slice pair {i}-{i + 1}...")
        (
            source_coordinates,
            target_coordinates,
            rotation_angle,
            x_translation,
            y_translation,
        ) = generalized_procrustes_analysis(
            aligned_coordinates[i],
            torch.Tensor(slices[i + 1].obsm["spatial"])
            .to(pis[i].dtype)
            .to(pis[i].device),
            pis[i],
            is_partial=False,
            allow_reflections=False,
        )

        # Check if rotation is reasonable
        rotation_angle_rad = float(
            rotation_angle.cpu().item()
            if hasattr(rotation_angle, "item")
            else rotation_angle
        )
        original_angle_deg = np.degrees(rotation_angle_rad)

        valid, reason = is_rotation_valid(original_angle_deg)
        if valid:
            typer.echo(f"      ✅ Rotation: {original_angle_deg:.2f}° ({reason})")
        else:
            typer.echo(
                "      ❌ ALIGNMENT FAILURE: "
                f"Rotation {original_angle_deg:.2f}° outside allowed ranges"
            )
            typer.echo("      → Skipping rotation, using translation only")
            rotation_angle = torch.tensor(
                0.0, dtype=rotation_angle.dtype, device=rotation_angle.device
            )
            target_coordinates = (
                torch.Tensor(slices[i + 1].obsm["spatial"])
                .to(pis[i].dtype)
                .to(pis[i].device)
            )
            weighted_target_skip = pis[i].sum(axis=0).matmul(target_coordinates)
            target_coordinates = target_coordinates - weighted_target_skip

        rotation_angles.append(rotation_angle)
        translations.append((x_translation, y_translation))
        aligned_coordinates.append(target_coordinates)

    # Create aligned slices
    new_slices = []
    for i in range(len(slices)):
        _slice = slices[i].copy()
        _slice.obsm["spatial"] = aligned_coordinates[i].cpu().numpy()
        new_slices.append(_slice)

    return new_slices, rotation_angles, translations


@app.command()
def align(
    output: str = typer.Option("aligned", help="Output filename prefix"),
    subsample: int = typer.Option(
        None, help="Cells per slice for alignment (None = use all)"
    ),
    n_top_genes: int = typer.Option(2000, help="Number of HVGs for alignment"),
    align_images: bool = typer.Option(False, help="Also align and save images"),
    max_rotation: float = typer.Option(
        20.0, help="Maximum allowed rotation per slice pair (degrees)"
    ),
    allow_90_rotation: bool = typer.Option(
        False, help="Also allow ~90° rotations (70-110° and -70 to -110°)"
    ),
    skip_image_viz: bool = typer.Option(
        False, help="Skip image visualization (saves memory)"
    ),
):
    """Run PASTE3 pairwise alignment on all slices.

    If subsample is set, runs PASTE3 on subsampled data to learn transforms,
    then applies those transforms to all cells and images (one pairwise
    transform per slice; see apply_transforms_to_coordinates).

    Rotations exceeding --max-rotation are rejected (alignment failure).

    Optionally aligns images if --align-images is set and images exist in uns['image'].
    """
    try:
        from paste3.paste import pairwise_align
    except ImportError as err:
        typer.echo("PASTE3 not installed. Install with: pip install paste3")
        raise typer.Exit(1) from err

    h5ad_files = [
        f
        for f in DATA_DIR.glob("*.h5ad")
        if "_clustered" not in f.name
        and "_aligned" not in f.name
        and "_labeled" not in f.name
        and not f.name.startswith("aligned")
        and not f.name.startswith("test_")
    ]
    h5ad_files = sorted(h5ad_files, key=get_slice_sort_key)

    if len(h5ad_files) < 2:
        typer.echo(f"Need at least 2 h5ad files, found {len(h5ad_files)}")
        raise typer.Exit(1)

    typer.echo(f"Loading {len(h5ad_files)} slices...")

    # Xenium pixel scale: 0.2125 µm per pixel
    # Convert coordinates from µm to pixels BEFORE PASTE (like datafication)
    PIXEL_SCALE = 0.2125

    full_slices = []  # Keep ALL cells
    subsampled_slices = []  # For PASTE3 alignment
    original_slices = []  # For image alignment (with images)
    has_images = True

    for i, f in enumerate(h5ad_files):
        adata = sc.read_h5ad(f)

        # Convert spatial coordinates from µm to pixels (like datafication)
        # This ensures PASTE outputs are in pixel space, matching images
        coords = adata.obsm["spatial"] / PIXEL_SCALE

        # The h5ad stores coords in global slide space but the image is a crop, so
        # subtract the crop origin (uns["crop_coords"], written by
        # convert_single_region.py) to get image-relative coords.
        if "crop_coords" in adata.uns:
            crop_min = np.array(adata.uns["crop_coords"]["min"])
            coords = coords - crop_min
            typer.echo(
                "      -> adjusted by crop offset "
                f"({crop_min[0]:.0f}, {crop_min[1]:.0f})"
            )

        adata.obsm["spatial"] = coords

        # Check for images
        if "image" not in adata.uns:
            has_images = False

        # Keep original for image alignment
        if has_images:
            original_slices.append(adata.copy())

        # Basic filtering
        sc.pp.filter_cells(adata, min_genes=10)
        sc.pp.filter_genes(adata, min_cells=3)

        # Store raw counts for Tangram compatibility
        if adata.raw is None:
            adata.raw = adata.copy()

        # Preprocessing for PASTE
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

        adata.obs["slice_name"] = f.stem
        # Authoritative slice-index z_level (10µm spacing) — overwrites any
        # per-cell Xenium z_level carried over from the raw table. `h5ad_files`
        # is already sorted in A-P order by `get_slice_sort_key`.
        adata.obs["z_level"] = float(i * 10)

        # Keep FULL slice before any subsampling
        full_slices.append(adata.copy())
        typer.echo(f"   {f.stem}: {adata.n_obs:,} cells")

        # Create subsampled copy for PASTE3 alignment
        if subsample and adata.n_obs > subsample:
            sc.pp.subsample(adata, n_obs=subsample)
            typer.echo(f"      -> subsampled to {adata.n_obs:,} for alignment")

        subsampled_slices.append(adata)

    if align_images and not has_images:
        typer.echo("⚠️  --align-images specified but no images found in uns['image']")
        align_images = False

    # Run PASTE pairwise alignment on subsampled data
    typer.echo(
        "\nRunning PASTE pairwise alignment "
        f"on {'subsampled' if subsample else 'full'} data..."
    )

    pis = []
    for i in range(len(subsampled_slices) - 1):
        typer.echo(f"   Aligning slices {i + 1} -> {i + 2}...")
        pi, _ = pairwise_align(subsampled_slices[i], subsampled_slices[i + 1])
        pis.append(pi)

    # Stack aligned slices to get rotation_angles and translations (with constraints)
    typer.echo("\nStacking aligned slices with rotation constraint...")
    aligned_subsampled, rotation_angles, translations = (
        constrained_stack_slices_pairwise(
            subsampled_slices,
            pis,
            max_rotation_deg=max_rotation,
            allow_90_rotation=allow_90_rotation,
        )
    )

    # Apply transforms to the full (non-subsampled) slices
    typer.echo("\nApplying transforms to all cells...")
    aligned_full = apply_transforms_to_coordinates(
        full_slices, rotation_angles, translations
    )

    # Convert torch tensors to numpy (if any remain)
    import torch

    for s in aligned_full:
        if isinstance(s.obsm["spatial"], torch.Tensor):
            s.obsm["spatial"] = s.obsm["spatial"].cpu().numpy()

    total_cells = sum(s.n_obs for s in aligned_full)
    typer.echo(f"   Aligned {total_cells:,} cells")

    # Save alignment validation figures
    typer.echo("\nSaving alignment validation figures...")
    slice_names = [f.stem for f in h5ad_files]
    save_alignment_validation(
        aligned_full,  # Use full slices for validation viz
        rotation_angles,
        translations,
        FIGURES_DIR,
        slice_names,
        original_slices=original_slices if original_slices else None,
    )
    save_coordinate_alignment_composite(aligned_full, slice_names, translations)

    # Transform images for validation (if available)
    if has_images and original_slices:
        typer.echo("\nTransforming images for validation...")
        result = align_images_with_paste(
            aligned_full, original_slices, rotation_angles, translations
        )
        if result:
            transformed_images, img_padding = result
            save_image_alignment_validation(
                original_slices,
                transformed_images,
                rotation_angles,
                translations,
                slice_names,
                aligned_slices=aligned_full,
                padding=img_padding,
            )
            save_image_alignment_composite(transformed_images, slice_names)

            if align_images:
                output_dir = FIGURES_DIR / "alignment_validation"
                output_dir.mkdir(parents=True, exist_ok=True)
                save_aligned_images(
                    transformed_images, output_dir / f"{output}_images.tif"
                )
                typer.echo(
                    f"   Saved aligned images to {output_dir / f'{output}_images.tif'}"
                )

    # Collect and preserve all label columns (in_*)
    all_label_cols = set()
    for s in aligned_full:
        label_cols = [c for c in s.obs.columns if c.startswith("in_")]
        all_label_cols.update(label_cols)

    if all_label_cols:
        typer.echo(f"   Preserving {len(all_label_cols)} anatomical label columns")
        for s in aligned_full:
            for col in all_label_cols:
                if col not in s.obs.columns:
                    s.obs[col] = False

    # Combine FULL aligned slices and save
    combined = ad.concat(aligned_full, axis=0, join="inner")
    combined.obs_names_make_unique()

    total_cells = combined.n_obs
    output_path = DATA_DIR / f"{output}.h5ad"
    combined.write_h5ad(output_path)
    typer.echo(f"\n✅ Saved: {output_path} ({total_cells:,} cells)")
    typer.echo("Ready for: cluster, visualize-3d-regions, Tangram, etc.")


# =============================================================================
# 3D VISUALIZATION
# =============================================================================


@app.command("visualize-3d")
def visualize_3d(
    input_file: str = typer.Option("aligned_clustered", help="Input h5ad file"),
    color_by: str = typer.Option("leiden", help="Column to color by"),
    z_spacing: float = typer.Option(
        10.0, help="Z spacing between slices (µm, physical=10)"
    ),
    point_size: float = typer.Option(2.0, help="Point size"),
    output: str = typer.Option("brain_3d", help="Output filename"),
):
    """Create interactive 3D rotating brain visualization."""
    input_path = DATA_DIR / f"{input_file}.h5ad"
    if not input_path.exists():
        typer.echo(f"❌ File not found: {input_path}")
        typer.echo("   Run 'cluster' first to create aligned_clustered.h5ad")
        raise typer.Exit(1)

    typer.echo(f"📂 Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)
    typer.echo(f"   Cells: {adata.n_obs}")

    # Get unique slices from slice_name column
    if "slice_name" not in adata.obs:
        typer.echo("❌ No 'slice_name' column found. Data may not be from alignment.")
        raise typer.Exit(1)

    slice_names = adata.obs["slice_name"].unique()
    typer.echo(f"   Slices: {len(slice_names)}")

    fig = go.Figure()

    # Color palette
    n_colors = 40
    colors = [f"hsl({i * 360 / n_colors}, 70%, 50%)" for i in range(n_colors)]

    for i, slice_name in enumerate(slice_names):
        slice_mask = adata.obs["slice_name"] == slice_name
        slice_adata = adata[slice_mask]
        coords = slice_adata.obsm["spatial"]
        z_coord = np.full(len(coords), i * z_spacing)

        if color_by in slice_adata.obs:
            for cluster in adata.obs[color_by].unique():
                cluster_mask = slice_adata.obs[color_by] == cluster
                if cluster_mask.sum() == 0:
                    continue
                cluster_idx = (
                    int(cluster) % n_colors
                    if str(cluster).isdigit()
                    else hash(cluster) % n_colors
                )

                fig.add_trace(
                    go.Scatter3d(
                        x=coords[cluster_mask, 0],
                        y=coords[cluster_mask, 1],
                        z=z_coord[cluster_mask],
                        mode="markers",
                        marker=dict(
                            size=point_size, color=colors[cluster_idx], opacity=0.7
                        ),
                        name=f"Cluster {cluster}",
                        legendgroup=f"cluster_{cluster}",
                        showlegend=(i == 0),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=z_coord,
                    mode="markers",
                    marker=dict(
                        size=point_size, color=colors[i % n_colors], opacity=0.7
                    ),
                    name=slice_name,
                )
            )

    # Layout with rotation
    fig.update_layout(
        title="3D Brain Reconstruction",
        scene=dict(
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            zaxis_title="Z (slice)",
            aspectmode="data",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.1,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="▶ Rotate",
                        method="animate",
                        args=[None, {"frame": {"duration": 50}, "fromcurrent": True}],
                    ),
                    dict(
                        label="⏸ Stop",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
    )

    # Rotation frames
    frames = [
        go.Frame(
            layout=dict(
                scene_camera=dict(
                    eye=dict(
                        x=1.5 * np.cos(np.radians(a)),
                        y=1.5 * np.sin(np.radians(a)),
                        z=0.5,
                    )
                )
            )
        )
        for a in range(0, 360, 5)
    ]
    fig.frames = frames

    output_path = FIGURES_DIR / f"{output}.html"
    fig.write_html(output_path)
    typer.echo(f"✅ Saved 3D visualization: {output_path}")


@app.command("visualize-3d-regions")
def visualize_3d_regions(
    input_file: str = typer.Option("aligned", help="Input h5ad file"),
    z_spacing: float = typer.Option(
        10.0, help="Z spacing between slices (µm, physical=10)"
    ),
    point_size: float = typer.Option(2.0, help="Point size"),
    output: str = typer.Option("brain_3d_regions", help="Output filename"),
    show_unlabeled: bool = typer.Option(True, help="Show unlabeled cells in gray"),
):
    """Create interactive 3D visualization colored by anatomical regions."""
    input_path = DATA_DIR / f"{input_file}.h5ad"
    if not input_path.exists():
        typer.echo(f"File not found: {input_path}")
        typer.echo("   Run 'cluster' first to create aligned.h5ad")
        raise typer.Exit(1)

    typer.echo(f"Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)
    typer.echo(f"   Cells: {adata.n_obs}")

    # Find region columns
    region_cols = [c for c in adata.obs.columns if c.startswith("in_")]
    if not region_cols:
        typer.echo("No anatomical region labels found (columns starting with 'in_')")
        typer.echo("   Run 'label-all' first to add anatomical labels")
        raise typer.Exit(1)

    region_names = [c.replace("in_", "") for c in region_cols]
    typer.echo(f"   Regions: {', '.join(region_names)}")

    # Get unique slices
    if "slice_name" not in adata.obs:
        typer.echo("No 'slice_name' column found. Data may not be from alignment.")
        raise typer.Exit(1)

    slice_names = adata.obs["slice_name"].unique()
    typer.echo(f"   Slices: {len(slice_names)}")

    fig = go.Figure()

    # Region color palette (distinct colors)
    region_colors = {
        "lamina": "#59A14F",
        "medulla": "#4E79A7",
        "lobula": "#E15759",
        "kc_oc": "#EDC948",
        "kc_ic": "#B07AA1",
        "kc_nc": "#F28E2B",
    }
    default_colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6", "#1ABC9C"]

    for i, slice_name in enumerate(sorted(slice_names)):
        slice_mask = adata.obs["slice_name"] == slice_name
        slice_adata = adata[slice_mask]
        coords = slice_adata.obsm["spatial"]
        z_coord = np.full(len(coords), i * z_spacing)

        # Track which cells have been plotted
        plotted = np.zeros(len(coords), dtype=bool)

        # Plot each region
        for j, region in enumerate(region_names):
            col = f"in_{region}"
            if col not in slice_adata.obs:
                continue

            region_mask = slice_adata.obs[col].values
            if region_mask.sum() == 0:
                continue

            color = region_colors.get(region, default_colors[j % len(default_colors)])
            plotted |= region_mask

            fig.add_trace(
                go.Scatter3d(
                    x=coords[region_mask, 0],
                    y=coords[region_mask, 1],
                    z=z_coord[region_mask],
                    mode="markers",
                    marker=dict(size=point_size, color=color, opacity=0.8),
                    name=region,
                    legendgroup=region,
                    showlegend=(i == 0),
                )
            )

        # Plot unlabeled cells
        if show_unlabeled:
            unlabeled = ~plotted
            if unlabeled.sum() > 0:
                fig.add_trace(
                    go.Scatter3d(
                        x=coords[unlabeled, 0],
                        y=coords[unlabeled, 1],
                        z=z_coord[unlabeled],
                        mode="markers",
                        marker=dict(
                            size=point_size * 0.5, color="lightgray", opacity=0.2
                        ),
                        name="unlabeled",
                        legendgroup="unlabeled",
                        showlegend=(i == 0),
                    )
                )

    # Layout with rotation
    fig.update_layout(
        title="3D Brain - Anatomical Regions",
        scene=dict(
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            zaxis_title="Z (slice)",
            aspectmode="data",
        ),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.1,
                x=0.5,
                xanchor="center",
                buttons=[
                    dict(
                        label="Rotate",
                        method="animate",
                        args=[None, {"frame": {"duration": 50}, "fromcurrent": True}],
                    ),
                    dict(
                        label="Stop",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    ),
                ],
            )
        ],
    )

    # Rotation frames
    frames = [
        go.Frame(
            layout=dict(
                scene_camera=dict(
                    eye=dict(
                        x=1.5 * np.cos(np.radians(a)),
                        y=1.5 * np.sin(np.radians(a)),
                        z=0.5,
                    )
                )
            )
        )
        for a in range(0, 360, 5)
    ]
    fig.frames = frames

    output_path = FIGURES_DIR / f"{output}.html"
    fig.write_html(output_path)
    typer.echo(f"Saved 3D regions visualization: {output_path}")

    # Stats
    typer.echo("\nRegion cell counts:")
    for region in region_names:
        col = f"in_{region}"
        if col in adata.obs:
            count = adata.obs[col].sum()
            pct = 100 * count / adata.n_obs
            typer.echo(f"  {region}: {count:,} cells ({pct:.1f}%)")


# =============================================================================
# VIDEO RENDERING
# =============================================================================


@app.command("render-video")
def render_video(
    input_file: str = typer.Option("aligned", help="Input h5ad file"),
    color_by: str = typer.Option(
        "leiden", help="Column to color by, or 'regions' for anatomical regions"
    ),
    z_spacing: float = typer.Option(
        10.0, help="Z spacing between slices (µm, physical=10)"
    ),
    z_exaggerate: float = typer.Option(
        1.0, help="Z exaggeration factor for visualization"
    ),
    point_size: float = typer.Option(5.0, help="Point/dot size (try 3-10)"),
    elevation: float = typer.Option(15.0, help="Camera elevation angle (degrees)"),
    zoom: float = typer.Option(2, help="Zoom factor (higher = closer)"),
    fps: int = typer.Option(30, help="Frames per second"),
    duration: float = typer.Option(10.0, help="Video duration in seconds"),
    output: str = typer.Option("brain_3d", help="Output filename (without extension)"),
    dpi: int = typer.Option(150, help="Resolution (DPI)"),
):
    """Render high-quality rotating 3D brain video."""
    import shutil
    import subprocess

    import matplotlib.pyplot as plt

    input_path = DATA_DIR / f"{input_file}.h5ad"
    if not input_path.exists():
        typer.echo(f"❌ File not found: {input_path}")
        raise typer.Exit(1)

    typer.echo(f"📂 Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)

    if "slice_name" not in adata.obs:
        typer.echo("❌ No 'slice_name' column found.")
        raise typer.Exit(1)

    slice_names = adata.obs["slice_name"].unique()
    typer.echo(f"   Cells: {adata.n_obs}, Slices: {len(slice_names)}")

    # Prepare coordinates with Z
    coords_list = []
    colors_list = []

    # Region color palette (for anatomical regions)
    region_colors = {
        "lamina": (1.0, 0.42, 0.42, 1.0),  # red
        "medulla": (0.31, 0.80, 0.77, 1.0),  # teal
        "lobula": (0.27, 0.72, 0.82, 1.0),  # blue
        "pcb": (0.59, 0.81, 0.70, 1.0),  # green
        "fsb": (1.0, 0.92, 0.65, 1.0),  # yellow
        "cc": (0.87, 0.63, 0.87, 1.0),  # plum
        "peduncle": (0.60, 0.85, 0.78, 1.0),  # mint
        "protocerebrum": (0.97, 0.86, 0.44, 1.0),  # gold
        "modula": (0.73, 0.56, 0.81, 1.0),  # purple
    }
    unlabeled_color = (0.3, 0.3, 0.3, 0.1)  # gray, super transparent

    # Color map for categorical columns
    if color_by == "regions":
        # Find region columns
        region_cols = [c for c in adata.obs.columns if c.startswith("in_")]
        if not region_cols:
            typer.echo(
                "❌ No anatomical region labels found (columns starting with 'in_')"
            )
            raise typer.Exit(1)
        typer.echo(
            f"   Coloring by regions: {[c.replace('in_', '') for c in region_cols]}"
        )
    elif color_by in adata.obs:
        # Sort categories numerically if possible for consistent coloring
        categories = adata.obs[color_by].astype(str).unique()
        categories = sorted(categories, key=lambda x: int(x) if x.isdigit() else 999)
        cmap = plt.cm.get_cmap("tab20", len(categories))
        color_map = {cat: cmap(i) for i, cat in enumerate(categories)}

    for i, slice_name in enumerate(sorted(slice_names)):
        mask = adata.obs["slice_name"] == slice_name
        slice_data = adata[mask]
        coords = slice_data.obsm["spatial"]
        z = np.full(len(coords), i * z_spacing)
        coords_list.append(np.column_stack([coords, z]))

        if color_by == "regions":
            # Assign colors based on anatomical regions
            slice_colors = []
            for cell_idx in range(len(slice_data)):
                cell_color = unlabeled_color
                for col in region_cols:
                    if slice_data.obs[col].iloc[cell_idx]:
                        region_name = col.replace("in_", "")
                        cell_color = region_colors.get(
                            region_name, (0.9, 0.3, 0.3, 1.0)
                        )
                        break
                slice_colors.append(cell_color)
            colors_list.extend(slice_colors)
        elif color_by in adata.obs:
            colors_list.extend([color_map[str(c)] for c in slice_data.obs[color_by]])
        else:
            cmap = plt.cm.get_cmap("tab20", 20)
            colors_list.extend([cmap(i % 20)] * len(coords))

    all_coords = np.vstack(coords_list)
    all_colors = np.array(colors_list)

    # Subsample if too many points
    max_points = 500000
    if len(all_coords) > max_points:
        typer.echo(
            f"   Subsampling {len(all_coords)} → {max_points} points for rendering..."
        )
        idx = np.random.choice(len(all_coords), max_points, replace=False)
        all_coords = all_coords[idx]
        all_colors = all_colors[idx]

    # Create frames directory
    frames_dir = FIGURES_DIR / "frames"
    frames_dir.mkdir(exist_ok=True)

    n_frames = int(fps * duration)
    typer.echo(f"🎬 Rendering {n_frames} frames at {dpi} DPI...")

    # Compute ranges for aspect ratio
    x_range = all_coords[:, 0].max() - all_coords[:, 0].min()
    y_range = all_coords[:, 1].max() - all_coords[:, 1].min()
    z_range = all_coords[:, 2].max() - all_coords[:, 2].min()

    # Apply Z exaggeration for visualization (brain is very flat)
    z_range_viz = z_range * z_exaggerate
    typer.echo(
        f"   Dimensions: {x_range:.0f} × {y_range:.0f} × {z_range:.0f} µm "
        f"(Z×{z_exaggerate})"
    )

    # Compute zoom-based margins (negative margins = zoom in, extend plot beyond figure)
    # zoom=1.0 means no zoom, zoom=2.0 means 2x closer
    zoom_margin = (1 - zoom) / (2 * zoom)  # e.g., zoom=2 -> margin=-0.25

    for frame_idx in range(n_frames):
        angle = (frame_idx / n_frames) * 360

        # Wider figure for landscape brain - fill the frame
        fig = plt.figure(figsize=(16, 9), facecolor="black")
        ax = fig.add_subplot(111, projection="3d", facecolor="black")

        # Use negative margins to zoom in (plot extends beyond figure bounds)
        fig.subplots_adjust(
            left=zoom_margin,
            right=1 - zoom_margin,
            bottom=zoom_margin,
            top=1 - zoom_margin,
        )

        ax.scatter(
            all_coords[:, 0],
            all_coords[:, 1],
            all_coords[:, 2],
            c=all_colors,
            s=point_size,
            edgecolors="none",
        )

        # Exaggerate Z for visualization
        ax.set_box_aspect([x_range, y_range, z_range_viz])

        # Set view angle (lower elevation to see layers)
        ax.view_init(elev=elevation, azim=angle)

        # Set tight axis limits (no extra margin)
        ax.set_xlim(all_coords[:, 0].min(), all_coords[:, 0].max())
        ax.set_ylim(all_coords[:, 1].min(), all_coords[:, 1].max())
        ax.set_zlim(all_coords[:, 2].min(), all_coords[:, 2].max())

        # Clean up axes for video
        ax.set_axis_off()
        ax.grid(False)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor("none")
        ax.yaxis.pane.set_edgecolor("none")
        ax.zaxis.pane.set_edgecolor("none")

        # Save frame (no tight crop to maintain 16:9 aspect)
        frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
        plt.savefig(frame_path, dpi=dpi, facecolor="black")
        plt.close()

        if (frame_idx + 1) % 30 == 0:
            typer.echo(f"   Frame {frame_idx + 1}/{n_frames}")

    # Combine frames with ffmpeg
    output_path = FIGURES_DIR / f"{output}.mp4"
    typer.echo("🎥 Encoding video with ffmpeg...")

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Ensure even dimensions for libx264
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "18",  # High quality
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        typer.echo(f"❌ ffmpeg error: {result.stderr}")
        raise typer.Exit(1)

    # Cleanup frames
    shutil.rmtree(frames_dir)

    typer.echo(f"✅ Saved video: {output_path}")


# =============================================================================
# ANATOMICAL LABELING
# =============================================================================

ANNOTATIONS_DIR = annotations_dir(BRAIN_DIR)


@app.command("list-annotations")
def list_annotations():
    """List available GeoJSON annotation files."""
    annotations = discover_annotations()
    if not annotations:
        typer.echo(f"No annotation files found in {ANNOTATIONS_DIR}")
        return

    for ann in annotations:
        info = parse_annotation_filename(ann)
        typer.echo(f"  {ann.name}")
        if info:
            typer.echo(f"    Slide {info['slide']}, Region {info['region']}")

        regions = load_geojson_regions(ann)
        typer.echo(f"    Regions: {', '.join(regions.keys())}")


@app.command("label")
def label_slice(
    input_file: str = typer.Argument(..., help="Input h5ad file (in data/)"),
    annotation: str = typer.Argument(
        ..., help="GeoJSON annotation file (in data/annotations/<brain>/v1/)"
    ),
    output: Optional[str] = typer.Option(
        None, help="Output filename (default: input_labeled.h5ad)"
    ),
    pixel_scale: float = typer.Option(
        0.2125, help="Scale factor to convert annotation pixels to µm (Xenium: 0.2125)"
    ),
    x_offset: float = typer.Option(
        0.0, help="X offset to add to cell coords (for coord system alignment)"
    ),
    y_offset: float = typer.Option(
        0.0, help="Y offset to add to cell coords (for coord system alignment)"
    ),
):
    """Add anatomical region labels to a slice using GeoJSON annotations.

    GeoJSON annotations from QuPath/Napari are typically in pixels. Use --pixel-scale
    to convert to microns (default: 0.2125 µm/pixel for Xenium data).
    """
    input_path = (
        DATA_DIR / input_file
        if not Path(input_file).is_absolute()
        else Path(input_file)
    )
    ann_path = (
        ANNOTATIONS_DIR / annotation
        if not Path(annotation).is_absolute()
        else Path(annotation)
    )

    if not input_path.exists():
        typer.echo(f"File not found: {input_path}")
        raise typer.Exit(1)

    if not ann_path.exists():
        typer.echo(f"Annotation not found: {ann_path}")
        raise typer.Exit(1)

    typer.echo(f"Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)
    typer.echo(f"  Cells: {adata.n_obs}")
    typer.echo(
        f"  Cell X range: {adata.obsm['spatial'][:, 0].min():.1f} "
        f"- {adata.obsm['spatial'][:, 0].max():.1f}"
    )
    typer.echo(
        f"  Cell Y range: {adata.obsm['spatial'][:, 1].min():.1f} "
        f"- {adata.obsm['spatial'][:, 1].max():.1f}"
    )

    typer.echo(f"Loading annotations from {ann_path.name}...")
    typer.echo(f"  Pixel scale: {pixel_scale} µm/pixel")
    regions = load_geojson_regions(ann_path, pixel_scale=pixel_scale)
    typer.echo(f"  Found {len(regions)} regions: {list(regions.keys())}")

    # Show annotation bounds in µm
    all_bounds = [
        (
            r["polygon"].bounds[0],
            r["polygon"].bounds[1],
            r["polygon"].bounds[2],
            r["polygon"].bounds[3],
        )
        for r in regions.values()
    ]
    min_x = min(b[0] for b in all_bounds)
    min_y = min(b[1] for b in all_bounds)
    max_x = max(b[2] for b in all_bounds)
    max_y = max(b[3] for b in all_bounds)
    typer.echo(f"  Annotation X range: {min_x:.1f} - {max_x:.1f} µm")
    typer.echo(f"  Annotation Y range: {min_y:.1f} - {max_y:.1f} µm")

    coord_offset = (x_offset, y_offset) if (x_offset != 0 or y_offset != 0) else None
    if coord_offset:
        typer.echo(f"  Using coordinate offset: ({x_offset}, {y_offset})")

    typer.echo("Assigning cells to regions...")
    adata = add_labels_to_adata(adata, regions, coord_offset=coord_offset)

    # Report statistics
    total_labeled = 0
    for region_name in regions:
        col = f"in_{region_name}"
        if col in adata.obs:
            n_cells = adata.obs[col].sum()
            pct = 100 * n_cells / adata.n_obs
            typer.echo(f"  {region_name}: {n_cells:,} cells ({pct:.1f}%)")
            total_labeled += n_cells

    typer.echo(f"  Total labeled: {total_labeled:,} cells")

    # Save
    if output:
        output_path = (
            DATA_DIR / output if not Path(output).is_absolute() else Path(output)
        )
    else:
        output_path = DATA_DIR / f"{input_path.stem}_labeled.h5ad"

    adata.write_h5ad(output_path)
    typer.echo(f"Saved: {output_path}")


@app.command("label-all")
def label_all_slices(
    pixel_scale: float = typer.Option(
        0.2125, help="Scale factor to convert annotation pixels to µm"
    ),
    inplace: bool = typer.Option(
        True, help="Overwrite source h5ad files (vs creating _labeled copies)"
    ),
    dry_run: bool = typer.Option(
        False, help="Show what would be labeled without actually doing it"
    ),
):
    """Label all h5ad files using the ANNOTATION_MAPPING in the brain's labeling.py.

    Edit ANNOTATION_MAPPING in the brain's labeling.py to specify which annotation files
    map to which crop suffixes (e.g., "2_7A", "2_7B").

    After labeling, run 'align' to create aligned data with labels preserved.
    """
    from labeling_core import ANNOTATION_MAPPING, discover_annotations

    annotations = discover_annotations()
    if not annotations:
        typer.echo(f"No annotation files found in {ANNOTATIONS_DIR}")
        raise typer.Exit(1)

    # Find all h5ad files (exclude derived files)
    h5ad_files = list(DATA_DIR.glob("*.h5ad"))
    h5ad_files = [
        f
        for f in h5ad_files
        if "_clustered" not in f.name
        and "_aligned" not in f.name
        and "_labeled" not in f.name
        and not f.name.startswith("aligned")
        and not f.name.startswith("test_")
    ]

    typer.echo(f"Found {len(annotations)} annotations and {len(h5ad_files)} h5ad files")
    typer.echo(f"ANNOTATION_MAPPING has {len(ANNOTATION_MAPPING)} entries\n")

    labeled_count = 0
    for ann in annotations:
        typer.echo(f"Annotation: {ann.name}")

        # Check if annotation is in mapping
        if ann.name not in ANNOTATION_MAPPING:
            typer.echo("  Not in ANNOTATION_MAPPING - skipping")
            typer.echo(f'  Add to labeling.py: "{ann.name}": ["crop_suffix"]')
            continue

        crop_suffixes = ANNOTATION_MAPPING[ann.name]
        typer.echo(f"  Mapped to crops: {crop_suffixes}")

        # Extract slide ID from annotation filename if present (e.g., "0040240" from the
        # name)
        import re

        slide_id_match = re.search(r"__(\d{7})__", ann.name)
        ann_slide_id = slide_id_match.group(1) if slide_id_match else None
        if ann_slide_id:
            typer.echo(f"  Annotation slide ID: {ann_slide_id}")

        # Find matching h5ad files by crop suffix AND slide ID (if available)
        matching_files = []
        for suffix in crop_suffixes:
            for h5ad_file in h5ad_files:
                if h5ad_file.stem.endswith("_" + suffix):
                    # If annotation has slide ID, also check h5ad has same slide ID
                    if ann_slide_id:
                        if f"__{ann_slide_id}__" in h5ad_file.name:
                            matching_files.append(h5ad_file)
                    else:
                        matching_files.append(h5ad_file)

        if not matching_files:
            typer.echo(f"  No matching h5ad files found for suffixes: {crop_suffixes}")
            continue

        # Load regions
        regions = load_geojson_regions(ann, pixel_scale=pixel_scale)
        typer.echo(f"  Loaded {len(regions)} regions: {list(regions.keys())}")

        for h5ad_file in matching_files:
            typer.echo(f"  Labeling: {h5ad_file.name}")

            if dry_run:
                typer.echo(f"    [DRY RUN] Would label with {len(regions)} regions")
                continue

            adata = sc.read_h5ad(h5ad_file)
            adata = add_labels_to_adata(adata, regions)

            # Report stats
            for region_name in regions:
                col = f"in_{region_name}"
                if col in adata.obs:
                    n_cells = adata.obs[col].sum()
                    pct = 100 * n_cells / adata.n_obs
                    typer.echo(f"    {region_name}: {n_cells:,} cells ({pct:.1f}%)")

            # Extract suffix for figure naming
            suffix = h5ad_file.stem.split("_")[-1]  # e.g., "2_1A" or "8A"

            # Save figure (cropped to cell bounding box)
            fig_dir = FIGURES_DIR / "annotations"
            fig_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fig_dir / f"labeling_{suffix}.png"
            plot_labels(adata, regions, output_path=fig_path, title=suffix)

            # Save h5ad
            if inplace:
                output_path = h5ad_file
            else:
                output_path = DATA_DIR / f"{h5ad_file.stem}_labeled.h5ad"

            adata.write_h5ad(output_path)
            typer.echo(f"    Saved: {output_path.name}")
            labeled_count += 1

    typer.echo(f"\nLabeled {labeled_count} h5ad files")
    if labeled_count > 0 and not dry_run:
        typer.echo("Run 'align' to create aligned data with labels preserved")


@app.command("propagate-labels")
def propagate_labels(
    input_file: str = typer.Option(
        "aligned_clustered", help="Aligned h5ad file with slice_name column"
    ),
    source_slice: str = typer.Option(..., help="Name of the labeled source slice"),
    annotation: str = typer.Option(
        ..., help="GeoJSON annotation file for source slice"
    ),
    pixel_scale: float = typer.Option(
        0.2125, help="Scale factor to convert annotation pixels to µm"
    ),
    output: Optional[str] = typer.Option(None, help="Output filename"),
):
    """Propagate anatomical labels from a source slice to all aligned slices."""
    input_path = DATA_DIR / f"{input_file}.h5ad"
    ann_path = (
        ANNOTATIONS_DIR / annotation
        if not Path(annotation).is_absolute()
        else Path(annotation)
    )

    if not input_path.exists():
        typer.echo(f"File not found: {input_path}")
        raise typer.Exit(1)

    if not ann_path.exists():
        typer.echo(f"Annotation not found: {ann_path}")
        raise typer.Exit(1)

    typer.echo(f"Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)

    if "slice_name" not in adata.obs:
        typer.echo("No 'slice_name' column - data must be from alignment")
        raise typer.Exit(1)

    slice_names = adata.obs["slice_name"].unique().tolist()
    typer.echo(f"  Found {len(slice_names)} slices")

    # Load regions
    regions = load_geojson_regions(ann_path, pixel_scale=pixel_scale)
    typer.echo(f"  Loaded {len(regions)} regions from annotation")

    # First label the source slice
    source_mask = adata.obs["slice_name"] == source_slice
    if source_mask.sum() == 0:
        typer.echo(f"Source slice '{source_slice}' not found in data")
        typer.echo(f"Available slices: {slice_names[:5]}...")
        raise typer.Exit(1)

    typer.echo(f"Labeling source slice: {source_slice}")
    source_adata = adata[source_mask].copy()
    source_adata = add_labels_to_adata(source_adata, regions)

    # Copy labels back to main adata
    for col in source_adata.obs.columns:
        if col.startswith("in_"):
            adata.obs[col] = False  # Initialize
            adata.obs.loc[source_mask, col] = source_adata.obs[col].values

    # Propagate to neighboring slices
    target_slices = [s for s in slice_names if s != source_slice]
    typer.echo(f"Propagating to {len(target_slices)} neighboring slices...")

    adata = propagate_labels_to_neighbors(
        adata,
        source_slice=source_slice,
        target_slices=target_slices,
        regions=regions,
    )

    # Report
    for region_name in regions:
        col = f"in_{region_name}"
        if col in adata.obs:
            n_cells = adata.obs[col].sum()
            typer.echo(f"  {region_name}: {n_cells:,} cells total")

    # Save
    if output:
        output_path = (
            DATA_DIR / output if not Path(output).is_absolute() else Path(output)
        )
    else:
        output_path = DATA_DIR / f"{input_file}_labeled.h5ad"

    adata.write_h5ad(output_path)
    typer.echo(f"Saved: {output_path}")


@app.command("visualize-labels")
def visualize_labels(
    input_file: str = typer.Option(
        "aligned_clustered_labeled", help="Labeled h5ad file"
    ),
    region: Optional[str] = typer.Option(
        None, help="Specific region to highlight (default: all)"
    ),
    output: str = typer.Option("brain_3d_labeled", help="Output filename"),
    z_spacing: float = typer.Option(10.0, help="Z spacing between slices"),
    point_size: float = typer.Option(2.0, help="Point size"),
):
    """Create 3D visualization with anatomical region coloring."""
    input_path = DATA_DIR / f"{input_file}.h5ad"
    if not input_path.exists():
        typer.echo(f"File not found: {input_path}")
        raise typer.Exit(1)

    typer.echo(f"Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)

    # Get region columns
    region_cols = [c for c in adata.obs.columns if c.startswith("in_")]
    if not region_cols:
        typer.echo("No region labels found (columns starting with 'in_')")
        raise typer.Exit(1)

    typer.echo(f"  Found {len(region_cols)} region labels")

    # Get region colors from uns if available
    region_colors = adata.uns.get("region_colors", {})

    slice_names = adata.obs["slice_name"].unique()
    fig = go.Figure()

    # Default colors if not in uns
    default_colors = [
        "#d2c800",
        "#21efa5",
        "#7cec37",
        "#43bcE7",
        "#fe9a30",
        "#ff6464",
        "#459bff",
        "#a177ff",
    ]

    for i, slice_name in enumerate(slice_names):
        slice_mask = adata.obs["slice_name"] == slice_name
        slice_adata = adata[slice_mask]
        coords = slice_adata.obsm["spatial"]
        z_coord = np.full(len(coords), i * z_spacing)

        if region:
            # Show specific region
            col = f"in_{region}"
            if col not in adata.obs.columns:
                typer.echo(f"Region '{region}' not found")
                raise typer.Exit(1)

            in_region = slice_adata.obs[col].values
            color = region_colors.get(region, "#ff0000")

            # Cells in region
            fig.add_trace(
                go.Scatter3d(
                    x=coords[in_region, 0],
                    y=coords[in_region, 1],
                    z=z_coord[in_region],
                    mode="markers",
                    marker=dict(size=point_size, color=color, opacity=0.8),
                    name=region,
                    legendgroup=region,
                    showlegend=(i == 0),
                )
            )

            # Other cells (gray)
            fig.add_trace(
                go.Scatter3d(
                    x=coords[~in_region, 0],
                    y=coords[~in_region, 1],
                    z=z_coord[~in_region],
                    mode="markers",
                    marker=dict(size=point_size * 0.5, color="#888888", opacity=0.2),
                    name="Other",
                    legendgroup="other",
                    showlegend=(i == 0),
                )
            )
        else:
            # Color by all regions
            # Assign each cell to its region (first match wins)
            cell_regions = ["unlabeled"] * len(coords)
            for col in region_cols:
                region_name = col.replace("in_", "")
                mask = slice_adata.obs[col].values
                for j, is_in in enumerate(mask):
                    if is_in and cell_regions[j] == "unlabeled":
                        cell_regions[j] = region_name

            # Plot each region
            unique_regions = list(set(cell_regions))
            for j, reg in enumerate(unique_regions):
                reg_mask = np.array([r == reg for r in cell_regions])
                if reg_mask.sum() == 0:
                    continue

                if reg == "unlabeled":
                    color = "#888888"
                    opacity = 0.2
                else:
                    color = region_colors.get(
                        reg, default_colors[j % len(default_colors)]
                    )
                    opacity = 0.8

                fig.add_trace(
                    go.Scatter3d(
                        x=coords[reg_mask, 0],
                        y=coords[reg_mask, 1],
                        z=z_coord[reg_mask],
                        mode="markers",
                        marker=dict(size=point_size, color=color, opacity=opacity),
                        name=reg,
                        legendgroup=reg,
                        showlegend=(i == 0),
                    )
                )

    fig.update_layout(
        title="3D Brain with Anatomical Regions",
        scene=dict(
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            zaxis_title="Z (slice)",
            aspectmode="data",
        ),
    )

    output_path = FIGURES_DIR / f"{output}.html"
    fig.write_html(output_path)
    typer.echo(f"Saved: {output_path}")


@app.command("plot-umap")
def plot_umap(
    input_file: str = typer.Option(
        "aligned_clustered", help="Input h5ad file (without .h5ad)"
    ),
):
    """Generate publication-quality UMAP plots with density and region views."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    input_path = DATA_DIR / f"{input_file}.h5ad"
    if not input_path.exists():
        typer.echo(f"❌ File not found: {input_path}")
        raise typer.Exit(1)

    typer.echo(f"📂 Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)
    typer.echo(f"   Cells: {adata.n_obs}")

    # Check if UMAP exists, compute if not
    if "X_umap" not in adata.obsm:
        typer.echo("   Computing UMAP...")
        if "X_pca" not in adata.obsm:
            sc.tl.pca(adata, n_comps=50)
        if "neighbors" not in adata.uns:
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50)
        sc.tl.umap(adata)

        # Save back to file so we don't recompute next time
        typer.echo(f"   Saving UMAP to {input_path.name}...")
        adata.write_h5ad(input_path)

    umap_dir = FIGURES_DIR / "umap"
    umap_dir.mkdir(parents=True, exist_ok=True)
    umap_coords = adata.obsm["X_umap"]

    # Set clean style
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    # --- 1. Leiden Clusters UMAP ---
    n_clusters = 0
    leiden_colors = None
    if "leiden" in adata.obs:
        typer.echo("   Plotting Leiden clusters...")
        n_clusters = adata.obs["leiden"].nunique()

        fig, ax = plt.subplots(figsize=(8, 7))

        # Use a nice categorical palette
        if n_clusters <= 20:
            leiden_colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))
        else:
            leiden_colors = plt.cm.turbo(np.linspace(0.1, 0.9, n_clusters))

        for i, cluster in enumerate(
            sorted(adata.obs["leiden"].unique(), key=lambda x: int(x))
        ):
            mask = adata.obs["leiden"] == cluster
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=[leiden_colors[i]],
                s=3,
                alpha=0.6,
                label=cluster,
                rasterized=True,
            )

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"Leiden Clusters (n={n_clusters})", fontweight="bold")
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            frameon=False,
            markerscale=3,
            fontsize=8,
            ncol=1 if n_clusters <= 15 else 2,
        )
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(
            umap_dir / f"{input_file}_umap_leiden.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        typer.echo(f"   ✅ Saved: {input_file}_umap_leiden.png")

    # --- 2. Density plot ---
    typer.echo("   Plotting density...")
    fig, ax = plt.subplots(figsize=(7, 6))

    # Compute density
    xy = np.vstack([umap_coords[:, 0], umap_coords[:, 1]])
    try:
        kde = gaussian_kde(xy, bw_method=0.1)
        density = kde(xy)
    except Exception:
        density = np.ones(len(umap_coords))

    # Sort by density for proper layering
    idx = density.argsort()
    x, y, z = umap_coords[idx, 0], umap_coords[idx, 1], density[idx]

    scatter = ax.scatter(x, y, c=z, s=2, cmap="viridis", alpha=0.7, rasterized=True)
    plt.colorbar(scatter, ax=ax, label="Density", shrink=0.8)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("Cell Density", fontweight="bold")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(
        umap_dir / f"{input_file}_umap_density.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    typer.echo(f"   ✅ Saved: {input_file}_umap_density.png")

    # --- 3. Region-specific density plots ---
    region_cols = [c for c in adata.obs.columns if c.startswith("in_")]
    if region_cols:
        typer.echo(f"   Plotting {len(region_cols)} regions...")

        # Color scheme for regions
        region_colors = {
            "optic_lobe": "#e41a1c",
            "central_brain": "#377eb8",
            "antennal_lobe": "#4daf4a",
            "mushroom_body": "#984ea3",
            "subesophageal": "#ff7f00",
            "protocerebrum": "#ffff33",
        }

        n_regions = len(region_cols)
        n_cols = min(3, n_regions)
        n_rows = (n_regions + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_regions == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        for idx, col in enumerate(region_cols):
            row, c = idx // n_cols, idx % n_cols
            ax = axes[row, c]

            region_name = col.replace("in_", "")
            mask = adata.obs[col].astype(bool)
            n_cells = mask.sum()

            # Background (gray)
            ax.scatter(
                umap_coords[~mask, 0],
                umap_coords[~mask, 1],
                c="#e0e0e0",
                s=1,
                alpha=0.3,
                rasterized=True,
            )

            # Region cells (colored)
            color = region_colors.get(region_name, "#e41a1c")
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=color,
                s=3,
                alpha=0.7,
                rasterized=True,
            )

            ax.set_title(f"{region_name}\n(n={n_cells:,})", fontweight="bold")
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_aspect("equal")

        # Hide empty axes
        for idx in range(n_regions, n_rows * n_cols):
            row, c = idx // n_cols, idx % n_cols
            axes[row, c].axis("off")

        plt.tight_layout()
        plt.savefig(
            umap_dir / f"{input_file}_umap_regions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        typer.echo(f"   ✅ Saved: {input_file}_umap_regions.png")

        # --- 3b. Region density contour map ---
        typer.echo("   Plotting region density contours...")
        fig, ax = plt.subplots(figsize=(10, 8))

        # Background scatter (all cells in light gray)
        ax.scatter(
            umap_coords[:, 0],
            umap_coords[:, 1],
            c="#f0f0f0",
            s=1,
            alpha=0.3,
            rasterized=True,
        )

        # Draw density contours for each region
        from scipy.stats import gaussian_kde

        for col in region_cols:
            region_name = col.replace("in_", "")
            mask = adata.obs[col].astype(bool)
            n_cells = mask.sum()

            if n_cells < 50:  # Skip if too few cells for KDE
                continue

            region_coords = umap_coords[mask]
            color = region_colors.get(region_name, "#e41a1c")

            try:
                # Compute KDE
                xy = np.vstack([region_coords[:, 0], region_coords[:, 1]])
                kde = gaussian_kde(xy, bw_method=0.3)

                # Create grid for contours
                x_grid = np.linspace(
                    umap_coords[:, 0].min(), umap_coords[:, 0].max(), 100
                )
                y_grid = np.linspace(
                    umap_coords[:, 1].min(), umap_coords[:, 1].max(), 100
                )
                X, Y = np.meshgrid(x_grid, y_grid)
                positions = np.vstack([X.ravel(), Y.ravel()])
                Z = kde(positions).reshape(X.shape)

                # Draw contours
                levels = np.percentile(Z[Z > 0], [50, 75, 90])
                ax.contour(
                    X,
                    Y,
                    Z,
                    levels=levels,
                    colors=[color],
                    linewidths=[0.5, 1, 1.5],
                    alpha=0.8,
                )
                ax.contourf(
                    X, Y, Z, levels=[levels[-1], Z.max()], colors=[color], alpha=0.3
                )

                # Add label at density peak
                peak_idx = np.unravel_index(Z.argmax(), Z.shape)
                ax.annotate(
                    region_name,
                    (X[peak_idx], Y[peak_idx]),
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                    ha="center",
                    va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.7,
                        edgecolor=color,
                    ),
                )
            except Exception:
                pass  # Skip if KDE fails

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Region Density Contours in UMAP Space", fontweight="bold")
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(
            umap_dir / f"{input_file}_umap_region_density.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()
        typer.echo(f"   ✅ Saved: {input_file}_umap_region_density.png")

    # --- 4. Slice origin ---
    slices = []
    slice_colors = None
    if "slice_name" in adata.obs:
        typer.echo("   Plotting slice origin...")
        fig, ax = plt.subplots(figsize=(8, 7))

        slices = sorted(adata.obs["slice_name"].unique())
        slice_colors = plt.cm.Spectral(np.linspace(0, 1, len(slices)))

        for i, s in enumerate(slices):
            mask = adata.obs["slice_name"] == s
            short_name = s.split("_")[-1] if "_" in s else s
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=[slice_colors[i]],
                s=3,
                alpha=0.6,
                label=short_name,
                rasterized=True,
            )

        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title("Slice Origin", fontweight="bold")
        ax.legend(
            loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, markerscale=3
        )
        ax.set_aspect("equal")

        plt.tight_layout()
        plt.savefig(
            umap_dir / f"{input_file}_umap_slices.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        typer.echo(f"   ✅ Saved: {input_file}_umap_slices.png")

    # --- 5. Combined summary figure ---
    typer.echo("   Creating summary figure...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))

    # A: Leiden
    ax = axes[0, 0]
    if "leiden" in adata.obs and leiden_colors is not None:
        for i, cluster in enumerate(
            sorted(adata.obs["leiden"].unique(), key=lambda x: int(x))
        ):
            mask = adata.obs["leiden"] == cluster
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=[leiden_colors[i]],
                s=1,
                alpha=0.5,
                rasterized=True,
            )
    ax.set_title("A. Leiden Clusters", fontweight="bold", loc="left")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")

    # B: Density
    ax = axes[0, 1]
    ax.scatter(x, y, c=z, s=1, cmap="viridis", alpha=0.6, rasterized=True)
    ax.set_title("B. Cell Density", fontweight="bold", loc="left")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")

    # C: Slice origin
    ax = axes[1, 0]
    if "slice_name" in adata.obs:
        for i, s in enumerate(slices):
            mask = adata.obs["slice_name"] == s
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=[plt.cm.Spectral(i / len(slices))],
                s=1,
                alpha=0.5,
                rasterized=True,
            )
    ax.set_title("C. Slice Origin", fontweight="bold", loc="left")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")

    # D: Regions overlay
    ax = axes[1, 1]
    ax.scatter(
        umap_coords[:, 0],
        umap_coords[:, 1],
        c="#e0e0e0",
        s=1,
        alpha=0.3,
        rasterized=True,
    )
    if region_cols:
        for i, col in enumerate(region_cols[:6]):  # Top 6 regions
            mask = adata.obs[col].astype(bool)
            region_name = col.replace("in_", "")
            color = region_colors.get(region_name, plt.cm.Set1(i / 6))
            ax.scatter(
                umap_coords[mask, 0],
                umap_coords[mask, 1],
                c=[color] if isinstance(color, str) else [color],
                s=2,
                alpha=0.6,
                label=region_name,
                rasterized=True,
            )
        ax.legend(loc="upper right", frameon=True, fontsize=7, markerscale=2)
    ax.set_title("D. Anatomical Regions", fontweight="bold", loc="left")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(
        umap_dir / f"{input_file}_umap_summary.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    typer.echo(f"   ✅ Saved: {input_file}_umap_summary.png")

    typer.echo(f"\n📊 All UMAP plots saved to {umap_dir}/")


if __name__ == "__main__":
    app()
