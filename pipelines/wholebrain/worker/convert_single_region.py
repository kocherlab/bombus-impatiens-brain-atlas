#!/usr/bin/env python3
"""
Convert Xenium regions to h5ad files with cropping.

Usage:
    python convert_single_region.py list                    # List available regions
    python convert_single_region.py base-images --all       # Generate all base images
    python convert_single_region.py base-images REGION      # Generate single base image
    python convert_single_region.py convert REGION          # Convert single region with
                                                            # crops
    python convert_single_region.py convert --all           # Convert all regions
"""

import time
from pathlib import Path
from typing import Optional

import dask
import matplotlib.pyplot as plt
import numpy as np
import typer
from anndata import AnnData

app = typer.Typer(help="Convert Xenium regions to h5ad files")

# =============================================================================
# CONFIG
# =============================================================================

BASE_PATH = Path(__file__).resolve().parents[3] / "data/20260115__193720__20260115"
OUTPUT_DIR = Path(__file__).parent / "data"
FIGURES_DIR = Path(__file__).parent / "figures"
CROPS_DIR = FIGURES_DIR / "crops"

N_JOBS = 8
CELLS_BOUNDARIES = True
NUCLEUS_BOUNDARIES = True
MORPHOLOGY_FOCUS = True
CELLS_AS_CIRCLES = True

SCALE = 0.2125  # µm/pixel

# =============================================================================
# CROP CONFIGURATIONS
# =============================================================================
# Crop boxes are set from the base morphology images. To redefine them:
# 1. Run: python convert_single_region.py base-images --all
# 2. Open figures/crops/*_base.png in QuPath/Napari
# 3. Update min/max coordinates below (in PIXELS)
# 4. Run: python convert_single_region.py convert --all
# =============================================================================

CROP_CONFIGS = {
    # Slide 1: XETG00151__0068401 (worker brain 1)
    # z_level: 10µm spacing between slices
    "output-XETG00151__0068401__Region_1__20260115__193900": {
        "1A": {"min": [0, 0], "max": [12500, 13000], "z_level": 0},
        "1B": {"min": [0, 19000], "max": [12000, 33000], "z_level": 10},
    },
    "output-XETG00151__0068401__Region_2__20260115__193900": {
        "2A": {"min": [0, 0], "max": [12000, 15000], "z_level": 20},
        "2B": {"min": [0, 21000], "max": [13000, 34000], "z_level": 30},
    },
    "output-XETG00151__0068401__Region_3__20260115__193900": {
        "3A": {"min": [3000, 0], "max": [18000, 13000], "z_level": 40},
        "3B": {"min": [2000, 17000], "max": [14000, 30000], "z_level": 50},
        "3C": {"min": [20000, 74000], "max": [32000, 88000], "z_level": 60},
        "3D": {"min": [22000, 94000], "max": [33000, 108000], "z_level": 70},
    },
    "output-XETG00151__0068401__Region_4__20260115__193900": {
        "4A": {"min": [3000, 0], "max": [14000, 18000], "z_level": 80},
        "4B": {"min": [1000, 18000], "max": [12000, 32000], "z_level": 90},
    },
    "output-XETG00151__0068401__Region_5__20260115__193900": {
        "5A": {"min": [0, 0], "max": [13000, 14000], "z_level": 100},
        "5B": {"min": [0, 19000], "max": [13000, 32000], "z_level": 110},
    },
    "output-XETG00151__0068401__Region_6__20260115__193900": {
        "6A": {"min": [1000, 1000], "max": [14000, 14000], "z_level": 120},
        "6B": {"min": [0, 21000], "max": [14000, 35000], "z_level": 130},
    },
    "output-XETG00151__0068401__Region_7__20260115__193900": {
        "7A": {"min": [0, 1000], "max": [15000, 17000], "z_level": 140},
        "7B": {"min": [2, 18000], "max": [14000, 32000], "z_level": 150},
    },
    "output-XETG00151__0068401__Region_8__20260115__193900": {
        "8A": {"min": [1000, 0], "max": [13000, 13000], "z_level": 160},
    },
    # Slide 2: XETG00151__0069041 (worker brain 2)
    "output-XETG00151__0069041__Region_1__20260115__193900": {
        "2_1A": {"min": [0, 1000], "max": [14000, 15000], "z_level": 170},
        "2_1B": {"min": [0, 19000], "max": [15000, 38000], "z_level": 180},
    },
    "output-XETG00151__0069041__Region_2__20260115__193900": {
        "2_2A": {"min": [3000, 0], "max": [15000, 13000], "z_level": 190},
        "2_2B": {"min": [2000, 17000], "max": [13000, 32000], "z_level": 200},
    },
    "output-XETG00151__0069041__Region_3__20260115__193900": {
        "2_3A": {"min": [0, 0], "max": [13000, 15000], "z_level": 210},
        "2_3B": {"min": [2000, 18000], "max": [14000, 33000], "z_level": 220},
    },
    "output-XETG00151__0069041__Region_4__20260115__193900": {
        "2_4A": {"min": [0, 2000], "max": [12000, 15000], "z_level": 230},
        "2_4B": {"min": [0, 22000], "max": [12000, 36000], "z_level": 240},
    },
    "output-XETG00151__0069041__Region_5__20260115__193900": {
        "2_5A": {"min": [2000, 0], "max": [14000, 13000], "z_level": 250},
        "2_5B": {"min": [2000, 18000], "max": [13000, 33000], "z_level": 260},
    },
    "output-XETG00151__0069041__Region_6__20260115__193900": {
        "2_6A": {"min": [2000, 0], "max": [15000, 11000], "z_level": 270},
        "2_6B": {"min": [2000, 17000], "max": [15000, 35000], "z_level": 280},
    },
    "output-XETG00151__0069041__Region_7__20260115__193900": {
        "2_7A": {"min": [1000, 2000], "max": [15000, 15000], "z_level": 290},
        "2_7B": {"min": [2000, 21000], "max": [15000, 35000], "z_level": 300},
    },
    "output-XETG00151__0069041__Region_8__20260115__193900": {
        "2_8A": {"min": [3000, 0], "max": [13000, 12000], "z_level": 310},
        "2_8B": {"min": [2000, 17000], "max": [13000, 30000], "z_level": 320},
        "2_8C": {"min": [500, 34000], "max": [12000, 49000], "z_level": 330},
        "2_8D": {"min": [2000, 54000], "max": [12000, 70000], "z_level": 340},
    },
}

# =============================================================================
# HELPERS
# =============================================================================


def get_region_info(region_name: str) -> dict:
    parts = region_name.split("__")
    return {
        "sample_id": parts[1] if len(parts) > 1 else "unknown",
        "region": parts[2] if len(parts) > 2 else "unknown",
    }


def get_available_regions() -> list[str]:
    """Return regions that exist on disk and have crop configs."""
    available = []
    for region in CROP_CONFIGS.keys():
        region_path = BASE_PATH / region
        if region_path.exists() and (region_path / "experiment.xenium").exists():
            available.append(region)
    return available


def load_xenium(region_path: Path):
    """Load Xenium data with standard settings."""
    import spatialdata_plot  # noqa: F401
    from spatialdata_io import xenium

    return xenium(
        path=str(region_path),
        n_jobs=N_JOBS,
        cells_boundaries=CELLS_BOUNDARIES,
        nucleus_boundaries=NUCLEUS_BOUNDARIES,
        morphology_focus=MORPHOLOGY_FOCUS,
        cells_as_circles=CELLS_AS_CIRCLES,
    )


# =============================================================================
# COMMANDS
# =============================================================================


@app.command("list")
def list_regions():
    """List all configured regions and their availability."""
    typer.echo("Configured regions:\n")

    available = get_available_regions()

    for region, crops in CROP_CONFIGS.items():
        status = "✓ ready" if region in available else "✗ not found"
        crop_names = ", ".join(crops.keys())
        typer.echo(f"  {status}  {region}")
        typer.echo(f"          Crops: {crop_names}\n")

    typer.echo(f"\nTotal: {len(available)}/{len(CROP_CONFIGS)} regions available")


@app.command("base-images")
def base_images(
    region: Optional[str] = typer.Argument(None, help="Region name (or use --all)"),
    all_regions: bool = typer.Option(
        False, "--all", help="Process all available regions"
    ),
):
    """Generate base morphology images for determining crop coordinates."""
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    if all_regions:
        regions = get_available_regions()
        if not regions:
            typer.echo("No regions available yet. Data may still be copying.")
            raise typer.Exit(1)
        typer.echo(f"Processing {len(regions)} available regions...\n")
    elif region:
        if region not in CROP_CONFIGS:
            typer.echo(f"Unknown region: {region}")
            typer.echo(
                "Run 'python convert_single_region.py list' to see available regions"
            )
            raise typer.Exit(1)
        regions = [region]
    else:
        typer.echo("Specify a region name or use --all")
        raise typer.Exit(1)

    dask.config.set(scheduler="threads", num_workers=N_JOBS)

    for region_name in regions:
        region_path = BASE_PATH / region_name
        if not (region_path / "experiment.xenium").exists():
            typer.echo(f"Skipping {region_name} (not ready)")
            continue

        info = get_region_info(region_name)
        typer.echo(f"Loading {region_name}...")

        start = time.time()
        sdata = load_xenium(region_path)
        typer.echo(f"  Loaded in {time.time() - start:.1f}s")

        # Save base morphology
        fig, ax = plt.subplots(figsize=(12, 12))
        sdata.pl.render_images("morphology_focus").pl.show(
            ax=ax,
            title=f"{info['sample_id']} - {info['region']}",
            coordinate_systems="global",
        )
        ax.set_aspect("equal")

        output_path = CROPS_DIR / f"{region_name}_base.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        typer.echo(f"  Saved: {output_path.name}")

    typer.echo(f"\nBase images saved to: {CROPS_DIR}")
    typer.echo(
        "Open in QuPath/Napari to determine crop coordinates, then update CROP_CONFIGS"
    )


@app.command()
def convert(
    region: Optional[str] = typer.Argument(None, help="Region name (or use --all)"),
    all_regions: bool = typer.Option(
        False, "--all", help="Process all available regions"
    ),
    skip_existing: bool = typer.Option(
        True, "--skip-existing/--no-skip", help="Skip regions with existing h5ad files"
    ),
):
    """Convert Xenium regions to h5ad files with crops."""
    from spatialdata import bounding_box_query

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    if all_regions:
        regions = get_available_regions()
        if not regions:
            typer.echo("No regions available yet. Data may still be copying.")
            raise typer.Exit(1)
        typer.echo(f"Processing {len(regions)} available regions...\n")
    elif region:
        if region not in CROP_CONFIGS:
            typer.echo(f"Unknown region: {region}")
            typer.echo(
                "Run 'python convert_single_region.py list' to see available regions"
            )
            raise typer.Exit(1)
        regions = [region]
    else:
        typer.echo("Specify a region name or use --all")
        raise typer.Exit(1)

    dask.config.set(scheduler="threads", num_workers=N_JOBS)

    total_cells = 0
    for region_name in regions:
        region_path = BASE_PATH / region_name
        crop_config = CROP_CONFIGS[region_name]

        if not (region_path / "experiment.xenium").exists():
            typer.echo(f"Skipping {region_name} (not ready)")
            continue

        # Check if all crops exist
        if skip_existing:
            all_exist = all(
                (OUTPUT_DIR / f"{region_name}_{crop_name}.h5ad").exists()
                for crop_name in crop_config.keys()
            )
            if all_exist:
                typer.echo(f"Skipping {region_name} (h5ad files exist)")
                continue

        info = get_region_info(region_name)
        typer.echo(f"\nProcessing {region_name}")
        typer.echo(f"  Sample: {info['sample_id']}, Region: {info['region']}")
        typer.echo(f"  Crops: {list(crop_config.keys())}")

        start = time.time()
        sdata = load_xenium(region_path)
        typer.echo(f"  Loaded in {time.time() - start:.1f}s")

        # Save base morphology if not exists
        base_path = CROPS_DIR / f"{region_name}_base.png"
        if not base_path.exists():
            fig, ax = plt.subplots(figsize=(12, 12))
            sdata.pl.render_images("morphology_focus").pl.show(
                ax=ax,
                title=f"{info['sample_id']} - {info['region']}",
                coordinate_systems="global",
            )
            ax.set_aspect("equal")
            fig.savefig(base_path, dpi=150, bbox_inches="tight")
            plt.close(fig)

        # Process each crop
        for crop_name, crop_coords in crop_config.items():
            crop_start = time.time()

            output_path = OUTPUT_DIR / f"{region_name}_{crop_name}.h5ad"
            if skip_existing and output_path.exists():
                typer.echo(f"  Skipping crop {crop_name} (exists)")
                continue

            typer.echo(f"  Crop '{crop_name}': {crop_coords}")

            min_x, min_y = crop_coords["min"]
            max_x, max_y = crop_coords["max"]

            # Crop image
            cropped = bounding_box_query(
                sdata,
                min_coordinate=[min_x, min_y],
                max_coordinate=[max_x, max_y],
                axes=("x", "y"),
                target_coordinate_system="global",
            )

            # Save crop visualization
            fig, ax = plt.subplots(figsize=(8, 8))
            cropped.pl.render_images("morphology_focus").pl.show(
                ax=ax, title=f"{region_name} - {crop_name}", coordinate_systems="global"
            )
            ax.set_aspect("equal")
            fig.savefig(
                CROPS_DIR / f"{region_name}_{crop_name}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)

            # Filter cells by spatial coordinates
            full_table = sdata.tables["table"]
            spatial_coords = full_table.obsm["spatial"]

            min_x_um, max_x_um = min_x * SCALE, max_x * SCALE
            min_y_um, max_y_um = min_y * SCALE, max_y * SCALE

            mask = (
                (spatial_coords[:, 0] >= min_x_um)
                & (spatial_coords[:, 0] <= max_x_um)
                & (spatial_coords[:, 1] >= min_y_um)
                & (spatial_coords[:, 1] <= max_y_um)
            )
            indices = np.where(mask)[0]

            if len(indices) == 0:
                typer.echo(
                    f"    WARNING: No cells in crop region! Skipping {crop_name}"
                )
                continue

            # Create AnnData
            adata = AnnData(
                X=full_table.X[indices, :],
                obs=full_table.obs.iloc[indices].copy(),
                var=full_table.var.copy(),
                uns=full_table.uns.copy(),
                obsm={k: v[indices] for k, v in full_table.obsm.items()},
            )

            if "morphology_focus" in cropped.images:
                image_data = cropped.images["morphology_focus"][
                    "scale0"
                ].image.compute()
                adata.uns["image"] = image_data.values

            adata.uns["region_info"] = info
            adata.uns["region_name"] = region_name
            adata.uns["crop_name"] = crop_name
            adata.uns["crop_coords"] = crop_coords

            # Add z_level from config (10 µm spacing between slices)
            z_level = crop_coords.get("z_level", 0)
            adata.obs["z_level"] = z_level

            adata.write_h5ad(output_path)
            total_cells += adata.n_obs

            typer.echo(
                f"    Cells: {adata.n_obs}, Time: {time.time() - crop_start:.1f}s"
            )

    typer.echo(f"\nDone! Total cells: {total_cells:,}")


if __name__ == "__main__":
    app()
