#!/usr/bin/env python3
"""
Convert a single Xenium region to h5ad files.

Usage:
    python convert_single_region.py <region_name>

Example:
    python convert_single_region.py \
        output-XETG00151__0040240__Region_1__20251204__195014
"""

import sys
import time
from pathlib import Path

import dask
import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from spatialdata import bounding_box_query
from spatialdata_io import xenium

# =============================================================================
# CONFIG
# =============================================================================

REPO = Path(__file__).resolve().parents[3]
BASE_PATH = REPO / "data/20251204__194847___20251204"
OUTPUT_DIR = REPO / "pipelines/wholebrain/queen/data"
FIGURES_DIR = REPO / "pipelines/wholebrain/queen/figures"
CROPS_DIR = FIGURES_DIR / "crops"

N_JOBS = 8  # Per-job parallelism
CELLS_BOUNDARIES = True
NUCLEUS_BOUNDARIES = True
MORPHOLOGY_FOCUS = True
CELLS_AS_CIRCLES = True

# Crop configs in PIXELS (microns / 0.2125)
# Original coords were in microns, scaled to pixels for bounding_box_query
SCALE = 0.2125  # µm/pixel
CROP_CONFIGS = {
    # Slide 1: XETG00151__0040240 (queen brain, slide 1)
    # z_level: 10µm spacing between slices
    "output-XETG00151__0040240__Region_1__20251204__195014": {
        "1A": {"min": [635, 132], "max": [14513, 20471], "z_level": 0},
        "1B": {"min": [635, 20471], "max": [14513, 40810], "z_level": 10},
    },
    "output-XETG00151__0040240__Region_2__20251204__195015": {
        "2A": {"min": [202, 461], "max": [14372, 20645], "z_level": 20},
        "2B": {"min": [202, 20645], "max": [14372, 40824], "z_level": 30},
    },
    "output-XETG00151__0040240__Region_3__20251204__195015": {
        "3A": {"min": [127, 9], "max": [15000, 20000], "z_level": 40},
        "3B": {"min": [20000, 85000], "max": [33412, 105516], "z_level": 50},
    },
    "output-XETG00151__0040240__Region_4__20251204__195015": {
        "4A": {"min": [28, 0], "max": [16960, 20349], "z_level": 60},
        "4B": {"min": [28, 20349], "max": [16960, 40903], "z_level": 70},
    },
    "output-XETG00151__0040240__Region_5__20251204__195015": {
        "5A": {"min": [0, 0], "max": [15435, 18702], "z_level": 80},
        "5B": {"min": [0, 18702], "max": [15435, 37525], "z_level": 90},
    },
    "output-XETG00151__0040240__Region_6__20251204__195015": {
        "6A": {"min": [0, 306], "max": [14170, 20278], "z_level": 100},
        "6B": {"min": [0, 20278], "max": [14170, 40249], "z_level": 110},
    },
    "output-XETG00151__0040240__Region_7__20251204__195015": {
        "7A": {"min": [414, 0], "max": [17346, 20414], "z_level": 120},
        "7B": {"min": [414, 20414], "max": [17346, 40993], "z_level": 130},
    },
    "output-XETG00151__0040240__Region_8__20251204__195015": {
        "8A": {"min": [1110, 1233], "max": [15802, 20555], "z_level": 140},
    },
    # Slide 2: XETG00151__0040243 (queen brain, slide 2)
    "output-XETG00151__0040243__Region_1__20251204__195015": {
        "2_1A": {"min": [42, 0], "max": [17355, 20457], "z_level": 150},
        "2_1B": {"min": [42, 20457], "max": [17355, 41063], "z_level": 160},
    },
    "output-XETG00151__0040243__Region_2__20251204__195015": {
        "2_2A": {"min": [0, 3063], "max": [14042, 21826], "z_level": 170},
        "2_2B": {"min": [0, 21826], "max": [14042, 40588], "z_level": 180},
    },
    "output-XETG00151__0040243__Region_3__20251204__195015": {
        "2_3A": {"min": [1666, 132], "max": [15000, 20000], "z_level": 190},
        "2_3B": {"min": [20000, 85000], "max": [34179, 104396], "z_level": 200},
    },
    "output-XETG00151__0040243__Region_4__20251204__195015": {
        "2_4A": {"min": [202, 0], "max": [16593, 21435], "z_level": 210},
        "2_4B": {"min": [202, 21435], "max": [16593, 43072], "z_level": 220},
    },
    "output-XETG00151__0040243__Region_5__20251204__195015": {
        "2_5A": {"min": [0, 645], "max": [14301, 20531], "z_level": 230},
        "2_5B": {"min": [0, 20531], "max": [14301, 40418], "z_level": 240},
    },
    "output-XETG00151__0040243__Region_6__20251204__195015": {
        "2_6A": {"min": [0, 0], "max": [15661, 19435], "z_level": 250},
        "2_6B": {"min": [0, 19435], "max": [15661, 39082], "z_level": 260},
    },
    "output-XETG00151__0040243__Region_7__20251204__195015": {
        "2_7A": {"min": [0, 0], "max": [16998, 18555], "z_level": 270},
        "2_7B": {"min": [0, 18555], "max": [16998, 37204], "z_level": 280},
    },
    "output-XETG00151__0040243__Region_8__20251204__195015": {
        "2_8A": {"min": [0, 569], "max": [14240, 20273], "z_level": 290},
    },
}

# =============================================================================
# PROCESSING
# =============================================================================


def get_region_info(region_name: str) -> dict:
    parts = region_name.split("__")
    return {
        "sample_id": parts[1] if len(parts) > 1 else "unknown",
        "region": parts[2] if len(parts) > 2 else "unknown",
    }


def process_region(region_name: str):
    """Process a single region."""
    if region_name not in CROP_CONFIGS:
        print(f"ERROR: No crop config for {region_name}")
        print(f"Available regions: {list(CROP_CONFIGS.keys())}")
        sys.exit(1)

    crop_config = CROP_CONFIGS[region_name]
    region_path = BASE_PATH / region_name
    info = get_region_info(region_name)

    print(f"Processing {region_name}")
    print(f"  Sample: {info['sample_id']}, Region: {info['region']}")
    print(f"  Crops: {list(crop_config.keys())}")

    # Load Xenium data
    print("  Loading Xenium data...")
    start = time.time()
    sdata = xenium(
        path=str(region_path),
        n_jobs=N_JOBS,
        cells_boundaries=CELLS_BOUNDARIES,
        nucleus_boundaries=NUCLEUS_BOUNDARIES,
        morphology_focus=MORPHOLOGY_FOCUS,
        cells_as_circles=CELLS_AS_CIRCLES,
    )
    print(f"  Loaded in {time.time() - start:.1f}s")

    # Save base morphology
    fig, ax = plt.subplots(figsize=(10, 10))
    sdata.pl.render_images("morphology_focus").pl.show(
        ax=ax,
        title=f"{info['sample_id']} - {info['region']}",
        coordinate_systems="global",
    )
    ax.set_aspect("equal")
    fig.savefig(CROPS_DIR / f"{region_name}_base.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Process each crop
    for crop_name, crop_coords in crop_config.items():
        crop_start = time.time()
        print(f"\n  Crop '{crop_name}': {crop_coords}")

        # Get crop bounds
        min_x, min_y = crop_coords["min"]
        max_x, max_y = crop_coords["max"]

        # Crop image using bounding_box_query
        cropped = bounding_box_query(
            sdata,
            min_coordinate=[min_x, min_y],
            max_coordinate=[max_x, max_y],
            axes=("x", "y"),
            target_coordinate_system="global",
        )

        # Plot crop
        fig, ax = plt.subplots(figsize=(8, 8))
        cropped.pl.render_images("morphology_focus").pl.show(
            ax=ax, title=f"{region_name} - {crop_name}", coordinate_systems="global"
        )
        ax.set_aspect("equal")
        fig.savefig(
            CROPS_DIR / f"{region_name}_{crop_name}.png", dpi=150, bbox_inches="tight"
        )
        plt.close(fig)

        # Manually subset table using spatial coordinates (more reliable than
        # bounding_box_query)
        # The table may not be properly linked to the coordinate system in spatialdata
        full_table = sdata.tables["table"]
        spatial_coords = full_table.obsm[
            "spatial"
        ]  # (n_cells, 2) array with x, y coords in MICRONS

        # Convert pixel crop box to microns for cell filtering
        min_x_um = min_x * SCALE
        max_x_um = max_x * SCALE
        min_y_um = min_y * SCALE
        max_y_um = max_y * SCALE

        # Filter cells within bounding box (in microns)
        mask = (
            (spatial_coords[:, 0] >= min_x_um)
            & (spatial_coords[:, 0] <= max_x_um)
            & (spatial_coords[:, 1] >= min_y_um)
            & (spatial_coords[:, 1] <= max_y_um)
        )
        indices = np.where(mask)[0]

        if len(indices) == 0:
            print(f"    WARNING: No cells in crop region! Skipping {crop_name}")
            continue

        print(f"    Found {len(indices)} cells in crop region")

        # Create AnnData from subset (use indices for sparse matrix compatibility)
        adata = AnnData(
            X=full_table.X[indices, :],
            obs=full_table.obs.iloc[indices].copy(),
            var=full_table.var.copy(),
            uns=full_table.uns.copy(),
            obsm={k: v[indices] for k, v in full_table.obsm.items()},
        )

        if "morphology_focus" in cropped.images:
            image_data = cropped.images["morphology_focus"]["scale0"].image.compute()
            adata.uns["image"] = image_data.values

        adata.uns["region_info"] = info
        adata.uns["region_name"] = region_name
        adata.uns["crop_name"] = crop_name
        adata.uns["crop_coords"] = crop_coords

        # Add z_level from config (10µm spacing between slices)
        z_level = crop_coords.get("z_level", 0)
        adata.obs["z_level"] = z_level

        output_path = OUTPUT_DIR / f"{region_name}_{crop_name}.h5ad"
        adata.write_h5ad(output_path)

        print(f"    Saved: {output_path.name}")
        print(f"    Cells: {adata.n_obs}, Time: {time.time() - crop_start:.1f}s")


def main():
    if len(sys.argv) != 2:
        print("Usage: python convert_single_region.py <region_name>")
        print("\nAvailable regions:")
        for r in CROP_CONFIGS.keys():
            print(f"  {r}")
        sys.exit(1)

    region_name = sys.argv[1]

    dask.config.set(scheduler="threads", num_workers=N_JOBS)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    CROPS_DIR.mkdir(parents=True, exist_ok=True)

    start = time.time()
    process_region(region_name)
    print(f"\nTotal time: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
