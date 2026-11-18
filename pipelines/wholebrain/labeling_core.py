"""
GeoJSON-based anatomical region labeling for whole-brain Xenium data.

Parses hand-drawn annotations exported from QuPath/Napari as GeoJSON,
assigns cells to regions using Shapely, and propagates labels across slices.

Shared by both brains. The brain is the current directory (see brain_dir.py);
its ``labeling.py`` holds only ANNOTATION_MAPPING, the table of which GeoJSON
covers which crop.
"""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import scanpy as sc
from brain_dir import BRAIN_DIR, annotations_dir, load_annotation_mapping
from shapely.geometry import Point, Polygon

# =============================================================================
# CONFIG
# =============================================================================

ANNOTATIONS_DIR = annotations_dir(BRAIN_DIR)
DATA_DIR = BRAIN_DIR / "data"

# Pixel-to-micron scale for Xenium
PIXEL_SCALE = 0.2125

# Annotation file -> crop suffixes, from the brain's labeling.py.
ANNOTATION_MAPPING = load_annotation_mapping(BRAIN_DIR)


def parse_annotation_filename(filepath: Path) -> dict:
    """
    Parse annotation filename to extract slide and region info.

    Format: YYYYMMDD_HHMM_slide_N_region_M.geojson
    Example: 20251210_1257_slide_2_region_7.geojson
    """
    match = re.match(
        r"(\d{8})_(\d{4})_slide_(\d+)_region_(\d+)\.geojson",
        filepath.name,
    )
    if not match:
        return {}

    return {
        "date": match.group(1),
        "time": match.group(2),
        "slide": int(match.group(3)),
        "region": int(match.group(4)),
    }


def load_geojson_regions(geojson_path: Path, pixel_scale: float | None = None) -> dict:
    """
    Load anatomical regions from a GeoJSON file.

    Args:
        geojson_path: Path to GeoJSON file
        pixel_scale: If provided, multiply coordinates by this value to convert
                     from pixels to microns. Xenium uses 0.2125 µm/pixel.

    Returns dict mapping region_name -> {polygon, color, z_plane}

    Note: Multiple polygons with the same name are merged using unary_union.
    """
    from shapely.ops import unary_union

    with open(geojson_path) as f:
        data = json.load(f)

    # First pass: collect all polygons by name
    polygons_by_name = {}
    metadata_by_name = {}

    for feature in data.get("features", []):
        geometry = feature.get("geometry", {})
        properties = feature.get("properties", {})

        # Get region name from classification (normalize to lowercase)
        classification = properties.get("classification", {})
        region_name = classification.get("name", properties.get("name", "unknown"))
        region_name = region_name.lower()
        # Fix known typos
        if region_name == "lobulla":
            region_name = "lobula"
        color = classification.get("color", [128, 128, 128])

        # Get z-plane if available
        z_plane = geometry.get("plane", {}).get("z")

        # Extract coordinates (handle nested structure)
        coords = geometry.get("coordinates", [])
        if geometry.get("type") == "Polygon" and coords:
            # GeoJSON Polygon: [[[x1,y1], [x2,y2], ...]]
            # First element is exterior ring
            exterior_coords = coords[0]

            # Apply pixel-to-micron scaling if specified
            if pixel_scale is not None:
                exterior_coords = [
                    [x * pixel_scale, y * pixel_scale] for x, y in exterior_coords
                ]

            if len(exterior_coords) >= 3:
                try:
                    polygon = Polygon(exterior_coords)
                    if not polygon.is_valid:
                        # Repair self-intersecting hand-drawn polygons rather
                        # than dropping them. Four are affected: one 214-vertex
                        # KC-NC polygon on queen slide 0040240 section 6
                        # (439 cells) and three worker KC-OC polygons (328 cells).
                        repaired = polygon.buffer(0)
                        if repaired.is_valid and not repaired.is_empty:
                            print(
                                f"Note: repaired self-intersecting polygon for "
                                f"{region_name} ({len(exterior_coords)} vertices)"
                            )
                            polygon = repaired
                        else:
                            print(
                                f"Warning: unrepairable polygon for {region_name} "
                                f"({len(exterior_coords)} vertices) - skipping"
                            )
                            continue
                    if region_name not in polygons_by_name:
                        polygons_by_name[region_name] = []
                        metadata_by_name[region_name] = {
                            "color": color,
                            "z_plane": z_plane,
                        }
                    polygons_by_name[region_name].append(polygon)
                except Exception as e:
                    print(f"Warning: Invalid polygon for {region_name}: {e}")

    # Second pass: merge polygons with same name
    regions = {}
    for region_name, polygon_list in polygons_by_name.items():
        if len(polygon_list) == 1:
            merged = polygon_list[0]
        else:
            # Merge multiple polygons into MultiPolygon or single geometry
            merged = unary_union(polygon_list)

        regions[region_name] = {
            "polygon": merged,
            "coordinates": None,  # Not meaningful for merged polygons
            "color": metadata_by_name[region_name]["color"],
            "z_plane": metadata_by_name[region_name]["z_plane"],
        }

    return regions


def discover_annotations() -> list[Path]:
    """Find all GeoJSON annotation files."""
    if not ANNOTATIONS_DIR.exists():
        return []
    return sorted(ANNOTATIONS_DIR.glob("*.geojson"))


def point_in_regions(coords: np.ndarray, regions: dict) -> dict:
    """
    Test which anatomical regions each coordinate point falls into.

    Args:
        coords: (N, 2) array of (x, y) coordinates
        regions: dict from load_geojson_regions()

    Returns:
        dict mapping region_name -> boolean array of length N
    """
    n_points = len(coords)
    assignments = {}

    for region_name, region_data in regions.items():
        polygon = region_data.get("polygon")
        if polygon is None:
            assignments[region_name] = np.zeros(n_points, dtype=bool)
            continue

        # Test each point (vectorized with prepared geometry for large datasets)
        from shapely.prepared import prep

        prepared_poly = prep(polygon)

        in_region = np.array([prepared_poly.contains(Point(x, y)) for x, y in coords])
        assignments[region_name] = in_region

    return assignments


def add_labels_to_adata(
    adata: sc.AnnData,
    regions: dict,
    prefix: str = "in_",
    coord_offset: tuple[float, float] | None = None,
) -> sc.AnnData:
    """
    Add anatomical region labels to an AnnData object.

    Args:
        adata: AnnData with spatial coordinates in obsm['spatial']
        regions: dict from load_geojson_regions()
        prefix: prefix for column names (e.g., "in_" -> "in_lamina")
        coord_offset: (x_offset, y_offset) to ADD to cell coords before testing.
                      Use this when annotation is in global coords and cells are local.
                      E.g., if annotation X starts at 3583 and cells start at 0,
                      use coord_offset=(3583, y_offset) to shift cells into annotation
                      space.

    Returns:
        Modified AnnData with boolean region columns in obs
    """
    if "spatial" not in adata.obsm:
        raise ValueError("No 'spatial' key in adata.obsm")

    coords = adata.obsm["spatial"].copy()

    # Apply offset to shift local coords into annotation coordinate space
    if coord_offset is not None:
        coords = coords + np.array(coord_offset)

    assignments = point_in_regions(coords, regions)

    # Add as boolean columns
    for region_name, in_region in assignments.items():
        col_name = f"{prefix}{region_name}"
        adata.obs[col_name] = in_region

    # Store region metadata
    if "region_colors" not in adata.uns:
        adata.uns["region_colors"] = {}

    for region_name, region_data in regions.items():
        color = region_data.get("color", [128, 128, 128])
        # Convert RGB to hex
        hex_color = "#{:02x}{:02x}{:02x}".format(*color)
        adata.uns["region_colors"][region_name] = hex_color

    return adata


def plot_labels(
    adata: sc.AnnData,
    regions: dict,
    output_path: Optional[Path] = None,
    title: str = "",
    padding: float = 200.0,
) -> None:
    """
    Plot labeled cells with region polygons, cropped to cell bounding box.

    Args:
        adata: AnnData with spatial coordinates and label columns
        regions: dict from load_geojson_regions()
        output_path: Optional path to save figure
        title: Title prefix for plots
        padding: Padding around cell bounding box in µm
    """
    import matplotlib.pyplot as plt
    from shapely.geometry import box

    coords = adata.obsm["spatial"]

    # Get cell bounding box
    xmin, xmax = coords[:, 0].min(), coords[:, 0].max()
    ymin, ymax = coords[:, 1].min(), coords[:, 1].max()
    cell_bbox = box(xmin - padding, ymin - padding, xmax + padding, ymax + padding)

    # Count regions with labels
    region_names = [
        name for name in regions.keys() if f"in_{name}" in adata.obs.columns
    ]
    n_regions = len(region_names)

    if n_regions == 0:
        print("No label columns found in adata.obs")
        return

    fig, axes = plt.subplots(1, n_regions + 1, figsize=(4 * (n_regions + 1), 4))
    if n_regions == 0:
        axes = [axes]

    # Base plot
    axes[0].scatter(coords[:, 0], coords[:, 1], s=0.5, c="lightgray", alpha=0.3)
    axes[0].set_title(
        f"{title}: {len(adata)} cells" if title else f"{len(adata)} cells"
    )
    axes[0].set_aspect("equal")
    axes[0].set_xlim(xmin - padding, xmax + padding)
    axes[0].set_ylim(ymin - padding, ymax + padding)

    # Region plots
    for i, name in enumerate(region_names):
        ax = axes[i + 1]
        col = f"in_{name}"
        in_region = adata.obs[col].values

        ax.scatter(
            coords[~in_region, 0],
            coords[~in_region, 1],
            s=0.5,
            c="lightgray",
            alpha=0.2,
        )
        ax.scatter(coords[in_region, 0], coords[in_region, 1], s=1, c="red", alpha=0.8)

        # Only draw polygons that intersect with cell bounding box
        poly = regions[name]["polygon"]
        if poly.geom_type == "MultiPolygon":
            for p in poly.geoms:
                if p.intersects(cell_bbox):
                    x, y = p.exterior.xy
                    ax.plot(x, y, "b-", linewidth=1)
        else:
            if poly.intersects(cell_bbox):
                x, y = poly.exterior.xy
                ax.plot(x, y, "b-", linewidth=1)

        ax.set_title(f"{name}: {in_region.sum()}")
        ax.set_aspect("equal")
        ax.set_xlim(xmin - padding, xmax + padding)
        ax.set_ylim(ymin - padding, ymax + padding)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")

    plt.close()


def propagate_labels_to_neighbors(
    adata: sc.AnnData,
    source_slice: str,
    target_slices: list[str],
    regions: dict,
    transform_matrix: Optional[np.ndarray] = None,
) -> sc.AnnData:
    """
    Propagate labels from a source slice to neighboring slices.

    For slices adjacent to the labeled slice, we assume the same
    anatomical regions exist with similar (transformed) boundaries.

    Args:
        adata: Combined AnnData with slice_name column
        source_slice: Name of the labeled slice
        target_slices: Names of slices to propagate to
        regions: Regions from the source slice
        transform_matrix: Optional affine transform to apply to regions

    Returns:
        Modified AnnData with propagated labels
    """
    # Get region columns from source
    region_cols = [c for c in adata.obs.columns if c.startswith("in_")]

    if not region_cols:
        print("No region columns found - running initial labeling first")
        return adata

    # For each target slice, apply the same regions
    # (optionally with a spatial transform from PASTE alignment)
    for target_slice in target_slices:
        mask = adata.obs["slice_name"] == target_slice
        if mask.sum() == 0:
            continue

        coords = adata[mask].obsm["spatial"]

        # Apply transform if provided (from PASTE alignment matrices)
        if transform_matrix is not None:
            # Affine transform: [x', y'] = M @ [x, y, 1]
            ones = np.ones((len(coords), 1))
            coords_h = np.hstack([coords, ones])
            coords = (transform_matrix @ coords_h.T).T[:, :2]

        # Test against original regions
        assignments = point_in_regions(coords, regions)

        for region_name, in_region in assignments.items():
            col_name = f"in_{region_name}"
            if col_name in adata.obs.columns:
                adata.obs.loc[mask, col_name] = in_region

    return adata


def get_slice_from_annotation(
    annotation_path: Path, h5ad_files: list[Path]
) -> Optional[Path]:
    """
    Match an annotation file to its corresponding h5ad slice.

    Uses the slide/region info from the annotation filename to find
    the matching data file.
    """
    info = parse_annotation_filename(annotation_path)
    if not info:
        return None

    # Pattern: look for files containing the slide and region numbers
    slide = info["slide"]
    region = info["region"]

    # Try various naming conventions
    patterns = [
        f"_slide{slide}_region{region}",
        f"_{slide}_{region}",
        f"_{region}A",  # Region 7 might be file "7A"
        f"_{region}B",
    ]

    for h5ad_file in h5ad_files:
        for pattern in patterns:
            if pattern.lower() in h5ad_file.stem.lower():
                return h5ad_file

    return None


def label_single_slice(
    h5ad_path: Path,
    geojson_path: Path,
    output_path: Optional[Path] = None,
) -> sc.AnnData:
    """
    Label a single slice with GeoJSON annotations.

    Args:
        h5ad_path: Path to h5ad file
        geojson_path: Path to GeoJSON annotation file
        output_path: Optional path to save labeled data

    Returns:
        Labeled AnnData
    """
    print(f"Loading {h5ad_path.name}...")
    adata = sc.read_h5ad(h5ad_path)

    print(f"Loading annotations from {geojson_path.name}...")
    regions = load_geojson_regions(geojson_path)
    print(f"  Found {len(regions)} regions: {list(regions.keys())}")

    print("Assigning cells to regions...")
    adata = add_labels_to_adata(adata, regions)

    # Report statistics
    for region_name in regions:
        col = f"in_{region_name}"
        if col in adata.obs:
            n_cells = adata.obs[col].sum()
            pct = 100 * n_cells / adata.n_obs
            print(f"  {region_name}: {n_cells:,} cells ({pct:.1f}%)")

    if output_path:
        adata.write_h5ad(output_path)
        print(f"Saved: {output_path}")

    return adata


# =============================================================================
# CLI (analysis_workflow.py imports the functions above)
# =============================================================================

if __name__ == "__main__":
    import typer

    app = typer.Typer(help="GeoJSON-based anatomical labeling")

    @app.command()
    def list_annotations():
        """List available annotation files."""
        annotations = discover_annotations()
        if not annotations:
            typer.echo(f"No annotation files found in {ANNOTATIONS_DIR}")
            return

        for ann in annotations:
            info = parse_annotation_filename(ann)
            typer.echo(f"  {ann.name}")
            if info:
                typer.echo(f"    Slide {info['slide']}, Region {info['region']}")

            # Quick peek at regions
            regions = load_geojson_regions(ann)
            typer.echo(f"    Regions: {', '.join(regions.keys())}")

    @app.command()
    def label(
        h5ad: str = typer.Argument(..., help="Input h5ad file"),
        annotation: str = typer.Argument(..., help="GeoJSON annotation file"),
        output: Optional[str] = typer.Option(None, help="Output h5ad file"),
    ):
        """Label a slice with GeoJSON annotations."""
        h5ad_path = DATA_DIR / h5ad if not Path(h5ad).is_absolute() else Path(h5ad)
        ann_path = (
            ANNOTATIONS_DIR / annotation
            if not Path(annotation).is_absolute()
            else Path(annotation)
        )

        if not h5ad_path.exists():
            typer.echo(f"File not found: {h5ad_path}")
            raise typer.Exit(1)

        if not ann_path.exists():
            typer.echo(f"Annotation not found: {ann_path}")
            raise typer.Exit(1)

        out_path = Path(output) if output else None
        label_single_slice(h5ad_path, ann_path, out_path)

    @app.command()
    def test():
        """Test labeling with first available annotation."""
        annotations = discover_annotations()
        if not annotations:
            typer.echo("No annotations found")
            raise typer.Exit(1)

        ann = annotations[0]
        typer.echo(f"Testing with: {ann.name}")

        regions = load_geojson_regions(ann)
        typer.echo(f"Loaded {len(regions)} regions:")
        for name, data in regions.items():
            poly = data["polygon"]
            color = data["color"]
            typer.echo(f"  {name}: {poly.area:.0f} sq units, color={color}")

    app()
