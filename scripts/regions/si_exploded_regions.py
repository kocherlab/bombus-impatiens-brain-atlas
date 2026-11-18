#!/usr/bin/env python3
"""Supplementary figure: Exploded anatomical regions.

Generates a publication-quality "exploded view" of a single brain section:
- Left hemisphere: flat colored regions at z=0 (labeled reference)
- Right hemisphere: regions lifted vertically with connector lines

Region outlines are drawn from the actual GeoJSON annotations.

    PYTHONPATH=src .venv/bin/python scripts/regions/si_exploded_regions.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from shapely.geometry import Polygon
from shapely.ops import unary_union

from bimp_st.nine_region_common import (
    DESCRIPTIVE_REGIONS,
    REGION_COLORS,
    attach_region_flags,
    canonical_name,
)
from bimp_st.nine_region_common import (
    display as region_display,
)
from bimp_st.si_common import OUTPUT_DIR, set_style

# ── Configuration ────────────────────────────────────────────────────────────
# Queen slide 0040243 section 6, crop 2_6B: a section carrying all eight
# regions (al 330 cells, central complex 184).
#
# Five queen sections carry all eight (0040243 R4-R8) and, with annotations_v4,
# so do five worker sections (0068401 R5/R7, 0069041 R3-R5), so the queen
# section is representative, not forced. (Under annotations_v3 no worker
# section qualified.)
#
# The section must really hold every region in DESCRIPTIVE_REGIONS; otherwise
# the caption would describe regions the artwork does not show.
RAW_H5AD = (
    "pipelines/wholebrain/queen/data/"
    "output-XETG00151__0040243__Region_6__20251204__195015_2_6B.h5ad"
)
# v2/ (not v1/) -- that's where the antennal lobe and the
# central complex are actually drawn. The file carries a "central complex"
# polygon that canonical_name() resolves to the pooled `central_complex` label
# (it splits it into EB/FB only when POOL_CENTRAL_COMPLEX is off).
GEOJSON = (
    "data/annotations/queen/v2/"
    "output-XETG00151__0040243__Region_6__20251204__195015.geojson"
)
PIXEL_SCALE = 0.2125

# Shared across figures via nine_region_common.REGION_COLORS so a region is
# never a different colour in two figures (see also fig_atlas_overview.py,
# fig_atlas_structure.py).
REGION_PALETTE = dict(REGION_COLORS)

# Lift order, medial and deep at the bottom, lateral and superficial at the
# top. The rung a region gets is the only ordering information this figure
# carries, so it should say something true: the central-brain structures (the
# calyx compartments, the antennal lobe, the central complex) sit low, and the
# optic neuropils stack above them in the order the visual system runs, lobula
# -> medulla -> lamina, ending with the lamina outermost.
#
# The antennal lobe is an anterior, medial structure, so it sits with the
# central brain rather than above the lamina.
#
# The known order is filtered by the live vocabulary so both settings of
# nine_region_common.POOL_CENTRAL_COMPLEX work (pooled, canonical_name returns
# "central_complex"). The check below makes a region missing from this order
# fail loudly instead of being silently left out of the figure.
_LIFT_ORDER = [
    "kc_nc",
    "kc_ic",
    "kc_oc",
    "central_complex",
    "ellipsoid_body",
    "fan_shaped_body",
    "al",
    "lobula",
    "medulla",
    "lamina",
]
_unknown = [r for r in DESCRIPTIVE_REGIONS if r not in _LIFT_ORDER]
if _unknown:
    raise SystemExit(
        f"figS_exploded_regions: {_unknown} is in DESCRIPTIVE_REGIONS but has "
        "no place in _LIFT_ORDER. Give it one (and a colour in "
        "nine_region_common.REGION_COLORS) rather than letting it drop out of "
        "the figure."
    )
REGION_ORDER = [r for r in _LIFT_ORDER if r in DESCRIPTIVE_REGIONS]

# One rung per region, 250 apart, in REGION_ORDER. Deriving the values rather
# than writing them out means a region can never be in the draw order without a
# rung.
Z_OFFSETS = {r: 250 * (i + 1) for i, r in enumerate(REGION_ORDER)}

# Hemisphere split: offset from Y median (µm)
HEMISPHERE_OFFSET = 150


# ── Region helpers ───────────────────────────────────────────────────────────


def get_region_labels(adata):
    """Extract region labels from ``in_*`` boolean columns.

    al/ellipsoid_body/fan_shaped_body are nullable booleans -- pd.NA means
    "this section has no tracing for that region yet", not "outside the
    region". ``.values.astype(bool)`` raises on the NA entries; a plain
    ``.fillna(False)`` would manufacture an absence out of missing data. For
    this single-section coverage render, NA cells simply get no region tag
    (they land in "unlabeled" with genuinely out-of-region cells) -- dropping
    them from the scatter instead would leave visible holes in the section, so
    this is a display simplification, not a quantitative claim.
    """
    in_cols = [c for c in adata.obs.columns if c.startswith("in_")]
    labels = np.full(adata.n_obs, "unlabeled", dtype=object)
    for col in in_cols:
        region = col.replace("in_", "")
        mask = adata.obs[col].to_numpy(dtype=bool, na_value=False)
        labels[mask] = region
    return labels


def format_region_label(region):
    """Pretty-print a region name for legends.

    Delegates to nine_region_common.display() so a region's legend text is
    the same everywhere (matches fig_s_regional_diversity,
    fig_s_regional_composition, figS_tangram_confidence, fig_atlas_overview.py).
    """
    return region_display(region)


# ── GeoJSON loading ──────────────────────────────────────────────────────────


def load_geojson_polygons(path, pixel_scale, y_min, y_max):
    """Load GeoJSON and return merged Shapely polygons per region.

    Name canonicalization (aliases like "lobulla", and splitting a drawn
    "central complex" polygon into ellipsoid_body/fan_shaped_body via the
    ``subregion``/``_eb``/``_fb`` convention) is delegated to
    nine_region_common.canonical_name -- not re-derived here, so this script
    can't silently diverge from the shared vocabulary.
    """
    with open(path) as f:
        data = json.load(f)

    by_name = {}
    for feat in data["features"]:
        props = feat.get("properties", {}) or {}
        name = canonical_name(props)
        if not name:
            continue
        coords = feat["geometry"]["coordinates"][0]
        scaled = [(x * pixel_scale, y * pixel_scale) for x, y in coords]
        if len(scaled) < 3:
            continue
        poly = Polygon(scaled)
        if not poly.is_valid:
            continue
        if poly.centroid.y < y_min or poly.centroid.y > y_max:
            continue
        by_name.setdefault(name, []).append(poly)

    return {name: unary_union(polys) for name, polys in by_name.items()}


# ── 3D drawing helpers ───────────────────────────────────────────────────────


def _polygon_geoms(polys):
    """Yield individual Polygon geometries from a Polygon or MultiPolygon."""
    if polys is None:
        return
    if polys.geom_type == "MultiPolygon":
        yield from polys.geoms
    else:
        yield polys


def draw_geojson_polys(ax, polys, color, z=0.0, alpha=0.12, lw=1.8):
    """Draw GeoJSON polygon(s) as filled footprints at a given z."""
    for g in _polygon_geoms(polys):
        ox, oy = g.exterior.coords.xy
        ox, oy = np.array(ox), np.array(oy)
        verts = [list(zip(ox, oy, [z] * len(ox), strict=False))]
        p3d = Poly3DCollection(
            verts,
            alpha=alpha,
            facecolor=color,
            edgecolor=color,
            linewidth=lw,
        )
        ax.add_collection3d(p3d)


def get_outline_points(polys, n_max=40):
    """Extract evenly-spaced outline points from polygon(s)."""
    all_pts = []
    for g in _polygon_geoms(polys):
        ox, oy = g.exterior.coords.xy
        all_pts.extend(zip(ox, oy, strict=False))
    if not all_pts:
        return None
    pts = np.array(all_pts)
    if len(pts) <= n_max:
        return pts
    step = max(1, len(pts) // n_max)
    return pts[::step]


def autocrop_whitespace(path, pad=20):
    """Crop white borders from a saved PNG."""
    img = Image.open(path)
    bg = Image.new(img.mode, img.size, (255, 255, 255))
    diff = Image.fromarray(
        np.abs(np.array(img, dtype=np.int16) - np.array(bg, dtype=np.int16)).astype(
            np.uint8
        )
    )
    bbox = diff.convert("L").point(lambda x: 255 if x > 5 else 0).getbbox()
    if bbox:
        crop_box = (
            max(0, bbox[0] - pad),
            max(0, bbox[1] - pad),
            min(img.width, bbox[2] + pad),
            min(img.height, bbox[3] + pad),
        )
        # Keep the resolution metadata the original savefig wrote.
        dpi = img.info.get("dpi", (600, 600))
        img.crop(crop_box).save(path, dpi=tuple(round(d) for d in dpi))


# ── Main plotting function ───────────────────────────────────────────────────


def plot_exploded_regions(adata, flat_polys, lift_polys, y_mid, output_dir):
    """Exploded 3D: left hemisphere flat-colored, right hemisphere lifted."""
    print("  Exploded regions...", flush=True)

    coords = np.asarray(adata.obsm["spatial"])
    labels = get_region_labels(adata)

    y_cut = y_mid + HEMISPHERE_OFFSET
    flat_mask = coords[:, 1] <= y_cut
    lift_mask = coords[:, 1] > y_cut

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Base layer: all cells at z=0 in gray
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        np.zeros(len(coords)),
        c="#D0D0D0",
        s=0.5,
        alpha=0.35,
        edgecolors="none",
        rasterized=True,
        depthshade=False,
    )

    handles = {}
    for region in REGION_ORDER:
        color = REGION_PALETTE[region]
        z_off = Z_OFFSETS[region]

        # Left hemisphere: flat colored cells + GeoJSON outlines
        fm = (labels == region) & flat_mask
        if fm.sum() > 0:
            ax.scatter(
                coords[fm, 0],
                coords[fm, 1],
                np.zeros(fm.sum()),
                c=color,
                s=1.5,
                alpha=0.7,
                edgecolors="none",
                rasterized=True,
                depthshade=False,
            )
        draw_geojson_polys(
            ax,
            flat_polys.get(region),
            color,
            z=0.0,
            alpha=0.08,
            lw=1.2,
        )

        # Right hemisphere: GeoJSON outlines on base and raised
        draw_geojson_polys(
            ax,
            lift_polys.get(region),
            color,
            z=0.0,
            alpha=0.12,
            lw=1.8,
        )
        draw_geojson_polys(
            ax,
            lift_polys.get(region),
            color,
            z=z_off,
            alpha=0.10,
            lw=1.5,
        )

        # Right hemisphere: lifted cells
        lm = (labels == region) & lift_mask
        n_lift = int(lm.sum())

        if n_lift:
            lx, ly = coords[lm, 0], coords[lm, 1]
            ax.scatter(
                lx,
                ly,
                np.full(n_lift, z_off),
                c=color,
                s=2.0,
                alpha=0.8,
                edgecolors="none",
                rasterized=True,
                depthshade=False,
            )

            # Connector lines from outline border points
            pts = get_outline_points(lift_polys.get(region), n_max=6)
            if pts is not None:
                for px, py in pts:
                    ax.plot(
                        [px, px],
                        [py, py],
                        [0, z_off],
                        color=color,
                        lw=0.5,
                        alpha=0.3,
                    )

        # Legend tracks what is drawn, not only what is lifted: a region with
        # no cells in the lifted hemisphere (e.g. the midline central-complex
        # substructures) is still plotted flat and needs a legend entry.
        if int(fm.sum()) or n_lift:
            handles[region] = Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=7,
                label=format_region_label(region),
            )
            print(f"      {region}: {int(fm.sum())} flat + {n_lift} lifted", flush=True)

    # View and layout
    ax.view_init(elev=25, azim=-65)
    ax.set_axis_off()
    ax.dist = 1.5
    ax.set_position([-0.25, -0.35, 1.5, 1.7])

    x_rng = coords[:, 0].max() - coords[:, 0].min()
    y_rng = coords[:, 1].max() - coords[:, 1].min()
    z_rng = max(Z_OFFSETS.values())
    max_rng = max(x_rng, y_rng, z_rng)
    ax.set_box_aspect([x_rng / max_rng, y_rng / max_rng, z_rng / max_rng])

    # The central complex is midline, so it is drawn flat only, with no lifted
    # copy to connect it to: name it in place, in its own palette colour.
    cx = flat_polys.get("central_complex")
    if cx is not None and "central_complex" in handles:
        from mpl_toolkits.mplot3d import proj3d

        c = cx.centroid
        px, py, _ = proj3d.proj_transform(c.x, c.y, 0.0, ax.get_proj())
        ax.annotate(
            format_region_label("central_complex"),
            xy=(px, py),
            xycoords="data",
            xytext=(-22, -8),
            textcoords="offset points",
            fontsize=5.0,
            color=REGION_PALETTE["central_complex"],
            fontweight="bold",
            ha="left",
            va="top",
            zorder=20,
        )

    # Two titled legends, one per group, each in an empty corner near the
    # structures it keys: the Kenyon-cell compartments (top left, beside the
    # lifted calyx columns) and the other neuropils (bottom right, below the
    # lifted optic lobe).
    kc = [r for r in REGION_ORDER if r.startswith("kc_") and r in handles]
    other = [r for r in REGION_ORDER if not r.startswith("kc_") and r in handles]
    leg_kw = dict(
        fontsize=7,
        title_fontsize=7.5,
        frameon=False,
        borderpad=0,
        labelspacing=0.3,
        handletextpad=0.3,
        alignment="left",
    )
    fig.legend(
        handles=[handles[r] for r in kc],
        title="Kenyon cell compartments",
        loc="upper left",
        bbox_to_anchor=(0.20, 0.93),
        **leg_kw,
    )
    fig.legend(
        handles=[handles[r] for r in other[::-1]],
        title="Other neuropils",
        loc="lower right",
        bbox_to_anchor=(0.93, 0.12),
        **leg_kw,
    )

    # Save and auto-crop
    out = output_dir / "figS_exploded_regions.png"
    fig.savefig(out, dpi=600)
    plt.close(fig)
    autocrop_whitespace(out)
    print(f"    Saved {out}", flush=True)


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from bimp_st.paths import explain_missing_inputs

    with explain_missing_inputs():
        set_style()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        print(f"Loading {RAW_H5AD}...", flush=True)
        adata = sc.read_h5ad(RAW_H5AD)
        sp = adata.obsm["spatial"]
        y_mid = np.median(sp[:, 1])
        y_cut = y_mid + HEMISPHERE_OFFSET
        print(f"  {adata.n_obs:,} cells, Y cut at {y_cut:.0f}", flush=True)

        # This raw per-crop h5ad ships the original six in_* columns but no
        # slice_name column; the nine-region sidecar table is keyed on
        # (slice_name, cell_id), and its slice_name for this crop IS the crop's
        # own filename stem (verified against results/nine_region/crops/worker/).
        adata.obs["slice_name"] = Path(RAW_H5AD).stem
        # Refresh all nine flags, not just the three new ones. The six baked into
        # the atlas object predate polygon repair in the sidecar (e.g. four polygons on
        # worker 0068401 R1 move 119 cells from lobula to antennal lobe), so mixing
        # sources would draw those cells as both.
        attach_region_flags(adata, regions=DESCRIPTIVE_REGIONS)

        # Load GeoJSON polygons for each hemisphere
        print("Loading GeoJSON...", flush=True)
        flat_polys = load_geojson_polygons(
            GEOJSON,
            PIXEL_SCALE,
            y_min=sp[:, 1].min() - 100,
            y_max=y_cut,
        )
        lift_polys = load_geojson_polygons(
            GEOJSON,
            PIXEL_SCALE,
            y_min=y_cut,
            y_max=sp[:, 1].max() + 100,
        )
        for name in REGION_ORDER:
            nf = nl = 0
            if name in flat_polys:
                p = flat_polys[name]
                nf = len(p.geoms) if p.geom_type == "MultiPolygon" else 1
            if name in lift_polys:
                p = lift_polys[name]
                nl = len(p.geoms) if p.geom_type == "MultiPolygon" else 1
            print(f"  {name}: flat={nf}, lift={nl} polygon(s)")

        plot_exploded_regions(adata, flat_polys, lift_polys, y_mid, OUTPUT_DIR)
        print(f"\nDone. Output in {OUTPUT_DIR}/", flush=True)
