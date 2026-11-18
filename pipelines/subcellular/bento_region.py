#!/usr/bin/env python3
"""
Bento subcellular analysis per Xenium region -- Typer CLI.

Same analysis flow as the bento-tools Xenium tutorial:
  Format Xenium data -> shape stats -> RNAflux -> fluxmap -> LP -> colocation -> comp

Outputs per region:
  - crop_shape_stats.png -- cell/nucleus area, aspect ratio, density
  - crop_flux.png -- RNAflux per-transcript embedding (local composition)
  - crop_fluxmap.png -- SOM-clustered flux domains vectorized as polygons
  - crop_lp_dist.png -- UpSet plot of subcellular localization patterns
  - crop_lp_genes.png -- RadViz projection of gene localization
  - crop_colocation.png -- gene colocalization tensor decomposition
  - crop_comp.png -- gene composition across cell/nucleus compartments

Zarr caches (raw + prepped) are stored under --cache-dir to skip expensive
reloads on reruns.
"""

import sys
import warnings

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
warnings.filterwarnings("ignore")

import typer

app = typer.Typer()


def _setup_style():
    """Publication-quality matplotlib defaults."""
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _transcript_scale(sdata):
    """Tutorial: ``sdata['transcripts'].attrs['transform']['global']``."""
    from spatialdata.transformations import get_transformation

    tx = sdata["transcripts"]
    try:
        return tx.attrs["transform"]["global"]
    except KeyError:
        pass
    return get_transformation(tx)


def _plot_lp_genes_by_pattern(sdata, fname, alpha=0.6, annotate=0):
    """RadViz of localization patterns colored by each gene's dominant pattern.

    Bento's ``bt.pl.lp_genes`` colors by mean expression (logcounts) -- that
    obscures which subcellular region each gene actually prefers. This recolors
    by argmax pattern using bento's own palette.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    from bento.plotting._lp import PATTERN_COLORS, PATTERN_NAMES
    from bento.tools._lp import lp_stats

    lp_stats(sdata)
    table = sdata.tables["table"]
    n_cells = table.n_obs
    gene_frac = table.uns["lp_stats"][PATTERN_NAMES] / n_cells

    # Drop genes with zero predictions
    gene_frac = gene_frac[gene_frac.sum(axis=1) > 0]

    # Dominant pattern per gene
    dominant = gene_frac.idxmax(axis=1)
    fraction = gene_frac.max(axis=1)

    # RadViz: project N-d gene fractions onto 2D pentagon vertices
    fig, ax = plt.subplots(figsize=(11, 11))
    df = gene_frac.copy()
    df["_class"] = ""  # required by pd.plotting.radviz
    pd.plotting.radviz(df, "_class", s=0, ax=ax)
    ax.get_legend().remove()

    # Extract scattered point coords from the radviz collection
    pts = []
    for c in ax.collections:
        pts.extend(c.get_offsets().data)
    pts = np.array(pts).reshape(-1, 2)
    xy = pd.DataFrame(pts, index=gene_frac.index, columns=["x", "y"])
    xy["pattern"] = dominant
    xy["max_frac"] = fraction
    # Cell fraction (% of cells expressing) for size
    xy["cell_pct"] = (
        100
        * (table.X > 0).sum(axis=0).A1[table.var_names.get_indexer(xy.index)]
        / n_cells
    )

    palette = dict(zip(PATTERN_NAMES, PATTERN_COLORS, strict=False))

    # KDE backdrop for density
    sns.kdeplot(
        data=xy,
        x="x",
        y="y",
        fill=True,
        cmap="binary",
        zorder=0.9,
        ax=ax,
        levels=8,
    )

    # Hide rectangular axes (radviz only needs the pentagon)
    vertices = ax.patches[1:]
    poly = plt.Polygon(
        [v.center for v in vertices],
        facecolor="none",
        edgecolor="#444",
        zorder=1,
        linewidth=1.2,
    )
    ax.add_patch(poly)
    for v in vertices:
        v.remove()

    sns.scatterplot(
        data=xy,
        x="x",
        y="y",
        hue="pattern",
        hue_order=PATTERN_NAMES,
        palette=palette,
        size="cell_pct",
        sizes=(15, 350),
        size_norm=(0, 100),
        alpha=alpha,
        edgecolor="white",
        linewidth=0.5,
        ax=ax,
    )

    # Annotate top-N genes by max localization fraction (excluding "none")
    if annotate and annotate > 0:
        try:
            from adjustText import adjust_text

            adjust = True
        except ImportError:
            adjust = False
        non_none = xy[xy["pattern"] != "none"].copy()
        non_none["sort_score"] = non_none["max_frac"] * non_none["cell_pct"]
        top = non_none.nlargest(annotate, "sort_score")
        texts = []
        for gene, row in top.iterrows():
            short = gene.split("|")[0] if "|" in gene else gene
            texts.append(
                ax.text(
                    row["x"],
                    row["y"],
                    short,
                    fontsize=9,
                    color="#222",
                    ha="center",
                    va="center",
                    zorder=5,
                )
            )
        if adjust and texts:
            adjust_text(
                texts,
                ax=ax,
                expand=(1.5, 1.8),
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.5, alpha=0.7),
            )

    ax.axis(False)
    ax.set_title("Gene localization patterns -- colored by dominant region")
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=11,
    )
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)


@app.command()
def run(
    xenium_path: str = typer.Option(..., help="Path to Xenium output directory"),
    brain: str = typer.Option(..., help="Brain ID (queen / worker)"),
    region: str = typer.Option(..., help="Region folder name"),
    output_dir: str = typer.Option(
        None, help="Output directory for figures (default: results/{brain}/{region}/)"
    ),
    cache_dir: str = typer.Option(
        None, help="Cache directory for zarr files (default: data/{brain}/{region}/)"
    ),
    full_slice: bool = typer.Option(
        True, help="Process full slice (skip slow rendering steps)"
    ),
    num_workers: int = typer.Option(64, help="Number of parallel workers"),
    flux_res: int = typer.Option(1, help="RNAflux resolution"),
    skip_lp: bool = typer.Option(False, help="Skip localization pattern analysis"),
    skip_coloc: bool = typer.Option(True, help="Skip colocalization analysis"),
):
    """Run bento subcellular analysis on a single Xenium region."""
    from pathlib import Path

    import bento as bt
    import matplotlib.pyplot as plt
    import spatialdata as sd
    from spatialdata.models import PointsModel
    from spatialdata.transformations import Scale
    from spatialdata_io import xenium

    _setup_style()

    # Resolve output and cache dirs
    if output_dir is None:
        output_dir = f"results/bento/{brain}/{region}"
    if cache_dir is None:
        cache_dir = f"data/bento/{brain}/{region}"

    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    CROP_PATH = Path(cache_dir) / "raw.zarr"
    PREPPED_PATH = Path(cache_dir) / "prepped.zarr"
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    crop = None

    # -- Format Xenium data ----------------------------------------------------
    if not CROP_PATH.exists():
        xdir = Path(xenium_path)
        print(f"Loading Xenium from {xdir}...", flush=True)
        sdata = xenium(
            xdir,
            morphology_mip=False,
            morphology_focus=False,
            aligned_images=False,
            cells_labels=False,
            nucleus_labels=False,
        )

        scale = _transcript_scale(sdata)
        scale2d = Scale(scale.to_scale_vector(["x", "y"]), axes=["x", "y"])
        sd.transformations.set_transformation(sdata.points["transcripts"], scale2d)

        tx = sdata.points["transcripts"]
        tfm = sd.transformations.get_transformation(tx, get_all=True)
        tx2d = tx.drop("z", axis=1)
        tx2d.attrs.clear()
        sdata.points["transcripts"] = PointsModel.parse(
            tx2d,
            coordinates={"x": "x", "y": "y"},
            feature_key="feature_name",
            instance_key="cell_id",
            transformations=tfm,
        )

        if "table" in sdata.tables:
            del sdata.tables["table"]

        if full_slice:
            print("Using full slice (no crop)...", flush=True)
            crop = sdata
        else:
            # Centroids are in local coords; transform to global for the query.
            cells = sdata.shapes["cell_boundaries"]
            cx_local = cells.geometry.centroid.x.median()
            cy_local = cells.geometry.centroid.y.median()
            cb_tfm = sd.transformations.get_transformation(
                cells,
                to_coordinate_system="global",
            )
            scale_xy = cb_tfm.to_scale_vector(["x", "y"])
            cx = cx_local * scale_xy[0]
            cy = cy_local * scale_xy[1]
            h = 1500
            print(
                f"Cropping around global ({cx:.0f}, {cy:.0f}) +/-{h}...",
                flush=True,
            )
            crop = sd.bounding_box_query(
                sdata,
                axes=["x", "y"],
                min_coordinate=[cx - h, cy - h],
                max_coordinate=[cx + h, cy + h],
                target_coordinate_system="global",
                filter_table=False,
            )

        for key in ("morphology_focus",):
            if key in crop.images:
                del crop.images[key]
        for key in ("cell_labels", "nucleus_labels"):
            if key in crop.labels:
                del crop.labels[key]

        if "transcripts" not in crop.points:
            raise RuntimeError(
                "Crop has no transcripts (empty bounding box). Check the Xenium path."
            )

        crop["transcripts"].attrs["spatialdata_attrs"]["instance_key"] = (
            "cell_boundaries"
        )

        nc = len(crop.shapes["cell_boundaries"])
        nt = len(crop.points["transcripts"].compute())
        print(f"Slice: {nc} cells, {nt} transcripts", flush=True)

        crop.write(str(CROP_PATH), overwrite=True)
        print(f"Saved -> {CROP_PATH}", flush=True)
    else:
        print(f"Crop cache hit ({CROP_PATH})", flush=True)

    # -- Load crop (prefer post-prep cache) ------------------------------------
    if PREPPED_PATH.exists():
        print(f"Loading prepped cache ({PREPPED_PATH})...", flush=True)
        crop = sd.read_zarr(str(PREPPED_PATH))
        n_cells = len(crop.shapes["cell_boundaries"])
        n_tx = len(crop.points["transcripts"].compute())
        print(
            f"Prepped loaded: {n_cells} cells, {n_tx} transcripts",
            flush=True,
        )
    else:
        if crop is None:
            crop = sd.read_zarr(str(CROP_PATH))
        n_cells = len(crop.shapes["cell_boundaries"])
        n_tx = len(crop.points["transcripts"].compute())
        print(f"Crop loaded: {n_cells} cells, {n_tx} transcripts", flush=True)
        print("Running bt.io.prep (spatial join)...", flush=True)
        bt.io.prep(crop)
        print(f"Saving prepped cache -> {PREPPED_PATH}...", flush=True)
        crop.write(str(PREPPED_PATH), overwrite=True)

    # Bigger figures + finer flux when running the full slice.
    if full_slice:
        figsize = (28, 22)
        fluxmap_clusters = range(3, 11)  # 3-10 SOM clusters
        fluxmap_iters = 2000
        fluxmap_min_count = 30
        coloc_ranks = [3, 5, 7, 9, 11]
        coloc_iters = 5
        coloc_plot_rank = 7
        lp_annotate = 20
    else:
        figsize = (14, 10)
        fluxmap_clusters = range(2, 6)
        fluxmap_iters = 1000
        fluxmap_min_count = 50
        coloc_ranks = [3, 5, 7]
        coloc_iters = 3
        coloc_plot_rank = 5
        lp_annotate = 10

    # -- 1. Spatial scatter + density ------------------------------------------
    if full_slice:
        print(
            "Skipping points + density (too slow at 20M+ tx / 50K cells)",
            flush=True,
        )
    else:
        print("Plotting points + density...", flush=True)
        fig, ax = plt.subplots(figsize=figsize)
        bt.pl.points(
            crop,
            ax=ax,
            shapes=["cell_boundaries", "nucleus_boundaries"],
        )
        ax.set_title("Transcript locations -- cell + nucleus boundaries")
        fig.savefig(OUT / "crop_points.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=figsize)
        bt.pl.density(
            crop,
            ax=ax,
            shapes=["cell_boundaries", "nucleus_boundaries"],
        )
        ax.set_title("Transcript density")
        fig.savefig(OUT / "crop_density.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # -- 2. Shape statistics ---------------------------------------------------
    print("Computing shape stats...", flush=True)
    bt.tl.shape_stats(crop)
    bt.pl.shape_stats(crop, fname=str(OUT / "crop_shape_stats.png"))
    plt.close("all")

    # -- 3. RNAflux ------------------------------------------------------------
    print(f"Computing flux (res={flux_res})...", flush=True)
    bt.tl.flux(crop, res=flux_res, recompute=True, num_workers=num_workers)
    if full_slice:
        print(
            "Skipping bt.pl.flux render (50K cells x raster too slow)",
            flush=True,
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        bt.pl.flux(crop, res=flux_res, ax=ax)
        ax.set_title("RNAflux -- local transcriptome composition")
        fig.savefig(OUT / "crop_flux.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # -- 4. Fluxmap (SOM clustering) -------------------------------------------
    print(
        f"Computing fluxmap (clusters={list(fluxmap_clusters)}, "
        f"iters={fluxmap_iters})...",
        flush=True,
    )
    bt.tl.fluxmap(
        crop,
        n_clusters=fluxmap_clusters,
        res=flux_res,
        num_iterations=fluxmap_iters,
        min_count=fluxmap_min_count,
        train_size=1,
        plot_error=True,
    )
    if full_slice:
        print(
            "Skipping bt.pl.fluxmap render (50K cells x clusters too slow)",
            flush=True,
        )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        bt.pl.fluxmap(crop, ax=ax)
        ax.set_title("Fluxmap -- subcellular transcriptome domains (SOM)")
        fig.savefig(OUT / "crop_fluxmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # -- 5. Localization patterns ----------------------------------------------
    if skip_lp:
        print("Skipping localization patterns (--skip-lp)", flush=True)
    else:
        print("Predicting localization patterns...", flush=True)
        try:
            bt.tl.lp(crop, num_workers=num_workers)

            # Check predictions aren't all zeros (broken XGB pickle)
            lp_df = crop.tables["table"].uns["lp"]
            pattern_cols = [
                "cell_edge",
                "cytoplasmic",
                "none",
                "nuclear",
                "nuclear_edge",
            ]
            if lp_df[pattern_cols].sum().sum() == 0:
                print(
                    "  LP model produced all-zero predictions (xgboost pickle compat)",
                    flush=True,
                )
            else:
                bt.pl.lp_dist(crop, fname=str(OUT / "crop_lp_dist.png"))
                plt.close("all")
                _plot_lp_genes_by_pattern(
                    crop,
                    fname=str(OUT / "crop_lp_genes.png"),
                    alpha=0.7,
                    annotate=lp_annotate,
                )
                plt.close("all")
                print("  LP plots saved", flush=True)

                # Save per-cell LP predictions for downstream differential
                # analysis (the prepped zarr only has spatialdata_attrs,
                # not the LP results).
                for key in ("lp", "lpp", "lp_stats"):
                    if key in crop.tables["table"].uns:
                        crop.tables["table"].uns[key].to_parquet(OUT / f"{key}.parquet")
                print(f"  LP data saved -> {OUT}/lp*.parquet", flush=True)
        except Exception as e:
            print(f"  Localization patterns skipped: {e}", flush=True)

    # -- 6. Colocalization -----------------------------------------------------
    if skip_coloc:
        print("Skipping colocalization (--skip-coloc)", flush=True)
    elif full_slice:
        print(
            "Skipping colocalization (O(cells x genes^2) too slow at 50K cells)",
            flush=True,
        )
    else:
        print(
            f"Computing colocalization (ranks={coloc_ranks}, iters={coloc_iters})...",
            flush=True,
        )
        try:
            bt.tl.coloc_quotient(
                crop,
                shapes=["cell_boundaries"],
                num_workers=num_workers,
            )
            bt.tl.colocation(crop, ranks=coloc_ranks, iterations=coloc_iters)
            bt.pl.colocation(
                crop,
                rank=coloc_plot_rank,
                fname=str(OUT / "crop_colocation.png"),
            )
            plt.close("all")
        except Exception as e:
            print(f"  Colocalization skipped: {e}", flush=True)

    # -- 7. Gene composition ---------------------------------------------------
    print("Computing composition...", flush=True)
    try:
        bt.tl.comp(
            crop,
            points_key="transcripts",
            shape_names=["cell_boundaries", "nucleus_boundaries"],
        )
        bt.pl.comp(crop, fname=str(OUT / "crop_comp.png"))
        plt.close("all")
    except Exception as e:
        print(f"  Composition skipped: {e}", flush=True)

    print(f"Done -> {OUT}/", flush=True)


if __name__ == "__main__":
    app()
