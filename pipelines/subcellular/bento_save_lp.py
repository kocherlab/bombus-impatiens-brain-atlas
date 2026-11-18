#!/usr/bin/env python3
"""Save per-cell LP predictions from bento prepped zarrs.

Loads a prepped zarr, runs bt.tl.lp(), joins with cell metadata from the
aligned h5ad, and saves the result as parquet. This is the data needed for
differential subcellular localization analysis.

Usage:
    .venv/bin/python pipelines/subcellular/bento_save_lp.py \
        --brain queen --region output-XETG00151__0040243__Region_1__...
"""

import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import pandas as pd
import typer

app = typer.Typer()


@app.command()
def save_lp(
    brain: str = typer.Option(..., help="queen or worker"),
    region: str = typer.Option(..., help="Xenium region folder name"),
    cache_dir: str = typer.Option(None, help="Override cache dir"),
    output_dir: str = typer.Option(None, help="Override output dir"),
    num_workers: int = typer.Option(64),
):
    """Compute LP and save per-cell predictions + metadata as parquet."""
    import anndata as ad
    import bento as bt
    import spatialdata as sd

    from bimp_st.cluster_labels import get_short

    REPO = Path(__file__).resolve().parents[2]
    CACHE = Path(cache_dir) if cache_dir else REPO / f"data/bento/{brain}/{region}"
    OUT = Path(output_dir) if output_dir else REPO / f"results/bento/{brain}/{region}"
    OUT.mkdir(parents=True, exist_ok=True)

    lp_out = OUT / "lp.parquet"
    if lp_out.exists():
        print(f"LP parquet already exists: {lp_out}", flush=True)
        return

    # ── Load prepped zarr ─────────────────────────────────────────────────
    prepped = CACHE / "prepped.zarr"
    if not prepped.exists():
        print(f"No prepped zarr at {prepped}", flush=True)
        raise typer.Exit(1)

    print(f"Loading {prepped}...", flush=True)
    crop = sd.read_zarr(str(prepped))
    n_cells = len(crop.shapes["cell_boundaries"])
    print(f"  {n_cells} cells", flush=True)

    # ── Run LP ────────────────────────────────────────────────────────────
    print("Computing LP...", flush=True)
    bt.tl.lp(crop, num_workers=num_workers)

    lp_df = crop.tables["table"].uns["lp"]
    pattern_cols = ["cell_edge", "cytoplasmic", "none", "nuclear", "nuclear_edge"]
    if lp_df[pattern_cols].sum().sum() == 0:
        print("  LP model produced all-zero predictions — aborting", flush=True)
        raise typer.Exit(1)

    # ── Load gene name mapping ────────────────────────────────────────────
    mapping_csv = REPO / "data/panel/np31_to_v43_mapping.csv"
    gene_map = {}
    if mapping_csv.exists():
        mdf = pd.read_csv(mapping_csv)
        mdf["short"] = mdf["description"].str.extract(r"\[protein=(.+?)\]")[0]
        gene_map = dict(zip(mdf["np31_id"], mdf["short"], strict=True))

    def readable_name(raw_name):
        """BIMP_vNP3.1_00052|MGEN_v3.1_01175 → insulin-like peptide receptor."""
        for part in str(raw_name).split("|"):
            if part in gene_map:
                return gene_map[part]
        return raw_name

    # ── Load cell metadata ────────────────────────────────────────────────
    # Aligned h5ad has cell_id, slice_name, in_* anatomical flags
    aligned_path = (
        REPO / "pipelines/wholebrain/queen/data/aligned_clustered.h5ad"
        if brain == "queen"
        else REPO / "pipelines/wholebrain/worker/data/aligned_clustered.h5ad"
    )

    meta = None
    if aligned_path.exists():
        print(f"Loading cell metadata from {aligned_path.name}...", flush=True)
        adata = ad.read_h5ad(aligned_path, backed="r")
        meta = adata.obs.copy()
        adata.file.close()
        # cell_id column has the Xenium barcode (obs.index may be numeric)
        if "cell_id" not in meta.columns:
            meta["cell_id"] = meta.index
        # Filter to this region: slice_name starts with region folder
        if "slice_name" in meta.columns:
            meta = meta[meta["slice_name"].str.startswith(region)]
        meta = meta.drop_duplicates(subset="cell_id")
        print(f"  {len(meta)} cells matched to region", flush=True)

    # Tangram cell types
    tangram_path = REPO / "results/tangram/combined_clustered_g2_R2_s42_tangram.h5ad"
    tangram_ct = None
    if tangram_path.exists():
        print("Loading tangram cell types...", flush=True)
        tdata = ad.read_h5ad(tangram_path, backed="r")
        tobs = tdata.obs.copy()
        tdata.file.close()
        # Use the cell_id column (barcode), not obs.index (may be numeric)
        # Filter to this brain to avoid cross-brain barcode collisions
        if "brain" in tobs.columns:
            tobs = tobs[tobs["brain"] == brain]
        tangram_ct = tobs[["cell_id", "tangram_cell_type"]].copy()
        tangram_ct = tangram_ct.drop_duplicates(subset="cell_id")
        print(f"  {len(tangram_ct)} tangram cells for {brain}", flush=True)

    # ── Join LP predictions with metadata ─────────────────────────────────
    instance_key = "cell_boundaries"
    lp_df = lp_df.copy()
    lp_df["gene_name"] = lp_df["feature_name"].apply(readable_name)
    lp_df["brain"] = brain
    lp_df["region"] = region

    # Dominant pattern per gene-cell
    lp_df["dominant_pattern"] = lp_df[pattern_cols].idxmax(axis=1)
    lp_df.loc[lp_df[pattern_cols].sum(axis=1) == 0, "dominant_pattern"] = "none"

    # Join cell type from tangram
    if tangram_ct is not None:
        lp_df = lp_df.merge(
            tangram_ct,
            left_on=instance_key,
            right_on="cell_id",
            how="left",
        )
        # tangram_cell_type is a cell-type name string → short label
        if "tangram_cell_type" in lp_df.columns:
            lp_df["cell_type"] = lp_df["tangram_cell_type"].apply(
                lambda ct: get_short(ct) if pd.notna(ct) else "Unknown"
            )

    # Join anatomical labels from aligned h5ad
    if meta is not None and len(meta) > 0:
        in_cols = [c for c in meta.columns if c.startswith("in_")]
        if in_cols:
            lp_df = lp_df.merge(
                meta[["cell_id"] + in_cols],
                left_on=instance_key,
                right_on="cell_id",
                how="left",
            )

    # ── Save ──────────────────────────────────────────────────────────────
    lp_df.to_parquet(lp_out, index=False)
    print(f"Saved {len(lp_df)} rows → {lp_out}", flush=True)

    # Also save lp_stats (per-gene summary)
    if "lp_stats" in crop.tables["table"].uns:
        stats = crop.tables["table"].uns["lp_stats"].copy()
        stats["gene_name"] = stats.index.map(readable_name)
        stats["brain"] = brain
        stats["region"] = region
        stats.to_parquet(OUT / "lp_stats.parquet")
        print(f"Saved lp_stats → {OUT}/lp_stats.parquet", flush=True)


if __name__ == "__main__":
    app()
