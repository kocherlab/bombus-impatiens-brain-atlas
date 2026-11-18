"""MALDI-MSI data ingest: SCiLS .spydata / wide CSV → per-slice h5ad files.

Usage:
    python maldi_ingest.py from-csv      --csv <path> --sample-id queen_80524 \
        --label queen --out-dir data/slices
    python maldi_ingest.py from-spydata  --spydata <path> --out-dir data/slices
    python maldi_ingest.py from-scils    --slx <path> --feature-list <uuid> \
        --out-dir data/slices
    python maldi_ingest.py inventory      --slice-dir data/slices
"""

import pickle
import re
import tarfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import typer

app = typer.Typer(pretty_exceptions_enable=False)

# Metadata columns expected in the wide-CSV format
_META_COLS = {"spot_id", "sample_name", "region", "region_id", "x", "y"}
_FEATURE_PATTERN = re.compile(r"^(BIMP_v4\.3_\d+)-R(\d+)_\[M\+(H|Na|K)\]\+$")


def _strip_pandas_dup_suffix(col: str) -> str:
    return re.sub(r"\.\d+$", "", col)


def _build_var_table(unique_features: list[str], mapping_path: Path) -> pd.DataFrame:
    """Map BIMP_v4.3 feature names to gene metadata via np31_to_v43_mapping.csv."""
    mapping = pd.read_csv(mapping_path)
    by_v43 = mapping.set_index("v43_id")

    rows = []
    for name in unique_features:
        m = _FEATURE_PATTERN.match(name)
        if not m:
            rows.append(
                {
                    "v43_id": None,
                    "replicate": None,
                    "adduct": None,
                    "np31_id": None,
                    "loc_id": None,
                    "gene_name": None,
                }
            )
            continue
        v43_id, rep, adduct = m.group(1), int(m.group(2)), m.group(3)
        row = {
            "v43_id": v43_id,
            "replicate": rep,
            "adduct": adduct,
            "np31_id": None,
            "loc_id": None,
            "gene_name": None,
        }
        if v43_id in by_v43.index:
            entry = by_v43.loc[v43_id]
            row["np31_id"] = entry["np31_id"]
            row["loc_id"] = entry["loc_id"]
            desc = str(entry["description"])
            gm = re.search(r"\[protein=([^\]]+)\]", desc)
            row["gene_name"] = gm.group(1) if gm else desc
        rows.append(row)

    var = pd.DataFrame(rows, index=pd.Index(unique_features, name="feature"))
    return var


# ---------------------------------------------------------------------------
# from-csv: ingest wide-format CSV (queen/worker MALDI-III export)
# ---------------------------------------------------------------------------
@app.command()
def from_csv(
    csv: Path = typer.Option(
        ..., help="Path to wide-format CSV (one sample, all regions)"
    ),
    sample_id: str = typer.Option(..., help="Sample identifier (e.g., queen_80524)"),
    label: str = typer.Option(..., help="Sample label (queen | worker)"),
    out_dir: Path = typer.Option(
        "data/slices", help="Output directory for per-(sample,region) h5ad"
    ),
    mapping: Path = typer.Option(
        "../../data/panel/np31_to_v43_mapping.csv",
        help="Gene name mapping (np31 ↔ v43 ↔ LOC)",
    ),
    chunksize: int = typer.Option(20000, help="CSV stream chunk size (rows)"),
    aggregation: str = typer.Option(
        "max", help="How to collapse duplicate peak columns: max | sum | mean"
    ),
):
    """Ingest a wide-format MALDI-III CSV into per-region h5ad slices.

    Collapses duplicate feature columns (same base label, different m/z peaks)
    via *aggregation* (default: max, which keeps the strongest matched peak
    per feature per spot — closest to SCiLS/spydata peak-picked intensities;
    sum/mean inflate everything by the noise floor in the m/z window).
    Recenters each region's (x, y) on its own centroid and joins gene
    metadata from the v4.3 panel mapping.
    """
    if aggregation not in ("max", "sum", "mean"):
        raise typer.BadParameter("aggregation must be max | sum | mean")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Pass 1: read header, group columns ---
    header = pd.read_csv(csv, nrows=0).columns.tolist()
    feat_groups: dict[str, list[int]] = {}  # base_name -> column positions
    meta_pos: dict[str, int] = {}  # meta_name -> column position
    for i, col in enumerate(header):
        base = _strip_pandas_dup_suffix(col)
        if base in _META_COLS:
            meta_pos[base] = i
        elif _FEATURE_PATTERN.match(base):
            feat_groups.setdefault(base, []).append(i)
        # else: drop unnamed index column and malformed headers
    missing = _META_COLS - set(meta_pos)
    if missing:
        raise typer.BadParameter(f"Missing required columns: {missing}")

    unique_features = sorted(feat_groups)
    n_feat = len(unique_features)
    n_peaks_per_feat = np.array(
        [len(feat_groups[f]) for f in unique_features], dtype=np.int32
    )
    typer.echo(
        f"  {n_feat} unique features, "
        f"{n_peaks_per_feat.sum()} raw peak columns "
        f"(avg {n_peaks_per_feat.mean():.1f} peaks/feature; "
        f"min={n_peaks_per_feat.min()}, max={n_peaks_per_feat.max()})"
    )

    feat_col_lists = [feat_groups[f] for f in unique_features]

    # --- Pass 2: stream chunks, route per region ---
    per_region_X: dict[str, list[np.ndarray]] = {}
    per_region_meta: dict[str, list[pd.DataFrame]] = {}

    typer.echo(f"Streaming {csv} (chunksize={chunksize}) ...")
    total_rows = 0
    for chunk in pd.read_csv(csv, chunksize=chunksize, low_memory=False):
        n = len(chunk)
        total_rows += n
        arr = chunk.to_numpy()  # mixed dtype; numeric cols are float
        X = np.zeros((n, n_feat), dtype=np.float32)
        for j, cols in enumerate(feat_col_lists):
            sub = arr[:, cols].astype(np.float32, copy=False)
            if aggregation == "max":
                # Mask NaN to -inf so they don't dominate max
                X[:, j] = np.where(np.isnan(sub), -np.inf, sub).max(axis=1)
                X[:, j] = np.where(np.isfinite(X[:, j]), X[:, j], 0.0)
            elif aggregation == "sum":
                X[:, j] = np.nansum(sub, axis=1)
            elif aggregation == "mean":
                X[:, j] = np.nanmean(sub, axis=1)

        meta = pd.DataFrame({name: arr[:, pos] for name, pos in meta_pos.items()})
        meta["region"] = meta["region"].astype(str)
        meta["x"] = meta["x"].astype(np.float64)
        meta["y"] = meta["y"].astype(np.float64)

        for region, sub_meta in meta.groupby("region"):
            idx = sub_meta.index.to_numpy()
            per_region_X.setdefault(region, []).append(X[idx])
            per_region_meta.setdefault(region, []).append(
                sub_meta.reset_index(drop=True)
            )

        typer.echo(f"  ... {total_rows} rows processed")

    # --- Build var table once ---
    var = _build_var_table(unique_features, mapping)

    # --- Write per-(sample, region) h5ad ---
    written = 0
    for region in sorted(per_region_X, key=lambda r: int(r) if r.isdigit() else r):
        X = np.vstack(per_region_X[region])
        meta = pd.concat(per_region_meta[region], ignore_index=True)

        xy_slide = meta[["x", "y"]].to_numpy(dtype=np.float64)
        # Recenter per slice (PASTE3 expects per-slice frames)
        xy_center = xy_slide - xy_slide.mean(axis=0, keepdims=True)

        adata = ad.AnnData(X=X)
        adata.var_names = unique_features
        adata.var = var.copy()
        adata.var["n_peaks_collapsed"] = n_peaks_per_feat

        adata.obsm["spatial"] = xy_center
        adata.obsm["spatial_slide"] = xy_slide
        adata.obs["spot_id"] = meta["spot_id"].astype(str).values
        adata.obs["sample_name"] = meta["sample_name"].astype(str).values
        adata.obs["region_id"] = meta["region_id"].astype(str).values
        adata.obs["region"] = region
        adata.obs["sample_id"] = sample_id
        adata.obs["label"] = label
        adata.obs["slide_id"] = (
            sample_id  # alias for downstream rules expecting slide_id
        )
        adata.obs_names = [f"{sample_id}_r{region}_{i}" for i in range(adata.n_obs)]

        adata.uns["sample_id"] = sample_id
        adata.uns["label"] = label
        adata.uns["region_name"] = str(region)
        adata.uns["slide_id"] = sample_id
        adata.uns["source_csv"] = str(csv)
        adata.uns["aggregation"] = aggregation

        out_path = out_dir / f"slice_{sample_id}_r{str(region).zfill(3)}.h5ad"
        adata.write_h5ad(out_path)
        written += 1
        typer.echo(f"  wrote {out_path.name}: n={adata.n_obs}, features={adata.n_vars}")

    typer.echo(f"Wrote {written} slice h5ad files to {out_dir}/ for sample={sample_id}")


# ---------------------------------------------------------------------------
# from-spydata: load collaborator's pre-extracted Spyder session
# ---------------------------------------------------------------------------
@app.command()
def from_spydata(
    spydata: Path = typer.Option(..., help="Path to .spydata file"),
    out_dir: Path = typer.Option(
        "data/slices", help="Output directory for per-slice h5ad"
    ),
    slide_id: str = typer.Option("78942", help="Slide identifier for file naming"),
):
    """Load pre-extracted MALDI data from a Spyder .spydata archive."""
    out_dir.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Loading {spydata} ...")
    with tarfile.open(spydata, "r") as tar:
        data = pickle.loads(tar.extractfile(tar.getnames()[0]).read())  # noqa: S301

    region_ids: dict = data["region_ids"]  # {uuid: "60", ...}
    spot_coords_dict: dict = data["spot_coords_dict"]  # {uuid: DataFrame(x,y)}
    intensity_dfs: dict = data["intensity_dfs"]  # {uuid: DataFrame}
    all_features: pd.DataFrame = data["all_features"]

    feature_names = list(all_features["name"])
    mz_centers = all_features[["mz_low", "mz_high"]].mean(axis=1).values

    typer.echo(f"  {len(region_ids)} regions, {len(feature_names)} m/z features")
    typer.echo(f"  Features: {feature_names}")

    written = 0
    for region_uuid, region_name in sorted(region_ids.items(), key=lambda x: int(x[1])):
        idf = intensity_dfs[region_uuid]
        coords = spot_coords_dict[region_uuid]

        # Intensity columns are the last len(feature_names) columns
        X = idf.iloc[:, 2:].values.astype(np.float32)
        assert X.shape[1] == len(feature_names), (
            f"Expected {len(feature_names)} features, got {X.shape[1]}"
        )

        adata = ad.AnnData(X=X)
        adata.var_names = feature_names
        adata.var["mz_center"] = mz_centers
        adata.obsm["spatial"] = coords.values.astype(np.float64)
        adata.obs["spot_id"] = idf.iloc[:, 0].values
        adata.obs["region"] = region_name
        adata.obs["slide_id"] = slide_id
        adata.obs_names = [f"{slide_id}_r{region_name}_{i}" for i in range(adata.n_obs)]
        adata.uns["slide_id"] = slide_id
        adata.uns["region_name"] = region_name
        adata.uns["region_uuid"] = region_uuid

        out_path = out_dir / f"slice_{slide_id}_r{region_name.zfill(3)}.h5ad"
        adata.write_h5ad(out_path)
        written += 1

    typer.echo(f"Wrote {written} slice h5ad files to {out_dir}/")


# ---------------------------------------------------------------------------
# from-scils: load directly from SCiLS .slx (requires scilslab SDK)
# ---------------------------------------------------------------------------
@app.command()
def from_scils(
    slx: Path = typer.Option(..., help="Path to .slx file"),
    feature_list: str = typer.Option(..., help="Feature list UUID from SCiLS"),
    out_dir: Path = typer.Option("data/slices", help="Output directory"),
    slide_id: str = typer.Option(..., help="Slide identifier"),
    port: int = typer.Option(8088, help="SCiLS local server port"),
):
    """Load MALDI data directly from a SCiLS .slx project file.

    Requires the scilslab Python SDK (proprietary, not on PyPI).
    """
    try:
        import scilslab
    except ImportError as err:
        typer.echo(
            "ERROR: scilslab SDK not installed. "
            "Get the wheel from Bruker and install with: "
            "uv pip install /path/to/scilslab-*.whl",
            err=True,
        )
        raise typer.Exit(1) from err

    from tqdm import tqdm

    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: enumerate regions
    with scilslab.LocalSession(str(slx), port=port) as session:
        dataset = session.dataset_proxy
        region_tree = dataset.get_region_tree()
        region_tree.print()

        region_ids = {}
        for region in region_tree.get_all_regions():
            actual_name = region.name.split("/")[-1]
            region_ids[region.id] = actual_name

    # Keep only numeric region names (tissue sections)
    region_ids = {k: v for k, v in region_ids.items() if v.isdigit()}
    region_ids = dict(sorted(region_ids.items(), key=lambda item: int(item[1])))
    typer.echo(f"Found {len(region_ids)} tissue sections")

    # Step 2: fetch features and coordinates
    with scilslab.LocalSession(str(slx)) as session:
        dataset = session.dataset_proxy
        all_features = dataset.feature_table.get_features(feature_list)
        all_features = all_features[all_features["has_data"]]
        feature_names = scilslab.make_feature_names(all_features)

        spot_coords_dict = {}
        for region_id in tqdm(region_ids, desc="Fetching coordinates"):
            spots = pd.DataFrame(dataset.get_region_spots(region_id))
            spot_coords_dict[region_id] = spots[["x", "y"]]

    # Step 3: extract intensities and write per-slice h5ad
    written = 0
    for region_id, region_name in tqdm(
        region_ids.items(), desc="Extracting intensities"
    ):
        with scilslab.LocalSession(str(slx)) as session:
            dataset = session.dataset_proxy
            peak_intensities = np.array(
                [
                    dataset.feature_table.get_feature_intensities(
                        f_id, region_id=region_id, mode="area"
                    ).values
                    for f_id in all_features["id"]
                ]
            ).T

        coords = spot_coords_dict[region_id]
        X = peak_intensities.astype(np.float32)

        adata = ad.AnnData(X=X)
        adata.var_names = feature_names
        adata.obsm["spatial"] = coords.values.astype(np.float64)
        adata.obs["region"] = region_name
        adata.obs["slide_id"] = slide_id
        adata.obs_names = [f"{slide_id}_r{region_name}_{i}" for i in range(adata.n_obs)]
        adata.uns["slide_id"] = slide_id
        adata.uns["region_name"] = region_name

        out_path = out_dir / f"slice_{slide_id}_r{region_name.zfill(3)}.h5ad"
        adata.write_h5ad(out_path)
        written += 1

    typer.echo(f"Wrote {written} slice h5ad files to {out_dir}/")


# ---------------------------------------------------------------------------
# inventory: summarize ingested slices
# ---------------------------------------------------------------------------
@app.command()
def inventory(
    slice_dir: Path = typer.Option(
        "data/slices", help="Directory with per-slice h5ad files"
    ),
):
    """Print a summary table of ingested MALDI slices."""
    h5ads = sorted(slice_dir.glob("slice_*.h5ad"))
    if not h5ads:
        typer.echo(f"No slice files found in {slice_dir}")
        raise typer.Exit(1)

    rows = []
    for p in h5ads:
        adata = ad.read_h5ad(p, backed="r")
        rows.append(
            {
                "file": p.name,
                "slide": adata.uns.get("slide_id", "?"),
                "region": adata.uns.get("region_name", "?"),
                "n_spots": adata.n_obs,
                "n_features": adata.n_vars,
                "x_range": f"{adata.obsm['spatial'][:, 0].min():.0f}-"
                f"{adata.obsm['spatial'][:, 0].max():.0f}",
                "y_range": f"{adata.obsm['spatial'][:, 1].min():.0f}-"
                f"{adata.obsm['spatial'][:, 1].max():.0f}",
            }
        )
        adata.file.close()

    df = pd.DataFrame(rows)
    typer.echo(df.to_string(index=False))
    typer.echo(f"\nTotal: {len(rows)} slices, {df['n_spots'].sum()} spots")


if __name__ == "__main__":
    app()
