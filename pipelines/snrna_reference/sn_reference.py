"""snRNA-seq reference processing pipeline for Bombus impatiens.

Commands:
    preprocess — Load CellBender h5 files, QC, combine into raw h5ad
    process    — Normalize, HVG, PCA, Harmony, neighbors, UMAP, Leiden
    annotate   — Apply cluster→cell_type mapping from YAML
    markers    — Export top N marker genes per cluster with descriptions
    inspect    — Print summary of any h5ad file

Adapted from I. Traniello's 2025 notebooks (sNuc_Worker_GEX.ipynb, 00_QC_AAUR).
Raw data flow: CellBender h5 → preprocess (QC/filter/combine) → process → annotate.
"""

from pathlib import Path

import typer

app = typer.Typer(help="snRNA-seq reference processing for Bombus impatiens")

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
FIGURES_DIR = OUTPUT_DIR / "figures"
ANNOTATIONS_DIR = OUTPUT_DIR  # cell_type_map.yaml sits beside this script

BIMP_LOOKUP = OUTPUT_DIR.parent / "combined_atlas/data/BIMP_v4.3_lookup.tsv"


# =============================================================================
# PREPROCESS — Load CellBender outputs, QC, combine
# =============================================================================


def _is_outlier(obs_series, nmads: int):
    """MAD-based outlier detection (the integrated notebook's protocol)."""
    from scipy.stats import median_abs_deviation

    med = obs_series.median()
    mad = median_abs_deviation(obs_series)
    return (obs_series < med - nmads * mad) | (obs_series > med + nmads * mad)


@app.command()
def preprocess(
    config_file: str = typer.Option(
        "", help="Path to config.yaml with sample definitions"
    ),
    min_genes: int = typer.Option(200, help="Min genes per cell"),
    min_cells: int = typer.Option(3, help="Min cells per gene"),
    nmads: int = typer.Option(5, help="MADs for outlier detection"),
    max_counts: int = typer.Option(15000, help="Max UMI counts per cell (0=disabled)"),
    mt_pct: float = typer.Option(5.0, help="Max mitochondrial % per cell"),
    doublet_score_max: float = typer.Option(0.2, help="Max doublet score"),
    n_jobs: int = typer.Option(4, help="Parallel jobs"),
):
    """Load CellBender-filtered h5 files, QC, filter, and combine.

    Matches the integrated notebook exactly:
      1. Load each CellBender h5 (10x format), prefix barcodes
      2. Concatenate all samples
      3. Filter cells (min_genes, max_counts) and genes (min_cells)
      4. Compute QC metrics
      5. Per-sample MAD-based outlier detection (total counts, n_genes) + mt% cutoff
      6. Batch-wide Scrublet (batch_key="sample") + doublet_score filter
      7. Save as data/raw.h5ad
    """
    import os

    os.environ.setdefault("NUMBA_NUM_THREADS", str(n_jobs))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_jobs))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_jobs))

    import matplotlib.pyplot as plt
    import scanpy as sc
    import yaml

    sc.settings.n_jobs = n_jobs
    sc.settings.verbosity = 2

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load config
    cfg_path = Path(config_file) if config_file else OUTPUT_DIR / "config.yaml"
    if not cfg_path.exists():
        typer.echo(f"ERROR: Config not found: {cfg_path}")
        raise typer.Exit(1)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    samples = cfg.get("samples", {})
    if not samples:
        typer.echo("ERROR: No samples defined in config.yaml")
        raise typer.Exit(1)

    # Override QC params from config if present
    min_genes = cfg.get("qc_min_genes", min_genes)
    min_cells = cfg.get("qc_min_cells", min_cells)
    nmads = cfg.get("qc_nmads", nmads)
    max_counts = cfg.get("qc_max_counts", max_counts)
    mt_pct = cfg.get("qc_mt_pct", mt_pct)
    doublet_score_max = cfg.get("qc_doublet_score_max", doublet_score_max)

    typer.echo(f"Loading {len(samples)} samples from CellBender h5 files...")
    typer.echo(
        f"QC params: min_genes={min_genes}, min_cells={min_cells}, nmads={nmads}, "
        f"max_counts={max_counts or 'none'}, mt_pct={mt_pct}%, "
        f"doublet_score_max={doublet_score_max}"
    )

    # ── Step 1: Load all samples ──────────────────────────────
    adatas_list = []
    sample_names = []
    for sample_name, sample_info in samples.items():
        h5_path = sample_info["path"]
        fmt = sample_info.get("format", "h5")
        typer.echo(f"\n{'=' * 60}")
        typer.echo(f"Sample: {sample_name}")
        typer.echo(f"  File: {h5_path}")

        if not Path(h5_path).exists():
            typer.echo("  ERROR: File not found, skipping")
            continue

        if fmt == "h5ad":
            adata = sc.read_h5ad(h5_path)
            subset_key = sample_info.get("subset_key")
            subset_value = sample_info.get("subset_value")
            if subset_key and subset_value:
                n_before = adata.n_obs
                adata = adata[adata.obs[subset_key] == subset_value].copy()
                typer.echo(
                    f"  Subset {subset_key}=={subset_value}: {n_before:,} "
                    f"→ {adata.n_obs:,}"
                )
            if "counts" in adata.layers:
                typer.echo("  Restoring raw counts from .layers['counts']")
                adata.X = adata.layers["counts"].copy()
            # Strip stale embeddings/clustering
            for key in list(adata.obsm.keys()):
                del adata.obsm[key]
            for key in list(adata.obsp.keys()):
                del adata.obsp[key]
            for col in [c for c in adata.obs.columns if c.startswith("leiden")]:
                del adata.obs[col]
            for key in [
                k
                for k in adata.uns
                if k.endswith("_colors")
                or k.startswith("leiden")
                or k in ("log1p", "neighbors", "pca", "umap", "hvg")
            ]:
                del adata.uns[key]
        else:
            adata = sc.read_10x_h5(h5_path)

        adata.var_names_make_unique()

        # Add metadata
        adata.obs["sample"] = sample_name
        for key in ["caste", "experiment", "condition"]:
            if key in sample_info:
                adata.obs[key] = sample_info[key]

        # Prefix barcodes with sample name (matches notebook)
        adata.obs_names = [f"{sample_name}_{bc}" for bc in adata.obs_names]

        typer.echo(f"  Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        adatas_list.append(adata)
        sample_names.append(sample_name)

    if not adatas_list:
        typer.echo("ERROR: No samples loaded successfully")
        raise typer.Exit(1)

    # ── Step 2: Concatenate (matches notebook .concatenate()) ─
    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"Concatenating {len(adatas_list)} samples...")
    combined = adatas_list[0].concatenate(
        adatas_list[1:],
        batch_key="sample",
        batch_categories=sample_names,
    )
    combined.var_names_make_unique()
    typer.echo(f"  Combined: {combined.n_obs:,} cells x {combined.n_vars:,} genes")

    # ── Step 3: Cell/gene filters BEFORE QC metrics ───────────
    n_before = combined.n_obs
    sc.pp.filter_cells(combined, min_genes=min_genes)
    typer.echo(f"  min_genes={min_genes}: {n_before:,} → {combined.n_obs:,} cells")

    if max_counts > 0:
        n_before = combined.n_obs
        sc.pp.filter_cells(combined, max_counts=max_counts)
        typer.echo(
            f"  max_counts={max_counts:,}: {n_before:,} → {combined.n_obs:,} "
            f"cells ({n_before - combined.n_obs:,} removed)"
        )

    n_genes_before = combined.n_vars
    sc.pp.filter_genes(combined, min_cells=min_cells)
    typer.echo(
        f"  min_cells={min_cells}: {n_genes_before:,} → {combined.n_vars:,} genes"
    )

    combined.var_names_make_unique()

    # ── Step 4: QC metrics (after filtering, matches notebook) ─
    combined.var["mt"] = combined.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(combined, qc_vars=["mt"], inplace=True, log1p=True)

    # ── Step 5: Per-sample MAD outlier detection ──────────────
    # Matches notebook: per-group MAD on log1p_total_counts and
    # log1p_n_genes_by_counts only (NOT pct_counts_in_top_20_genes),
    # plus hard mt% cutoff combined into the outlier mask.
    def _is_outlier_per_group(obs_df, metric, nmads_val, group_col="sample"):
        from scipy.stats import median_abs_deviation

        grp = obs_df.groupby(group_col)[metric]
        med = grp.transform("median")
        mad = grp.transform(lambda x: median_abs_deviation(x, nan_policy="omit"))
        return (obs_df[metric] < med - nmads_val * mad) | (
            obs_df[metric] > med + nmads_val * mad
        )

    combined.obs["outlier"] = (
        _is_outlier_per_group(combined.obs, "log1p_total_counts", nmads)
        | _is_outlier_per_group(combined.obs, "log1p_n_genes_by_counts", nmads)
        | (combined.obs["pct_counts_mt"] > mt_pct)
    )
    n_outliers = combined.obs["outlier"].sum()
    typer.echo(
        f"  Per-sample MAD outliers ({nmads} MADs) + "
        f"mt>{mt_pct}%: {n_outliers:,} / {combined.n_obs:,}"
    )

    n_before = combined.n_obs
    combined = combined[~combined.obs["outlier"].astype(bool)].copy()
    typer.echo(
        f"  After outlier removal: {combined.n_obs:,} cells "
        f"({n_before - combined.n_obs:,} removed)"
    )

    # ── Step 6: Batch-wide Scrublet + doublet_score filter ────
    typer.echo("  Running Scrublet (batch_key='sample')...")
    try:
        sc.pp.scrublet(combined, batch_key="sample")
        n_doublets = combined.obs["predicted_doublet"].sum()
        typer.echo(f"  Predicted doublets: {n_doublets:,} / {combined.n_obs:,}")

        n_before = combined.n_obs
        combined = combined[combined.obs["predicted_doublet"].eq(False)].copy()
        typer.echo(f"  After predicted_doublet removal: {combined.n_obs:,} cells")

        n_before = combined.n_obs
        combined = combined[combined.obs["doublet_score"] <= doublet_score_max].copy()
        typer.echo(
            f"  After doublet_score<={doublet_score_max}: {combined.n_obs:,} "
            f"cells ({n_before - combined.n_obs:,} removed)"
        )
    except Exception as e:
        typer.echo(f"  WARNING: Scrublet failed: {e}")

    # ── Summary ───────────────────────────────────────────────
    typer.echo(
        f"\nFinal combined dataset: {combined.n_obs:,} cells "
        f"x {combined.n_vars:,} genes"
    )
    typer.echo("Samples:")
    for s, n in combined.obs["sample"].value_counts().items():
        typer.echo(f"  {s}: {n:,}")

    # QC figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric, label in zip(
        axes,
        ["total_counts", "n_genes_by_counts", "pct_counts_mt"],
        ["Total counts", "Genes detected", "% mito"],
        strict=True,
    ):
        sc.pl.violin(combined, metric, groupby="sample", ax=ax, show=False)
        ax.set_title(label)
        ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "preprocess_qc.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo("Saved: figures/preprocess_qc.png")

    # Save
    output_path = DATA_DIR / "raw.h5ad"
    combined.write_h5ad(output_path)
    typer.echo(f"Saved: {output_path}")


# =============================================================================
# PROCESS
# =============================================================================


@app.command()
def process(
    input_file: str = typer.Option("", help="Input h5ad path (default: data/raw.h5ad)"),
    batch_key: str = typer.Option("sample", help="Batch key for Harmony"),
    pca_n_comps: int = typer.Option(50, help="Number of PCA components to compute"),
    n_pcs: int = typer.Option(20, help="Number of PCs for neighbors/UMAP"),
    harmony_max_iter: int = typer.Option(15, help="Harmony max iterations"),
    n_neighbors: int = typer.Option(15, help="Number of neighbors for UMAP/Leiden"),
    resolution: float = typer.Option(1.0, help="Leiden clustering resolution"),
    n_jobs: int = typer.Option(4, help="Parallel jobs for scanpy"),
    min_dist: float = typer.Option(0.5, help="UMAP min_dist"),
):
    """Normalize, HVG, PCA, Harmony, neighbors, UMAP, Leiden, rank_genes_groups.

    Starts from the QC'd raw h5ad (output of preprocess). Stores raw counts in
    .layers["counts"], then normalizes, selects HVGs, runs PCA, Harmony batch
    integration, and Leiden clustering.
    """
    import os

    # Set thread limits before any numba/BLAS-dependent import (scanpy → numba)
    os.environ.setdefault("NUMBA_NUM_THREADS", str(n_jobs))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_jobs))
    os.environ.setdefault("MKL_NUM_THREADS", str(n_jobs))

    import matplotlib.pyplot as plt
    import numpy as np
    import scanpy as sc
    from scipy.sparse import issparse

    sc.settings.n_jobs = n_jobs
    sc.settings.verbosity = 2

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Load — default to data/raw.h5ad (output of preprocess)
    if input_file:
        input_path = Path(input_file)
    else:
        input_path = DATA_DIR / "raw.h5ad"
    if not input_path.exists():
        typer.echo(f"ERROR: Input file not found: {input_path}")
        typer.echo("Run `preprocess` first to generate data/raw.h5ad")
        raise typer.Exit(1)
    typer.echo(f"Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)
    typer.echo(f"  Cells: {adata.n_obs:,}, Genes: {adata.n_vars}")
    typer.echo(f"  obs columns: {list(adata.obs.columns)}")

    # Handle raw counts: if data is already log-normalized, restart from counts layer
    already_normalized = "log1p" in adata.uns
    if "counts" in adata.layers:
        if already_normalized:
            typer.echo(
                "Data is already log-normalized. Restoring raw counts from "
                ".layers['counts']..."
            )
            adata.X = adata.layers["counts"].copy()
            # Clear embeddings/graphs/colors carried in from upstream processing
            for key in ["X_pca", "X_pca_harmony", "X_umap"]:
                adata.obsm.pop(key, None)
            adata.obsp.pop("connectivities", None)
            adata.obsp.pop("distances", None)
            # Remove stale uns keys (colors, clustering params, etc.)
            stale_uns = [
                k for k in adata.uns if k.endswith("_colors") or k.startswith("leiden")
            ]
            stale_uns += ["log1p", "neighbors", "pca", "umap", "hvg"]
            for key in stale_uns:
                adata.uns.pop(key, None)
            # Remove old leiden obs columns
            for col in [c for c in adata.obs.columns if c.startswith("leiden")]:
                del adata.obs[col]
        else:
            typer.echo("Storing raw counts in .layers['counts'] (already present).")
    else:
        typer.echo("Storing raw counts in .layers['counts']...")
        if issparse(adata.X):
            adata.layers["counts"] = adata.X.copy()
        else:
            adata.layers["counts"] = np.array(adata.X)

    # Normalize
    typer.echo("Normalizing (target_sum=1e4, log1p)...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Fix random seed for reproducibility
    np.random.seed(0)

    # HVG selection
    typer.echo("Selecting highly variable genes...")
    sc.pp.highly_variable_genes(adata, batch_key=batch_key)
    n_hvg = adata.var["highly_variable"].sum()
    typer.echo(f"  HVGs: {n_hvg}")

    # PCA
    typer.echo(f"Running PCA (n_comps={pca_n_comps}, neighbor n_pcs={n_pcs})...")
    sc.tl.pca(adata, n_comps=pca_n_comps, use_highly_variable=True, svd_solver="arpack")

    # Pre-Harmony UMAP (for comparison)
    typer.echo("Computing pre-Harmony UMAP...")
    sc.pp.neighbors(adata, n_pcs=n_pcs, n_neighbors=n_neighbors, random_state=0)
    sc.tl.umap(adata, min_dist=min_dist, random_state=0)
    adata.obsm["X_umap_pre_harmony"] = adata.obsm["X_umap"].copy()

    # QC figure: pre-Harmony
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sc.pl.umap(
        adata, color=batch_key, ax=axes[0], show=False, title="Pre-Harmony (batch)"
    )
    if "caste" in adata.obs.columns:
        sc.pl.umap(
            adata, color="caste", ax=axes[1], show=False, title="Pre-Harmony (caste)"
        )
    else:
        axes[1].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pre_harmony_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo("  Saved: figures/pre_harmony_umap.png")

    # Harmony (harmonypy directly — sce.pp.harmony_integrate has an obsm shape bug)
    typer.echo(
        f"Running Harmony integration (key={batch_key}, max_iter={harmony_max_iter})..."
    )
    import harmonypy

    ho = harmonypy.run_harmony(
        adata.obsm["X_pca"],
        adata.obs,
        batch_key,
        max_iter_harmony=harmony_max_iter,
        random_state=0,
    )
    Z = np.asarray(ho.Z_corr)
    assert Z.shape == (adata.n_obs, pca_n_comps), f"Unexpected Harmony shape: {Z.shape}"
    adata.obsm["X_pca_harmony"] = Z
    typer.echo(f"  Harmony output shape: {Z.shape}")

    # Post-Harmony neighbors + UMAP
    typer.echo("Computing post-Harmony neighbors + UMAP...")
    sc.pp.neighbors(
        adata,
        use_rep="X_pca_harmony",
        n_pcs=n_pcs,
        n_neighbors=n_neighbors,
        random_state=0,
    )
    sc.tl.umap(adata, min_dist=min_dist, random_state=0)

    # Leiden clustering
    typer.echo(f"Leiden clustering (resolution={resolution})...")
    sc.tl.leiden(
        adata, resolution=resolution, flavor="igraph", n_iterations=2, random_state=0
    )
    n_clusters = adata.obs["leiden"].nunique()
    typer.echo(f"  Clusters: {n_clusters}")

    # Marker genes
    typer.echo("Computing marker genes (Wilcoxon rank-sum)...")
    sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")

    # Post-Harmony QC figures
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    sc.pl.umap(
        adata, color=batch_key, ax=axes[0], show=False, title="Post-Harmony (batch)"
    )
    sc.pl.umap(
        adata,
        color="leiden",
        ax=axes[1],
        show=False,
        legend_loc="on data",
        legend_fontsize=6,
        title="Leiden clusters",
    )
    if "caste" in adata.obs.columns:
        sc.pl.umap(
            adata, color="caste", ax=axes[2], show=False, title="Post-Harmony (caste)"
        )
    else:
        axes[2].set_visible(False)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "post_harmony_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo("  Saved: figures/post_harmony_umap.png")

    # Cluster composition
    _plot_cluster_composition(adata)

    # Dendrogram for downstream dotplots
    sc.tl.dendrogram(adata, groupby="leiden")

    # Save
    output_path = DATA_DIR / "processed.h5ad"
    adata.write_h5ad(output_path)
    typer.echo(f"Saved: {output_path}")
    typer.echo(f"  Final shape: {adata.n_obs:,} cells x {adata.n_vars} genes")


def _plot_cluster_composition(adata):
    """Plot cluster composition by sample/caste."""
    import matplotlib.pyplot as plt
    import pandas as pd

    cols = [c for c in ["sample", "caste"] if c in adata.obs.columns]
    if not cols:
        return

    fig, axes = plt.subplots(1, len(cols), figsize=(7 * len(cols), 5))
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols, strict=True):
        ct = pd.crosstab(adata.obs["leiden"], adata.obs[col], normalize="index")
        ct.plot.bar(stacked=True, ax=ax, legend=True)
        ax.set_title(f"Cluster composition by {col}")
        ax.set_xlabel("Leiden cluster")
        ax.set_ylabel("Proportion")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo("  Saved: figures/cluster_composition.png")


# =============================================================================
# ANNOTATE
# =============================================================================


@app.command()
def annotate(
    input_file: str = typer.Option("processed", help="Input h5ad stem in data/"),
    mapping_file: str = typer.Option(
        "", help="Path to cell_type_map.yaml (default: ./cell_type_map.yaml)"
    ),
):
    """Apply cluster→cell_type mapping from YAML file.

    Reads the mapping from cell_type_map.yaml and adds 'cell_type'
    and optionally 'major_cell_type' columns to the AnnData obs.
    """
    import scanpy as sc
    import yaml

    input_path = DATA_DIR / f"{input_file}.h5ad"
    if not input_path.exists():
        typer.echo(f"ERROR: File not found: {input_path}")
        raise typer.Exit(1)

    mapping_path = (
        Path(mapping_file) if mapping_file else ANNOTATIONS_DIR / "cell_type_map.yaml"
    )
    if not mapping_path.exists():
        typer.echo(f"ERROR: Mapping file not found: {mapping_path}")
        typer.echo("Create it by editing cell_type_map.yaml")
        raise typer.Exit(1)

    with open(mapping_path) as f:
        mapping = yaml.safe_load(f)

    cluster_map = mapping.get("cluster_to_cell_type", {})
    if not cluster_map:
        typer.echo("ERROR: cluster_to_cell_type is empty in mapping file")
        typer.echo("Run `python sn_reference.py markers` first, then edit the YAML")
        raise typer.Exit(1)

    typer.echo(f"Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)
    typer.echo(f"  Cells: {adata.n_obs:,}")

    # Map cluster IDs to cell types
    cluster_map_str = {str(k): v for k, v in cluster_map.items()}
    adata.obs["cell_type"] = adata.obs["leiden"].map(cluster_map_str)

    unmapped = adata.obs["cell_type"].isna().sum()
    if unmapped > 0:
        missing = set(adata.obs.loc[adata.obs["cell_type"].isna(), "leiden"].unique())
        typer.echo(f"WARNING: {unmapped:,} cells have unmapped clusters: {missing}")
        adata.obs["cell_type"] = adata.obs["cell_type"].fillna("Unknown")

    typer.echo("Cell type counts:")
    for ct, n in adata.obs["cell_type"].value_counts().items():
        typer.echo(f"  {ct}: {n:,}")

    # Major cell types (optional)
    major_types = mapping.get("major_types", {})
    if major_types:
        reverse_map = {}
        for major, subtypes in major_types.items():
            for st in subtypes:
                reverse_map[st] = major
        adata.obs["major_cell_type"] = adata.obs["cell_type"].map(reverse_map)
        unmapped_major = adata.obs["major_cell_type"].isna().sum()
        if unmapped_major > 0:
            adata.obs["major_cell_type"] = adata.obs["major_cell_type"].fillna("Other")
        typer.echo("\nMajor cell type counts:")
        for ct, n in adata.obs["major_cell_type"].value_counts().items():
            typer.echo(f"  {ct}: {n:,}")

    # Recompute markers by cell_type
    typer.echo("\nRecomputing marker genes by cell_type...")
    sc.tl.rank_genes_groups(adata, groupby="cell_type", method="wilcoxon")

    adata.write_h5ad(input_path)
    typer.echo(f"\nSaved: {input_path}")


# =============================================================================
# MARKERS
# =============================================================================


@app.command()
def markers(
    input_file: str = typer.Option("processed", help="Input h5ad stem in data/"),
    n_genes: int = typer.Option(20, help="Number of top marker genes per cluster"),
    groupby: str = typer.Option("leiden", help="Grouping key (leiden or cell_type)"),
    lookup_file: str = typer.Option("", help="Path to BIMP lookup TSV"),
):
    """Export top N marker genes per cluster with descriptions.

    Outputs a CSV and markdown summary to help with manual cell type annotation.
    """
    import pandas as pd
    import scanpy as sc

    input_path = DATA_DIR / f"{input_file}.h5ad"
    if not input_path.exists():
        typer.echo(f"ERROR: File not found: {input_path}")
        raise typer.Exit(1)

    typer.echo(f"Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)

    if adata.uns.get("rank_genes_groups") is None:
        typer.echo("Computing marker genes...")
        sc.tl.rank_genes_groups(adata, groupby=groupby, method="wilcoxon")

    # Load lookup table for gene descriptions
    lookup_path = Path(lookup_file) if lookup_file else BIMP_LOOKUP
    lookup = None
    if lookup_path.exists():
        typer.echo(f"Loading gene lookup: {lookup_path.name}")
        lookup = pd.read_csv(lookup_path, sep="\t", index_col=0)
    else:
        typer.echo(f"WARNING: Lookup file not found: {lookup_path}")
        typer.echo("  Markers will be exported without descriptions")

    # Extract markers
    groups = (
        adata.obs[groupby].cat.categories
        if hasattr(adata.obs[groupby], "cat")
        else adata.obs[groupby].unique()
    )
    rows = []
    for group in groups:
        result = sc.get.rank_genes_groups_df(adata, group=str(group))
        top = result.head(n_genes)
        for _, row in top.iterrows():
            entry = {
                "cluster": group,
                "gene": row["names"],
                "score": round(row["scores"], 2),
                "logfoldchange": round(row["logfoldchanges"], 3),
                "pval_adj": f"{row['pvals_adj']:.2e}",
            }
            if lookup is not None and row["names"] in lookup.index:
                gene_info = lookup.loc[row["names"]]
                if "description" in gene_info.index:
                    entry["description"] = gene_info["description"]
                if "gene" in gene_info.index:
                    entry["gene_name"] = gene_info["gene"]
            rows.append(entry)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = DATA_DIR / f"markers_{groupby}_top{n_genes}.csv"
    df.to_csv(csv_path, index=False)
    typer.echo(f"Saved: {csv_path}")

    # Save markdown summary
    md_path = DATA_DIR / f"markers_{groupby}_top{n_genes}.md"
    with open(md_path, "w") as f:
        f.write(f"# Top {n_genes} markers per {groupby}\n\n")
        for group in groups:
            f.write(f"## Cluster {group}\n\n")
            group_df = df[df["cluster"] == group]
            f.write("| Gene | Score | logFC | p_adj |")
            if "description" in df.columns:
                f.write(" Description |")
            f.write("\n")
            f.write("|------|-------|-------|-------|")
            if "description" in df.columns:
                f.write("-------------|")
            f.write("\n")
            for _, row in group_df.iterrows():
                f.write(
                    f"| {row['gene']} | {row['score']} "
                    f"| {row['logfoldchange']} | {row['pval_adj']} |"
                )
                if "description" in df.columns:
                    desc = row.get("description", "")
                    if pd.notna(desc):
                        f.write(f" {desc} |")
                    else:
                        f.write(" |")
                f.write("\n")
            f.write("\n")
    typer.echo(f"Saved: {md_path}")
    typer.echo(
        "\nNext: edit cell_type_map.yaml, then run `uv run "
        "python sn_reference.py annotate`"
    )


# =============================================================================
# INSPECT
# =============================================================================


@app.command()
def inspect(
    input_file: str = typer.Option(
        "processed", help="Input h5ad stem in data/ (or full path)"
    ),
):
    """Print summary of an h5ad file (shape, obs columns, cluster counts, caste
    distribution)."""
    import scanpy as sc

    # Allow both stem and full path
    input_path = Path(input_file)
    if not input_path.exists():
        input_path = DATA_DIR / f"{input_file}.h5ad"
    if not input_path.exists():
        typer.echo(f"ERROR: File not found: {input_file}")
        raise typer.Exit(1)

    adata = sc.read_h5ad(input_path)

    typer.echo(f"File: {input_path}")
    typer.echo(f"Shape: {adata.n_obs:,} cells x {adata.n_vars} genes")
    typer.echo(f"obs columns: {list(adata.obs.columns)}")
    typer.echo(f"var columns: {list(adata.var.columns)}")
    typer.echo(f"Layers: {list(adata.layers.keys())}")
    typer.echo(f"obsm: {list(adata.obsm.keys())}")

    if "leiden" in adata.obs.columns:
        typer.echo(f"\nLeiden clusters ({adata.obs['leiden'].nunique()}):")
        for cluster, n in adata.obs["leiden"].value_counts().sort_index().items():
            typer.echo(f"  {cluster}: {n:,}")

    if "cell_type" in adata.obs.columns:
        typer.echo(f"\nCell types ({adata.obs['cell_type'].nunique()}):")
        for ct, n in adata.obs["cell_type"].value_counts().items():
            typer.echo(f"  {ct}: {n:,}")

    if "caste" in adata.obs.columns:
        typer.echo("\nCaste distribution:")
        for caste, n in adata.obs["caste"].value_counts().items():
            typer.echo(f"  {caste}: {n:,}")

    if "sample" in adata.obs.columns:
        typer.echo("\nSample distribution:")
        for sample, n in adata.obs["sample"].value_counts().items():
            typer.echo(f"  {sample}: {n:,}")


if __name__ == "__main__":
    app()
