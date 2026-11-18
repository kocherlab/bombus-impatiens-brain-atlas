"""MALDI-MSI QC and preprocessing.

Usage:
    python maldi_qc.py preprocess --slice-dir data/slices --out-dir \
        data/processed
    python maldi_qc.py plot-raw    --slice-dir data/slices --fig-dir figures/raw
    python maldi_qc.py plot-qc     --slice-dir data/processed --fig-dir \
        figures/qc
"""

from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(pretty_exceptions_enable=False)


def _load_slices(slice_dir: Path) -> list[ad.AnnData]:
    """Load all slice h5ad files from a directory, sorted by region number."""
    paths = sorted(slice_dir.glob("slice_*.h5ad"))
    if not paths:
        typer.echo(f"No slice files in {slice_dir}", err=True)
        raise typer.Exit(1)
    slices = [ad.read_h5ad(p) for p in paths]
    typer.echo(f"Loaded {len(slices)} slices from {slice_dir}")
    return slices


@app.command()
def preprocess(
    slice_dir: Path = typer.Option("data/slices", help="Input slice directory"),
    out_dir: Path = typer.Option("data/processed", help="Output directory"),
    normalization: str = typer.Option(
        "tic", help="Normalization: tic | median | zscore"
    ),
    nan_strategy: str = typer.Option("zero", help="NaN handling: zero | drop"),
    tic_min_quantile: float = typer.Option(0.01, help="Min TIC quantile for spot QC"),
):
    """Clean NaNs, filter low-quality spots, and normalize intensities."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slices = _load_slices(slice_dir)

    for i, adata in enumerate(slices):
        region = adata.uns.get("region_name", str(i))
        X = np.asarray(adata.X, dtype=np.float32)

        # --- NaN handling ---
        n_nan = np.isnan(X).sum()
        if n_nan > 0:
            typer.echo(f"  Slice {region}: {n_nan} NaNs → {nan_strategy}")
            if nan_strategy == "zero":
                X = np.nan_to_num(X, nan=0.0)
            elif nan_strategy == "drop":
                mask = ~np.isnan(X).any(axis=1)
                X = X[mask]
                adata = adata[mask].copy()

        # --- Spot QC: remove low TIC spots ---
        tic = X.sum(axis=1)
        tic_thresh = np.quantile(tic, tic_min_quantile)
        keep = tic > tic_thresh
        n_removed = (~keep).sum()
        if n_removed > 0:
            typer.echo(
                f"  Slice {region}: removed {n_removed} low-TIC spots "
                f"(threshold={tic_thresh:.2f})"
            )
            X = X[keep]
            adata = adata[keep].copy()

        # --- Normalization ---
        if normalization == "tic":
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            X = X / row_sums * np.median(row_sums)
        elif normalization == "median":
            col_medians = np.median(X, axis=0, keepdims=True)
            col_medians[col_medians == 0] = 1.0
            X = X / col_medians
        elif normalization == "zscore":
            col_means = X.mean(axis=0, keepdims=True)
            col_stds = X.std(axis=0, keepdims=True)
            col_stds[col_stds == 0] = 1.0
            X = (X - col_means) / col_stds

        adata.X = X
        adata.uns["normalization"] = normalization
        adata.uns["nan_strategy"] = nan_strategy
        adata.uns["tic_min_quantile"] = tic_min_quantile

        out_path = out_dir / f"slice_{adata.uns['slide_id']}_r{region.zfill(3)}.h5ad"
        adata.write_h5ad(out_path)

    typer.echo(f"Wrote {len(slices)} preprocessed slices to {out_dir}/")


@app.command()
def plot_raw(
    slice_dir: Path = typer.Option("data/slices", help="Slice directory"),
    fig_dir: Path = typer.Option("figures/raw", help="Output figure directory"),
    channels: str = typer.Option(
        "all", help="Comma-separated channel indices or 'all'"
    ),
    max_slices: int = typer.Option(12, help="Max slices to show per figure"),
):
    """Plot spatial heatmaps of raw intensities per m/z channel."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    slices = _load_slices(slice_dir)

    if len(slices) > max_slices:
        typer.echo(f"  Showing first {max_slices} of {len(slices)} slices")
        slices = slices[:max_slices]

    n_features = slices[0].n_vars
    if channels == "all":
        ch_indices = list(range(n_features))
    else:
        ch_indices = [int(c) for c in channels.split(",")]

    for ch in ch_indices:
        ch_name = slices[0].var_names[ch]
        ncols = min(6, len(slices))
        nrows = (len(slices) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.atleast_2d(axes)

        vmin = min(np.nanmin(s.X[:, ch]) for s in slices)
        vmax = max(np.nanquantile(s.X[:, ch], 0.99) for s in slices)

        for idx, adata in enumerate(slices):
            ax = axes[idx // ncols, idx % ncols]
            coords = adata.obsm["spatial"]
            vals = np.asarray(adata.X[:, ch]).ravel()
            sc = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=vals,
                s=1,
                cmap="magma",
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_title(f"r{adata.uns.get('region_name', idx)}", fontsize=9)
            ax.set_aspect("equal")
            ax.axis("off")

        # Hide empty axes
        for idx in range(len(slices), nrows * ncols):
            axes[idx // ncols, idx % ncols].axis("off")

        fig.suptitle(f"{ch_name}", fontsize=12, y=1.01)
        fig.colorbar(sc, ax=axes, shrink=0.6, label="Intensity")
        fig.tight_layout()
        fig.savefig(
            fig_dir / f"raw_{ch_name.replace(' ', '_')}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        typer.echo(f"  Saved {ch_name}")

    typer.echo(f"Wrote {len(ch_indices)} channel plots to {fig_dir}/")


@app.command()
def plot_qc(
    slice_dir: Path = typer.Option("data/processed", help="Processed slice directory"),
    fig_dir: Path = typer.Option("figures/qc", help="Output figure directory"),
):
    """Plot QC diagnostics: TIC distributions, spot counts, feature correlations."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    slices = _load_slices(slice_dir)

    # --- TIC distribution per slice ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    tics = []
    labels = []
    spot_counts = []
    for adata in slices:
        region = adata.uns.get("region_name", "?")
        tic = np.asarray(adata.X).sum(axis=1)
        tics.append(tic)
        labels.append(region)
        spot_counts.append(adata.n_obs)

    # Boxplot of TIC
    ax = axes[0]
    positions = range(len(tics))
    ax.boxplot(tics, positions=positions, widths=0.6, showfliers=False)
    ax.set_xlabel("Slice")
    ax.set_ylabel("Total Ion Count")
    ax.set_title("TIC per slice")
    if len(labels) <= 30:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)

    # Spot count bar
    ax = axes[1]
    ax.bar(positions, spot_counts, color="steelblue")
    ax.set_xlabel("Slice")
    ax.set_ylabel("# Spots")
    ax.set_title("Spots per slice")
    if len(labels) <= 30:
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)

    fig.tight_layout()
    fig.savefig(fig_dir / "qc_tic_spots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Feature correlation heatmap (across all spots) ---
    all_X = np.vstack([np.asarray(s.X) for s in slices])
    corr = np.corrcoef(all_X.T)
    feature_names = slices[0].var_names.tolist()

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, fontsize=7)
    ax.set_yticklabels(feature_names, fontsize=7)
    ax.set_title("Feature correlation (all spots)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(fig_dir / "qc_feature_corr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    typer.echo(f"QC plots saved to {fig_dir}/")


if __name__ == "__main__":
    app()
