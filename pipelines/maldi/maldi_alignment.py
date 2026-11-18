"""MALDI-MSI alignment and 3D reconstruction.

Usage:
    python maldi_alignment.py align      --slice-dir data/processed --out \
        data/aligned.h5ad
    python maldi_alignment.py voxelize   --aligned data/aligned.h5ad --out \
        data/volume.zarr
    python maldi_alignment.py plot-align  --aligned data/aligned.h5ad --fig-dir \
        figures/alignment
"""

from pathlib import Path

import anndata as ad
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import typer

app = typer.Typer(pretty_exceptions_enable=False)


def _load_slices(slice_dir: Path) -> list[ad.AnnData]:
    paths = sorted(slice_dir.glob("slice_*.h5ad"))
    if not paths:
        typer.echo(f"No slice files in {slice_dir}", err=True)
        raise typer.Exit(1)
    slices = [ad.read_h5ad(p) for p in paths]
    typer.echo(f"Loaded {len(slices)} slices")
    return slices


def _clean_nans(slices: list[ad.AnnData]) -> list[ad.AnnData]:
    for i, s in enumerate(slices):
        X = np.asarray(s.X)
        if np.isnan(X).any():
            typer.echo(f"  Cleaning NaNs in slice {i}")
            s.X = np.nan_to_num(X, nan=0.0)
    return slices


@app.command()
def align(
    slice_dir: Path = typer.Option(
        "data/processed", help="Preprocessed slice directory"
    ),
    out: Path = typer.Option(
        "data/aligned.h5ad", help="Output aligned concatenated h5ad"
    ),
    method: str = typer.Option("both", help="Alignment: pairwise | center | both"),
    max_iter: int = typer.Option(100000000, help="Max iterations for alignment"),
    seed: int = typer.Option(5, help="Random seed for center alignment"),
):
    """Align serial MALDI sections using PASTE3."""
    import os
    import time
    from concurrent.futures import ThreadPoolExecutor

    from paste3 import helper as paste3_helper
    from paste3.helper import get_common_genes, match_spots_using_spatial_heuristic
    from paste3.paste import center_align, pairwise_align
    from paste3.visualization import stack_slices_center, stack_slices_pairwise

    # Monkey-patch compute_slice_weights to clean NaN/inf from degenerate FGW pi
    # matrices.
    # PASTE3's FGW optimal transport can return pi with NaN entries, which then
    # propagate
    # into the feature matrix via sum(w_i * pi_i @ X_i), crashing NMF in center_NMF.
    _original_csw = paste3_helper.compute_slice_weights

    def _patched_compute_slice_weights(slice_weights, pis, slices, device):
        result = _original_csw(slice_weights, pis, slices, device)
        # result is a torch.Tensor on `device`
        if hasattr(result, "isnan"):
            n_bad = int((~torch.isfinite(result)).sum().item())
            if n_bad:
                typer.echo(f"  [patch] compute_slice_weights: cleaning {n_bad} NaN/inf")
                result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
                # NMF requires non-negative
                result = torch.clamp(result, min=0.0)
        return result

    # Patch in BOTH places it's looked up (paste3.paste imports it directly)
    paste3_helper.compute_slice_weights = _patched_compute_slice_weights
    import paste3.paste as _pp

    _pp.compute_slice_weights = _patched_compute_slice_weights

    # ------------------------------------------------------------------
    # Parallel center_ot patch
    # ------------------------------------------------------------------
    # center_align's inner loop calls pairwise_align(center_slice, slices[i])
    # sequentially, which dominates runtime (~40 min per outer iteration for 59
    # slices on one core). The calls are independent given the center slice, and
    # POT + torch release the GIL, so they run in threads.
    # ------------------------------------------------------------------
    from anndata import AnnData

    _orig_pairwise_inner = _pp.pairwise_align

    def _parallel_center_ot(
        feature_matrix,
        coeff_matrix,
        slices,
        center_coordinates,
        common_genes,
        alpha,
        use_gpu,
        exp_dissim_metric="kl",
        norm=False,
        pi_inits=None,
        spot_weights=None,
        numItermax=200,
    ):
        center_slice = AnnData(np.asarray(feature_matrix) @ np.asarray(coeff_matrix))
        center_slice.var.index = common_genes
        center_slice.obsm["spatial"] = center_coordinates
        if spot_weights is None:
            spot_weights = [None] * len(slices)
        if pi_inits is None:
            pi_inits = [None] * len(slices)

        n = int(os.environ.get("PASTE3_WORKERS", "8"))
        typer.echo(f"  [parallel center_ot] {n} workers × {len(slices)} slices")

        def _task(i):
            pi, loss = _orig_pairwise_inner(
                center_slice,
                slices[i],
                alpha=alpha,
                exp_dissim_metric=exp_dissim_metric,
                pi_init=pi_inits[i],
                b_spots_weight=spot_weights[i],
                norm=norm,
                numItermax=numItermax,
                use_gpu=use_gpu,
            )
            return i, pi, loss["loss"][-1].item()

        with ThreadPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(_task, range(len(slices))))
        results.sort(key=lambda r: r[0])
        pis = [r[1] for r in results]
        losses = np.array([r[2] for r in results])
        return pis, losses

    _pp.center_ot = _parallel_center_ot
    typer.echo("Patched paste3.paste.center_ot for parallel execution")

    slices = _load_slices(slice_dir)
    slices = _clean_nans(slices)
    region_names = [s.uns.get("region_name", str(i)) for i, s in enumerate(slices)]

    # --- Pairwise alignment ---
    # Match example: use very large numItermax for pairwise (effectively unlimited)
    if method in ("pairwise", "both"):
        # Cache pairwise result to avoid re-running 2hr alignment if center fails
        pw_cache = out.parent / f"{out.stem}_pairwise_intermediate.h5ad"
        if pw_cache.exists():
            typer.echo(f"Loading cached pairwise alignment from {pw_cache}")
            pw_combined = ad.read_h5ad(pw_cache)
            # Split back into list per z_slice
            pw_slices = []
            for z in sorted(pw_combined.obs["z_slice"].unique()):
                pw_slices.append(pw_combined[pw_combined.obs["z_slice"] == z].copy())
        else:
            typer.echo("Running pairwise alignment (parallel)...")
            t0 = time.time()
            n_workers = int(os.environ.get("PASTE3_WORKERS", "8"))

            def _pw(i):
                pi, _ = pairwise_align(
                    slices[i],
                    slices[i + 1],
                    numItermax=10**37,
                )
                return i, pi

            with ThreadPoolExecutor(max_workers=n_workers) as ex:
                futures = [ex.submit(_pw, i) for i in range(len(slices) - 1)]
                results = [f.result() for f in futures]
            results.sort(key=lambda r: r[0])
            pis = [r[1] for r in results]
            pw_slices, _, _ = stack_slices_pairwise(slices, pis)
            typer.echo(f"Pairwise alignment done in {time.time() - t0:.1f}s")

            # Save pairwise intermediate cache
            for z_idx, s in enumerate(pw_slices):
                s.obs["z_slice"] = z_idx
            pw_cache.parent.mkdir(parents=True, exist_ok=True)
            ad.concat(pw_slices, join="outer").write_h5ad(pw_cache)
            typer.echo(f"Cached pairwise result to {pw_cache}")

    # --- Center alignment ---
    # Match example: re-clean NaNs after pairwise, copy initial before get_common_genes
    if method in ("center", "both"):
        typer.echo("Running center alignment...")
        t0 = time.time()
        # Re-clean NaNs (pairwise alignment can introduce them)
        slices = _clean_nans(slices)
        initial_slice = slices[0].copy()
        lmbda = [1.0 / len(slices)] * len(slices)
        slices_common, _ = get_common_genes(slices)
        # NOTE: match_spots_using_spatial_heuristic takes spatial coordinates
        # (.obsm["spatial"]), not the 14-channel feature matrix; passing .X gives
        # meaningless pi_inits and worse center_align convergence.
        pi_inits = [
            torch.Tensor(
                match_spots_using_spatial_heuristic(
                    slices_common[0].obsm["spatial"],
                    s.obsm["spatial"],
                )
            ).double()
            for s in slices_common
        ]
        # NaN check before alignment
        for i, s in enumerate(slices_common):
            if np.isnan(np.asarray(s.X)).any():
                typer.echo(f"  [WARNING] Slice {i} still has NaNs before center_align")
        center_slice, pis_center = center_align(
            initial_slice,
            slices_common,
            lmbda,
            random_seed=seed,
            pi_inits=pi_inits,
            max_iter=max_iter,
        )
        # NaN check after alignment
        for i, s in enumerate(slices_common):
            if np.isnan(np.asarray(s.X)).any():
                typer.echo(f"  [WARNING] Slice {i} has NaNs after center_align")
        center, new_slices, _, _ = stack_slices_center(
            center_slice, slices_common, pis_center
        )
        typer.echo(f"Center alignment done in {time.time() - t0:.1f}s")

    # --- Build aligned output ---
    # Use center-aligned slices if available, otherwise pairwise
    if method in ("center", "both"):
        aligned = new_slices
    else:
        aligned = pw_slices

    # Add z-coordinate and concatenate
    for z_idx, s in enumerate(aligned):
        s.obs["z_slice"] = z_idx
        s.obs["region_name"] = (
            region_names[z_idx] if z_idx < len(region_names) else str(z_idx)
        )
        s.obsm["spatial_3d"] = np.column_stack(
            [
                s.obsm["spatial"],
                np.full(s.n_obs, z_idx, dtype=np.float64),
            ]
        )

    combined = ad.concat(aligned, join="outer")
    # PASTE3's center_align drops var metadata; restore from input slices
    if slices and slices[0].var.shape[1] > 0:
        src_var = slices[0].var
        shared = combined.var_names.intersection(src_var.index)
        for col in src_var.columns:
            combined.var.loc[shared, col] = src_var.loc[shared, col]
    combined.uns["alignment_method"] = method
    combined.uns["n_slices"] = len(aligned)
    combined.uns["region_names"] = region_names

    out.parent.mkdir(parents=True, exist_ok=True)
    combined.write_h5ad(out)
    typer.echo(
        f"Wrote aligned data: {out} ({combined.n_obs} "
        f"spots, {combined.n_vars} features)"
    )


@app.command()
def voxelize(
    aligned: Path = typer.Option("data/aligned.h5ad", help="Aligned h5ad"),
    out: Path = typer.Option("data/volume.zarr", help="Output zarr volume"),
    voxel_size: int = typer.Option(20, help="Voxel size in coordinate units"),
):
    """Voxelize aligned MALDI data into a 4D volume (channels, z, y, x)."""
    import zarr

    adata = ad.read_h5ad(aligned)
    typer.echo(f"Loaded {adata.n_obs} spots, {adata.n_vars} features")

    coords_3d = adata.obsm["spatial_3d"]
    X = np.asarray(adata.X, dtype=np.float32)
    n_channels = X.shape[1]

    # Voxelize
    vx = (coords_3d[:, 0] // voxel_size).astype(int)
    vy = (coords_3d[:, 1] // voxel_size).astype(int)
    vz = coords_3d[:, 2].astype(int)

    # Shift to non-negative
    vx -= vx.min()
    vy -= vy.min()
    vz -= vz.min()

    shape = (n_channels, vz.max() + 1, vy.max() + 1, vx.max() + 1)
    typer.echo(f"Volume shape: {shape} (channels, z, y, x)")

    volume = np.zeros(shape, dtype=np.float32)
    counts = np.zeros(shape[1:], dtype=np.int32)

    for i in range(len(vx)):
        volume[:, vz[i], vy[i], vx[i]] += X[i]
        counts[vz[i], vy[i], vx[i]] += 1

    # Average where multiple spots map to same voxel
    mask = counts > 0
    for c in range(n_channels):
        volume[c][mask] /= counts[mask]

    # Save as zarr
    store = zarr.open(str(out), mode="w")
    store.create_dataset("volume", data=volume, chunks=(1, shape[1], 64, 64))
    store.create_dataset("counts", data=counts, chunks=(shape[1], 64, 64))
    store.attrs["voxel_size"] = voxel_size
    store.attrs["feature_names"] = adata.var_names.tolist()
    store.attrs["shape"] = list(shape)

    typer.echo(f"Wrote volume to {out}")
    typer.echo(f"  Non-empty voxels: {mask.sum()} / {np.prod(shape[1:])}")


@app.command()
def plot_align(
    aligned: Path = typer.Option("data/aligned.h5ad", help="Aligned h5ad"),
    fig_dir: Path = typer.Option("figures/alignment", help="Output figure directory"),
):
    """Plot alignment results: overlay of all slices, per-slice view."""
    fig_dir.mkdir(parents=True, exist_ok=True)

    adata = ad.read_h5ad(aligned)
    z_slices = sorted(adata.obs["z_slice"].unique())
    n_slices = len(z_slices)

    cmap = plt.get_cmap("tab20")
    colors = [mcolors.to_hex(cmap(i / max(n_slices, 1))) for i in range(n_slices)]

    # --- Overlay plot ---
    fig, ax = plt.subplots(figsize=(8, 8))
    for idx, z in enumerate(z_slices):
        mask = adata.obs["z_slice"] == z
        coords = adata[mask].obsm["spatial"]
        ax.scatter(
            coords[:, 0], coords[:, 1], c=colors[idx], s=1, alpha=0.5, rasterized=True
        )

    handles = [
        mpatches.Patch(color=colors[i], label=f"z={z}") for i, z in enumerate(z_slices)
    ]
    if n_slices <= 20:
        ax.legend(handles=handles, fontsize=6, ncol=2)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(f"Aligned MALDI slices (n={n_slices})")
    fig.tight_layout()
    fig.savefig(fig_dir / "aligned_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- Per-slice grid ---
    ncols = min(6, n_slices)
    nrows = (n_slices + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes)

    for idx, z in enumerate(z_slices):
        ax = axes[idx // ncols, idx % ncols]
        mask = adata.obs["z_slice"] == z
        coords = adata[mask].obsm["spatial"]
        ax.scatter(coords[:, 0], coords[:, 1], c=colors[idx], s=1, rasterized=True)
        region = (
            adata.obs.loc[mask, "region_name"].iloc[0]
            if "region_name" in adata.obs
            else str(z)
        )
        ax.set_title(f"r{region}", fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")

    for idx in range(n_slices, nrows * ncols):
        axes[idx // ncols, idx % ncols].axis("off")

    fig.suptitle("Per-slice aligned coordinates")
    fig.tight_layout()
    fig.savefig(fig_dir / "aligned_per_slice.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    typer.echo(f"Alignment plots saved to {fig_dir}/")


if __name__ == "__main__":
    app()
