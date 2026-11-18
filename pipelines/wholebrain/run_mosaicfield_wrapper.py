#!/usr/bin/env python3
"""MOSAICField alignment wrapper for wholebrain pipeline.

Runs MOSAICField 4-step pipeline on per-slice h5ad files, collects
*_cells_final.npy outputs, and assembles a single aligned.h5ad
matching the PASTE3 output format.

Shared by both brains; run it from the brain's directory (see brain_dir.py).

Usage (from pipelines/wholebrain/queen or pipelines/wholebrain/worker):
    python ../run_mosaicfield_wrapper.py align --device cuda:0 --epochs 3000

Snakefile integration:
    This script is called by the `align` rule when config["aligner"] == "mosaicfield".
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import anndata as ad
import numpy as np
import scanpy as sc
import typer
from brain_dir import BRAIN_DIR
from MOSAICField.pipeline import (
    step1_affine,
    step2_chain_affine,
    step3_nonlinear,
    step4_apply_flows,
)

app = typer.Typer(help="MOSAICField alignment wrapper")

DATA_DIR = BRAIN_DIR / "data"
PS = 0.2125  # pixel size (microns/pixel)

# Bounding box for rasterization (microns) — covers full Xenium FOV
MINX, MAXX = 0, int(21000 * 0.22)
MINY, MAXY = 0, int(14000 * 0.22)
REGION = (MINX, MAXX, MINY, MAXY)


def _discover_slices() -> list[str]:
    """Find all per-slice h5ad files and return ordered short IDs."""
    # Pattern: *_1A.h5ad, *_2B.h5ad, *_2_3A.h5ad
    import re

    h5ads = sorted(DATA_DIR.glob("output-*.h5ad"))
    ids = []
    paths = {}
    for p in h5ads:
        m = re.search(r"_(\d_?\d[A-D])\.h5ad$", p.name)
        if m:
            sid = m.group(1)
            ids.append(sid)
            paths[sid] = p
    return ids, paths


def _load_coords(paths):
    """Return callback for step1: (coords_xy, features)."""

    def fn(sid):
        adata = sc.read_h5ad(paths[sid])
        coords = np.array(adata.obsm["spatial"]) / PS - adata.uns["crop_coords"]["min"]
        features = (
            adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
        )
        return coords, features

    return fn


def _load_anndata(paths):
    """Return callback for step2: AnnData."""

    def fn(sid):
        return sc.read_h5ad(paths[sid])

    return fn


def _assemble_aligned(ids: list[str], paths: dict, out_dir: Path, output_path: Path):
    """Assemble per-slice *_cells_final.npy into a single aligned.h5ad."""
    typer.echo("Assembling aligned.h5ad from per-slice coordinates...")
    adatas = []
    for i, sid in enumerate(ids):
        final = out_dir / f"{sid}_cells_final.npy"
        if not final.exists():
            typer.echo(f"  Warning: {final.name} not found, using affine coords")
            final = out_dir / f"{sid}_cells.npy"
        if not final.exists():
            typer.echo(f"  Skipping {sid}: no aligned coordinates found")
            continue

        coords = np.load(final)
        adata = sc.read_h5ad(paths[sid])
        # Store MOSAICField-aligned coords as spatial_aligned
        adata.obsm["spatial_aligned"] = coords[: len(adata)]
        adata.obs["z_level"] = float(i * 10)  # 10µm spacing
        adata.obs["slice_name"] = sid
        adatas.append(adata)

    if not adatas:
        raise RuntimeError("No slices assembled — check MOSAICField output")

    combined = ad.concat(adatas, join="outer")
    combined.obs_names_make_unique()
    typer.echo(
        f"Writing {output_path.name}: {combined.shape[0]} cells, {len(adatas)} slices"
    )
    combined.write_h5ad(output_path)


@app.command()
def align(
    device: str = typer.Option("cuda:0", help="Torch device"),
    epochs: int = typer.Option(3000, help="Training epochs per pair"),
    lr: float = typer.Option(0.005, help="Learning rate"),
    lambda_j: float = typer.Option(2.5, "--lambda-J", help="Jacobian regularization"),
    lambda_v: float = typer.Option(
        0.0005, "--lambda-v", help="Magnitude regularization"
    ),
    chunk_size: int = typer.Option(0, "--chunk-size", help="0 = no chunking"),
    steps: str = typer.Option("1,2,3,4", help="Steps to run"),
    force: bool = typer.Option(False, "--force", help="Recompute all"),
    output_dir: Optional[Path] = typer.Option(None, help="MOSAICField cache dir"),
):
    """Run MOSAICField alignment and produce aligned.h5ad."""
    ids, paths = _discover_slices()
    if not ids:
        typer.echo("No slice h5ad files found in data/")
        raise typer.Exit(1)

    out = output_dir or BRAIN_DIR / "mosaicfield_output"
    out.mkdir(parents=True, exist_ok=True)
    run = {int(s) for s in steps.split(",")}

    config = SimpleNamespace(
        epoches=epochs,
        lr=lr,
        lambda_J=lambda_j,
        lambda_v=lambda_v,
        device=device,
    )
    typer.echo(
        f"Slices: {len(ids)}, device={device}, epochs={epochs}, chunk_size={chunk_size}"
    )

    # Chunking
    if chunk_size > 0:
        chunks = [ids[i : i + chunk_size] for i in range(0, len(ids), chunk_size)]
    else:
        chunks = [ids]

    for ci, chunk in enumerate(chunks):
        if len(chunks) > 1:
            typer.echo(f"\n--- Chunk {ci}: {chunk[0]}..{chunk[-1]} ---")

        if 1 in run:
            step1_affine(chunk, out, _load_coords(paths), force=force)

        if 2 in run:
            step2_chain_affine(
                chunk,
                out,
                _load_anndata(paths),
                REGION,
                PS,
                force=force,
            )

        if 3 in run:
            step3_nonlinear(chunk, set(), out, config, force=force)

        if 4 in run:
            step4_apply_flows(chunk, set(), out)

    # Assemble final h5ad
    _assemble_aligned(ids, paths, out, DATA_DIR / "aligned.h5ad")
    typer.echo("Done!")


if __name__ == "__main__":
    app()
