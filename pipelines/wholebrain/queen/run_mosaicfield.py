#!/usr/bin/env python
"""Bee brain serial slice alignment pipeline (MOSAICField).

4-step pipeline for registering serial Xenium spatial transcriptomics slices:
  1. Affine alignment via minibatch Fused Gromov-Wasserstein optimal transport
  2. Chain affine transforms, export aligned images and coordinates
  3. Nonlinear alignment via neural displacement field (GPU recommended)
  4. Apply composed nonlinear flows to cell coordinates

Usage:
    python run_mosaicfield.py                               # first 6 slices
    python run_mosaicfield.py --slices all                   # all 30 slices
    python run_mosaicfield.py --slices 1A,1B,2B,3A,3B,4A,4B # pick specific
    python run_mosaicfield.py --slices 0:15                  # first 15
    python run_mosaicfield.py --slices all --chunk-size 5    # chunked alignment
    python run_mosaicfield.py --steps 3,4
"""

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import scanpy as sc
import typer
from MOSAICField.pipeline import (
    step1_affine,
    step2_chain_affine,
    step3_nonlinear,
    step4_apply_flows,
)
from MOSAICField.plotting import (
    plot_step1_affine,
    plot_step2_grid,
    plot_step2_sequence,
    plot_step3_nonlinear,
    plot_step4_overview,
    plot_step4_sequence,
)

# ---------------------------------------------------------------------------
# Project-specific constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "data"
PS = 0.2125  # pixel size (microns/pixel)
MINX, MAXX = 0, int(21000 * 0.22)
MINY, MAXY = 0, int(14000 * 0.22)
REGION = (MINX, MAXX, MINY, MAXY)

ALL_IDS = [
    "1A",
    "1B",
    "2A",
    "2B",
    "3A",
    "3B",
    "4A",
    "4B",
    "5A",
    "5B",
    "6A",
    "6B",
    "7A",
    "7B",
    "8A",
    "2_1A",
    "2_1B",
    "2_2A",
    "2_2B",
    "2_3A",
    "2_3B",
    "2_4A",
    "2_4B",
    "2_5A",
    "2_5B",
    "2_6A",
    "2_6B",
    "2_7A",
    "2_7B",
    "2_8A",
]
SKIP = {"2A", "7B", "2_3A", "2_3B", "2_6A", "2_7A"}


def id_to_path(sid: str) -> Path:
    """Map a slice ID like '1A' or '2_3B' to its h5ad file path."""
    if sid.startswith("2_"):
        region = int(sid[2:-1])
        return (
            DATA_DIR
            / f"output-XETG00151__0040243__Region_{region}__20251204__195015_{sid}.h5ad"
        )
    region = int(sid[:-1])
    ts = "195014" if region == 1 else "195015"
    return (
        DATA_DIR
        / f"output-XETG00151__0040240__Region_{region}__20251204__{ts}_{sid}.h5ad"
    )


def parse_slices(spec: str, skip: set[str] | None = None) -> list[str]:
    """Parse a slice specification string into a list of slice IDs.

    For range specs (``0:4``), skipped slices don't count toward the total —
    the range expands until enough active slices are collected.

    Formats:
      "all"        -> ALL_IDS (minus skip)
      "0:4"        -> first 4 non-skipped slices starting at index 0
      "1A,1B,2B"   -> ["1A", "1B", "2B"] (explicit, no filtering)
    """
    skip = skip or set()
    spec = spec.strip()
    if spec.lower() == "all":
        return [s for s in ALL_IDS if s not in skip]
    if ":" in spec and "," not in spec:
        parts = spec.split(":")
        start = int(parts[0]) if parts[0] else 0
        count = (int(parts[1]) if parts[1] else len(ALL_IDS)) - start
        result = []
        for sid in ALL_IDS[start:]:
            if sid not in skip:
                result.append(sid)
                if len(result) >= count:
                    break
        return result
    return [s.strip() for s in spec.split(",")]


# ---------------------------------------------------------------------------
# Callbacks for pipeline steps
# ---------------------------------------------------------------------------


def load_slice_coords(sid):
    """Load slice for step1: returns (coords_xy, features)."""
    ad = sc.read_h5ad(id_to_path(sid))
    coords = np.array(ad.obsm["spatial"]) / PS - ad.uns["crop_coords"]["min"]
    features = ad.X.toarray()
    return coords, features


def load_slice_anndata(sid):
    """Load slice for step2: returns AnnData."""
    return sc.read_h5ad(id_to_path(sid))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
app = typer.Typer(help="Bee brain serial slice alignment (MOSAICField).")


@app.command()
def main(
    device: str = typer.Option("cuda:0", help="Torch device for nonlinear step"),
    epochs: int = typer.Option(3000, help="Training epochs per slice pair"),
    lr: float = typer.Option(0.005, help="Learning rate"),
    lambda_j: float = typer.Option(
        2.5, "--lambda-J", help="Jacobian regularization weight"
    ),
    lambda_v: float = typer.Option(
        0.0005, "--lambda-v", help="Deformation magnitude weight"
    ),
    slices: str = typer.Option("0:6", help="Slice spec: 'all', '0:6', or '1A,1B,2B'"),
    chunk_size: int = typer.Option(
        0, "--chunk-size", help="Chunk size (0 = no chunking)"
    ),
    output_dir: Optional[Path] = typer.Option(  # noqa: B008
        None, "--output-dir", help="Output directory"
    ),
    steps: str = typer.Option(
        "1,2,3,4", help="Comma-separated steps to run, e.g. '3,4'"
    ),
    debug: bool = typer.Option(
        False, "--debug", help="Save extra debug plots to output/debug/"
    ),
    force: bool = typer.Option(
        False, "--force", help="Recompute all steps, ignoring cached results"
    ),
    align_skipped: bool = typer.Option(
        False,
        "--align-skipped",
        help="Include skipped slices in affine alignment (steps 1-2)",
    ),
    no_skip: bool = typer.Option(
        False,
        "--no-skip",
        help="Disable SKIP set entirely — include all slices in all steps",
    ),
):
    skip = set() if no_skip else SKIP
    active = parse_slices(slices, skip=skip)
    out = output_dir or SCRIPT_DIR / "output"
    out.mkdir(parents=True, exist_ok=True)
    run = {int(s) for s in steps.split(",")}

    # When --align-skipped, re-insert skipped slices that fall within the range
    # so they get affine-aligned (steps 1-2) but still skip nonlinear (3-4).
    if align_skipped and len(active) >= 2:
        first = ALL_IDS.index(active[0])
        last = ALL_IDS.index(active[-1])
        all_in_range = list(ALL_IDS[first : last + 1])
        skipped_in_range = SKIP & set(all_in_range)
    else:
        all_in_range = list(active)
        skipped_in_range = set()

    # nonlinear_alignment.py uses `config.epoches` (upstream spelling)
    config = SimpleNamespace(
        epoches=epochs, lr=lr, lambda_J=lambda_j, lambda_v=lambda_v, device=device
    )
    if skipped_in_range:
        print(
            f"slices={all_in_range}  affine-only={skipped_in_range}  output={out}  "
            f"device={device}  epochs={epochs}  debug={debug}"
        )
    else:
        print(
            f"slices={active}  output={out}  "
            f"device={device}  epochs={epochs}  debug={debug}"
        )

    if chunk_size > 0:
        chunks = [active[i : i + chunk_size] for i in range(0, len(active), chunk_size)]
    else:
        chunks = [active]

    for ci, chunk in enumerate(chunks):
        if len(chunks) > 1:
            print(f"\n--- Chunk {ci}: {chunk[0]}..{chunk[-1]} (ref={chunk[-1]}) ---")

        # For affine steps, include skipped slices that fall within this chunk
        if skipped_in_range:
            first = ALL_IDS.index(chunk[0])
            last = ALL_IDS.index(chunk[-1])
            affine_ids = list(ALL_IDS[first : last + 1])
            pass  # chunk_skip computed implicitly via affine_ids - chunk
        else:
            affine_ids = chunk

        # Prepare debug dir once
        dbg = out / "debug"
        if debug:
            dbg.mkdir(exist_ok=True)

        # Step 1: Pairwise affine alignment
        if 1 in run:

            def _on_step1_pair(a, b, t, _dbg=dbg, _ids=affine_ids):
                if not debug:
                    return
                png = _dbg / f"step1_affine_{a}_{b}.png"
                if force or not png.exists():
                    pts_a, _ = load_slice_coords(a)
                    pts_b, _ = load_slice_coords(b)
                    plot_step1_affine(a, b, t, _ids, _dbg, pts_a, pts_b, PS)

            step1_affine(
                affine_ids, out, load_slice_coords, force=force, on_pair=_on_step1_pair
            )

        # Step 2: Chain affine transforms
        if 2 in run:
            # Track state for grid plots (need accumulated results)
            s2_imgs = {}
            s2_tfs = {}

            def _on_step2_slice(
                sid,
                aligned_img,
                chained_tf,
                _imgs=s2_imgs,
                _tfs=s2_tfs,
                _ids=affine_ids,
            ):
                _imgs[sid] = aligned_img
                _tfs[sid] = chained_tf
                if force or not (out / f"{sid}_grid.png").exists():
                    plot_step2_grid(sid, _ids, _tfs, _imgs, out, REGION, PS)

            aligned_imgs, chained_transforms = step2_chain_affine(
                affine_ids,
                out,
                load_slice_anndata,
                REGION,
                PS,
                force=force,
                on_slice=_on_step2_slice,
            )
            if debug:
                plot_step2_sequence(affine_ids, aligned_imgs, dbg)

        # Step 3: Nonlinear alignment (skipped slices excluded)
        if 3 in run:

            def _on_step3_pair(src, tgt, _dbg=dbg):
                if not debug:
                    return
                png = _dbg / f"step3_nonlinear_{src}_{tgt}.png"
                if force or not png.exists():
                    plot_step3_nonlinear(src, tgt, out, _dbg)

            step3_nonlinear(
                chunk, set(), out, config, force=force, on_pair=_on_step3_pair
            )

        # Step 4: Apply flows to cell coordinates (skipped slices excluded)
        if 4 in run:
            orig_img, orig_pts, warped_img, warped_pts = step4_apply_flows(
                chunk, set(), out
            )
            # Skipped slices: carry affine-only coords as final
            affine_only = set(affine_ids) - set(chunk)
            for sid in affine_only:
                cells_path = out / f"{sid}_cells.npy"
                if cells_path.exists():
                    shutil.copy(cells_path, out / f"{sid}_cells_final.npy")
                    print(f"  {sid} (affine only)")

            plot_step4_overview(
                chunk, orig_img, orig_pts, warped_img, warped_pts, out, REGION
            )
            if debug:
                plot_step4_sequence(chunk, orig_img, warped_img, dbg)

    print("Done!")


if __name__ == "__main__":
    app()
