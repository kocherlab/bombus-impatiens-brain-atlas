#!/usr/bin/env python
"""Stitch the chunked MOSAICField alignment into one 3D point cloud (queen).

Builds the cache that ``scripts/atlas/si_mosaicfield.py`` (Fig. mosaicfield_si,
panels E-F) reads, ``results/mosaicfield_stitched/stitched_coords_3d.npy``.

Run order (queen brain, 30 sections, repo root):

  1. chunked nonlinear alignment, 6 sections per chunk (GPU; MOSAICField env):
       python pipelines/wholebrain/queen/run_mosaicfield.py --slices all \\
           --chunk-size 6 --output-dir results/mosaicfield_chunked
     -> results/mosaicfield_chunked/<section>_cells_final.npy (chunk-local um)
  2. this script:
       PYTHONPATH=src:scripts .venv/bin/python scripts/atlas/mosaicfield_stitch.py
     a. affine-aligns consecutive chunk references (3B -> 6B -> 2_2A -> 2_5A -> 2_8A,
        the last section of each chunk) by minibatch FGW optimal transport on the
        chunk-aligned cells, cached as results/mosaicfield_stitched/T_ref_<a>_<b>.npy;
     b. chains them to the last chunk's frame, applies them to every section's
        cells, and stacks sections at z = 10 um x section index.
     Step a needs the MOSAICField package (submodules/MOSAICField; not in .venv) and
     runs only for a transform that is not cached; the four stored transforms are
     cached, so a rerun from them is deterministic and needs only numpy.
  3. scripts/atlas/si_mosaicfield.py renders the figure.

Outputs: results/mosaicfield_stitched/{stitched_coords_3d,stitched_slice_ids}.npy.
``--force`` rebuilds the stitched arrays even if they exist (transforms still come
from the cache unless deleted).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
CHUNK_DIR = REPO / "results" / "mosaicfield_chunked"
STITCH_DIR = REPO / "results" / "mosaicfield_stitched"
DATA_DIR = REPO / "pipelines" / "wholebrain" / "queen" / "data"

PS = 0.2125  # pixel size, um/px

# fmt: off
ALL_IDS = [
    "1A", "1B", "2A", "2B", "3A", "3B", "4A", "4B", "5A", "5B",
    "6A", "6B", "7A", "7B", "8A",
    "2_1A", "2_1B", "2_2A", "2_2B", "2_3A", "2_3B", "2_4A", "2_4B",
    "2_5A", "2_5B", "2_6A", "2_6B", "2_7A", "2_7B", "2_8A",
]
# fmt: on
CHUNK_SIZE = 6
CHUNKS = [ALL_IDS[i : i + CHUNK_SIZE] for i in range(0, len(ALL_IDS), CHUNK_SIZE)]
CHUNK_REFS = [chunk[-1] for chunk in CHUNKS]  # 3B, 6B, 2_2A, 2_5A, 2_8A


def id_to_h5ad(sid: str) -> Path:
    """Raw per-crop h5ad of a queen section."""
    if sid.startswith("2_"):
        reg = int(sid[2:-1])
        return DATA_DIR / (
            f"output-XETG00151__0040243__Region_{reg}__20251204__195015_{sid}.h5ad"
        )
    reg = int(sid[:-1])
    ts = "195014" if reg == 1 else "195015"
    return DATA_DIR / (
        f"output-XETG00151__0040240__Region_{reg}__20251204__{ts}_{sid}.h5ad"
    )


def transform_coords(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 3x3 affine to Nx2 points."""
    hom = np.hstack([pts, np.ones((len(pts), 1))])
    out = hom @ T.T
    return out[:, :2] / out[:, 2:3]


def align_chunk_refs() -> list[np.ndarray]:
    """Affine transforms between consecutive chunk references (cached)."""
    STITCH_DIR.mkdir(parents=True, exist_ok=True)
    transforms = []
    for src_id, tgt_id in zip(CHUNK_REFS[:-1], CHUNK_REFS[1:], strict=True):
        tf_path = STITCH_DIR / f"T_ref_{src_id}_{tgt_id}.npy"
        if tf_path.exists():
            print(f"  cached {src_id} -> {tgt_id}", flush=True)
            transforms.append(np.load(tf_path))
            continue
        print(f"  aligning {src_id} -> {tgt_id} (minibatch FGW)", flush=True)
        import scanpy as sc
        from MOSAICField.affine_alignment import minibatch_FGW_affine

        coords_src = np.load(CHUNK_DIR / f"{src_id}_cells_final.npy") / PS
        coords_tgt = np.load(CHUNK_DIR / f"{tgt_id}_cells_final.npy") / PS
        feat_src = sc.read_h5ad(id_to_h5ad(src_id)).X.toarray()
        feat_tgt = sc.read_h5ad(id_to_h5ad(tgt_id)).X.toarray()
        _, _, T = minibatch_FGW_affine(
            coords_src,
            coords_tgt,
            feat_src,
            feat_tgt,
            batch_size=3000,
            max_iter=10,
            alpha=0.1,
            device="cpu",
        )
        np.save(tf_path, T)
        transforms.append(T)
    return transforms


def stitch_chunks(inter_chunk_tfs: list[np.ndarray]):
    """Chain the inter-chunk affines to the last chunk's frame and stack sections."""
    n = len(CHUNKS)
    chained = [None] * n
    chained[n - 1] = np.eye(3)
    for i in range(n - 2, -1, -1):
        chained[i] = chained[i + 1] @ inter_chunk_tfs[i]

    pts_all, z_all, ids_all = [], [], []
    for ci, chunk in enumerate(CHUNKS):
        for sid in chunk:
            p = CHUNK_DIR / f"{sid}_cells_final.npy"
            if not p.exists():
                continue
            pts = transform_coords(np.load(p) / PS, chained[ci]) * PS
            pts_all.append(pts)
            z_all.append(np.full(len(pts), ALL_IDS.index(sid) * 10.0))
            ids_all.append(np.full(len(pts), sid, dtype=object))
            print(f"  {sid}: {len(pts):,} cells", flush=True)
    pts_3d = np.hstack([np.vstack(pts_all), np.concatenate(z_all)[:, None]])
    slice_ids = np.concatenate(ids_all)
    np.save(STITCH_DIR / "stitched_coords_3d.npy", pts_3d)
    np.save(STITCH_DIR / "stitched_slice_ids.npy", slice_ids)
    print(f"{len(pts_3d):,} cells -> {STITCH_DIR}", flush=True)
    return pts_3d, slice_ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--force", action="store_true", help="rebuild the stitched arrays")
    args = ap.parse_args()
    if (STITCH_DIR / "stitched_coords_3d.npy").exists() and not args.force:
        print("stitched cache exists; pass --force to rebuild", flush=True)
        return
    stitch_chunks(align_chunk_refs())


if __name__ == "__main__":
    main()
