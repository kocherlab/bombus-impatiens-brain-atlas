"""Write clean, release-ready copies of the atlas objects into the deposit staging area.

The canonical objects are correct but carry traps that are only safe inside
this repo, behind ``bimp_st.paths.load_tangram()``. A public copy has to be
safe for someone who opens it with plain ``anndata.read_h5ad``. This script
reads the canonical objects (never modifies them) and writes new files under
``<staging>/zenodo/Z1_atlas/public/``.

What the cleanup does, trap by trap (see ``bimp_st.paths``):

* **Duplicate obs_names** (562,028 in the canonical object). Rewritten as
  ``<brain>__<slice_name>__<cell_id>``, asserted unique; the original index is
  kept in ``obs['source_obs_name']``.
* **Poisoned column** ``tangram_cell_type_stitched`` (built from runs whose
  columns were named by position, not identity). Dropped. The correctly
  recomposed stitched labels go into the label table from
  ``paths.CANONICAL_STITCHED`` instead.
* **Pre-relabel region flags.** The canonical object's six ``in_*`` columns predate
  the nine-region relabel (``annotations_v4``; the per-brain 9-region objects
  keep the old values as ``in_*__prerepair``). They are dropped and replaced by
  the eight-region flags from ``nine_region_common.attach_region_flags``,
  joined on (slice_name, cell_id), kept as nullable booleans (NA = untraced,
  never filled with False).
* **PASTE-frame ``obsm['spatial']``.** Verified identical to
  ``obsm['spatial_aligned']`` and then removed, so no reader mistakes the
  registration frame for slide microns. ``obsm['spatial_aligned']`` is kept and
  ``obsm['spatial_um']`` is added: raw section coordinates in microns from the
  per-crop h5ads, joined on (slice_name, cell_id), never positionally.
* **Cell-type identity.** ``tangram_cell_type`` (per-cell argmax of the
  canonical run) is kept, and ``cell_type`` is added as the canonical label:
  ``cluster_labels.hierarchical_vote(cluster_col='leiden',
  label_col='tangram_cell_type')``, exactly as fig_atlas_overview /
  caste_comparison_sccoda /
  webviz export_data call it. ``anatomy_guided_vote`` is never used.
* The neighbour graph (``obsp``, ``uns['neighbors']``) is dropped; it is
  reproducible and is most of the file.

Outputs:

    public/bimp_atlas_xenium_public.h5ad      every atlas cell x the panel genes
    public/tangram_labels_public.parquet       per-cell labels from the canonical run,
                                               its two seed replicates, and the
                                               central / optic / stitched runs
    public/EXPORT_REPORT.json                  counts and checks behind every step

Default is a dry run that lists inputs and outputs. ``--run`` writes. Needs
~64 GB RAM; run it through Slurm, not on a login node (site options in
config/profiles/slurm/sbatch.env):

    sbatch $BIMP_SBATCH_CPU $BIMP_SBATCH_SHORT \\
      -c 4 --mem=96G -t 59:00 -o logs/deposit_export_%j.out \\
      --wrap "cd $PWD && PYTHONPATH=src:scripts .venv/bin/python \\
              scripts/deposit/export_public_objects.py --run"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from bimp_st import paths  # noqa: E402
from bimp_st.site import get as site_path  # noqa: E402

STAGING = site_path("deposit_staging")
SEED_RUNS = ("combined_clustered_g2_R2_s43", "combined_clustered_g2_R2_s44")
# Per-crop h5ads live in the checkout that holds pipelines/*/data (site key
# data_repo; a second checkout shares results/ by symlink but not those).
DATA_ROOT = site_path("data_repo")
CROP_DIRS = {
    "queen": DATA_ROOT / "pipelines/wholebrain/queen/data",
    "worker": DATA_ROOT / "pipelines/wholebrain/worker/data",
}
LABEL_COLS = (
    "tangram_cell_type",
    "tangram_confidence",
    "tangram_margin",
    "tangram_entropy",
)


def crop_files(brain: str) -> list[Path]:
    out = []
    for f in sorted(CROP_DIRS[brain].glob("output-XETG00151__*.h5ad")):
        if "_clustered" in f.name or "_labeled" in f.name:
            continue
        out.append(f)
    return out


def inputs() -> dict[str, list[Path]]:
    t = paths.TANGRAM_DIR
    return {
        "canonical": [paths.CANONICAL_TANGRAM],
        "seed_replicates": [t / f"{r}_tangram.h5ad" for r in SEED_RUNS],
        "region_runs": [
            paths.CANONICAL_CENTRAL,
            paths.CANONICAL_OPTIC,
            paths.CANONICAL_STITCHED,
        ],
        "region_flags": [paths.RESULTS / "nine_region/region_flags.parquet"],
        "crops": [f for b in CROP_DIRS for f in crop_files(b)],
    }


def raw_um_coords() -> pd.DataFrame:
    """(slice_name, cell_id) -> raw section microns, without loading uns['image']."""
    import h5py
    from anndata.io import read_elem  # anndata >= 0.11; fall back below

    frames = []
    for brain in CROP_DIRS:
        for f in crop_files(brain):
            with h5py.File(f, "r") as h:
                obs = h["obs"]
                cid = (
                    read_elem(obs["cell_id"])
                    if "cell_id" in obs
                    else read_elem(obs[obs.attrs["_index"]])
                )
                xy = h["obsm"]["spatial"][:, :2].astype(np.float64)
            frames.append(
                pd.DataFrame(
                    {
                        "slice_name": f.stem,
                        "cell_id": np.asarray(cid).astype(str),
                        "x_um": xy[:, 0],
                        "y_um": xy[:, 1],
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True)
    if df.duplicated(["slice_name", "cell_id"]).any():
        raise SystemExit("(slice_name, cell_id) not unique across the per-crop h5ads")
    return df


def key(obs: pd.DataFrame) -> np.ndarray:
    return (
        obs["slice_name"].astype(str).to_numpy()
        + "||"
        + obs["cell_id"].astype(str).to_numpy()
    )


def export_atlas(out_dir: Path, report: dict) -> Path:
    import scipy.sparse as sp

    from bimp_st.cluster_labels import hierarchical_vote
    from bimp_st.nine_region_common import attach_region_flags

    a = paths.load_tangram()  # renames the poisoned column out of the way
    report["n_cells"] = int(a.n_obs)
    report["dup_obs_names_in_source"] = int(a.obs_names.duplicated().sum())

    a.obs["source_obs_name"] = a.obs_names.astype(str)
    a.obs_names = (
        a.obs["brain"].astype(str)
        + "__"
        + a.obs["slice_name"].astype(str)
        + "__"
        + a.obs["cell_id"].astype(str)
    ).to_numpy()
    if a.obs_names.duplicated().any():
        raise SystemExit("brain__slice_name__cell_id is not unique")

    drop = [c for c in a.obs.columns if c.endswith(paths.SUPERSEDED_SUFFIX)]
    stale = [c for c in a.obs.columns if c.startswith("in_")]
    report["dropped_obs_columns"] = drop + stale
    a.obs.drop(columns=drop + stale, inplace=True)
    report["region_columns_added"] = attach_region_flags(a, strict=True)
    report["region_counts"] = {
        c: int(a.obs[c].sum(skipna=True)) for c in report["region_columns_added"]
    }
    report["region_nulls"] = {
        c: int(a.obs[c].isna().sum()) for c in report["region_columns_added"]
    }

    hierarchical_vote(
        a, cluster_col="leiden", label_col="tangram_cell_type", dst="cell_type"
    )
    report["cell_type_counts"] = a.obs["cell_type"].value_counts().to_dict()

    if not np.array_equal(
        np.asarray(a.obsm["spatial"]), np.asarray(a.obsm["spatial_aligned"])
    ):
        raise SystemExit(
            "obsm['spatial'] differs from obsm['spatial_aligned']; inspect before "
            "dropping"
        )
    del a.obsm["spatial"]
    raw = raw_um_coords()
    idx = pd.Index(key(raw)).get_indexer(key(a.obs))
    report["cells_without_raw_coords"] = int((idx < 0).sum())
    if (idx < 0).any():
        raise SystemExit(
            f"{(idx < 0).sum():,} atlas cells have no raw per-crop coordinates"
        )
    a.obsm["spatial_um"] = raw[["x_um", "y_um"]].to_numpy()[idx]

    # Expression: keep the stored X (log1p-normalised) and, if .raw holds counts,
    # as layers['counts'].
    if a.raw is not None:
        rx = a.raw.X
        sample = rx[:2000].toarray() if sp.issparse(rx) else np.asarray(rx[:2000])
        if np.allclose(sample, np.round(sample)) and a.raw.var_names.equals(
            a.var_names
        ):
            a.layers["counts"] = sp.csr_matrix(rx)
            report["counts_layer"] = "from .raw (integer-valued, same genes)"
        else:
            report["counts_layer"] = (
                "not added: .raw is not integer counts over the same genes"
            )
        a.raw = None
    for k in list(a.obsp.keys()):
        del a.obsp[k]
    for k in ("neighbors",):
        a.uns.pop(k, None)
    a.uns["deposit_provenance"] = {
        "source": str(paths.CANONICAL_TANGRAM),
        "region_flags": (
            "results/nine_region (annotations_v4), via "
            "nine_region_common.attach_region_flags"
        ),
        "cell_type": (
            "cluster_labels.hierarchical_vote(cluster_col='leiden', "
            "label_col='tangram_cell_type')"
        ),
        "spatial_um": (
            "raw per-crop obsm['spatial'] (section microns), joined on (slice_name, "
            "cell_id)"
        ),
        "spatial_aligned": "PASTE3 registration frame; NOT slide microns",
    }
    out = out_dir / "bimp_atlas_xenium_public.partial.h5ad"
    a.write_h5ad(out, compression="gzip")
    return out


def export_labels(out_dir: Path, report: dict) -> Path:
    """Per-cell label table across runs, written before the big h5ad.

    Rows follow the canonical object. The central / optic runs map a subset of
    cells (1,185,039 and 115,153 of 1,300,192), so cells a run did not map get NA
    and the report records each run's coverage. Duplicate keys within a run abort.
    """
    import anndata as ad

    canon = ad.read_h5ad(paths.CANONICAL_TANGRAM, backed="r")
    ref = (
        canon.obs[["brain", "slice_name", "cell_id"]].astype(str).reset_index(drop=True)
    )
    canon.file.close()
    tab = ref.copy()
    tab.insert(
        0,
        "obs_name",
        (ref["brain"] + "__" + ref["slice_name"] + "__" + ref["cell_id"]).to_numpy(),
    )
    if tab["obs_name"].duplicated().any():
        raise SystemExit("brain__slice_name__cell_id is not unique")
    srcs = {
        "s42": paths.CANONICAL_TANGRAM,
        **{
            r.rsplit("_", 1)[-1]: paths.TANGRAM_DIR / f"{r}_tangram.h5ad"
            for r in SEED_RUNS
        },
        "central": paths.CANONICAL_CENTRAL,
        "optic": paths.CANONICAL_OPTIC,
        "stitched": paths.CANONICAL_STITCHED,
    }
    for tag, p in srcs.items():
        b = ad.read_h5ad(p, backed="r")
        o = b.obs
        same = (
            len(o) == len(ref)
            and (
                o["slice_name"].astype(str).to_numpy() == ref["slice_name"].to_numpy()
            ).all()
            and (o["cell_id"].astype(str).to_numpy() == ref["cell_id"].to_numpy()).all()
        )
        if same:
            ix = np.arange(len(o))
        else:  # keyed join; a subset run leaves -1 for the cells it did not map
            okey = pd.Index(key(o))
            if okey.duplicated().any():
                raise SystemExit(f"{p.name}: duplicate (slice_name, cell_id) keys")
            ix = okey.get_indexer(key(ref))
        for c in LABEL_COLS:
            if c in o.columns:
                tab[f"{c}__{tag}"] = pd.Series(o[c].to_numpy()).reindex(ix).to_numpy()
        report.setdefault("label_sources", {})[tag] = {
            "path": str(p),
            "row_aligned": bool(same),
            "n_run": int(len(o)),
            "n_covered": int((ix >= 0).sum()),
        }
        b.file.close()
    out = out_dir / "tangram_labels_public.partial.parquet"
    tab.to_parquet(out, index=False)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--run", action="store_true", help="write the exports (default: list only)"
    )
    ap.add_argument("--staging", type=Path, default=STAGING)
    args = ap.parse_args()
    out_dir = args.staging / "zenodo/Z1_atlas/public"

    ins = inputs()
    print("Inputs (read-only):")
    for grp, fs in ins.items():
        sz = sum(f.stat().st_size for f in fs if f.exists()) / 1e9
        miss = [f for f in fs if not f.exists()]
        print(
            f"  {grp:16s} {len(fs):3d} files  {sz:7.1f} GB"
            + (f"  MISSING: {miss}" if miss else "")
        )
    print(f"Outputs -> {out_dir}:")
    for n in (
        "bimp_atlas_xenium_public.h5ad (~2-3 GB gzip)",
        "tangram_labels_public.parquet (~0.2 GB)",
        "EXPORT_REPORT.json",
    ):
        print(f"  {n}")
    if not args.run:
        print("Dry run. Pass --run (inside a Slurm job) to write.")
        return
    if any(not f.exists() for fs in ins.values() for f in fs):
        raise SystemExit("missing inputs; see list above")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in ("bimp_atlas_xenium_public.h5ad", "tangram_labels_public.parquet"):
        if (out_dir / f).exists():
            raise SystemExit(
                f"{out_dir / f} exists; refusing to overwrite (remove it by hand if "
                "intended)"
            )
    report: dict = {}
    # Labels first (cheap, and they validate every run's row alignment), then the
    # atlas. Both are written as *.partial.* and renamed only when everything passed,
    # so a failed run never leaves a final-named file that blocks a retry.
    parts = [export_labels(out_dir, report), export_atlas(out_dir, report)]
    finals = []
    for part in parts:
        final = part.with_name(part.name.replace(".partial", ""))
        part.rename(final)
        finals.append(str(final))
    report["outputs"] = finals
    (out_dir / "EXPORT_REPORT.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    print(json.dumps({k: report[k] for k in ("n_cells", "outputs")}, indent=2))


if __name__ == "__main__":
    main()
