"""ELLA Newton per section on Kenyon-compartment cells.

  manifest           : write results/ella/manifest.csv (brain, slice_name, region)
  run <task_index>   : build ELLA input for one section and run the pipeline

Inputs are raw Xenium transcripts / cell and nucleus boundaries of the capture
region (same micron frame as each other). Cells are sampled per compartment
(CAP_PER_COMP, seed 0). Genes kept per compartment: >= MIN_TX panel transcripts.
ELLA Newton is CPU-only (no torch); runs on the main partition.
"""

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regroup_common import (  # noqa: E402
    CONTROLS,
    DIST_BINS,
    DIST_LABELS,
    NB_DIR,
    OPTIC_REGIONS,
    REGION_NAME,
    REPO,
    cell_map,
    region_cell_map,
)

RAW = REPO / "data"
import os

MODE = os.environ.get("ELLA_MODE", "kc")  # "kc" (default) or "regions"
OUT = REPO / {"regions": "results/ella_regions", "subs": "results/ella_subclusters"}.get(MODE, "results/ella")
CAP_PER_COMP = 150
MIN_TX = 20
QV_MIN = 20.0
COMP_TYPE = {"KC-IC": "Small-type", "KC-NC": "Large-type", "KC-OC": "Class II"}
# Large-type cells are also fitted per distance-from-Small-type-core bin (as in
# bento_regroup and `fig:kc_subtypes`), so ELLA patterns can be read along the
# gradient.
COMP_TYPE.update({f"LT {b} um": f"Large-type {b} um" for b in DIST_LABELS})
if MODE == "regions":
    COMP_TYPE = dict(REGION_NAME)
if MODE == "subs":  # Large-type cells by projected snRNA subcluster x near/far from the Small-type core
    COMP_TYPE = {f"Sub{s} {z}": f"Sub{s} {z}" for s in "012" for z in ("near", "far")}


def get_cells(brain):
    if MODE == "subs":
        cells, n = cell_map(brain)
        calls = pd.read_parquet(
            REPO / "results/bento/by_section/ltkc_subcluster_calls.parquet",
            columns=["slice_name", "cell_id", "sub"],
        ).drop_duplicates(["slice_name", "cell_id"])
        cells = cells.merge(calls, on=["slice_name", "cell_id"], how="left")
        keep = (cells.comp == "KC-NC") & cells["dist"].notna() & cells["sub"].notna()
        cells = cells[keep].copy()
        cells["comp"] = "Sub" + cells["sub"].astype(str) + np.where(cells["dist"] < 60, " near", " far")
        return cells.reset_index(drop=True), n
    if MODE == "regions":
        cells, n = region_cell_map(brain)
        return cells[cells.comp.isin(OPTIC_REGIONS)].reset_index(drop=True), n
    return cell_map(brain)


def region_dir(region):
    """Raw Xenium folder for a capture-region folder name."""
    run = (
        "20251204__194847___20251204"
        if "__0040" in region
        else "20260115__193720__20260115"
    )
    return RAW / run / region


def manifest():
    rows = []
    for brain in ("queen", "worker"):
        cells, _ = get_cells(brain)
        kc = cells[cells.comp != ""]
        for (sl, r), _g in kc.groupby(["slice_name", "region"]):
            rows.append({"brain": brain, "slice_name": sl, "region": r})
    m = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    m.to_csv(OUT / "manifest.csv", index=False)
    print(len(m), "sections", flush=True)


def build_inputs(brain, sl, region):
    import pyarrow.parquet as pq

    cells, _ = get_cells(brain)
    sec = cells[(cells.slice_name == sl)]
    sec = sec[sec.comp != ""]
    rng = np.random.default_rng(0)
    pick = {}
    lt = sec[(sec.comp == "KC-NC") & sec.dist.notna()]
    lt_bin = pd.cut(lt.dist, DIST_BINS, labels=DIST_LABELS, right=False)
    for comp in COMP_TYPE:
        if comp.startswith("LT "):
            b = comp[3:-3]
            ids = lt.loc[(lt_bin == b).to_numpy(), "cell_id"].unique()
        else:
            ids = sec.loc[sec.comp == comp, "cell_id"].unique()
        if len(ids):
            pick[comp] = rng.choice(
                ids, size=min(CAP_PER_COMP, len(ids)), replace=False
            )
    keep_ids = np.array(sorted({c for ids in pick.values() for c in ids}))
    rd = region_dir(region)
    tx = pq.read_table(
        rd / "transcripts.parquet",
        columns=[
            "cell_id",
            "feature_name",
            "x_location",
            "y_location",
            "qv",
            "is_gene",
        ],
        filters=[("cell_id", "in", list(keep_ids))],
    ).to_pandas()
    tx["feature_name"] = tx["feature_name"].astype(str)
    tx = tx[(tx.qv >= QV_MIN) & tx.is_gene & ~tx.feature_name.str.startswith(CONTROLS)]
    tx = tx.rename(columns={"x_location": "x", "y_location": "y"})
    tx["cell_id"] = tx["cell_id"].astype(str)
    # polygons: cell boundaries and nucleus centroids, same capture frame
    cb = pq.read_table(
        rd / "cell_boundaries.parquet", filters=[("cell_id", "in", list(keep_ids))]
    ).to_pandas()
    cb["cell_id"] = cb["cell_id"].astype(str)
    nb = pq.read_table(
        rd / "nucleus_boundaries.parquet", filters=[("cell_id", "in", list(keep_ids))]
    ).to_pandas()
    nb["cell_id"] = nb["cell_id"].astype(str)
    ncen = nb.groupby("cell_id")[["vertex_x", "vertex_y"]].mean()
    poly = {
        k: g[["vertex_x", "vertex_y"]].to_numpy()
        for k, g in cb.groupby("cell_id")
        if len(g) >= 3
    }
    # frame check against the crop: section coordinates vs polygon centroids
    import anndata as ad

    crop = ad.read_h5ad(REPO / f"pipelines/{NB_DIR[brain]}/data/{sl}.h5ad", backed="r")
    cid = crop.obs["cell_id"].astype(str).to_numpy()
    xy = np.asarray(crop.obsm["spatial"], float)
    crop.file.close()
    pos = pd.Series(np.arange(len(cid)), index=cid)
    pos = pos[~pos.index.duplicated()]
    common = [c for c in poly if c in pos.index][:200]
    cen = np.array([poly[c].mean(0) for c in common])
    dif = np.linalg.norm(cen - xy[pos[common].to_numpy()], axis=1)
    frame_med = float(np.median(dif))
    if frame_med > 30:
        raise RuntimeError(
            f"frame mismatch: median crop-vs-polygon distance {frame_med:.1f} um"
        )
    expr_parts, cells_d, genes_d, types = [], {}, {}, []
    for comp, ids in pick.items():
        t = COMP_TYPE[comp]
        ids = [c for c in ids if c in poly]
        if not ids:
            continue
        s = tx[tx.cell_id.isin(ids)].copy()
        tot = s.groupby("cell_id").size().rename("sc_total")
        s = s.join(tot, on="cell_id")
        counts = s.groupby("feature_name").size()
        genes = sorted(counts[counts >= MIN_TX].index)
        s = s[s.feature_name.isin(genes)]
        s["type"] = t
        s["cell"] = s.cell_id
        s["gene"] = s.feature_name
        s["umi"] = 1
        ce = ncen.reindex(s.cell_id).to_numpy()
        s["centerX"], s["centerY"] = ce[:, 0], ce[:, 1]
        s = s.dropna(subset=["centerX"])
        if len(s) == 0 or not genes:
            continue
        expr_parts.append(
            s[
                [
                    "x",
                    "y",
                    "umi",
                    "centerX",
                    "centerY",
                    "type",
                    "cell",
                    "gene",
                    "sc_total",
                ]
            ]
        )
        cells_d[t] = [c for c in ids if c in set(s.cell_id)]
        genes_d[t] = genes
        types.append(t)
    expr = pd.concat(expr_parts, ignore_index=True)
    cells_all = sorted({c for v in cells_d.values() for c in v})
    data = dict(
        expr=expr,
        cell_poly={c: poly[c] for c in cells_all},
        types=types,
        cells={t: cells_d[t] for t in types},
        genes={t: genes_d[t] for t in types},
        cells_all=cells_all,
    )
    return data, frame_med


def check_parity(data):
    import pickle

    ref = Path(
        os.environ.get("ELLA_SRC", "ELLA_src") + "/tutorials/mini_demo/input/mini_demo_data.pkl"
    )
    demo = pickle.load(open(ref, "rb"))
    # ELLA v1 consumes polygons; the demo's raster cell_seg/nucleus_seg are unused
    missing = set(demo) - {"cell_seg", "nucleus_seg"} - set(data)
    assert not missing, f"input lacks keys {missing}"
    miss_cols = set(demo["expr"].columns) - set(data["expr"].columns)
    assert not miss_cols, f"expr lacks columns {miss_cols}"


def run(task):
    m = pd.read_csv(OUT / "manifest.csv")
    if task >= len(m):
        print('no such task', task, flush=True)
        return
    row = m.iloc[task]
    brain, sl, region = row.brain, row.slice_name, row.region
    d = OUT / brain / sl
    d.mkdir(parents=True, exist_ok=True)
    try:
        data, frame_med = build_inputs(brain, sl, region)
        check_parity(data)
        import pickle

        pickle.dump(data, open(d / "ella_input.pkl", "wb"))
        import os

        from ELLA import ELLA

        os.chdir(d)
        e = ELLA(dataset=sl)
        e.load_data(data_dict=data)
        e.register_cells()
        e.nhpp_prepare()
        e.nhpp_fit()
        e.weighted_density_est()
        e.compute_pv()
        rows = []
        for t in data["types"]:
            pv = e.pv_fdr_tl[t]
            sc = e.scores[t]
            for i, g in enumerate(e.gene_list_dict[t]):
                rows.append(
                    {
                        "brain": brain,
                        "slice_name": sl,
                        "type": t,
                        "feature_name": g,
                        "pv_fdr": float(pv[i]),
                        "score": float(sc[i]),
                        "sig": float(pv[i]) <= 0.05,
                        "n_cells": len(data["cells"][t]),
                        "frame_median_um": frame_med,
                        "cap_per_comp": 150,
                    }
                )
        pd.DataFrame(rows).to_csv(d / "ella_results.csv", index=False)
        print("done", brain, sl, flush=True)
    except Exception:
        (d / "FAILED.txt").write_text(traceback.format_exc())
        raise


if __name__ == "__main__":
    if sys.argv[1] == "manifest":
        manifest()
    else:
        run(int(sys.argv[2]))
