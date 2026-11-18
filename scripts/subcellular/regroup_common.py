"""Shared loaders for the by-section and Kenyon-compartment bento analyses.

Join logic (the only place it lives):
  * bento lp.parquet rows are keyed by (region, cell_boundaries). ``region`` is the
    capture-region folder, which holds 2-4 sections. A cell is mapped to its
    section by requiring the atlas ``slice_name`` to start with that folder name
    (the rule bento_save_lp.py uses). Barcodes that resolve to more than one
    section inside the same capture region are dropped and counted.
  * Compartments are the nine-region atlas flags, precedence Small-type (in_kc_ic)
    over Large-type (in_kc_nc) over Class II (in_kc_oc), as atlas_cells() in
    scripts/kenyon/fig_kc_subtypes.py.
  * Distance to the Small-type core is computed as in the Kenyon-cell figure's
    _subzone_data():
    Large-type cell to nearest Small-type cell in the same section, in the raw
    per-crop coordinates (joined on (slice_name, cell_id), never positionally).
"""

import glob
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
# src/ on the path even without PYTHONPATH (ella_section.py runs in ELLA's own
# env); nine_region_common needs only the standard library at import
sys.path.insert(0, str(REPO / "src"))
from bimp_st.nine_region_common import region_mask, region_of, traced_mask  # noqa: E402

PATTERNS = ["cell_edge", "cytoplasmic", "none", "nuclear", "nuclear_edge"]
# Per-row counts summed next to the pattern flags. bento's LP flags are
# multi-label 0/1 and most rows carry none of them: ``n_calls`` counts every
# cell-gene row (also the read-depth covariate), ``n_classified`` the rows with
# at least one flag, ``none_only`` the rows flagged "none" and nothing else.
COUNTS = PATTERNS + ["n_calls", "n_classified", "none_only"]
CONTROLS = ("NegControl", "Unassigned", "Deprecated")
NB_DIR = {"queen": "wholebrain/queen", "worker": "wholebrain/worker"}
COMP_NAME = {"KC-IC": "Small-type", "KC-NC": "Large-type", "KC-OC": "Class II"}
COMP_ORDER = ["KC-NC", "KC-IC", "KC-OC"]
DIST_BINS = [0, 30, 60, 120, np.inf]
DIST_LABELS = ["0-30", "30-60", "60-120", ">120"]


def atlas_obs(brain):
    """Cell rows of the nine-region atlas: brain, cell_id, slice_name, comp."""
    path = REPO / f"results/nine_region/atlas/{brain}_aligned_clustered_9region.h5ad"
    a = ad.read_h5ad(path, backed="r")
    o = a.obs[["cell_id", "slice_name", "in_kc_ic", "in_kc_nc", "in_kc_oc"]].copy()
    a.file.close()
    # every section is traced for the Kenyon compartments (traced_mask raises
    # otherwise), so "not in KC-IC" below is a traced absence
    ic, nc, oc = (traced_mask(region_mask(o, [r])) for r in ("kc_ic", "kc_nc", "kc_oc"))
    out = pd.DataFrame(
        {
            "brain": brain,
            "cell_id": o["cell_id"].astype(str).to_numpy(),
            "slice_name": o["slice_name"].astype(str).to_numpy(),
            "comp": np.select(
                [ic, nc & ~ic, oc & ~ic & ~nc], ["KC-IC", "KC-NC", "KC-OC"], ""
            ),
        }
    )
    return out


def core_distance(obs):
    """Distance (um, raw per-crop frame) of each Large-type cell to the nearest
    Small-type cell of the same section. Returns cell rows with a ``dist`` column."""
    obs = obs.copy()
    dist = np.full(len(obs), np.nan)
    groups = obs.groupby(["brain", "slice_name"], sort=False).indices
    for (brain, sl), idx in groups.items():
        g = obs.iloc[idx]
        if not ((g.comp == "KC-IC").any() and (g.comp == "KC-NC").any()):
            continue
        crop = REPO / f"pipelines/{NB_DIR[brain]}/data/{sl}.h5ad"
        c = ad.read_h5ad(crop, backed="r")
        cid = c.obs["cell_id"].astype(str).to_numpy()
        xy_raw = np.asarray(c.obsm["spatial"], dtype=float)
        c.file.close()
        pos = pd.Series(np.arange(len(cid)), index=cid)
        pos = pos[~pos.index.duplicated()]
        found = g.cell_id.isin(pos.index).to_numpy()
        xy = np.full((len(g), 2), np.nan)
        xy[found] = xy_raw[pos[g.cell_id[found]].to_numpy()]
        ic = (g.comp == "KC-IC").to_numpy() & np.isfinite(xy[:, 0])
        nc = (g.comp == "KC-NC").to_numpy() & np.isfinite(xy[:, 0])
        d = np.full(len(g), np.nan)
        if ic.any() and nc.any():
            d[nc] = cKDTree(xy[ic]).query(xy[nc])[0]
        dist[idx] = d
    obs["dist"] = dist
    return obs


def region_folders(brain):
    """Capture-region folder names bento produced for this brain."""
    return sorted(
        Path(p).parent.name
        for p in glob.glob(str(REPO / f"results/bento/{brain}/output-XETG*/lp.parquet"))
    )


def cell_map(brain):
    """One row per cell with its section, region folder, compartment and distance.
    Barcodes that map to more than one section of the same capture region are
    dropped (counted in the return value)."""
    obs = core_distance(atlas_obs(brain))
    regions = region_folders(brain)
    obs["region"] = ""
    for r in regions:
        m = obs.slice_name.str.startswith(r)
        obs.loc[m, "region"] = r
    obs = obs[obs.region != ""]
    amb = obs.duplicated(["region", "cell_id"], keep=False)
    n_amb = int(obs.loc[amb, ["region", "cell_id"]].drop_duplicates().shape[0])
    return obs[~amb].reset_index(drop=True), n_amb


def aggregate_lp(brain, cells):
    """Stream the region parquets and return three small aggregate tables.

    section : (slice_name, comp) sums over every LP row (all cells)
    cell    : (slice_name, cell_id, comp, dist) sums per cell
    gene    : (slice_name, comp, feature_name) sums, Kenyon compartments only
    Every table carries ``n_calls`` (LP rows), ``n_classified`` (rows with any
    pattern flag) and ``none_only`` next to the pattern sums (see COUNTS).
    """
    import pyarrow.parquet as pq

    sec_p, cell_p, gene_p = [], [], []
    for f in sorted(
        glob.glob(str(REPO / f"results/bento/{brain}/output-XETG*/lp.parquet"))
    ):
        region = Path(f).parent.name
        cm = cells[cells.region == region]
        if not len(cm):
            continue
        t = pq.read_table(f, columns=["cell_boundaries", "feature_name", *PATTERNS])
        d = t.to_pandas()
        d["feature_name"] = d["feature_name"].astype(str)
        d = d[~d["feature_name"].str.startswith(CONTROLS)]
        d = d.rename(columns={"cell_boundaries": "cell_id"})
        d["cell_id"] = d["cell_id"].astype(str)
        d = d.merge(
            cm[["cell_id", "slice_name", "comp", "dist"]], on="cell_id", how="inner"
        )
        flags = d[PATTERNS].astype(int)
        other = flags.drop(columns="none").sum(axis=1) > 0
        d["n_calls"] = 1
        d["n_classified"] = (other | (flags["none"] > 0)).astype(int)
        d["none_only"] = ((flags["none"] > 0) & ~other).astype(int)
        sec_p.append(d.groupby(["slice_name", "comp"])[COUNTS].sum())
        cell_p.append(
            d.groupby(["slice_name", "cell_id", "comp", "dist"], dropna=False)[
                COUNTS
            ].sum()
        )
        k = d[d["comp"] != ""]
        gene_p.append(k.groupby(["slice_name", "comp", "feature_name"])[COUNTS].sum())
    sec = pd.concat(sec_p).groupby(level=[0, 1]).sum().reset_index()
    cell = (
        pd.concat(cell_p).groupby(level=[0, 1, 2, 3], dropna=False).sum().reset_index()
    )
    gene = pd.concat(gene_p).groupby(level=[0, 1, 2]).sum().reset_index()
    for df in (sec, cell, gene):
        df.insert(0, "brain", brain)
    return sec, cell, gene


def fractions(df):
    """Localization-class fractions among CLASSIFIED calls, plus the unclassified share.

    bento flags each cell-gene row with zero or more of the five patterns; about
    60% of rows carry no flag at all. The denominator is ``n_classified`` (rows
    with at least one flag), not ``n_calls`` (all rows). A row flagged with a
    pattern and with "none" counts for the pattern: ``frac_none`` is
    ``none_only / n_classified``. The five fractions then sum to one, up to the
    rare rows flagged with two non-"none" patterns. ``frac_unclassified``
    is the share of all rows with no flag, reported separately.
    """
    out = df.copy()
    tot = out["n_classified"].replace(0, np.nan)
    for p in PATTERNS:
        num = out["none_only"] if p == "none" else out[p]
        out[f"frac_{p}"] = num / tot
    out["frac_unclassified"] = 1 - out["n_classified"] / out["n_calls"].replace(
        0, np.nan
    )
    return out


OPTIC_REGIONS = ["lamina", "medulla", "lobula", "al"]
REGION_NAME = {
    "lamina": "Lamina", "medulla": "Medulla", "lobula": "Lobula", "al": "Antennal lobe",
}


def region_cell_map(brain):
    """Like cell_map, but comp is the single anatomical region a cell is in
    (KC compartments and optic/antennal lobe regions; cells in >1 region get "")."""
    flags = ["in_kc_ic", "in_kc_nc", "in_kc_oc"] + [f"in_{r}" for r in OPTIC_REGIONS]
    a = ad.read_h5ad(
        REPO / f"results/nine_region/atlas/{brain}_aligned_clustered_9region.h5ad",
        backed="r",
    )
    o = a.obs[["cell_id", "slice_name", *flags]].copy()
    a.file.close()
    regions = ["kc_ic", "kc_nc", "kc_oc", *OPTIC_REGIONS]
    names = dict(zip(regions, ["KC-IC", "KC-NC", "KC-OC", *OPTIC_REGIONS], strict=True))
    comp = np.array([names.get(r, "") for r in region_of(o, regions)])
    obs = pd.DataFrame(
        {
            "brain": brain,
            "cell_id": o.cell_id.astype(str).to_numpy(),
            "slice_name": o.slice_name.astype(str).to_numpy(),
            "comp": comp,
            "dist": np.nan,
        }
    )
    obs["region"] = ""
    for r in region_folders(brain):
        obs.loc[obs.slice_name.str.startswith(r), "region"] = r
    obs = obs[obs.region != ""]
    amb = obs.duplicated(["region", "cell_id"], keep=False)
    n_amb = int(obs.loc[amb, ["region", "cell_id"]].drop_duplicates().shape[0])
    return obs[~amb].reset_index(drop=True), n_amb
