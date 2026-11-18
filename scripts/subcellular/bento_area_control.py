"""Segmentation-geometry control for the region comparison.

'Nuclear' transcripts depend on how much of the cell outline the nucleus covers. Compute
per-cell cell area, nucleus area and their ratio from the raw Xenium boundaries, then
(1) report area ratio by region and (2) compare regions WITHIN bins of area ratio. Nuclear fractions are among classified
calls (n_classified, rows with any bento pattern flag).

Inputs : raw cell_boundaries / nucleus_boundaries parquets (capture-region folders),
         results/bento/by_section/region_cells_{queen,worker}.parquet (per-cell bento calls)
Outputs: results/bento/regions_summary/area_control_*.csv, area_control.png
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regroup_common import REPO, region_folders  # noqa: E402

from bimp_st.si_common import QUEEN_COLOR, WORKER_COLOR, set_style  # noqa: E402

R = REPO / "results/bento"
OUT = R / "regions_summary"
KC = ["KC-IC", "KC-NC", "KC-OC"]
OPT = ["lamina", "medulla", "lobula", "al"]
CAST = {"queen": QUEEN_COLOR, "worker": WORKER_COLOR}
lines = []


def log(s=""):
    print(s, flush=True)
    lines.append(s)


def region_dir(folder):
    run = "20251204__194847___20251204" if "__0040" in folder else "20260115__193720__20260115"
    return REPO / "data" / run / folder


def poly_area(df):
    """Shoelace area per cell_id from vertex_x / vertex_y (vertices in order)."""
    df = df.sort_values("cell_id", kind="stable")
    cid = df["cell_id"].to_numpy()
    x, y = df["vertex_x"].to_numpy(float), df["vertex_y"].to_numpy(float)
    start = np.r_[0, np.flatnonzero(cid[1:] != cid[:-1]) + 1]
    end = np.r_[start[1:], len(cid)]
    xn = np.roll(x, -1)
    yn = np.roll(y, -1)
    # close each polygon: last vertex wraps to the first of its own cell
    xn[end - 1] = x[start]
    yn[end - 1] = y[start]
    cross = x * yn - xn * y
    area = 0.5 * np.abs(np.add.reduceat(cross, start))
    return pd.Series(area, index=cid[start])


rows = []
for brain in CAST:
    cells = pd.read_parquet(R / f"by_section/region_cells_{brain}.parquet")
    cells = cells[cells.comp.isin(KC + OPT)].copy()
    folders = region_folders(brain)
    cells["folder"] = ""
    for f in folders:
        cells.loc[cells.slice_name.str.startswith(f), "folder"] = f
    cells = cells[cells.folder != ""]
    for f, c in cells.groupby("folder"):
        d = region_dir(f)
        ids = c.cell_id.astype(str).unique().tolist()
        cb = pq.read_table(d / "cell_boundaries.parquet", filters=[("cell_id", "in", ids)]).to_pandas()
        nb = pq.read_table(d / "nucleus_boundaries.parquet", filters=[("cell_id", "in", ids)]).to_pandas()
        for t in (cb, nb):
            t["cell_id"] = t["cell_id"].astype(str)
        ca, na = poly_area(cb), poly_area(nb)
        m = c.assign(cell_id=c.cell_id.astype(str)).drop_duplicates(["slice_name", "cell_id"])
        m["cell_area"] = m.cell_id.map(ca)
        m["nuc_area"] = m.cell_id.map(na)
        rows.append(m[["brain", "slice_name", "cell_id", "comp", "nuclear", "n_calls", "n_classified",
                       "cell_area", "nuc_area"]])
        print("done", f, len(m), flush=True)
A = pd.concat(rows, ignore_index=True)
A = A.dropna(subset=["cell_area", "nuc_area"])
A = A[(A.cell_area > 0) & (A.nuc_area > 0)]
A["ratio"] = (A.nuc_area / A.cell_area).clip(upper=1.0)
A.to_parquet(OUT / "area_control_cells.parquet")

log("== Nucleus / cell area by region (median across cells)")
t = A.groupby(["brain", "comp"]).agg(cells=("ratio", "size"), cell_area_um2=("cell_area", "median"),
                                     nuc_area_um2=("nuc_area", "median"), area_ratio=("ratio", "median"),
                                     pooled_nuclear=("nuclear", "sum"), calls=("n_classified", "sum"))
t["nuclear_fraction"] = t.pooled_nuclear / t.calls
t = t.drop(columns=["pooled_nuclear", "calls"])
t.to_csv(OUT / "area_control_by_region.csv")
log(t.round(3).to_string())

# nuclear fraction relative to the area null (frac nuclear transcripts / nucleus area share)
t["enrichment_vs_area"] = t.nuclear_fraction / t.area_ratio
log("\nnuclear fraction divided by median nucleus/cell area ratio (1 = uniform density):")
log(t[["area_ratio", "nuclear_fraction", "enrichment_vs_area"]].round(3).to_string())

# matched area-ratio bins: optic vs KC
EDGES = [0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 1.01]
A["bin"] = pd.cut(A.ratio, EDGES, right=False)
log("\n== Optic+AL vs KC nuclear fraction within nucleus/cell area-ratio bins (>=200 cells each side)")
con = []
for brain, g in A.groupby("brain"):
    for b, d in g.groupby("bin", observed=True):
        k, o = d[d.comp.isin(KC)], d[d.comp.isin(OPT)]
        if len(k) >= 200 and len(o) >= 200:
            kn, on = k.nuclear.sum() / k.n_classified.sum(), o.nuclear.sum() / o.n_classified.sum()
            con.append(dict(brain=brain, area_bin=str(b), kc_cells=len(k), optic_cells=len(o),
                            kc_nuclear=kn, optic_nuclear=on, ratio=on / kn))
con = pd.DataFrame(con)
con.to_csv(OUT / "area_control_matched.csv", index=False)
log(con.round(3).to_string(index=False))
if len(con):
    log(f"\nratio range {con.ratio.min():.2f}-{con.ratio.max():.2f}; optic > KC in {(con.ratio > 1).sum()}/{len(con)} bins")
(OUT / "area_control_summary.txt").write_text("\n".join(lines))

set_style()
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for brain in CAST:
    g = A[A.brain == brain]
    for comp in KC + OPT:
        v = g[g.comp == comp].ratio
        ax[0].hist(v, bins=40, range=(0, 1), histtype="step", density=True,
                   lw=1.2, ls="-" if brain == "queen" else "--", label=f"{comp}" if brain == "queen" else None)
ax[0].set_xlabel("nucleus area / cell area"); ax[0].set_ylabel("density"); ax[0].legend(frameon=False, fontsize=7)
for brain in CAST:
    c = con[con.brain == brain]
    ax[1].plot(range(len(c)), c.ratio, "-o", color=CAST[brain], label=brain)
    ax[1].set_xticks(range(len(c))); ax[1].set_xticklabels(c.area_bin, rotation=40, fontsize=6)
ax[1].axhline(1, color="grey", lw=0.6)
ax[1].set_xlabel("nucleus/cell area-ratio bin"); ax[1].set_ylabel("optic+AL / KC nuclear fraction")
ax[1].legend(frameon=False)
for a in ax:
    a.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "area_control.png", dpi=300)
