"""Spatial breadth of cell types: how globally distributed is each unit?

Shows how broadly spread or spatially distinct each cell type is
across the whole brain. Breadth needs no region tracing, so every atlas
cell is used, including the cells that lie outside the eight traced
regions (which no regional analysis touches).

Units. Per-cell Tangram labels are too noisy to profile (about a third of calls
are confident), so every measure is taken on
  (1) the 32 Leiden spatial clusters of the canonical combined object, and
  (2) the vote-assigned groups: `hierarchical_vote` over the canonical
      Tangram run, exactly as fig:atlas_overview does (expression only, no
      anatomy).
Both are cluster-level; neither says anything about single cells.

Measures (per unit, per scope = pooled / queen / worker)
  (a) distribution over the 65 sections (`slice_name` crops): normalized Shannon
      entropy and Gini of the unit's counts, and the same on the size-normalized
      fraction-of-section profile (separates "present everywhere but rare" from
      "concentrated"), number of sections holding >=1% of the unit, number of
      sections where the unit is >=1% of the section.
  (b) anterior-posterior (stack) extent, per brain only (stacks are not
      co-registered): fractional z position, span of the central 90% of the
      fraction-weighted profile, centre of mass, first/last occupied section.
  (c) distribution over the eight traced regions plus 'outside traced regions'.
      Region view is restricted to sections where the region was traced (nulls in
      the nine-region flags mean untraced, never False).
  (d) within-section clustering from raw per-crop coordinates: mean
      nearest-neighbour distance among the unit's cells divided by the same
      statistic for random subsets of the section's own cells (same size), per
      section, summarized by the median over sections with >= NMIN cells.
  (e) composite distinctness = mean percentile rank of three tracing-free
      components (section concentration, AP span, local clustering), reported
      alongside the components. The region view is descriptive and is not in it.
Control. Every section/AP measure is shown against a size-matched null: random
groupings of the same number of cells drawn without replacement from the scope
(multivariate hypergeometric across sections, so the per-section totals are
those of the real atlas). x-chance is not scale-free; breadth is judged against
this null, not against an absolute scale.

Descriptive only. One brain per caste (n = 1); no p-values; the queen and worker
stacks are not co-registered.

Run (from a worktree whose results/ and pipelines/*/data hold symlinks to the
inputs):
  PYTHONPATH=src:scripts .venv/bin/python \
      scripts/regions/celltype_spatial_breadth.py [compute|figure]
Outputs only to results/handoff/G_spatial_breadth/.
"""

from __future__ import annotations

import json
import os
import sys
import zlib
from multiprocessing import Pool
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- config
MAIN = Path(os.environ.get("BIMP_MAIN", Path(__file__).resolve().parents[2]))
OUT = Path(os.environ.get("BIMP_SPB_OUT", MAIN / "results/handoff/G_spatial_breadth"))
N_NULL = 500  # size-matched null draws per unit and scope
NN_NULL = 10  # label-permutation draws per unit and section
NMIN_SECTION = 50  # min unit cells in a section for the NN statistic
NMIN_SCOPE = 200  # min unit cells in a scope to report it
PRESENT_FRAC = 0.01  # "present" threshold
N_PROC = int(os.environ.get("N_PROC", 12))
SEED = 20261008
SCOPES = ("pooled", "queen", "worker")
RAW_DIRS = {
    "queen": MAIN / "pipelines/wholebrain/queen/data",
    "worker": MAIN / "pipelines/wholebrain/worker/data",
}


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------- loading
def load_cells():
    """obs of the canonical Tangram run + region flags, joined on
    (slice_name, cell_id). The canonical object's own leiden column is the
    canonical clustering; labels come from the same object (corrected run)."""
    from anndata import AnnData

    from bimp_st import nine_region_common as nr
    from bimp_st.cluster_labels import hierarchical_vote
    from bimp_st.io import read_obs
    from bimp_st.paths import CANONICAL_TANGRAM

    o = read_obs(CANONICAL_TANGRAM)[
        ["cell_id", "slice_name", "leiden", "brain", "z_level", "tangram_cell_type"]
    ]
    o = o.reset_index(drop=True)
    o["leiden"] = o["leiden"].astype(int)
    # positional vote on an obs-only AnnData (canonical obs_names are duplicated)
    a = AnnData(
        obs=pd.DataFrame(
            {
                "leiden": o["leiden"].astype(str).values,
                "tangram_cell_type": o["tangram_cell_type"].astype(str).values,
            }
        )
    )
    tab = hierarchical_vote(
        a, cluster_col="leiden", label_col="tangram_cell_type", dst="cluster_vote"
    )
    o["vote"] = a.obs["cluster_vote"].astype(str).values
    tab.index = tab.index.astype(int)

    flags = pd.read_parquet(nr.RESULTS / "region_flags.parquet")
    regions = list(nr.DESCRIPTIVE_REGIONS)
    cols = [f"in_{r}" for r in regions]
    miss = [c for c in cols if c not in flags.columns]
    assert not miss, miss
    flags = flags.drop_duplicates(["slice_name", "cell_id"])
    m = o[["slice_name", "cell_id"]].merge(
        flags[["slice_name", "cell_id"] + cols],
        on=["slice_name", "cell_id"],
        how="left",
    )
    assert len(m) == len(o)
    assert m[cols[0]].notna().all(), "cells without region flags"
    # eligibility: a region counts as traced in a section if any cell has a
    # non-null flag
    elig = {}
    for c in cols:
        has = m[c].notna().groupby(o["slice_name"].values).any()
        elig[c] = has
    elig = pd.DataFrame(elig)
    # membership only (known inside); eligibility (traced) is kept separately in elig
    memb = np.column_stack(
        [
            pd.array(m[c], dtype="boolean").to_numpy(dtype=bool, na_value=False)
            for c in cols
        ]
    )
    return o, tab, regions, memb, elig


# ---------------------------------------------------------------- measures
def _H(v):
    v = np.asarray(v, float)
    s = v.sum()
    if s <= 0:
        return np.nan
    p = v / s
    nz = p > 0
    return float(-(p[nz] * np.log(p[nz])).sum() / np.log(len(v)))


def _gini(v):
    v = np.sort(np.asarray(v, float))
    n = len(v)
    s = v.sum()
    if s <= 0:
        return np.nan
    return float((2 * np.sum(np.arange(1, n + 1) * v) / (n * s)) - (n + 1) / n)


def _wq(pos, w, q):
    """weighted quantile on discrete ordered positions (mid-point cdf)."""
    o = np.argsort(pos)
    pos, w = pos[o], w[o]
    cdf = np.cumsum(w) - w / 2
    return float(np.interp(q, cdf / w.sum(), pos))


def section_measures(c, N, pos):
    """c: unit counts per section, N: section totals, pos: fractional stack
    position per section (None for pooled scope). Returns dict."""
    tot = c.sum()
    f = c / N
    d = dict(
        H_cnt=_H(c),
        gini_cnt=_gini(c),
        H_frac=_H(f),
        gini_frac=_gini(f),
        n_sec_ge1pct_of_unit=int((c / tot >= PRESENT_FRAC).sum()),
        n_sec_unit_ge1pct_of_sec=int((f >= PRESENT_FRAC).sum()),
    )
    if pos is not None and f.sum() > 0:
        w = f / f.sum()
        d["ap_span90"] = _wq(pos, w, 0.95) - _wq(pos, w, 0.05)
        d["ap_center"] = float((pos * w).sum())
        occ = c / tot >= PRESENT_FRAC
        d["ap_first"] = float(pos[occ].min()) if occ.any() else np.nan
        d["ap_last"] = float(pos[occ].max()) if occ.any() else np.nan
    return d


MEAS = [
    "H_cnt",
    "gini_cnt",
    "H_frac",
    "gini_frac",
    "n_sec_ge1pct_of_unit",
    "n_sec_unit_ge1pct_of_sec",
    "ap_span90",
]


def null_for(n_u, N, pos, rng):
    """size-matched null: n_u cells drawn at random from the scope; same
    per-section totals N. Returns array (N_NULL, len(MEAS))."""
    draws = rng.multivariate_hypergeometric(N.astype(np.int64), int(n_u), size=N_NULL)
    rows = []
    for c in draws:
        d = section_measures(c.astype(float), N.astype(float), pos)
        rows.append([d.get(k, np.nan) for k in MEAS])
    return np.array(rows)


# ---------------------------------------------------------------- NN (raw coords)
def _nn_mean(P):
    from scipy.spatial import cKDTree

    if len(P) < 3:
        return np.nan
    d = cKDTree(P).query(P, k=2)[0][:, 1]
    return float(d.mean())


def _nn_section(args):
    """per-section NN statistic for one crop. Cells are those of the atlas that
    have a raw coordinate (join on cell_id within the crop)."""
    sname, brain, cell_ids, ucl, uvt = args
    import h5py
    from anndata.io import read_elem

    p = RAW_DIRS[brain] / f"{sname}.h5ad"
    with h5py.File(p, "r") as f:
        obs = read_elem(f["obs"])
        xy = np.asarray(read_elem(f["obsm"]["spatial"]))[:, :2]
    raw_ids = (
        obs["cell_id"].astype(str).values
        if "cell_id" in obs
        else obs.index.astype(str).values
    )
    assert len(set(raw_ids)) == len(raw_ids), f"duplicate cell_id in {sname}"
    idx = pd.Series(np.arange(len(raw_ids)), index=raw_ids)
    j = idx.reindex(cell_ids.astype(str)).values
    ok = ~np.isnan(j)
    P = xy[j[ok].astype(int)]
    cl, vt = ucl[ok], uvt[ok]
    rng = np.random.default_rng(zlib.crc32(sname.encode()) + SEED)
    rows = []
    for level, lab in (("cluster", cl), ("vote", vt)):
        for u in pd.unique(lab):
            m = lab == u
            n = int(m.sum())
            if n < NMIN_SECTION:
                continue
            obs_nn = _nn_mean(P[m])
            nulls = [
                _nn_mean(P[rng.choice(len(P), n, replace=False)])
                for _ in range(NN_NULL)
            ]
            rows.append(
                dict(
                    level=level,
                    unit=u,
                    slice_name=sname,
                    brain=brain,
                    n_cells=n,
                    n_section=len(P),
                    nn_obs=obs_nn,
                    nn_null=float(np.mean(nulls)),
                    nn_null_sd=float(np.std(nulls)),
                    coverage=float(ok.mean()),
                )
            )
    return rows


# ---------------------------------------------------------------- compute
def compute():
    OUT.mkdir(parents=True, exist_ok=True)
    o, tab, regions, memb, elig = load_cells()
    tab.reset_index().to_csv(OUT / "cluster_to_vote_group.csv", index=False)
    rng = np.random.default_rng(SEED)

    sec = o.groupby("slice_name", sort=True).agg(
        brain=("brain", "first"), z=("z_level", "first"), n=("leiden", "size")
    )
    sec["pos"] = sec["z"] / sec.groupby("brain")["z"].transform("max")
    sec.to_csv(OUT / "sections.csv")
    log("sections", len(sec), sec.groupby("brain").size().to_dict())

    traced_any = memb.any(axis=1)
    log(
        "cells",
        len(o),
        "inside any traced region",
        int(traced_any.sum()),
        "outside",
        int((~traced_any).sum()),
    )
    sidx = pd.Categorical(o["slice_name"], categories=sec.index).codes
    reg_names = [f"in_{r}" for r in regions]

    unit_rows, region_rows, count_rows = [], [], []
    levels = {"cluster": o["leiden"].values, "vote": o["vote"].values}
    for level, lab in levels.items():
        for u in pd.unique(lab):
            um = lab == u
            ct = np.bincount(sidx[um], minlength=len(sec)).astype(float)
            for s, name in enumerate(sec.index):
                if ct[s]:
                    count_rows.append((level, u, name, int(ct[s])))
            for scope in SCOPES:
                sm = (
                    np.ones(len(sec), bool)
                    if scope == "pooled"
                    else (sec["brain"].values == scope)
                )
                c = ct[sm]
                Nn = sec["n"].values[sm].astype(float)
                row = dict(
                    level=level,
                    unit=u,
                    scope=scope,
                    n_cells=int(c.sum()),
                    n_cells_total=int(um.sum()),
                    n_sections=int(sm.sum()),
                )
                if c.sum() < NMIN_SCOPE:
                    unit_rows.append(row)
                    continue
                if scope == "pooled":
                    d = section_measures(c, Nn, None)
                    nl = null_for(c.sum(), Nn, None, rng)
                else:
                    d = section_measures(c, Nn, sec["pos"].values[sm])
                    nl = null_for(c.sum(), Nn, sec["pos"].values[sm], rng)
                for k, key in enumerate(MEAS):
                    row[key] = d.get(key, np.nan)
                    row[key + "_null_mean"] = (
                        float(np.nanmean(nl[:, k]))
                        if not np.all(np.isnan(nl[:, k]))
                        else np.nan
                    )
                    row[key + "_null_lo"] = (
                        float(np.nanpercentile(nl[:, k], 2.5))
                        if not np.all(np.isnan(nl[:, k]))
                        else np.nan
                    )
                    row[key + "_null_hi"] = (
                        float(np.nanpercentile(nl[:, k], 97.5))
                        if not np.all(np.isnan(nl[:, k]))
                        else np.nan
                    )
                for k in ("ap_center", "ap_first", "ap_last"):
                    row[k] = d.get(k, np.nan)
                # region view (restricted to sections where the region was traced)
                bm = um & (
                    np.ones(len(o), bool)
                    if scope == "pooled"
                    else (o["brain"].values == scope)
                )
                nb = int(bm.sum())
                inside = traced_any[bm]
                row["outside_frac"] = float(1 - inside.mean())
                row["inside_n"] = int(inside.sum())
                cats = []
                for j, rn in enumerate(reg_names):
                    ok_sec = (
                        elig[rn].reindex(o["slice_name"].values[bm]).values.astype(bool)
                    )
                    den = int(ok_sec.sum())
                    num = int((memb[bm, j] & ok_sec).sum())
                    row[f"frac_{rn}"] = num / den if den else np.nan
                    region_rows.append(
                        dict(
                            level=level,
                            unit=u,
                            scope=scope,
                            region=rn,
                            n_in=num,
                            n_eligible=den,
                            frac_of_eligible=num / den if den else np.nan,
                            frac_of_unit=memb[bm, j].sum() / nb,
                        )
                    )
                    cats.append(memb[bm, j].astype(float))
                cats = np.column_stack(cats)
                k = cats.sum(axis=1, keepdims=True)
                alloc = np.where(k > 0, cats / np.maximum(k, 1), 0.0)
                vec = np.append(alloc.sum(axis=0), (k[:, 0] == 0).sum())
                row["region_H9"] = _H(vec)
                region_rows.append(
                    dict(
                        level=level,
                        unit=u,
                        scope=scope,
                        region="outside",
                        n_in=int((k[:, 0] == 0).sum()),
                        n_eligible=nb,
                        frac_of_eligible=float(row["outside_frac"]),
                        frac_of_unit=float(row["outside_frac"]),
                    )
                )
                unit_rows.append(row)
        log("measures done:", level)

    units = pd.DataFrame(unit_rows)
    pd.DataFrame(count_rows, columns=["level", "unit", "slice_name", "n_cells"]).to_csv(
        OUT / "unit_section_counts.csv", index=False
    )
    pd.DataFrame(region_rows).to_csv(OUT / "unit_region_profile.csv", index=False)

    # NN, per section, raw coordinates
    jobs = []
    for name, g in o.groupby("slice_name", sort=True):
        jobs.append(
            (
                name,
                g["brain"].iloc[0],
                g["cell_id"].values,
                g["leiden"].values,
                g["vote"].values,
            )
        )
    log("NN over", len(jobs), "sections with", N_PROC, "procs")
    with Pool(N_PROC) as pool:
        res = pool.map(_nn_section, jobs, chunksize=1)
    nn = pd.DataFrame([r for rr in res for r in rr])
    nn["nn_ratio"] = nn["nn_obs"] / nn["nn_null"]
    nn.to_csv(OUT / "nn_by_section.csv", index=False)
    log(
        "raw-coordinate join coverage: min %.3f, mean %.3f"
        % (nn.coverage.min(), nn.coverage.mean())
    )

    # summarize NN per unit and scope
    nrows = []
    for (level, u), g in nn.groupby(["level", "unit"]):
        for scope in SCOPES:
            h = g if scope == "pooled" else g[g.brain == scope]
            if len(h) == 0:
                continue
            nrows.append(
                dict(
                    level=level,
                    unit=u,
                    scope=scope,
                    nn_n_sections=len(h),
                    nn_ratio_median=float(h.nn_ratio.median()),
                    nn_ratio_wmean=float(np.average(h.nn_ratio, weights=h.n_cells)),
                    nn_ratio_q25=float(h.nn_ratio.quantile(0.25)),
                    nn_ratio_q75=float(h.nn_ratio.quantile(0.75)),
                )
            )
    nns = pd.DataFrame(nrows)
    units["unit"] = units["unit"].astype(str)
    nns["unit"] = nns["unit"].astype(str)
    units = units.merge(nns, on=["level", "unit", "scope"], how="left")

    # labels
    lab_map = tab["label"].astype(str).to_dict()
    units["vote_label"] = [
        lab_map.get(int(u)) if lvl == "cluster" else u
        for lvl, u in zip(units["level"], units["unit"], strict=True)
    ]
    units = add_composite(units)
    units.to_csv(OUT / "unit_measures.csv", index=False)

    cover = dict(
        n_cells=int(len(o)),
        n_sections=int(len(sec)),
        n_inside_any_traced=int(traced_any.sum()),
        n_outside_traced=int((~traced_any).sum()),
        outside_fraction=float((~traced_any).mean()),
        by_brain={
            b: dict(
                n=int((o.brain == b).sum()),
                outside=int(((~traced_any) & (o.brain.values == b)).sum()),
            )
            for b in ("queen", "worker")
        },
        note=(
            "inside = member of >=1 of the eight traced regions (central complex null "
            "counted as not-inside); the 460,274 quoted elsewhere differs from this by "
            "the cells counted here from the flags table directly"
        ),
    )
    (OUT / "coverage.json").write_text(json.dumps(cover, indent=2))
    log(json.dumps(cover, indent=2))


def add_composite(units):
    """(e) composite: mean percentile rank of three tracing-free components,
    each oriented so larger = more spatially distinct. Components are kept as
    columns. Computed within (level, scope) among units with all components."""
    u = units.copy()
    u["comp_section"] = u["H_frac_null_mean"] - u["H_frac"]  # concentration beyond null
    u["comp_ap"] = u["ap_span90_null_mean"] - u["ap_span90"]  # narrower than null
    u["comp_local"] = 1 - u["nn_ratio_median"]  # tighter than section-matched null
    u["distinctness"] = np.nan
    for (lv, sc), g in u.groupby(["level", "scope"]):
        if sc == "pooled":
            # pooled AP = mean of the two brains' components
            for k in ("comp_ap",):
                b = (
                    u[(u.level == lv) & (u.scope.isin(["queen", "worker"]))]
                    .groupby("unit")[k]
                    .mean()
                )
                u.loc[g.index, k] = g["unit"].map(b).values
        gg = u.loc[g.index]
        ok = gg[["comp_section", "comp_ap", "comp_local"]].notna().all(axis=1)
        if ok.sum() < 3:
            continue
        r = gg.loc[ok, ["comp_section", "comp_ap", "comp_local"]].rank(pct=True)
        u.loc[r.index, "distinctness"] = r.mean(axis=1)
    return u


# ---------------------------------------------------------------- figure
def figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from bimp_st import nine_region_common as nr
    from bimp_st.si_common import (
        LEGEND_KW,
        QUEEN_COLOR,
        WORKER_COLOR,
        panel_label_size,
        set_style,
    )

    set_style()
    units = pd.read_csv(OUT / "unit_measures.csv", dtype={"unit": str})
    prof = pd.read_csv(OUT / "unit_region_profile.csv", dtype={"unit": str})
    regions = list(nr.DESCRIPTIVE_REGIONS)
    rcols = plt.get_cmap("tab10")
    rcol = {f"in_{r}": rcols(i) for i, r in enumerate(regions)}

    for level in ("cluster", "vote"):
        draw_main(
            units,
            prof,
            level,
            regions,
            rcol,
            plt,
            GridSpec,
            (LEGEND_KW, QUEEN_COLOR, WORKER_COLOR, panel_label_size),
            with_examples=(level == "cluster"),
        )


def display_label(level, unit, vote_label):
    from bimp_st.cluster_labels import get_canonical_display

    try:
        v = get_canonical_display(vote_label)
    except Exception:
        v = vote_label
    return f"{unit} {v}" if level == "cluster" else str(v)


def draw_main(units, prof, level, regions, rcol, plt, GridSpec, sty, with_examples):
    LEGEND_KW, QC, WC, panel_label_size = sty
    P = units[(units.level == level) & (units.scope == "pooled")].dropna(
        subset=["distinctness"]
    )
    P = P.sort_values("distinctness", ascending=False)
    order = list(P["unit"])
    n = len(order)
    y = np.arange(n)[::-1]
    qw = units[(units.level == level) & units.scope.isin(["queen", "worker"])]
    rows = {(r.unit, r.scope): r for r in qw.itertuples()}
    nrow_ex = 1 if with_examples else 0
    h = max(4.0, 0.2 * n + 1.6)
    fig = plt.figure(figsize=(11.5, h * 1.15 + (3.6 if with_examples else 0)))
    gs = GridSpec(
        1 + nrow_ex,
        8,
        figure=fig,
        height_ratios=[h * 1.15] + ([3.6] if with_examples else []),
        hspace=0.28,
        wspace=0.28,
        width_ratios=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4, 0.05],
    )
    axs = [fig.add_subplot(gs[0, 0])]
    for k in range(1, 7):
        axs.append(fig.add_subplot(gs[0, k], sharey=axs[0]))
    names = [
        display_label(level, u, P.set_index("unit").loc[u, "vote_label"]) for u in order
    ]
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(names)
    for a in axs[1:]:
        plt.setp(a.get_yticklabels(), visible=False)
        a.tick_params(left=False)
    # col 0 composite
    ax = axs[0]
    ax.barh(y, P["distinctness"].values, color="0.35", height=0.7)
    ax.set_xlim(0, 1)
    ax.set_title("Distinct-\nness", fontsize=8)
    ax.set_xlabel("rank mean", fontsize=7)
    # col 1 section spread (fraction profile)
    ax = axs[1]
    for yy, u in zip(y, order, strict=False):
        p = P.set_index("unit").loc[u]
        ax.plot(
            [p.H_frac_null_lo, p.H_frac_null_hi],
            [yy, yy],
            color="0.8",
            lw=3,
            solid_capstyle="butt",
        )
        for sc, c in (("queen", QC), ("worker", WC)):
            r = rows.get((u, sc))
            if r is not None and not np.isnan(getattr(r, "H_frac", np.nan)):
                ax.plot(r.H_frac, yy, "o", color=c, ms=2.5, mfc="none", mew=0.8)
        ax.plot(p.H_frac, yy, "o", color="k", ms=3)
    ax.set_title("Section\nspread", fontsize=8)
    ax.set_xlabel("entropy", fontsize=7)
    ax.set_xticks([0.9, 1.0])
    ax.set_xlim(0.85, 1.005)
    # col 2 sections where unit is >=1% of section
    ax = axs[2]
    for yy, u in zip(y, order, strict=False):
        p = P.set_index("unit").loc[u]
        ax.plot(
            [p.n_sec_unit_ge1pct_of_sec_null_lo, p.n_sec_unit_ge1pct_of_sec_null_hi],
            [yy, yy],
            color="0.8",
            lw=3,
            solid_capstyle="butt",
        )
        ax.plot(p.n_sec_unit_ge1pct_of_sec, yy, "o", color="k", ms=3)
    ax.set_xlim(left=0)
    ax.set_title("Sections\nwith >=1%", fontsize=8)
    ax.set_xlabel("of 65", fontsize=7)
    ax.set_xticks([0, 30, 65])
    ax.set_xlim(0, 66)
    # col 3 AP span
    ax = axs[3]
    for yy, u in zip(y, order, strict=False):
        for sc, c in (("queen", QC), ("worker", WC)):
            r = rows.get((u, sc))
            if r is not None and not np.isnan(getattr(r, "ap_span90", np.nan)):
                ax.plot(
                    [r.ap_span90_null_lo, r.ap_span90_null_hi],
                    [yy, yy],
                    color="0.85",
                    lw=2.2,
                    solid_capstyle="butt",
                )
                ax.plot(r.ap_span90, yy, "o", color=c, ms=3)
    ax.set_xlim(0, 1)
    ax.set_title("Stack\nspan", fontsize=8)
    ax.set_xlabel("central 90%", fontsize=7)
    ax.set_xticks([0, 0.5, 1])
    # col 4 local clustering
    ax = axs[4]
    for yy, u in zip(y, order, strict=False):
        p = P.set_index("unit").loc[u]
        ax.plot(
            [p.nn_ratio_q25, p.nn_ratio_q75],
            [yy, yy],
            color="0.8",
            lw=3,
            solid_capstyle="butt",
        )
        for sc, c in (("queen", QC), ("worker", WC)):
            r = rows.get((u, sc))
            if r is not None and not np.isnan(getattr(r, "nn_ratio_median", np.nan)):
                ax.plot(
                    r.nn_ratio_median, yy, "o", color=c, ms=2.5, mfc="none", mew=0.8
                )
        ax.plot(p.nn_ratio_median, yy, "o", color="k", ms=3)
    ax.axvline(1, color="0.5", lw=0.6, ls="--")
    ax.set_title("Local\nclustering", fontsize=8)
    ax.set_xlabel("NN / null", fontsize=7)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xlim(0, 1.05)
    # col 5 outside fraction
    ax = axs[5]
    ax.barh(y, P["outside_frac"].values, color="0.6", height=0.7)
    glob = units[(units.level == level) & (units.scope == "pooled")]
    tot = glob["n_cells"].sum()
    base = (glob["outside_frac"] * glob["n_cells"]).sum() / tot
    ax.axvline(base, color="k", lw=0.6, ls="--")
    ax.set_xlim(0, 1)
    ax.set_title("Outside\ntracing", fontsize=8)
    ax.set_xlabel("fraction", fontsize=7)
    ax.set_xticks([0, 0.5, 1])
    # col 6 region profile stacked
    ax = axs[6]
    pf = prof[(prof.level == level) & (prof.scope == "pooled")]
    left = np.zeros(n)
    for rn in [f"in_{r}" for r in regions]:
        v = (
            pf[pf.region == rn]
            .set_index("unit")["frac_of_unit"]
            .reindex(order)
            .fillna(0)
            .values
        )
        ax.barh(y, v, left=left, color=rcol[rn], height=0.7, lw=0)
        left += v
    ax.set_xlim(0, 1)
    ax.set_title("Inside tracing,\nby region", fontsize=8)
    ax.set_xlabel("fraction", fontsize=7)
    ax.set_xticks([0, 0.5, 1])
    from bimp_st.nine_region_common import display as rdisp

    hs = [plt.Rectangle((0, 0), 1, 1, color=rcol[f"in_{r}"]) for r in regions]
    ax.legend(
        hs,
        [rdisp(r) for r in regions],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        **{**LEGEND_KW, "frameon": False, "fontsize": 6},
    )
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    pl = panel_label_size(11.5)
    fig.text(0.005, 0.995, "A", fontsize=pl, fontweight="bold", va="top")
    fig.text(0.76, 0.995, "B", fontsize=pl, fontweight="bold", va="top")
    # dot legend
    axs[1].plot([], [], "o", color="k", ms=3, label="pooled")
    axs[1].plot([], [], "o", color=QC, ms=2.5, mfc="none", label="queen")
    axs[1].plot([], [], "o", color=WC, ms=2.5, mfc="none", label="worker")
    axs[1].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        **{**LEGEND_KW, "frameon": False, "fontsize": 6},
    )
    if with_examples:
        draw_examples(fig, gs, units, P, level, order, panel_label_size, plt)
    stem = (
        "fig_spatial_breadth"
        if level == "cluster"
        else "fig_spatial_breadth_vote_groups"
    )
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    log("saved", OUT / f"{stem}.png")


def draw_examples(fig, gs, units, P, level, order, panel_label_size, plt):
    """C: a broad and a distinct cluster on one representative queen section,
    from raw per-crop coordinates."""
    import h5py
    from anndata.io import read_elem

    big = P[P.n_cells >= 20000]
    broad_u, dist_u = (
        big.sort_values("nn_ratio_median").iloc[-1]["unit"],
        big.iloc[0]["unit"],
    )
    o, _, _, _, _ = load_cells()
    o["leiden"] = o["leiden"].astype(str)
    sub = o[o.brain == "queen"]
    cnt = (
        sub[sub.leiden.isin([broad_u, dist_u])]
        .groupby(["slice_name", "leiden"])
        .size()
        .unstack(fill_value=0)
    )
    cnt = cnt[(cnt[broad_u] > 500) & (cnt[dist_u] > 500)]
    z = sub.groupby("slice_name").z_level.first()
    # mid-stack section with both units well represented
    cand = cnt.assign(
        score=np.minimum(cnt[broad_u], cnt[dist_u]),
        mid=(z.reindex(cnt.index) - z.median()).abs(),
    )
    sname = (
        cand.sort_values(["score"], ascending=False).head(5).sort_values("mid").index[0]
    )
    g = o[o.slice_name == sname]
    with h5py.File(RAW_DIRS["queen"] / f"{sname}.h5ad", "r") as f:
        ro = read_elem(f["obs"])
        xy = np.asarray(read_elem(f["obsm"]["spatial"]))[:, :2]
    ids = (
        (ro["cell_id"] if "cell_id" in ro else ro.index.to_series()).astype(str).values
    )
    pos = (
        pd.Series(np.arange(len(ids)), index=ids)
        .reindex(g["cell_id"].astype(str))
        .values
    )
    keep = ~np.isnan(pos)
    P_ = xy[pos[keep].astype(int)]
    lab = g["leiden"].values[keep]
    tab = P.set_index("unit")
    for k, (u, _ttl) in enumerate(((broad_u, "broad"), (dist_u, "distinct"))):
        ax = fig.add_subplot(
            gs[1, 2 * k + 0 : 2 * k + 2] if False else gs[1, [0, 3][k] : [3, 6][k]]
        )
        ax.scatter(P_[:, 0], P_[:, 1], s=0.15, c="0.85", lw=0, rasterized=True)
        m = lab == u
        ax.scatter(
            P_[m, 0],
            P_[m, 1],
            s=0.6,
            c="#D53E4F" if k else "#2166AC",
            lw=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.axis("off")
        nm = display_label(level, u, tab.loc[u, "vote_label"])
        ax.set_title(f"{nm}", loc="left", fontsize=8)
        ax.plot(
            [P_[:, 0].min(), P_[:, 0].min() + 1000],
            [P_[:, 1].max() + 80] * 2,
            "k",
            lw=1.5,
            clip_on=False,
        )
        ax.text(P_[:, 0].min(), P_[:, 1].max() + 330, "1 mm", fontsize=6, va="top")
    pl = panel_label_size(11.5)
    fig.text(0.005, 0.22, "C", fontsize=pl, fontweight="bold", va="top")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("compute", "all"):
        compute()
    if step in ("figure", "all"):
        figure()
