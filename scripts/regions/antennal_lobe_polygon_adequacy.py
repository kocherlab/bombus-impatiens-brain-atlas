"""Antennal lobe: are the worker slide 0069041 polygons under-drawn?

The 0069041 AL polygons (sections 3-5, the only ones traced on that slide) are
much smaller than 0068401's and hold a much smaller share of that slide's cells.
Two readings call for opposite handling:

  under-drawing  the true AL is larger than what was traced, so untraced AL
                 cells sit just outside the polygon and are counted non-AL;
  oblique cut    the polygon is right and the section clips the edge of the
                 structure; there is nothing outside it to recover.

Test. Everything is computed in raw per-crop micrometres (never the PASTE3
frame) and cells are keyed by (slice_name, cell_id). For every traced AL
polygon we build rings outside it (0-25, 25-50, 50-100, 100-200 um from the
boundary; cells inside any other traced region excluded) and give each cell
three "AL-likeness" scores that are 0 for the brain's average cell and 1 for the
average cell inside the reference AL polygons:

  e  expression:   projection of the cell's z-scored 101-gene profile on the AL
                   signature (mean z-profile of reference AL cells);
  l  composition:  log-odds weight of the cell's canonical-17 label, AL vs the
                   brain background, rescaled the same way;
  v  the same with the expression-only cluster-vote label;
  top  fraction of ring cells carrying the AL's top label (astrocyte-like glia).

The signature for a reference section never contains that section (leave one
section out). Reference = every queen section and worker slide 0068401 (all
well traced); "same_brain" uses the cell's own brain only, "pooled" uses both.

Controls. (i) matched distance rings around the optic-neuropil polygons (lamina,
medulla, lobula) and around the Kenyon-cell polygons of the same sections, so a
generic "next to a neuropil edge" effect is controlled; control cells within
200 um of the AL are dropped. (ii) the method is shown to be able to see
under-drawing: well-traced polygons are eroded to 50% and to 1/16 of their area
(the area ratio of 0069041 to 0068401) and re-ringed; the cells of the original
polygon that now lie outside the eroded one are genuine AL, so the ring profile
must rise. The oblique-cut prediction is the un-eroded reference profile.

Prevalence. If rings were partly AL, how many cells would that add? Mixing
model: a ring's AL fraction f = clip((s_ring - s_ctrl)/(1 - s_ctrl), 0, 1).
The "literal" variant counts every cell of every ring whose excess over the
control is positive with a bootstrap interval excluding zero. Descriptive only:
AL is in NO_PVALUE_REGIONS, and no p-value is attached to any caste contrast.

Outputs: results/handoff/C_antennal_lobe/adequacy/
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
from shapely.ops import unary_union

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from bimp_st import nine_region_common as nrc  # noqa: E402
from bimp_st.cluster_labels import add_canonical, hierarchical_vote  # noqa: E402
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.paths import CANONICAL_TANGRAM, SPATIAL_COMBINED  # noqa: E402
from bimp_st.style import set_pub_style  # noqa: E402

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ config
OUT = REPO / "results" / "handoff" / "C_antennal_lobe" / "adequacy"
CACHE = OUT / "cache"
CROP_TABLE = REPO / "results/handoff/C_antennal_lobe/step1_ap_crop_table.csv"
RING_EDGES = [0, 25, 50, 100, 200]  # um from the polygon boundary
RING_NAMES = ["0-25", "25-50", "50-100", "100-200"]
FINE_EDGES = list(range(-100, 201, 20))  # signed distance curve, um
MAX_D = 200.0
ERODE_FRACTIONS = {"e50": 0.5, "e16": 1 / 16}  # remaining polygon area
OPTIC = ("lamina", "medulla", "lobula")
KC = ("kc_oc", "kc_ic", "kc_nc")
TEST_SLIDE = "0069041"  # worker slide under test
REF_SLIDE_WORKER = "0068401"
N_BOOT = 2000
SEED = 0
Z_SD_FLOOR = 0.05
METRICS = ("e", "l", "v", "top")
METRIC_LABEL = {
    "e": "expression AL-likeness",
    "l": "canonical-17 composition AL-likeness",
    "v": "cluster-vote composition AL-likeness",
    "top": "fraction astrocyte-like glia",
}


# ------------------------------------------------------------------ labels
def label_table() -> pd.DataFrame:
    """(slice_name, cell_id) -> canonical-17 + cluster-vote label, cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / "labels.parquet"
    if f.exists():
        return pd.read_parquet(f)
    obs = read_obs(SPATIAL_COMBINED)
    t = read_obs(CANONICAL_TANGRAM)
    if len(t) != len(obs) or not (t["cell_id"].values == obs["cell_id"].values).all():
        raise SystemExit("corrected labels do not line up with combined object")
    obs = obs.copy()
    obs["tangram_cell_type"] = t["tangram_cell_type"].astype(str).values
    obs.index = pd.RangeIndex(len(obs)).astype(str)
    a = ad.AnnData(obs=obs)
    add_canonical(a, src="tangram_cell_type")
    hierarchical_vote(
        a, cluster_col="leiden", label_col="tangram_cell_type", dst="cluster_vote"
    )
    o = a.obs[
        ["slice_name", "cell_id", "brain", "cell_type_canonical", "cluster_vote"]
    ].astype(str)
    if o.duplicated(["slice_name", "cell_id"]).any():
        raise SystemExit("(slice_name, cell_id) is not unique in the atlas")
    o.to_parquet(f)
    return o


# ------------------------------------------------------------------ geometry
def erode_to_area(geom, frac: float):
    """Negative buffer that leaves ``frac`` of the area (bisection)."""
    target, lo, hi = geom.area * frac, 0.0, 600.0
    for _ in range(30):
        mid = (lo + hi) / 2
        a = geom.buffer(-mid).area
        lo, hi = (mid, hi) if a > target else (lo, mid)
    g = geom.buffer(-lo)
    return None if g.is_empty else g


def signed_distance(geom, xy: np.ndarray) -> np.ndarray:
    """Distance to ``geom`` boundary: >0 outside, <0 inside; inf if > MAX_D out."""
    d = np.full(len(xy), np.inf)
    if geom is None or geom.is_empty:
        return d
    x0, y0, x1, y1 = geom.bounds
    near = (
        (xy[:, 0] > x0 - MAX_D)
        & (xy[:, 0] < x1 + MAX_D)
        & (xy[:, 1] > y0 - MAX_D)
        & (xy[:, 1] < y1 + MAX_D)
    )
    if not near.any():
        return d
    pts = shapely.points(xy[near])
    inside = shapely.contains_xy(geom, xy[near, 0], xy[near, 1])
    dd = shapely.distance(geom.boundary, pts)
    d[near] = np.where(inside, -dd, dd)
    d[d > MAX_D] = np.inf
    return d


def find_crop(data_dir: Path, slide: str, suffix: str) -> Path | None:
    for f in sorted(data_dir.glob("*.h5ad")):
        n = f.name
        if "_clustered" in n or "_labeled" in n or n.startswith(("aligned", "test_")):
            continue
        if f.stem.endswith("_" + suffix) and f"__{slide}__" in n:
            return f
    return None


# ------------------------------------------------------------------ pass 1
def collect(labels: pd.DataFrame):
    """Walk every crop of every traced-AL section once.

    Returns (cells, X, bg, sections) where ``cells`` holds one row per kept cell
    (within MAX_D of an AL / control polygon, or inside the AL), ``X`` its
    log1p(CP10k) profile, ``bg`` per-brain background sums over all atlas cells
    of the crops visited, and ``sections`` per-section metadata.
    """
    import scipy.sparse as sp

    lab = labels.set_index(["slice_name", "cell_id"])
    rows, Xs, secs = [], [], []
    bg = {}
    for brain in ("queen", "worker"):
        src = nrc.annotation_sources(brain)
        cm = nrc.crop_map(brain)
        data_dir = nrc.BRAINS[brain]["dir"] / "data"
        bg[brain] = dict(n=0, s=np.zeros(101), ss=np.zeros(101))
        for (slide, sec), d in sorted(src.items()):
            path = d["v2"] or d["old"]
            regs = nrc.load_regions(path, vocabulary=None)
            al = regs.get("al")
            sid = f"{slide}_R{sec}"
            if al is None:
                secs.append(
                    dict(
                        section_id=sid,
                        brain=brain,
                        slide=slide,
                        section=sec,
                        al_traced=False,
                    )
                )
                continue
            eroded = {k: erode_to_area(al, f) for k, f in ERODE_FRACTIONS.items()}
            opt = [regs[r] for r in OPTIC if r in regs]
            kc = [regs[r] for r in KC if r in regs]
            g_opt = unary_union(opt) if opt else None
            g_kc = unary_union(kc) if kc else None
            others = {r: g for r, g in regs.items() if r != "al"}
            secs.append(
                dict(
                    section_id=sid,
                    brain=brain,
                    slide=slide,
                    section=sec,
                    al_traced=True,
                    al_area_um2=al.area,
                    n_al_parts=len(getattr(al, "geoms", [al])),
                    e50_area_um2=eroded["e50"].area if eroded["e50"] else 0.0,
                    e16_area_um2=eroded["e16"].area if eroded["e16"] else 0.0,
                )
            )
            for suf in cm.get((slide, sec), []):
                cf = find_crop(data_dir, slide, suf)
                if cf is None:
                    print(f"  !! no crop h5ad for {slide} R{sec} {suf}", flush=True)
                    continue
                a = ad.read_h5ad(cf)
                cid = a.obs["cell_id"].astype(str).to_numpy()
                if len(np.unique(cid)) != len(cid):
                    raise SystemExit(f"cell_id not unique within {cf.name}")
                # keep only atlas cells: they carry labels and define the atlas
                # prevalence (the raw crop holds cells QC later dropped)
                key = pd.MultiIndex.from_arrays([np.full(len(cid), cf.stem), cid])
                pos = lab.index.get_indexer(key)
                keep = pos >= 0
                xy = np.asarray(a.obsm["spatial"], float)[:, :2][keep]
                Xc = sp.csr_matrix(a.X)[keep].astype(np.float32)
                tot = np.asarray(Xc.sum(1)).ravel()
                Xc = sp.diags(1e4 / np.maximum(tot, 1)) @ Xc
                Xc.data = np.log1p(Xc.data)
                Xd = Xc.toarray().astype(np.float32)
                b = bg[brain]
                b["n"] += len(Xd)
                b["s"] += Xd.sum(0)
                b["ss"] += (Xd.astype(np.float64) ** 2).sum(0)

                d_al = signed_distance(al, xy)
                d_e = {k: signed_distance(g, xy) for k, g in eroded.items()}
                d_opt = signed_distance(g_opt, xy)
                d_kc = signed_distance(g_kc, xy)
                in_other = np.zeros(len(xy), bool)
                for g in others.values():
                    in_other |= shapely.contains_xy(g, xy[:, 0], xy[:, 1])
                sel = (
                    (d_al < np.inf)
                    | ((d_opt > 0) & (d_opt < np.inf))
                    | ((d_kc > 0) & (d_kc < np.inf))
                )
                lb = lab.iloc[pos[keep]]
                rows.append(
                    pd.DataFrame(
                        dict(
                            section_id=sid,
                            brain=brain,
                            slide=slide,
                            slice_name=cf.stem,
                            d_al=d_al,
                            d_e50=d_e["e50"],
                            d_e16=d_e["e16"],
                            d_opt=d_opt,
                            d_kc=d_kc,
                            in_other=in_other,
                            label=lb["cell_type_canonical"].to_numpy(),
                            vote=lb["cluster_vote"].to_numpy(),
                            n_crop_cells=len(xy),
                        )
                    ).loc[sel]
                )
                Xs.append(Xd[sel])
                print(
                    f"  {cf.stem[-24:]}: {len(xy):,} atlas cells, "
                    f"{int((d_al < 0).sum())} in AL, kept {int(sel.sum()):,}",
                    flush=True,
                )
    cells = pd.concat(rows, ignore_index=True)
    return cells, np.vstack(Xs), bg, pd.DataFrame(secs)


# ------------------------------------------------------------------ scoring
def label_background(labels: pd.DataFrame) -> dict:
    return {
        b: d["cell_type_canonical"].value_counts(normalize=True)
        for b, d in labels.groupby("brain")
    } | {
        f"{b}_vote": d["cluster_vote"].value_counts(normalize=True)
        for b, d in labels.groupby("brain")
    }


def score_cells(cells, X, bg, labels, ref_mode: str):
    """Add e, l, v, top columns using leave-one-section-out AL signatures."""
    mu, sd = {}, {}
    for b, v in bg.items():
        mu[b] = v["s"] / v["n"]
        sd[b] = np.maximum(
            np.sqrt(np.maximum(v["ss"] / v["n"] - mu[b] ** 2, 0)), Z_SD_FLOOR
        )
    Z = np.empty_like(X)
    for b in ("queen", "worker"):
        m = (cells["brain"] == b).to_numpy()
        Z[m] = (X[m] - mu[b]) / sd[b]

    inside = (cells["d_al"] < 0).to_numpy()
    ref_sections = set(
        cells.loc[
            (cells.brain == "queen") | (cells.slide == REF_SLIDE_WORKER), "section_id"
        ]
    )
    bgl = label_background(labels)
    lab_levels = sorted(set(labels["cell_type_canonical"]))
    vote_levels = sorted(set(labels["cluster_vote"]))

    sec_sig, sec_n, sec_lab, sec_vote = {}, {}, {}, {}
    for sid, _d in cells.groupby("section_id"):
        m = (cells["section_id"] == sid).to_numpy() & inside
        sec_sig[sid] = Z[m].sum(0)
        sec_n[sid] = int(m.sum())
        sec_lab[sid] = (
            cells.loc[m, "label"].value_counts().reindex(lab_levels, fill_value=0)
        )
        sec_vote[sid] = (
            cells.loc[m, "vote"].value_counts().reindex(vote_levels, fill_value=0)
        )
    brain_of = cells.groupby("section_id")["brain"].first().to_dict()

    out = {k: np.full(len(cells), np.nan) for k in ("e", "l", "v", "top")}
    ref_inside = inside & cells["section_id"].isin(ref_sections).to_numpy()
    top_label = cells.loc[ref_inside, "label"].value_counts().idxmax()
    top = top_label
    for sid in cells["section_id"].unique():
        b = brain_of[sid]
        pool = [
            s
            for s in ref_sections
            if s != sid and (ref_mode == "pooled" or brain_of[s] == b)
        ]
        n = sum(sec_n[s] for s in pool)
        sig = sum(sec_sig[s] for s in pool) / n
        m = (cells["section_id"] == sid).to_numpy()
        out["e"][m] = Z[m] @ sig / (sig @ sig)

        for tag, levels, cnt, col, bgk in (
            ("l", lab_levels, sec_lab, "label", b),
            ("v", vote_levels, sec_vote, "vote", f"{b}_vote"),
        ):
            c = sum(cnt[s] for s in pool) + 0.5
            p_al = c / c.sum()
            p_bg = bgl[bgk].reindex(levels, fill_value=0) + 1e-4
            p_bg = p_bg / p_bg.sum()
            w = np.log2(p_al / p_bg)
            e_al, e_bg = float((p_al * w).sum()), float((p_bg * w).sum())
            out[tag][m] = (cells.loc[m, col].map(w).to_numpy() - e_bg) / (e_al - e_bg)
            if tag == "l":
                out["top"][m] = (cells.loc[m, "label"].to_numpy() == top).astype(float)
    return pd.DataFrame(out, index=cells.index), Z, top_label


# ------------------------------------------------------------------ rings
def assign(d: np.ndarray, edges) -> np.ndarray:
    """Bin index into edges (left-open, right-closed); -1 outside the range."""
    i = np.digitize(d, edges, right=True) - 1
    ok = np.isfinite(d) & (d > edges[0]) & (d <= edges[-1])
    return np.where(ok, i, -1)


def build_long(cells, sc, edges, names, signed_inside: bool) -> pd.DataFrame:
    """Per section x variant x target x bin sufficient statistics."""
    c = cells.join(sc)
    c["excl_other"] = c["in_other"]
    pieces = []
    far_from_al = (c["d_al"] > MAX_D).to_numpy()  # controls: >200 um from AL
    specs = []
    for v, col in (("orig", "d_al"), ("e50", "d_e50"), ("e16", "d_e16")):
        specs.append((v, "AL", c[col].to_numpy(), np.ones(len(c), bool)))
    specs.append(("orig", "ctrl_optic", c["d_opt"].to_numpy(), far_from_al))
    specs.append(("orig", "ctrl_kc", c["d_kc"].to_numpy(), far_from_al))
    for variant, target, d, allowed in specs:
        dd = d.copy()
        if target != "AL":
            dd = np.where(dd > 0, dd, np.inf)  # outside the control polygon
        inside = (dd <= 0) & allowed
        idx = np.where(allowed, assign(dd, edges), -1)
        if signed_inside and target == "AL":
            idx = np.where(inside, -2, idx)
        for k in [-2] + list(range(len(names))):
            m = idx == k
            if not m.any():
                continue
            g = c[m]
            keep = ~g["excl_other"]
            agg = (
                g[keep]
                .groupby("section_id")
                .agg(
                    n=("e", "size"),
                    e=("e", "sum"),
                    l=("l", "sum"),
                    v=("v", "sum"),
                    top=("top", "sum"),
                )
            )
            agg["n_excl_other"] = (
                g[~keep].groupby("section_id").size().reindex(agg.index, fill_value=0)
            )
            agg = agg.reset_index()
            agg["variant"], agg["target"] = variant, target
            agg["bin"] = "inside" if k == -2 else names[k]
            pieces.append(agg)
    return pd.concat(pieces, ignore_index=True)


def pooled(long, sids, variant, target, bin_, rng_idx):
    """Cell-weighted pooled mean per metric, point + bootstrap over sections."""
    d = long[
        (long.variant == variant) & (long.target == target) & (long.bin == bin_)
    ].set_index("section_id")
    d = d.reindex(sids).fillna(0)
    n = d["n"].to_numpy()
    res = {}
    for m in METRICS:
        s = d[m].to_numpy()
        pt = s.sum() / n.sum() if n.sum() else np.nan
        bs = np.full(len(rng_idx), np.nan)
        nn = n[rng_idx].sum(1)
        ok = nn > 0
        bs[ok] = s[rng_idx].sum(1)[ok] / nn[ok]
        res[m] = (pt, bs)
    return int(n.sum()), int((n > 0).sum()), res


def ci(bs):
    b = bs[np.isfinite(bs)]
    return (
        (np.percentile(b, 2.5), np.percentile(b, 97.5))
        if len(b) > 50
        else (np.nan,) * 2
    )


def summarise(long, groups: dict, tag: str) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for gname, sids in groups.items():
        idx = rng.integers(0, len(sids), (N_BOOT, len(sids)))
        bins = [b for b in long["bin"].unique()]
        for variant in ("orig", "e50", "e16"):
            for bin_ in bins:
                tg = {}
                for target in ("AL", "ctrl_optic", "ctrl_kc"):
                    if target != "AL" and variant != "orig":
                        continue
                    tg[target] = pooled(long, sids, variant, target, bin_, idx)
                # pooled control = optic + kc ring cells together
                if "ctrl_optic" in tg:
                    d = long[
                        (long.variant == "orig")
                        & long.target.isin(["ctrl_optic", "ctrl_kc"])
                        & (long.bin == bin_)
                    ]
                    d = d.groupby("section_id")[["n", *METRICS]].sum()
                    d = d.reindex(sids).fillna(0)
                    n = d["n"].to_numpy()
                    r = {}
                    for m in METRICS:
                        s = d[m].to_numpy()
                        nn = n[idx].sum(1)
                        bs = np.full(N_BOOT, np.nan)
                        bs[nn > 0] = (s[idx].sum(1))[nn > 0] / nn[nn > 0]
                        r[m] = (s.sum() / n.sum() if n.sum() else np.nan, bs)
                    tg["ctrl_all"] = (int(n.sum()), int((n > 0).sum()), r)
                for target, (nc, ns, res) in tg.items():
                    for m, (pt, bs) in res.items():
                        lo, hi = ci(bs)
                        row = dict(
                            group=gname,
                            variant=variant,
                            target=target,
                            bin=bin_,
                            metric=m,
                            n_cells=nc,
                            n_sections=ns,
                            mean=pt,
                            ci_lo=lo,
                            ci_hi=hi,
                        )
                        if target == "AL" and "ctrl_all" in tg and bin_ != "inside":
                            cpt, cbs = tg["ctrl_all"][2][m]
                            elo, ehi = ci(bs - cbs)
                            row.update(
                                ctrl_mean=cpt,
                                excess=pt - cpt,
                                excess_lo=elo,
                                excess_hi=ehi,
                            )
                        rows.append(row)
    return pd.DataFrame(rows).assign(set=tag)


# ------------------------------------------------------------------ cosine
def ring_cosine(cells, Z, groups, label_sig):
    """Cosine of the ring mean z-profile with the AL signature, size matched."""
    rows = []
    rng = np.random.default_rng(SEED)
    c = cells
    far = (c["d_al"] > MAX_D).to_numpy()
    for gname, sids in groups.items():
        gm = c["section_id"].isin(sids).to_numpy() & ~c["in_other"].to_numpy()
        sig = label_sig[gname]
        specs = {
            "AL": c["d_al"].to_numpy(),
            "ctrl": np.where(c["d_opt"] > 0, c["d_opt"], np.inf),
        }
        ctrl2 = np.where(c["d_kc"] > 0, c["d_kc"], np.inf)
        for k, nm in enumerate(RING_NAMES):
            lo, hi = RING_EDGES[k], RING_EDGES[k + 1]
            ma = gm & (specs["AL"] > lo) & (specs["AL"] <= hi)
            mc = (
                gm
                & far
                & (
                    ((specs["ctrl"] > lo) & (specs["ctrl"] <= hi))
                    | ((ctrl2 > lo) & (ctrl2 <= hi))
                )
            )
            n = min(int(ma.sum()), int(mc.sum()), 300)
            if n < 20:
                continue
            for tgt, m in (("AL", ma), ("ctrl", mc)):
                ii = np.flatnonzero(m)
                cs = []
                for _ in range(200):
                    mean = Z[rng.choice(ii, n, replace=False)].mean(0)
                    cs.append(mean @ sig / (np.linalg.norm(mean) * np.linalg.norm(sig)))
                rows.append(
                    dict(
                        group=gname,
                        ring=nm,
                        target=tgt,
                        n_matched=n,
                        cosine_median=np.median(cs),
                        cosine_lo=np.percentile(cs, 2.5),
                        cosine_hi=np.percentile(cs, 97.5),
                    )
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ prevalence
RING_SETS = {"0-50": [0, 1], "0-100": [0, 1, 2], "0-200": [0, 1, 2, 3]}
SCENARIO_NAMES = (
    "full_range_as_shipped",
    "traced_worker_sections_only",
    "matched_windows",
)


def ring_arrays(long, sids, variant, targets=("AL",), metric="e"):
    """(sections x 4 rings) sums of the score and cell counts."""
    d = long[
        (long.variant == variant)
        & long.target.isin(targets)
        & long.bin.isin(RING_NAMES)
    ]
    s = (
        d.pivot_table(index="section_id", columns="bin", values=metric, aggfunc="sum")
        .reindex(index=sids, columns=RING_NAMES)
        .fillna(0)
    )
    n = (
        d.pivot_table(index="section_id", columns="bin", values="n", aggfunc="sum")
        .reindex(index=sids, columns=RING_NAMES)
        .fillna(0)
    )
    return s.to_numpy(), n.to_numpy()


def ring_fraction(ts, tn, bs, bn):
    """Mixing-model AL fraction of each ring relative to a baseline profile."""
    st = ts.sum(0) / np.maximum(tn.sum(0), 1)
    sb = bs.sum(0) / np.maximum(bn.sum(0), 1)
    return np.clip((st - sb) / (1 - sb), 0, 1)


def recovery(test, base, rng):
    """Point and bootstrap (sections resampled in both arms) ring fractions."""
    ts, tn = test
    bs, bn = base
    point = ring_fraction(ts, tn, bs, bn)
    boot = np.empty((N_BOOT, 4))
    for i in range(N_BOOT):
        it = rng.integers(0, len(ts), len(ts))
        ib = rng.integers(0, len(bs), len(bs))
        boot[i] = ring_fraction(ts[it], tn[it], bs[ib], bn[ib])
    return point, boot


def prevalence(long, groups, crop_table):
    """Recover-the-deficit arithmetic for the 0069041 test sections."""
    rng = np.random.default_rng(SEED)
    ct = crop_table.copy()
    traced = ct.al_traced.astype(bool)
    test_crops = ct[(ct.slide.astype(str) == TEST_SLIDE) & traced]
    ref_w = ct[ct.slide.astype(str) == REF_SLIDE_WORKER]
    pct_ref = ref_w.n_al.sum() / ref_w.n_cells.sum()
    deficit = float(test_crops.n_cells.sum() * pct_ref - test_crops.n_al.sum())

    test_sids = groups["worker_0069041"]
    test = ring_arrays(long, test_sids, "orig")
    N = test[1].sum(0)
    baselines = {
        "well_traced_0068401_edge": ring_arrays(long, groups["worker_0068401"], "orig"),
        "well_traced_all_edge": ring_arrays(long, groups["ref_all"], "orig"),
        "neuropil_edge_control": None,
    }
    # control rings from the TEST sections themselves (same sections, optic + kc)
    baselines["neuropil_edge_control"] = ring_arrays(
        long, test_sids, "orig", targets=("ctrl_optic", "ctrl_kc")
    )

    q = ct[ct.brain == "queen"]
    w = ct[ct.brain == "worker"]
    scen = {
        "full_range_as_shipped": (q, w),
        "traced_worker_sections_only": (q, w[w.al_traced.astype(bool)]),
        "matched_windows": (
            q[q.in_matched_window.astype(bool)],
            w[w.al_traced.astype(bool)],
        ),
    }
    rows, det = [], []
    for bname, base in baselines.items():
        point, boot = recovery(test, base, rng)
        det.append(
            pd.DataFrame(
                dict(
                    baseline=bname,
                    ring=RING_NAMES,
                    n_ring_cells_test_sections=N.astype(int),
                    al_fraction=point,
                    al_fraction_lo=np.percentile(boot, 2.5, axis=0),
                    al_fraction_hi=np.percentile(boot, 97.5, axis=0),
                )
            )
        )
        for sname, idx in RING_SETS.items():
            rec = float((point[idx] * N[idx]).sum())
            rb = (boot[:, idx] * N[idx]).sum(1)
            lo, hi = np.percentile(rb, [2.5, 97.5])
            for scn, (qq, ww) in scen.items():
                qp = qq.n_al.sum() / qq.n_cells.sum()
                wc = ww.n_cells.sum()
                wp0 = ww.n_al.sum() / wc
                n_al_w = ww.n_al.sum()

                def wp(r, n_al=n_al_w, wc=wc):
                    return (n_al + r) / wc

                rows.append(
                    dict(
                        baseline=bname,
                        rings=sname,
                        scenario=scn,
                        recovered_cells=rec,
                        recovered_lo=lo,
                        recovered_hi=hi,
                        frac_of_deficit=rec / deficit,
                        frac_of_deficit_lo=lo / deficit,
                        frac_of_deficit_hi=hi / deficit,
                        queen_al_pct=100 * qp,
                        worker_al_pct_as_traced=100 * wp0,
                        worker_al_pct_with_recovery=100 * wp(rec),
                        ratio_as_traced=qp / wp0,
                        ratio_with_recovery=qp / wp(rec),
                        ratio_lo=qp / wp(hi),
                        ratio_hi=qp / wp(lo),
                    )
                )
    meta = dict(
        pct_ref_0068401=100 * pct_ref,
        test_cells=int(test_crops.n_cells.sum()),
        test_al_cells=int(test_crops.n_al.sum()),
        deficit_cells=deficit,
    )
    return pd.DataFrame(rows), pd.concat(det, ignore_index=True), meta


def detection_control(cells, long, groups):
    """Can the ring method see under-drawing?  Eroded well-traced polygons.

    Baseline is the same group's un-eroded ring profile, so the un-eroded polygon
    recovers nothing by construction; the question is how much of the genuine AL
    that was cut away (cells of the original polygon outside the eroded one but
    within 200 um of it) is recovered, and what a well-traced group recovers when
    judged against the other brain's well-traced profile (a noise floor).
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for gname in ("queen", "worker_0068401", "ref_all"):
        sids = groups[gname]
        base = ring_arrays(long, sids, "orig")
        for variant, col in (("e50", "d_e50"), ("e16", "d_e16")):
            test = ring_arrays(long, sids, variant)
            point, boot = recovery(test, base, rng)
            cc = cells[cells.section_id.isin(sids) & ~cells.in_other]
            d = cc[col].to_numpy()
            known_r = [
                int(((cc.d_al < 0) & (assign(d, RING_EDGES) == k)).sum())
                for k in range(4)
            ]
            N = test[1].sum(0)
            for sname, idx in RING_SETS.items():
                rec = float((point[idx] * N[idx]).sum())
                lo, hi = np.percentile((boot[:, idx] * N[idx]).sum(1), [2.5, 97.5])
                kn = sum(known_r[i] for i in idx)
                rows.append(
                    dict(
                        group=gname,
                        variant=variant,
                        rings=sname,
                        known_al_cells_in_rings=kn,
                        recovered=rec,
                        recovered_lo=lo,
                        recovered_hi=hi,
                        detected_fraction=rec / kn if kn else np.nan,
                    )
                )
    # noise floor: a well-traced group judged against the other brain's profile
    for gname, other in (("queen", "worker_0068401"), ("worker_0068401", "queen")):
        point, boot = recovery(
            ring_arrays(long, groups[gname], "orig"),
            ring_arrays(long, groups[other], "orig"),
            rng,
        )
        N = ring_arrays(long, groups[gname], "orig")[1].sum(0)
        inside = int(
            cells[cells.section_id.isin(groups[gname]) & (cells.d_al < 0)].shape[0]
        )
        for sname, idx in RING_SETS.items():
            rec = float((point[idx] * N[idx]).sum())
            rows.append(
                dict(
                    group=gname,
                    variant=f"orig_vs_{other}_baseline",
                    rings=sname,
                    known_al_cells_in_rings=0,
                    recovered=rec,
                    detected_fraction=np.nan,
                    recovered_over_polygon_cells=rec / inside,
                )
            )
    return pd.DataFrame(rows)


def halo_tables(long, secs):
    """Per-section AL-likeness of the 0-25 and 25-100 um zones."""
    d = long[long.target == "AL"].copy()
    d["zone"] = d["bin"].map(
        {
            "0-25": "0-25",
            "25-50": "25-100",
            "50-100": "25-100",
            "100-200": "100-200",
            "inside": "inside",
        }
    )
    g = d.groupby(["section_id", "variant", "zone"])[["n", "e", "l", "v", "top"]].sum()
    for m in METRICS:
        g[m] = g[m] / g["n"]
    g = g.reset_index().merge(secs[["section_id", "brain", "slide"]], on="section_id")
    g["group"] = np.where(
        g.brain == "queen", "queen", "worker_" + g.slide.astype(str).str.zfill(7)
    )
    return g


def halo_summary(h):
    rows = []
    t = h[(h.group == "worker_0069041") & (h.variant == "orig")]
    for zone in ("0-25", "25-100", "100-200"):
        for m in ("e", "l", "v"):
            tv = t[t.zone == zone].set_index("section_id")[m]
            for variant in ("orig", "e50", "e16"):
                r = h[
                    (h.group.isin(["queen", "worker_0068401"]))
                    & (h.variant == variant)
                    & (h.zone == zone)
                ]
                rv = r[m].dropna()
                rows.append(
                    dict(
                        zone=zone,
                        metric=m,
                        reference_variant=variant,
                        n_ref_sections=len(rv),
                        ref_mean=rv.mean(),
                        ref_q05=rv.quantile(0.05),
                        ref_median=rv.median(),
                        ref_q95=rv.quantile(0.95),
                        test_mean=tv.mean(),
                        test_values="; ".join(f"{k}={v:.2f}" for k, v in tv.items()),
                        frac_ref_sections_at_or_below_test_mean=(
                            rv <= tv.mean()
                        ).mean(),
                        frac_ref_sections_at_or_above_test_mean=(
                            rv >= tv.mean()
                        ).mean(),
                    )
                )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figure
def curve(long_fine, sids, variant, target, metric, rng_idx, edges):
    names = [f"{a}:{b}" for a, b in zip(edges[:-1], edges[1:], strict=True)]
    mid, mean, lo, hi = [], [], [], []
    for (a, b), nm in zip(zip(edges[:-1], edges[1:], strict=True), names, strict=False):
        n, ns, res = pooled(long_fine, sids, variant, target, nm, rng_idx)
        if n < 15:
            continue
        pt, bs = res[metric]
        lo_ci, hi_ci = ci(bs)
        mid.append((a + b) / 2), mean.append(pt), lo.append(lo_ci), hi.append(hi_ci)
    return np.array(mid), np.array(mean), np.array(lo), np.array(hi)


def figure(summ, long_fine, groups, prev, halo, fname):
    set_pub_style()
    col = {
        "queen": "#8E4A9C",
        "worker_0068401": "#E8A317",
        "worker_0069041": "#1B7F8C",
        "ctrl": "0.55",
    }
    nice = {
        "queen": "queen (16 sections)",
        "worker_0068401": "worker 0068401 (8)",
        "worker_0069041": "worker 0069041 (3)",
    }
    fig = plt.figure(figsize=(10.5, 6.0))
    gs = fig.add_gridspec(2, 3, hspace=1.0, wspace=0.42)
    rng = np.random.default_rng(SEED)

    # a: signed-distance curve
    ax = fig.add_subplot(gs[0, 0])
    fine = long_fine.copy()
    for g in ("queen", "worker_0068401", "worker_0069041"):
        sids = groups[g]
        idx = rng.integers(0, len(sids), (500, len(sids)))
        x, y, lo, hi = curve(fine, sids, "orig", "AL", "e", idx, FINE_EDGES)
        ax.plot(x, y, "-o", ms=3, lw=1.2, color=col[g], label=nice[g])
        ax.fill_between(x, lo, hi, color=col[g], alpha=0.15, lw=0)
    sids = groups["ref_all"]
    idx = rng.integers(0, len(sids), (500, len(sids)))
    x, y, lo, hi = curve(fine, sids, "e16", "AL", "e", idx, FINE_EDGES)
    ax.plot(x, y, "--", lw=1.2, color="k", label="reference eroded to 1/16")
    ax.axvline(0, color="0.6", lw=0.6)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.set_xlabel("distance from edge (um)")
    ax.set_ylabel("expression AL-likeness")
    ax.legend(
        frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.28)
    )

    # b, c: ring profiles
    def ring_panel(ax, metric, ylab):
        xs = np.arange(len(RING_NAMES) + 1)
        labs = ["inside"] + RING_NAMES
        spec = [
            ("worker_0069041", "orig", "worker_0069041", "-", "o"),
            ("ref_all", "orig", "ref_all", "-", "s"),
            ("ref_all", "e50", "ref_all", ":", "^"),
            ("ref_all", "e16", "ref_all", "--", "v"),
        ]
        legn = {
            ("worker_0069041", "orig"): "0069041 polygons",
            ("ref_all", "orig"): "well traced",
            ("ref_all", "e50"): "well traced, eroded 50%",
            ("ref_all", "e16"): "well traced, eroded to 1/16",
        }
        cm = {
            ("worker_0069041", "orig"): col["worker_0069041"],
            ("ref_all", "orig"): "#444444",
            ("ref_all", "e50"): "#C0504D",
            ("ref_all", "e16"): "#7A1F1F",
        }
        for g, v, _, ls, mk in spec:
            s = summ[
                (summ.group == g)
                & (summ.variant == v)
                & (summ.target == "AL")
                & (summ.metric == metric)
            ].set_index("bin")
            y = [s.loc[b, "mean"] for b in labs if b in s.index]
            lo = [s.loc[b, "ci_lo"] for b in labs if b in s.index]
            hi = [s.loc[b, "ci_hi"] for b in labs if b in s.index]
            ax.errorbar(
                xs[: len(y)] + 0.07 * (spec.index((g, v, _, ls, mk)) - 1.5),
                y,
                yerr=[np.array(y) - lo, np.array(hi) - y],
                fmt=mk + ls,
                ms=3.5,
                lw=1,
                capsize=1.5,
                color=cm[(g, v)],
                label=legn[(g, v)],
            )
        s = summ[
            (summ.group == "ref_all")
            & (summ.variant == "orig")
            & (summ.target == "ctrl_all")
            & (summ.metric == metric)
        ].set_index("bin")
        ax.plot(
            xs[1:],
            [s.loc[b, "mean"] for b in RING_NAMES],
            "-",
            color=col["ctrl"],
            lw=3,
            alpha=0.5,
            label="control rings (neuropil edges)",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(labs, fontsize=7)
        ax.set_xlabel("distance outside polygon (um)")
        ax.set_ylabel(ylab)
        ax.axhline(0, color="0.85", lw=0.6)

    axb = fig.add_subplot(gs[0, 1])
    ring_panel(axb, "e", "expression AL-likeness")
    axb.legend(
        frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.28)
    )
    axc = fig.add_subplot(gs[0, 2])
    ring_panel(axc, "l", "composition AL-likeness")

    # d: per-section halo (25-100 um) -- the test sections against the references
    axe = fig.add_subplot(gs[1, 0])
    h = halo[(halo.zone == "25-100")]
    cols_e = [
        (
            "well traced\n(24)",
            h[(h.variant == "orig") & h.group.isin(["queen", "worker_0068401"])],
            "#444444",
        ),
        (
            "eroded\n50%",
            h[(h.variant == "e50") & h.group.isin(["queen", "worker_0068401"])],
            "#C0504D",
        ),
        (
            "eroded\nto 1/16",
            h[(h.variant == "e16") & h.group.isin(["queen", "worker_0068401"])],
            "#7A1F1F",
        ),
        (
            "0069041\n(3)",
            h[(h.variant == "orig") & (h.group == "worker_0069041")],
            col["worker_0069041"],
        ),
    ]
    jit = np.random.default_rng(1)
    for i, (_lab, d, c) in enumerate(cols_e):
        axe.scatter(
            i + jit.uniform(-0.15, 0.15, len(d)), d["e"], s=14, color=c, alpha=0.8, lw=0
        )
        axe.plot([i - 0.25, i + 0.25], [d["e"].mean()] * 2, color="k", lw=1.2)
    axe.set_xticks(range(len(cols_e)))
    axe.set_xticklabels([c[0] for c in cols_e], fontsize=7)
    axe.set_ylabel("AL-likeness (25-100 um)")
    axe.axhline(0, color="0.85", lw=0.6)

    # e: prevalence ratio under the hypotheses
    axd = fig.add_subplot(gs[1, 1:])
    s = prev[prev.scenario == "traced_worker_sections_only"]
    sh = prev[prev.scenario == "full_range_as_shipped"].iloc[0]
    pick = lambda b, r: s[(s.baseline == b) & (s.rings == r)].iloc[0]  # noqa: E731
    base_pt = float(s.ratio_as_traced.iloc[0])
    vals = [
        ("shipped", float(sh.ratio_as_traced), None, "0.6"),
        ("traced\nsections", base_pt, None, "#444444"),
    ]
    for lab, b, r, c in (
        (
            "+ edge-\ncalibrated\n0-50",
            "well_traced_0068401_edge",
            "0-50",
            col["worker_0069041"],
        ),
        ("+ edge-\ncalibrated\n0-200", "well_traced_0068401_edge", "0-200", "#7FB5BC"),
        ("+ control\nrings\n0-200", "neuropil_edge_control", "0-200", "#CFCFCF"),
    ):
        row = pick(b, r)
        vals.append(
            (
                lab,
                float(row.ratio_with_recovery),
                (float(row.ratio_lo), float(row.ratio_hi)),
                c,
            )
        )
    axd.bar(
        range(len(vals)), [v[1] for v in vals], color=[v[3] for v in vals], width=0.65
    )
    for i, v in enumerate(vals):
        if v[2]:
            axd.errorbar(
                i,
                v[1],
                yerr=[[v[1] - v[2][0]], [v[2][1] - v[1]]],
                color="k",
                lw=0.8,
                capsize=2,
            )
        axd.text(
            i,
            (v[2][1] if v[2] else v[1]) + 0.05,
            f"{v[1]:.2f}",
            ha="center",
            fontsize=7,
        )
    axd.axhline(1, color="0.6", lw=0.6, ls=":")
    axd.set_xticks(range(len(vals)))
    axd.set_xticklabels([v[0] for v in vals], fontsize=6.5)
    axd.set_ylabel("queen : worker AL prevalence")
    axes = (ax, axb, axc, axe, axd)
    for a in axes:
        for s_ in ("top", "right"):
            a.spines[s_].set_visible(False)
    for a, L in zip(axes, "abcde", strict=False):
        a.text(-0.28, 1.06, L, transform=a.transAxes, fontsize=11, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{fname}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels = label_table()
    print("collecting crops ...", flush=True)
    import pickle

    cache = CACHE / "collected.pkl"
    if cache.exists():
        with open(cache, "rb") as fh:
            cells, X, bg, secs = pickle.load(fh)
    else:
        cells, X, bg, secs = collect(labels)
        with open(cache, "wb") as fh:
            pickle.dump((cells, X, bg, secs), fh, protocol=4)
    secs.to_csv(OUT / "sections_tracing.csv", index=False)

    # the in-polygon counts must reproduce the atlas's counts per crop
    crop_table = pd.read_csv(CROP_TABLE, dtype={"slide": str})
    mine = cells[cells.d_al < 0].groupby("slice_name").size()
    ref = crop_table.set_index("slice_name")["n_al"]
    j = pd.concat([mine.rename("mine"), ref.rename("shipped")], axis=1).fillna(0)
    j = j[(j.mine > 0) | (j.shipped > 0)].join(
        crop_table.set_index("slice_name")["al_traced"]
    )
    bad = j[(j.mine != j.shipped)]
    print(f"in-polygon cells vs shipped n_al: {len(j)} crops, {len(bad)} differ")
    j.to_csv(OUT / "check_inpolygon_vs_shipped.csv")
    if len(bad):
        raise SystemExit(
            f"polygon membership does not reproduce shipped AL counts:\n{bad}"
        )

    traced = secs[secs.al_traced]
    sec_ids = lambda m: list(traced.loc[m, "section_id"])  # noqa: E731
    groups = {
        "queen": sec_ids(traced.brain == "queen"),
        "worker_0068401": sec_ids(traced.slide == REF_SLIDE_WORKER),
        "worker_0069041": sec_ids(traced.slide == TEST_SLIDE),
    }
    groups["ref_all"] = groups["queen"] + groups["worker_0068401"]
    print({k: len(v) for k, v in groups.items()}, flush=True)

    for ref_mode in ("same_brain", "pooled"):
        print(f"scoring ({ref_mode}) ...", flush=True)
        sc, Z, top_label = score_cells(cells, X, bg, labels, ref_mode)
        long = build_long(cells, sc, RING_EDGES, RING_NAMES, signed_inside=True)
        fine_names = [
            f"{a}:{b}" for a, b in zip(FINE_EDGES[:-1], FINE_EDGES[1:], strict=True)
        ]
        long_fine = build_long(cells, sc, FINE_EDGES, fine_names, signed_inside=False)
        # fine bins: signed edges apply to AL only (negative = inside); the
        # control polygons were set to inf inside, so they only fill d > 0 bins
        long.to_csv(OUT / f"ring_sufficient_stats_{ref_mode}.csv", index=False)
        summ = summarise(long, groups, "rings")
        summ.to_csv(OUT / f"ring_summary_{ref_mode}.csv", index=False)
        if ref_mode != "same_brain":
            continue
        fine_s = summarise(
            long_fine,
            {
                k: groups[k]
                for k in ("queen", "worker_0068401", "worker_0069041", "ref_all")
            },
            "fine",
        )
        fine_s.to_csv(OUT / "distance_curve_summary_same_brain.csv", index=False)

        # signatures for the ring cosine: mean z-profile of the reference AL
        # cells of the same brain (all reference sections for ref_all)
        inside = (cells.d_al < 0).to_numpy()
        src_sections = {
            "queen": groups["queen"],
            "worker_0068401": groups["worker_0068401"],
            "worker_0069041": groups["worker_0068401"],
            "ref_all": groups["ref_all"],
        }
        lsig = {
            g: Z[cells.section_id.isin(ids).to_numpy() & inside].mean(0)
            for g, ids in src_sections.items()
        }
        cos = ring_cosine(cells, Z, {k: groups[k] for k in lsig}, lsig)
        cos.to_csv(OUT / "ring_cosine_matched_n.csv", index=False)

        prev, detail, meta = prevalence(long, groups, crop_table)
        prev.to_csv(OUT / "prevalence_under_hypotheses.csv", index=False)
        detail.to_csv(OUT / "prevalence_ring_fractions.csv", index=False)
        pd.Series(meta).to_csv(OUT / "prevalence_deficit_summary.csv", header=["value"])
        det = detection_control(cells, long, groups)
        det.to_csv(OUT / "detection_control_eroded_polygons.csv", index=False)
        halo = halo_tables(long, secs)
        halo.to_csv(OUT / "halo_per_section.csv", index=False)
        hs = halo_summary(halo)
        hs.to_csv(OUT / "halo_summary.csv", index=False)
        pd.set_option("display.width", 250)
        print(prev[prev.scenario == "traced_worker_sections_only"].round(3).to_string())
        print(detail.round(3).to_string())
        print(meta)
        print(det.round(3).to_string())
        print(hs[hs.metric == "e"].round(3).to_string())
        # excluded-by-other-region bookkeeping
        ex = long[long.target == "AL"].merge(
            secs[["section_id", "slide", "brain"]], on="section_id"
        )
        ex["group"] = np.where(ex.brain == "queen", "queen", "worker_" + ex.slide)
        ex.groupby(["group", "variant", "bin"])[["n", "n_excl_other"]].sum().assign(
            frac_excluded=lambda d: d.n_excl_other / (d.n + d.n_excl_other)
        ).to_csv(OUT / "ring_exclusion_other_regions.csv")
        figure(summ, long_fine, groups, prev, halo, "fig_al_polygon_adequacy")
        print("top AL label:", top_label)
    print("done", OUT)


if __name__ == "__main__":
    main()
