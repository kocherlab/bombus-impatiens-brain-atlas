"""Is the queen sNPF excess near the Small-type core (`fig:kc_subtypes` I) just
Middle-type KCs?

Reuses fig_kc_subtypes._subzone_data() / panel_distance()._fold() definitions
verbatim (Large-type compartment = in_kc_nc & ~in_kc_ic on the ltKC h5ad;
distance = raw per-crop um to nearest in_kc_ic cell of the same slice;
detection = X > 0).  Middle-type probability = restricted 5-population
reclassification (kc5pop_spatial_allKC_dsource.h5ad, tangram_ct_pred
renormalised over TYPE_ORDER, as load_kc5pop()), joined on
(brain, slice_name, cell_id).

Output: results/handoff/middle_type_control/

Run
    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/kenyon/snpf_middle_type_control.py
"""

import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata.io import read_elem
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, spearmanr

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results/handoff/middle_type_control"
OUT.mkdir(parents=True, exist_ok=True)

LTKC_H5AD = REPO / "results/tangram_ltKC_cb/ltKC_CB_spatial_subclusters_dsource.h5ad"
KC5POP_H5AD = REPO / "results/tangram_ltKC/kc5pop_spatial_allKC_dsource.h5ad"
SN_REF = REPO / "data/sn/BIMP_atlas_qw_CentralBrain.h5ad"
CROP_DIRS = {
    "queen": REPO / "pipelines/wholebrain/queen/data",
    "worker": REPO / "pipelines/wholebrain/worker/data",
}
TYPE_ORDER = [
    "Large-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Middle-type Kenyon cells",
    "Other Kenyon cells",
]
SNPF = "bimp_v4.3_005730"
ZONE = 120.0
BINS = [0, 40, 80, 120, 200, np.inf]
BIN_LAB = ["0-40", "40-80", "80-120", "120-200", ">200"]
rng = np.random.default_rng(0)
N_BOOT = 2000

log = open(OUT / "log.txt", "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log.write(s + "\n")
    log.flush()


# ── 1. kc_subtypes_data._subzone_data(), verbatim logic ────────────────────
ad = sc.read_h5ad(LTKC_H5AD)
o = ad.obs
n = ad.n_obs
xy = np.full((n, 2), np.nan)
for (b, sl), idx in o.groupby(["brain", "slice_name"], observed=True).indices.items():
    with h5py.File(CROP_DIRS[b] / f"{sl}.h5ad", "r") as h:
        cid = np.asarray(read_elem(h["obs"]["cell_id"])).astype(str)
        raw = h["obsm/spatial"][:].astype(float)
    pos = pd.Series(np.arange(len(cid)), index=cid)
    ids = o["cell_id"].values[idx].astype(str)
    ok = pd.Index(ids).isin(pos.index)
    xy[idx[ok]] = raw[pos[ids[ok]].values]
ic = o["in_kc_ic"].values.astype(bool)
nc = o["in_kc_nc"].values.astype(bool) & ~ic
have = np.isfinite(xy[:, 0])
brain = o["brain"].astype(str).values
sl_all = o["slice_name"].astype(str).values
dist = np.full(n, np.nan)
for idx in o.groupby(["brain", "slice_name"], observed=True).indices.values():
    m = np.zeros(n, bool)
    m[idx] = True
    a_, n_ = m & ic & have, m & nc & have
    if a_.any() and n_.any():
        dist[n_] = cKDTree(xy[a_]).query(xy[n_])[0]
X = np.asarray(ad.X)
var = {v.lower(): i for i, v in enumerate(ad.var_names)}
det_snpf = X[:, var[SNPF]] > 0
use = np.isfinite(dist)


def fold(mask, lo, hi):
    q = det_snpf[mask & (brain == "queen") & (dist >= lo) & (dist < hi)]
    w = det_snpf[mask & (brain == "worker") & (dist >= lo) & (dist < hi)]
    return float(q.mean() / w.mean())


say("== baseline reproduction ==")
for b in ("queen", "worker"):
    say(
        f"n_cells_{b}",
        int((use & (brain == b)).sum()),
        "sections",
        len(np.unique(sl_all[use & (brain == b)])),
    )
say("within120", fold(use, 0, ZONE))
say("beyond120", fold(use, ZONE, 1e9))
say("all", fold(use, 0, 1e9))

# sanity: in_kc_* flags baked into the object vs nine-region flags (read-only cache)
try:
    ft = pd.read_parquet(REPO / "results/nine_region/region_flags.parquet")
    k_ft = ft["slice_name"].astype(str) + "||" + ft["cell_id"].astype(str)
    ft = ft.assign(_k=k_ft).drop_duplicates("_k").set_index("_k")
    k = sl_all + "||" + o["cell_id"].astype(str).values
    gi = ft.index.get_indexer(k)
    say("nine-region flags matched", int((gi >= 0).sum()), "/", n)
    for col in ("in_kc_ic", "in_kc_nc"):
        if col in ft.columns:
            v = pd.array(ft[col].to_numpy()[gi[gi >= 0]], dtype="boolean")
            v = np.asarray(v.fillna(False), bool)
            say(
                f"  agreement {col}:",
                float((v == o[col].values.astype(bool)[gi >= 0]).mean()),
            )
except Exception as e:  # noqa: BLE001
    say("nine-region flag comparison skipped:", repr(e))

# ── 2. join restricted 5-pop reclassification on (brain, slice, cell_id) ──
k5 = sc.read_h5ad(KC5POP_H5AD, backed="r")
p = k5.obsm["tangram_ct_pred"]
if isinstance(p, pd.DataFrame):
    say("kc5pop obsm columns:", list(p.columns))
    p = p[TYPE_ORDER].values
else:
    say("kc5pop obsm is ndarray; trusting TYPE_ORDER")
p = np.asarray(p, float)
p = p / p.sum(1, keepdims=True)
o5 = k5.obs
key5 = (
    o5["brain"].astype(str)
    + "|"
    + o5["slice_name"].astype(str)
    + "|"
    + o5["cell_id"].astype(str)
).values
keyL = brain + "|" + sl_all + "|" + o["cell_id"].astype(str).values
say(
    "unique keys kc5pop",
    len(set(key5)) == len(key5),
    "ltkc",
    len(set(keyL)) == len(keyL),
)
gi = pd.Index(key5).get_indexer(keyL)
assert (gi >= 0).all(), "unmatched cells"
P = p[gi]
am = P.argmax(1)
call = o5["kc5pop_call"].astype(str).values[gi]
say(
    "argmax vs stored kc5pop_call agreement:",
    float((np.array(TYPE_ORDER)[am] == call).mean()),
)
pM = P[:, 3]
isMid = am == 3
isLarge = am == 0

# ── 3. snRNA reference: sNPF per population, pairwise markers ───────────────
ref = sc.read_h5ad(SN_REF)
ref = ref[ref.obs["cell_type"].astype(str).isin(TYPE_ORDER)].copy()
say("snRNA KC nuclei:", ref.n_obs)
panel = set(var)
pg = [g for g in ref.var_names if g.lower() in panel]
say("panel genes in reference:", len(pg))
desc = ref.var["description"].astype(str).to_dict()
cnt = ref.layers["counts"]
gidx = {g: i for i, g in enumerate(ref.var_names)}
ref.obs["snpf_det"] = np.asarray(
    (cnt[:, gidx["BIMP_v4.3_005730"]] > 0).todense()
).ravel()
sn_tab = (
    ref.obs.groupby(["cell_type", "sample"], observed=True)["snpf_det"]
    .agg(["mean", "size"])
    .reset_index()
)
sn_tab.to_csv(OUT / "snrna_snpf_by_population_sample.csv", index=False)
say("snRNA sNPF detection by population x caste (mean of per-sample %):")
sn_tab["caste"] = sn_tab["sample"].str.extract(r"_(Queen|Worker)_")[0]
say((sn_tab.groupby(["cell_type", "caste"])["mean"].mean() * 100).round(1).to_string())
say("population sizes per caste:")
say(ref.obs.groupby(["cell_type", "caste"], observed=True).size().unstack().to_string())

rp = ref[:, pg].copy()
res = {}
for grp, refg in (
    ("Middle-type Kenyon cells", "Large-type Kenyon cells"),
    ("Small-type Kenyon cells", "Large-type Kenyon cells"),
):
    sc.tl.rank_genes_groups(
        rp,
        "cell_type",
        groups=[grp],
        reference=refg,
        method="wilcoxon",
        n_genes=len(pg),
    )
    df = sc.get.rank_genes_groups_df(rp, group=grp)
    res[grp] = df.set_index("names")
pw = pd.DataFrame(
    {
        "mid_vs_large_lfc": res["Middle-type Kenyon cells"]["logfoldchanges"],
        "mid_vs_large_padj": res["Middle-type Kenyon cells"]["pvals_adj"],
        "mid_vs_large_score": res["Middle-type Kenyon cells"]["scores"],
        "small_vs_large_lfc": res["Small-type Kenyon cells"]["logfoldchanges"],
    }
)
# detection fractions per population
for t in TYPE_ORDER:
    sub = cnt[(ref.obs["cell_type"] == t).values][:, [gidx[g] for g in pw.index]]
    pw["det_" + t.split()[0].split("-")[0]] = np.asarray((sub > 0).mean(0)).ravel()
pw["description"] = [desc.get(g, "") for g in pw.index]
pw = pw.sort_values("mid_vs_large_score", ascending=False)
pw.to_csv(OUT / "snrna_middle_vs_large_panel_genes.csv")
say("top panel genes Middle vs Large (pairwise Wilcoxon):")
say(pw.head(12).round(3).to_string())
mk = pw[
    (pw.mid_vs_large_lfc > 1)
    & (pw.mid_vs_large_padj < 0.05)
    & (pw.small_vs_large_lfc < 0.5)
    & (pw.index.str.lower() != SNPF)
]
markers = list(mk.index[:4])
if len(markers) < 2:
    markers = list(pw[(pw.mid_vs_large_padj < 0.05)].index[:4])
say("chosen Middle-specific markers:", markers, [desc.get(g) for g in markers])
ovr = ["BIMP_v4.3_006191", "BIMP_v4.3_006836", "BIMP_v4.3_001161", "BIMP_v4.3_002889"]
say("one-vs-rest on-panel markers also tabulated:", ovr)
allm = list(dict.fromkeys(markers + ovr))
det_m = {g: X[:, var[g.lower()]] > 0 for g in allm}

# ── 4. tables by distance bin ───────────────────────────────────────────────
binv = pd.cut(dist, BINS, right=False, labels=BIN_LAB)
rows = []
for b in ("queen", "worker"):
    for lab in BIN_LAB:
        m = use & (brain == b) & (np.asarray(binv) == lab)
        r = dict(
            caste=b,
            bin=lab,
            n=int(m.sum()),
            pMiddle_mean=pM[m].mean(),
            pMiddle_median=np.median(pM[m]),
            argmax_Middle_pct=100 * isMid[m].mean(),
            argmax_Large_pct=100 * isLarge[m].mean(),
            sNPF_det_pct=100 * det_snpf[m].mean(),
        )
        for g in allm:
            r[f"det_{g[-6:]}_pct"] = 100 * det_m[g][m].mean()
        rows.append(r)
bt = pd.DataFrame(rows)
bt.to_csv(OUT / "middle_by_distance_bin.csv", index=False)
say("== by distance bin ==")
say(bt.round(3).to_string())

# ── 5. exclusions & ratios ──────────────────────────────────────────────────
A = use
q10 = np.quantile(pM[A], 0.90)
q25 = np.quantile(pM[A], 0.75)
qc = {
    b: (
        np.quantile(pM[A & (brain == b)], 0.90),
        np.quantile(pM[A & (brain == b)], 0.75),
    )
    for b in ("queen", "worker")
}
say("pooled thresholds p90", q10, "p75", q25, "per-caste", qc)
pc10 = np.where(brain == "queen", qc["queen"][0], qc["worker"][0])
pc25 = np.where(brain == "queen", qc["queen"][1], qc["worker"][1])
keeps = {
    "baseline (shipped)": np.ones(n, bool),
    "excl top10% pMiddle (pooled)": pM < q10,
    "excl top25% pMiddle (pooled)": pM < q25,
    "excl top10% pMiddle (per-caste)": pM < pc10,
    "excl top25% pMiddle (per-caste)": pM < pc25,
    "excl argmax==Middle": ~isMid,
    "only argmax==Large": isLarge,
    "only argmax==Middle": isMid,
}
secs = {b: np.unique(sl_all[A & (brain == b)]) for b in ("queen", "worker")}


def ratio_stats(keep, lo, hi):
    out = {}
    m = A & keep & (dist >= lo) & (dist < hi)
    kq = int(det_snpf[m & (brain == "queen")].sum())
    nq = int((m & (brain == "queen")).sum())
    kw = int(det_snpf[m & (brain == "worker")].sum())
    nw = int((m & (brain == "worker")).sum())
    if nq == 0 or nw == 0 or kw == 0:
        return dict(nq=nq, nw=nw)
    rr = (kq / nq) / (kw / nw)
    se = np.sqrt(1 / kq - 1 / nq + 1 / kw - 1 / nw) if kq else np.nan
    out.update(
        nq=nq,
        nw=nw,
        q_pct=100 * kq / nq,
        w_pct=100 * kw / nw,
        ratio=rr,
        ci_cell_lo=rr * np.exp(-1.96 * se),
        ci_cell_hi=rr * np.exp(1.96 * se),
    )
    # section bootstrap (resample sections within each brain)
    agg = {}
    for b in ("queen", "worker"):
        mb = m & (brain == b)
        s_ = (
            pd.DataFrame({"s": sl_all[mb], "k": det_snpf[mb]})
            .groupby("s")["k"]
            .agg(["sum", "size"])
        )
        s_ = s_.reindex(secs[b], fill_value=0)
        agg[b] = s_.values
    bs = []
    for _ in range(N_BOOT):
        r_ = []
        for b in ("queen", "worker"):
            v = agg[b][rng.integers(0, len(agg[b]), len(agg[b]))]
            r_.append(v[:, 0].sum() / max(v[:, 1].sum(), 1))
        bs.append(r_[0] / r_[1] if r_[1] > 0 else np.nan)
    bs = np.array(bs)
    out.update(
        ci_sec_lo=np.nanpercentile(bs, 2.5), ci_sec_hi=np.nanpercentile(bs, 97.5)
    )
    return out


rows = []
for name, keep in keeps.items():
    for zone, (lo, hi) in (("within120", (0, ZONE)), ("beyond120", (ZONE, 1e9))):
        r = dict(set=name, zone=zone)
        r.update(ratio_stats(keep, lo, hi))
        # fraction excluded per caste in this zone
        for b in ("queen", "worker"):
            mm = A & (brain == b) & (dist >= lo) & (dist < hi)
            r[f"excl_pct_{b}"] = 100 * (1 - keep[mm].mean())
        rows.append(r)
rt = pd.DataFrame(rows)
rt.to_csv(OUT / "snpf_ratio_exclusions.csv", index=False)
say("== sNPF queen/worker ratio ==")
say(rt.round(3).to_string())

# ── 6. is pMiddle itself caste-different in the zone? correlations ─────────
zs = {}
for zone, (lo, hi) in (("within120", (0, ZONE)), ("beyond120", (ZONE, 1e9))):
    mq = A & (brain == "queen") & (dist >= lo) & (dist < hi)
    mw = A & (brain == "worker") & (dist >= lo) & (dist < hi)
    secmed = {
        b: pd.Series(pM[mm]).groupby(sl_all[mm]).mean()
        for b, mm in (("queen", mq), ("worker", mw))
    }
    zs[zone] = dict(
        pM_mean_q=float(pM[mq].mean()),
        pM_mean_w=float(pM[mw].mean()),
        argmaxMid_pct_q=float(100 * isMid[mq].mean()),
        argmaxMid_pct_w=float(100 * isMid[mw].mean()),
        mwu_p_cells=float(mannwhitneyu(pM[mq], pM[mw]).pvalue),
        sec_mean_pM_q_median=float(secmed["queen"].median()),
        sec_mean_pM_w_median=float(secmed["worker"].median()),
        mwu_p_sections=float(mannwhitneyu(secmed["queen"], secmed["worker"]).pvalue),
        spearman_pM_snpf_q=float(spearmanr(pM[mq], det_snpf[mq]).statistic),
        spearman_pM_snpf_w=float(spearmanr(pM[mw], det_snpf[mw]).statistic),
        snpf_det_pct_in_argmaxMid_q=float(100 * det_snpf[mq & isMid].mean()),
        snpf_det_pct_in_argmaxMid_w=float(100 * det_snpf[mw & isMid].mean()),
        snpf_det_pct_in_argmaxLarge_q=float(100 * det_snpf[mq & isLarge].mean()),
        snpf_det_pct_in_argmaxLarge_w=float(100 * det_snpf[mw & isLarge].mean()),
    )
    # counterfactual: how much of the queen excess could Middle composition explain?
    # queen's rate if it had worker's argmax-Middle share, per-class rates kept
    sq_m, sq_o = det_snpf[mq & isMid].mean(), det_snpf[mq & ~isMid].mean()
    fw = isMid[mw].mean()
    zs[zone]["queen_rate_at_worker_mid_share_pct"] = float(
        100 * (fw * sq_m + (1 - fw) * sq_o)
    )
    zs[zone]["worker_rate_pct"] = float(100 * det_snpf[mw].mean())
json.dump(zs, open(OUT / "zone_middle_caste.json", "w"), indent=1)
say("== zone Middle-prob caste comparison ==")
say(json.dumps(zs, indent=1))

# ── 7. figure ───────────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 4, figsize=(16, 3.8))
x = np.arange(len(BIN_LAB))
cols = {"queen": "#C9A227", "worker": "#3B6EA8"}
for b in ("queen", "worker"):
    s = bt[bt.caste == b]
    axs[0].plot(x, s.pMiddle_mean, "o-", c=cols[b], label=b)
    axs[1].plot(x, s.argmax_Middle_pct, "o-", c=cols[b], label=b)
    axs[2].plot(x, s.sNPF_det_pct, "o-", c=cols[b], label=f"{b} sNPF")
    for g, ls in zip(markers[:3], ("--", ":", "-."), strict=False):
        axs[2].plot(
            x, s[f"det_{g[-6:]}_pct"], ls, c=cols[b], alpha=0.7, label=f"{b} {g[-6:]}"
        )
for a, t in zip(
    axs[:3],
    (
        "mean Middle-type prob",
        "% argmax = Middle",
        "% detected (sNPF solid; Middle markers dashed)",
    ),
    strict=True,
):
    a.set_xticks(x, BIN_LAB)
    a.set_xlabel("distance from Small-type core (um)")
    a.set_title(t, fontsize=9)
axs[0].legend(fontsize=7)
axs[2].legend(fontsize=5.5, ncol=2)
w = rt[rt.zone == "within120"].reset_index(drop=True)
bb = rt[rt.zone == "beyond120"].reset_index(drop=True)
yy = np.arange(len(w))
axs[3].errorbar(
    w.ratio,
    yy - 0.12,
    xerr=[w.ratio - w.ci_sec_lo, w.ci_sec_hi - w.ratio],
    fmt="o",
    c="k",
    label="<120 um",
)
axs[3].errorbar(
    bb.ratio,
    yy + 0.12,
    xerr=[bb.ratio - bb.ci_sec_lo, bb.ci_sec_hi - bb.ratio],
    fmt="s",
    c="grey",
    label=">=120 um",
)
axs[3].axvline(1, c="r", lw=0.6)
axs[3].set_yticks(yy, w.set, fontsize=6)
axs[3].invert_yaxis()
axs[3].set_xlabel("queen/worker sNPF detection (section-bootstrap 95% CI)")
axs[3].legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT / "middle_type_check.png", dpi=150)
say("done")
