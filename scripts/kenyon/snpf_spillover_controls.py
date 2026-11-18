"""Transcript-level spillover controls for the queen sNPF zone
(`fig:kc_subtypes` panel I).

The queen excess of sNPF detection in Large-type compartment cells near
the Small-type core could come from transcripts assigned across the core
boundary. This recounts transcripts per cell from the Xenium transcript
tables and repeats the caste comparison three ways:

1. nuclear transcripts only (``overlaps_nucleus == 1``, cells with a nucleus);
2. excluding Large-type cells within 10 or 20 um of any Small-type cell;
3. control genes enriched in the Small-type core (the top panel genes by
   core / Large-type detection ratio among genes detected in >= 15% of core
   cells), whose detection in Large-type cells by distance from the core
   shows what boundary bleed looks like.

It also tabulates Oamb in Large-type cells by distance from the core after
excluding cells within 30 um of the Class II compartment.

Cells and compartments are those of the Large-type subcluster projection
`fig:kc_subtypes` reads
(``results/tangram_ltKC_cb/ltKC_CB_spatial_subclusters_dsource.h5ad``):
Small-type core = ``in_kc_ic``, Large-type = ``in_kc_nc & ~in_kc_ic``, Class II
= ``in_kc_oc`` and neither. Distances are in raw per-crop section coordinates
(micrometres), joined on (slice_name, cell_id), to the nearest core cell of
the same crop. Transcripts are kept at quality value >= 20; detection is >= 1
transcript. The queen-to-worker ratio is the ratio of pooled detection
fractions over Large-type cells in a distance window.

Outputs (results/handoff/snpf_spillover/)
  bins_by_distance.csv   detection % per gene, mode (all / nuclear),
                         distance bin and caste
  summary.csv            per gene, mode and exclusion distance: per-section
                         median near the core and the within / beyond 120 um
                         queen-to-worker ratios
  control_ranges.csv     detection of the core-enriched controls under
                         several definitions of "adjacent"
  control_ranges_named.csv  min / max of the same over the three named controls
  oamb_excl_class2.csv   Oamb bins after excluding cells near Class II
  genes.json             the genes used

Run (about 20 min; reads 32 transcript tables)
    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/kenyon/snpf_spillover_controls.py
"""

import json
import re

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from anndata.io import read_elem
from scipy.spatial import cKDTree

from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.paths import (
    ALIGNED_QUEEN,
    ALIGNED_WORKER,
    PANEL_CROSSWALK,  # noqa: E402
    REPO,
    RESULTS,
)

OUT = RESULTS / "handoff" / "snpf_spillover"
LTKC = RESULTS / "tangram_ltKC_cb" / "ltKC_CB_spatial_subclusters_dsource.h5ad"
RUN = {
    "queen": REPO / "data" / "20251204__194847___20251204",
    "worker": REPO / "data" / "20260115__193720__20260115",
}
CROP = {"queen": ALIGNED_QUEEN.parent, "worker": ALIGNED_WORKER.parent}
MAPPING = PANEL_CROSSWALK
FOCAL = {"sNPF": "bimp_v4.3_005730", "Oamb": "bimp_v4.3_009024"}
N_CORE_CONTROLS = 5
# The three reported controls (the other two picked by the
# rule are broadly expressed in Large-type cells and so are not clean bleed
# probes): octopamine receptor 009612, Nmdar2, 5-HT2A.
NAMED_CONTROLS = ["bimp_v4.3_009612", "bimp_v4.3_003250", "bimp_v4.3_003007"]
BINS = [0, 10, 20, 30, 60, 120, np.inf]
CASTES = ("queen", "worker")


def load_cells():
    L = ad.read_h5ad(LTKC, backed="r")
    o = L.obs[
        [
            "cell_id",
            "slice_name",
            "brain",
            "in_kc_ic",
            "in_kc_nc",
            "in_kc_oc",
            "nucleus_area",
        ]
    ].copy()
    for c in ("cell_id", "slice_name", "brain"):
        o[c] = o[c].astype(str)
    X = L.X[:]
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    var = [v.lower() for v in L.var_names]
    ic = o.in_kc_ic.to_numpy().astype(bool)
    nc = o.in_kc_nc.to_numpy().astype(bool) & ~ic
    oc = o.in_kc_oc.to_numpy().astype(bool) & ~ic & ~nc
    return o, X > 0, var, ic, nc, oc


def pick_genes(det, var, ic, nc):
    dI, dN = det[ic].mean(0), det[nc].mean(0)
    sel = pd.DataFrame(dict(gene=var, det_core=dI, det_large=dN))
    sel["log2_core_over_large"] = np.log2((dI + 1e-3) / (dN + 1e-3))
    ctrl = (
        sel[(sel.det_core >= 0.15) & ~sel.gene.isin(FOCAL.values())]
        .sort_values("log2_core_over_large", ascending=False)
        .head(N_CORE_CONTROLS)
    )
    genes = dict(FOCAL)
    for i, g in enumerate(ctrl.gene, 1):
        genes[f"core_control_{i}"] = g
    return genes, sel


def count_transcripts(o, genes, ic, nc, oc):
    mp = read_panel_crosswalk()
    np2v = {
        str(a).strip(): str(b).strip().lower()
        for a, b in zip(mp.np31_id, mp.v43_id, strict=True)
    }

    def to_v43(f):
        for c in f.split("|"):
            if c.strip() in np2v:
                return np2v[c.strip()]
        return f.lower()

    inv = {v: k for k, v in genes.items()}
    G = list(genes)
    gi = {g: i for i, g in enumerate(G)}
    n = len(o)
    CA = np.zeros((n, len(G)), np.int32)
    CN = np.zeros((n, len(G)), np.int32)
    xy = np.full((n, 2), np.nan)
    d_core = np.full(n, np.nan)
    d_oc = np.full(n, np.nan)
    groups = o.groupby(["brain", "slice_name"]).indices
    regions = {}
    for (b, sl), idx in groups.items():
        reg = re.match(r"(output-.*?__\d{8}__\d{6})", sl).group(1)
        regions.setdefault((b, reg), []).append((sl, idx))
    for (b, reg), lst in sorted(regions.items()):
        allidx = np.concatenate([i for _, i in lst])
        pos = pd.Series(allidx, index=o.cell_id.to_numpy()[allidx])
        assert pos.index.is_unique, (b, reg)
        f = RUN[b] / reg / "transcripts.parquet"
        feats = (
            pq.read_table(f, columns=["feature_name"]).column(0).unique().to_pylist()
        )
        want = [x for x in feats if to_v43(x) in inv]
        t = pq.read_table(
            f,
            columns=["cell_id", "overlaps_nucleus", "feature_name", "qv"],
            filters=[("feature_name", "in", want), ("qv", ">=", 20)],
        ).to_pandas()
        t = t[t.cell_id.isin(pos.index)]
        g = t.feature_name.astype(str).map(lambda x: gi[inv[to_v43(x)]]).to_numpy()
        row = pos.loc[t.cell_id.to_numpy()].to_numpy()
        np.add.at(CA, (row, g), 1)
        m = t.overlaps_nucleus.to_numpy() == 1
        np.add.at(CN, (row[m], g[m]), 1)
        for sl, idx in lst:
            with h5py.File(CROP[b] / f"{sl}.h5ad", "r") as h:
                c2 = np.asarray(read_elem(h["obs"]["cell_id"])).astype(str)
                raw = h["obsm/spatial"][:].astype(float)
            p2 = pd.Series(np.arange(len(c2)), index=c2)
            ids = o.cell_id.to_numpy()[idx]
            ok = pd.Index(ids).isin(p2.index)
            xy[idx[ok]] = raw[p2[ids[ok]].to_numpy()]
            here = np.zeros(n, bool)
            here[idx] = True
            here &= np.isfinite(xy[:, 0])
            if (here & ic).any() and (here & nc).any():
                d_core[here & nc] = cKDTree(xy[here & ic]).query(xy[here & nc])[0]
            if (here & oc).any() and (here & nc).any():
                d_oc[here & nc] = cKDTree(xy[here & oc]).query(xy[here & nc])[0]
        print(b, reg, len(t), flush=True)
    return G, gi, CA, CN, d_core, d_oc


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    o, det, var, ic, nc, oc = load_cells()
    genes, sel = pick_genes(det, var, ic, nc)
    json.dump(genes, open(OUT / "genes.json", "w"), indent=1)
    G, gi, CA, CN, dist, d_oc = count_transcripts(o, genes, ic, nc, oc)

    j = var.index(FOCAL["sNPF"])
    agree = float(((CA[:, gi["sNPF"]] > 0) == det[:, j]).mean())
    print("sNPF detection agreement, transcripts vs object:", agree, flush=True)
    assert agree > 0.99

    brain = o.brain.to_numpy()
    sl = o.slice_name.to_numpy()
    hasnuc = np.isfinite(o.nucleus_area.to_numpy(float)) & (
        o.nucleus_area.to_numpy(float) > 0
    )
    use = nc & np.isfinite(dist)

    def detected(g, mode):
        return (CA if mode == "all" else CN)[:, gi[g]] > 0

    def base(mode):
        return use & (hasnuc if mode == "nuc" else True)

    def binrows(g, mode, d, b0, tag):
        rows = []
        for lo, hi in zip(BINS[:-1], BINS[1:], strict=True):
            r = dict(gene=g, mode=mode, set=tag, bin_lo=lo, bin_hi=hi)
            for c in CASTES:
                m = b0 & (brain == c) & (d >= lo) & (d < hi)
                r[f"n_{c}"] = int(m.sum())
                r[f"pct_{c}"] = 100 * detected(g, mode)[m].mean()
            rows.append(r)
        return rows

    T = []
    for g in G:
        for mode in ("all", "nuc"):
            T += binrows(g, mode, dist, base(mode), "large_type_vs_core_distance")
    T = pd.DataFrame(T)
    T.to_csv(OUT / "bins_by_distance.csv", index=False)

    S = []
    for g in G:
        for mode in ("all", "nuc"):
            a_ = detected(g, mode)
            for ex in (0, 10, 20):
                hi = 30 if ex < 20 else 40
                r = dict(
                    gene=g, mode=mode, exclude_within_um=ex, near_window=f"{ex}-{hi}"
                )
                for c in CASTES:
                    v = []
                    m0 = base(mode) & (brain == c) & (dist >= ex) & (dist < hi)
                    for s in np.unique(sl[m0]):
                        mm = m0 & (sl == s)
                        if mm.sum() > 100:
                            v.append(100 * a_[mm].mean())
                    r[f"near_median_{c}"] = float(np.median(v)) if v else np.nan
                    r[f"n_sections_{c}"] = len(v)
                win = base(mode) & (dist >= ex) & (dist < 120)
                far = base(mode) & (dist >= 120)
                q, w = (
                    a_[win & (brain == "queen")].mean(),
                    a_[win & (brain == "worker")].mean(),
                )
                r["pct_queen_within120"], r["pct_worker_within120"] = 100 * q, 100 * w
                r["ratio_within120"] = q / w
                r["ratio_beyond120"] = (
                    a_[far & (brain == "queen")].mean()
                    / a_[far & (brain == "worker")].mean()
                )
                S.append(r)
    S = pd.DataFrame(S)
    S.to_csv(OUT / "summary.csv", index=False)

    # Detection of the core-enriched controls in adjacent Large-type cells,
    # under the definitions of "adjacent" a reader could mean.
    ctrl = [g for g in G if g.startswith("core_control_")]
    C = []
    for g in ctrl:
        for mode in ("all", "nuc"):
            a_ = detected(g, mode)
            for lo, hi in ((0, 10), (0, 20), (0, 30)):
                for c in CASTES:
                    m = base(mode) & (brain == c) & (dist >= lo) & (dist < hi)
                    C.append(
                        dict(
                            gene=g,
                            v43_id=genes[g],
                            mode=mode,
                            definition=f"pooled {lo}-{hi} um",
                            caste=c,
                            pct=100 * a_[m].mean(),
                        )
                    )
    C = pd.DataFrame(C)
    tb = T[T.gene.isin(ctrl) & (T.bin_hi <= 30)]
    for mode in ("all", "nuc"):
        for g in ctrl:
            x = tb[(tb.gene == g) & (tb["mode"] == mode)]
            for c in CASTES:
                C = pd.concat(
                    [
                        C,
                        pd.DataFrame(
                            [
                                dict(
                                    gene=g,
                                    v43_id=genes[g],
                                    mode=mode,
                                    definition="per 10-um bin, 0-30 um, min",
                                    caste=c,
                                    pct=x[f"pct_{c}"].min(),
                                ),
                                dict(
                                    gene=g,
                                    v43_id=genes[g],
                                    mode=mode,
                                    definition="per 10-um bin, 0-30 um, max",
                                    caste=c,
                                    pct=x[f"pct_{c}"].max(),
                                ),
                            ]
                        ),
                    ]
                )
    C = C.merge(
        sel[["gene", "det_core", "det_large"]].rename(columns={"gene": "v43_id"}),
        on="v43_id",
        how="left",
    )
    C.to_csv(OUT / "control_ranges.csv", index=False)
    named = C[C.v43_id.isin(NAMED_CONTROLS)]
    R = []
    for (mode, d), x in named.groupby(["mode", "definition"]):
        R.append(
            dict(
                mode=mode,
                definition=d,
                n_genes=x.v43_id.nunique(),
                min_pct=x.pct.min(),
                max_pct=x.pct.max(),
            )
        )
    pd.DataFrame(R).to_csv(OUT / "control_ranges_named.csv", index=False)
    print(pd.DataFrame(R).round(2).to_string(index=False))

    use2 = use & ~(d_oc < 30)
    oamb_rows = []
    for mode in ("all", "nuc"):
        oamb_rows += binrows(
            "Oamb",
            mode,
            dist,
            use2 & (hasnuc if mode == "nuc" else True),
            "excluding <30 um from Class II",
        )
    pd.DataFrame(oamb_rows).to_csv(OUT / "oamb_excl_class2.csv", index=False)

    pd.set_option("display.width", 220)
    print(S[S.gene == "sNPF"].round(3).to_string(index=False))
    print(T[T.gene.isin(ctrl) & (T["mode"] == "all")].round(2).to_string(index=False))
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
