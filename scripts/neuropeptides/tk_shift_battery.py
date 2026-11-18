"""The Tk compartment-shift battery (specificity of the tachykinin compartment
shift) on the nine-region Kenyon cell flags.

The ``in_kc_{ic,nc,oc}`` columns stored in ``combined_clustered.h5ad`` give
one set of Kenyon-compartment cells. The headline queen / worker fold uses
the nine-region flags from ``attach_region_flags`` (a slightly different
set), joined on (slice_name, cell_id). This computes every quantity in the
battery both ways: the nine-region values, and the stored-flag values
for comparison.

Quantities (per caste unless noted), for flags in {stored, nine}:
  fold                Kenyon-vs-rest Tk detection fold (fraction of cells with
                      Tk > 0 inside the Kenyon compartments over outside;
                      fig_neuropeptide_spatial._kc_enrichment)
  depth_ratio         mean total_counts of Kenyon-compartment cells over
                      non-Kenyon cells
  band_fold           fold restricted to cells whose total_counts lie in the
                      band common to both castes: from the larger of the two
                      castes' 10th percentiles to the smaller of their 90th
  bleed_fold          fold after dropping non-Kenyon cells within 50 um of a
                      Kenyon-compartment cell of the same section (raw per-crop
                      coordinates in micrometres, joined on (slice_name,
                      cell_id)); bleed_fold_aligned50 is the same with a radius
                      of 50 units in the aligned atlas frame (about 10.6 um),
                      for comparison only
  det_in, mean_in, mean_out, prevalence   Tk detection inside the compartments,
                      mean log-normalised Tk inside/outside, whole-brain detection;
                      the same means for sNPF
  section_*           per-section fold over sections with >= 20 cells on each
                      side of the Kenyon boundary: min / max / median / n, and
                      whether every queen section is below every worker section
  panel_*             the panel-wide specificity ranking of
                      tk_detection_sensitivity.py: Tk and sNPF ranks on
                      log(fold_worker / fold_queen), Tk z against the 99 other
                      genes, and the decile of queen non-Kenyon detection

Inputs
  paths.SPATIAL_COMBINED (expression in .X, total_counts, slice_name, cell_id,
  stored flags; read through h5py, positionally, no label column), the
  nine-region flags table (bimp_st.nine_region_common.attach_region_flags),
  and the per-crop h5ads next to paths.ALIGNED_QUEEN / ALIGNED_WORKER for raw
  coordinates.

Outputs (results/handoff/tk_battery_nine_region/)
  summary.csv        name, flags, caste, value
  per_section.csv    per-section fold, both flag sets
  per_gene.csv       panel-wide table, both flag sets

Run (about 10 min, ~60 GB)
    OPENBLAS_NUM_THREADS=4 PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/neuropeptides/tk_shift_battery.py
"""

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem
from scipy.spatial import cKDTree

from bimp_st.nine_region_common import attach_region_flags, kc_mask, traced_mask
from bimp_st.paths import ALIGNED_QUEEN, ALIGNED_WORKER, RESULTS, SPATIAL_COMBINED

OUT = RESULTS / "handoff" / "tk_battery_nine_region"
TK = "BIMP_vNP3.1_07029"
SNPF = "BIMP_vNP3.1_01359"
CROP = {"queen": ALIGNED_QUEEN.parent, "worker": ALIGNED_WORKER.parent}
KC = ("kc_ic", "kc_nc", "kc_oc")
BLEED_UM = 50.0
MIN_PER_ARM = 20
CASTES = ("queen", "worker")


class _Shim:
    def __init__(self, obs):
        self.obs = obs


def fold(det, m, kc):
    return float(det[m & kc].mean() / det[m & ~kc].mean())


def raw_xy(obs):
    xy = np.full((len(obs), 2), np.nan)
    sl = obs["slice_name"].astype(str).to_numpy()
    cid = obs["cell_id"].astype(str).to_numpy()
    br = obs["brain"].astype(str).to_numpy()
    for s in np.unique(sl):
        idx = np.flatnonzero(sl == s)
        with h5py.File(CROP[br[idx[0]]] / f"{s}.h5ad", "r") as h:
            c2 = np.asarray(read_elem(h["obs"]["cell_id"])).astype(str)
            raw = h["obsm/spatial"][:].astype(float)
        p2 = pd.Series(np.arange(len(c2)), index=c2)
        p2 = p2[~p2.index.duplicated()]
        ok = pd.Index(cid[idx]).isin(p2.index)
        xy[idx[ok]] = raw[p2[cid[idx][ok]].to_numpy()]
    return xy


def near_kc(coords, sl, kc, radius):
    near = np.zeros(len(sl), bool)
    for s in np.unique(sl):
        ms = sl == s
        k = ms & kc & np.isfinite(coords[:, 0])
        o = np.flatnonzero(ms & ~kc & np.isfinite(coords[:, 0]))
        if not k.any() or not len(o):
            continue
        d, _ = cKDTree(coords[k]).query(coords[o], k=1)
        near[o[d < radius]] = True
    return near


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with h5py.File(SPATIAL_COMBINED, "r") as f:
        obs = read_elem(f["obs"])
        X = sp.csc_matrix(read_elem(f["X"]))
        names = pd.Index(read_elem(f["var"]).index.astype(str))
        aligned = f["obsm"]["spatial_aligned"][:]
    tki = int(np.flatnonzero(names.str.startswith(TK))[0])
    sni = int(np.flatnonzero(names.str.startswith(SNPF))[0])
    tk = np.asarray(X[:, tki].todense()).ravel()
    snpf = np.asarray(X[:, sni].todense()).ravel()
    det_tk = tk > 0
    brain = obs["brain"].astype(str).to_numpy()
    sl = obs["slice_name"].astype(str).to_numpy()
    zl = obs["z_level"].to_numpy(float)
    tc = obs["total_counts"].to_numpy(float)

    flags = {"stored": np.zeros(len(obs), bool)}
    for r in KC:
        flags["stored"] |= obs[f"in_{r}"].to_numpy().astype(bool)
    d = obs[["slice_name", "cell_id"]].copy()
    attach_region_flags(_Shim(d), regions=list(KC))
    # Untraced is NA, not False: traced_mask raises if any KC flag is NA.
    flags["nine"] = traced_mask(kc_mask(d))

    xy = raw_xy(obs)
    rows = [("n_cells_no_raw_xy", "both", "both", int(np.isnan(xy[:, 0]).sum()))]
    q10 = [np.percentile(tc[brain == b], [10, 90]) for b in CASTES]
    lo, hi = max(q10[0][0], q10[1][0]), min(q10[0][1], q10[1][1])
    band = (tc >= lo) & (tc <= hi)
    rows += [
        ("band_lo_total_counts", "both", "both", lo),
        ("band_hi_total_counts", "both", "both", hi),
    ]
    sec_rows, gene_rows = [], []
    Xd = (X > 0).astype(np.float32).tocsc()
    for fl, kc in flags.items():
        rows.append(("n_kc_cells", fl, "both", int(kc.sum())))
        near_um = near_kc(xy, sl, kc, BLEED_UM)
        near_al = near_kc(aligned, sl, kc, BLEED_UM)
        for b in CASTES:
            m = brain == b
            rows += [
                ("fold", fl, b, fold(det_tk, m, kc)),
                ("depth_ratio", fl, b, tc[m & kc].mean() / tc[m & ~kc].mean()),
                ("band_fold", fl, b, fold(det_tk, m & band, kc)),
                ("bleed_fold", fl, b, fold(det_tk, m & ~near_um, kc)),
                ("bleed_fold_aligned50", fl, b, fold(det_tk, m & ~near_al, kc)),
                ("bleed_n_dropped", fl, b, int((m & near_um).sum())),
                ("det_in", fl, b, float(det_tk[m & kc].mean())),
                ("mean_in", fl, b, float(tk[m & kc].mean())),
                ("mean_out", fl, b, float(tk[m & ~kc].mean())),
                ("prevalence_pct", fl, b, 100 * float(det_tk[m].mean())),
                ("snpf_mean_in", fl, b, float(snpf[m & kc].mean())),
                ("snpf_mean_out", fl, b, float(snpf[m & ~kc].mean())),
                ("tk_out_pos_cells", fl, b, int((m & ~kc & det_tk).sum())),
            ]
            for s in sorted(set(sl[m]), key=lambda s: zl[sl == s][0]):
                ms = m & (sl == s)
                n_kc, n_out = int((ms & kc).sum()), int((ms & ~kc).sum())
                if n_kc < MIN_PER_ARM or n_out < MIN_PER_ARM:
                    continue
                sec_rows.append(
                    dict(
                        flags=fl,
                        brain=b,
                        slice_name=s,
                        z_level=float(zl[ms][0]),
                        n_kc=n_kc,
                        n_out=n_out,
                        fold=fold(det_tk, ms, kc),
                    )
                )
        sec = pd.DataFrame([r for r in sec_rows if r["flags"] == fl])
        for b in CASTES:
            v = sec[sec.brain == b]["fold"]
            rows += [
                ("section_min", fl, b, v.min()),
                ("section_max", fl, b, v.max()),
                ("section_median", fl, b, v.median()),
                ("section_n", fl, b, len(v)),
            ]
        rows.append(
            (
                "section_complete_separation",
                fl,
                "both",
                int(
                    sec[sec.brain == "queen"].fold.max()
                    < sec[sec.brain == "worker"].fold.min()
                ),
            )
        )

        # Panel-wide specificity (tk_detection_sensitivity.py logic).
        def det(mask):
            return np.asarray(Xd[np.flatnonzero(mask)].mean(0)).ravel()

        q, w = brain == "queen", brain == "worker"
        g = pd.DataFrame(
            {
                "gene": names,
                "det_in_queen": det(q & kc),
                "det_out_queen": det(q & ~kc),
                "det_in_worker": det(w & kc),
                "det_out_worker": det(w & ~kc),
            }
        )
        g["fold_queen"] = g.det_in_queen / g.det_out_queen
        g["fold_worker"] = g.det_in_worker / g.det_out_worker
        g["shift"] = np.log(g.fold_worker / g.fold_queen)
        g["d_out_pp"] = 100 * (g.det_out_worker - g.det_out_queen)
        g["decile_out_queen"] = pd.qcut(
            g.det_out_queen.rank(method="first"), 10, labels=False
        )
        g["is_tk"] = np.arange(len(g)) == tki
        g["is_snpf"] = np.arange(len(g)) == sni
        g["flags"] = fl
        gene_rows.append(g)
        shift = g["shift"].replace([np.inf, -np.inf], np.nan)
        rank = shift.rank(ascending=False)
        null = g[shift.notna() & ~g.is_tk & ~g.is_snpf]
        t = g.loc[tki]
        dec = g[(g.decile_out_queen == t.decile_out_queen) & ~g.is_tk]
        rows += [
            ("panel_tk_rank", fl, "both", float(rank[tki])),
            ("panel_snpf_rank", fl, "both", float(rank[sni])),
            ("panel_n_null", fl, "both", len(null)),
            (
                "panel_tk_z",
                fl,
                "both",
                float((t["shift"] - null["shift"].mean()) / null["shift"].std(ddof=1)),
            ),
            ("panel_tk_d_out_pp", fl, "both", float(t.d_out_pp)),
            (
                "panel_decile_min_det_out_queen_pct",
                fl,
                "both",
                100 * dec.det_out_queen.min(),
            ),
            (
                "panel_decile_max_det_out_queen_pct",
                fl,
                "both",
                100 * dec.det_out_queen.max(),
            ),
            ("panel_decile_median_d_out_pp", fl, "both", float(dec.d_out_pp.median())),
        ]
    pd.DataFrame(sec_rows).to_csv(OUT / "per_section.csv", index=False)
    pd.concat(gene_rows).to_csv(OUT / "per_gene.csv", index=False)
    s = pd.DataFrame(rows, columns=["name", "flags", "caste", "value"])
    s.to_csv(OUT / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(
        s.pivot_table(
            index="name", columns=["flags", "caste"], values="value", aggfunc="first"
        ).to_string()
    )


if __name__ == "__main__":
    main()
