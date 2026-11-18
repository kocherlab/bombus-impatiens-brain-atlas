#!/usr/bin/env python
"""What changes when the generic optic-lobe "Kenyon cells" reference entry is dropped
from the whole-atlas mapping?

Compares, per seed (42, 43, 44), the canonical run ``combined_clustered_g2_R2_s<seed>``
(entry present) with the rerun ``combined_clustered_r6_noolkc_s<seed>`` (same recipe,
reference ``BIMP_atlas_qw_combined_noOLKC.h5ad``, new tags;
``scripts/identity/run_tangram_remap_no_optic_kc.sh`` recipe):

(a) where the cells that carried the generic label go, per cell and under the cluster
    vote;
(b) the 17-type per-cell composition and the vote-group composition; per-cell own-type
    region accuracy (three leave-section splits, seeds 0-2); Kenyon cell F1 of the
    expression-only vote against the calyx flags; the queen-worker difference over
    within-brain SD per vote group;
(c) the cross-modal panel concordance (rho: mean over cell-type means, types with
    >= 100 cells in both modalities), recomputed with the new labels and with the
    generic nuclei removed from the snRNA side.

Leiden clusters do not depend on the labels, so the cluster-only region accuracy
cannot change and is not recomputed here. The n=3 DESeq2
uses the central brain object, which has no generic nuclei, so it cannot change either
(checked here by counting them).

Writes ``results/round6/noolkc/``.

    PYTHONPATH=src:scripts .venv/bin/python \\
        scripts/identity/drop_generic_optic_kc_entry.py [--skip-concordance]
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
os.chdir(REPO)
from bimp_st.cluster_labels import (
    add_canonical,
    get_canonical,
    get_pooled,
    hierarchical_vote,
)  # noqa: E402
from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.io import read_obs  # noqa: E402
from bimp_st.nine_region_common import STATISTICAL_REGIONS, attach_region_flags  # noqa: E402
from bimp_st.paths import SN_CB, SN_OL, SPATIAL_COMBINED  # noqa: E402

T = REPO / "results/tangram"
OUT = REPO / "results/round6/noolkc"
SEEDS = (42, 43, 44)
GENERIC = "Kenyon cells"
MIXED = "Mixed"


class _Shim:
    def __init__(self, obs):
        self.obs, self.n_obs = obs, len(obs)

    @property
    def obs_names(self):
        return self.obs.index


def labelled(obs: pd.DataFrame) -> pd.DataFrame:
    """Per-cell canonical/pooled labels and the expression-only vote (fine and
    pooled)."""
    o = pd.DataFrame(
        {
            "leiden": obs["leiden"].astype(str).values,
            "slice": obs["slice_name"].astype(str).values,
            "brain": obs["brain"].astype(str).values,
        }
    )
    o["canon"] = [get_canonical(v) for v in obs["tangram_cell_type"].astype(str)]
    o["pooled"] = [get_pooled(v) for v in o["canon"]]
    shim = _Shim(
        obs[["leiden", "tangram_cell_type"]].assign(leiden=obs["leiden"].astype(str))
    )
    tab = hierarchical_vote(
        shim, cluster_col="leiden", label_col="tangram_cell_type", dst="cluster_vote"
    )
    o["vote_fine"] = shim.obs["cluster_vote"].astype(str).values
    o["vote"] = [v if v == MIXED else get_pooled(v) for v in o["vote_fine"]]
    return o, tab.reset_index()


def own_type_region_accuracy(obs: pd.DataFrame, canon: np.ndarray) -> list[float]:
    """Leave-section accuracy of the one-feature (own cell type) model = per-label
    training plurality."""
    w = _Shim(obs.copy())
    attach_region_flags(w, regions=STATISTICAL_REGIONS)
    o = w.obs
    FL = [f"in_{r}" for r in STATISTICAL_REGIONS]
    traced = o["in_al"].notna().to_numpy()
    flags = o[FL].astype(bool).to_numpy()
    keep = traced & (flags.sum(1) == 1)
    y = np.array(FL)[flags.argmax(1)][keep]
    x = np.asarray(canon)[keep]
    g = o["slice_name"].astype(str).to_numpy()[keep]
    accs = []
    for seed in (0, 1, 2):
        tr, te = next(
            GroupShuffleSplit(1, test_size=0.3, random_state=seed).split(x, y, g)
        )
        maj = pd.crosstab(x[tr], y[tr]).idxmax(axis=1)
        pred = (
            pd.Series(x[te])
            .map(maj)
            .fillna(pd.Series(y[tr]).value_counts().idxmax())
            .to_numpy()
        )
        accs.append(float((pred == y[te]).mean()))
    return accs


def kenyon_f1(o: pd.DataFrame, obs: pd.DataFrame) -> float:
    from bimp_st.nine_region_common import kc_mask, traced_mask

    calyx = traced_mask(kc_mask(obs))
    kc = (o["vote_fine"] == "Kenyon cells").to_numpy()
    p, r = calyx[kc].mean(), kc[calyx].mean()
    return float(2 * p * r / (p + r))


def sd_ratios(o: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cls in sorted(set(o["vote"])):
        share = (
            o.assign(hit=(o["vote"] == cls))
            .groupby(["brain", "slice"])
            .hit.mean()
            .reset_index()
        )
        q, w = share[share.brain == "queen"].hit, share[share.brain == "worker"].hit
        sd = np.mean([q.std(ddof=1), w.std(ddof=1)])
        rows.append(
            dict(
                vote_group=cls,
                mean_queen=q.mean(),
                mean_worker=w.mean(),
                diff=abs(q.mean() - w.mean()),
                mean_within_brain_sd=sd,
                ratio=abs(q.mean() - w.mean()) / sd if sd > 0 else np.nan,
            )
        )
    return pd.DataFrame(rows)


def concordance(spatial_labels: dict[str, np.ndarray]) -> pd.DataFrame:
    """rho of mean-over-type-means, Xenium vs snRNA, for each label set (as in
    atlas.si_panel_concordance)."""
    import atlas.si_panel_concordance as C

    panel = read_panel_crosswalk()
    v43 = list(dict.fromkeys(panel["v43_id"].dropna()))
    cb, ol = sc.read_h5ad(SN_CB), sc.read_h5ad(SN_OL)
    n_generic_cb = int((cb.obs["cell_type"].astype(str) == GENERIC).sum())
    for a in (cb, ol):
        add_canonical(a, src="cell_type")
    shared = [g for g in v43 if g in cb.var_names and g in ol.var_names]
    sn = ad.concat(
        [cb[:, shared], ol[:, shared]], join="outer", label="atlas", keys=["cb", "ol"]
    )
    del cb, ol
    is_generic = (sn.obs["cell_type"].astype(str) == GENERIC).to_numpy()
    sp = sc.read_h5ad(SPATIAL_COMBINED)
    m = dict(zip(panel["np31_id"], panel["v43_id"], strict=True))
    keep, ids = [], []
    for n in sp.var_names:
        v = m.get(str(n).split("|")[0])
        if v is not None and v in set(shared):
            keep.append(n)
            ids.append(v)
    sp = sp[:, keep].copy()
    sp.var_names = ids
    sp = sp[:, ~pd.Index(sp.var_names).duplicated()].copy()
    genes = [g for g in shared if g in set(sp.var_names)]
    rows = [
        dict(
            label_set="generic nuclei in the central brain object (DESeq2 input)",
            value=n_generic_cb,
        )
    ]
    for name, lab in spatial_labels.items():
        sp.obs["tangram_cell_type"] = lab
        add_canonical(sp, src=("tangram_cell_type", "cell_type"))
        snx = sn[~is_generic].copy() if name.startswith("noolkc") else sn
        A = C.cpm_by_type(snx, C.LABEL, genes, "sn")
        B = C.cpm_by_type(sp, C.LABEL, genes, "xen")
        cols = [c for c in A.columns if c in B.columns]
        r, p = spearmanr(A[cols].mean(1), B[cols].mean(1))
        rows.append(
            dict(
                label_set=name,
                value=float(r),
                p=float(p),
                n_types=len(cols),
                generic_type_present=GENERIC in cols,
            )
        )
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-concordance", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summ, comp_rows, move_rows, cl_rows = [], [], [], []
    sp_labels = {}
    for s in SEEDS:
        old_p, new_p = (
            T / f"combined_clustered_g2_R2_s{s}_tangram.h5ad",
            T / f"combined_clustered_r6_noolkc_s{s}_tangram.h5ad",
        )
        if not new_p.exists():
            print("missing", new_p, flush=True)
            continue
        ob_o, ob_n = read_obs(old_p), read_obs(new_p)
        assert (
            len(ob_o) == len(ob_n)
            and (
                ob_o["leiden"].astype(str).values == ob_n["leiden"].astype(str).values
            ).all()
        )
        assert (
            ob_o["slice_name"].astype(str).values
            == ob_n["slice_name"].astype(str).values
        ).all()
        lo, to = labelled(ob_o)
        ln, tn = labelled(ob_n)
        n = len(lo)
        gen = (lo["canon"] == GENERIC).to_numpy()
        # (a) where the generic cells go
        for scheme, col in (("per-cell label", "canon"), ("cluster vote", "vote_fine")):
            d = ln.loc[gen, col].value_counts()
            for k, v in d.items():
                move_rows.append(
                    dict(
                        seed=s,
                        scheme=scheme,
                        cells_with_generic_label=int(gen.sum()),
                        new_label=k,
                        n=int(v),
                        pct=100 * v / gen.sum(),
                    )
                )
        oldv = lo["vote_fine"].to_numpy()
        newv = ln["vote_fine"].to_numpy()
        ch = oldv != newv
        cl = to.merge(tn, on="cl", suffixes=("_old", "_new"))
        cl = cl[cl.label_old != cl.label_new]
        cl["seed"] = s
        cl_rows.append(cl)
        # (b) composition
        for scheme, col in (
            ("per-cell canonical-17", "canon"),
            ("cluster vote (fine/pooled names)", "vote_fine"),
            ("cluster vote pooled groups", "vote"),
        ):
            po, pn = lo[col].value_counts(), ln[col].value_counts()
            for k in sorted(set(po.index) | set(pn.index)):
                comp_rows.append(
                    dict(
                        seed=s,
                        scheme=scheme,
                        label=k,
                        n_old=int(po.get(k, 0)),
                        n_new=int(pn.get(k, 0)),
                        pct_old=100 * po.get(k, 0) / n,
                        pct_new=100 * pn.get(k, 0) / n,
                        delta_pp=100 * (pn.get(k, 0) - po.get(k, 0)) / n,
                    )
                )
        acc_o, acc_n = (
            own_type_region_accuracy(ob_o, lo["canon"].to_numpy()),
            own_type_region_accuracy(ob_n, ln["canon"].to_numpy()),
        )
        f1_o, f1_n = kenyon_f1(lo, ob_o), kenyon_f1(ln, ob_n)
        sr_o, sr_n = (
            sd_ratios(lo).set_index("vote_group"),
            sd_ratios(ln).set_index("vote_group"),
        )
        summ += [
            dict(
                seed=s,
                quantity="cells with the generic per-cell label (old run)",
                old=int(gen.sum()),
                new=int((ln["canon"] == GENERIC).sum()),
            ),
            dict(
                seed=s,
                quantity="cells whose fine vote label changes",
                old=np.nan,
                new=int(ch.sum()),
                pct=100 * ch.mean(),
            ),
            dict(
                seed=s, quantity="clusters whose vote changes", old=np.nan, new=len(cl)
            ),
            dict(
                seed=s,
                quantity="Other (Mixed) share of cells, %",
                old=100 * (lo["vote_fine"] == MIXED).mean(),
                new=100 * (ln["vote_fine"] == MIXED).mean(),
            ),
            dict(
                seed=s,
                quantity="own-cell-type region accuracy, mean of 3 splits",
                old=np.mean(acc_o),
                new=np.mean(acc_n),
            ),
            dict(
                seed=s,
                quantity="Kenyon cell F1, expression-only vote vs calyx",
                old=f1_o,
                new=f1_n,
            ),
            dict(
                seed=s,
                quantity="per-cell vs vote differing, % of all cells",
                old=100 * (lo["pooled"] != lo["vote"]).mean(),
                new=100 * (ln["pooled"] != ln["vote"]).mean(),
            ),
        ]
        for k in sorted(set(sr_o.index) | set(sr_n.index)):
            summ.append(
                dict(
                    seed=s,
                    quantity=f"queen-worker diff / within-brain SD, {k}",
                    old=sr_o.ratio.get(k, np.nan),
                    new=sr_n.ratio.get(k, np.nan),
                )
            )
        prev = T / f"combined_clustered_g2_R2_noolkc_s{s}_tangram.h5ad"
        if (
            prev.exists()
        ):  # an earlier run of the same recipe: is the rerun reproducible?
            same = float(
                (
                    read_obs(prev)["tangram_cell_type"].astype(str).to_numpy()
                    == ob_n["tangram_cell_type"].astype(str).to_numpy()
                ).mean()
            )
            summ.append(
                dict(
                    seed=s,
                    quantity=(
                        "per-cell labels identical to the 2026-09-11 noolkc run, %"
                    ),
                    old=np.nan,
                    new=100 * same,
                )
            )
        print(
            f"seed {s}: generic cells {int(gen.sum()):,}; fine-vote changes "
            f"{int(ch.sum()):,} cells in {len(cl)} clusters; "
            f"region acc {np.mean(acc_o):.4f} -> {np.mean(acc_n):.4f}; Kenyon F1 "
            f"{f1_o:.4f} -> {f1_n:.4f}",
            flush=True,
        )
        if s == 42:
            sp_labels = {
                "canonical s42": ob_o["tangram_cell_type"].astype(str).to_numpy(),
                "noolkc s42": ob_n["tangram_cell_type"].astype(str).to_numpy(),
            }
    pd.DataFrame(summ).to_csv(OUT / "summary_by_seed.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(OUT / "composition_old_vs_new.csv", index=False)
    pd.DataFrame(move_rows).to_csv(OUT / "generic_cells_relabelled.csv", index=False)
    if cl_rows:
        pd.concat(cl_rows).to_csv(OUT / "clusters_whose_vote_changes.csv", index=False)
    if sp_labels and not a.skip_concordance:
        concordance(sp_labels).to_csv(OUT / "panel_concordance.csv", index=False)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
