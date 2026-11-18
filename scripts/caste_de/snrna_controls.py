#!/usr/bin/env python
"""Controls on the snRNA-seq caste-DE and composition analyses.

1. scCODA reference sensitivity (``--stage sccoda``).
   The ``fig:caste_comparison`` scCODA
   (``regions/caste_comparison_sccoda.py::compute_abundance``)
   fixes the reference category at the most abundant label, *Optic lobe*.
   This refits the same model -- same per-section count table, same labels,
   ``formula="brain"``, ``num_warmup=500, num_samples=1000, rng_key=42``,
   ``est_fdr=0.05``, pertpy defaults otherwise -- with the reference set to
   Optic lobe (a reproduction of the original run, checked against
   ``abundance_tangram_ct.csv``), Kenyon cells, Astrocyte-like glia, Glia
   (the unresolved glial group) and scCODA's automatic selection.  Four more
   seeds per reference (0-3) give the MCMC noise on an inclusion probability.
   The count table is rebuilt from ``results/figures/fig2/slice_composition.csv``
   (per-section proportions x n_cells, written by the same run as the
   saved scCODA CSV) and expanded back to one row per cell, so the model
   goes through the same ``Sccoda.load(type="cell_level")`` path.

2. Ambient / dissection contamination of the caste DEGs
   (``--stage contamination``).  Three gene scores, each tested against the
   DEG lists of the n=3 DESeq2 table and the recurring core:
     * ``fat_log2enr``: log2((CPM in the fat-body nuclei + 10) /
       (CPM in all other queen central-brain nuclei + 10)).  Fat-body nuclei
       are defined as in ``vgn_fat_body_origin.py`` (queen, Vgn UMI >= 5 and
       Kunitz-inhibitor UMI >= 10); the stage asserts their count.
       ``fat_high`` = fat_log2enr >= 2 and detected in >= 3 of them.
     * ``soup_log2ratio``: CellBender's learned ambient profile
       (``global_latents/ambient_expression`` of the
       ``*_CellbenderOut_halfLR_force.h5`` runs the atlas was built from, one
       per caste x trial library) over that library's pooled nuclear profile,
       log2, averaged over the six libraries.  ``soup_high`` = top 5% of the
       genes in the universe.  The castes were loaded on separate inlets
       (on-chip multiplexing) and CellBender was run per caste, so each
       library has its own ambient profile.
     * ``ambient_caste_lfc``: mean over trials of log2(queen ambient / worker
       ambient).  ``ambient_queen_high`` = >= 1 (log2) and soup-enriched
       (soup_log2ratio > 0).
   For every set: overlap, expected overlap (n K / M), hypergeometric upper
   tail, and a Mann-Whitney test of each score in the set vs the rest of the
   universe.  Sets: queen-up and worker-up DEGs (FDR < 0.05) in each of the
   seven coarse lists (universe = genes with a padj in that list; worker-up
   is the negative control, the fat body being a queen-dissection problem),
   and the recurring core from ``paths.CASTE_DE_RECURRENCE / cross_list_counts.csv``
   (``si_deg_recurrence.py``) at three stated cut-offs -- n_queen_up >= 5,
   n_coarse_queen_up >= 3, and the ten queen-up genes the SI figure shows --
   with the worker-side analogues (universe = genes with a padj in the
   whole-brain list).  Two direct checks besides: the share of each queen-up
   DEG's queen pseudobulk contributed by the fat-body nuclei, and per-trial
   queen/worker whole-brain log2 ratios of the core genes set against the
   per-trial queen Vgn level per million, the natural contamination gradient.

3. JH / ecdysone pathway genes in Kenyon cells (``--stage pathway``).
   A priori list of Drosophila genes, by category (FlyBase gene reports;
   Jindra, Palli & Riddiford 2013 Annu Rev Entomol 58:181 for JH; King-Jones &
   Thummel 2005 Nat Rev Genet 6:311 and Niwa & Niwa 2014 Biosci Biotechnol
   Biochem 78:1283 for ecdysone):
     JH biosynthesis      Jhamt, Cyp15a1 (absent from Drosophila; matched by
                          the NCBI description "methyl farnesoate epoxidase")
     JH degradation       Jhe, Jheh1, Jheh2, Jheh3
     JH reception/effect  Met, gce, Tai, Kr-h1, Eip93F (E93)
     ecdysone synthesis   nvd, spok, phm, dib, sad, shd; inactivation Cyp18a1
     ecdysone reception   EcR, usp, DopEcR
     early response       Eip75B (E75), Eip74EF (E74), br (Broad), Hr3 (Hr46)
     late/mid response    Hr4, ftz-f1 (betaFTZ-F1), Hr39, Eip78C (E78), ImpL2
   Mapped to BIMP v4.3 genes through ``results/round5/glia_labels/v43_dmel_rbh.csv.gz``
   (DIAMOND BIMP v4.3 vs Dmel; ``dmel_symbol`` of the best hit, any call, and
   ``relaxed_dmel_symbol``, case-insensitive), and separately by the NCBI
   product name of the same list members (``DESC_ROUTE``), which reaches the
   genes the Drosophila route cannot (Cyp15a1 has no Drosophila ortholog;
   Kr-h1, BIMP_v4.3_005433, has no strict hit; Jhamt, Jhe, Jheh are
   RBH-ambiguous); ``route`` says which.  Upstream mevalonate-pathway genes
   and JH-binding proteins are deliberately
   excluded (generic isoprenoid metabolism).  ``panel_dmel_orthologs.csv``
   covers only the 101 panel genes, so it is not used here.  Met and Jheh2/3 come
   back unmatched by design: Hymenoptera carry one Met/gce gene (matched as
   gce) and one Jheh.  Each gene gets
   its Kenyon-cell baseMean, log2FC, lfcSE, pvalue, padj from
   ``results/caste_de/de_coarse_blocked.csv`` (the ``~ trial +
   caste`` fit) and the fine Kenyon types from
   ``results/caste_de/de_fine_blocked.csv``, with two
   detectable effect sizes: 1.96 times lfcSE (nominal alpha 0.05) and
   z_crit x lfcSE, where z_crit is the Wald z of the weakest gene that
   passed FDR 0.05 in the Kenyon-cell list.

Inputs: ``results/figures/fig2/{slice_composition,abundance_tangram_ct}.csv``;
``paths.SN_CB`` (layers['counts']); the CellBender ambient profiles, frozen
by ``freeze_cellbender_ambient.py`` into ``AMBIENT``;
``results/caste_de/de_coarse_blocked.csv``;
``results/caste_de/de_fine_blocked.csv``;
``paths.CASTE_DE_RECURRENCE / cross_list_{counts,top20}.csv``;
``results/round5/glia_labels/v43_dmel_rbh.csv.gz``.

Output: results/handoff/sn_controls/
  sccoda_effects.csv          reference x seed x category: final parameter,
                              log2FC, inclusion probability, credible
  sccoda_summary.csv          reference x seed: credible effects, perineurial
                              inclusion probability, reproduction check
  contamination_gene_scores.csv   per gene: the three scores and flags
  contamination_tests.csv     per set x score: overlap, expected, p values
  contamination_fat_share.csv per queen-up DEG: share of queen pseudobulk
                              from the fat-body nuclei
  contamination_trial_gradient.csv  per core gene x trial queen/worker log2
  contamination_trial_summary.csv   per set x trial median, with Vgn level
  contamination_meta.json     fat-body nuclei, flag sizes, soup threshold,
                              per-trial queen Vgn per million
  pathway_kc.csv              JH/ecdysone genes in Kenyon cells (written to
                              results/caste_de/sn_controls/)
  pathway_unmatched.csv       list symbols with no BIMP v4.3 match (same dir)
  summary.md                  one sentence per result

Environment: the shared ``.venv`` (pertpy 1.1.1, numpyro 0.21, jax 0.10.2 --
the same interpreter caste_comparison_sccoda uses); no pyDESeq2 refit is done.

Run (via Slurm; scCODA ~1 h on 8 cpus, the others minutes)
    PYTHONPATH=src:scripts .venv/bin/python -u \\
        scripts/caste_de/snrna_controls.py --stage STAGE
    (STAGE: sccoda | contamination | pathway | summary | all)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts")]
from bimp_st.paths import (  # noqa: E402
    CASTE_DE_AMBIENT_DIR,
    CASTE_DE_COARSE,
    CASTE_DE_DIR,
    CASTE_DE_FINE,
    CASTE_DE_RECURRENCE,
    SN_CB,
)

OUT = REPO / "results/handoff/sn_controls"
FIG2 = REPO / "results/figures/fig2"
PATHWAY_OUT = CASTE_DE_DIR / "sn_controls"
DE = CASTE_DE_COARSE  # ~ trial + caste
FINE = CASTE_DE_FINE
XLIST = CASTE_DE_RECURRENCE
RBH = REPO / "results/round5/glia_labels/v43_dmel_rbh.csv.gz"
# CellBender ambient profile of each central-brain library (the runs behind
# BIMP_atlas_qw_CentralBrain.h5ad), frozen with md5s by
# freeze_cellbender_ambient.py; columns Queen_t1 ... Worker_t3, index gene id.
AMBIENT = CASTE_DE_AMBIENT_DIR / "cellbender_ambient.csv.gz"
FDR = 0.05

# ── 1. scCODA ────────────────────────────────────────────────────────────────
REFERENCES = ["Optic lobe", "Kenyon cells", "Astrocyte-like glia", "Glia", "automatic"]
SEEDS = [42, 0, 1, 2, 3]  # 42 is the seed of the original run
SHIPPED_REF = "Optic lobe"
PERI = "Perineurial glia"


def _count_table() -> tuple[pd.DataFrame, pd.Series]:
    from regions.caste_comparison_sccoda import _pooled_order

    sc_ = pd.read_csv(FIG2 / "slice_composition.csv")
    raw = sc_.prop * sc_.n_cells
    assert np.allclose(raw, raw.round(), atol=1e-6), "non-integer counts"
    sc_["count"] = raw.round().astype(int)
    tab = sc_.pivot(index="slice_name", columns="cluster", values="count")
    tab = tab[_pooled_order(tab.columns)]
    n = sc_.groupby("slice_name").n_cells.first()
    assert (tab.sum(1) == n.loc[tab.index]).all()
    brain = sc_.groupby("slice_name").brain.first().loc[tab.index]
    return tab, brain


def _run_sccoda(tab, brain, ref, seed):
    import anndata as ad
    import scanpy._utils as _su

    for _old, _new in (
        ("check_use_raw", "_check_use_raw"),
        ("sanitize_anndata", "_sanitize_anndata"),
    ):
        if not hasattr(_su, _old) and hasattr(_su, _new):
            setattr(_su, _old, getattr(_su, _new))
    from pertpy.tools import Sccoda

    ct_col = "cell_type_pooled"
    long = tab.stack().rename("n").reset_index()
    long.columns = ["slice_name", ct_col, "n"]
    idx = np.repeat(np.arange(len(long)), long.n.to_numpy())
    obs = long.iloc[idx][["slice_name", ct_col]].reset_index(drop=True)
    obs["brain"] = obs.slice_name.map(brain).astype(str)
    obs = obs[["slice_name", "brain", ct_col]].astype(str)
    cell = ad.AnnData(obs=obs)
    cell.obs_names_make_unique()
    sccoda = Sccoda()
    mdata = sccoda.load(
        cell,
        type="cell_level",
        generate_sample_level=True,
        cell_type_identifier=ct_col,
        sample_identifier="slice_name",
        covariate_obs=["brain"],
    )
    sccoda.prepare(mdata, modality_key="coda", formula="brain", reference_cell_type=ref)
    mdata["coda"].obs.index.name = None
    sccoda.run_nuts(
        mdata, modality_key="coda", num_warmup=500, num_samples=1000, rng_key=seed
    )
    sccoda.credible_effects(mdata, modality_key="coda", est_fdr=0.05)
    coda = mdata["coda"]
    key = [k for k in coda.varm.keys() if "effect_df" in k][0]
    eff = coda.varm[key].copy()
    params = coda.uns.get("scCODA_params", {})
    ref_idx = params.get("reference_index")
    chosen = str(coda.var_names[int(ref_idx)]) if ref_idx is not None else ref
    eff = eff.reset_index(names="cluster")
    eff["cluster"] = eff.cluster.astype(str)
    return eff, chosen, key


def stage_sccoda():
    tab, brain = _count_table()
    print("count table", tab.shape, "cells", int(tab.values.sum()), flush=True)
    shipped = pd.read_csv(FIG2 / "abundance_tangram_ct.csv").set_index("cluster")
    rows, summ = [], []
    for ref in REFERENCES:
        for seed in SEEDS:
            eff, chosen, key = _run_sccoda(tab, brain, ref, seed)
            eff["credible"] = eff["Final Parameter"] != 0
            eff.insert(0, "reference_requested", ref)
            eff.insert(1, "reference_used", chosen)
            eff.insert(2, "seed", seed)
            eff["is_reference"] = eff.cluster == chosen
            eff["contrast"] = key
            rows.append(eff)
            e = eff.set_index("cluster")
            s = dict(
                reference_requested=ref,
                reference_used=chosen,
                seed=seed,
                n_credible=int(e.credible.sum()),
                credible_effects="; ".join(
                    f"{c} ({e.loc[c, 'log2-fold change']:+.3f})"
                    for c in e.index[e.credible]
                ),
                perineurial_inclusion_prob=float(e.loc[PERI, "Inclusion probability"]),
                perineurial_credible=bool(e.loc[PERI, "credible"]),
                max_inclusion_prob=float(e["Inclusion probability"].max()),
                max_inclusion_category=str(e["Inclusion probability"].idxmax()),
            )
            if ref == SHIPPED_REF and seed == 42:
                d = (
                    e["Inclusion probability"] - shipped.loc[e.index, "inclusion_prob"]
                ).abs()
                s["max_abs_diff_vs_shipped"] = float(d.max())
            summ.append(s)
            print(s, flush=True)
    pd.concat(rows, ignore_index=True).to_csv(OUT / "sccoda_effects.csv", index=False)
    pd.DataFrame(summ).to_csv(OUT / "sccoda_summary.csv", index=False)


# ── 2. contamination ─────────────────────────────────────────────────────────
VGN, KUNITZ = "BIMP_v4.3_005500", "BIMP_v4.3_000091"
HIGH, FAT = 5, 10  # as vgn_fat_body_origin.py
COARSE = {
    "all": "Whole brain",
    "Kenyon_cells": "Kenyon cells",
    "Glia": "Glia",
    "Hemocytes": "Hemocytes",
    "Projection_neurons": "Projection neurons",
    "Other_neurons": "Other neurons",
    "Photoreceptors": "Photoreceptors",
}
MAJOR = {
    "Kenyon_cells": "Kenyon cells",
    "Glia": "Glia",
    "Hemocytes": "Hemocytes",
    "Projection_neurons": "Projection neurons",
    "Other_neurons": "Other neurons",
    "Photoreceptors": "Photoreceptors",
}
EPS = 1e-6


def _hyper(sel: set, flag: set, universe: set) -> dict:
    sel, flag = sel & universe, flag & universe
    M, K, n = len(universe), len(flag), len(sel)
    k = len(sel & flag)
    return dict(
        universe=M,
        flagged_in_universe=K,
        set_size=n,
        overlap=k,
        expected=n * K / M if M else np.nan,
        fold=(k / (n * K / M)) if n and K else np.nan,
        p_hyper=float(stats.hypergeom.sf(k - 1, M, K, n)) if n else np.nan,
    )


def _mw(score: pd.Series, sel: set, universe: set) -> dict:
    s = score.reindex(sorted(universe)).dropna()
    a, b = s[s.index.isin(sel)], s[~s.index.isin(sel)]
    if len(a) == 0:
        return dict(median_in_set=np.nan, median_rest=np.nan, p_mw_greater=np.nan)
    p = stats.mannwhitneyu(a, b, alternative="greater").pvalue
    return dict(
        median_in_set=float(a.median()),
        median_rest=float(b.median()),
        p_mw_greater=float(p),
    )


def _ambient_profiles(genes: pd.Index) -> pd.DataFrame:
    """Per-library ambient profile (sums to 1 over all features) on ``genes``."""
    amb = pd.read_csv(AMBIENT, index_col=0)
    for c in amb.columns:
        print(f"ambient {c}: {int(amb[c].reindex(genes).notna().sum())} matched")
    return amb.reindex(genes)


def stage_contamination():
    import anndata as ad
    import scipy.sparse as sp

    a = ad.read_h5ad(SN_CB)
    X = sp.csr_matrix(a.layers["counts"])
    obs = a.obs[["sample", "caste", "trial", "major_cell_type", "cell_type"]].copy()
    genes = a.var_names
    vgn = X[:, genes.get_loc(VGN)].toarray().ravel()
    kun = X[:, genes.get_loc(KUNITZ)].toarray().ravel()
    queen = (obs.caste == "Queen").to_numpy()
    fat = queen & (vgn >= HIGH) & (kun >= FAT)
    assert fat.sum() == 13, fat.sum()
    rest = queen & ~fat
    fat_umi = np.asarray(X[fat].sum(0)).ravel()
    fat_det = np.asarray((X[fat] > 0).sum(0)).ravel()
    rest_umi = np.asarray(X[rest].sum(0)).ravel()
    fat_cpm = fat_umi / fat_umi.sum() * 1e6
    rest_cpm = rest_umi / rest_umi.sum() * 1e6
    score = pd.DataFrame(index=genes)
    score["fat_umi"] = fat_umi
    score["fat_nuclei_detected"] = fat_det
    score["fat_cpm"] = fat_cpm
    score["other_queen_cpm"] = rest_cpm
    score["fat_log2enr"] = np.log2((fat_cpm + 10) / (rest_cpm + 10))
    score["fat_high"] = (score.fat_log2enr >= 2) & (score.fat_nuclei_detected >= 3)

    # per-sample pooled nuclear profile, and the CellBender ambient profile
    samples = sorted(obs["sample"].unique())
    S = sp.csr_matrix(
        (
            np.ones(len(obs)),
            (
                pd.Categorical(obs["sample"], categories=samples).codes,
                np.arange(len(obs)),
            ),
        ),
        shape=(len(samples), len(obs)),
    )
    pb = pd.DataFrame(np.asarray((S @ X).todense()), index=samples, columns=genes).T
    nuc = pb / pb.sum(0)
    amb = _ambient_profiles(genes)
    lib = {c: f"BIMP_{c}" for c in amb.columns}  # Queen_t1 -> BIMP_Queen_t1
    ratios = pd.DataFrame(
        {c: np.log2((amb[c] + EPS) / (nuc[lib[c]] + EPS)) for c in amb.columns}
    )
    score["soup_log2ratio"] = ratios.mean(1)
    score["ambient_caste_lfc"] = pd.concat(
        [
            np.log2((amb[f"Queen_t{t}"] + EPS) / (amb[f"Worker_t{t}"] + EPS))
            for t in (1, 2, 3)
        ],
        axis=1,
    ).mean(1)
    for c in amb.columns:
        score[f"ambient_frac_{c}"] = amb[c]

    de = pd.read_csv(DE, index_col=0)
    de = de[de.cell_type.isin(COARSE)]
    tested = {k: set(g.gene[g.padj.notna()]) for k, g in de.groupby("cell_type")}
    wb = tested["all"]
    q95 = score.loc[sorted(wb), "soup_log2ratio"].quantile(0.95)
    score["soup_high"] = score.soup_log2ratio >= q95
    score["ambient_queen_high"] = (score.ambient_caste_lfc >= 1) & (
        score.soup_log2ratio > 0
    )
    score.to_csv(OUT / "contamination_gene_scores.csv", index_label="gene")

    flags = {
        k: set(score.index[score[k]])
        for k in ("fat_high", "soup_high", "ambient_queen_high")
    }
    scores = {
        k: score[k] for k in ("fat_log2enr", "soup_log2ratio", "ambient_caste_lfc")
    }
    sets = []
    for ct, g in de.groupby("cell_type"):
        sig = g[g.padj < FDR]
        sets.append(
            (
                f"{COARSE[ct]} queen-up",
                set(sig.gene[sig.log2FoldChange > 0]),
                tested[ct],
            )
        )
        sets.append(
            (
                f"{COARSE[ct]} worker-up",
                set(sig.gene[sig.log2FoldChange < 0]),
                tested[ct],
            )
        )
    xc = pd.read_csv(XLIST / "cross_list_counts.csv")
    top = pd.read_csv(XLIST / "cross_list_top20.csv")
    core = {
        "core queen-up, n_queen_up >= 5": set(xc.gene[xc.n_queen_up >= 5]),
        "core queen-up, n_coarse_queen_up >= 3": set(
            xc.gene[xc.n_coarse_queen_up >= 3]
        ),
        "core queen-up, SI figure top 10": set(top.gene[top.side == "queen-up"]),
        "core worker-up, n_worker_up >= 5": set(xc.gene[xc.n_worker_up >= 5]),
        "core worker-up, n_coarse_worker_up >= 3": set(
            xc.gene[xc.n_coarse_worker_up >= 3]
        ),
        "core worker-up, SI figure top 10": set(top.gene[top.side == "worker-up"]),
    }
    sets += [(k, v, wb) for k, v in core.items()]
    rows = []
    for name, sel, uni in sets:
        for fk, fl in flags.items():
            rows.append(
                dict(set=name, test="hypergeometric", flag=fk, **_hyper(sel, fl, uni))
            )
        for sk, sc_ in scores.items():
            rows.append(
                dict(
                    set=name,
                    test="mann_whitney",
                    score=sk,
                    set_size=len(sel & uni),
                    universe=len(uni),
                    **_mw(sc_, sel, uni),
                )
            )
    pd.DataFrame(rows).to_csv(OUT / "contamination_tests.csv", index=False)

    # share of each queen-up DEG's queen pseudobulk from the 13 nuclei
    share = []
    for ct, g in de.groupby("cell_type"):
        qup = g[(g.padj < FDR) & (g.log2FoldChange > 0)].gene
        m = (
            queen.copy()
            if ct == "all"
            else queen & (obs.major_cell_type == MAJOR[ct]).to_numpy()
        )
        qsum = np.asarray(X[m].sum(0)).ravel()
        fsum = np.asarray(X[m & fat].sum(0)).ravel()
        loc = genes.get_indexer(qup)
        share.append(
            pd.DataFrame(
                dict(
                    list=COARSE[ct],
                    gene=qup.values,
                    queen_umi=qsum[loc],
                    fat_nuclei_umi=fsum[loc],
                    fat_share=np.where(
                        qsum[loc] > 0, fsum[loc] / np.maximum(qsum[loc], 1), np.nan
                    ),
                    n_fat_nuclei_in_list=int((m & fat).sum()),
                )
            )
        )
    pd.concat(share).to_csv(OUT / "contamination_fat_share.csv", index=False)

    # per-trial queen/worker log2 ratio (whole-brain CPM) of the core genes
    cpm = pb / pb.sum(0) * 1e6
    vgn_pm = {t: float(cpm.loc[VGN, f"BIMP_Queen_{t}"]) for t in ("t1", "t2", "t3")}
    grad, gsum = [], []
    gsets = dict(core)
    gsets["Whole brain queen-up"] = sets[
        [s[0] for s in sets].index("Whole brain queen-up")
    ][1]
    gsets["Whole brain worker-up"] = sets[
        [s[0] for s in sets].index("Whole brain worker-up")
    ][1]
    # the flagged part of the core: if it is contamination it should track
    # the queen Vgn gradient even where the core as a whole does not
    c3 = core["core queen-up, n_coarse_queen_up >= 3"]
    gsets["core queen-up (coarse >= 3) & fat_high"] = c3 & flags["fat_high"]
    gsets["core queen-up (coarse >= 3) & ambient_queen_high"] = (
        c3 & flags["ambient_queen_high"]
    )
    gsets["core queen-up (coarse >= 3), unflagged"] = (
        c3 - flags["fat_high"] - (flags["ambient_queen_high"])
    )
    gsets["Vgn"] = {VGN}
    for name, sel in gsets.items():
        sel = sorted(set(sel) & set(genes))
        for t in ("t1", "t2", "t3"):
            lr = np.log2(
                (cpm.loc[sel, f"BIMP_Queen_{t}"] + 1)
                / (cpm.loc[sel, f"BIMP_Worker_{t}"] + 1)
            )
            grad.append(
                pd.DataFrame(
                    dict(set=name, gene=sel, trial=t, log2_queen_over_worker=lr.values)
                )
            )
            gsum.append(
                dict(
                    set=name,
                    trial=t,
                    n_genes=len(sel),
                    median_log2_queen_over_worker=float(lr.median()),
                    frac_positive=float((lr > 0).mean()),
                    queen_vgn_per_million=vgn_pm[t],
                )
            )
    pd.concat(grad).to_csv(OUT / "contamination_trial_gradient.csv", index=False)
    pd.DataFrame(gsum).to_csv(OUT / "contamination_trial_summary.csv", index=False)
    meta = dict(
        n_fat_nuclei=int(fat.sum()),
        fat_nuclei_major_type=obs.loc[fat, "major_cell_type"].value_counts().to_dict(),
        n_fat_high=len(flags["fat_high"]),
        n_soup_high=len(flags["soup_high"]),
        soup_high_threshold=float(q95),
        n_ambient_queen_high=len(flags["ambient_queen_high"]),
        queen_vgn_per_million=vgn_pm,
    )
    (OUT / "contamination_meta.json").write_text(
        json.dumps(meta, indent=1, default=str)
    )
    print(meta, flush=True)


# ── 3. JH / ecdysone pathway ─────────────────────────────────────────────────
PATHWAY = [  # (category, Drosophila symbol, aliases also matched)
    ("JH biosynthesis", "Jhamt", []),
    ("JH degradation", "Jhe", []),
    ("JH degradation", "Jheh1", []),
    ("JH degradation", "Jheh2", []),
    ("JH degradation", "Jheh3", []),
    ("JH reception/effector", "Met", []),
    ("JH reception/effector", "gce", []),
    ("JH reception/effector", "Tai", []),
    ("JH reception/effector", "Kr-h1", []),
    ("JH reception/effector", "Eip93F", ["E93"]),
    ("ecdysone biosynthesis", "nvd", []),
    ("ecdysone biosynthesis", "spok", ["Cyp307a1"]),
    ("ecdysone biosynthesis", "phm", ["Cyp306a1"]),
    ("ecdysone biosynthesis", "dib", ["Cyp302a1"]),
    ("ecdysone biosynthesis", "sad", ["Cyp315a1"]),
    ("ecdysone biosynthesis", "shd", ["Cyp314a1"]),
    ("ecdysone inactivation", "Cyp18a1", []),
    ("ecdysone reception", "EcR", []),
    ("ecdysone reception", "usp", []),
    ("ecdysone reception", "DopEcR", []),
    ("ecdysone early response", "Eip75B", ["E75"]),
    ("ecdysone early response", "Eip74EF", ["E74"]),
    ("ecdysone early response", "br", ["Broad", "BR-C"]),
    ("ecdysone early response", "Hr3", ["Hr46", "DHR3"]),
    ("ecdysone mid/late response", "Hr4", []),
    ("ecdysone mid/late response", "ftz-f1", []),
    ("ecdysone mid/late response", "Hr39", []),
    ("ecdysone mid/late response", "Eip78C", ["E78"]),
    ("ecdysone mid/late response", "ImpL2", []),
]
# Second route: the NCBI (BIMP v4.3) product name of the same list members.
# It reaches genes the Drosophila route misses -- Cyp15a1 (no Drosophila
# ortholog), Kr-h1 (no strict hit; the repo's own ID is BIMP_v4.3_005433),
# Jhe / Jheh (annotated but RBH-ambiguous), Met/gce -- and adds no product
# outside the list.
DESC_ROUTE = {
    "Jhamt": r"juvenile hormone acid (O-)?methyltransferase",
    "Cyp15a1": r"methyl farnesoate epoxidase",
    "Jhe": r"^juvenile hormone esterase",
    "Jheh1": r"juvenile hormone epoxide hydrolase",
    "Met": r"methoprene-tolerant|germ cell-expressed bHLH-PAS",
    "Tai": r"taiman",
    "Kr-h1": r"^kr(u|ue|\u00fc)ppel homolog 1$",
    "Eip93F": r"ecdysone-induced protein 93F",
    "EcR": r"^ecdysone receptor",
    "usp": r"ultraspiracle",
    "Eip75B": r"ecdysone-induced protein 75",
    "Eip74EF": r"ecdysone-induced protein E?74",
    "Eip78C": r"ecdysone-induced protein 78C",
    "br": r"broad-complex",
    "ImpL2": r"ecdysone-inducible gene L2",
}
KC_FINE = [
    "Large-type Kenyon cells",
    "Middle-type Kenyon cells",
    "Small-type Kenyon cells",
    "Class II Kenyon cells",
    "Other Kenyon cells",
]


def stage_pathway():
    rbh = pd.read_csv(RBH)

    def clean(s):
        return (
            s.fillna("").astype(str).str.replace("Dmel\\", "", regex=False).str.lower()
        )

    sym, rel = clean(rbh.dmel_symbol), clean(rbh.relaxed_dmel_symbol)
    desc = rbh.description.fillna("")
    cat_of = {s: c for c, s, _ in PATHWAY}
    cat_of["Cyp15a1"] = "JH biosynthesis"
    hits = []
    for cat, s, aliases in PATHWAY:
        names = {x.lower() for x in (s, *aliases)}
        m = rbh[sym.isin(names) | rel.isin(names)].copy()
        m["route"] = np.where(
            sym[m.index].isin(names), "dmel_symbol", "relaxed_dmel_symbol"
        )
        m["category"], m["query_symbol"] = cat, s
        hits.append(m)
    for s, rx in DESC_ROUTE.items():
        m = rbh[desc.str.contains(rx, case=False, regex=True)].copy()
        m["route"], m["category"], m["query_symbol"] = "NCBI description", cat_of[s], s
        hits.append(m)
    h = pd.concat(hits, ignore_index=True)
    # one row per (list symbol, BIMP gene); keep the Drosophila route if both hit
    h["route"] = h.groupby(["query_symbol", "gene"]).route.transform(
        lambda r: " + ".join(sorted(set(r)))
    )
    h = h.drop_duplicates(["query_symbol", "gene"])
    done = set(h.query_symbol)
    unmatched = [
        dict(category=c, dmel_symbol=s) for c, s, _ in PATHWAY if s not in done
    ]
    h = h[
        [
            "category",
            "query_symbol",
            "gene",
            "route",
            "description",
            "dmel_symbol",
            "call",
            "reciprocal",
            "pident",
            "qcov",
            "evalue",
        ]
    ]

    de = pd.read_csv(DE, index_col=0)
    kc = de[de.cell_type == "Kenyon_cells"].set_index("gene")
    sig = kc[kc.padj < FDR]
    z_crit = float(stats.norm.isf(sig.pvalue.max() / 2))
    t = h.join(kc[["baseMean", "log2FoldChange", "lfcSE", "pvalue", "padj"]], on="gene")
    t["in_kc_table"] = t.gene.isin(kc.index)
    t["tested"] = t.padj.notna()
    t["sig_fdr05"] = t.padj < FDR
    t["mde_nominal_log2fc"] = 1.96 * t.lfcSE
    t["mde_fdr05_log2fc"] = z_crit * t.lfcSE
    t["z_crit_fdr05"] = z_crit
    tt = t[t.tested].drop_duplicates("gene")
    from statsmodels.stats.multitest import multipletests

    bh = pd.Series(multipletests(tt.pvalue, method="fdr_bh")[1], index=tt.gene)
    t["padj_within_pathway"] = t.gene.map(bh)
    fine = pd.read_csv(FINE, index_col=0)
    fine = fine[fine.cell_type.isin(KC_FINE)]
    fs = fine[fine.padj < FDR]
    t["fine_kc_sig"] = t.gene.map(
        fs.groupby("gene").apply(
            lambda g: "; ".join(
                f"{c} ({v:+.2f})"
                for c, v in zip(g.cell_type, g.log2FoldChange, strict=True)
            )
        )
    )
    PATHWAY_OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(PATHWAY_OUT / "pathway_kc.csv", index=False)
    pd.DataFrame(unmatched).to_csv(PATHWAY_OUT / "pathway_unmatched.csv", index=False)
    print(
        t[
            [
                "query_symbol",
                "gene",
                "route",
                "call",
                "log2FoldChange",
                "lfcSE",
                "padj",
                "mde_fdr05_log2fc",
                "fine_kc_sig",
            ]
        ].to_string(),
        flush=True,
    )
    print("z_crit", z_crit, "unmatched", unmatched, flush=True)


# ── summary ──────────────────────────────────────────────────────────────────
def stage_summary():
    L = [
        "# snRNA / composition controls",
        "",
        "Generated by `scripts/caste_de/snrna_controls.py`; every number "
        "below is in the CSV named beside it.",
        "",
    ]
    p = OUT / "sccoda_summary.csv"
    if p.exists():
        s = pd.read_csv(p)
        s42 = s[s.seed == 42]
        L.append("## scCODA reference sensitivity")
        rep = s42[s42.reference_requested == SHIPPED_REF].iloc[0]
        L.append(
            f"- Reproduction of the shipped run (reference Optic lobe, seed 42): "
            f"max |inclusion probability difference| = "
            f"{rep.max_abs_diff_vs_shipped:.3f} (`sccoda_summary.csv`, "
            f"`max_abs_diff_vs_shipped`)."
        )
        for _, r in s42.iterrows():
            rng = s[
                s.reference_requested == r.reference_requested
            ].perineurial_inclusion_prob
            L.append(
                f"- Reference {r.reference_requested} (used: {r.reference_used}): "
                f"{r.n_credible} credible effects"
                f"{' (' + r.credible_effects + ')' if r.n_credible else ''}; "
                "perineurial glia inclusion probability "
                f"{r.perineurial_inclusion_prob:.3f} "
                f"(seeds 0-3 and 42: {rng.min():.3f}-{rng.max():.3f}) "
                f"(`sccoda_summary.csv`, `perineurial_inclusion_prob`)."
            )
        L.append("")
    p = OUT / "contamination_tests.csv"
    if p.exists():
        c = pd.read_csv(p)
        meta = json.loads((OUT / "contamination_meta.json").read_text())
        L.append("## Ambient / fat-body contamination of the DEGs")
        L.append(
            f"- Flag sizes: {meta['n_fat_high']} fat-body-high genes, "
            f"{meta['n_soup_high']} soup-high genes, "
            f"{meta['n_ambient_queen_high']} queen-ambient-high genes "
            f"(`contamination_meta.json`)."
        )
        hy = c[c.test == "hypergeometric"]
        for _, r in hy.iterrows():
            if r.flag != "soup_high" and r.set.startswith(
                ("core queen-up", "Whole brain", "Kenyon cells queen", "Glia queen")
            ):
                L.append(
                    f"- {r.set} x {r.flag}: {int(r.overlap)} of {int(r.set_size)} "
                    f"(expected {r.expected:.1f}, p = {r.p_hyper:.2g}) "
                    f"(`contamination_tests.csv`, `overlap`/`expected`/`p_hyper`)."
                )
        fs = pd.read_csv(OUT / "contamination_fat_share.csv")
        for lst, g in fs.groupby("list"):
            if lst not in ("Projection neurons", "Whole brain", "Glia", "Kenyon cells"):
                continue
            L.append(
                f"- {lst} queen-up DEGs: {int((g.fat_share > 0.10).sum())} of {len(g)} "
                f"take > 10% of their queen pseudobulk from the 13 fat-body nuclei, "
                f"median share {g.fat_share.median():.4f} "
                f"(`contamination_fat_share.csv`, `fat_share`)."
            )
        ts = pd.read_csv(OUT / "contamination_trial_summary.csv")
        for name, g in ts.groupby("set", sort=False):
            vals = ", ".join(
                f"{r.trial} {r.median_log2_queen_over_worker:+.2f}"
                for r in g.itertuples()
            )
            L.append(
                f"- {name}: median queen/worker log2 by trial {vals}, against "
                f"queen Vgn {', '.join(f'{v:.1f}' for v in g.queen_vgn_per_million)} "
                f"per million (`contamination_trial_summary.csv`, "
                f"`median_log2_queen_over_worker`)."
            )
        L.append("")
    p = PATHWAY_OUT / "pathway_kc.csv"
    if p.exists():
        t = pd.read_csv(p)
        L.append("## JH / ecdysone pathway genes in Kenyon cells")
        u = t.drop_duplicates("gene")
        L.append(
            f"- {u.gene.nunique()} BIMP genes matched "
            f"{t.query_symbol.nunique()} list symbols; {int(u.tested.sum())} "
            f"tested in Kenyon cells, {int(u.sig_fdr05.sum())} at FDR < 0.05 "
            f"({', '.join(u[u.sig_fdr05.fillna(False)].query_symbol)}) "
            f"(`pathway_kc.csv`, `padj`)."
        )
        L.append(
            f"- Median detectable |log2FC| among tested genes: "
            f"{u[u.tested].mde_fdr05_log2fc.median():.2f} at the list's FDR "
            f"threshold, {u[u.tested].mde_nominal_log2fc.median():.2f} nominal "
            f"(`pathway_kc.csv`, `mde_fdr05_log2fc`, `mde_nominal_log2fc`)."
        )
        for r in u[u.sig_fdr05.fillna(False)].itertuples():
            L.append(
                f"- {r.query_symbol} ({r.gene}): Kenyon-cell log2FC "
                f"{r.log2FoldChange:+.2f}, lfcSE {r.lfcSE:.2f}, padj {r.padj:.2g}, "
                f"baseMean {r.baseMean:.0f} "
                f"(`pathway_kc.csv`, `log2FoldChange`/`padj`)."
            )
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        default="all",
        choices=["sccoda", "contamination", "pathway", "summary", "all"],
    )
    st = ap.parse_args().stage
    OUT.mkdir(parents=True, exist_ok=True)
    if st in ("pathway", "all"):
        stage_pathway()
    if st in ("contamination", "all"):
        stage_contamination()
    if st in ("sccoda", "all"):
        stage_sccoda()
    stage_summary()


if __name__ == "__main__":
    main()
