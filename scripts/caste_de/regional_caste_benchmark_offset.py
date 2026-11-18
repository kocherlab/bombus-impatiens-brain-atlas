#!/usr/bin/env python3
"""Robustness of the per-region caste benchmark (regional_caste_benchmark.py)
to a brain/run-wide offset, region specificity, a chance baseline for snRNA
direction agreement, and one yardstick for expression and composition.

The original analysis is left untouched; its functions and constants are
imported (``section_means``, ``l2fc``, ``N_SPLITS``, ``SEED``, ...).  Its split
draws are reproduced exactly (same rng, same call order) and the original
per-region exceed counts are recomputed and asserted against the saved
``results/caste_de/B_caste_benchmark/caste_benchmark_counts.csv``.

Normalisation in the original: ``.X`` of the nine-region atlas objects is
``normalize_total(target_sum=1e4)`` then ``log1p`` per cell (analysis_workflow
cluster step), then section means.  So it is NOT raw means; it is per-cell
total-count normalised to a fixed 1e4, which on a ~30-count cell turns one
transcript into log1p(~345) and makes the log1p mean depth-sensitive.  The
depth variant ("robust") therefore rebuilds the matrix from ``raw.X`` counts:
log1p(count / cell_total * T) with T = the median per-cell raw total pooled
over both brains (one target for both runs), then the identical section-mean
pipeline.

Variants, per region (``variant`` column):
  original          the original contrast and null (|fc| vs p95 of |split|)
  centred           fc - median_g(fc); each split draw d - median_g(d)
  robust            pooled-median-depth log1p, uncentred
  robust_centred    robust, then the same centring

Analyses
  1  offset-removed exceed counts per region (offset_benchmark_*.csv)
  2  region specificity over the six matched-coverage regions (lamina,
     medulla, lobula, kc_oc, kc_ic, kc_nc; al/central_complex in an
     eight-region block): pairwise Jaccard of exceeding sets with a
     random-gene-set expectation, genes exceeding in exactly one / all,
     sign discordance, pairwise correlation of fc, and per gene the SD
     across regions of the (centred) fc vs the same SD over split draws.
     The interaction null pairs draw i of every region (independent splits
     per region, queen draws first then worker): a heuristic, not a
     shared-split null.
  3  direction agreement with snRNA (Kenyon_cells for KC compartments, all
     nuclei for every region): analytic chance p_q*p_s + (1-p_q)(1-p_s) from
     the marginal worker-up rates of the gene set, and a permutation p
     (snRNA signs shuffled within the same gene set, N_PERM draws).
  4  composition (results/figures/fig2/slice_composition.csv, the per-section
     cluster-vote shares behind `fig:caste_comparison` panel B) benchmarked
     with the identical
     split-half construction, and expression put on the composition SD-ratio
     yardstick, so both are compared on both scales.  Composition uses no
     pseudocount (every class share is > 0 in every section; asserted).

Inputs: results/nine_region/atlas/{queen,worker}_aligned_clustered_9region.h5ad,
results/caste_de/de_coarse_blocked.csv, data/panel/np31_to_v43_mapping.csv,
results/figures/fig2/slice_composition.csv,
results/caste_de/B_caste_benchmark/caste_benchmark_counts.csv (assertion only).

Output: results/caste_de/B_caste_benchmark_offset/

Run
    PYTHONPATH=src:scripts .venv/bin/python \
        scripts/caste_de/regional_caste_benchmark_offset.py
"""

import sys as _sys  # noqa: E402
from pathlib import Path as _Path  # noqa: E402

_sys.path.insert(
    0, str(_Path(__file__).resolve().parents[1])
)  # scripts/, for topic-package imports

from itertools import combinations

import anndata as ad
import numpy as np
import pandas as pd
from caste_de.regional_caste_benchmark import (
    ATLAS,
    CUT_Q,
    DESEQ,
    MAPPING,
    MIN_SECTIONS,
    N_SPLITS,
    REGION_CELLTYPE,
    SEED,
    l2fc,
    section_means,
)

from bimp_st.nine_region_common import DESCRIPTIVE_REGIONS
from bimp_st.paths import (
    CASTE_DE_DIR,  # noqa: E402
    REPO,  # noqa: E402
)

OUT = CASTE_DE_DIR / "B_caste_benchmark_offset"
SHIPPED_COUNTS = (
    CASTE_DE_DIR / "B_caste_benchmark/caste_benchmark_counts.csv"
)
COMPOSITION = REPO / "results/figures/fig2/slice_composition.csv"

SIX = ["lamina", "medulla", "lobula", "kc_oc", "kc_ic", "kc_nc"]
KC = ["kc_oc", "kc_ic", "kc_nc"]
WHOLE = "whole_section"  # all cells of a section, no region mask
VARIANTS = ["original", "centred", "robust", "robust_centred"]
N_PERM = 10_000
PERM_SEED = 20261027
COMP_SEED = SEED + 1
GLIA_PARTS = ["Glia", "Astrocyte-like glia", "Perineurial glia"]
GLIA_POOL = "Glia (all three pooled)"


def draw_starts(k, rng):
    """Block starts exactly as split_half_null draws them (same rng calls)."""
    h = k // 2
    return np.array([rng.integers(0, k - h + 1) for _ in range(N_SPLITS)])


def signed_splits(S, starts, pseudo=True):
    """Signed log2FC of each contiguous half-block vs the remainder."""
    k = S.shape[0]
    h = k // 2
    out = np.empty((len(starts), S.shape[1]))
    for i, st in enumerate(starts):
        sel = np.zeros(k, bool)
        sel[st : st + h] = True
        a, b = S[sel].mean(0), S[~sel].mean(0)
        out[i] = l2fc(a, b) if pseudo else np.log2(a / b)
    return out


def sd_ratio(Sq, Sw):
    """|mean_w - mean_q| / mean(per-brain SD across sections), per column."""
    sd = (Sq.std(0, ddof=1) + Sw.std(0, ddof=1)) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.abs(Sw.mean(0) - Sq.mean(0)) / sd


def load():
    ads, X, XR = {}, {}, {}
    totals = {}
    for brain, p in ATLAS.items():
        a = ad.read_h5ad(p)
        x = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        X[brain] = x.astype(np.float64)
        r = a.raw.X
        r = r.toarray() if hasattr(r, "toarray") else np.asarray(r)
        assert list(a.raw.var_names) == list(a.var_names)
        XR[brain] = r.astype(np.float64)
        totals[brain] = XR[brain].sum(1)
        a.obs[f"in_{WHOLE}"] = True
        ads[brain] = a
    T = float(np.median(np.concatenate(list(totals.values()))))
    XN = {}
    for b in ads:
        t = np.maximum(totals[b], 1.0)[:, None]
        XN[b] = np.log1p(XR[b] / t * T)
        del XR[b]
    return ads, X, XN, T, totals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ads, X, XN, T, totals = load()
    genes = list(ads["queen"].var_names)
    assert genes == list(ads["worker"].var_names)
    G = len(genes)
    short = [g.split("|")[0] for g in genes]
    m = pd.read_csv(MAPPING).set_index("np31_id")
    v43 = [m.v43_id.get(s) for s in short]

    # ---- reproduce the original rng stream, then reuse the same splits ----
    rng = np.random.default_rng(SEED)
    regions = list(DESCRIPTIVE_REGIONS) + [WHOLE]
    S = {}  # (variant_base, region, brain) -> section matrix
    starts = {}  # (region, brain) -> block starts
    elig_of, nsec_of = {}, {}
    for region in regions:
        if region == WHOLE:  # after the original stream: own rng
            rng = np.random.default_rng(SEED + 2)
        for base, M in (("original", X), ("robust", XN)):
            for b in ads:
                S[(base, region, b)] = section_means(ads[b], M[b], region)
        nsec = {b: S[("original", region, b)].shape[0] for b in ads}
        nsec_of[region] = nsec
        elig = [b for b in ads if nsec[b] >= MIN_SECTIONS]
        elig_of[region] = elig
        if nsec["queen"] == 0 or nsec["worker"] == 0:
            continue
        for b in elig:
            starts[(region, b)] = draw_starts(nsec[b], rng)

    rows = []
    null_signed = {}  # (variant, region) -> (n_draws, G) signed, centred if so
    fc_of = {}
    for region in regions:
        elig = elig_of[region]
        if not elig or 0 in nsec_of[region].values():
            continue
        for v in VARIANTS:
            base = "robust" if v.startswith("robust") else "original"
            Sq, Sw = S[(base, region, "queen")], S[(base, region, "worker")]
            fc = l2fc(Sw.mean(0), Sq.mean(0))
            d = np.vstack(
                [signed_splits(S[(base, region, b)], starts[(region, b)]) for b in elig]
            )
            off = 0.0
            if v.endswith("centred"):
                off = float(np.median(fc))
                fc = fc - off
                d = d - np.median(d, axis=1, keepdims=True)
            p95 = np.percentile(np.abs(d), CUT_Q, axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(p95 > 0, np.abs(fc) / p95, np.nan)
            sdr = sd_ratio(Sq, Sw)
            null_signed[(v, region)] = d
            fc_of[(v, region)] = fc
            for gi, g in enumerate(genes):
                rows.append(
                    dict(
                        region=region,
                        variant=v,
                        gene=g,
                        gene_id_v43=v43[gi],
                        offset_subtracted=round(off, 4),
                        log2fc_worker_over_queen=fc[gi],
                        null_p95=p95[gi],
                        ratio=ratio[gi],
                        exceeds=bool(ratio[gi] > 1),
                        sd_ratio=sdr[gi],
                        n_null_draws=d.shape[0],
                        null_brains="+".join(elig),
                    )
                )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "offset_benchmark_table.csv", index=False)

    # ---- 1. counts, with the saved counts beside them ----
    shipped = pd.read_csv(SHIPPED_COUNTS).set_index("region")["n_exceed_null"]
    cnt = (
        tab.groupby(["region", "variant"], sort=False)
        .agg(
            offset_median_log2fc=("offset_subtracted", "first"),
            uncentred_median_log2fc=("log2fc_worker_over_queen", "median"),
            n_exceed=("exceeds", "sum"),
            n_genes=("gene", "size"),
            n_null_draws=("n_null_draws", "first"),
        )
        .reset_index()
    )
    wide = cnt.pivot(index="region", columns="variant", values="n_exceed")
    wide = wide.reindex([r for r in regions if r in wide.index])[VARIANTS]
    wide.columns = [f"n_exceed_{c}" for c in wide.columns]
    offs = cnt.pivot(index="region", columns="variant", values="offset_median_log2fc")
    wide["offset_original"] = offs["centred"]
    wide["offset_robust"] = offs["robust_centred"]
    wide["n_exceed_shipped"] = shipped.reindex(wide.index)
    wide = wide.reset_index()
    for r in wide.itertuples():
        if r.region in shipped.index:
            assert r.n_exceed_original == r.n_exceed_shipped, (
                f"{r.region}: recomputed {r.n_exceed_original} != shipped "
                f"{r.n_exceed_shipped}; reuse of the original is not faithful"
            )
    wide.to_csv(OUT / "offset_benchmark_counts.csv", index=False)
    print(wide.to_string(index=False))

    # ---- 2. region specificity ----
    prng = np.random.default_rng(PERM_SEED)
    jac_rows, spec_rows, int_rows = [], [], []
    eight = [r for r in DESCRIPTIVE_REGIONS if ("original", r) in fc_of]
    for v in VARIANTS:
        for scope, regs in (("six", SIX), ("eight", eight)):
            E = np.vstack(
                [
                    tab.loc[
                        (tab.variant == v) & (tab.region == r), "exceeds"
                    ].to_numpy()
                    for r in regs
                ]
            )
            F = np.vstack([fc_of[(v, r)] for r in regs])
            k = E.sum(0)
            sgn = np.sign(F)
            multi = k >= 2
            disc = np.array([len(set(sgn[E[:, g], g])) > 1 for g in range(G)])
            spec_rows.append(
                dict(
                    variant=v,
                    scope=scope,
                    n_regions=len(regs),
                    n_exceed_none=int((k == 0).sum()),
                    n_exceed_exactly_one=int((k == 1).sum()),
                    n_exceed_ge2=int(multi.sum()),
                    n_exceed_all=int((k == len(regs)).sum()),
                    n_ge2_sign_discordant=int((disc & multi).sum()),
                    exceed_in_all_expected_independent=float(np.prod(E.mean(1)) * G),
                )
            )
            if scope != "six":
                continue
            # Jaccard, with expectation for random sets of the same sizes
            for (i, a), (j, b) in combinations(enumerate(regs), 2):
                na, nb = int(E[i].sum()), int(E[j].sum())
                both = int((E[i] & E[j]).sum())
                uni = int((E[i] | E[j]).sum())
                jr = np.empty(N_PERM)
                for t in range(N_PERM):
                    A = np.zeros(G, bool)
                    A[prng.choice(G, na, replace=False)] = True
                    B = np.zeros(G, bool)
                    B[prng.choice(G, nb, replace=False)] = True
                    u = (A | B).sum()
                    jr[t] = (A & B).sum() / u if u else np.nan
                jac = both / uni if uni else np.nan
                rho = np.corrcoef(F[i], F[j])[0, 1]
                jac_rows.append(
                    dict(
                        variant=v,
                        region_a=a,
                        region_b=b,
                        n_a=na,
                        n_b=nb,
                        n_both=both,
                        jaccard=jac,
                        jaccard_random_mean=float(np.nanmean(jr)),
                        jaccard_random_p_ge=float(
                            (1 + (jr >= jac).sum()) / (1 + N_PERM)
                        ),
                        pearson_log2fc=rho,
                    )
                )
            # interaction: SD across the six regions vs SD over split draws
            D = [null_signed[(v, r)] for r in regs]
            nd = {x.shape[0] for x in D}
            assert len(nd) == 1, f"unequal null draws across regions: {nd}"
            Dn = np.stack(D)  # (regions, draws, G)
            sd_obs = F.std(0, ddof=1)
            sd_null = Dn.std(0, ddof=1)  # (draws, G)
            p95 = np.percentile(sd_null, CUT_Q, axis=0)
            pe = (1 + (sd_null >= sd_obs).sum(0)) / (1 + sd_null.shape[0])
            mean_obs = F.mean(0)
            mean_null = np.percentile(np.abs(Dn.mean(0)), CUT_Q, axis=0)
            for g in range(G):
                int_rows.append(
                    dict(
                        variant=v,
                        gene=genes[g],
                        gene_id_v43=v43[g],
                        mean_log2fc_six=mean_obs[g],
                        mean_abs_null_p95=mean_null[g],
                        shared_exceeds=bool(abs(mean_obs[g]) > mean_null[g]),
                        sd_across_six=sd_obs[g],
                        null_sd_p95=p95[g],
                        null_sd_median=float(np.median(sd_null[:, g])),
                        sd_ratio_to_null_p95=sd_obs[g] / p95[g],
                        interaction_exceeds=bool(sd_obs[g] > p95[g]),
                        interaction_p_emp=pe[g],
                        n_regions_exceed=int(k[g]),
                    )
                )
    pd.DataFrame(jac_rows).to_csv(OUT / "region_jaccard.csv", index=False)
    spec = pd.DataFrame(spec_rows)
    spec.to_csv(OUT / "region_specificity_summary.csv", index=False)
    inter = pd.DataFrame(int_rows)
    inter.to_csv(OUT / "region_interaction_by_gene.csv", index=False)
    isum = (
        inter.groupby("variant", sort=False)
        .agg(
            n_genes=("gene", "size"),
            n_interaction_exceeds=("interaction_exceeds", "sum"),
            n_shared_exceeds=("shared_exceeds", "sum"),
            median_sd_ratio_to_null_p95=("sd_ratio_to_null_p95", "median"),
        )
        .reset_index()
    )
    isum.to_csv(OUT / "region_interaction_summary.csv", index=False)

    # ---- 3. direction agreement with snRNA, against chance ----
    de = pd.read_csv(DESEQ, index_col=0)
    de = de.drop_duplicates(["cell_type", "gene"]).set_index(["cell_type", "gene"])

    def sn_sign(ct):
        out = np.full(G, np.nan)
        for gi, gid in enumerate(v43):
            if ct is not None and (ct, gid) in de.index:
                x = -float(de.loc[(ct, gid)].log2FoldChange)  # worker/queen
                out[gi] = np.sign(x) if x != 0 else np.nan
        return out

    sn = {"Kenyon_cells": sn_sign("Kenyon_cells"), "all": sn_sign("all")}
    arows = []
    for v in VARIANTS:
        for region in regions:
            if (v, region) not in fc_of:
                continue
            refs = ["all"] + (
                [REGION_CELLTYPE.get(region)] if REGION_CELLTYPE.get(region) else []
            )
            sub = tab[(tab.variant == v) & (tab.region == region)]
            ex = sub.exceeds.to_numpy()
            ours = np.sign(fc_of[(v, region)])
            for ref in refs:
                s = sn[ref]
                for gset in ("exceeding", "all_tested"):
                    keep = ~np.isnan(s) & (ours != 0)
                    if gset == "exceeding":
                        keep &= ex
                    a, b = ours[keep] > 0, s[keep] > 0
                    n = int(keep.sum())
                    if n == 0:
                        continue
                    obs = int((a == b).sum())
                    pq, ps = a.mean(), b.mean()
                    exp = pq * ps + (1 - pq) * (1 - ps)
                    perm = np.empty(N_PERM)
                    for t in range(N_PERM):
                        perm[t] = (a == prng.permutation(b)).sum()
                    arows.append(
                        dict(
                            variant=v,
                            region=region,
                            snrna_reference=ref,
                            gene_set=gset,
                            n_genes=n,
                            n_agree=obs,
                            agree_rate=obs / n,
                            p_spatial_worker_up=pq,
                            p_snrna_worker_up=ps,
                            chance_agree_rate=exp,
                            chance_n_agree=exp * n,
                            perm_mean_n_agree=float(perm.mean()),
                            perm_p_ge=float((1 + (perm >= obs).sum()) / (1 + N_PERM)),
                        )
                    )
    agr = pd.DataFrame(arows)
    agr.to_csv(OUT / "direction_agreement.csv", index=False)

    # ---- 4. composition on the expression yardstick, and vice versa ----
    comp = pd.read_csv(COMPOSITION)
    z = {}
    for b in ads:
        o = ads[b].obs[["slice_name", "z_level"]].drop_duplicates("slice_name")
        z.update(
            dict(zip(o.slice_name.astype(str), o.z_level.astype(float), strict=True))
        )
    missing = set(comp.slice_name) - set(z)
    assert not missing, f"{len(missing)} composition sections lack z_level"
    comp["z_level"] = comp.slice_name.map(z)
    P = comp.pivot(index="slice_name", columns="cluster", values="prop").fillna(0)
    # The registry's canonical "Glia" is one vote group; slice_composition.csv
    # splits Astrocyte-like and Perineurial glia out of it, so add the pool.
    P[GLIA_POOL] = P[[c for c in GLIA_PARTS if c in P.columns]].sum(1)
    classes = list(P.columns)
    brain_of = comp.drop_duplicates("slice_name").set_index("slice_name").brain
    Sc = {}
    for b in ("queen", "worker"):
        sl = [s for s in P.index if brain_of[s] == b]
        sl = sorted(sl, key=lambda s: z[s])
        Sc[b] = P.loc[sl].to_numpy(float)
    assert all((Sc[b] > 0).all() for b in Sc), "a class share is 0; pseudocount needed"
    crng = np.random.default_rng(COMP_SEED)
    cst = {b: draw_starts(Sc[b].shape[0], crng) for b in Sc}
    cfc = np.log2(Sc["worker"].mean(0) / Sc["queen"].mean(0))
    cd = np.vstack([signed_splits(Sc[b], cst[b], pseudo=False) for b in Sc])
    cp95 = np.percentile(np.abs(cd), CUT_Q, axis=0)
    csdr = sd_ratio(Sc["queen"], Sc["worker"])
    crows = [
        dict(
            cls=c,
            mean_queen=Sc["queen"][:, i].mean(),
            mean_worker=Sc["worker"][:, i].mean(),
            log2fc_worker_over_queen=cfc[i],
            null_p95=cp95[i],
            ratio=abs(cfc[i]) / cp95[i],
            exceeds=bool(abs(cfc[i]) > cp95[i]),
            sd_ratio=csdr[i],
            n_queen_sections=Sc["queen"].shape[0],
            n_worker_sections=Sc["worker"].shape[0],
        )
        for i, c in enumerate(classes)
    ]
    compt = pd.DataFrame(crows)
    compt.to_csv(OUT / "composition_benchmark.csv", index=False)

    yrows = []
    for v in ("original", "centred"):
        for region in regions:
            sub = tab[(tab.variant == v) & (tab.region == region)]
            if sub.empty:
                continue
            yrows.append(
                dict(
                    quantity="expression",
                    variant=v,
                    unit=region,
                    n_items=len(sub),
                    median_split_half_ratio=sub.ratio.median(),
                    n_split_half_exceed=int((sub.ratio > 1).sum()),
                    # the SD ratio is on the linear scale and is not centred
                    median_sd_ratio=sub.sd_ratio.median()
                    if v == "original"
                    else np.nan,
                    n_sd_ratio_gt1=(
                        int((sub.sd_ratio > 1).sum()) if v == "original" else np.nan
                    ),
                )
            )
    yrows.append(
        dict(
            quantity="composition",
            variant="original",
            unit="whole_section_classes",
            n_items=len(compt),
            median_split_half_ratio=compt.ratio.median(),
            n_split_half_exceed=int(compt.exceeds.sum()),
            median_sd_ratio=compt.sd_ratio.median(),
            n_sd_ratio_gt1=int((compt.sd_ratio > 1).sum()),
        )
    )
    yard = pd.DataFrame(yrows)
    yard.to_csv(OUT / "yardstick_comparison.csv", index=False)

    write_summary(wide, spec, pd.DataFrame(jac_rows), isum, agr, compt, yard, T, totals)


def write_summary(wide, spec, jac, isum, agr, compt, yard, T, totals):
    w = wide.set_index("region")

    def rng_(col, regs):
        v = w.loc[regs, col]
        return f"{int(v.min())} to {int(v.max())}"

    L = ["# Caste benchmark: offset, region specificity, chance, yardstick", ""]
    L.append(
        f"Original exceed counts reproduce the shipped table exactly "
        f"(asserted). Robust depth target T = {T:.0f} counts/cell "
        f"(pooled median; queen median {np.median(totals['queen']):.0f}, "
        f"worker {np.median(totals['worker']):.0f})."
    )
    L += ["", "## 1. Offset removed (offset_benchmark_counts.csv)", ""]
    kc_offs = ", ".join(f"{w.loc[r, 'offset_original']:+.3f}" for r in KC)
    L.append(
        f"- KC compartments: {rng_('n_exceed_original', KC)} genes exceed "
        f"originally, {rng_('n_exceed_centred', KC)} after subtracting the "
        f"per-region median log2FC (offsets "
        f"{kc_offs}"
        f"), {rng_('n_exceed_robust', KC)} with pooled-depth normalisation, "
        f"{rng_('n_exceed_robust_centred', KC)} with both."
    )
    ol = ["lamina", "medulla", "lobula"]
    L.append(
        f"- Optic neuropils: {rng_('n_exceed_original', ol)} originally, "
        f"{rng_('n_exceed_centred', ol)} centred, "
        f"{rng_('n_exceed_robust_centred', ol)} robust+centred."
    )
    L.append(
        "- Centring removes any caste shift common to most genes as well as "
        "the run offset, so the centred count is a floor, not the truth."
    )
    L.append(
        "- Centring removes only a uniform offset. A gene-specific run or "
        "probe effect is indistinguishable from caste with one brain per run "
        "and no within-brain null can absorb it."
    )
    L.append(
        f"- The per-region medians here ({w.offset_original.min():+.3f} to "
        f"{w.offset_original.max():+.3f}; whole section "
        f"{w.loc[WHOLE, 'offset_original']:+.3f}) are not the paper's +0.083 "
        "(67/101 genes), which is a different QC-figure quantity."
    )
    L += ["", "## 2. Region specificity (six matched regions)", ""]
    for v in ("original", "centred"):
        s = spec[(spec.variant == v) & (spec.scope == "six")].iloc[0]
        j = jac[jac.variant == v]
        jk = j[j.region_a.isin(KC) & j.region_b.isin(KC)]
        L.append(
            f"- {v}: {s.n_exceed_exactly_one} genes exceed in exactly one "
            f"region, {s.n_exceed_all} in all six (expected "
            f"{s.exceed_in_all_expected_independent:.1f} if independent), "
            f"{s.n_exceed_none} in none; {s.n_ge2_sign_discordant} of "
            f"{s.n_exceed_ge2} multi-region genes change sign. Jaccard "
            f"median {j.jaccard.median():.2f} "
            f"(random {j.jaccard_random_mean.median():.2f}); among KC "
            f"compartments {jk.jaccard.min():.2f}-{jk.jaccard.max():.2f}, "
            f"log2FC r {jk.pearson_log2fc.min():.2f}-"
            f"{jk.pearson_log2fc.max():.2f}."
        )
    for r in isum.itertuples():
        L.append(
            f"- interaction ({r.variant}): {r.n_interaction_exceeds}/{r.n_genes} "
            f"genes have an across-region SD of log2FC above the 95th "
            f"percentile of the split-draw SD; {r.n_shared_exceeds} have a "
            f"six-region mean log2FC above its null "
            f"(region_interaction_summary.csv)."
        )
    for v in ("original", "centred"):
        j = jac[jac.variant == v]
        kk = j.region_a.isin(KC) & j.region_b.isin(KC)
        oo = ~j.region_a.isin(KC) & ~j.region_b.isin(KC)
        ko = ~kk & ~oo
        L.append(
            f"- {v} Jaccard by pair type (observed / random-set mean): "
            f"KC-KC {j[kk].jaccard.median():.2f}/"
            f"{j[kk].jaccard_random_mean.median():.2f}, "
            f"optic-optic {j[oo].jaccard.median():.2f}/"
            f"{j[oo].jaccard_random_mean.median():.2f}, "
            f"KC-optic {j[ko].jaccard.median():.2f}/"
            f"{j[ko].jaccard_random_mean.median():.2f}; log2FC r KC-KC "
            f"{j[kk].pearson_log2fc.median():.2f}, KC-optic "
            f"{j[ko].pearson_log2fc.median():.2f}."
        )
    L.append(
        "- Interpretation (written after reading these numbers, not "
        "computed): the three Kenyon compartments share one caste signature "
        "(overlap well above random sets, log2FC r ~0.8-0.9); Kenyon vs optic "
        "neuropils overlap no more than random sets of the same size. So the "
        "data support 'differs between Kenyon and optic neuropils', not "
        "'specific to each region'."
    )
    L += ["", "## 3. Direction agreement vs chance (direction_agreement.csv)", ""]
    for v in ("original", "centred"):
        for r in KC:
            a = agr[
                (agr.variant == v)
                & (agr.region == r)
                & (agr.snrna_reference == "Kenyon_cells")
                & (agr.gene_set == "exceeding")
            ]
            if a.empty:
                continue
            a = a.iloc[0]
            L.append(
                f"- {v} {r}: {a.n_agree}/{a.n_genes} agree "
                f"({a.agree_rate:.2f}); chance {a.chance_agree_rate:.2f} "
                f"(p_spatial_up {a.p_spatial_worker_up:.2f}, p_snRNA_up "
                f"{a.p_snrna_worker_up:.2f}); permutation p {a.perm_p_ge:.4f}."
            )
    L.append(
        "- The three KC compartments share most exceeding genes (Jaccard "
        "0.55-0.66), so their permutation p values are one comparison, "
        "not three."
    )
    L.append(
        "- The interaction null compares ~15 vs ~15 sections against a "
        "30 vs 35 caste contrast, so its SD is inflated and the "
        "interaction count is a floor."
    )
    L += [
        "",
        "## 4. One yardstick (composition_benchmark.csv, yardstick_comparison.csv)",
        "",
    ]
    for c in compt.itertuples():
        L.append(
            f"- {c.cls}: |log2FC| {abs(c.log2fc_worker_over_queen):.3f} vs "
            f"split-half p95 {c.null_p95:.3f} (ratio {c.ratio:.2f}); "
            f"SD ratio {c.sd_ratio:.2f}."
        )
    y = yard.set_index(["quantity", "variant", "unit"])
    for r in KC + ol:
        e = y.loc[("expression", "original", r)]
        L.append(
            f"- expression {r}: median split-half ratio "
            f"{e.median_split_half_ratio:.2f}, median SD ratio "
            f"{e.median_sd_ratio:.2f} ({e.n_sd_ratio_gt1}/{e.n_items} > 1)."
        )
    (OUT / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
