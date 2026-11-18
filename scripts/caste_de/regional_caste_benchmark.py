#!/usr/bin/env python3
"""Benchmark per-region caste panel-gene contrasts against within-brain
split-half variation and snRNA-seq (n=3) direction.

    PYTHONPATH=src .venv/bin/python scripts/caste_de/regional_caste_benchmark.py

Normalisation and contrast are reused verbatim from
``scripts/caste_de/si_regional_degs.py::si_regional_degs``: the object's ``.X``
(log1p-normalised), mean per section (``slice_name`` crop, sections with fewer
than MIN_CELLS_PER_SLICE=5 cells in the region dropped), mean of section
means per brain, ``log2((worker + 0.01) / (queen + 0.01))``.  Sign convention:
**positive = worker-higher** (paper convention for the spatial arm).  The
snRNA DESeq2 table is queen-vs-worker (positive = queen-up) and is flipped
here so both are worker/queen.

Null: within each brain and region, sections ordered by ``z_level`` (serial
anterior-posterior order).  A split-half is a random contiguous block of
floor(k/2) sections vs the remaining ceil(k/2).  N_SPLITS draws per brain,
|log2FC| pooled over eligible brains.  Brains with < MIN_SECTIONS sections
in a region contribute no null draws and the region is flagged.

Output only under results/caste_de/B_caste_benchmark/ (snRNA directions
from the trial-blocked fit, results/caste_de/de_coarse_blocked.csv).
"""


import anndata as ad
import numpy as np
import pandas as pd

from bimp_st.genes import read_panel_crosswalk  # noqa: E402
from bimp_st.nine_region_common import (
    DESCRIPTIVE_REGIONS,
    NO_PVALUE_REGIONS,
)
from bimp_st.paths import (
    CASTE_DE_COARSE,  # noqa: E402
    CASTE_DE_DIR,  # noqa: E402
    PANEL_CROSSWALK,  # noqa: E402
    REPO,  # noqa: E402
)

OUT = CASTE_DE_DIR / "B_caste_benchmark"
ATLAS = {
    "queen": REPO / "results/nine_region/atlas/queen_aligned_clustered_9region.h5ad",
    "worker": REPO / "results/nine_region/atlas/worker_aligned_clustered_9region.h5ad",
}
DESEQ = CASTE_DE_COARSE  # ~ trial + caste
MAPPING = PANEL_CROSSWALK

MIN_CELLS_PER_SLICE = 5  # as si_regional_degs.py
MIN_SECTIONS = 4
N_SPLITS = 200
SEED = 20261005
PSEUDO = 0.01  # as si_regional_degs.py
CUT_Q = 95

# Region -> snRNA central-brain cell type. The snRNA central-brain atlas has
# no optic-lobe nuclei and no lamina/medulla/lobula-specific type, and no type
# that corresponds to the antennal lobe or central complex as a whole, so those
# are NA rather than forced.  Kenyon-cell compartments map to Kenyon_cells.
REGION_CELLTYPE = {
    "kc_oc": "Kenyon_cells",
    "kc_ic": "Kenyon_cells",
    "kc_nc": "Kenyon_cells",
    "lamina": None,
    "medulla": None,
    "lobula": None,
    "al": None,
    "central_complex": None,
}


def l2fc(w, q):
    return np.log2((w + PSEUDO) / (q + PSEUDO))


def section_means(adata, X, region):
    """(sections ordered by z_level) x genes matrix of mean expression."""
    col = f"in_{region}"
    if col not in adata.obs.columns:
        return np.empty((0, X.shape[1]))
    mask = adata.obs[col].to_numpy(dtype=bool, na_value=False)
    if not mask.any():
        return np.empty((0, X.shape[1]))
    obs = adata.obs.loc[mask, ["slice_name", "z_level"]]
    Xr = X[mask]
    rows, z = [], []
    for _sl, idx in obs.groupby("slice_name", observed=True).indices.items():
        if len(idx) < MIN_CELLS_PER_SLICE:
            continue
        rows.append(Xr[idx].mean(axis=0))
        z.append(float(obs["z_level"].iloc[idx[0]]))
    if not rows:
        return np.empty((0, X.shape[1]))
    order = np.argsort(z)
    return np.vstack(rows)[order]


def split_half_null(S, rng):
    """|log2FC| of random contiguous half-block vs remainder; (N_SPLITS, genes)."""
    k = S.shape[0]
    h = k // 2
    out = np.empty((N_SPLITS, S.shape[1]))
    for i in range(N_SPLITS):
        start = rng.integers(0, k - h + 1)
        sel = np.zeros(k, bool)
        sel[start : start + h] = True
        out[i] = np.abs(l2fc(S[sel].mean(0), S[~sel].mean(0)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    ads, Xs = {}, {}
    for brain, p in ATLAS.items():
        a = ad.read_h5ad(p)
        X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        ads[brain], Xs[brain] = a, X.astype(np.float64)
    genes = list(ads["queen"].var_names)
    assert genes == list(ads["worker"].var_names)
    short = [g.split("|")[0] for g in genes]
    m = read_panel_crosswalk().set_index("np31_id")
    v43 = [m.v43_id.get(s) for s in short]
    loc = [m.loc_id.get(s) for s in short]

    de = pd.read_csv(DESEQ, index_col=0)
    de = de.drop_duplicates(["cell_type", "gene"]).set_index(["cell_type", "gene"])

    def sn_lookup(ct, gid):
        """worker/queen-signed snRNA log2FC, padj; NaN if untested."""
        if ct is None or (ct, gid) not in de.index:
            return np.nan, np.nan
        r = de.loc[(ct, gid)]
        return -float(r.log2FoldChange), r.padj

    rows, meta = [], []
    for region in DESCRIPTIVE_REGIONS:
        S = {b: section_means(ads[b], Xs[b], region) for b in ads}
        nsec = {b: S[b].shape[0] for b in S}
        elig = [b for b in S if nsec[b] >= MIN_SECTIONS]
        flag = (
            "ok"
            if len(elig) == 2
            else "null_from_" + "+".join(elig) + "_only"
            if elig
            else "null_unavailable_lt%d_sections" % MIN_SECTIONS
        )
        meta.append(
            dict(
                region=region,
                queen_sections=nsec["queen"],
                worker_sections=nsec["worker"],
                null_flag=flag,
            )
        )
        if nsec["queen"] == 0 or nsec["worker"] == 0:
            print(f"{region}: absent in a brain {nsec}; skipped")
            continue
        fc = l2fc(S["worker"].mean(0), S["queen"].mean(0))
        if elig:
            null = np.vstack([split_half_null(S[b], rng) for b in elig])
            p95 = np.percentile(null, CUT_Q, axis=0)
        else:
            p95 = np.full(len(genes), np.nan)
        ct = REGION_CELLTYPE[region]
        for gi, g in enumerate(genes):
            sn_fc, sn_padj = sn_lookup(ct, v43[gi])
            sn_all, _ = sn_lookup("all", v43[gi])
            ratio = (abs(fc[gi]) / p95[gi]) if p95[gi] > 0 else np.nan
            exceeds = bool(ratio > 1) if not np.isnan(ratio) else False

            def dirn(x):
                return (
                    "NA"
                    if np.isnan(x) or x == 0
                    else "worker_up"
                    if x > 0
                    else "queen_up"
                )

            sn_dir = dirn(sn_fc)
            ours = "worker_up" if fc[gi] > 0 else "queen_up" if fc[gi] < 0 else "none"
            agrees = np.nan if sn_dir == "NA" else bool(sn_dir == ours)
            rows.append(
                dict(
                    region=region,
                    gene=g,
                    gene_id_v43=v43[gi],
                    gene_short=loc[gi],
                    log2fc_worker_over_queen=round(fc[gi], 4),
                    null_p95=round(p95[gi], 4) if not np.isnan(p95[gi]) else np.nan,
                    ratio=round(ratio, 3) if not np.isnan(ratio) else np.nan,
                    exceeds=exceeds,
                    snrna_celltype=ct or "NA",
                    snrna_log2fc_worker_over_queen=sn_fc,
                    snrna_padj=sn_padj,
                    snrna_direction=sn_dir,
                    agrees=agrees,
                    snrna_all_nuclei_direction=dirn(sn_all),
                    agrees_all_nuclei=(
                        np.nan if dirn(sn_all) == "NA" else bool(dirn(sn_all) == ours)
                    ),
                    null_flag=flag,
                    no_pvalue_region=region in NO_PVALUE_REGIONS,
                    n_sections_queen=nsec["queen"],
                    n_sections_worker=nsec["worker"],
                )
            )
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "caste_benchmark_table.csv", index=False)

    out = []
    for region, g in tab.groupby("region", sort=False):
        mt = g.loc[g.snrna_direction != "NA"]
        mall = g.loc[g.snrna_all_nuclei_direction != "NA"]
        out.append(
            dict(
                region=region,
                n_genes=len(g),
                queen_sections=int(g.n_sections_queen.iloc[0]),
                worker_sections=int(g.n_sections_worker.iloc[0]),
                null_flag=g.null_flag.iloc[0],
                n_exceed_null=int(g.exceeds.sum()),
                snrna_celltype=g.snrna_celltype.iloc[0],
                n_with_snrna=len(mt),
                n_exceed_and_agree=int((mt.exceeds & (mt.agrees.eq(True))).sum()),
                n_exceed_and_disagree=int((mt.exceeds & (mt.agrees.eq(False))).sum()),
                n_exceed_all_nuclei_and_agree=int(
                    (mall.exceeds & (mall.agrees_all_nuclei.eq(True))).sum()
                ),
                n_with_all_nuclei=len(mall),
                descriptive_only_no_pvalue=region in NO_PVALUE_REGIONS,
                median_abs_log2fc=round(g.log2fc_worker_over_queen.abs().median(), 3),
                median_null_p95=round(g.null_p95.median(), 3),
            )
        )
    cnt = pd.DataFrame(out)
    cnt.to_csv(OUT / "caste_benchmark_counts.csv", index=False)
    pd.DataFrame(meta).to_csv(OUT / "caste_benchmark_sections.csv", index=False)
    with open(OUT / "caste_benchmark_counts.md", "w") as f:
        f.write(
            "| region | sections (Q/W) | genes exceeding null | snRNA cell type | "
            "genes with snRNA | exceed and agree | exceed, all-nuclei and agree |\n"
        )
        f.write("|---|---|---|---|---|---|---|\n")
        for r in out:
            star = "*" if r["descriptive_only_no_pvalue"] else ""
            f.write(
                f"| {r['region']}{star} | {r['queen_sections']}/"
                f"{r['worker_sections']} | {r['n_exceed_null']}/{r['n_genes']} | "
                f"{r['snrna_celltype']} | {r['n_with_snrna'] or 'NA'} | "
                f"{r['n_exceed_and_agree'] if r['n_with_snrna'] else 'NA'} | "
                f"{r['n_exceed_all_nuclei_and_agree']}/{r['n_with_all_nuclei']} |\n"
            )
        f.write("\n* descriptive only (NO_PVALUE_REGIONS).\n")
    print(cnt.to_string())


if __name__ == "__main__":
    main()
