#!/usr/bin/env python3
"""Aggregate bento LP parquets to per-gene caste stats + confound covariates.

Computes the per-gene caste-differential localization statistics (genes with a
significant caste difference, their direction, and their dominant pattern), and
additionally extracts the segmentation/depth covariates needed to test whether
the caste difference tracks a mechanical confound.

Replicate unit = Xenium region (15 queen, 13 worker), matching
pipelines/subcellular/bento_differential.py. Test = Mann-Whitney U on
region-level LP fractions, BH-corrected across the 100 genes per pattern.

Denominator: bento's LP flags are multi-label 0/1 and many cell-gene rows carry
no flag. Fractions are among CLASSIFIED rows (at least one flag), a pattern
taking precedence over "none" (``none`` = rows flagged "none" and nothing else),
as in regroup_common.fractions; ``unclassified`` is the share of all rows with no
flag. The per-(brain, region, gene) counts come from ``--counts``
(region_gene_counts.csv), run first.

Caveat carried downstream: the 28 regions are sections of one queen and one
worker brain, run on two different instrument runs (queen 20251204, worker
20260115). They are technical, not biological, replicates.

Writes:
    results/bento/differential/region_gene_counts.csv   (--counts)
    results/bento/differential/pergene_caste_stats.csv
    results/bento/differential/region_gene_means.csv    (classified fractions)
    results/bento/differential/region_covariates.csv

Usage:
    PYTHONPATH=src .venv/bin/python scripts/subcellular/bento_pergene_stats.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bimp_st.paths import CANONICAL_TANGRAM  # noqa: E402

OUT = REPO / "results/bento/differential"
PATTERNS = ["cell_edge", "cytoplasmic", "none", "nuclear", "nuclear_edge"]


def region_gene_means() -> pd.DataFrame:
    """Mean LP flag per (brain, region, gene) over ALL rows, one parquet at a time.

    The all-rows denominator, for comparison only; main() uses
    classified_region_gene_means.
    """
    parquets = sorted(REPO.glob("results/bento/*/output-XETG*/lp.parquet"))
    if not parquets:
        raise FileNotFoundError("No lp.parquet files found under results/bento/")
    print(f"Loading {len(parquets)} LP parquets...", flush=True)

    cols = ["feature_name", "gene_name", "brain", "region", "cell_type", *PATTERNS]
    out, n_cells = [], []
    for p in parquets:
        df = pd.read_parquet(p, columns=cols)
        # Match bento_differential.py exactly: drop Xenium controls, drop cells
        # with no Tangram cell-type join.
        df = df[~df["feature_name"].str.startswith(("NegControl", "Unassigned"))]
        df = df[df["cell_type"] != "Unknown"]
        if not len(df):
            continue
        g = df.groupby(["brain", "region", "gene_name"], observed=True)[PATTERNS]
        out.append(g.mean().reset_index())
        n_cells.append(
            df.groupby(["brain", "region"], observed=True)["gene_name"]
            .size()
            .reset_index(name="n_obs")
        )
        print(f"  {p.parent.name}: {len(df):,} obs", flush=True)

    rgm = pd.concat(out, ignore_index=True)
    counts = pd.concat(n_cells, ignore_index=True)
    print(f"\nregion x gene means: {rgm.shape}", flush=True)
    print(counts.groupby("brain")["n_obs"].agg(["count", "sum"]), flush=True)
    return rgm


def region_gene_counts() -> pd.DataFrame:
    """Per (brain, region, gene) counts of LP rows, classified rows and pattern flags.

    bento's LP flags are multi-label 0/1 and about 60% of rows carry none of them.
    ``n_obs`` counts every row, ``n_classified`` the rows with at least one flag,
    ``none_only`` the rows flagged "none" with no other pattern. Same row filters
    as region_gene_means().
    """
    parquets = sorted(REPO.glob("results/bento/*/output-XETG*/lp.parquet"))
    cols = ["feature_name", "gene_name", "brain", "region", "cell_type", *PATTERNS]
    out = []
    for p in parquets:
        df = pd.read_parquet(p, columns=cols)
        df = df[~df["feature_name"].str.startswith(("NegControl", "Unassigned"))]
        df = df[df["cell_type"] != "Unknown"]
        if not len(df):
            continue
        flags = df[PATTERNS].astype(int)
        other = flags.drop(columns="none").sum(axis=1) > 0
        df = df.assign(
            n_obs=1,
            n_classified=(other | (flags["none"] > 0)).astype(int),
            none_only=((flags["none"] > 0) & ~other).astype(int),
        )
        df[PATTERNS] = flags
        g = df.groupby(["brain", "region", "gene_name"], observed=True)
        out.append(
            g[[*PATTERNS, "n_obs", "n_classified", "none_only"]].sum().reset_index()
        )
        print(f"  {p.parent.name}: {len(df):,} obs", flush=True)
    return pd.concat(out, ignore_index=True)


def classified_region_gene_means(rgc: pd.DataFrame) -> pd.DataFrame:
    """Per (brain, region, gene) fractions among classified rows.

    ``none`` is none_only / n_classified, every other pattern its flag count over
    n_classified, so the five sum to one (up to the rare rows with two
    non-"none" flags). ``unclassified`` = 1 - n_classified / n_obs. Region x gene
    cells with no classified row are dropped.
    """
    d = rgc[rgc["n_classified"] > 0].copy()
    out = d[["brain", "region", "gene_name"]].copy()
    for p in PATTERNS:
        num = d["none_only"] if p == "none" else d[p]
        out[p] = num / d["n_classified"]
    out["unclassified"] = 1 - d["n_classified"] / d["n_obs"]
    return out.reset_index(drop=True)


def pergene_stats(rgm: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U per gene per pattern on region-level means, then BH."""
    rows = []
    for gene, gdf in rgm.groupby("gene_name", observed=True):
        q = gdf[gdf["brain"] == "queen"]
        w = gdf[gdf["brain"] == "worker"]
        if len(q) < 3 or len(w) < 3:
            continue
        row = {"gene": gene, "n_queen": len(q), "n_worker": len(w)}
        for pat in PATTERNS:
            qv, wv = q[pat].values, w[pat].values
            u, pval = stats.mannwhitneyu(qv, wv, alternative="two-sided")
            row[f"{pat}_queen"] = qv.mean()
            row[f"{pat}_worker"] = wv.mean()
            row[f"{pat}_diff"] = qv.mean() - wv.mean()
            # Cliff's delta (rank-biserial), positive = queen higher
            row[f"{pat}_delta"] = (2 * u) / (len(qv) * len(wv)) - 1
            row[f"{pat}_pval"] = pval
        rows.append(row)

    res = pd.DataFrame(rows)
    for pat in PATTERNS:
        res[f"{pat}_padj"] = multipletests(res[f"{pat}_pval"], method="fdr_bh")[1]
    return res


def loo_stability(rgm: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-region-out sweep: how much does each headline count move?

    Re-runs the whole per-gene test 28 times, each time dropping one section.
    Shows whether "N of 100 genes" is a stable quantity or an artifact of which
    sections happened to be included (a 27-region subset gives different
    counts, so this is the relevant sensitivity).
    """
    print("\nLeave-one-region-out stability sweep (28 refits)...", flush=True)
    rows = []
    for dropped in sorted(rgm["region"].unique()):
        sub = rgm[rgm["region"] != dropped]
        caste = rgm[rgm["region"] == dropped]["brain"].iloc[0]
        res = pergene_stats(sub)
        for pat in PATTERNS:
            sig = res[f"{pat}_padj"] < 0.05
            rows.append(
                {
                    "pattern": pat,
                    "dropped_region": dropped,
                    "dropped_caste": caste,
                    "n_sig": int(sig.sum()),
                    "n_queen": int((sig & (res[f"{pat}_diff"] > 0)).sum()),
                    "n_worker": int((sig & (res[f"{pat}_diff"] < 0)).sum()),
                }
            )
    loo = pd.DataFrame(rows)
    for pat in PATTERNS:
        d = loo[loo["pattern"] == pat]
        print(
            f"  {pat:14s} sig {d.n_sig.min()}-{d.n_sig.max()}  "
            f"queen {d.n_queen.min()}-{d.n_queen.max()}  "
            f"worker {d.n_worker.min()}-{d.n_worker.max()}",
            flush=True,
        )
    return loo


def region_covariates() -> pd.DataFrame:
    """Per-region segmentation covariates from the tangram h5ad obs."""
    import anndata as ad

    h5 = CANONICAL_TANGRAM
    print(f"\nReading covariates from {h5.name}...", flush=True)
    obs = (
        ad.read_h5ad(h5, backed="r")
        .obs[
            [
                "slice_name",
                "brain",
                "cell_area",
                "nucleus_area",
                "total_counts",
                "segmentation_method",
            ]
        ]
        .copy()
    )

    # obs['region'] is useless ("cell_circles"). slice_name is the bento region
    # name plus optional slide ("_2") and sub-slice ("_1A", "_3C") suffixes, so
    # map each cell to the longest bento region name that prefixes its
    # slice_name rather than trying to strip suffixes by pattern.
    bento_regions = sorted(
        {p.parent.name for p in REPO.glob("results/bento/*/output-XETG*/lp.parquet")},
        key=len,
        reverse=True,
    )
    slices = obs["slice_name"].astype(str)
    lookup = {}
    for s in slices.unique():
        match = next((b for b in bento_regions if s.startswith(b)), None)
        if match:
            lookup[s] = match
    obs["region"] = slices.map(lookup)
    n_unmapped = obs["region"].isna().sum()
    obs = obs[obs["region"].notna()]
    print(
        f"  mapped {len(obs):,} cells to bento regions "
        f"({n_unmapped:,} cells in non-bento slices dropped)",
        flush=True,
    )
    obs["nuc_frac_area"] = obs["nucleus_area"] / obs["cell_area"]
    obs["density"] = obs["total_counts"] / obs["cell_area"]

    cov = (
        obs.groupby(["brain", "region"], observed=True)
        .agg(
            n_cells=("cell_area", "size"),
            cell_area=("cell_area", "mean"),
            nucleus_area=("nucleus_area", "mean"),
            nuc_frac_area=("nuc_frac_area", "mean"),
            total_counts=("total_counts", "mean"),
            density=("density", "mean"),
        )
        .reset_index()
    )
    print(f"  {len(cov)} regions with covariates", flush=True)

    # Segmentation-method composition per caste: nucleus-expansion cells have a
    # geometrically fixed nucleus:cell relationship, so a run difference here
    # would mechanically drive any nuclear-fraction difference.
    seg = (
        obs.groupby(
            ["brain", "segmentation_method"],
            observed=True,
        )
        .size()
        .rename("n")
        .reset_index()
    )
    seg["frac"] = seg["n"] / seg.groupby("brain")["n"].transform("sum")
    print(seg.to_string(), flush=True)
    seg.to_csv(OUT / "segmentation_method_by_caste.csv", index=False)
    return cov


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if "--counts" in sys.argv:
        rgc = region_gene_counts()
        rgc.to_csv(OUT / "region_gene_counts.csv", index=False)
        print(f"Wrote region_gene_counts.csv {rgc.shape}", flush=True)
        return

    counts = OUT / "region_gene_counts.csv"
    if not counts.exists():
        region_gene_counts().to_csv(counts, index=False)
    rgm = classified_region_gene_means(pd.read_csv(counts))
    rgm.to_csv(OUT / "region_gene_means.csv", index=False)

    res = pergene_stats(rgm)
    res.to_csv(OUT / "pergene_caste_stats.csv", index=False)

    print("\n" + "=" * 62, flush=True)
    print("PER-GENE CASTE STATS  (n = %d genes)" % len(res), flush=True)
    print("=" * 62, flush=True)
    for pat in PATTERNS:
        sig = res[f"{pat}_padj"] < 0.05
        up = (sig & (res[f"{pat}_diff"] > 0)).sum()
        dn = (sig & (res[f"{pat}_diff"] < 0)).sum()
        print(
            f"  {pat:14s} sig={sig.sum():3d}  queen>={up:3d}  worker>={dn:3d}",
            flush=True,
        )

    loo = loo_stability(rgm)
    loo.to_csv(OUT / "loo_stability.csv", index=False)

    # Segmentation covariates do not depend on the bento denominator.
    cov_path = OUT / "region_covariates.csv"
    if cov_path.exists():
        print(f"Using cached {cov_path.name} (delete to recompute)", flush=True)
    else:
        region_covariates().to_csv(cov_path, index=False)
    print(f"\nWrote 5 CSVs to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
