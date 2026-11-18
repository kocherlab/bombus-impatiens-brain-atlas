"""Differential expression and composition analysis for BIMP snRNA-seq.

Commands:
    region-degs  -- 1-vs-rest Wilcoxon DEGs per cluster (region comparison)
    tau          -- Tau specificity scores for region DEGs
    composition  -- Cluster composition analysis across samples

Adapted from I. Traniello's 2025 notebooks.
All comparisons are cluster-based (region), not caste-based.
"""

from pathlib import Path

import typer

app = typer.Typer(help="Differential analysis for BIMP snRNA-seq")

# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path(__file__).parent
DATA_DIR = OUTPUT_DIR / "data"
FIGURES_DIR = OUTPUT_DIR / "figures"


# =============================================================================
# REGION-DEGS: 1-vs-rest Wilcoxon per cluster
# =============================================================================


@app.command()
def region_degs(
    input_file: str = typer.Option("processed", help="Input h5ad stem in data/"),
    cell_type_key: str = typer.Option("leiden", help="Grouping key for clusters"),
    fdr_threshold: float = typer.Option(0.05, help="FDR threshold for significance"),
    n_genes: int = typer.Option(0, help="Max genes per cluster (0=all)"),
):
    """1-vs-rest Wilcoxon rank-sum DEGs per cluster.

    For each cluster, tests all genes against the rest of the dataset.
    Outputs a combined CSV with all results and per-cluster volcano plots.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    input_path = DATA_DIR / f"{input_file}.h5ad"
    if not input_path.exists():
        typer.echo(f"ERROR: File not found: {input_path}")
        raise typer.Exit(1)

    typer.echo(f"Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)
    typer.echo(f"  Cells: {adata.n_obs:,}")

    groups = sorted(
        adata.obs[cell_type_key].unique(), key=lambda x: int(x) if x.isdigit() else x
    )
    typer.echo(f"  Clusters ({len(groups)}): {list(groups)}")

    # Run 1-vs-rest Wilcoxon (may already be in .uns from process step)
    typer.echo("Running Wilcoxon 1-vs-rest per cluster...")
    sc.tl.rank_genes_groups(adata, groupby=cell_type_key, method="wilcoxon")

    all_results = []
    for group in groups:
        result = sc.get.rank_genes_groups_df(adata, group=str(group))
        result["cell_type"] = group
        if n_genes > 0:
            result = result.head(n_genes)
        all_results.append(result)

    df = pd.concat(all_results, ignore_index=True)
    df["significant"] = df["pvals_adj"] < fdr_threshold

    n_sig = df["significant"].sum()
    typer.echo(f"\nSignificant DEGs (FDR < {fdr_threshold}): {n_sig:,} / {len(df):,}")

    # Per-cluster summary
    summary = df.groupby("cell_type")["significant"].sum().sort_values(ascending=False)
    typer.echo("\nDEGs per cluster:")
    for ct, n in summary.items():
        typer.echo(f"  {ct}: {n}")

    output_path = DATA_DIR / "region_degs.csv"
    df.to_csv(output_path, index=False)
    typer.echo(f"\nSaved: {output_path}")

    # Volcano plots (top 12 clusters by n DEGs)
    top_clusters = summary.head(12).index
    if len(top_clusters) > 0:
        n_cols = min(4, len(top_clusters))
        n_rows = (len(top_clusters) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_2d(axes).ravel()

        for i, ct in enumerate(top_clusters):
            ax = axes[i]
            ct_df = df[df["cell_type"] == ct].copy()
            ct_df["-log10(p)"] = -np.log10(ct_df["pvals_adj"].clip(lower=1e-300))
            sig_mask = ct_df["significant"]
            ax.scatter(
                ct_df.loc[~sig_mask, "logfoldchanges"],
                ct_df.loc[~sig_mask, "-log10(p)"],
                s=2,
                alpha=0.3,
                c="grey",
            )
            ax.scatter(
                ct_df.loc[sig_mask, "logfoldchanges"],
                ct_df.loc[sig_mask, "-log10(p)"],
                s=4,
                alpha=0.5,
                c="red",
            )
            ax.set_title(f"Cluster {ct}", fontsize=9)
            ax.set_xlabel("logFC")
            ax.set_ylabel("-log10(p_adj)")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "region_volcanos.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        typer.echo("Saved: figures/region_volcanos.png")


# =============================================================================
# TAU
# =============================================================================


@app.command()
def tau(
    input_file: str = typer.Option("processed", help="Input h5ad stem in data/"),
    deg_file: str = typer.Option("region_degs", help="DEG CSV stem in data/"),
    cell_type_key: str = typer.Option("leiden", help="Grouping key for clusters"),
    fdr_threshold: float = typer.Option(0.05, help="FDR threshold"),
    tau_threshold: float = typer.Option(0.8, help="Tau threshold for specificity"),
):
    """Compute tau specificity scores for region DEGs.

    Tau measures how specific a gene's expression is to one cluster (0 = ubiquitous,
    1 = perfectly specific). Identifies concordant DEGs where the cluster showing
    differential expression is also the cluster with maximum expression.
    """
    import numpy as np
    import pandas as pd
    import scanpy as sc

    input_path = DATA_DIR / f"{input_file}.h5ad"
    deg_path = DATA_DIR / f"{deg_file}.csv"

    if not input_path.exists():
        typer.echo(f"ERROR: File not found: {input_path}")
        raise typer.Exit(1)
    if not deg_path.exists():
        typer.echo(f"ERROR: DEG file not found: {deg_path}")
        typer.echo("Run `region-degs` first")
        raise typer.Exit(1)

    typer.echo(f"Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)
    degs = pd.read_csv(deg_path)

    typer.echo("Computing mean expression per cluster...")
    groups = sorted(
        adata.obs[cell_type_key].unique(),
        key=lambda x: int(x) if str(x).isdigit() else x,
    )
    mean_expr = {}
    for group in groups:
        mask = adata.obs[cell_type_key] == group
        sub = adata[mask]
        if hasattr(sub.X, "toarray"):
            mean_expr[group] = np.array(sub.X.mean(axis=0)).ravel()
        else:
            mean_expr[group] = sub.X.mean(axis=0).ravel()

    mean_df = pd.DataFrame(mean_expr, index=adata.var_names)

    typer.echo("Computing tau scores...")
    # Tau = 1 - mean(x_hat_i) where x_hat_i = x_i / max(x)
    tau_scores = {}
    max_cell_type = {}
    for gene in mean_df.index:
        expr = mean_df.loc[gene].values
        max_val = expr.max()
        if max_val == 0:
            tau_scores[gene] = 0.0
            max_cell_type[gene] = groups[0]
            continue
        x_hat = expr / max_val
        n = len(expr)
        tau_scores[gene] = (1 - x_hat).sum() / (n - 1) if n > 1 else 0.0
        max_cell_type[gene] = groups[np.argmax(expr)]

    # Merge with DEGs
    degs["tau_score"] = degs["names"].map(tau_scores)
    degs["specific_cluster"] = degs["names"].map(max_cell_type)
    degs["concordant"] = degs["cell_type"].astype(str) == degs[
        "specific_cluster"
    ].astype(str)

    # Filter significant + high-tau + concordant
    sig_mask = (
        (degs["pvals_adj"] < fdr_threshold)
        & (degs["tau_score"] > tau_threshold)
        & degs["concordant"]
    )
    n_concordant = sig_mask.sum()
    typer.echo(
        f"\nConcordant DEGs (FDR<{fdr_threshold}, tau>{tau_threshold}): {n_concordant}"
    )

    # Summary by cluster
    if n_concordant > 0:
        concordant = degs[sig_mask]
        for ct, group_df in concordant.groupby("cell_type"):
            typer.echo(f"  Cluster {ct}: {len(group_df)} genes")

    output_path = DATA_DIR / "tau_degs.csv"
    degs.to_csv(output_path, index=False)
    typer.echo(f"\nSaved: {output_path}")

    # Top concordant markers
    if n_concordant > 0:
        top = concordant.sort_values("tau_score", ascending=False).head(50)
        md_path = DATA_DIR / "top_tau_concordant_degs.md"
        with open(md_path, "w") as f:
            f.write("# Top concordant region DEGs (tau > 0.8)\n\n")
            f.write("| Gene | Cluster | Tau | logFC | p_adj |\n")
            f.write("|------|---------|-----|-------|-------|\n")
            for _, row in top.iterrows():
                f.write(
                    f"| {row['names']} | {row['cell_type']} | "
                    f"{row['tau_score']:.3f} | {row['logfoldchanges']:.3f} | "
                    f"{row['pvals_adj']:.2e} |\n"
                )
        typer.echo(f"Saved: {md_path}")


# =============================================================================
# COMPOSITION
# =============================================================================


@app.command()
def composition(
    input_file: str = typer.Option("processed", help="Input h5ad stem in data/"),
    cell_type_key: str = typer.Option("leiden", help="Grouping key for clusters"),
    sample_key: str = typer.Option("sample", help="obs column for sample"),
):
    """Cluster composition analysis across samples.

    Computes cluster proportions per sample, runs chi-squared test of
    independence, and generates composition plots.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import scanpy as sc
    from scipy.stats import chi2_contingency

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results_dir = DATA_DIR / "composition_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    input_path = DATA_DIR / f"{input_file}.h5ad"
    if not input_path.exists():
        typer.echo(f"ERROR: File not found: {input_path}")
        raise typer.Exit(1)

    typer.echo(f"Loading {input_path.name}...")
    adata = sc.read_h5ad(input_path)
    typer.echo(f"  Cells: {adata.n_obs:,}")

    # Build composition table: samples x clusters
    comp = pd.crosstab(adata.obs[sample_key], adata.obs[cell_type_key])
    typer.echo(f"Composition table: {comp.shape[0]} samples x {comp.shape[1]} clusters")
    typer.echo(f"\n{comp}")

    # Proportions
    props = comp.div(comp.sum(axis=1), axis=0)
    props.to_csv(results_dir / "proportions_by_sample.csv")
    typer.echo("\nSaved: proportions_by_sample.csv")

    # Raw counts
    comp.to_csv(results_dir / "counts_by_sample.csv")
    typer.echo("Saved: counts_by_sample.csv")

    # Chi-squared test of independence
    chi2, p, dof, expected = chi2_contingency(comp)
    typer.echo("\nChi-squared test of independence:")
    typer.echo(f"  chi2={chi2:.2f}, p={p:.2e}, dof={dof}")

    # Save expected counts
    expected_df = pd.DataFrame(expected, index=comp.index, columns=comp.columns)
    expected_df.to_csv(results_dir / "expected_counts.csv")

    # Per-cluster chi-squared residuals (observed - expected) / sqrt(expected)
    residuals = (comp - expected_df) / (expected_df**0.5)
    residuals.to_csv(results_dir / "chi2_residuals.csv")
    typer.echo("Saved: chi2_residuals.csv")

    # Summary stats
    summary = {
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "n_samples": comp.shape[0],
        "n_clusters": comp.shape[1],
        "total_cells": int(comp.sum().sum()),
    }
    pd.Series(summary).to_csv(results_dir / "summary.csv")

    # Composition barplot
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # Stacked bar: proportions by sample
    props.plot.bar(stacked=True, ax=axes[0], legend=False)
    axes[0].set_title("Cluster proportions by sample")
    axes[0].set_xlabel("Sample")
    axes[0].set_ylabel("Proportion")
    axes[0].legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=6, title="Cluster"
    )

    # Heatmap of residuals
    im = axes[1].imshow(residuals.values, aspect="auto", cmap="RdBu_r", vmin=-5, vmax=5)
    axes[1].set_xticks(range(residuals.shape[1]))
    axes[1].set_xticklabels(residuals.columns, rotation=90, fontsize=7)
    axes[1].set_yticks(range(residuals.shape[0]))
    axes[1].set_yticklabels(residuals.index)
    axes[1].set_title("Chi-squared residuals")
    axes[1].set_xlabel("Cluster")
    axes[1].set_ylabel("Sample")
    plt.colorbar(im, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    typer.echo("Saved: figures/composition.png")


if __name__ == "__main__":
    app()
