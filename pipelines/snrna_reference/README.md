# snRNA-seq Reference Processing

Processes CellBender-filtered single-nucleus RNA-seq data into a clustered, annotated reference for Tangram cell type mapping.

## Pipeline

```
CellBender h5 (per sample)
    → preprocess (QC/MAD/Scrublet/combine) → raw.h5ad
        → process (normalize/HVG/PCA/Harmony/UMAP/Leiden) → processed.h5ad
            ├→ markers (top N per cluster + descriptions)
            ├→ region_degs (1-vs-rest Wilcoxon) → tau (specificity scores)
            └→ composition (cluster proportions across samples)
```

## Scripts

| File | Purpose | CLI |
|------|---------|-----|
| `sn_reference.py` | QC, integration, clustering, annotation | Typer: `preprocess`, `process`, `annotate`, `markers`, `inspect` |
| `sn_differential.py` | Differential expression + specificity | Typer: `region-degs`, `tau`, `composition` |

## Usage

```bash
# Full pipeline via Snakemake
uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake --profile ../../config/profiles/slurm
```

## Output

| File | Description |
|------|-------------|
| `data/raw.h5ad` | QC'd, combined nuclei (pre-normalization) |
| `data/processed.h5ad` | Normalized, Harmony-integrated, clustered nuclei |
| `data/markers/` | Top marker genes per cluster |
| `data/degs/` | Regional DEGs + tau specificity scores |

## Input

CellBender-filtered h5 files using BIMP_v4.3 gene IDs. Input paths configured in `config.yaml`.
