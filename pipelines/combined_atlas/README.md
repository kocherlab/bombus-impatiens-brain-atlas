# Combined atlas pipeline

Combines the aligned queen and worker atlases into one object, clusters it, maps
snRNA-seq cell types onto it with Tangram2, and runs the queen-worker
differential analysis.

## Pipeline

```
[queen aligned_clustered] + [worker aligned_clustered]
    → combine → cluster ─┬→ differential
                          └→ tangram, tangram_loocv   (only with --config out_tag=<new tag>)
```

The default target writes nothing under `results/`. Tangram rules run the
canonical recipe (`bimp_st.paths.CANONICAL_TANGRAM`, tag `g2_R2_s42`) under a
new tag; the untagged basename and the canonical tag itself are refused. The
shared profiles rerun on mtime only, so editing a shell line does not schedule
a rebuild of the canonical `data/combined_clustered.h5ad`.

## Scripts

| File | Purpose | CLI |
|------|---------|-----|
| `combined_analysis.py` | Combine brains, cluster, differential | Typer: `combine`, `cluster`, `differential` |
| `tangram_mapping.py` | Tangram cell type mapping + validation | Typer: `train`, `qc`, `cross-validate` |



## Usage

```bash
# Combine, cluster, differential
uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake --profile ../../config/profiles/slurm

# A new Tangram run (canonical recipe, another seed)
uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake --profile ../../config/profiles/slurm --config out_tag=g2_R2_s43 tangram_seed=43

# Requires both per-brain pipelines to have completed first
```

## Output

| File | Description |
|------|-------------|
| `data/combined_clustered.h5ad` | Combined cells from both brains, clustered |
| `results/tangram/combined_clustered_<tag>_tangram.h5ad` | + Tangram cell type predictions (the canonical run, tag `g2_R2_s42`, came from `tangram_mapping.py train` via `scripts/identity/run_tangram_remap_grid*.sh`; load it with `bimp_st.paths.load_tangram`) |
| `results/tangram/loocv_<tag>/` | gene-imputation leave-one-out CV (`tangram_loocv`) |

`tangram_mapping.py qc` is not a rule: it writes fixed, untagged PNGs into
`results/tangram_qc/` whatever the tag.
