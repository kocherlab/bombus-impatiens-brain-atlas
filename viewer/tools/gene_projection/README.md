# Atlas gene projection (the web atlas's gene set)

This rebuilds the gene set the web viewer shows: the 100 measured Xenium panel genes
and the Tangram-inferred ones. The inferred genes are re-projected through the **canonical**
Tangram mapping (the one that assigned the cell types), so genes and labels come from
the same mapping. It writes a new imputed object and never touches the existing ones,
which the analyses read.

## Inputs (all from `bimp_st.paths`; no hard-coded `results/tangram` paths)

| constant | file | role |
|---|---|---|
| `paths.CANONICAL_TANGRAM` | `combined_clustered_g2_R2_s42_tangram.h5ad` | `obsm['tangram_ct_pred']`, the row-normalised cell-type mapping (29 types) |
| sibling `…_g2_R2_s42_imputed.h5ad` | the canonical run's own projection | recovers each cell's mapping scale; must be reproduced exactly |
| `paths.SN_COMBINED` | `data/sn/BIMP_atlas_qw_combined.h5ad` | snRNA reference (`layers['counts']`, `obs['cell_type']`) |
| `REPO/data/panel/BIMPv4.3_neuropeptides_MALDI.tsv` | tracked | neuropeptide precursors, always included |
| `REPO/data/panel/np31_to_v43_mapping.csv` | tracked | panel genes, always included |

## Gene rule (`select_genes.py`)

The rule takes every reference gene with `max_type_frac >= 0.01`, meaning a nonzero raw
count in at least 1% of the nuclei of at least one of the 29 `cell_type` groups. It
then adds every neuropeptide precursor and every panel gene present in the reference.

The list is checked in as `genes_atlas_v2.txt`. The script asserts that allatostatin A (`BIMP_v4.3_009248`),
allatotropin (`BIMP_v4.3_000471`) and tachykinin (`BIMP_v4.3_009257`) are in the list.

The floor is applied per cell type on purpose. The earlier rule (HVGs and markers
detected in at least 5% of *all* nuclei) dropped allatostatin, which is rare over all
nuclei but common in its top type. The reference is already filtered, so this rule keeps
nearly every gene. That is also why receptors and amine enzymes are covered: no
genome-wide annotation existed to pick them out by name.

## How the projection works (`project_genes.py`)

The canonical run used Tangram in `mode="clusters"`. For that mode,
`tangram2.project_genes` computes:

    imputed[i, g] = Σ_t M[t, i] · C[t, g]

Here `C[t, g]` is the **sum**, over the nuclei of type `t`, of
`normalize_total(counts)` (`adata_to_cluster_expression(scale=True)` sums rather than
averages). The library-size normalisation runs over all reference genes before any
subsetting.

The mapping matrix `M` was never saved. The tangram object keeps only its row-normalised
transpose, and the per-cell scale lost there is recovered by least squares from the
canonical imputed object. The script aborts unless the rebuild reproduces all 2,104
genes of that object to a relative error of at most 1e-4 (measured: 1.6e-15). Run it
with `--check-only` to do just that check.

The output is `paths.CANONICAL_TANGRAM` with `_tangram` replaced by `_imputed_atlas_v2`,
written uncompressed so the export's row-block reads stay fast:

- shape 1,300,192 × 9,710, float32, about 51 GB;
- `obs['tangram_map_scale']`, the recovered per-cell scale;
- `uns['provenance']`.

## Order of commands

Run everything from the repo root, under Slurm (`bash config/sbatch.sh`, which adds the
partition and QOS from `config/profiles/slurm/sbatch.env`). None of these steps needs a GPU.

```bash
mkdir -p logs
bash config/sbatch.sh viewer/tools/gene_projection/01_ref_gene_stats.sbatch   # -> results/gene_projection/ref_gene_stats.csv
bash config/sbatch.sh viewer/tools/gene_projection/02_select_genes.sbatch     # -> results/gene_projection/genes_atlas_v2.txt
                                                                   #    (should equal the checked-in genes_atlas_v2.txt)
bash config/sbatch.sh viewer/tools/gene_projection/03_project_genes.sbatch    # -> results/tangram/combined_clustered_g2_R2_s42_imputed_atlas_v2.h5ad
bash config/sbatch.sh viewer/tools/gene_projection/04_export_pages.sbatch     # -> viewer/pages/data/{base*.bin,genes/,manifest.json,inferred_meta.json}
```

Steps 1 and 2 only re-derive the list; step 3 reads the checked-in `genes_atlas_v2.txt`
by default. To project a different list, pass it as the first argument to `03_…`. Any
further arguments go to `project_genes.py`, for example `--out PATH`. Step 3 refuses to
overwrite an existing output.

| step | partition / QOS | CPUs | memory | wall | measured (2026-10-16) |
|---|---|---|---|---|---|
| 01 ref stats | cpu, short | 4 | 32G | 0:30 | ~1 min |
| 02 select | cpu, short | 1 | 4G | 0:10 | seconds |
| 03 project | cpu, short | 4 | 96G | 0:45 | 6.7 min, 52 GB RSS |
| 04 export | cpu, day | 4 | 128G | 2:00 | 39 min, 53 GB RSS |

## Landing the output in `pages/data`

Step 4 runs `export_data.py --pages --genes 100 --imputed-genes --imputed-h5ad <v2>
--inferred-mapping canonical`. Before writing, it deletes the old `genes/i*.b4` and
`inferred_meta.json`, because the inferred list changes wholesale. It never touches
`pages/data/histo*`.

The export writes:

- `genes/g000–g099.bin`: the measured genes;
- `genes/i0100.b4` onward: one gzipped, 4-bit-packed file per inferred gene, about
  640 KB each, 6.1 GB in total;
- `manifest.json`, which records:
  - `inferredMapping: "canonical"`, telling the viewer the genes share the cell types'
    mapping;
  - `genesBase` (default `"./data/"`, set with `--genes-base-url`), the URL prefix the
    viewer fetches gene files from.

The gene files are well over GitHub Pages' 1 GB limit, so a deployment must serve them
from another host and set `--genes-base-url`.

When the export ran on a cluster, copy `genes/` back to your checkout. Copy only `genes/`,
never a full `pages/data` sync: that would overwrite a `manifest.json` whose gene names
have already been patched.

```bash
rsync -rlt --partial <cluster>:<checkout>/viewer/pages/data/genes/ viewer/pages/data/genes/
```

Bring over `manifest.json` and `inferred_meta.json` once, deliberately, and only after
re-running the export.

## Follow-up: gene names

The export leaves inferred genes unnamed. Patch names, fly orthologs and search aliases
into the exported manifest and its sidecar:

```bash
python3 viewer/gene_names.py viewer/pages/data
```

The argument is the `pages/data` directory, relative to where you run it. From
`viewer` it is `python3 gene_names.py pages/data`. Then rebuild the site with
`node build-pages.mjs` in `viewer`.

## Caveats

- In clusters mode every cell's inferred value is a mixture of 29 cell-type profiles, so
  it is never exactly zero. The viewer's "cells expressing" share reads 100% for inferred
  genes, and the low-signal flag (coefficient of variation < 1) marks none of them.
- The inferred-gene count in `viewer/pages/data/manifest.json` changes with this export;
  any check pinned to the earlier count must be updated with it.
