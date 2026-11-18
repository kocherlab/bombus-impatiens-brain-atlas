# caste_de/cnmf_gsea: joint cNMF and GSEA of the caste analysis, ported

Analysis design and source notebooks: I. Traniello.

`fig:caste_de` panel d comes from a joint cNMF (k = 50), a per-program caste
test, a label-swap null and GSEA program labels. This directory ports that code
from the analysis notebook
`01_QC_BIMP-Atlas_DifferentialExpression_clean.ipynb` (md5
184efd9af8192fd09f7329f1eeaa9f17) and, for the label-swap null and the
all-program GO labels, from the original notebook
`01_QC_BIMP-Atlas_DifferentialExpression.ipynb` (md5
5d120475e946d3953bc106195cad70ce), both in
`<snrna_source>/src/scripts/03_analysis/`.
The cells are restructured into functions and formatted to repo style (ruff
check and format, no exclusions); the computations are the notebook's. Each
file's docstring names its source cells and md5.

The notebook's DESeq2 functions (clean cell 6) are not ported: the DE fit is the
repo's (`src/bimp_st/caste_de.py`, `~ trial + caste`, whole trial pairs).
`--source repo` reads the fine-type table `bimp_st.paths.CASTE_DE_FINE` from
`../caste_de_blocked_all.py`, the same table `fig_caste_de.py` reads. The
`out/fig4_supp/scripts/` (six-program supplementary tables) produce nothing
panel d shows and are not ported; neither is the per-cell-type cNMF
(`cnmf_program_caste.csv`, `cnmf_program_per_trial.csv`), which no figure or
table reads. Its `cnmf_chosen_k.json` is copied into the saved-inputs directory
(`cnmf_gsea_paths.SAVED`) for the record only; no step reads it (the joint run
fixes k = 50).

Env: `.venv-cnmf` at the repo root, built from `config/envs/cnmf.txt` (the
analysis's sn-tools versions: Python 3.10, cnmf 1.7.1, GSEApy 1.2.1,
scikit-learn 1.7.2, numpy 1.26.4):
`uv venv --python 3.10 .venv-cnmf && uv pip install --python .venv-cnmf -r config/envs/cnmf.txt`.
`CNMF_ENV=<venv dir>` selects another build; the outputs in
`results/caste_programs/` came from a separate build of `config/envs/cnmf.txt`,
which freezes to exactly that file. Never the shared `.venv`.

## Run

From the repo root, one command submits every step as a chained Slurm job:

```bash
bash scripts/caste_de/cnmf_gsea/run_pipeline.sh all
```

Outputs go to `results/caste_programs/pipeline/` (logs in `logs/`). Single
stages: `run_pipeline.sh <stage> [repo|saved]`, chained by hand with
`DEP="--dependency=afterok:<id>"`. `--source repo` (default) is the repo's own
factorization and DESeq2 table; `--source saved` runs the same code on the saved
cNMF and DE, to check it against the saved tables.

| stage | file | source |
|---|---|---|
| freeze | `freeze_inputs.py` | new: copies the eggNOG GO annotation, GO OBO, gene lookup and saved outputs into `inputs/` (PROVENANCE.md, md5s); the only step that reads the snRNA-seq analysis tree |
| inputs | `check_inputs.py` | new: repo atlas `bimp_st.paths.SN_CB` == the atlas the analysis read (content fingerprint) |
| (shared) | `cnmf_gsea_common.py`, `cnmf_gsea_paths.py` | cells 2, 4, 6 (DE read only), 9, 11, 13, 14 |
| prepare | `cnmf_prepare.py` | cell 21: photoreceptors dropped, genes in >= 10 nuclei, 3,000 overdispersed genes, k 30-70, 10 iterations, seed 14 |
| factorize | `pipeline/cnmf_joint/run_joint.sh` | the script cell 21 writes: 16 workers, combine, k-selection plot |
| caste | `cnmf_caste.py` | cells 23, 25: consensus k = 40 / 50 at density 2.0; per trial, (queen - worker mean usage) / SD of usage in the type |
| map | `program_map.py` | new: saved program numbers -> the repo run's, by gene-spectra correlation |
| null | `label_swap_null.py` | original cells 111, 113: replicated calls (same sign in all trials, mean >= 0.5 SD, weakest trial >= 0.25 SD, mean usage >= 2%) and the trial sign flips; with three trial pairs there are 8 relabelings, so the smallest attainable permutation P is 0.25 (`label_swap_null.csv`) |
| go | `program_go.py` | cell 27 and original cell 102: GSEApy prerank on program gene scores, eggNOG GO, size 10-500, 1,000 permutations, seed 0; panel d is labelled from cell 27's table. `--cell109`: also original cell 109's table (500 permutations for programs 18 and 12; same top term for program 12); comparison only |
| gsea | `gsea_celltypes.py` | cell 15: prerank on the caste Wald statistic (baseMean > 10), 11 glial / neuronal types |
| fig3inputs | `fig3_inputs.py` | new: panel d tables; labels by cell 30's `row_label` (top BP term at FDR < 0.05, else the notebook's dagger label) on cell 27's table; `--cell109` writes the comparison set to `fig3_inputs_cell109/` |
| parity | `parity.py` | new: `pipeline/parity/parity_summary.txt` |
| figure | `../fig_caste_de.py` | `fig:caste_de` (`.venv`); `--saved-inputs` draws panel d from the saved outputs instead, for comparison |

Parity reference: the outputs in `results/caste_programs/` (`cnmf_joint/` and
one directory per source) come from a verbatim copy of the notebook cells
(commit cd8d4278); `parity.py` checks the port against them and against the
saved tables.

## Supplementary Data tables

Three Supplementary Data tables of the analysis are rebuilt here (with the
saved tables' columns) and written out by `../../deposit/supplementary_data_csvs.py`,
which calls each `build()`; outputs of the CLIs go to
`results/caste_programs/si_tables/`.

| file | table | source |
|---|---|---|
| `sd_pseudobulk_degs.py` | `snrna_caste_degs.csv` | cell 9 annotation of the fine-type DE (padj < 0.05) |
| `sd_celltype_go.py` | `snrna_caste_gsea_celltypes.csv` | cell 15 GSEA (`gsea_celltypes.py`), unfiltered |
| `program_go_all.py`, `sd_program_go.py` | `cnmf_program_go_terms.csv` | `prog_gsea.py` / `supp_full.py`: all-program GSEA (Slurm array, the cNMF env) and the table |
| `sd_common.py` | shared helpers | `supp_full.py` helpers |
| `compare_si_tables.py` | diff against the saved tables | `si_tables/compare_si_tables.txt` |

`--source repo` (what ships) reads the repo's fit, the one behind
`fig:caste_de`; `--source saved` the saved pyDESeq2 0.5.2 fit, for comparison
only. The library QC table is `../../deposit/sd_cellranger_qc.py`.
