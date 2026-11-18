# Whole-brain pipeline: shared code

One Snakemake pipeline for both brains. Each brain directory (`queen/`,
`worker/`) keeps its data and a two-line `Snakefile` that loads its
`config.yaml` and includes `wholebrain.smk`. The brains' hand-drawn region
annotations are in `data/annotations/<brain>/` at the repository root (`v1/` is
the set `label` runs on; `v2/`-`v4/` are read by `pipelines/analyses/`).

| Brain | Directory | Brain-specific files |
|---|---|---|
| queen (brain1) | `queen/` | `config.yaml`, `labeling.py` (ANNOTATION_MAPPING), `convert_single_region.py`, `data/`, plus `run_mosaicfield.py` and `compositional_analysis.py` |
| worker (brain2) | `worker/` | `config.yaml`, `labeling.py`, `convert_single_region.py`, `data/` |

`config.yaml` holds everything that differs between the brains: the PASTE3
flags as run (queen `--subsample 3000 --align-images`, worker `--subsample 8000
--allow-90-rotation`), `align` and `label` memory, the video colouring, and
the raw regions the worker converts (`convert_regions`; the queen's crops
predate the pipeline).

| File | Purpose |
|---|---|
| `wholebrain.smk` | the rules: [base_image, convert_region] → label → align → cluster → visualize_3d, render_video |
| `analysis_workflow.py` | Typer CLI: `label-all`, `align`, `cluster`, `visualize-3d`, `render-video`, `plot-umap`, ... |
| `labeling_core.py` | GeoJSON annotations to `in_<region>` cell flags (Shapely) |
| `run_mosaicfield_wrapper.py` | the experimental `--config aligner=mosaicfield` alignment |
| `brain_dir.py` | resolves the brain directory: the current directory, or `BIMP_BRAIN_DIR` |

The scripts work on the brain directory they are run from and refuse any other
directory. Run everything from a brain directory:

```bash
cd pipelines/wholebrain/queen    # or worker
uvx --from snakemake --with snakemake-executor-plugin-slurm snakemake -n --profile ../../../config/profiles/slurm   # dry run
PYTHONPATH=../../../src ../../../.venv/bin/python ../analysis_workflow.py label-all
```

Never re-run `label`, `align` or `cluster` on the canonical data: `align` is a
stochastic GPU job and every downstream spatial number depends on its
output. The eight-region relabel is `pipelines/analyses/region_relabel.smk` for that reason.
