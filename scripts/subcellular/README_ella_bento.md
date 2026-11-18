# Subcellular localization across regions and compartments (bento + ELLA)

Subcellular localization across regions and compartments: the analyses behind
the supplementary figure `subcellular_regions`. One brain per caste: everything is
descriptive, no caste test. Run from the repo root with
`PYTHONPATH=src:scripts .venv/bin/python scripts/subcellular/<script>`.

Inputs: per-region bento `lp.parquet` under `results/bento/{queen,worker}/`, the nine-region
atlas objects under `results/nine_region/atlas/`, raw per-crop h5ads under
`pipelines/*/data/`, and the raw Xenium `transcripts.parquet` / `cell_boundaries.parquet` /
`nucleus_boundaries.parquet` under `data/`.

| step | script | writes |
|---|---|---|
| section and compartment tables | `bento_regroup.py` | `results/bento/by_section/`, `kc_compartments/` |
| per-region tables with per-cell and per-gene detail | `bento_regions.py` | `results/bento/by_section/region_*` |
| ELLA per section (`ELLA_MODE=kc`, `regions`, `subs`; `ella_section.py manifest`, then `run <i>` per section) | `ella_section.py` | `results/ella*/<brain>/<section>/` |
| controls | `bento_depth_control.py`, `bento_area_control.py`, `bento_size_control.py` | `results/bento/regions_summary/` |
| exploratory, no figure reads it (subclusters) | `bento_kc_subclusters.py` | `kc_subclusters/` |
| summaries | `si_ella_summary.py kc\|regions`, `si_bento_regions.py` | `results/ella*/summary/`, `results/bento/*_summary/` |
| figure | `si_subcellular_summary.py` | `results/figures/si/figS_subcellular_regions.png` |

ELLA (https://github.com/jadexq/ELLA) is CPU-only and needs its own Python 3.11
environment; point `ELLA_SRC` at a clone of the repository for the demo-pickle parity check
in `ella_section.py`.

Coverage: the section tables describe 55 sections (28 queen, 27 worker). Ten crops (queen
0040243 region 4; worker 0068401 region 3; worker 0069041 regions 2 and 3) have no
`lp.parquet` and are absent from every table until bento is rerun on them.
