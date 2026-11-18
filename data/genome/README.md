# Genome-annotation inputs

The inputs of the BIMP_v4.3 genome annotation (EGAPx) that Supplementary Data
and the Methods are computed from.

- `egapx_run_params.yaml`: the EGAPx run's own `nextflow/run_params.yaml`
  (EGAPx 0.4.0-alpha): the RNA-seq and Iso-Seq read sets given to the run.
  `scripts/deposit/sd_egapx_inputs.py` builds Supplementary Data
  `sd:egapx_samples` and `sd:egapx_isoseq` from it.
- `egapx_rnaseq_samples.csv`: `EGAPx_sample_tables.csv` (md5
  1d70ed0cb73e848e100afa4765728ebb), the RNA-seq sample table supplied by the
  authors; `sd_egapx_inputs.py` checks that the table it builds from
  `egapx_run_params.yaml` reproduces this file byte for byte.
