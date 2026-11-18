#!/bin/bash
# bimp-sbatch: none  (submit: bash config/sbatch.sh scripts/gene_annotation/run_genome_busco.sh)
#SBATCH --job-name=bimp_v4_busco
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=results/handoff/genome_stats/busco_%j.log
# BUSCO completeness of the BIMP_v4 assembly against hymenoptera_odb12
# (the set the SI snail plot names), then the assembly statistics with the
# BUSCO short summary folded into results/handoff/genome_stats/summary.json.
#
# BUSCO is not in the project venv; this uses a BUSCO 6.1.0 conda env (site
# key busco_env, or BUSCO_ENV). The lineage is downloaded once into
# BUSCO_DL (default ~/busco_downloads: shared storage, outside the repo, so
# SLURM nodes reuse it rather than re-downloading).
#
#   bash config/sbatch.sh scripts/gene_annotation/run_genome_busco.sh
set -euo pipefail
BUSCO_ENV=${BUSCO_ENV:-$(PYTHONPATH=src .venv/bin/python -m bimp_st.site busco_env)}
BUSCO_DL=${BUSCO_DL:-$HOME/busco_downloads}
FASTA=${FASTA:-$(PYTHONPATH=src .venv/bin/python -m bimp_st.site genome_dir)/genome/BIMP_genome_v4.fasta}
OUT=results/handoff/genome_stats
mkdir -p "$OUT" "$BUSCO_DL"
export PATH="$BUSCO_ENV/bin:$PATH"
busco --version
busco -i "$FASTA" -m genome -l hymenoptera_odb12 -c "${SLURM_CPUS_PER_TASK:-16}" \
      --download_path "$BUSCO_DL" --out_path "$OUT" -o busco_hymenoptera_odb12 -f
SUM=$(ls "$OUT"/busco_hymenoptera_odb12/short_summary.specific.hymenoptera_odb12.*.json | head -1)
PYTHONPATH=src:scripts .venv/bin/python scripts/gene_annotation/genome_assembly_stats.py --fasta "$FASTA" --busco "$SUM"
