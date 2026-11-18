#!/bin/bash
# bimp-sbatch: cpu_day  (submit: bash config/sbatch.sh scripts/gene_annotation/run_annotation_busco.sh)
#SBATCH --job-name=bimp_v43_annot_busco
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=results/genome/annotation_busco/busco_%j.log
# BUSCO completeness of the BIMP v4.3 annotation (Methods, "Genome annotation":
# BUSCO v6.0.0, hymenoptera_odb12): proteins mode on the longest
# protein isoform of each protein-coding gene of the released
# BIMP_OGS_v4.3_pep.fa (scripts/gene_annotation/longest_isoform_proteins.py).
#
# BUSCO comes from its own env (config/envs/busco.yml), never the project .venv.
# The lineage is fetched once into BUSCO_DL; --offline afterwards.
#
#   bash config/sbatch.sh scripts/gene_annotation/run_annotation_busco.sh
#
# Output: results/genome/annotation_busco/{BIMP_OGS_v4.3_pep.longest.fa,
#   longest_isoform.json, busco_hymenoptera_odb12/short_summary.*.json,
#   summary.json}.
set -euo pipefail
BUSCO6_ENV=${BUSCO6_ENV:-$(PYTHONPATH=src .venv/bin/python -m bimp_st.site busco6_env)}
BUSCO_DL=${BUSCO_DL:-$(PYTHONPATH=src .venv/bin/python -m bimp_st.site busco_downloads)}
PY=${PY:-.venv/bin/python}
OUT=results/genome/annotation_busco
LINEAGE=hymenoptera_odb12
mkdir -p "$OUT" "$BUSCO_DL"

PYTHONPATH=src:scripts "$PY" scripts/gene_annotation/longest_isoform_proteins.py --out "$OUT"

export PATH="$BUSCO6_ENV/bin:$PATH"
busco --version
if [ ! -d "$BUSCO_DL/lineages/$LINEAGE" ]; then
    busco --download_path "$BUSCO_DL" --download "$LINEAGE"
fi
busco -i "$OUT/BIMP_OGS_v4.3_pep.longest.fa" -m proteins \
      -l "$BUSCO_DL/lineages/$LINEAGE" --offline \
      -c "${SLURM_CPUS_PER_TASK:-16}" \
      --out_path "$OUT" -o "busco_$LINEAGE" -f

SUM=$(ls "$OUT/busco_$LINEAGE"/short_summary.specific.$LINEAGE.*.json | head -1)
PYTHONPATH=src:scripts "$PY" - "$SUM" "$OUT" "$BUSCO_DL/lineages/$LINEAGE/dataset.cfg" <<'EOF'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/gene_annotation")
from genome_assembly_stats import busco_summary
s, out, cfg = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
S = {"busco_source": s, "busco_mode": "proteins"}
S.update(json.loads((out / "longest_isoform.json").read_text()))
S.update(busco_summary(s))
S["lineage_dataset_cfg"] = cfg.read_text() if cfg.exists() else None
(out / "summary.json").write_text(json.dumps(S, indent=1) + "\n")
print(json.dumps(S, indent=1))
EOF
