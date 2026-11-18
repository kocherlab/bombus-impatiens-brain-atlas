#!/bin/bash
REPO=$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)
# Submit LP save jobs for all completed bento regions.
# Each job loads the prepped zarr, runs LP (~5.5h), saves parquet.

cd $REPO
mkdir -p logs
# shellcheck disable=SC1090
source "${BIMP_SBATCH_ENV:-config/profiles/slurm/sbatch.env}"

for brain in queen worker; do
    for dir in results/bento/${brain}/output-XETG*; do
        region=$(basename "$dir")
        cache="data/bento/${brain}/${region}"

        # Skip if no prepped zarr
        [ ! -d "${cache}/prepped.zarr" ] && continue
        # Skip if already has lp.parquet
        [ -f "${dir}/lp.parquet" ] && continue

        echo "Submitting ${brain}/${region}"
        sbatch --job-name="lp_${brain}_${region:(-3)}" \
               $BIMP_SBATCH_CPU \
               --cpus-per-task=64 \
               --mem=128G \
               --time=10:00:00 \
               --output="logs/lp_save_${brain}_${region:(-3)}_%j.log" \
               --error="logs/lp_save_${brain}_${region:(-3)}_%j.log" \
               --export=ALL,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1 \
               --wrap ".venv/bin/python pipelines/subcellular/bento_save_lp.py \
                   --brain ${brain} --region ${region} --num-workers 64"
    done
done

echo "All LP save jobs submitted"
