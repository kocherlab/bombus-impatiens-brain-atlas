#!/usr/bin/env bash
# Submit a stand-alone Slurm launcher with this site's partition / QOS options.
#
#   bash config/sbatch.sh <launcher.sh> [launcher args...]
#
# The launcher names its class in a line "# bimp-sbatch: <class>":
#   cpu       $BIMP_SBATCH_CPU
#   cpu_short $BIMP_SBATCH_CPU $BIMP_SBATCH_SHORT
#   cpu_day   $BIMP_SBATCH_CPU $BIMP_SBATCH_DAY
#   gpu       $BIMP_SBATCH_GPU        (the launcher requests the GPU itself)
#   gpu_h100  $BIMP_SBATCH_GPU_H100
#   none      no site options
# The values come from config/profiles/slurm/sbatch.env, or the file named by
# $BIMP_SBATCH_ENV. Extra sbatch options can be passed in $BIMP_SBATCH_EXTRA.
# The job is submitted from the repository root, so relative #SBATCH --output
# paths resolve there; their directories are created first.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1090
source "${BIMP_SBATCH_ENV:-$REPO/config/profiles/slurm/sbatch.env}"

[ $# -ge 1 ] || { echo "usage: bash config/sbatch.sh <launcher.sh> [args...]" >&2; exit 2; }
script=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
shift
cls=$(sed -n 's/^# bimp-sbatch: *\([a-z0-9_]*\).*/\1/p' "$script" | head -1)
case "$cls" in
  cpu) opts=$BIMP_SBATCH_CPU ;;
  cpu_short) opts="$BIMP_SBATCH_CPU $BIMP_SBATCH_SHORT" ;;
  cpu_day) opts="$BIMP_SBATCH_CPU $BIMP_SBATCH_DAY" ;;
  gpu) opts=$BIMP_SBATCH_GPU ;;
  gpu_h100) opts=$BIMP_SBATCH_GPU_H100 ;;
  none) opts="" ;;
  *) echo "$script: no '# bimp-sbatch: cpu|cpu_short|cpu_day|gpu|gpu_h100|none' line" >&2; exit 2 ;;
esac

cd "$REPO"
while read -r out; do
  [ -n "$out" ] && mkdir -p "$(dirname "$out")"
done < <(sed -n 's/^#SBATCH --\(output\|error\)=\(.*\)/\2/p' "$script")

# shellcheck disable=SC2086
exec sbatch $opts ${BIMP_SBATCH_EXTRA:-} "$script" "$@"
