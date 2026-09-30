#!/bin/bash
#SBATCH --job-name=vlfz_fullstream
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=48:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue
#SBATCH --output=lrz/logs/fullstream_%j.out
#SBATCH --error=lrz/logs/fullstream_%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source lrz/fullstream_env.sh
child=''
trap 'if [ -n "$child" ]; then kill -USR1 "$child"; fi' USR1
python -m vlfz.ssl.fullstream &
child=$!
set +e
wait "$child"
rc=$?
if kill -0 "$child" 2>/dev/null; then
  wait "$child"
  rc=$?
fi
set -e
if [ "$rc" = 75 ] && [ "${STREAM_RESUMES:-0}" -lt 100 ]; then
  export STREAM_RESUMES=$(( ${STREAM_RESUMES:-0} + 1 ))
  sbatch "$SLURM_SUBMIT_DIR/lrz/sbatch_fullstream.sbatch"
elif [ "$rc" != 0 ]; then
  echo "Stopped with rc=$rc; inspect logs before manual retry."
  exit "$rc"
fi
