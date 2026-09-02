#!/bin/bash
# End-to-end eval smoke on a synthetic REAL-Colon tree (CPU, ~2 min).
#   bash tests/smoke_eval.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh
export OMP_NUM_THREADS=4

SMK="${SMK:-$(mktemp -d)}"
export REALCOLON_ROOT="$SMK/fake_real_colon"
export OUT_ROOT="$SMK/runs"
export DATA_ROOT="${DATA_ROOT:-$MCMLSCRATCH/zeiss_data}"
CFG=tests/smoke_config.yaml

echo "[smoke] workdir $SMK"
python tests/make_fake_realcolon.py "$REALCOLON_ROOT"
python -m vlfz.data.realcolon --config $CFG --inspect

python -m vlfz.eval.run_eval --config $CFG --init imagenet --stage pre  --task all --protocols knn,eknn --edl-head

# a throwaway 2-step SSL checkpoint so the 'post' path is exercised
python -m vlfz.ssl.pretrain --config $CFG --objective lejepa --init imagenet --stage full \
  --data smoke --fresh --limit-steps 2 --bs 2 --nw 0 --max-images 6
python -m vlfz.eval.run_eval --config $CFG --init imagenet --objective lejepa --stage post \
  --task all --protocols knn,eknn --edl-head

python -m vlfz.report.aggregate --config $CFG
echo "[smoke] report:"; cat "$OUT_ROOT/results/report.md"
echo "[smoke] OK  (artefacts under $SMK)"
