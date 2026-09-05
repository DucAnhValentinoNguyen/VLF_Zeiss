#!/bin/bash
# End-to-end eval smoke on the STAGED HyperKvasir data.
# Core (pre-SSL zero-shot + aggregate) runs on a login node.
# The SSL + post-SSL + segmentation steps need a GPU node (ViT-B OOMs a login
# node); they are best-effort here (guarded with '|| echo skip').
#   bash tests/smoke_eval.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh
export OMP_NUM_THREADS=4

SMK="${SMK:-$(mktemp -d)}"
export OUT_ROOT="$SMK/runs"
CFG=tests/smoke_config.yaml
set -e

echo "[smoke] workdir $SMK  (HKV_ROOT=$HKV_ROOT)"
python -m vlfz.data.hyperkvasir --config $CFG --inspect

echo "[smoke] pre-SSL zero-shot (hkv_tract, hkv_category)"
python -m vlfz.eval.run_eval --config $CFG --dataset hyperkvasir --init imagenet --stage pre \
  --task all --protocols knn

set +e
echo "[smoke] (GPU-node) throwaway SSL checkpoint + post eval + segmentation"
python -m vlfz.ssl.pretrain --config $CFG --objective lejepa --init imagenet --stage full \
  --corpus hkv_unlabeled --fresh --limit-steps 2 --bs 2 --nw 0 --max-images 6 \
  && python -m vlfz.eval.run_eval --config $CFG --dataset hyperkvasir --init imagenet \
       --objective lejepa --corpus hkv_unlabeled --stage post --task all \
       --protocols knn \
  || echo "[smoke] SSL/post skipped (needs a GPU node)"
python -m vlfz.eval.seg --config $CFG --init imagenet --stage pre || echo "[smoke] seg skipped"
set -e

python -m vlfz.report.aggregate --config $CFG
echo "[smoke] report:"; cat "$OUT_ROOT/results/report.md"
echo "[smoke] OK  (artefacts under $SMK)"
