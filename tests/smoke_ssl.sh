#!/bin/bash
# SSL pretraining smoke. LeJEPA runs anywhere; DINO+ViT-B needs a GPU node
# (deepcopy'd EMA teacher = 2x ViT-B in RAM, OOMs a login node).
#   bash tests/smoke_ssl.sh            # LeJEPA only (login-node safe)
#   ALL=1 bash tests/smoke_ssl.sh      # + DINO  (run on a GPU node / srun)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source lrz/job_env.sh
export OMP_NUM_THREADS=4
export DATA_ROOT="${DATA_ROOT:-$MCMLSCRATCH/zeiss_data}"
export OUT_ROOT="${OUT_ROOT:-$(mktemp -d)/runs}"
CFG=tests/smoke_config.yaml

OBJS=(lejepa)
[ "${ALL:-0}" = "1" ] && OBJS=(lejepa dino)

for OBJ in "${OBJS[@]}"; do
  for INIT in siglip2 imagenet; do
    echo "=== SSL smoke: $OBJ / $INIT ==="
    python -m vlfz.ssl.pretrain --config $CFG --objective "$OBJ" --init "$INIT" --stage smoke \
      --data smoke --fresh --limit-steps 3 --bs 2 --nw 0 --max-images 8
    D="$OUT_ROOT/ssl/${OBJ}_${INIT}_smoke"
    test -f "$D/train_state.pt" && echo "  ok: $D/train_state.pt"
    # resume must not crash
    python -m vlfz.ssl.pretrain --config $CFG --objective "$OBJ" --init "$INIT" --stage smoke \
      --data smoke --resume --limit-steps 5 --bs 2 --nw 0 --max-images 8
  done
done
echo "[smoke_ssl] OK  ($OUT_ROOT)"
