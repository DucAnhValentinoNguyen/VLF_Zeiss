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

# The staged HyperKvasir tree may not have hyper_kvasir_unlabeled_images (it's
# only an SSL-corpus fallback, and has been removed at times) -- fabricate a
# tiny one so this smoke stays self-contained regardless.
UNLAB="$(python -c "
from vlfz.cfg import load_cfg
from vlfz.data.hyperkvasir import unlabeled_dir
print(unlabeled_dir(load_cfg('$CFG')))
")"
if [ -z "$(find "$UNLAB" -maxdepth 1 -iname '*.jpg' -o -iname '*.png' 2>/dev/null | head -1)" ]; then
  FAKE_HKV="$(mktemp -d)"
  export HKV_ROOT="$FAKE_HKV"
  python -c "
import os
from PIL import Image
d = os.path.join('$FAKE_HKV', 'hyper_kvasir_unlabeled_images')
os.makedirs(d, exist_ok=True)
for i in range(8):
    Image.new('RGB', (256, 256), ((i * 20) % 255, 100, 150)).save(os.path.join(d, f'img_{i}.jpg'))
"
  echo "[smoke_ssl] no real unlabeled pool -> fabricated 8 images under $FAKE_HKV"
fi

OBJS=(lejepa)
[ "${ALL:-0}" = "1" ] && OBJS=(lejepa dino)

for OBJ in "${OBJS[@]}"; do
  for INIT in siglip2 imagenet; do
    echo "=== SSL smoke: $OBJ / $INIT ==="
    python -m vlfz.ssl.pretrain --config $CFG --objective "$OBJ" --init "$INIT" --stage smoke \
      --corpus hkv_unlabeled --fresh --limit-steps 3 --bs 2 --nw 0 --max-images 8
    D="$OUT_ROOT/ssl/${OBJ}_${INIT}_hkv_unlabeled_smoke"
    test -f "$D/last.ckpt" && echo "  ok: $D/last.ckpt"
    test -f "$D/ema_backbone.pt" && echo "  ok: $D/ema_backbone.pt"
    # resume must not crash
    python -m vlfz.ssl.pretrain --config $CFG --objective "$OBJ" --init "$INIT" --stage smoke \
      --corpus hkv_unlabeled --resume --limit-steps 5 --bs 2 --nw 0 --max-images 8
  done
done
echo "[smoke_ssl] OK  ($OUT_ROOT)"
