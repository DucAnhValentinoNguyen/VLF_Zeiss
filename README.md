# VLF_Zeiss

Zero-shot **calibration** (ECE / NLL) of self-supervised vision foundation models
on **HyperKvasir**, measured **before vs after** in-domain SSL pretraining on
**GastroNet-5M**.

Independent repo — its own venv (`.venv`), its own code. No dependency on any
sibling project.

## The experiment

4 model settings = **{DINO v1, LeJEPA}** × **{SigLIP-2 ViT-B/16 init, ImageNet
ViT-B/16 init}**.

1. Evaluate all 4 **zero-shot on HyperKvasir** (frozen init).
2. **SSL-pretrain** each on GastroNet-5M (unlabeled): DINO v1 / LeJEPA, full
   backbone, ~0.7 M-image subset, single GPU, self-resubmitting.
   (`--corpus hkv_unlabeled` uses the staged 99 k HyperKvasir-unlabeled pool as a
   fallback until GastroNet-5M is downloaded.)
3. Evaluate all 4 **zero-shot on HyperKvasir again**.

Zero-shot = **weighted k-NN** on frozen features (k=20, τ=0.07). Also:
**evidential k-NN** (Dirichlet from the neighbour vote mass — Sensoy et al. 2018,
arXiv:1806.01768) for an epistemic *vacuity* and better-calibrated probabilities;
optional **evidential linear head** (`--edl-head`).

Downstream tasks (from `image-labels.csv`: Organ / Classification / Finding):
- `hkv_findings`  — 23-way finding classification
- `hkv_category`  — 4-way (anatomical / pathological / therapeutic / quality)
- `hkv_tract`     — 2-way upper vs lower GI
- `hkv_pathology` — 2-way pathological vs other
- `hkv_seg`       — zero-shot polyp segmentation (dense patch-feature k-NN label
  transfer on the 1000 mask pairs; Dice / mIoU + pixel-wise ECE/NLL)

Metrics: **ECE** (15-bin equal-width + adaptive) and **NLL**, before and after
scalar temperature / evidence-scale (fit on a held-out `cal` split), plus
accuracy / balanced-acc / AUROC / vacuity. One global stratified **image-level**
split (reference / cal / query = .60/.15/.25), reused across tasks — HyperKvasir
has no patient id, so this is image-level (recorded as a caveat).

## Layout

```
vlfz/
  cfg.py                     config (OmegaConf) + provenance
  models/  vit_backbone.py   ViT-B/16 factory (siglip2 | imagenet | ssl-ckpt), uniform .feature()
           convert_tv_vit.py torchvision vit_b_16 -> timm state_dict
           load_weights.py   weight sources + strict-load report
           heads.py ema.py   Projector/Predictor/DINOHead ; weight-EMA
  ssl/     lejepa_lib.py      SIGReg (Epps-Pulley) + variance hinge + stop-grad prediction
           dino_lib.py        DINO loss + centering + schedules
           pretrain.py        unified SSL entrypoint (--objective/--init/--stage)
  data/    realcolon.py       REAL-Colon loader + rc_frame / rc_lesion + video splits
           gastronet.py       zip-shard manifest + lazy reader + subset sampler
           transforms.py folds.py
  eval/    features.py        cached frozen-feature extraction
           knn.py             weighted k-NN -> probs / logits / vote_mass
           calibration.py     ECE / NLL / Brier / temperature scaling
           edl.py             evidential k-NN + evidential linear head
           seg.py             zero-shot polyp segmentation (dense patch k-NN)
           metrics.py run_eval.py
  data/    hyperkvasir.py     HyperKvasir tasks + splits ; registry.py dataset dispatch
           realcolon.py       (kept, unused — REAL-Colon dropped for storage)
  report/  aggregate.py pivot.py leakage_audit.py
lrz/       job_env.sh  setup_env.sh  sbatch_*.sbatch  submit_*.sh  download_*.sh
tests/     unit tests + smoke_eval.sh / smoke_ssl.sh
```

## Setup (LRZ login node)

```bash
bash lrz/setup_env.sh          # creates .venv with the pinned torch 2.5.1+cu121 stack
source lrz/job_env.sh          # exports DATA_ROOT / HKV_ROOT / OUT_ROOT(home) / venv
python -m pytest -q            # unit tests
bash tests/smoke_eval.sh       # full pipeline on a small HyperKvasir subset (CPU)
python -m vlfz.data.hyperkvasir --inspect --make-splits
```

## Data

HyperKvasir is already staged read-only under `$HKV_ROOT`
(`/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra82sat2/zeiss_data`). GastroNet-5M
(authorised via cortex.thetavision.nl) — put the shard URLs in
`lrz/gastronet_urls.txt`, then:

```bash
bash lrz/download_gastronet.sh          # -> $GASTRONET_ROOT (=$HOME/vlf_zeiss_data/gastronet5m)
```

## Run

```bash
# 1. pre-SSL baselines on HyperKvasir
bash lrz/submit_eval_chain.sh ONLY_PRE=1

# 2. SSL pretraining (4 runs, self-resubmitting).  CORPUS=hkv_unlabeled until GastroNet lands.
CORPUS=gastronet bash lrz/submit_ssl_matrix.sh    # or TIER=lejepa for the cheap 2-run cut

# 3. post-SSL eval + aggregate (chained; picks up whatever checkpoints exist)
CORPUS=gastronet bash lrz/submit_eval_chain.sh

cat "$OUT_ROOT/results/report.md"
#   long_results.csv  delta_results.csv  report.md  leakage_report.json
```

## Notes / caveats

- **k-NN probabilities are not posteriors** — raw ECE/NLL mostly reflect vote
  sharpness. Always compare the temperature-scaled rows and the pre→post **Δ**,
  not absolute values; a k-sensitivity sweep (k ∈ {10,20,50,200}) is recorded.
- **Domain gap**: GastroNet-5M is upper-GI-weighted; HyperKvasir spans upper+lower
  GI. Expect the SSL lift to vary by task (largest on fine-grained `hkv_findings`).
- **Image-level split** — HyperKvasir's `image-labels.csv` has no patient id, so
  near-duplicate procedure frames may span splits (`SUSPECT ⚠` flag + a note in
  every results JSON). Storage lives on the home quota (DSS scratch is full).
- DINO's EMA teacher is a deepcopy of a ViT-B → 2× in RAM; DINO SSL needs a GPU
  node, not a login node. LeJEPA is lighter.
- Deep Evidential *Regression* (Amini et al. 2019, arXiv:1910.02600) is not used
  (tasks are classification); it would slot into `vlfz/eval/edl.py` if a polyp-size
  regression target is added later.
