# VLF_Zeiss

Zero-shot **calibration** (ECE / NLL) of self-supervised vision foundation models
on **REAL-Colon**, measured **before vs after** in-domain SSL pretraining on
**GastroNet-5M**.

Independent repo — its own venv (`.venv`), its own code. No dependency on any
sibling project.

## The experiment

4 model settings = **{DINO v1, LeJEPA}** × **{SigLIP-2 ViT-B/16 init, ImageNet
ViT-B/16 init}**.

1. Evaluate all 4 **zero-shot on REAL-Colon** (frozen init).
2. **SSL-pretrain** each on GastroNet-5M (unlabeled): DINO v1 / LeJEPA, full
   backbone, ~1–1.5 M-image subset, single GPU, self-resubmitting.
3. Evaluate all 4 **zero-shot on REAL-Colon again**.

Zero-shot = **weighted k-NN** on frozen features (k=20, τ=0.07). Also:
**evidential k-NN** (Dirichlet from the neighbour vote mass — Sensoy et al. 2018,
arXiv:1806.01768) for an epistemic *vacuity* and better-calibrated probabilities;
optional **evidential linear head** (`--edl-head`).

Two binary tasks derived from REAL-Colon:
- `rc_frame` — frame-level polyp vs no-polyp
- `rc_lesion` — lesion-crop adenoma vs non-adenoma

Metrics: **ECE** (15-bin equal-width + adaptive) and **NLL**, before and after
scalar temperature / evidence-scale (fit on a held-out `cal` video split), plus
accuracy / balanced-acc / AUROC / vacuity. Splits are **by video** (query /
reference / cal) — no frame leakage.

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
           metrics.py run_eval.py
  report/  aggregate.py pivot.py leakage_audit.py
lrz/       job_env.sh  setup_env.sh  sbatch_*.sbatch  submit_*.sh  download_*.sh
tests/     unit tests + smoke_eval.sh / smoke_ssl.sh + fake-REAL-Colon generator
```

## Setup (LRZ login node)

```bash
bash lrz/setup_env.sh          # creates .venv with the pinned torch 2.5.1+cu121 stack
source lrz/job_env.sh          # exports MCMLSCRATCH / DATA_ROOT / OUT_ROOT / venv
python -m pytest -q            # 17 unit tests
bash tests/smoke_eval.sh       # full pipeline on synthetic REAL-Colon (~2 min, CPU)
```

## Data (user-run, login node — both are large)

```bash
bash lrz/download_realcolon.sh          # ~1 TB, Figshare, CC BY 4.0
python -m vlfz.data.realcolon --make-splits --inspect
bash lrz/download_gastronet.sh          # gated: cortex.thetavision.nl  (~0.5-1 TB)
```

## Run

```bash
# 1. pre-SSL baselines
bash lrz/submit_eval_chain.sh ONLY_PRE=1

# 2. SSL pretraining (4 runs, self-resubmitting)
bash lrz/submit_ssl_matrix.sh                 # or TIER=lejepa for the cheap 2-run cut

# 3. post-SSL eval + aggregate (chained; picks up whatever checkpoints exist)
bash lrz/submit_eval_chain.sh

# results
cat "$OUT_ROOT/results/report.md"
#   long_results.csv  delta_results.csv  report.md  leakage_report.json
```

## Notes / caveats

- **k-NN probabilities are not posteriors** — raw ECE/NLL mostly reflect vote
  sharpness. Always compare the temperature-scaled rows and the pre→post **Δ**,
  not absolute values; a k-sensitivity sweep (k ∈ {10,20,50,200}) is recorded.
- **Domain gap**: GastroNet-5M is upper-GI-weighted; REAL-Colon is colonoscopy.
  Expect `rc_frame` to benefit from SSL more than `rc_lesion`.
- DINO's EMA teacher is a deepcopy of a ViT-B → 2× in RAM; DINO SSL needs a GPU
  node, not a login node. LeJEPA is lighter.
- Deep Evidential *Regression* (Amini et al. 2019, arXiv:1910.02600) is not used
  (tasks are classification); it would slot into `vlfz/eval/edl.py` if a polyp-size
  regression target is added later.
