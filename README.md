# VLF_Zeiss

Zero-shot **calibration** (ECE / NLL) of self-supervised vision foundation models
on **HyperKvasir**, measured **before vs after** in-domain SSL pretraining on
**GastroNet-5M**. SSL training + eval feature extraction run on **PyTorch
Lightning**; GastroNet-5M is curated straight from the Cortex portal into a
local WebDataset tier on LRZ (see `pipeline/`).

Independent repo — its own venv (`.venv`), its own code. No dependency on any
sibling project.

## Results and released checkpoint

Two studies, same evaluation (weighted k-NN on HyperKvasir, de-duplicated split):

1. **Full-corpus study (main result, 2026-09-29).** ImageNet ViT-B/16 init, then
   SSL on **all 506 GastroNet-5M archives** (4,823,974 images/epoch, 3 epochs =
   113,064 steps), `DINO` vs `LeJEPA`. Run with `lrz/submit_fullstream.sh`; the
   loader streams the portal zips directly (`--source portal_zip`).
2. **Curated-subset study (2026-09-08..13).** 4 settings, {DINO, LeJEPA} ×
   {SigLIP-2, ImageNet}, on a 60-zip / 581k-image local WebDataset tier. Kept as an
   earlier, smaller comparison.

Full-corpus results, mean balanced accuracy over the 4 classification tasks and
polyp-segmentation Dice (frozen ImageNet baseline → after SSL):

| model | mean bal. acc. | Δ | hkv_seg Dice | Δ | gate |
|---|---|---|---|---|---|
| frozen ImageNet (baseline) | 0.8605 | – | 0.667 | – | – |
| **DINO + ImageNet** | **0.8689** | +0.008 | **0.745** | +0.079 | pass |
| LeJEPA + ImageNet | 0.8279 | −0.033 | 0.588 | −0.079 | fail (regresses) |

The **DINO/ImageNet** model is the promoted foundation model: no regression on any
task, effective rank stable (14.0 → 14.3, no collapse), 0 decode failures in
13.4M decode attempts. `hkv_tract` is structurally easy (tract-exclusive labels),
so its ~0.99 is a caveat, not a finding. Selection rule, per-task numbers and
provenance are in `foundation_model/CARD.json` (on LRZ, under
`$HOME/vlf_zeiss_fullstream/`) and in the W&B project
`vlf-zeiss` (artifact `dino-imagenet-s506-raw-e3`).

**Weights.** Private Hugging Face repo `DucAnhValentinoNguyen/vlf-zeiss-dino-imagenet`
(`ema_backbone.pt`, ViT-B/16 trunk, 343 MB). The weights are trained on
GastroNet-5M, which is under a data-use agreement: do not redistribute. Load:

```python
import torch, timm
ck = torch.load("ema_backbone.pt", map_location="cpu", weights_only=False)
m = timm.create_model("vit_base_patch16_224", num_classes=0, global_pool="")
m.load_state_dict(ck["ema_backbone"])   # feature = LayerNorm(mean of patch tokens), 768-d
```

## The experiment

Original design: 4 model settings = **{DINO v1, LeJEPA}** × **{SigLIP-2 ViT-B/16
init, ImageNet ViT-B/16 init}**. The full-corpus study above dropped SigLIP-2
(it lost to ImageNet on every task before SSL); its code stays as dormant
reference.

1. Evaluate all 4 **zero-shot on HyperKvasir** (frozen init).
2. **SSL-pretrain** each on GastroNet-5M (unlabeled): DINO v1 / LeJEPA, full
   backbone, single GPU. Data is read from the curated 224px WebDataset tier on
   local LRZ disk (`gastronet_source: local_webdataset`, the default). See
   `pipeline/README.md` for how that tier is built.
3. Evaluate all 4 **zero-shot on HyperKvasir again**.

Zero-shot = **weighted k-NN** on frozen features (k=20, τ=0.07), temperature
scaled. (Evidential deep learning — Sensoy et al. 2018, arXiv:1806.01768 — was
evaluated and dropped to keep the benchmark to one clean protocol; the
reference implementation stays at `vlfz/eval/edl.py`, unused.)

Downstream tasks (from `image-labels.csv`: Organ / Classification / Finding):
- `hkv_findings`  — 23-way finding classification
- `hkv_category`  — 4-way (anatomical / pathological / therapeutic / quality)
- `hkv_tract`     — 2-way upper vs lower GI
- `hkv_pathology` — 2-way pathological vs other
- `hkv_seg`       — zero-shot polyp segmentation (dense patch-feature k-NN label
  transfer on the 1000 mask pairs; Dice / mIoU + pixel-wise ECE/NLL)

Metrics: **ECE** (15-bin equal-width + adaptive) and **NLL**, before and after
scalar temperature scaling (fit on a held-out `cal` split), plus accuracy /
balanced-acc / AUROC. One global stratified **image-level** split (reference /
cal / query = .60/.15/.25), reused across tasks — HyperKvasir has no patient id,
so this is image-level (recorded as a caveat).

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
           pretrain.py        Lightning SSLModule + GastroNetDataModule (--objective/--init/--stage)
  data/    gastronet.py       curated WebDataset reader (local, the live path; S3-streamed mode dormant)
           hyperkvasir.py     HyperKvasir tasks + splits ; registry.py dataset dispatch
           transforms.py folds.py
  eval/    features.py        cached frozen-feature extraction via Lightning Trainer.predict()
           knn.py             weighted k-NN -> probs / logits / vote_mass
           calibration.py     ECE / NLL / Brier / temperature scaling
           edl.py             evidential k-NN + evidential linear head (dormant, unused)
           seg.py             zero-shot polyp segmentation (dense patch k-NN)
           metrics.py run_eval.py
  report/  aggregate.py pivot.py leakage_audit.py
pipeline/  GastroNet-5M ingest: portal_to_wds.py (portal -> curated WebDataset on
           LRZ, the live path) + a dormant AWS S3 lake (Terraform/catalog/curate,
           retired — see pipeline/README.md).
lrz/       job_env.sh  setup_env.sh  sbatch_*.sbatch  submit_*.sh  download_gastronet.sh
tests/     unit tests + smoke_eval.sh / smoke_ssl.sh
```

## Setup (LRZ login node)

```bash
bash lrz/setup_env.sh          # creates .venv: pinned torch 2.5.1+cu121 + lightning + webdataset/boto3
source lrz/job_env.sh          # exports DATA_ROOT / HKV_ROOT / OUT_ROOT / AWS_PROFILE / venv
python -m pytest -q            # unit tests
bash tests/smoke_eval.sh       # full pipeline on a small HyperKvasir subset (CPU)
bash tests/smoke_ssl.sh        # LeJEPA SSL smoke, self-contained (fabricates tiny images if needed)
python -m vlfz.data.hyperkvasir --inspect --make-splits
```

## Data

**HyperKvasir** (evaluation & zero-shot calibration benchmark) is staged read-only under `$HKV_ROOT`
(`/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra82sat2/zeiss_data`), serving as the primary evaluation dataset.

Layout under `$HKV_ROOT`:
- `hyper_kvasir_labeled_images/`: 10,662 labeled endoscopy images across upper and lower GI findings (`image-labels.csv`).
- `hyper_kvasir_segmented_images/`: 1,000 polyp images with ground-truth segmentation masks (`hkv_seg`).
- `hyper_kvasir_unlabeled_images/`: ~99k unlabeled images (`images/*.jpg`), usable as a local fallback SSL corpus (`--corpus hkv_unlabeled`).

**Splits & Cross-Split Deduplication:**
- Stratified image-level split (reference / cal / query = 60% / 15% / 25%) shared across all classification tasks.
- Perceptual hash deduplication (`imagehash.phash`, Hamming distance ≤ 6 bits) filters out cal and query frames that are near-duplicates of reference images, preventing cross-split video frame leakage.

**GastroNet-5M** (SSL corpus) is curated directly from the Cortex portal onto
LRZ local disk — `pipeline/ingest/portal_to_wds.py` streams each portal zip,
decodes/dedups/resizes to 224px, packs it into WebDataset `.tar` shards under
`$GASTRONET_ROOT/webdataset/`, then deletes the zip, so the ~1TB raw corpus
never has to land at once. This is the live path; training then just works:

```bash
python -m vlfz.ssl.pretrain --objective lejepa --init siglip2 --stage full --corpus gastronet
# gastronet_source defaults to local_webdataset: reads $GASTRONET_ROOT/webdataset/*.tar directly.
```

Full-corpus runs use `--source portal_zip` instead: the loader streams all 506
portal archives (4,823,974 images per epoch) directly from the portal without
staging the corpus locally, with deterministic shuffling and a blank-sample
fallback so step counts stay exact across SLURM restarts (3 epochs = 113,064
optimizer steps per setting). The local tier above holds only 60 of 506 portal
zips (581,518 images, ~8GB curated) and backs the curated-subset study
(~1.7M image-presentations across 3 epochs). To ingest more locally:

```bash
bash lrz/backfill_gastronet_full.sh   # resumes from pipeline/ingest's _ingested.txt, skips done shards
```

An earlier **AWS S3 data lake** design (Terraform + an ephemeral ingest EC2,
`pipeline/infra/`) is retired — the EC2 kept dying ~20min after boot on a
new-account restriction, so this repo never depends on it. The code
(`portal_to_s3.py`, `pipeline/infra/*.tf`, and the `s3_webdataset` /
`local_zip` source modes in `vlfz/data/gastronet.py`) is kept as a dormant
reference, not on the active path — see `pipeline/README.md`.

## Run

```bash
# 1. pre-SSL baselines on HyperKvasir
bash lrz/submit_eval_chain.sh ONLY_PRE=1

# 2. SSL pretraining (4 runs, self-resubmitting), reading GastroNet-5M from local LRZ disk.
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
- **Image-level split & pHash dedup** — HyperKvasir's `image-labels.csv` has no patient id,
  so near-duplicate procedure frames across splits are pruned via perceptual hashing
  (Hamming distance ≤ 6 bits). Outputs and run artifacts are placed on `$MCMLSCRATCH`
  to ensure ample disk space.
- DINO's EMA teacher is a deepcopy of a ViT-B → 2× in RAM; DINO SSL needs a GPU
  node, not a login node. LeJEPA is lighter. (Unchanged by the Lightning move —
  it's inherent to the algorithm, not the training-loop implementation.)
- **Checkpoint/resume is Lightning-native**: `last.ckpt` / `ckpt-{step}.ckpt`
  under each run's output dir, resumed via `Trainer.fit(..., ckpt_path=...)`.
  The one file downstream eval reads, `ema_backbone.pt`, keeps its original
  (pre-Lightning) shape on purpose.
- Deep Evidential *Regression* (Amini et al. 2019, arXiv:1910.02600) is not used
  (tasks are classification); it would slot into `vlfz/eval/edl.py` if a polyp-size
  regression target is added later.
