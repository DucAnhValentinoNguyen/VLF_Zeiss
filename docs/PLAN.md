# Zero-shot Calibration on HyperKvasir, before vs after SSL on GastroNet-5M

## UPDATE 2026-09-03 — pivot from REAL-Colon to HyperKvasir

REAL-Colon (946 GB) does not fit: both DSS containers for project `pr74ze` are
full (`pr74ze-dss-0001` 200 GB cap / 199 used; `pr74ze-dss-0000` quota exceeded).
GastroNet-5M is also not on HuggingFace (the HF repo has 3 files) — it comes from
`cortex.thetavision.nl` behind a data-use agreement (user has authorisation).

**New setup (same before/after loop):**
- **Eval dataset = HyperKvasir**, already staged read-only at
  `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra82sat2/zeiss_data`
  (`hyper_kvasir_{labeled,unlabeled,segmented}_images/`).
- **Zero-shot on all downstream tasks** from the label hierarchy
  (`image-labels.csv`: Organ / Classification / Finding):
  `hkv_findings` (23-way), `hkv_category` (4-way), `hkv_tract` (2-way, upper vs
  lower GI), `hkv_pathology` (2-way, pathological vs other), plus `hkv_seg`
  (zero-shot polyp segmentation on the 1000 mask pairs, dense patch-feature k-NN
  label transfer).
- **SSL corpus = GastroNet-5M** (user authorised; drop shard URLs in
  `lrz/gastronet_urls.txt`). Fallback wired: `--corpus hkv_unlabeled` uses the
  staged 99 k unlabeled HyperKvasir pool so a real pre→post result can be produced
  before GastroNet-5M lands.
- **Storage** moved off the full DSS scratch onto the **home quota** (~55 GB free):
  `OUT_ROOT=$HOME/vlf_zeiss_runs`, `GASTRONET_ROOT=$HOME/vlf_zeiss_data/gastronet5m`.
  SSL `subset_images` cut to ~700 k to fit. LeJEPA `train_state.pt` ≈ 1.4 GB;
  keep ≤ 2–3 checkpoints resident.
- One global stratified image-level split (`reference`/`cal`/`query` = .60/.15/.25),
  reused across all tasks. HyperKvasir's CSV has no patient id → image-level split,
  recorded as a caveat (near-duplicate procedure frames may span splits).

REAL-Colon code (`vlfz/data/realcolon.py`) is kept, unused, for if storage frees
up later.

## Context (original — REAL-Colon)

Simple, focused experiment: does in-domain self-supervised pretraining on
**GastroNet-5M** improve the **calibration** (ECE, NLL) of frozen vision
foundation features on **REAL-Colon** binary classification, measured **zero-shot
(weighted k-NN)** — for each of 4 backbone/objective combinations?

**4 model settings** = {DINO v1 self-distillation, LeJEPA} × {SigLIP-2 ViT-B/16
init, ImageNet ViT-B/16 init}.

**Loop:**
1. Evaluate all 4 settings **zero-shot on REAL-Colon** (frozen init, no SSL yet).
2. **SSL-pretrain** each of the 4 on GastroNet-5M (unlabeled): DINO v1 / LeJEPA.
3. Evaluate all 4 **zero-shot on REAL-Colon again**.

**Zero-shot = weighted k-NN on frozen features** (DINO protocol, k=20, τ=0.07);
these backbones are vision-only, so there is no text zero-shot. Normalized
weighted votes give the probability vector; `logit_c = log(vote_mass_c + eps)`
feeds temperature scaling.

**Uncertainty quantification — Evidential Deep Learning** (Sensoy et al. 2018,
arXiv:1806.01768). Two levels, both feeding the same ECE/NLL machinery:
- **Evidential k-NN (training-free, default ON)** — the weighted neighbour vote
  mass *is* Dirichlet evidence: `α_c = λ·vote_mass_c + 1`. Yields the Dirichlet
  mean (a Laplace-smoothed, better-calibrated predictive than raw normalized
  votes), an epistemic *vacuity* `u = C/S`, and error-detection AUROC (does `u`
  predict misclassification?). `λ` is fit on the `cal` split — the evidential
  analogue of temperature scaling. Adds `protocol=eknn` rows + `vacuity_mean`,
  `err_auroc` columns.
- **Evidential linear probe (opt-in, `--edl-head`)** — `Linear(768→C)` on frozen
  cached features, softplus evidence, Sensoy Type-II MLE (digamma) loss + annealed
  KL-to-uniform on misleading evidence. Trains in seconds on cached features. Adds
  `protocol=elin` rows.
**Evidential DL scope (decided 2026-09-03).** The supervisor asked for evidential
deep learning; it is considered **satisfied by the two classification protocols
above** (`eknn` default-on + `elin` opt-in, Sensoy et al. 2018, arXiv:1806.01768).
Deliberately **not** added, to keep the evaluation clean and simple:
- a full end-to-end EDL-trained head + the standard uncertainty benchmarks
  (entropy of correct vs incorrect, accuracy-vs-rejection, vacuity OOD) — deferred;
- **Deep Evidential *Regression*** (Amini et al. 2019, arXiv:1910.02600), the
  Normal-Inverse-Gamma analogue for continuous targets — **deferred**. HyperKvasir
  has no polyp-size-in-mm labels (that was REAL-Colon's `lesion_info.csv`, dropped
  for storage). If revisited, the target is **polyp area-fraction from the 1000
  HyperKvasir segmentation masks** (a new `hkv_polyp_size` task, NIG head on frozen
  features, sibling of `vlfz/eval/edl.py`) — no REAL-Colon needed.

**Metrics:** ECE (15-bin equal-width + adaptive/equal-mass) and NLL, reported
**before and after scalar temperature scaling** (fit on a held-out calibration
split). Context columns: accuracy, balanced accuracy, AUROC.

**Downstream tasks (both, binary):**
- **T1 — frame-level polyp vs no-polyp**: positives = frames with ≥1 polyp box,
  negatives = clean frames.
- **T2 — lesion-crop adenoma vs non-adenoma**: cropped lesion boxes, label from
  `histology_class` (AD vs HP/SSL/TSA).

**Results table:** 4 settings × 2 stages (pre-SSL / post-SSL) × 2 tasks = 16 rows
(each with pre/post-temperature calibration metrics).

`VLF_Zeiss` is empty. **It must be a fully independent repo** — its own venv, its
own code, no runtime import of / dependency on
`~/VL-Foundation-with-Surgeon-Level-Intellect`. That predecessor is used only as
a **read-only design reference**; every helper (DINO v1, LeJEPA, ECE, k-NN,
provenance, folds) is re-implemented cleanly inside `vlfz/`. Missing and to be
built: a calibration module, a weighted k-NN evaluator, backbone-agnostic
full-backbone SSL, and the REAL-Colon / GastroNet-5M loaders.

### Facts / assumptions

- **GastroNet-5M**: ~4.82M unlabeled PNGs in ≤10k-image zip shards; gated via
  `cortex.thetavision.nl` (not HuggingFace). SSL pretraining corpus only. Skewed
  toward **upper GI** (gastroscopy / Barrett's) — a known sub-domain gap vs
  REAL-Colon's colonoscopy; noted as an interpretation caveat, not blocking.
- **REAL-Colon**: 60 videos / 2.76M frames / ~1 TB, CC BY 4.0, from Figshare
  (`figshare_dataset.py`, download takes days). VOC-XML boxes + `video_info.csv` +
  `lesion_info.csv` (`histology_class`). **No official split** → we split by
  **video / patient** into a k-NN **reference** set and a **query** set.
- **Neither dataset is currently on the cluster.** Both downloads (login node,
  run by the user) are on the critical path. Check project quota on
  `dssmcmlfs01` first (~37 TB free, project may be capped).
- **SigLIP-2 ViT-B/16** = timm `vit_base_patch16_siglip*` (v2/webli tag) or
  open_clip `hf-hub:timm/ViT-B-16-SigLIP2`. **ImageNet ViT-B/16** = torchvision
  `vit_b_16(weights=IMAGENET1K_V1)`, converted into the timm module.
- **Dedicated venv** `$VLF_ROOT/.venv` (uv-managed), independent of the predecessor's
  `surg`. Pinned to the same known-good stack: torch 2.5.1+cu121, torchvision
  0.20.1, timm ~1.0.27, open_clip_torch ~3.3, torchmetrics ~1.9 (incl.
  `MulticlassCalibrationError`), scikit-learn, omegaconf, imagehash, tensorboard.
  All LRZ jobs single-GPU H100-94G / A100-80G. No DDP / Hydra / wandb.
- **SSL scope**: full-backbone ViT-B/16 (not LoRA), short schedule on a
  **~1–1.5M-image subset** of GastroNet-5M, single-GPU self-resubmitting
  checkpointed jobs, EMA weights as the eval artifact. `--lora` kept wired as a
  fast fallback. (A full-corpus × 4-run schedule is 10–40 GPU-days/run —
  infeasible single-GPU.)

## Architectural decisions

1. **One importable package `vlfz/`** + thin CLI entrypoints + `lrz/` SLURM dir +
   `config.yaml` + `tests/`.
2. **Backbone standard = timm `vit_base_patch16_224` @ 224px, feature =
   LayerNorm(mean over patch tokens) → 768-dim**, for both inits and both stages.
   Mean-pool erases the SigLIP "no-CLS + attn-pool head" vs ImageNet "CLS token"
   difference; 224 is native for both weight sources and ~4× cheaper than the
   predecessor's 384/patch14 path.
3. **Dedicated weight-ingestion module** with explicit key maps + a forward-parity
   test (torchvision→timm cosine > 0.999; timm-native vs open_clip SigLIP-2
   > 0.99).
4. **k-NN emits probs AND logits.** Temperature scaling (LBFGS on `CE(logits/T,y)`,
   optimise `log T`; Guo et al. 2017) operates on logits.
5. **3-way video-level split of REAL-Colon**: `query` (held-out test videos),
   `reference` (k-NN support videos), `cal` (a few videos, used only to fit T).
   No video appears in more than one; the SSL GastroNet-5M corpus never contains
   REAL-Colon frames (trivially true here, but asserted).
6. **Single-GPU everywhere**, bf16, grad-checkpointing, self-resubmitting
   checkpointed jobs — copy the predecessor SLURM pattern verbatim.

## Reusable modules (predecessor repo — read-only reference, port into `vlfz/`)

| Predecessor file | Reuse |
|---|---|
| `pipeline_demo/jepa_lib.py` | LeJEPA loss stack (`sigreg`, `epps_pulley_1d`, `variance_hinge`, `prediction_loss` w/ stop-grad, `lejepa_loss`, `Predictor`, `TwoViewDataset`, `build_ssl_augment` — NO flips) → port **unchanged** (collapse/NaN fixes are load-bearing) |
| `pipeline_demo/ssl_dino.py` | DINO v1: `MultiCropDataset`, `mc_collate`, `DINOHead`, `dino_loss` (centering+sharpening), EMA teacher, cosine LR / momentum / teacher-temp schedules → port helpers, drop the model class, **remove flips + grayscale** from the multicrop aug |
| `pipeline_demo/jepa_pretrain.py` | SSL loop skeleton: `load_cfg`, checkpoint/resume, grad-accum, NaN-skip guard |
| `pipeline_demo/full_ft_micro.py` | `llrd_param_groups` (layer-wise LR decay) for the SSL optimiser |
| `pipeline_demo/eval_sota.py` | `metrics()` (extend with probs), frozen-feature extractor idiom |
| `pipeline_demo/cnn_cv_benchmark.py` | `provenance()` (seed, git SHA, lib versions, GPU) |
| `pipeline_demo/refresh_report_section2.py` | JSON → markdown table generator |
| `pipeline_demo/config.yaml` | OmegaConf block layout + `${oc.env:VAR,default}` |
| `pipeline_demo/lrz/job_env.sh`, `sbatch_ssl_dino.sbatch`, `submit_evals_chain.sh` | SLURM env (`MCMLSCRATCH`, `wait_for_gpu`, HF-token resolution), single-GPU self-resubmit, `afterany` dependency chain |

## Target repo layout (all new under `VLF_Zeiss/`)

```
config.yaml            README.md
vlfz/
  cfg.py               # load_cfg (OmegaConf) + provenance()   [port]
  data/
    realcolon.py       # NEW: Figshare loader; T1 frame index (+ negatives); T2 lesion crops; video-level splits
    gastronet.py       # NEW: manifest over zip shards + lazy zip-member reader (no extraction) + subset sampler
    transforms.py      # build_ssl_augment (no flips) + MultiCropDataset + eval tfm (224, ImageNet norm)   [port]
  models/
    vit_backbone.py    # NEW: timm ViT-B/16 factory + uniform .feature(x) / .forward_tokens(x)
    load_weights.py    # NEW: 3-source state_dict loader (siglip2 / imagenet / our SSL ckpt) + strict report
    convert_tv_vit.py  # NEW: torchvision vit_b_16 -> timm state_dict (fused-qkv, name map, pos-embed identity)
    heads.py           # Predictor + DINOHead + Projector   [port]
    ema.py             # NEW: backbone weight-EMA wrapper (SSL eval artifact)
  ssl/
    lejepa_lib.py      # = jepa_lib.py ported UNCHANGED
    dino_lib.py        # ssl_dino.py helpers / schedules / dino_loss   [port]
    pretrain.py        # NEW unified entrypoint: --objective {lejepa,dino} --init {siglip2,imagenet}
                       #     --stage {smoke,full} --resume [--lora]
  eval/
    features.py        # NEW: one backbone-agnostic frozen-feature extractor -> feat_<key>.npz (cached)
    knn.py             # NEW: weighted k-NN (k=20, tau=0.07) -> (probs, logits, vote_mass); k-sensitivity sweep
    calibration.py     # NEW: ECE (15-bin equal-width + adaptive), NLL, Brier, reliability bins, fit_temperature (LBFGS)
    edl.py             # NEW: evidential k-NN (Dirichlet from vote mass) + evidence-scale fit + opt-in evidential linear head
    metrics.py         # accuracy / balanced_acc / macro_f1 / AUROC + flat-row assembly
    run_eval.py        # NEW orchestrator: (setting x stage x task x protocol x pre/post-T) -> results/*.json
  report/
    aggregate.py       # NEW: scan results/*.json -> long_results.csv + delta (post-SSL minus pre-SSL)
    pivot.py           # markdown pivot + SUSPECT / leakage flags   [port refresh_report_section2.py]
    leakage_audit.py   # NEW: pHash overlap (GastroNet-5M subset vs REAL-Colon query frames) -> expect ~0
lrz/
  job_env.sh                    # port; OUT_ROOT=$MCMLSCRATCH/vlf_zeiss_runs; reuse surg venv
  sbatch_ssl_pretrain.sbatch    # self-resubmitting SSL (variant via env OBJ/INIT/STAGE)   [port sbatch_ssl_dino.sbatch]
  sbatch_extract_features.sbatch  sbatch_eval.sbatch  sbatch_aggregate.sbatch
  submit_ssl_matrix.sh          # queue the 4 SSL runs
  submit_eval_chain.sh          # afterany chain: features -> eval -> aggregate   [port submit_evals_chain.sh]
tests/
  test_backbone_parity.py  test_calibration.py  test_knn_probs.py
  smoke_ssl.sh  smoke_eval.sh
```

## Phased plan

Build order: **P0 → (P1 ∥ P2) → P3 → P4 (first results) → P5 → P6 (final results)**.
P5 depends on the GastroNet-5M download; P3/P4 depend on the REAL-Colon download.
Both downloads are user-run on login nodes and are the critical path.

### Phase 0 — Scaffold + config  (~0.5 day)
Package skeleton; `config.yaml` porting the predecessor block layout
(`paths`, `backbones`, `ssl`, `eval`, `calibration`, `smoke`/`full`,
`${oc.env:...}`); `lrz/job_env.sh` (copy predecessor, change only `OUT_ROOT` →
`$MCMLSCRATCH/vlf_zeiss_runs`); `vlfz/cfg.py` (`load_cfg` + `provenance`).
**Verify**: `python -c "import vlfz"`; `run_eval.py --help`.

### Phase 1 — Backbone-agnostic ViT-B/16 + weight ingestion  (~2–3 days)
- `vit_backbone.py`: `build_vit_b16(init, img_size=224, dynamic_img_size=True)`
  over `timm.create_model("vit_base_patch16_224", num_classes=0, global_pool="")`;
  `.feature(x)`→(B,768) LN(mean tokens); `.forward_tokens(x)`→(B,196,768).
- `load_weights.py`: `load_siglip2` (timm-native tag first; open_clip
  `visual.trunk.*` remap fallback), `load_imagenet` (→ converter), `load_ssl_ckpt`;
  print `missing`/`unexpected` counts.
- `convert_tv_vit.py`: `conv_proj→patch_embed.proj`, `class_token→cls_token`,
  `encoder.pos_embedding→pos_embed`, `encoder.ln→norm`, `ln_1/ln_2→norm1/norm2`,
  `mlp.linear_1/linear_2→mlp.fc1/fc2`, `self_attention.out_proj→attn.proj`; fused
  `in_proj_{weight,bias}` (3·768) → timm fused `attn.qkv`. Pos-embed 14×14+1 at
  224 → identity.
- **Risks**: fused-attention layout; SigLIP has no CLS + attn-pool head (mean-pool
  sidesteps it — verify token count / pos-embed interpolation); open_clip SigLIP-2
  config-name drift.
- **Smoke** `test_backbone_parity.py`: 8 fixed JPEGs → finite (8,768) each;
  torchvision path cosine vs reference `torchvision.models.vit_b_16` pre-head
  > 0.999; timm-native vs open_clip SigLIP-2 > 0.99.

### Phase 2 — Calibration module + k-NN evaluator  (~2 days, parallel with P1)
- `calibration.py`: `ece_equal_width(probs,y,n_bins=15,top_label=True)`
  (cross-checked vs `torchmetrics ... MulticlassCalibrationError(n_bins=15,
  norm="l1")`; returns per-bin conf/acc/count); `ece_adaptive` (equal-mass bins);
  `nll(probs,y,eps=1e-7)`; `brier`; `fit_temperature(logits,y)→T` (LBFGS on
  `log T`); `calibration_report(...)` → `{pre:{ece_ew,ece_adaptive,nll,brier},
  post:{...}, T, reliability_bins}`.
- `knn.py`: `knn_predict(Xref,yref,Xq,k=20,tau=0.07)` — cosine sims, top-k, weights
  `softmax(sim/τ)`, per-class vote mass → normalized `probs`,
  `logits=log(vote_mass+eps)`; GPU torch; helper to sweep k∈{20,50,200}.
- `metrics.py`: `context_metrics` = ported `eval_sota.metrics` + `auroc`
  (`roc_auc_score`, binary) + `balanced_acc`; `full_row(...)` merges
  context + calibration.
- **Smoke** `test_calibration.py`: perfectly-calibrated synthetic → `ece_ew<0.02`;
  overconfident → `T>1`, post-T NLL < pre-T on held-out; one-hot-wrong + eps →
  NLL finite; own ECE vs torchmetrics within 1e-6. `test_knn_probs.py`: 3-blob toy
  → acc high, `probs.sum(1)=1`.

### Phase 3 — REAL-Colon acquisition + loader + tasks + splits  (~2–3 days; blocked on download)
- `lrz/download_realcolon.sh` (login node, user-run): `figshare_dataset.py` →
  `$MCMLSCRATCH/real_colon/` (~1 TB — confirm quota; resumable).
- `realcolon.py`:
  - parse `video_info.csv`, `lesion_info.csv` (`histology_class`), VOC-XML boxes;
  - **T1 `frame_task()`**: per-frame label (≥1 box → polyp; else no-polyp);
    subsample clean frames to a fixed ratio (e.g. 1:1 or 1:3) to bound size and
    balance;
  - **T2 `lesion_crop_task()`**: crop each box (with margin) → adenoma
    (AD) vs non-adenoma (HP/SSL/TSA); drop `NO POLYP`/`OTHER`;
  - **`video_splits(seed=0)`**: partition the 60 videos into `query` / `reference`
    / `cal` (e.g. 24 / 28 / 8), stratified on lesion prevalence and, where
    recorded, clinical center; persisted to `splits.json`.
- **Risks**: T2 sample count may be small (a few hundred lesions) → wide CIs on
  ECE; class/center imbalance → stratify splits, report balanced-acc + AUROC;
  frame near-duplicates → the video-level split already prevents leakage.
- **Smoke**: build both task manifests on 4 videos; assert no video spans two
  splits; assert T1 label balance within target ratio.

### Phase 4 — "Before SSL" evaluation  (~0.5 day compute)  ← FIRST RESULTS
- `run_eval.py --stage pre --settings dino_siglip2,dino_imagenet,lejepa_siglip2,lejepa_imagenet`
  — note pre-SSL the objective is irrelevant, so this is really just the 2 frozen
  inits (siglip2, imagenet); recorded under all 4 setting names for a clean
  before/after diff, or collapsed to 2 rows with a note.
- Per (init × task): extract `reference` + `cal` + `query` features → `knn_predict`
  → `fit_temperature` on `cal` → pre/post-T `calibration_report` + `context_metrics`
  → `results/{init}__{task}__pre.json`.
- `aggregate.py` → first `long_results.csv` + markdown pivot (the "no in-domain
  SSL" baseline).

### Phase 5 — SSL pretraining on GastroNet-5M  (compute-heavy; 4 runs; blocked on download)
- `lrz/download_gastronet.sh` (login node, user-run; requires the thetavision data
  agreement) → zip shards under `$MCMLSCRATCH/gastronet5m/`.
- `gastronet.py`: `build_manifest(root)→manifest.tsv` of `(zip_path, member)` via
  one `namelist()` pass (cached); `GastroNetDataset` opens zips lazily per worker,
  decodes bytes → PIL, no extraction; `--subset-frac` for the ~1–1.5M class/shard-
  balanced subset; corrupt-member guard → black placeholder.
- `ssl/pretrain.py`: model = `build_vit_b16(init)` + (`Projector` for lejepa /
  `DINOHead` for dino). **Full backbone trainable, no LoRA** (`--lora` fallback).
  grad-checkpointing; bf16; AdamW; base LR ~1e-4 cosine; LLRD ~0.75
  (`llrd_param_groups`); grad-clip + NaN-skip; EMA backbone → `ckpt/ema_backbone.pt`;
  `train_state.pt` for resume. Data = `TwoViewDataset` (lejepa) / `MultiCropDataset`
  (dino, no flips). TensorBoard scalars: loss terms, feature-std, effective-rank
  collapse proxy.
- `sbatch_ssl_pretrain.sbatch` (port `sbatch_ssl_dino.sbatch`): single-GPU,
  `--time=12:00:00`, self-resubmits until `ckpt/DONE`; `submit_ssl_matrix.sh`
  queues the 4 runs {lejepa,dino}×{siglip2,imagenet}, ~1–1.5M subset, ~40–60M
  images seen (≈2–4 resubmits each). LeJEPA ~2–3× cheaper than DINO multicrop.
- **Gate**: after each run, `run_eval.py --stage post --settings <one>` quick pass;
  if k-NN balanced-acc on T1 regresses vs the frozen init, flag "SSL regression —
  investigate" (do not silently publish).
- **Risks**: full-backbone SSL from a strong init drifts/forgets on a short
  schedule → low LR + LLRD + short schedule + EMA weights + the regression gate;
  upper-GI→colon domain gap limits T2 gains (interpretation caveat);
  representation collapse → SIGReg + variance-hinge + stop-grad (LeJEPA),
  centering + sharpening (DINO), smoke asserts feature-std > 0.1.
- **Smoke** `smoke_ssl.sh`: `STAGE=smoke` on ~2000 images (or staged
  `hyper_kvasir_unlabeled_images` if GastroNet-5M not yet down), 1 epoch, all 4
  settings → loss finite & decreasing, `ema_backbone.pt` written, `--resume`
  restores step.

### Phase 6 — "After SSL" evaluation + final report  (~0.5 day compute, chained)
- `submit_eval_chain.sh` (`--dependency=afterany`): for each of the 4 SSL
  checkpoints → `extract_features` → `eval` (k-NN + calibration on T1 & T2) →
  final `aggregate`.
- `aggregate.py` → `long_results.csv` with the 16 rows + a **delta view**
  (post-SSL minus pre-SSL ECE / NLL / AUROC per setting × task).
- `pivot.py` → markdown pivot (`init × objective × stage × task × {pre,post}-T`).
- `leakage_audit.py` → pHash overlap GastroNet-5M-subset vs REAL-Colon query
  frames (expected ~0; recorded in every JSON).

## Results schema (`long_results.csv`)

`init` (siglip2|imagenet) · `objective` (none|lejepa|dino) · `stage` (pre|post) ·
`task` (rc_frame|rc_lesion) · `protocol` (knn|eknn|elin) · `temp_scaled` (bool) ·
`T` (or evidence-scale λ) · `ece_ew` · `ece_adaptive` · `nll` · `brier` ·
`accuracy` · `balanced_acc` · `macro_f1` · `auroc` · `vacuity_mean` (eknn/elin) ·
`err_auroc_vacuity` (eknn/elin) · `k` · `n_query` · `overlap_count` · `git_sha` ·
`seed` · `timestamp` · `ckpt_path`.

## Cross-cutting risks & mitigations

| Risk | Mitigation |
|---|---|
| tv/open_clip/timm ViT-B/16 key mismatch (fused qkv, no-CLS SigLIP, pos-embed) | dedicated `convert_tv_vit.py` + `load_weights.py` strict report; parity test cosine gate (>0.999 tv / >0.99 siglip) |
| Full SSL on 4.8M single-GPU too costly | 4 runs × 8 epochs × 4.82M ≈ 154M imgs seen ≈ ~85 GPU-h total on free LRZ H100s (spread across self-resubmits); LeJEPA ~2–3× cheaper than DINO multicrop. Curated 224px tier keeps the working set local so steps are compute- not IO-bound. |
| k-NN "probabilities" are not posteriors → raw ECE/NLL misleading | `logits = log(vote_mass + eps)`; mandatory temperature scaling; eps smoothing; report k∈{20,50,200} sensitivity; compare RELATIVE (pre vs post) not absolute |
| Upper-GI (GastroNet-5M) → colon (REAL-Colon) domain gap | stated as interpretation caveat; T1 expected to benefit more than T2; natural follow-up = add unlabeled REAL-Colon frames to the SSL corpus |
| T2 lesion count small → wide ECE CIs | report bootstrap CIs; stratify video split on lesion prevalence + center; keep T1 as the primary story |
| Full-backbone SSL forgets the strong init | low LR + LLRD 0.75 + short schedule + EMA weights + "k-NN must not regress" gate before publishing |
| SSL representation collapse | LeJEPA SIGReg + variance-hinge + stop-grad ported unchanged; DINO centering + sharpening; log feature-std + effective-rank; smoke asserts std > 0.1 |
| GastroNet-5M / REAL-Colon storage & gated/slow downloads | **Phase 7**: S3 lake holds the full corpus; a hard-curated 224px tier (~50 GB) is staged once to LRZ so all epochs read local disk (one-time egress ~$5). REAL-Colon slots into the same `raw/` → `curate` → `stage` path later. |

## Implementation status (as built in `VLF_Zeiss/`)

**Done + verified on the login node (17 unit tests green, CPU smokes green):**
- `.venv` (uv, torch 2.5.1+cu121 stack) via `lrz/setup_env.sh`.
- `vlfz/models/`: `build_vit_b16("siglip2"|"imagenet"|ckpt)` — SigLIP-2 via timm
  tag `vit_base_patch16_siglip_gap_224.v2_webli` (open_clip fallback), ImageNet
  via **exact** torchvision→timm conversion (parity test: cosine > 0.999,
  missing=unexpected=0). `heads.py` (Projector/Predictor/DINOHead — no
  weight_norm, deepcopy-safe), `ema.py`.
- `vlfz/ssl/`: `lejepa_lib.py` (SIGReg/variance-hinge/stop-grad, numerics tested),
  `dino_lib.py` (loss + centering + schedules, tested), `pretrain.py` unified
  entrypoint — LeJEPA end-to-end incl. checkpoint **and resume** verified.
- `vlfz/data/`: `realcolon.py` (VOC parse, `rc_frame`/`rc_lesion` builders,
  by-video `grouped_stratified_split`), `gastronet.py` (zip manifest + lazy
  reader + shard-balanced subset), `transforms.py`, `folds.py`. Exercised
  end-to-end on a synthetic REAL-Colon tree (`tests/make_fake_realcolon.py`).
- `vlfz/eval/`: `features.py` (cached extraction), `knn.py` (probs/logits/
  vote_mass; k-sweep), `calibration.py` (ECE ew+adaptive / NLL / Brier /
  temperature — matches torchmetrics to 1e-6, T clamped to [1e-2,1e3]),
  `edl.py` (evidential k-NN + evidence-scale fit + optional evidential linear
  head), `metrics.py`, `run_eval.py` orchestrator.
- `vlfz/report/`: `aggregate.py` (long_results.csv + post−pre delta), `pivot.py`
  (markdown, SUSPECT flag), `leakage_audit.py` (pHash overlap).
- `lrz/`: `job_env.sh`, self-resubmitting `sbatch_ssl_pretrain.sbatch`,
  `sbatch_eval.sbatch`, `sbatch_aggregate.sbatch`, `submit_ssl_matrix.sh`,
  `submit_eval_chain.sh` (afterany chain), `download_realcolon.sh`,
  `download_gastronet.sh`.

**Needs a GPU compute node (login node OOMs on 2× ViT-B):**
- DINO+ViT-B SSL end-to-end (`ALL=1 bash tests/smoke_ssl.sh` under `srun`).
- Full-memory LeJEPA resume.

**Blocked on user-run data downloads (login node, hours–days):**
- `lrz/download_realcolon.sh` (~1 TB) → then `python -m vlfz.data.realcolon
  --make-splits --inspect`.
- `lrz/download_gastronet.sh` (gated, cortex.thetavision.nl, ~0.5–1 TB).

**Not committed** — branch `feat/calibration-eval-pipeline` created, files staged,
awaiting the user's go-ahead to commit.

---

### 2026-09-03 pivot addendum (HyperKvasir)

**Added / changed:**
- `vlfz/data/hyperkvasir.py` — `image-labels.csv` label table, `scan_labeled_images`
  (memoised), one global stratified image-level split, four classification task
  builders (`hkv_findings/category/tract/pathology`), `seg_pairs` for the 1000
  mask pairs, `unlabeled_paths` (99k SSL fallback corpus).
- `vlfz/data/registry.py` — `get_dataset("hyperkvasir"|"realcolon")` uniform
  provider so `run_eval` is dataset-agnostic.
- `vlfz/eval/seg.py` — zero-shot polyp segmentation: dense ViT patch tokens (14×14)
  → k-NN label transfer from a small support set → Dice / mIoU + pixel-wise
  ECE/NLL (global + foreground).
- `vlfz/eval/run_eval.py` — `--dataset` + `--corpus`; `--task all` honours
  `cfg.hyperkvasir.tasks`; results JSON named `<dataset>__<tag>__<task>.json`;
  rows carry `dataset` + `corpus`.
- `vlfz/ssl/pretrain.py` — `--corpus {gastronet,hkv_unlabeled}`; ckpt dir is
  `{obj}_{init}_{corpus}_{stage}`; auto-falls back to `hkv_unlabeled` when no
  GastroNet shards present.
- `config.yaml` / `lrz/job_env.sh` — storage on the **home quota**
  (`OUT_ROOT=$HOME/vlf_zeiss_runs`, `GASTRONET_ROOT=$HOME/vlf_zeiss_data/...`);
  `ssl.subset_images` 700k; new `hyperkvasir:` block.
- SLURM: `sbatch_eval.sbatch` (+seg step), `submit_eval_chain.sh`,
  `submit_ssl_matrix.sh` take `DATASET` / `CORPUS`.
- Smokes rewritten for HyperKvasir (`tests/smoke_config.yaml`,
  `tests/smoke_eval.sh`); the fake-REAL-Colon fixture is retained for realcolon.

**Verified on the login node:** unit tests green; HyperKvasir `--inspect`
(10,619 labelled images, task class counts as expected); pre-SSL zero-shot on
`hkv_tract` + `hkv_category` writes calibrated results JSON; SSL smoke + post eval
+ aggregate in progress.

**Storage reality:** DSS project scratch full (200 GB cap, 199 used; second
container quota-exceeded). Home has ~55 GB free — enough for outputs + a ~0.7 M
GastroNet-5M subset, not the full corpus or REAL-Colon.

**Still needs the user:** GastroNet-5M shard URLs from the authorised
`cortex.thetavision.nl` session into `lrz/gastronet_urls.txt` (the HF repo has no
data). Until then SSL runs on `--corpus hkv_unlabeled` (staged 99k).

---

## Phase 7 — GastroNet-5M cloud data lake + pipeline (AWS)  [2026-09-03]

### Context

GastroNet-5M (~4.82M images, ~1 TB in 506 portal zips) will not fit LRZ: total
writable quota is ~90 GB (home ~61 + `dss-0001` ~29; `dss-0000` quota-exceeded).
The `hyper_kvasir_unlabeled_images` fallback pool **has been deleted** — verified
2026-09-03, only `hyper_kvasir_{labeled,segmented}_images/` remain — so SSL
(Phase 5) has **no corpus at all** until GastroNet-5M lands. The user wants the
**full dataset** for training quality, and wants the acquisition layer built as a
**clean, scalable, cost-effective data pipeline** for an AI/Data-Engineer job
application (REAL-Colon likely added the same way later). Budget: a ~$100 AWS
promotional credit (EC2/S3/Athena/Batch/SageMaker eligible), un-shared,
single-account. Chosen stack: **AWS S3 + Athena**, region **eu-north-1**
(Stockholm), ingest via an **ephemeral EC2**. Outcome: S3 becomes the durable
data lake holding 100% of GastroNet-5M; a hard-curated tier (~40–60 GB) is staged
once to LRZ so every SSL epoch iterates all ~4.8M images from local disk.

### Key decisions

- **S3 bucket = the lake** (`gastronet5m-lake-<acct>-euno1`, versioned, SSE-S3,
  block-public-access), prefixes:
  - `raw/` — the 506 portal zips as-is (~1 TB). Lifecycle → **Glacier Instant
    Retrieval** after 30 d (~$4/mo); Deep Archive (~$1/mo) if credit runs tight.
    This is the pristine, reproducible copy and the "add REAL-Colon later" slot.
  - `webdataset/` — **curated tier**: decode → resize to 224 px → re-encode
    (JPEG q87, or WebP q80) → drop corrupt + pHash-near-duplicates → pack into
    ~1 GB WebDataset `.tar` shards. Estimated **30–65 GB** (4.82M × ~10–14 KB,
    less ~20–40% dedup on extracted-frame near-dups).
  - `catalog/manifest/` (raw) + `catalog/curated/` — **Parquet**, partitioned by
    `shard_group`; Glue external tables; queried with **Athena**.
  - `ingest_log/`, `dq/` — per-run JSON logs + the data-quality report.
- **Training data flow — stage once, read local (recommended, ~$5 egress total):**
  `aws s3 sync s3://<bucket>/webdataset/ $GASTRONET_ROOT` on the LRZ login node —
  the whole curated tier (~50 GB) fits the home quota (or node-local
  `$SLURM_TMPDIR` per job). All 4 SSL runs × 8 epochs iterate every image from
  local disk. One-time egress ≈ 50 GB × $0.09 ≈ **$4.50**.
- **`s3_webdataset` streaming mode — built + documented, not the default:** a
  `webdataset` shardlist of `s3://` URLs with a bounded local LRU shard cache, for
  when the curated tier outgrows local storage (e.g. after adding REAL-Colon).
  Costs ~1 full-corpus egress per pass (~$18 at 200 GB) — never run multi-epoch
  against it. Needs compute-node outbound HTTPS (job `5772201` pending; the
  recommended login-node stage-in does **not**).
- **Training stays on free LRZ H100s.** SageMaker is a non-goal (4 multi-day runs
  would exhaust the credit); the `s3_webdataset` source mode makes a SageMaker /
  EC2-GPU training job a drop-in if ever wanted — a talking point, not a build.
- **Ingest = ephemeral t3.small in eu-north-1**, Terraform-provisioned, **instance
  profile** with write to `raw/` (no static key ever created), `rclone`/`wget`
  portal → `s3://<bucket>/raw/`, auto-terminate on completion.
- **Curation = AWS Batch on Fargate Spot**, array job over shard-groups (container
  in ECR runs `curate/curate_shard.py`); ~30 vCPU-h ≈ <$1. Fallback: run the same
  script on a larger spot EC2.
- **Two IAM principals:** `gastronet-ingest` (instance-profile role, RW
  `raw/`+`webdataset/`+`catalog/`) and `gastronet-reader` (**read-only**, whole
  bucket — this is the key already supplied; used only by LRZ).

### Cost (full dataset; fits the $100 credit)

| Item | One-time | Monthly |
|---|---|---|
| S3 ingress (portal → bucket) | $0 | — |
| EC2 t3.small ingest box (~24 h) | ~$0.50 | — |
| AWS Batch curation (Fargate Spot) | ~$1–10 | — |
| S3 Standard — `raw/` ~1 TB, first 30 d before lifecycle | ~$8 (prorated) | — |
| S3 Glacier IR — `raw/` ~1 TB after lifecycle | — | ~$4.00 |
| S3 Standard — `webdataset/` ~50 GB | — | ~$1.20 |
| Athena (Parquet, partitioned → MB/query) | — | ~$0 |
| Egress — one-time curated stage-in to LRZ (~50 GB) | ~$4.50 | — |
| **Total** | **~$25–35** | **~$5 / mo** |

~$90 over a full year; ~$40 if the project finishes in ~2 months and the bucket
is deleted. AWS Budgets alarm at $25/mo (soft) + $60 (hard) → SNS email.

### Repo layout (new — independent, all under `VLF_Zeiss/`)

```
pipeline/
  infra/           Terraform: S3 (versioning, SSE-S3, lifecycle, block-public),
                   Glue DB + Athena workgroup + external tables, IAM
                   (gastronet-ingest role / gastronet-reader user), AWS Batch
                   (Fargate Spot compute env + queue + job def) + ECR repo,
                   ephemeral-ingest EC2 launch template, Budgets alarm + SNS.
                   Remote state in a Terraform-bootstrapped tfstate bucket.
  ingest/          portal_to_s3.py  — rclone/boto3 multipart, sha256 per object,
                   retry/backoff, writes ingest_log/*.json ; idempotent (skip
                   objects already present with matching size+etag).
  catalog/         build_manifest.py — one zipfile.namelist() pass over every
                   shard → Parquet (shard, shard_group, member, bytes, sha256,
                   ext, width, height, phash) → catalog/manifest/ .
  curate/          Dockerfile + curate_shard.py — resize 224 / re-encode / drop
                   corrupt + pHash-dup / pack WebDataset .tar → webdataset/ +
                   catalog/curated/ partial-manifest rows.
  stage/           select_subset.py (Athena/parquet query → shard list; default
                   = all) + stage_in.sh (aws s3 sync → $GASTRONET_ROOT,
                   sha256-verified, resumable).
  dq/              checks.py → dq/data_report.md + PNGs: row count vs ~4.82M,
                   corrupt %, dup-cluster %, image-dim & RGB/grayscale
                   histograms. Reuses vlfz/report/leakage_audit.py pHash code.
  flow.py          orchestration DAG: ingest → catalog → curate → dq → stage →
                   sbatch. Prefect (nice UI screenshot); `make` target fallback.
  README.md        architecture diagram + runbook + the cost table above.
```

- **`vlfz/data/gastronet.py`** — add `source ∈ {local_zip, local_webdataset,
  s3_webdataset}`. `local_webdataset` becomes the SSL default; `local_zip` kept
  for back-compat. Reuse the existing `subset_indices` / shard-balance logic;
  swap the `.tsv` manifest reader for the Parquet `catalog/` (pyarrow). WebDataset
  iteration replaces the per-worker `zipfile` handle cache.
- **`vlfz/ssl/pretrain.py::_make_dataset`** — thread `--source` through for
  `--corpus gastronet`; change the now-dead `hkv_unlabeled` auto-fallback (pool
  deleted) into a clear error: *"no local GastroNet tier — run `pipeline/stage`"*.
- **`pyproject.toml`** — new `pipeline` optional-dependency extra: `boto3`,
  `s3fs`, `webdataset`, `pyarrow`, `prefect` (optional). Core training env
  unchanged; `imagehash` already present.
- **`lrz/job_env.sh`** — export `AWS_PROFILE=gastronet-reader`,
  `AWS_DEFAULT_REGION=eu-north-1`. **`lrz/setup_env.sh`** — `uv tool install
  awscli` + fetch the `rclone` static binary into `~/bin`.
- **`.gitignore`** — add `pipeline/infra/.terraform*`, `*.tfstate*`, `*.tfvars`,
  `.aws/`, `*.csv`, `*accessKeys*`, `pipeline/**/secrets*`.
- **`lrz/download_gastronet.sh`** — repoint at `pipeline/stage/stage_in.sh`
  (keep the URL-list path as a manual fallback).

### Secrets handling

- The supplied key is **read-only** → LRZ only: `~/.aws/credentials`
  `[gastronet-reader]`, `chmod 600`; mirrored to a git-ignored `lrz/.aws_env`.
  Never echoed, never committed.
- The write path uses the EC2 **instance profile** — no static write key exists.
- Terraform runs on the **user's laptop / an admin box** with their admin creds
  (never on LRZ, never committed); state in a dedicated `tfstate` bucket.
- **Recommend rotating the pasted reader key** in IAM once stage-in completes
  (low urgency: read-only + single-bucket scope, but it is now in chat history).

### Build order

`P7.1` Terraform: bucket + IAM + Glue/Athena + Batch + Budgets →
`P7.2` ephemeral-EC2 ingest: portal → `raw/` (~1 TB) →
`P7.3` `catalog/build_manifest.py` → Parquet + Athena verify (`count(*)` ≈ 4.82M,
`sum(bytes)` sane) →
`P7.4` Batch curation → `webdataset/` + `catalog/curated/` →
`P7.5` `dq/checks.py` → `data_report.md` (inspect before proceeding) →
`P7.6` `stage_in.sh` → LRZ `$GASTRONET_ROOT`; `gastronet.py` source modes; SSL
smoke on the staged tier →
`P7.7` `flow.py` DAG + `README.md` + architecture diagram →
**unblocks Phase 5** (`--corpus gastronet --source local_webdataset`, real runs).

### Implementation status (as built, 2026-09-04 — not yet applied to AWS)

**Code complete, verified read-only (py_compile clean; no test depends on the
changed internals), nothing committed:**
- `pipeline/infra/*.tf` — bucket (versioned/SSE/private/lifecycle), Glue+Athena
  (workgroup + `manifest`/`curated` external tables), IAM (`gastronet-ingest`
  instance-profile role scoped to the bucket; `gastronet-lake-ro` managed policy
  to attach to the existing `read-only-agent` user — deliberately NOT auto-
  attached by Terraform, since that user/key already exists outside this state),
  ephemeral spot EC2 (`c7i.2xlarge`, SSM-driven, no inbound, ships `pipeline/` via
  an `archive_file` → `bootstrap/pipeline_src.zip`), Budgets alarm (40%/100% →
  SNS email). `terraform validate` not run (no `terraform` binary on LRZ by
  design — it's meant to run from the user's admin box).
- `pipeline/{ingest,catalog,curate,dq,stage}/` — portal→`raw/` (idempotent,
  sha256, `--url-list` or `--portal-json` cookie auth), Parquet manifest builder
  (+ phash), curator (resize 224 / JPEG q87 / within-group phash-dedup / pack
  WebDataset `.tar`, multiprocessing over `shard_group`s, doubles as the
  `curate/Dockerfile` AWS-Batch array-job image if ever needed), DQ report
  (counts vs 4.82M, corrupt %, dup %, dim histograms → `data_report.md`),
  `select_subset.py` (default = every shard; optional global-dedup / cap) +
  `stage_in.sh` (sha256-verified `aws s3 sync` to `$GASTRONET_ROOT/webdataset/`).
  `flow.py` + `Makefile` sequence all of it over SSM from a laptop.
- `vlfz/data/gastronet.py` — `resolve_source()` + `webdataset_shards()` +
  `webdataset_loader()` for `local_webdataset` (default) / `s3_webdataset`;
  `local_zip` path untouched. `vlfz/ssl/pretrain.py` — `_make_loader()` replaces
  `_make_dataset()`, handles both map-style and iterable (wds) loaders and
  derives `steps_per_epoch` for the latter; the dead `hkv_unlabeled` auto-
  fallback now raises a clear actionable error instead of silently misrouting.
  `config.yaml` gained `ssl.gastronet_source/_bucket/steps_per_epoch/
  wds_images_per_shard`. `pyproject.toml` gained a `pipeline` extra (boto3,
  pyarrow, webdataset, matplotlib). `lrz/job_env.sh` exports
  `AWS_PROFILE=gastronet-reader` + resolves `$GASTRONET_BUCKET` from
  `~/.gastronet_bucket`; `lrz/setup_env.sh` installs the `pipeline` extra +
  `awscli` + a static `rclone`. `lrz/download_gastronet.sh` now just calls
  `stage_in.sh` (`--zips` kept as the legacy raw-shard fallback). `.gitignore`
  covers `pipeline/infra/.terraform/`, `*.tfstate*`, `*accessKeys*`.
- **AWS reachability confirmed from an LRZ H100 compute node** (job `5772201`,
  not just the login node): `s3.eu-north-1.amazonaws.com`, `storage.googleapis.com`,
  and `cortex.thetavision.nl` all answered directly (no proxy needed) — so the
  optional `s3_webdataset` streaming mode is genuinely usable if the curated tier
  ever outgrows local disk, not just a paper option.
- **Read-only reader key stored & verified**: `~/.aws/credentials` profile
  `gastronet-reader` (chmod 600) on LRZ; `aws sts get-caller-identity` confirms
  account `709569057971`, user `read-only-agent`; it correctly **cannot**
  `s3:ListAllMyBuckets` / read IAM / Athena / EC2 — i.e. it has no standing
  permissions yet (none attached before the bucket + policy exist), consistent
  with "attach `gastronet-lake-ro` after `terraform apply`."
- Smoke job `5772000`: 17/17 unit tests green; the SSL-smoke step now fails
  **fast and clearly** (`corpus=hkv_unlabeled but no images … use --corpus
  gastronet after pipeline/stage/stage_in.sh`) instead of silently doing the
  wrong thing — expected until either `terraform apply` + ingest/curate/stage
  produces a local `webdataset/` tier, or `tests/smoke_ssl.sh` is pointed at a
  tiny `--corpus gastronet --source local_zip` fixture for CI purposes.

**Not done — needs the user, off LRZ:**
- `cd pipeline/infra && cp terraform.tfvars.example terraform.tfvars` (set
  `alert_email`), `terraform init && terraform apply` from an admin-credentialed
  laptop/box — **not** from LRZ (only the scoped reader key lives there).
- Portal auth mechanics for `ingest/portal_to_s3.py` (URL list vs. a session
  cookie / file-list JSON) — still open, see below.
- After `apply`: `aws iam attach-user-policy --user-name read-only-agent
  --policy-arn <gastronet_reader_policy_arn output>`, drop the bucket name into
  `~/.gastronet_bucket` on LRZ, then `make -C pipeline ingest catalog curate dq`
  (or `flow.py run --step all`), then `bash pipeline/stage/stage_in.sh` on the
  LRZ login node.

### From the user before execution

- AWS **account ID**; confirm region **eu-north-1**; admin creds available
  locally for `terraform apply` (not on LRZ).
- **Portal download mechanics**: is there a URL list / API token / resumable
  client, or is it browser-SSO only? (decides whether the EC2 ingest runs fully
  headless or needs a short-lived session cookie pasted onto the box).
- Keep `raw/` in **Glacier IR** (~$4/mo) vs **Deep Archive** (~$1/mo, slower
  restore)?
- Confirm the bucket name, or accept the generated `gastronet5m-lake-<acct>-euno1`.

---

## 2026-09-04 addendum — simplify the benchmark, stream+cache GastroNet, Lightning

Four decisions from the same round; each **supersedes** the matching earlier
text rather than rewriting history in place.

**1. Eval dataset — reaffirmed, no new work.** HyperKvasir is (still, since
2026-09-03) the only eval dataset. `vlfz/data/realcolon.py` and
`realcolon_tasks:` stay in the repo but are **not referenced** by
`registry.py`'s active path or `run_eval.py` defaults — already true today, so
this is a confirmation, not a build item.

**2. Evidential deep learning — DITCHED** (supersedes the 2026-09-03 EDL-scope
decision, which kept `eknn`/`elin`). Reason: keep the benchmark clean and
simple — a single `knn` protocol, temperature-scaled, is the whole story.
- `vlfz/eval/run_eval.py`: default `--protocols` becomes `knn` only; drop the
  `eknn` block and the `edl_head`/`elin` branch; drop the `from .edl import ...`.
- `vlfz/eval/edl.py` stays in the repo, **unimported**, as a dormant reference
  (same treatment as `realcolon.py`) — not deleted, not wired.
- `vlfz/report/aggregate.py` / `pivot.py`: drop `vacuity_mean`, `err_auroc`,
  evidence-scale λ from the results schema and pivot columns; `_PRIMARY` /
  `_CONTEXT` unchanged otherwise.
- `README.md`: replace the EDL paragraph with one line noting it was evaluated
  and dropped in favour of a simpler benchmark.
- `pipeline`/Phase 7 code is untouched by this — it only reused `imagehash`,
  never `edl.py`.

**3. GastroNet-5M — stream from S3 to LRZ by default, with a local shard
cache** (supersedes Phase 7's "stage once, `local_webdataset` default" — the
user wants the pipeline to actually *demonstrate* AWS→cluster streaming, which
is also the more interesting Data-Engineer portfolio artifact). Confirmed
technically live: LRZ H100 compute nodes reach `s3.eu-north-1.amazonaws.com`
directly (job `5772201`), and `webdataset==1.0.2` (already in the `pipeline`
extra) has native `cache_dir=`/`cache_size=` support — no new library needed.
- `config.yaml`: `ssl.gastronet_source` default flips to `s3_webdataset`.
  `local_webdataset` (pre-staged) and `local_zip` (raw shards) stay available
  as fallbacks — e.g. if a compute node is ever firewalled.
- `vlfz/data/gastronet.py::webdataset_loader`: for `s3_webdataset`, pass
  `cache_dir=$GASTRONET_ROOT/.wds_cache` (persists across job resubmits — it's
  on the login-node-visible home quota, not node-local scratch) and
  `cache_size=<a few hundred GB>` to `wds.WebDataset(...)`. First read of a
  shard downloads + caches it (one egress, ≈ the curated tier size, ≈ $5 total
  across all 4 runs' first epochs); every later epoch or resubmit reads the
  cache, not S3. `pipe:aws s3 cp` shelling-out is dropped in favour of wds's
  built-in HTTP(S) fetch (works directly against a virtual-hosted-style S3 URL,
  no `aws` CLI dependency inside the DataLoader worker).
  `AWS_PROFILE=gastronet-reader` must be exported (`job_env.sh` already does
  this) so boto3/the underlying HTTP client can sign requests — or, simpler,
  since `gastronet-lake-ro`'s scope already covers `GetObject`, a plain
  authenticated HTTPS GET works without the CLI.
- `pipeline/stage/stage_in.sh` / `select_subset.py` stay as-is — still the
  right tool for a deliberate one-time bulk copy (e.g. before a network
  outage, or to seed the cache in one shot instead of paying it out over
  epoch 1), just no longer the *default* code path.
- Cost is unchanged from Phase 7's table (~$5 one-time egress) — streaming +
  caching does not add per-epoch cost, it just changes *when* the fetch
  happens and demonstrates the pipeline moving data live rather than a bulk
  `sync` step.

**4. Training + inference move to PyTorch Lightning** (new; `lightning` is not
yet a dependency anywhere in the repo).
- **Scope**: SSL pretraining (`vlfz/ssl/pretrain.py`) **and** eval frozen-
  feature extraction (`vlfz/eval/features.py`). The k-NN / calibration /
  temperature-scaling math (`knn.py`, `calibration.py`, `metrics.py`) stays
  plain numpy/sklearn — there is no model training there, so Lightning
  wouldn't simplify it; it just consumes the feature `.npz` cache as today.
- **SSL**: `GastroNetLitModule(pl.LightningModule)` wraps the existing
  backbone + LeJEPA/DINO aux heads. `training_step` = today's loss branch
  (`lejepa_loss` / `dino_loss`); `configure_optimizers` returns the existing
  `llrd_param_groups` AdamW + a `LambdaLR` built from `cosine_lr`. The DINO
  EMA-teacher update + centering, and the LeJEPA EMA-backbone update, move to
  `on_train_batch_end` (Lightning does the optimizer step; these are pure
  post-step bookkeeping, same as today's manual loop). Grad-clip via
  `Trainer(gradient_clip_val=...)`; bf16 via `Trainer(precision="bf16-mixed")`;
  `set_grad_checkpointing(True)` unchanged (a backbone-level call, orthogonal
  to Lightning). Checkpoint/resume via `Trainer(ckpt_path=...)` +
  `ModelCheckpoint(every_n_train_steps=cfg.ssl.ckpt_every_steps)`; the existing
  `DONE` flag + self-resubmit `sbatch` pattern (`lrz/sbatch_ssl_pretrain.sbatch`)
  is unchanged — Lightning lives *inside* one job's `train()`, it doesn't
  replace the SLURM self-resubmit loop.
- **Data**: a `GastroNetDataModule(pl.LightningDataModule)` wraps
  `_make_loader`/`webdataset_loader` from decision 3 (`train_dataloader`); a
  sibling `HyperKvasirDataModule` wraps the eval transform + split for
  `predict_dataloader`. Same DataModule class family is reused by both
  `Trainer.fit()` (SSL) and `Trainer.predict()` (feature extraction), which is
  the "training/inference" unification the user asked for.
- **Eval**: `vlfz/eval/features.py`'s cached extractor becomes a thin
  `LightningModule` whose `predict_step` returns pooled features; call
  `Trainer(devices=1, precision="bf16-mixed").predict(module, datamodule)` and
  write the same `feat_<key>.npz` cache as today — `knn.py` downstream is
  untouched.
- **Deps**: add `lightning>=2.4,<3` to `pyproject.toml` main deps (not the
  `pipeline` extra — every training/eval job needs it). `lrz/setup_env.sh`
  picks it up via the existing `uv pip install -e ".[dev]"` once added.
- **Risk**: EMA-teacher-update-after-optimizer-step ordering and the
  DINO/LeJEPA dual-objective branching are the load-bearing, previously-tuned
  parts of `pretrain.py` (see "Errors and fixes" — DINOHead deepcopy, LeJEPA
  unpack bug, OOM). Port them into Lightning hooks **unchanged in substance**,
  not rewritten from scratch; add a smoke assertion that loss curves match the
  pre-migration run on the same 2000-image smoke set within noise, before
  trusting a real `--stage full` run under the new trainer.
- **Verify**: `tests/test_calibration.py`-style unit tests for the LR/momentum
  schedule math (already-tested pure functions, untouched); a new smoke —
  `STAGE=smoke` LeJEPA + DINO under `Trainer` — loss finite & decreasing,
  `ema_backbone.pt` written in the same format `run_eval.py` already loads
  (no eval-side changes needed if the checkpoint dict shape is preserved).

### Debugging status (2026-09-05, concluded for now)

Rewrote `vlfz/ssl/pretrain.py` to Lightning (`SSLModule` + `GastroNetDataModule`,
manual optimization) and `vlfz/eval/features.py` (`Trainer.predict()`); both
`py_compile` clean. `docs/PLAN.md` added to the repo (a copy of this file) per
the user's request to keep the plan under version control.

**Found and fixed**: `ModelCheckpoint(save_top_k=2, monitor=None)` is rejected
by Lightning 2.6 (`save_top_k` must be -1/0/1 without a monitored metric,
since this is unsupervised SSL with nothing to rank checkpoints by) →
`save_top_k=1` + `save_last=True`.

**Investigated at length, not conclusively root-caused**: a LeJEPA CPU smoke
(`--limit-steps 2`, `tests/smoke_config.yaml`, a fabricated tiny image set)
intermittently hangs with zero CPU activity for minutes, pinpointed via debug
print markers to different points on different runs — the same unmodified
code sometimes completes both steps cleanly end-to-end (checkpoints +
`ema_backbone.pt` written correctly), and sometimes stalls with no forward
progress until killed by an external timeout. Tried, in order, each verified
NOT to make it deterministic on its own:
1. `opt.step()` -> `opt.optimizer.step()` (bypass Lightning's optimizer-step
   dispatch) — completed once, then hung again on a later run.
2. `foreach=False` on the `AdamW` construction (batched/fused CPU kernels are
   a known hang source) — still hung, now inside the very same
   `opt.optimizer.step()` call.
3. Disabled `set_grad_checkpointing` on CPU (checkpointing's reentrant
   backward is a plausible destabiliser, and buys nothing on CPU anyway) —
   still hung, this time inside plain `manual_backward()` itself, i.e. before
   ever reaching the optimizer step.
4. `torch.set_num_threads(1)` for CPU-only runs (OpenMP busy-wait barriers are
   a classic shared-login-node hazard) — still hung, back at the optimizer
   step.

The hang point moves between runs with no code change, which rules out a
logic bug in this repo (a real bug reproduces at a fixed line) and is more
consistent with **contention on this specific shared LRZ login node** — the
same filesystem showed multi-minute variance all session for unrelated cold
imports/`uv pip install`s, and this node is explicitly not the intended
compute target (`README.md` already says DINO needs a GPU node, not a login
node; this investigation shows even a tiny LeJEPA CPU smoke isn't reliable
there either). All four mitigations above were kept (harmless, individually
justified) as defense-in-depth, but none is claimed to be *the* fix. Debug
print markers were removed; the code is otherwise unchanged from the design
in the "4. Training + inference move to PyTorch Lightning" section above.

**Not yet done**: a clean run on an actual GPU compute node (`sbatch
tests/smoke_ssl.sh`-equivalent, or `lrz/sbatch_ssl_pretrain.sbatch`), which is
the real target environment and was never the thing hanging — only the
login-node CPU smoke was. That is the next concrete verification step once
the user green-lights it (an sbatch submission, not something to run silently
in the background).


## Verification

- **Smoke mode** for every script (tiny cap, 1 epoch/pass); `tests/` unit tests
  for backbone parity, calibration correctness (vs torchmetrics), k-NN prob
  validity.
- **Sanity anchor**: ImageNet ViT-B/16 frozen k-NN on T1 should beat chance
  clearly; SigLIP-2 frozen should beat ImageNet on both tasks pre-SSL.
- **Leakage report** generated and inspected before results are final;
  `provenance()` embedded in every JSON.
- **End-to-end**: Phase 4 produces the pre-SSL table; Phase 6 produces the full
  16-row table + delta view with finite pre/post-temperature ECE and NLL for all
  4 settings on both tasks.

### Phase 7 (cloud pipeline)

- **Scope gate first**: `aws sts get-caller-identity --profile gastronet-reader`
  then `aws s3 ls s3://<bucket>/` — confirm the reader key is bucket-scoped and
  cannot write, before anything else runs.
- **IaC**: `terraform plan` reviewed and committed as evidence before
  `terraform apply`; `terraform destroy` returns to zero standing cost.
- **Ingest**: `ingest_log/` shows 506 objects, each with a matching sha256; the
  EC2 box is `terminated` afterwards (check the console / `aws ec2
  describe-instances`).
- **Catalog**: Athena `SELECT count(*), sum(bytes), count(DISTINCT phash) FROM
  gastronet_manifest` → rows ≈ 4.82M, byte total ≈ portal size, dup ratio logged.
- **Curation**: `webdataset/` shard count × images-per-shard ≈ deduped total;
  open one `.tar`, decode 10 members, assert RGB / 224².
- **DQ report**: `dq/data_report.md` generated and eyeballed — corrupt rate
  < 0.1%, dup clusters enumerated, dimension histogram sane.
- **Stage-in**: `stage_in.sh` sha256-verifies every shard; re-running is a no-op.
- **Dataloader**: `python -m vlfz.data.gastronet --source local_webdataset
  --smoke` iterates ≥100 batches with finite tensors; `--source s3_webdataset`
  iterates ≥5 batches (only if job `5772201` shows compute-node HTTPS egress).
- **End-to-end**: `STAGE=smoke … --corpus gastronet --source local_webdataset`
  SSL smoke on the staged tier → loss finite; then one real `--stage full` job;
  then the existing eval chain (Phase 6) runs unchanged on the new checkpoints.
- **Cost**: AWS Budgets alarm armed; Cost Explorer after week 1 shows egress
  ≈ one-time stage-in only (no per-epoch S3 GET spikes).

---

## Execution status — 2026-09-06

**Phase 4 (before-SSL eval): DONE.** Both siglip2 + imagenet frozen ViT-B/16,
zero-shot weighted k-NN on all 4 HyperKvasir classification tasks + zero-shot
polyp segmentation. First real report at `$OUT_ROOT/results/report.md`:

| task | imagenet acc / AUROC | siglip2 acc / AUROC |
|---|---|---|
| hkv_tract     | 0.995 / 0.999 ⚠ | 0.987 / 0.998 ⚠ |
| hkv_pathology | 0.942 / 0.983    | 0.908 / 0.957   |
| hkv_category  | 0.940 / 0.992    | 0.897 / 0.980   |
| hkv_findings (20-way) | 0.846 / 0.947 | 0.763 / 0.923 |
| hkv_seg       | Dice 0.667      | Dice 0.314      |

ImageNet init beats SigLIP-2 on every HyperKvasir task here — the SSL runs test
whether in-domain pretraining changes that and (the actual question) the
calibration.

**Phase 5 (SSL on GastroNet-5M): in progress.**
- SSL code migrated to PyTorch Lightning, validated end-to-end on an A100
  (all 4 settings, checkpoints + `ema_backbone.pt` in the shape `run_eval`
  reads). The intermittent CPU hang chased earlier is login-node-only.
- EDL dropped from the benchmark (one temperature-scaled `knn` protocol).
- `hkv_findings` label-alignment bug fixed (`canonical_classes`, global not
  per-split) — 2.9% → 84.6%.
- AWS data lake fully provisioned (bucket / IAM / Glue+Athena / cost budget /
  ephemeral spot EC2). Read-only `gastronet-lake-ro` policy attached to the
  LRZ `read-only-agent` user (inline — the 10-managed-policy quota was hit).
- Cortex portal ingest reverse-engineered + implemented: per-file
  `POST /api/provided_file/<id>/download_url/` → 10-min presigned URL on
  `s3.thetavision.nl` → streamed to `raw/` with Range-resume. `portal.json`
  (session cookie + 506-file list) lives at `s3://<bucket>/bootstrap/`.
- **Running now:** `pipeline/kick_ingest.sh 60` — a 60-zip subset (~250 GB,
  ~800k–1M images; full corpus is 1.88 TB / 506 zips) ingesting on the spot
  EC2, then catalog → curate → dq. The spot instance was reclaimed once and
  recreated (`i-091679f565fb6d44d`).

**Next:** when `dq/data_report.md` appears, stage or stream the curated tier and
`bash lrz/submit_ssl_matrix.sh` (4 self-resubmitting runs). Then Phase 6:
`submit_eval_chain.sh` picks up the `ema_backbone.pt`s and produces the
16-row pre-vs-post delta table.

### 2026-09-07 — Phase 5 launched

EC2 ingest abandoned (instances vanish ~20 min post-create in this account,
spot and on-demand alike). Pivoted to `pipeline/ingest/portal_to_wds.py` on an
LRZ login node: portal presign -> stream zip -> curate (224 / phash-dedup /
JPEG / WebDataset tar) -> delete zip. 60-zip subset done in 155 min ->
**581,518 images, 60 shards, 8.1 GB** at
`$MCMLSCRATCH/gastronet5m/webdataset/`. Downloads held ~90 MB/s to Hetzner.

`gastronet_source` default reverted to `local_webdataset`; `GASTRONET_ROOT`
default -> `$MCMLSCRATCH/gastronet5m`. `ssl.full.epochs` env-overridable
(`SSL_EPOCHS`). `sbatch_ssl_pretrain.sbatch` runs the venv python directly (no
`srun` — it dropped the sbatch env on this cluster).

**SSL matrix queued** (jobs 5775945-48): LeJEPA/DINO x siglip2/imagenet,
`STAGE=full SSL_EPOCHS=3`, self-resubmitting. ~4500 steps/epoch (581k / bs128).
Then Phase 6: `submit_eval_chain.sh` picks up the 4 `ema_backbone.pt`s ->
pre-vs-post calibration delta.

AWS lake infra stays built + committed (Terraform, ingest/catalog/curate/dq,
kick_ingest.sh) as the data-engineering artefact; this run just doesn't route
through it. The ephemeral EC2 should be torn down
(`terraform destroy -target=aws_instance.ingest`).
