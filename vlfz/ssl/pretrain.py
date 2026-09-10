"""Unified SSL pretraining entrypoint — LeJEPA or DINO v1, from a SigLIP-2 or
ImageNet ViT-B/16 init, full-backbone (``--lora`` is a fallback), single-GPU.

Built on **PyTorch Lightning**: ``SSLModule`` is a ``LightningModule`` (manual
optimization — the dual LeJEPA/DINO objective and EMA-teacher bookkeeping are
ported unchanged from the original hand-rolled loop, just inside Lightning
hooks); ``GastroNetDataModule`` wraps ``_make_loader`` below. Checkpoint/resume
is Lightning's own (``ModelCheckpoint`` + ``ckpt_path=``) — every trainable
piece (backbone, aux heads, EMA backbone, DINO teacher, the centering buffer)
is a registered submodule/buffer, so it round-trips for free. The one file
downstream code depends on, ``ema_backbone.pt``, keeps its exact pre-Lightning
shape (see ``SSLModule.save_eval_backbone``).

    STAGE=smoke python -m vlfz.ssl.pretrain --objective lejepa --init siglip2 --corpus hkv_unlabeled
    STAGE=full  python -m vlfz.ssl.pretrain --objective dino   --init imagenet --corpus gastronet
"""
from __future__ import annotations

import argparse
import os

import lightning.pytorch as pl
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint

from ..cfg import ensure_dirs, load_cfg, provenance, set_seed
from ..models.ema import EMA, cosine_momentum
from ..models.heads import DINOHead, Predictor, Projector
from ..models.vit_backbone import build_vit_b16
from .dino_lib import cosine_lr, dino_loss, teacher_temp_at, update_center
from .lejepa_lib import effective_rank, lejepa_loss

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


# ---------------------------------------------------------------- data
class _PathListDataset:
    def __init__(self, paths, transform):
        self.paths, self.t = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.t(img)


def _make_loader(cfg, args, transform, collate, *, bs, nw, dev):
    """-> (dataloader, steps_per_epoch). Map-style corpora build a normal
    DataLoader; the curated WebDataset tier (local or streamed from S3, see
    vlfz/data/gastronet.py) builds an iterable one with a fixed epoch length."""
    smoke_n = int(args.max_images or cfg.ssl.smoke.max_images)

    def _plain(ds):
        dl = torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=True, num_workers=nw,
            pin_memory=(dev == "cuda"), drop_last=True, collate_fn=collate,
            persistent_workers=(nw > 0))
        return dl, max(1, len(dl))

    if args.corpus == "hkv_unlabeled":
        from ..data.hyperkvasir import unlabeled_paths

        paths = unlabeled_paths(cfg, cap=(smoke_n if args.stage == "smoke" else 0))
        if not paths:
            raise SystemExit(
                "corpus=hkv_unlabeled but no images under $HKV_ROOT/hyper_kvasir_unlabeled_images "
                "(the pool was removed). Use --corpus gastronet after pipeline/stage/stage_in.sh.")
        return _plain(_PathListDataset(paths, transform))

    from ..data.gastronet import resolve_source, webdataset_loader, webdataset_shards

    source = resolve_source(cfg, args.source)
    if source == "local_zip":
        from ..data.gastronet import GastroNetDataset

        sub = smoke_n if args.stage == "smoke" else int(cfg.ssl.subset_images)
        return _plain(GastroNetDataset(cfg, transform, subset=sub, seed=int(cfg.seed)))

    # WebDataset (local or streamed from the S3 lake): iterable -> fix the
    # epoch length explicitly.
    per_shard = int(getattr(cfg.ssl, "wds_images_per_shard", 10000))
    n_shards = len(webdataset_shards(cfg, source))
    if args.stage == "smoke":
        steps = max(1, smoke_n // bs)
    elif int(getattr(cfg.ssl, "steps_per_epoch", 0)) > 0:
        steps = int(cfg.ssl.steps_per_epoch)
    else:
        steps = max(1, (n_shards * per_shard) // bs)
    dl = webdataset_loader(cfg, transform, collate, source=source, batch_size=bs,
                           num_workers=nw, steps_per_epoch=steps, seed=int(cfg.seed))
    print(f"[pretrain] {source}: {n_shards} shards ~{n_shards * per_shard} imgs "
          f"-> {steps} steps/epoch @ bs {bs}")
    return dl, steps


class GastroNetDataModule(pl.LightningDataModule):
    """Wraps ``_make_loader`` so the same corpus-resolution logic serves both
    ``Trainer.fit`` here and, via the sibling loader in ``eval/features.py``,
    ``Trainer.predict`` for frozen-feature extraction. Memoised: the S3 shard
    listing / presign (for ``s3_webdataset``) happens at most once per job."""

    def __init__(self, cfg, args, transform, collate, *, bs, nw, dev):
        super().__init__()
        self.cfg, self.args, self.transform, self.collate = cfg, args, transform, collate
        self.bs, self.nw, self.dev = bs, nw, dev
        self._dl = None
        self.steps_per_epoch = None

    def setup(self, stage=None):
        if self._dl is None:
            self._dl, self.steps_per_epoch = _make_loader(
                self.cfg, self.args, self.transform, self.collate,
                bs=self.bs, nw=self.nw, dev=self.dev)

    def train_dataloader(self):
        self.setup()
        return self._dl


# ---------------------------------------------------------------- optim
def llrd_param_groups(backbone, other_params, base_lr, decay):
    blocks = backbone.trunk.blocks
    depth = len(blocks)
    groups, seen = [], set()

    def add(params, scale):
        ps = [p for p in params if p.requires_grad and id(p) not in seen]
        for p in ps:
            seen.add(id(p))
        if ps:
            groups.append({"params": ps, "lr": base_lr * scale})

    add(other_params, 1.0)
    add([p for n, p in backbone.trunk.named_parameters()
         if n.startswith(("patch_embed", "pos_embed", "cls_token", "reg_token"))],
        decay ** (depth + 1))
    for i, blk in enumerate(blocks):
        add(list(blk.parameters()), decay ** (depth - i))
    add([p for n, p in backbone.trunk.named_parameters()
         if n.startswith(("norm", "fc_norm", "head", "attn_pool"))], 1.0)
    add(backbone.trunk.parameters(), decay ** (depth + 1))  # any stragglers
    return groups


def _maybe_lora(backbone):
    try:
        from peft import LoraConfig, get_peft_model

        backbone.trunk = get_peft_model(
            backbone.trunk,
            LoraConfig(r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
                       target_modules=["qkv", "proj", "fc1", "fc2"]),
        )
        print("[pretrain] LoRA adapters on backbone")
    except Exception as e:  # noqa: BLE001
        print(f"[pretrain] LoRA requested but unavailable ({e}); full-backbone")
    return backbone


# ---------------------------------------------------------------- module
class SSLModule(pl.LightningModule):
    def __init__(self, cfg, args, out_dir, total_steps):
        super().__init__()
        self.cfg, self.args, self.out_dir = cfg, args, out_dir
        self.total_steps = total_steps
        self.automatic_optimization = False  # LLRD groups + EMA/teacher updates need manual control
        self._limit_hit = False

        backbone = build_vit_b16(args.init, cfg, pretrained_init=True)
        # grad checkpointing trades compute for GPU memory; it's pointless on
        # CPU (no memory pressure at these batch sizes) and its reentrant
        # backward is the prime suspect for an intermittent CPU hang inside
        # the optimizer step right after (reproduced repeatedly; foreach=False
        # alone did not fix it -- see docs/PLAN.md debugging notes).
        if torch.cuda.is_available():
            backbone.set_grad_checkpointing(True)
        if args.lora:
            backbone = _maybe_lora(backbone)
        self.backbone = backbone
        d = backbone.embed_dim

        if args.objective == "lejepa":
            lp = cfg.ssl.lejepa
            self.proj = Projector(d, int(lp.proj_dim))
            self.pred = Predictor(int(lp.proj_dim), int(lp.pred_hidden_mult))
            self.aux = nn.ModuleList([self.proj, self.pred])
            self.teacher, self.teacher_net = None, None
            ema_bb = EMA(self.backbone, base_decay=float(cfg.ssl.ema_base))
            self.ema_bb, self.ema_bb_net = ema_bb, ema_bb.ema  # registers as a submodule
        else:
            dp = cfg.ssl.dino
            self.head = DINOHead(d, int(dp.out_dim))
            self.aux = nn.ModuleList([self.head])
            student = nn.ModuleDict({"bb": self.backbone, "head": self.head})
            teacher = EMA(student, base_decay=0.996)
            self.teacher, self.teacher_net = teacher, teacher.ema  # registers as a submodule
            self.ema_bb, self.ema_bb_net = None, None
            self.register_buffer("center", torch.zeros(int(dp.out_dim)))

        self.other_params = [p for m in self.aux for p in m.parameters()]

    # -- optim --------------------------------------------------------
    def configure_optimizers(self):
        cfg = self.cfg
        groups = llrd_param_groups(self.backbone, self.other_params,
                                   float(cfg.ssl.base_lr), float(cfg.ssl.llrd))
        # foreach=False only on CPU: the batched foreach/fused AdamW kernels were
        # seen to intermittently hang on the shared login node right after a
        # backward pass (non-deterministic). On a GPU the batched kernels are a
        # real speedup and the hang was never reproduced there, so keep the
        # default (foreach=None -> batched).
        fe = False if not torch.cuda.is_available() else None
        opt = torch.optim.AdamW(groups, lr=float(cfg.ssl.base_lr),
                                weight_decay=float(cfg.ssl.weight_decay), foreach=fe)
        self._base_lrs = [g["lr"] for g in opt.param_groups]
        return opt

    def _set_lr(self, opt) -> float:
        cfg = self.cfg
        warmup = int(cfg.ssl.warmup_frac * self.total_steps)
        lr = cosine_lr(self.global_step, self.total_steps, 1.0,
                       float(cfg.ssl.min_lr) / float(cfg.ssl.base_lr), warmup)
        for g, b in zip(opt.param_groups, self._base_lrs):
            g["lr"] = b * lr
        return opt.param_groups[0]["lr"]

    # -- step -----------------------------------------------------------
    def training_step(self, batch, batch_idx):
        cfg, opt = self.cfg, self.optimizers()
        cur_lr = self._set_lr(opt)
        gstep = self.global_step

        if self.args.objective == "lejepa":
            v1, v2 = batch[0], batch[1]
            z1, z2 = self.proj(self.backbone.feature(v1)), self.proj(self.backbone.feature(v2))
            loss, parts = lejepa_loss(
                z1, z2, self.pred,
                sigreg_lambda=float(cfg.ssl.lejepa.sigreg_lambda),
                var_weight=float(cfg.ssl.lejepa.var_weight),
                n_slices=int(cfg.ssl.lejepa.sigreg_slices),
                n_freq=int(cfg.ssl.lejepa.sigreg_freqs),
            )
            t_out = None
        else:
            dp = cfg.ssl.dino
            crops = batch
            t_temp = teacher_temp_at(gstep, self.total_steps, float(dp.teacher_temp),
                                     float(dp.teacher_temp_final),
                                     float(dp.teacher_temp_warmup_frac))
            s_out = [self.head(self.backbone.feature(c)) for c in crops]
            with torch.no_grad():
                t_out = [self.teacher_net["head"](self.teacher_net["bb"].feature(crops[0])),
                         self.teacher_net["head"](self.teacher_net["bb"].feature(crops[1]))]
            loss = dino_loss(s_out, t_out, self.center, float(dp.student_temp), t_temp)
            parts = {"dino": float(loss.detach())}

        if not torch.isfinite(loss):
            print(f"  [skip] non-finite loss @ step {gstep}")
            opt.zero_grad(set_to_none=True)
            return None

        self.manual_backward(loss)
        self.clip_gradients(opt, gradient_clip_val=float(cfg.ssl.grad_clip),
                            gradient_clip_algorithm="norm")
        # GPU (the real training target): the normal LightningOptimizer step, so
        # trainer.global_step advances. The LR / teacher-temp / EMA-momentum
        # schedules AND ModelCheckpoint(every_n_train_steps) all key off
        # global_step; the raw opt.optimizer.step() below leaves it pinned at 0
        # -> LR frozen at 0 -> the model never trains and no checkpoint fires.
        # Keep the raw-step bypass ONLY for CPU login-node smokes, where an
        # intermittent post-backward hang in the LightningOptimizer step path
        # was seen (docs/PLAN.md) and never reproduced on GPU; foreach=False
        # (configure_optimizers) + single-thread (train()) are the other
        # CPU-only guards.
        if torch.cuda.is_available():
            opt.step()
        else:
            opt.optimizer.step()
        opt.zero_grad(set_to_none=True)

        if self.ema_bb is not None:
            self.ema_bb.update(self.backbone)
        if self.teacher is not None:
            m = cosine_momentum(gstep, self.total_steps, 0.996)
            student = nn.ModuleDict({"bb": self.backbone, "head": self.head})
            self.teacher.update(student, decay=m)
            update_center(self.center, [o.detach() for o in t_out], float(cfg.ssl.dino.center_momentum))

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        for k, v in parts.items():
            self.log(f"loss/{k}", v, on_step=True, on_epoch=False)
        self.log("lr", cur_lr, on_step=True, on_epoch=False)

        if (gstep + 1) % int(cfg.ssl.ckpt_every_steps) == 0:
            self.save_eval_backbone()
        if self.args.limit_steps and (gstep + 1) >= self.args.limit_steps:
            print(f"[pretrain] --limit-steps {self.args.limit_steps} reached")
            self._limit_hit = True
            self.save_eval_backbone()
            self.trainer.should_stop = True

        return loss

    def on_train_epoch_end(self):
        try:
            self.backbone.eval()
            with torch.no_grad():
                probe = next(iter(self.trainer.train_dataloader))
                v = probe[0][:64].to(self.device)
                er = effective_rank(self.backbone.feature(v))
            print(f"epoch {self.current_epoch + 1} done  eff_rank {er:.1f}")
            self.log("collapse/effective_rank", er, on_epoch=True)
        except Exception as e:  # noqa: BLE001
            print(f"epoch {self.current_epoch + 1} done  (eff_rank skipped: {e})")
        finally:
            self.backbone.train()
        self.save_eval_backbone()

    # -- the one artefact downstream eval reads; format unchanged pre-Lightning
    def save_eval_backbone(self):
        trunk = (self.teacher_net["bb"].trunk if self.teacher is not None
                 else self.ema_bb_net.trunk)
        torch.save(
            {"ema_backbone": trunk.state_dict(), "init": self.args.init,
             "objective": self.args.objective, "corpus": self.args.corpus,
             "gstep": self.global_step,
             "provenance": provenance(int(self.cfg.seed), objective=self.args.objective,
                                      init=self.args.init, corpus=self.args.corpus)},
            os.path.join(self.out_dir, "ema_backbone.pt"),
        )


# ---------------------------------------------------------------- train
def train(cfg, args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        # Shared LRZ login nodes: PyTorch's intraop thread pool defaults to
        # nproc, and its busy-wait barrier can hang for minutes when those
        # threads are contended with other users' processes (reproduced:
        # intermittent hangs inside plain backward()/optimizer.step(), not
        # specific to grad checkpointing -- see docs/PLAN.md). CPU-only smoke
        # runs a handful of tiny images; single-threaded is plenty and removes
        # the hazard entirely. Real training runs on a dedicated GPU node.
        torch.set_num_threads(1)
    set_seed(int(cfg.seed))
    st = cfg.ssl.smoke if args.stage == "smoke" else cfg.ssl.full
    epochs = int(st.epochs)
    bs = int(args.bs or getattr(st, "batch_size", cfg.ssl.batch_size))
    nw = int(args.nw if args.nw is not None else getattr(st, "num_workers", cfg.ssl.num_workers))
    if args.stage == "smoke":  # keep the login-node/CI footprint tiny
        from omegaconf import OmegaConf, open_dict

        OmegaConf.set_struct(cfg, False)
        with open_dict(cfg):
            cfg.ssl.dino.n_local = min(int(cfg.ssl.dino.n_local), int(args.dino_n_local))
            cfg.ssl.dino.out_dim = min(int(cfg.ssl.dino.out_dim), int(args.dino_out_dim))

    out_dir = args.out or os.path.join(
        os.path.expanduser(str(cfg.paths.ckpts)),
        f"{args.objective}_{args.init}_{args.corpus}_{args.stage}",
    )
    ensure_dirs(out_dir)
    done_flag = os.path.join(out_dir, "DONE")
    if os.path.exists(done_flag) and not args.fresh:
        print(f"[pretrain] {done_flag} exists -> nothing to do")
        return out_dir

    # transforms + data
    from ..data.transforms import multicrop_collate

    if args.objective == "lejepa":
        from ..data.transforms import two_view_transform

        transform, collate = two_view_transform(cfg), multicrop_collate
    else:
        from ..data.transforms import MultiCropTransform

        transform, collate = MultiCropTransform(cfg), multicrop_collate

    dm = GastroNetDataModule(cfg, args, transform, collate, bs=bs, nw=nw, dev=dev)
    dm.setup()
    steps_per_epoch = dm.steps_per_epoch
    total_steps = epochs * steps_per_epoch
    print(f"[pretrain] {args.objective}/{args.init}/{args.stage}  "
          f"bs {bs} | {epochs} ep x {steps_per_epoch} = {total_steps} steps")

    model = SSLModule(cfg, args, out_dir, total_steps)

    last_ckpt = os.path.join(out_dir, "last.ckpt")
    resume_from = last_ckpt if (os.path.exists(last_ckpt) and args.resume and not args.fresh) else None
    if resume_from:
        print(f"[pretrain] RESUME <- {resume_from}")

    # no metric to monitor (unsupervised SSL) -> save_top_k is limited to
    # {-1, 0, 1} by Lightning; 1 + save_last keeps just the latest periodic
    # snapshot plus last.ckpt (the resume pointer) resident, as intended.
    ckpt_cb = ModelCheckpoint(dirpath=out_dir, filename="ckpt-{step}",
                              every_n_train_steps=int(cfg.ssl.ckpt_every_steps),
                              save_last=True, save_top_k=1)
    trainer = pl.Trainer(
        accelerator=("gpu" if dev == "cuda" else "cpu"), devices=1,
        precision=("bf16-mixed" if dev == "cuda" else 32),
        # bound on absolute global_step, not epochs: on resume from last.ckpt
        # Lightning restores global_step but its iterable-dataloader epoch counter
        # resets, so `max_epochs` would run a FRESH `epochs` epochs after a resume
        # (a run that TIMEOUT'd at 1 epoch then resumed did ~4 epochs total).
        # max_steps=total_steps makes "3 epochs" mean 3*steps_per_epoch of training
        # regardless of how many allocations it took.
        max_epochs=-1, max_steps=(args.limit_steps or total_steps),
        default_root_dir=out_dir, callbacks=[ckpt_cb],
        logger=pl.loggers.TensorBoardLogger(out_dir, name="tb"),
        enable_progress_bar=True, log_every_n_steps=50,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=dm, ckpt_path=resume_from)

    if not model._limit_hit:
        open(done_flag, "w").write(__import__("time").strftime("%Y-%m-%d %H:%M:%S\n"))
        print(f"[pretrain] DONE -> {out_dir}/ema_backbone.pt")
        # Reclaim the resume/periodic checkpoints (~1.4 GB each) now that the run
        # is finished -- eval only ever loads ema_backbone.pt. Four runs each
        # leaving last.ckpt + ckpt-step=*.ckpt filled the home quota and killed
        # a resume mid-checkpoint (OSError 28). Keep ema_backbone.pt + tb/ + DONE.
        import glob as _glob

        for _f in [os.path.join(out_dir, "last.ckpt"), *_glob.glob(os.path.join(out_dir, "ckpt-*.ckpt"))]:
            try:
                os.remove(_f)
                print(f"[pretrain] cleaned {os.path.basename(_f)}")
            except OSError:
                pass
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--objective", choices=["lejepa", "dino"],
                    default=os.environ.get("OBJ", "lejepa"))
    ap.add_argument("--init", choices=["siglip2", "imagenet"],
                    default=os.environ.get("INIT", "siglip2"))
    ap.add_argument("--stage", choices=["smoke", "full"],
                    default=os.environ.get("STAGE", "smoke"))
    ap.add_argument("--corpus", choices=["gastronet", "hkv_unlabeled"],
                    default=os.environ.get("CORPUS", "gastronet"))
    ap.add_argument("--source", default=os.environ.get("GASTRONET_SOURCE"),
                    choices=["local_zip", "local_webdataset", "s3_webdataset"],
                    help="gastronet read path (default: cfg.ssl.gastronet_source)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--limit-steps", type=int, default=0, help="stop early (smoke)")
    ap.add_argument("--bs", type=int, default=0, help="override batch size")
    ap.add_argument("--nw", type=int, default=None, help="override num_workers")
    ap.add_argument("--max-images", type=int, default=0, help="override smoke image cap")
    ap.add_argument("--dino-n-local", type=int, default=2, help="smoke: cap DINO local crops")
    ap.add_argument("--dino-out-dim", type=int, default=1024, help="smoke: cap DINO head dim")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    if args.corpus == "gastronet":
        from ..data.gastronet import resolve_source, shard_paths, webdataset_shards

        src = resolve_source(cfg, args.source)
        ok = (bool(shard_paths(os.path.expanduser(str(cfg.paths.gastronet))))
              if src == "local_zip" else True)
        if src != "local_zip":
            try:
                ok = bool(webdataset_shards(cfg, src))
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"[pretrain] {src} shard listing failed: {e}")
        if not ok:
            raise SystemExit(
                f"[pretrain] corpus=gastronet source={src} has no data. "
                "Stage it first: pipeline/stage/stage_in.sh (see pipeline/README.md).")
    train(cfg, args)


if __name__ == "__main__":
    main()
