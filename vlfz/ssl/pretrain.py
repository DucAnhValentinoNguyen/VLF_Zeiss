"""Unified SSL pretraining entrypoint — LeJEPA or DINO v1, from a SigLIP-2 or
ImageNet ViT-B/16 init, full-backbone (``--lora`` is a fallback), single-GPU with
self-resubmitting checkpoint/resume.

    STAGE=smoke python -m vlfz.ssl.pretrain --objective lejepa --init siglip2 --corpus hkv_unlabeled
    STAGE=full  python -m vlfz.ssl.pretrain --objective dino   --init imagenet --corpus gastronet
"""
from __future__ import annotations

import argparse
import math
import os
import time
from glob import glob

import torch
import torch.nn as nn

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


def _make_dataset(cfg, args, transform):
    smoke_n = int(args.max_images or cfg.ssl.smoke.max_images)
    n = smoke_n if args.stage == "smoke" else 0
    if args.corpus == "hkv_unlabeled":
        from ..data.hyperkvasir import unlabeled_paths

        return _PathListDataset(unlabeled_paths(cfg, cap=n), transform)
    from ..data.gastronet import GastroNetDataset

    sub = smoke_n if args.stage == "smoke" else int(cfg.ssl.subset_images)
    return GastroNetDataset(cfg, transform, subset=sub, seed=int(cfg.seed))


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


# ---------------------------------------------------------------- train
def train(cfg, args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
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
    ds = _make_dataset(cfg, args, transform)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=(dev == "cuda"),
        drop_last=True, collate_fn=collate, persistent_workers=(nw > 0),
    )
    steps_per_epoch = max(1, len(dl))
    total_steps = epochs * steps_per_epoch
    warmup = int(cfg.ssl.warmup_frac * total_steps)
    print(f"[pretrain] {args.objective}/{args.init}/{args.stage}  "
          f"{len(ds)} imgs | bs {bs} | {epochs} ep x {steps_per_epoch} = {total_steps} steps")

    # model
    backbone = build_vit_b16(args.init, cfg, pretrained_init=True).to(dev)
    backbone.set_grad_checkpointing(True)
    if args.lora:
        backbone = _maybe_lora(backbone).to(dev)
    d = backbone.embed_dim

    if args.objective == "lejepa":
        lp = cfg.ssl.lejepa
        proj = Projector(d, int(lp.proj_dim)).to(dev)
        pred = Predictor(int(lp.proj_dim), int(lp.pred_hidden_mult)).to(dev)
        aux = nn.ModuleList([proj, pred])
        teacher = center = None
    else:
        dp = cfg.ssl.dino
        head = DINOHead(d, int(dp.out_dim)).to(dev)
        student = nn.ModuleDict({"bb": backbone, "head": head})
        teacher = EMA(student, base_decay=0.996).to(dev)
        center = torch.zeros(int(cfg.ssl.dino.out_dim), device=dev)
        aux = nn.ModuleList([head])

    # LeJEPA: EMA of the backbone is the eval artefact.
    # DINO: the teacher backbone already IS that EMA -> no extra copy.
    ema_bb = EMA(backbone, base_decay=float(cfg.ssl.ema_base)).to(dev) if teacher is None else None

    other = [p for m in aux for p in m.parameters()]
    opt = torch.optim.AdamW(
        llrd_param_groups(backbone, other, float(cfg.ssl.base_lr), float(cfg.ssl.llrd)),
        lr=float(cfg.ssl.base_lr), weight_decay=float(cfg.ssl.weight_decay),
    )
    base_lrs = [g["lr"] for g in opt.param_groups]

    # tensorboard
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(os.path.join(out_dir, "tb"))
    except Exception:
        tb = None

    # resume
    state_path = os.path.join(out_dir, "train_state.pt")
    gstep, start_ep = 0, 0
    if os.path.exists(state_path) and args.resume and not args.fresh:
        ck = torch.load(state_path, map_location=dev)
        backbone.load_state_dict(ck["backbone"])
        for m, s in zip(aux, ck["aux"]):
            m.load_state_dict(s)
        opt.load_state_dict(ck["opt"])
        if ema_bb is not None and ck.get("ema_bb") is not None:
            ema_bb.ema.load_state_dict(ck["ema_bb"])
        if teacher is not None:
            teacher.ema.load_state_dict(ck["teacher"])
            center = ck["center"].to(dev)
        gstep, start_ep = int(ck["gstep"]), int(ck["epoch_done"])
        print(f"[pretrain] RESUME @ epoch {start_ep} step {gstep}")

    def save_state(epoch_done):
        torch.save(
            {"backbone": backbone.state_dict(),
             "aux": [m.state_dict() for m in aux],
             "opt": opt.state_dict(),
             "ema_bb": (ema_bb.ema.state_dict() if ema_bb is not None else None),
             "teacher": (teacher.ema.state_dict() if teacher is not None else None),
             "center": (center.detach().cpu() if center is not None else None),
             "gstep": gstep, "epoch_done": epoch_done,
             "init": args.init, "objective": args.objective, "corpus": args.corpus},
            state_path,
        )

    def save_eval_backbone():
        trunk = (teacher.ema["bb"].trunk if teacher is not None else ema_bb.ema.trunk)
        torch.save(
            {"ema_backbone": trunk.state_dict(), "init": args.init,
             "objective": args.objective, "corpus": args.corpus, "gstep": gstep,
             "provenance": provenance(int(cfg.seed), objective=args.objective,
                                      init=args.init, corpus=args.corpus)},
            os.path.join(out_dir, "ema_backbone.pt"),
        )

    dp = cfg.ssl.dino
    for ep in range(start_ep, epochs):
        backbone.train()
        [m.train() for m in aux]
        t0, running = time.time(), 0.0
        for it, batch in enumerate(dl):
            lr = cosine_lr(gstep, total_steps, 1.0, float(cfg.ssl.min_lr) / float(cfg.ssl.base_lr), warmup)
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = b * lr

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                if args.objective == "lejepa":
                    v1, v2 = batch[0].to(dev, non_blocking=True), batch[1].to(dev, non_blocking=True)
                    z1, z2 = aux[0](backbone.feature(v1)), aux[0](backbone.feature(v2))
                    loss, parts = lejepa_loss(
                        z1, z2, aux[1],
                        sigreg_lambda=float(cfg.ssl.lejepa.sigreg_lambda),
                        var_weight=float(cfg.ssl.lejepa.var_weight),
                        n_slices=int(cfg.ssl.lejepa.sigreg_slices),
                        n_freq=int(cfg.ssl.lejepa.sigreg_freqs),
                    )
                else:
                    crops = [c.to(dev, non_blocking=True) for c in batch]
                    t_temp = teacher_temp_at(gstep, total_steps, float(dp.teacher_temp),
                                             float(dp.teacher_temp_final),
                                             float(dp.teacher_temp_warmup_frac))
                    s_out = [aux[0](backbone.feature(c)) for c in crops]
                    with torch.no_grad():
                        t_out = [teacher.ema["head"](teacher.ema["bb"].feature(crops[0])),
                                 teacher.ema["head"](teacher.ema["bb"].feature(crops[1]))]
                    loss = dino_loss(s_out, t_out, center, float(dp.student_temp), t_temp)
                    parts = {"dino": float(loss.detach())}

            if not torch.isfinite(loss):
                print(f"  [skip] non-finite loss @ step {gstep}")
                opt.zero_grad(set_to_none=True)
                gstep += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in backbone.parameters() if p.grad is not None] + other,
                float(cfg.ssl.grad_clip),
            )
            opt.step()
            opt.zero_grad(set_to_none=True)

            if ema_bb is not None:
                ema_bb.update(backbone)
            if teacher is not None:
                m = cosine_momentum(gstep, total_steps, 0.996)
                teacher.update(student, decay=m)
                update_center(center, [o.detach() for o in t_out], float(dp.center_momentum))

            running += float(loss.detach())
            gstep += 1
            if args.limit_steps and gstep >= args.limit_steps:
                print(f"[pretrain] --limit-steps {args.limit_steps} reached")
                save_state(ep)
                save_eval_backbone()
                if tb:
                    tb.flush()
                return out_dir
            if gstep % 50 == 0:
                msg = f"  ep{ep+1} step {gstep}/{total_steps} loss {running/(it+1):.4f} lr {opt.param_groups[0]['lr']:.2e}"
                print(msg + " | " + " ".join(f"{k}={v:.3f}" for k, v in parts.items()))
                if tb:
                    for k, v in parts.items():
                        tb.add_scalar(f"loss/{k}", v, gstep)
                    tb.add_scalar("lr", opt.param_groups[0]["lr"], gstep)
            if gstep % int(cfg.ssl.ckpt_every_steps) == 0:
                save_state(ep)
                save_eval_backbone()

        # end epoch: collapse proxy on one batch
        try:
            backbone.eval()
            with torch.no_grad():
                probe = next(iter(dl))
                v = (probe[0] if args.objective == "lejepa" else probe[0]).to(dev)[:64]
                er = effective_rank(backbone.feature(v))
            print(f"epoch {ep+1} done in {time.time()-t0:.0f}s  loss {running/steps_per_epoch:.4f}  eff_rank {er:.1f}")
            if tb:
                tb.add_scalar("collapse/effective_rank", er, gstep)
        except Exception as e:  # noqa: BLE001
            print(f"epoch {ep+1} done  (eff_rank skipped: {e})")
        save_state(ep + 1)
        save_eval_backbone()

    open(done_flag, "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"[pretrain] DONE -> {out_dir}/ema_backbone.pt")
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
        from ..data.gastronet import shard_paths

        if not shard_paths(os.path.expanduser(str(cfg.paths.gastronet))):
            args.corpus = "hkv_unlabeled"
            print("[pretrain] no GastroNet shards -> corpus = hkv_unlabeled (staged 99k)")
    train(cfg, args)


if __name__ == "__main__":
    main()
