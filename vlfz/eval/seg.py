"""Zero-shot polyp segmentation on HyperKvasir-segmented via dense patch-feature
k-NN label transfer.

No training. Extract ViT patch tokens (14x14) for a small labelled support set,
label each support patch by its mask overlap, then for every query patch do a
weighted k-NN vote over all support patches -> per-patch polyp probability ->
bilinear upsample -> Dice / mIoU + pixel-wise ECE / NLL (global and foreground).
"""
from __future__ import annotations

import json
import os

import numpy as np

from ..cfg import ensure_dirs, load_cfg, provenance, set_seed
from .calibration import metric_block
from .knn import knn_vote

_GRID = 14  # 224/16


def _load_img(path, size):
    from PIL import Image

    return Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)


def _load_mask_grid(path, grid):
    from PIL import Image

    m = np.asarray(Image.open(path).convert("L").resize((grid, grid), Image.BILINEAR))
    return (m > 127).astype(np.int64)                       # (grid, grid) patch labels


def _load_mask_full(path, size):
    from PIL import Image

    m = np.asarray(Image.open(path).convert("L").resize((size, size), Image.NEAREST))
    return (m > 127).astype(np.int64)


def _patch_features(backbone, pil_imgs, cfg, device):
    import torch

    from ..data.transforms import eval_transform

    tf = eval_transform(cfg)
    x = torch.stack([tf(im) for im in pil_imgs]).to(device)
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            tok = backbone.forward_tokens(x).float()        # (B, 196, C)
    return tok.cpu().numpy()


def run_seg(cfg, *, dataset="hyperkvasir", init, objective, stage, corpus="gastronet") -> dict:
    import torch

    from ..data import hyperkvasir as H
    from ..models.vit_backbone import build_vit_b16

    set_seed(int(cfg.seed))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    size = int(cfg.eval.img_size)
    k = int(cfg.hyperkvasir.seg.patch_k)
    tau = float(cfg.hyperkvasir.seg.patch_tau)

    if stage == "pre":
        backbone = build_vit_b16(init, cfg, pretrained_init=True); tag = f"{init}_pre"
    else:
        ckpt = os.path.join(os.path.expanduser(str(cfg.paths.ckpts)),
                            f"{objective}_{init}_{corpus}_full", "ema_backbone.pt")
        backbone = build_vit_b16(init, cfg, ckpt=ckpt)
        tag = f"{init}_{objective}_{corpus}_post"
    backbone = backbone.to(device).eval()

    sup = H.seg_pairs(cfg, "support")
    qry = H.seg_pairs(cfg, "query")
    # support patch bank
    sup_imgs = [_load_img(p, size) for p, _ in sup]
    sup_feat = _patch_features(backbone, sup_imgs, cfg, device)          # (S, P, C)
    sup_lab = np.stack([_load_mask_grid(m, _GRID).reshape(-1) for _, m in sup])  # (S, P)
    Xref = sup_feat.reshape(-1, sup_feat.shape[-1])
    yref = sup_lab.reshape(-1)

    # query
    B = 16
    probs_all, gt_all = [], []
    dices, ious = [], []
    for i in range(0, len(qry), B):
        chunk = qry[i:i + B]
        imgs = [_load_img(p, size) for p, _ in chunk]
        feat = _patch_features(backbone, imgs, cfg, device)             # (b, P, C)
        b, P, C = feat.shape
        out = knn_vote(Xref, yref, feat.reshape(-1, C), n_classes=2, k=k, tau=tau)
        pp = out["probs"].reshape(b, _GRID, _GRID, 2)[..., 1]           # polyp prob per patch
        for j, (_, mp) in enumerate(chunk):
            pmap = _upsample(pp[j], size)
            gt = _load_mask_full(mp, size)
            pred = (pmap >= 0.5).astype(np.int64)
            inter = int((pred & gt).sum()); union = int((pred | gt).sum())
            psum = int(pred.sum()); gsum = int(gt.sum())
            dices.append((2 * inter / (psum + gsum)) if (psum + gsum) else 1.0)
            ious.append((inter / union) if union else 1.0)
            probs_all.append(np.stack([1 - pmap.ravel(), pmap.ravel()], 1))
            gt_all.append(gt.ravel())

    P = np.concatenate(probs_all); G = np.concatenate(gt_all)
    nb, ab = int(cfg.calibration.n_bins), int(cfg.calibration.adaptive_bins)
    glob = metric_block(P, G, nb, ab)
    fg = metric_block(P[G == 1], G[G == 1], nb, ab) if (G == 1).any() else {}

    res = {
        "dataset": dataset, "task": "hkv_seg", "tag": tag, "init": init,
        "objective": ("none" if stage == "pre" else objective),
        "corpus": ("none" if stage == "pre" else corpus), "stage": stage,
        "n_support": len(sup), "n_query": len(qry),
        "rows": [{
            "dataset": dataset, "task": "hkv_seg", "init": init, "stage": stage,
            "objective": ("none" if stage == "pre" else objective),
            "corpus": ("none" if stage == "pre" else corpus),
            "protocol": "seg_knn", "temp_scaled": False, "T": None,
            "dice": float(np.mean(dices)), "miou": float(np.mean(ious)),
            "ece_ew": glob["ece_ew"], "ece_adaptive": glob["ece_adaptive"],
            "nll": glob["nll"], "brier": glob["brier"],
            "nll_fg": fg.get("nll"), "ece_ew_fg": fg.get("ece_ew"),
            "k": k, "n_query": len(qry),
        }],
        "provenance": provenance(int(cfg.seed), dataset=dataset, init=init, objective=objective,
                                 stage=stage, corpus=corpus, task="hkv_seg"),
    }
    out_dir = os.path.expanduser(str(cfg.paths.results))
    ensure_dirs(out_dir)
    fp = os.path.join(out_dir, f"{dataset}__{tag}__hkv_seg.json")
    json.dump(res, open(fp, "w"), indent=2, default=float)
    print(f"[seg] {tag}: Dice {np.mean(dices):.3f}  mIoU {np.mean(ious):.3f}  "
          f"pixel-NLL {glob['nll']:.3f} -> {fp}")
    return res


def _upsample(patch_prob: np.ndarray, size: int) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    t = torch.as_tensor(patch_prob, dtype=torch.float32)[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--init", choices=["siglip2", "imagenet"], required=True)
    ap.add_argument("--objective", choices=["lejepa", "dino"], default="lejepa")
    ap.add_argument("--corpus", default="gastronet", choices=["gastronet", "hkv_unlabeled"])
    ap.add_argument("--stage", choices=["pre", "post"], required=True)
    a = ap.parse_args()
    run_seg(load_cfg(a.config), init=a.init, objective=a.objective, stage=a.stage, corpus=a.corpus)


if __name__ == "__main__":
    main()
