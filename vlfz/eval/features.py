"""Frozen-feature extraction with an on-disk cache.

One code path for every backbone (all are ViTBackbone). Given a list of
``(image_path, label)`` samples it returns ``{X: (N, 768), y: (N,)}`` and caches
it under ``<cache>/<key>.npz`` where key hashes (backbone tag, img_size, sample
paths + labels).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np


def _key(backbone_tag: str, img_size: int, samples: list[tuple[str, int]]) -> str:
    h = hashlib.sha1()
    h.update(f"{backbone_tag}|{img_size}|{len(samples)}".encode())
    for p, y in samples[:100000]:
        h.update(f"{os.path.basename(p)}|{y}|".encode())
    return h.hexdigest()[:16]


class _PathDataset:
    def __init__(self, samples, transform):
        self.samples = samples
        self.t = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from PIL import Image

        p, y = self.samples[i]
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.t(img), int(y), i


@np.errstate(all="ignore")
def extract_features(
    backbone,
    samples: list[tuple[str, int]],
    cfg,
    *,
    backbone_tag: str,
    cache_dir: str,
    device: str | None = None,
    recompute: bool = False,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    from ..data.transforms import eval_transform

    os.makedirs(cache_dir, exist_ok=True)
    key = _key(backbone_tag, int(cfg.eval.img_size), samples)
    path = os.path.join(cache_dir, f"feat_{key}.npz")
    if os.path.exists(path) and not recompute:
        d = np.load(path)
        return {"X": d["X"], "y": d["y"], "cache": path, "hit": True}

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(dev).eval()
    dl = DataLoader(
        _PathDataset(samples, eval_transform(cfg)),
        batch_size=int(cfg.eval.batch_size),
        num_workers=int(cfg.eval.num_workers),
        pin_memory=(dev == "cuda"),
        shuffle=False,
    )
    feats = np.empty((len(samples), backbone.embed_dim), dtype=np.float32)
    ys = np.empty(len(samples), dtype=np.int64)
    with torch.no_grad():
        for xb, yb, idx in dl:
            xb = xb.to(dev, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                f = backbone.feature(xb).float()
            idx = idx.numpy()
            feats[idx] = f.cpu().numpy()
            ys[idx] = yb.numpy()
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    np.savez(path, X=feats, y=ys)
    return {"X": feats, "y": ys, "cache": path, "hit": False}
