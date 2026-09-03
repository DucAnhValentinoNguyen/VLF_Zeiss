"""GastroNet-5M loader — lazy reads from zip shards, no extraction.

Layout (from cortex.thetavision.nl): ``<root>/*.zip``, each shard <= 10k PNGs.
~4.82M images total. We build a manifest of (shard_index, member_name) once and
cache it; the dataset opens each zip lazily with a per-worker handle and decodes
bytes -> PIL on __getitem__.

Used for SSL pretraining only (unlabeled). A class/shard-balanced subset of
``ssl.subset_images`` images keeps single-GPU training tractable.
"""
from __future__ import annotations

import io
import os
import random
import zipfile
from glob import glob

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def _root(cfg) -> str:
    return os.path.abspath(os.path.expanduser(str(cfg.paths.gastronet)))


def shard_paths(root: str) -> list[str]:
    return sorted(glob(os.path.join(root, "*.zip")))


def build_manifest(cfg, *, force: bool = False) -> str:
    """Write ``<cache>/gastronet_manifest.tsv`` with lines '<shard_idx>\\t<member>'."""
    root = _root(cfg)
    cache = os.path.abspath(os.path.expanduser(str(cfg.paths.cache)))
    os.makedirs(cache, exist_ok=True)
    mf = os.path.join(cache, "gastronet_manifest.tsv")
    if os.path.exists(mf) and not force:
        return mf

    shards = shard_paths(root)
    if not shards:
        raise FileNotFoundError(f"no *.zip shards under {root}")
    n = 0
    with open(mf, "w") as out:
        for si, sp in enumerate(shards):
            with zipfile.ZipFile(sp) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    if info.filename.lower().endswith(_IMG_EXTS):
                        out.write(f"{si}\t{info.filename}\n")
                        n += 1
            print(f"[gastronet] shard {si + 1}/{len(shards)}: {os.path.basename(sp)}  (running total {n})")
    with open(mf + ".meta", "w") as m:
        m.write(f"shards={len(shards)}\nimages={n}\n")
    print(f"[gastronet] manifest -> {mf}  ({n} images across {len(shards)} shards)")
    return mf


def _read_manifest(mf: str) -> list[tuple[int, str]]:
    rows = []
    with open(mf) as f:
        for line in f:
            si, name = line.rstrip("\n").split("\t", 1)
            rows.append((int(si), name))
    return rows


def subset_indices(mf: str, n: int, seed: int = 0) -> list[int]:
    """Shard-balanced subset of `n` manifest rows (0 -> all)."""
    rows = _read_manifest(mf)
    if not n or n >= len(rows):
        return list(range(len(rows)))
    by_shard: dict = {}
    for i, (si, _) in enumerate(rows):
        by_shard.setdefault(si, []).append(i)
    rng = random.Random(seed)
    for v in by_shard.values():
        rng.shuffle(v)
    per = max(1, n // len(by_shard))
    picked = []
    for v in by_shard.values():
        picked.extend(v[:per])
    rng.shuffle(picked)
    return picked[:n]


class GastroNetDataset:
    """Yields a transformed PIL image (SSL: the transform returns >1 view)."""

    def __init__(self, cfg, transform, *, subset: int | None = None, seed: int = 0):
        self.root = _root(cfg)
        self.shards = shard_paths(self.root)
        self.mf = build_manifest(cfg)
        rows = _read_manifest(self.mf)
        idx = subset_indices(
            self.mf, int(subset if subset is not None else 0), seed
        )
        self.items = [rows[i] for i in idx]
        self.transform = transform
        self._zh: dict[int, zipfile.ZipFile] = {}

    def __len__(self):
        return len(self.items)

    def _zip(self, si: int) -> zipfile.ZipFile:
        zf = self._zh.get(si)
        if zf is None:
            zf = zipfile.ZipFile(self.shards[si])
            self._zh[si] = zf
        return zf

    def __getitem__(self, i):
        from PIL import Image

        si, name = self.items[i]
        try:
            data = self._zip(si).read(name)
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.transform(img)


def main():
    import argparse

    from ..cfg import load_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--build-manifest", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.build_manifest:
        build_manifest(cfg, force=a.force)
    else:
        mf = build_manifest(cfg)
        rows = _read_manifest(mf)
        print(f"{len(rows)} images; sample subset of 5:",
              [rows[i] for i in subset_indices(mf, 5)])


if __name__ == "__main__":
    main()
