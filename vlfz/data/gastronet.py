"""GastroNet-5M loader.

Three read paths, chosen by ``source`` (config ``ssl.gastronet_source`` / env
``GASTRONET_SOURCE``):

* ``local_zip`` — the original: ``<root>/*.zip`` portal shards, a cached manifest
  of ``(shard_index, member)``, lazy per-worker zip handles. Map-style.
* ``local_webdataset`` (default) — the curated tier staged by ``pipeline/stage``:
  ``<root>/webdataset/*.tar`` of 224px JPEGs. Iterable (WebDataset).
* ``s3_webdataset`` — the same ``.tar`` shards streamed from
  ``s3://$GASTRONET_BUCKET/webdataset/`` with a bounded local cache. Iterable.
  For scale-out only; every full pass is a full-corpus egress.

Used for SSL pretraining only (unlabeled).
"""
from __future__ import annotations

import io
import os
import random
import zipfile
from glob import glob

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def resolve_source(cfg, override: str | None = None) -> str:
    src = override or os.environ.get("GASTRONET_SOURCE") or str(
        getattr(cfg.ssl, "gastronet_source", "local_webdataset"))
    if src not in ("local_zip", "local_webdataset", "s3_webdataset"):
        raise ValueError(f"bad gastronet source: {src}")
    return src


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


# ---------------------------------------------------------------- WebDataset tier
def _wds_root(cfg) -> str:
    return os.path.join(_root(cfg), "webdataset")


def webdataset_shards(cfg, source: str) -> list[str]:
    """URLs for wds.WebDataset: local paths, or presigned HTTPS GETs for
    s3_webdataset (wds's built-in HTTP(S) fetcher handles these directly — no
    `aws` CLI in the DataLoader worker, and the signature carries the auth so
    the bucket can stay private)."""
    if source == "local_webdataset":
        sh = sorted(glob(os.path.join(_wds_root(cfg), "*.tar")))
        if not sh:
            raise FileNotFoundError(
                f"no *.tar under {_wds_root(cfg)} — run pipeline/stage/stage_in.sh")
        return sh
    if source == "s3_webdataset":
        bucket = os.environ.get("GASTRONET_BUCKET") or str(getattr(cfg.ssl, "gastronet_bucket", ""))
        if not bucket:
            raise RuntimeError("s3_webdataset needs $GASTRONET_BUCKET (terraform output bucket)")
        import boto3

        cl = boto3.client("s3")
        keys, tok = [], None
        while True:
            kw = {"Bucket": bucket, "Prefix": "webdataset/"}
            if tok:
                kw["ContinuationToken"] = tok
            r = cl.list_objects_v2(**kw)
            keys += [o["Key"] for o in r.get("Contents", []) if o["Key"].endswith(".tar")]
            if not r.get("IsTruncated"):
                break
            tok = r["NextContinuationToken"]
        if not keys:
            raise FileNotFoundError(f"no webdataset/*.tar in s3://{bucket}")
        expires = int(getattr(cfg.ssl, "s3_presign_seconds", 86400))
        return [cl.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": k}, ExpiresIn=expires)
            for k in sorted(keys)]
    raise ValueError(source)


def webdataset_loader(cfg, transform, collate, *, source, batch_size, num_workers,
                      steps_per_epoch, shuffle_buf=2000, seed=0):
    """A DataLoader over the curated .tar tier. Yields the same crop-major list of
    stacked tensors as ``multicrop_collate`` — drop-in for the map-style path.

    ``s3_webdataset`` streams each shard over a presigned HTTPS GET and caches it
    to ``ssl.wds_cache_dir`` on first read (``url_to_cache_name`` keys the cache
    file on the S3 *path*, so it's stable across presign regenerations) — epoch 1
    pays the S3 egress, every later epoch or job resubmit reads local disk."""
    import webdataset as wds

    urls = webdataset_shards(cfg, source)
    cache_kw = {}
    if source == "s3_webdataset":
        cache_dir = os.path.expanduser(str(getattr(cfg.ssl, "wds_cache_dir", "~/.cache/wds")))
        os.makedirs(cache_dir, exist_ok=True)
        cache_kw = {"cache_dir": cache_dir,
                    "cache_size": int(getattr(cfg.ssl, "wds_cache_size_bytes", -1)),
                    "url_to_name": wds.cache.url_to_cache_name}

    def _decode(sample):
        from PIL import Image

        key = [k for k in sample if k.split(".")[-1] in ("jpg", "jpeg", "png", "webp")][0]
        img = Image.open(io.BytesIO(sample[key])).convert("RGB")
        return transform(img)

    ds = (
        wds.WebDataset(urls, resampled=True, shardshuffle=False,
                       nodesplitter=wds.split_by_node,
                       workersplitter=wds.split_by_worker, seed=seed,
                       handler=wds.warn_and_continue, **cache_kw)
        .shuffle(shuffle_buf)
        .map(_decode, handler=wds.warn_and_continue)
        .batched(batch_size, collation_fn=collate, partial=False)
    )

    # .with_epoch() has to bound the COMBINED stream, not the per-worker
    # pipeline. With num_workers=N the whole dataset pipeline runs inside every
    # worker, so ds.with_epoch(k) yields N*k batches per Lightning epoch (with
    # N=12 that made "epoch 0" ~56k batches and no epoch-end / schedule / ckpt
    # ever fired within the wall clock). WebLoader.with_epoch() is applied after
    # the multiprocessing fan-in -> exactly steps_per_epoch batches per epoch.
    dl = wds.WebLoader(ds, batch_size=None, num_workers=num_workers,
                       pin_memory=True, persistent_workers=(num_workers > 0))
    return dl.with_epoch(steps_per_epoch)


def main():
    import argparse

    from ..cfg import load_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--build-manifest", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--source", default=None,
                    choices=["local_zip", "local_webdataset", "s3_webdataset"])
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="iterate a few wds batches")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    src = resolve_source(cfg, a.source)

    if a.build_manifest:
        build_manifest(cfg, force=a.force)
        return
    if src == "local_zip":
        mf = build_manifest(cfg)
        rows = _read_manifest(mf)
        print(f"[local_zip] {len(rows)} images; sample:",
              [rows[i] for i in subset_indices(mf, 5)])
        return

    shards = webdataset_shards(cfg, src)
    print(f"[{src}] {len(shards)} .tar shards; first: {shards[0]}")
    if a.smoke:
        from .transforms import multicrop_collate, two_view_transform

        dl = webdataset_loader(cfg, two_view_transform(cfg), multicrop_collate,
                               source=src, batch_size=4, num_workers=0,
                               steps_per_epoch=5)
        for i, b in enumerate(dl):
            print(f"  batch {i}: {len(b)} views, view0 {tuple(b[0].shape)}")
        print("[smoke] ok")


if __name__ == "__main__":
    main()
