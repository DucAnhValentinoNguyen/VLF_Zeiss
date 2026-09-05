"""Curate the raw corpus into a training-ready WebDataset tier.

Per ``shard_group``: pull its ``raw/*.zip`` from S3, and for every image
  * decode, drop if corrupt,
  * drop exact perceptual-hash duplicates (keep the first seen in the group),
  * resize so the short side == ``--size`` (default 224; downscale only),
  * re-encode JPEG q``--quality`` (default 87), RGB,
  * append to a ``.tar`` WebDataset shard (~``--pack`` images each).
Then upload the ``.tar``(s) to ``s3://<bucket>/webdataset/`` and a Snappy Parquet
row-set to ``s3://<bucket>/catalog/curated/shard_group=<g>/``.

Columns: wds_shard, key, src_shard, src_member, bytes, width, height, phash, dup_of.

Runs multi-process over shard_groups (``--jobs``). Same code path works as an AWS
Batch array job — see curate/Dockerfile — with ``--only-group`` from the array
index.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import zipfile
from multiprocessing import Pool

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import download_file, groups, shard_refs, upload_file  # noqa: E402

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
_SCHEMA = pa.schema([
    ("wds_shard", pa.string()), ("key", pa.string()), ("src_shard", pa.string()),
    ("src_member", pa.string()), ("bytes", pa.int64()), ("width", pa.int32()),
    ("height", pa.int32()), ("phash", pa.string()), ("dup_of", pa.string()),
])


def _resize_encode(raw: bytes, size: int, quality: int):
    from PIL import Image

    im = Image.open(io.BytesIO(raw))
    im.draft("RGB", (size * 2, size * 2))
    im = im.convert("RGB")
    w, h = im.size
    s = size / min(w, h)
    if s < 1.0:
        im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue(), im.size


def _curate_group(job: dict) -> dict:
    import imagehash
    from PIL import Image

    g, grefs = job["group"], job["refs"]
    bucket, size, quality, pack = job["bucket"], job["size"], job["quality"], job["pack"]
    work = os.path.join(job["workdir"], g)
    os.makedirs(work, exist_ok=True)

    seen: dict[str, str] = {}
    rows: list[dict] = []
    kept = dropped_dup = dropped_bad = 0
    tar_idx, in_tar = 0, 0
    tar_path = os.path.join(work, f"{g}-{tar_idx:04d}.tar")
    tar = tarfile.open(tar_path, "w")
    made_tars = [tar_path]

    def _rotate():
        nonlocal tar, tar_idx, in_tar, tar_path
        tar.close()
        tar_idx += 1
        in_tar = 0
        tar_path = os.path.join(work, f"{g}-{tar_idx:04d}.tar")
        tar = tarfile.open(tar_path, "w")
        made_tars.append(tar_path)

    for r in grefs:
        zp = os.path.join(work, r.name + ".zip")
        download_file(bucket, r.key, zp)
        with zipfile.ZipFile(zp) as zf:
            for info in zf.infolist():
                if info.is_dir() or os.path.splitext(info.filename)[1].lower() not in _IMG_EXTS:
                    continue
                raw = zf.read(info.filename)
                key = f"{r.name}/{os.path.splitext(os.path.basename(info.filename))[0]}"
                try:
                    ph = str(imagehash.phash(Image.open(io.BytesIO(raw)).convert("RGB")))
                except Exception:
                    dropped_bad += 1
                    rows.append({"wds_shard": "", "key": key, "src_shard": r.name,
                                 "src_member": info.filename, "bytes": 0, "width": 0,
                                 "height": 0, "phash": "", "dup_of": "CORRUPT"})
                    continue
                if ph in seen:
                    dropped_dup += 1
                    rows.append({"wds_shard": "", "key": key, "src_shard": r.name,
                                 "src_member": info.filename, "bytes": 0, "width": 0,
                                 "height": 0, "phash": ph, "dup_of": seen[ph]})
                    continue
                seen[ph] = key
                try:
                    jpg, (w, h) = _resize_encode(raw, size, quality)
                except Exception:
                    dropped_bad += 1
                    continue
                ti = tarfile.TarInfo(f"{key}.jpg")
                ti.size = len(jpg)
                tar.addfile(ti, io.BytesIO(jpg))
                kept += 1
                in_tar += 1
                rows.append({"wds_shard": os.path.basename(tar_path), "key": key,
                             "src_shard": r.name, "src_member": info.filename,
                             "bytes": len(jpg), "width": w, "height": h,
                             "phash": ph, "dup_of": ""})
                if in_tar >= pack:
                    _rotate()
        os.remove(zp)
    tar.close()
    made_tars = [t for t in made_tars if os.path.getsize(t) > 1024]

    cat_local = os.path.join(work, f"curated_{g}.parquet")
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows), schema=_SCHEMA, preserve_index=False),
                   cat_local, compression="snappy")

    if job["upload"]:
        for t in made_tars:
            upload_file(bucket, f"{job['out_prefix']}{os.path.basename(t)}", t)
        upload_file(bucket, f"{job['curated_catalog']}shard_group={g}/part-{g}.parquet", cat_local)
        for t in made_tars:
            os.remove(t)

    return {"group": g, "kept": kept, "dropped_dup": dropped_dup,
            "dropped_bad": dropped_bad, "tars": len(made_tars)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--raw-prefix", default="raw/")
    ap.add_argument("--out-prefix", default="webdataset/")
    ap.add_argument("--curated-catalog", default="catalog/curated/")
    ap.add_argument("--workdir", default="/mnt/work/cur")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--quality", type=int, default=87)
    ap.add_argument("--pack", type=int, default=10000, help="images per .tar shard")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--only-group", default="")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    gm = groups(shard_refs(args.bucket, args.raw_prefix))
    if not gm:
        sys.exit(f"no *.zip under s3://{args.bucket}/{args.raw_prefix}")
    items = [{"group": g, "refs": refs, "bucket": args.bucket, "size": args.size,
              "quality": args.quality, "pack": args.pack, "workdir": args.workdir,
              "out_prefix": args.out_prefix, "curated_catalog": args.curated_catalog,
              "upload": args.upload}
             for g, refs in sorted(gm.items()) if not args.only_group or g == args.only_group]

    res = (list(map(_curate_group, items)) if args.jobs <= 1
           else Pool(args.jobs).map(_curate_group, items))
    kept = sum(r["kept"] for r in res)
    dd = sum(r["dropped_dup"] for r in res)
    db = sum(r["dropped_bad"] for r in res)
    tars = sum(r["tars"] for r in res)
    print(f"[curate] groups={len(res)} kept={kept} dup_dropped={dd} bad_dropped={db} tars={tars}")
    if args.upload:
        print("[curate] next: Athena  MSCK REPAIR TABLE gastronet.curated;")


if __name__ == "__main__":
    main()
