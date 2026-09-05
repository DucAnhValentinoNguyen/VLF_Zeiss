"""Build the raw-corpus Parquet catalog: one row per image inside the portal zips.

Streams each ``raw/*.zip`` from S3, reads its central directory + decodes each
member's header (cheap — PIL only touches the first few KB for size/mode), and
writes a Snappy Parquet file per ``shard_group`` to
``s3://<bucket>/catalog/manifest/shard_group=<g>/part-<name>.parquet``.

Columns: shard, member, bytes, sha256, ext, width, height, mode, phash, corrupt.
``phash`` is computed here (one decode) so dedup in curation is a pure join.
After upload, run ``MSCK REPAIR TABLE gastronet.manifest`` (or add partitions) in
Athena.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import zipfile

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import download_file, groups, s3, sha256_bytes, shard_refs, upload_file  # noqa: E402

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

_SCHEMA = pa.schema([
    ("shard", pa.string()), ("member", pa.string()), ("bytes", pa.int64()),
    ("sha256", pa.string()), ("ext", pa.string()), ("width", pa.int32()),
    ("height", pa.int32()), ("mode", pa.string()), ("phash", pa.string()),
    ("corrupt", pa.bool_()),
])


def _rows_for_zip(zip_path: str, shard_name: str) -> list[dict]:
    import imagehash
    from PIL import Image

    rows: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            ext = os.path.splitext(info.filename)[1].lower()
            if ext not in _IMG_EXTS:
                continue
            raw = zf.read(info.filename)
            row = {"shard": shard_name, "member": info.filename, "bytes": len(raw),
                   "sha256": sha256_bytes(raw), "ext": ext, "width": 0, "height": 0,
                   "mode": "", "phash": "", "corrupt": False}
            try:
                im = Image.open(io.BytesIO(raw))
                im.draft("RGB", (256, 256))
                row["width"], row["height"] = im.size
                row["mode"] = im.mode
                row["phash"] = str(imagehash.phash(im.convert("RGB")))
            except Exception:
                row["corrupt"] = True
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--raw-prefix", default="raw/")
    ap.add_argument("--catalog-prefix", default="catalog/manifest/")
    ap.add_argument("--workdir", default="/mnt/work/cat")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--only-group", default="", help="restrict to one shard_group (debug)")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    refs = shard_refs(args.bucket, args.raw_prefix)
    if not refs:
        sys.exit(f"no *.zip under s3://{args.bucket}/{args.raw_prefix}")
    total = 0
    for g, grefs in sorted(groups(refs).items()):
        if args.only_group and g != args.only_group:
            continue
        out_local = os.path.join(args.workdir, f"manifest_{g}.parquet")
        if os.path.exists(out_local):
            print(f"[catalog] {g}: cached")
        else:
            rows: list[dict] = []
            for r in grefs:
                zp = os.path.join(args.workdir, r.name + ".zip")
                download_file(args.bucket, r.key, zp)
                rr = _rows_for_zip(zp, r.name)
                os.remove(zp)
                rows.extend(rr)
                print(f"[catalog] {g}/{r.name}: {len(rr)} imgs")
            df = pd.DataFrame(rows)
            pq.write_table(pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False),
                           out_local, compression="snappy")
        n = pq.read_metadata(out_local).num_rows
        total += n
        if args.upload:
            key = f"{args.catalog_prefix}shard_group={g}/part-{g}.parquet"
            upload_file(args.bucket, key, out_local)
            print(f"[catalog] {g}: {n} rows -> s3://{args.bucket}/{key}")
    print(f"[catalog] total images catalogued: {total}")
    if args.upload:
        print("[catalog] next: Athena  MSCK REPAIR TABLE gastronet.manifest;")


if __name__ == "__main__":
    main()
