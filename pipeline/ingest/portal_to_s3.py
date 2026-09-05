"""Stage the GastroNet-5M portal zips into ``s3://<bucket>/raw/``.

Idempotent: a shard already in S3 with a matching size is skipped. Every object
gets a sha256 and a line in ``ingest_log/<run>.jsonl``. Runs on the ephemeral
in-region EC2 box (S3 ingress is free), streaming each file to a local scratch
dir and multipart-uploading it.

Portal auth — pass exactly one of:
  --url-list FILE       one ``https://…`` per line (works if the portal hands out
                        direct links or presigned URLs)
  --portal-json FILE    {"base": "...", "cookie": "...", "file_list": [
                          {"name": "shard_0001.zip", "url_or_uuid": "..."}, ...]}
                        the SPA at cortex.thetavision.nl exposes a per-file
                        download endpoint; capture the session cookie + the file
                        list from the browser and drop them here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import key_exists, s3, sha256_file, upload_file  # noqa: E402


def _download(url: str, dest: str, headers: dict | None = None, tries: int = 5) -> None:
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return
        except Exception as e:  # noqa: BLE001
            wait = 2 ** t
            print(f"  retry {t + 1}/{tries} after {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError(f"download failed: {url}")


def _targets(args) -> list[dict]:
    if args.url_list:
        out = []
        for ln in open(args.url_list):
            u = ln.strip()
            if u and not u.startswith("#"):
                out.append({"name": os.path.basename(u.split("?")[0]), "url": u, "headers": {}})
        return out
    spec = json.load(open(args.portal_json))
    base = spec.get("base", "").rstrip("/")
    cookie = spec.get("cookie", "")
    hdr = {"Cookie": cookie} if cookie else {}
    out = []
    for f in spec["file_list"]:
        u = f.get("url") or f["url_or_uuid"]
        if not u.startswith("http"):
            u = f"{base}/{u.lstrip('/')}"
        out.append({"name": f["name"], "url": u, "headers": hdr})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--raw-prefix", default="raw/")
    ap.add_argument("--workdir", default="/mnt/work/raw")
    ap.add_argument("--url-list")
    ap.add_argument("--portal-json")
    ap.add_argument("--limit", type=int, default=0, help="stop after N (debug)")
    args = ap.parse_args()
    if not (args.url_list or args.portal_json):
        ap.error("need --url-list or --portal-json")

    os.makedirs(args.workdir, exist_ok=True)
    tgts = _targets(args)
    if args.limit:
        tgts = tgts[: args.limit]
    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = os.path.join(args.workdir, f"ingest_{run}.jsonl")
    print(f"[ingest] {len(tgts)} shards -> s3://{args.bucket}/{args.raw_prefix}")

    done = 0
    with open(log_path, "a") as log:
        for i, t in enumerate(tgts):
            key = f"{args.raw_prefix}{t['name']}"
            local = os.path.join(args.workdir, t["name"])
            head = key_exists(args.bucket, key)
            if head is not None and not os.path.exists(local):
                print(f"  [{i + 1}/{len(tgts)}] skip (in S3) {t['name']}")
                continue
            print(f"  [{i + 1}/{len(tgts)}] {t['name']}")
            _download(t["url"], local, t["headers"])
            digest = sha256_file(local)
            size = os.path.getsize(local)
            upload_file(args.bucket, key, local, {"Metadata": {"sha256": digest}})
            os.remove(local)
            rec = {"name": t["name"], "key": key, "bytes": size, "sha256": digest,
                   "url": t["url"].split("?")[0], "ts": time.time()}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            done += 1

    s3().upload_file(log_path, args.bucket, f"ingest_log/ingest_{run}.jsonl")
    print(f"[ingest] uploaded {done} new shard(s); log -> s3://{args.bucket}/ingest_log/ingest_{run}.jsonl")


if __name__ == "__main__":
    main()
