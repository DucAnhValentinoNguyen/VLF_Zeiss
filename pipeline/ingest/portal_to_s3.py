"""Stage the GastroNet-5M portal zips into ``s3://<bucket>/raw/``.

The cortex.thetavision.nl portal has no static download links: for each file you
POST ``/api/provided_file/<id>/download_url/`` (session cookie + CSRF) and get a
short-lived (10 min) presigned URL on their own S3. This script does that dance
per file, streams the presigned URL to our bucket with HTTP-Range resume (so a
mid-file expiry just triggers a fresh presign + continue), verifies size, writes
a sha256, and skips anything already uploaded.

Auth + file list come from ``--portal-json``:
  {
    "cortex_base": "https://cortex.thetavision.nl",
    "session":     {"sessionid": "...", "csrftoken": "..."},
    "files":       [{"id": 509, "file_name": "0000.zip", "size": 4176856448}, ...]
  }
``session`` values are the browser cookie values (DevTools → Application →
Cookies). The session is long-lived; if it expires mid-run the download_url POST
returns 401/403 -- refresh the cookie and re-run, it resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import key_exists, s3, sha256_file  # noqa: E402

_CHUNK = 8 << 20


def _session(spec: dict) -> requests.Session:
    s = spec["session"]
    sess = requests.Session()
    sess.cookies.set("sessionid", s["sessionid"], domain="cortex.thetavision.nl")
    sess.cookies.set("csrftoken", s["csrftoken"], domain="cortex.thetavision.nl")
    sess.headers.update({
        "x-csrftoken": s["csrftoken"],
        "content-type": "application/json",
        "origin": spec.get("cortex_base", "https://cortex.thetavision.nl"),
        "referer": spec.get("cortex_base", "https://cortex.thetavision.nl")
                   + "/dataset-provider/request/download/",
        "user-agent": "gastronet-ingest/1.0",
    })
    return sess


def _presign(sess: requests.Session, base: str, file_id: int) -> str:
    r = sess.post(f"{base}/api/provided_file/{file_id}/download_url/", data="{}", timeout=60)
    if r.status_code in (401, 403):
        raise SystemExit(f"portal auth rejected ({r.status_code}) — refresh the session cookie in portal.json")
    r.raise_for_status()
    return r.json()["url"]


def _fetch_to(sess, base, file_id, dest, expected, tries=6) -> None:
    for attempt in range(tries):
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if expected and have >= expected:
            return
        url = _presign(sess, base, file_id)
        try:
            hdrs = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, headers=hdrs, stream=True, timeout=(30, 120)) as g:
                if g.status_code not in (200, 206):
                    raise RuntimeError(f"GET {g.status_code}")
                mode = "ab" if have and g.status_code == 206 else "wb"
                with open(dest, mode) as f:
                    for chunk in g.iter_content(_CHUNK):
                        f.write(chunk)
            if not expected or os.path.getsize(dest) >= expected:
                return
            print(f"    short read ({os.path.getsize(dest)}/{expected}), resuming")
        except Exception as e:  # noqa: BLE001
            wait = min(60, 2 ** attempt)
            print(f"    retry {attempt + 1}/{tries} after {wait}s ({e})")
            time.sleep(wait)
    raise RuntimeError(f"download failed after {tries} tries: id={file_id}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--raw-prefix", default="raw/")
    ap.add_argument("--portal-json", required=True)
    ap.add_argument("--workdir", default="/mnt/work/raw")
    ap.add_argument("--limit", type=int, default=0, help="stop after N (debug)")
    args = ap.parse_args()

    spec = json.load(open(args.portal_json))
    base = spec.get("cortex_base", "https://cortex.thetavision.nl").rstrip("/")
    files = spec["files"]
    if args.limit:
        files = files[: args.limit]
    os.makedirs(args.workdir, exist_ok=True)
    sess = _session(spec)

    run = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    log_path = os.path.join(args.workdir, f"ingest_{run}.jsonl")
    total = sum(f.get("size", 0) for f in files)
    print(f"[ingest] {len(files)} files (~{total / 1e12:.2f} TB) -> s3://{args.bucket}/{args.raw_prefix}")

    done = skipped = 0
    with open(log_path, "a") as log:
        for i, f in enumerate(files, 1):
            name, fid, exp = f["file_name"], f["id"], f.get("size", 0)
            key = f"{args.raw_prefix}{name}"
            head = key_exists(args.bucket, key)
            if head is not None and (not exp or head["ContentLength"] == exp):
                skipped += 1
                continue
            local = os.path.join(args.workdir, name)
            print(f"  [{i}/{len(files)}] {name}  ({exp / 1e9:.2f} GB)")
            t0 = time.time()
            _fetch_to(sess, base, fid, local, exp)
            digest = sha256_file(local)
            size = os.path.getsize(local)
            s3().upload_file(local, args.bucket, key, ExtraArgs={"Metadata": {"sha256": digest}})
            os.remove(local)
            dt = time.time() - t0
            print(f"      {size / 1e9:.2f} GB in {dt:.0f}s ({size / dt / 1e6:.0f} MB/s) -> {key}")
            rec = {"name": name, "id": fid, "key": key, "bytes": size, "sha256": digest, "ts": time.time()}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            done += 1

    s3().upload_file(log_path, args.bucket, f"ingest_log/ingest_{run}.jsonl")
    print(f"[ingest] done: {done} uploaded, {skipped} already present. "
          f"log -> s3://{args.bucket}/ingest_log/ingest_{run}.jsonl")


if __name__ == "__main__":
    main()
