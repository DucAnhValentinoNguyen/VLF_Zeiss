"""Portal zips -> local curated WebDataset tier, in one pass, on LRZ.

Bypasses the S3 lake (the ephemeral EC2 proved unreliable in this account).
Per portal file: POST /api/provided_file/<id>/download_url/ -> 10-min presigned
URL -> stream the zip to a scratch file (Range-resume) -> decode every image,
drop phash duplicates, resize short side to --size, re-encode JPEG, append to
~--pack-image .tar shards under --out -> delete the zip. Idempotent-ish: a
(src_shard) already recorded in <out>/_ingested.txt is skipped.

    python -m pipeline.ingest.portal_to_wds \
      --portal-json ~/.gastronet_portal/portal.json \
      --out $GASTRONET_ROOT/webdataset --scratch $SCRATCH/gnzips \
      --limit 60 --jobs 8
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import zipfile
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portal_to_s3 import _fetch_to, _session  # noqa: E402

_IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _one_image(args):
    raw, size, quality = args
    import imagehash
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(raw))
        ph = str(imagehash.phash(im.convert("RGB")))
        im = Image.open(io.BytesIO(raw)); im.draft("RGB", (size * 2, size * 2))
        im = im.convert("RGB")
        w, h = im.size
        s = size / min(w, h)
        if s < 1.0:
            im = im.resize((max(size, round(w * s)), max(size, round(h * s))), Image.BICUBIC)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True)
        return ph, buf.getvalue()
    except Exception:
        return None, None


class ShardWriter:
    def __init__(self, out_dir, pack):
        self.out, self.pack = out_dir, pack
        self.idx = self._resume_idx()
        self.n_in = 0
        self.tar = None
        self._open()

    def _resume_idx(self):
        ex = [f for f in os.listdir(self.out) if f.startswith("gn-") and f.endswith(".tar")]
        return (max(int(f[3:-4]) for f in ex) + 1) if ex else 0

    def _open(self):
        self.path = os.path.join(self.out, f"gn-{self.idx:05d}.tar")
        self.tar = tarfile.open(self.path, "w")

    def add(self, key, jpg):
        ti = tarfile.TarInfo(f"{key}.jpg"); ti.size = len(jpg)
        self.tar.addfile(ti, io.BytesIO(jpg))
        self.n_in += 1
        if self.n_in >= self.pack:
            self.tar.close(); self.idx += 1; self.n_in = 0; self._open()

    def close(self):
        self.tar.close()
        if os.path.getsize(self.path) <= 1024:
            os.remove(self.path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--portal-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", default="/tmp/gnzips")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--quality", type=int, default=87)
    ap.add_argument("--pack", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    spec = json.load(open(a.portal_json))
    base = spec.get("cortex_base", "https://cortex.thetavision.nl").rstrip("/")
    files = sorted(spec["files"], key=lambda x: x["file_name"])
    if a.limit:
        files = files[: a.limit]
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.scratch, exist_ok=True)
    sess = _session(spec)

    done_log = os.path.join(a.out, "_ingested.txt")
    done = set(open(done_log).read().split()) if os.path.exists(done_log) else set()
    seen_ph: set[str] = set()
    for m in (os.path.join(a.out, f) for f in os.listdir(a.out) if f.endswith(".phash")):
        seen_ph |= set(open(m).read().split())

    sw = ShardWriter(a.out, a.pack)
    pool = Pool(a.jobs)
    kept = dropped = 0
    t_all = time.time()
    for i, f in enumerate(files, 1):
        name, fid = f["file_name"], f["id"]
        if name in done:
            print(f"[{i}/{len(files)}] {name}  skip (done)"); continue
        zp = os.path.join(a.scratch, name)
        t0 = time.time()
        _fetch_to(sess, base, fid, zp, f.get("size", 0))
        dl = time.time() - t0
        new_ph = []
        with zipfile.ZipFile(zp) as zf:
            infos = [x for x in zf.infolist()
                     if not x.is_dir() and os.path.splitext(x.filename)[1].lower() in _IMG_EXTS]
            batch = ((zf.read(x.filename), a.size, a.quality) for x in infos)
            for x, (ph, jpg) in zip(infos, pool.imap(_one_image, batch, chunksize=64)):
                if jpg is None or ph in seen_ph:
                    dropped += 1; continue
                seen_ph.add(ph); new_ph.append(ph)
                key = f"{name[:-4]}/{os.path.splitext(os.path.basename(x.filename))[0]}"
                sw.add(key, jpg); kept += 1
        os.remove(zp)
        with open(os.path.join(a.out, f"{name[:-4]}.phash"), "w") as m:
            m.write("\n".join(new_ph))
        with open(done_log, "a") as d:
            d.write(name + "\n")
        print(f"[{i}/{len(files)}] {name}  {f.get('size',0)/1e9:.1f}GB dl {dl:.0f}s | "
              f"kept {kept} dropped {dropped} | tar {sw.idx}")
    sw.close(); pool.close(); pool.join()
    print(f"[wds] done in {(time.time()-t_all)/60:.0f}m: kept {kept}, dropped {dropped}, "
          f"{sw.idx + 1} shards -> {a.out}")


if __name__ == "__main__":
    main()
