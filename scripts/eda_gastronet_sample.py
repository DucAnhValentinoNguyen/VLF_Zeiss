"""EDA of a random sample of GastroNet-5M portal archives (measured, not assumed).

The full corpus is ~1.75 TiB; decoding all of it for an EDA is not needed to characterise it.
This draws a seeded random sample of whole .zip archives, downloads each, and for every image
records: strict-decode status (ok / truncated-but-recoverable / unreadable), size, mode, file
bytes, intensity statistics and a 64-bit perceptual hash. Redundancy is then computed within
each archive (frames of one procedure live together) and exact-byte duplicates across the sample.

  python scripts/eda_gastronet_sample.py --n 24 --workers 6 --seed 0
"""
import argparse
import collections
import hashlib
import io
import json
import os
import random
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_common import redundancy  # noqa: E402

MODE_CODE = {"RGB": 0, "RGBA": 1, "L": 2, "P": 3, "I;16": 4, "other": 5}
STATUS_CODE = {"ok": 0, "truncated": 1, "unreadable": 2}


def scan_archive(job):
    import imagehash
    from PIL import Image, ImageFile
    from vlfz.data.portal import PortalClient

    item, members, portal_json, cache_dir, out_dir = job
    t0 = time.time()
    path = Path(cache_dir) / item["file_name"]
    client = PortalClient(portal_json)
    client.download({k: item[k] for k in ("id", "file_name", "size")}, path)
    t_dl = time.time() - t0

    n = len(members)
    cols = {k: np.zeros(n, dtype=np.float32) for k in ("w", "h", "bytes", "mean", "std", "dark", "bright", "color")}
    status = np.zeros(n, dtype=np.int8)
    mode = np.zeros(n, dtype=np.int8)
    hashes = np.zeros(n, dtype=np.uint64)
    has_hash = np.zeros(n, dtype=bool)
    md5 = []
    folders = collections.Counter()
    with zipfile.ZipFile(path) as z:
        for i, name in enumerate(members):
            folders[name.split("/")[0] if "/" in name else ""] += 1
            try:
                raw = z.read(name)
            except Exception:
                status[i] = 2
                md5.append("")
                continue
            md5.append(hashlib.md5(raw).hexdigest())
            cols["bytes"][i] = len(raw)
            im = None
            ImageFile.LOAD_TRUNCATED_IMAGES = False
            try:
                im = Image.open(io.BytesIO(raw))
                im.load()
                status[i] = 0
            except Exception:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                try:
                    im = Image.open(io.BytesIO(raw))
                    im.load()
                    status[i] = 1
                except Exception:
                    status[i] = 2
                    im = None
                finally:
                    ImageFile.LOAD_TRUNCATED_IMAGES = False
            if im is None:
                continue
            cols["w"][i], cols["h"][i] = im.size
            mode[i] = MODE_CODE.get(im.mode, MODE_CODE["other"])
            rgb = im.convert("RGB")
            th = np.asarray(rgb.resize((64, 64)), dtype=np.float32)
            g = th.mean(axis=2)
            cols["mean"][i], cols["std"][i] = g.mean(), g.std()
            cols["dark"][i], cols["bright"][i] = (g < 12).mean(), (g > 243).mean()
            cols["color"][i] = (np.abs(th[..., 0] - th[..., 1]) + np.abs(th[..., 1] - th[..., 2])).mean()
            hashes[i] = int(str(imagehash.phash(rgb)), 16)
            has_hash[i] = True
    os.remove(path)

    red = redundancy(hashes[has_hash])
    md = collections.Counter(m for m in md5 if m)
    exact_dups = int(sum(c - 1 for c in md.values() if c > 1))
    np.savez_compressed(Path(out_dir) / f"{item['file_name']}.npz", status=status, mode=mode, hashes=hashes,
                        has_hash=has_hash, **cols)
    return {"archive": item["file_name"], "n_members": n, "download_s": round(t_dl, 1), "total_s": round(time.time() - t0, 1),
            "ok": int((status == 0).sum()), "truncated": int((status == 1).sum()), "unreadable": int((status == 2).sum()),
            "exact_byte_dups": exact_dups, "n_folders": len(folders), "redundancy": red,
            "md5": [m for m in md5 if m]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_root = Path(os.environ.get("OUT_ROOT", os.path.expanduser("~/vlf_zeiss_fullstream")))
    out_dir = out_root / "eda" / "gastronet_sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_root / "eda_zip_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    portal_json = os.environ.get("PORTAL_JSON", os.path.expanduser("~/.gastronet_portal/portal.json"))
    manifest = json.load(open(out_root / "portal_manifest.json"))
    arch = manifest["archives"]
    zips = [x for x in arch if x["file_name"].lower().endswith(".zip")]
    sample = random.Random(a.seed).sample(zips, a.n)
    print(f"[eda-gn] {len(arch)} archives ({len(zips)} zip); sampling {a.n} (seed {a.seed}): "
          + ", ".join(s["file_name"] for s in sample), flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(scan_archive, ({k: s[k] for k in ("id", "file_name", "size")}, s["members"], portal_json,
                                         str(cache_dir), str(out_dir))): s["file_name"] for s in sample}
        for f in as_completed(futs):
            try:
                r = f.result()
                results.append(r)
                print(f"[eda-gn] done {r['archive']}  n={r['n_members']} ok={r['ok']} trunc={r['truncated']} "
                      f"unread={r['unreadable']} exact_dups={r['exact_byte_dups']} {r['total_s']}s", flush=True)
            except Exception as e:  # keep going; record the failure
                print(f"[eda-gn] FAILED {futs[f]}: {type(e).__name__}: {str(e)[:200]}", flush=True)
                results.append({"archive": futs[f], "error": f"{type(e).__name__}"})

    good = [r for r in results if "error" not in r]
    allmd = collections.Counter(m for r in good for m in r["md5"])
    cross_exact = int(sum(c - 1 for c in allmd.values() if c > 1))
    within_exact = int(sum(r["exact_byte_dups"] for r in good))
    for r in good:
        r.pop("md5")
    summary = {"seed": a.seed, "archives_total": len(arch), "zip_archives": len(zips), "sampled": a.n,
               "archives_ok": len(good), "archives_failed": [r["archive"] for r in results if "error" in r],
               "images": int(sum(r["n_members"] for r in good)),
               "sample_archive_names": sorted(r["archive"] for r in good),
               "exact_byte_dups_within_archive": within_exact, "exact_byte_dups_across_sample": cross_exact,
               "per_archive": sorted(good, key=lambda r: r["archive"])}
    fp = out_dir.parent / "gastronet_sample.json"
    json.dump(summary, open(fp, "w"), indent=1)
    print("[eda-gn] wrote", fp, flush=True)


if __name__ == "__main__":
    main()
