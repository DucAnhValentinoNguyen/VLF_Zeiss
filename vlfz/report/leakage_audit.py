"""Perceptual-hash overlap between the SSL corpus subset and REAL-Colon query
frames. Expected ~0 (different endoscopy modality); recorded so a non-zero count
is caught before results are trusted.
"""
from __future__ import annotations

import argparse
import json
import os

from ..cfg import load_cfg


def phash_set(paths, max_n=200000):
    import imagehash
    from PIL import Image

    hs = set()
    for i, p in enumerate(paths):
        if i >= max_n:
            break
        try:
            hs.add(str(imagehash.phash(Image.open(p).convert("RGB"))))
        except Exception:
            continue
    return hs


def audit(cfg, *, ssl_sample=20000, task="rc_frame") -> dict:
    from ..data import realcolon as RC
    from ..data.gastronet import GastroNetDataset, shard_paths

    splits = RC.video_splits(cfg)
    q_samples = RC.build_task(cfg, task, splits["query"])
    q_hashes = phash_set([p for p, _ in q_samples])

    g_hashes = set()
    if shard_paths(os.path.expanduser(str(cfg.paths.gastronet))):
        ds = GastroNetDataset(cfg, transform=lambda im: im, subset=ssl_sample,
                              seed=int(cfg.seed))
        # iterate a bounded number of decoded PILs
        import imagehash

        for i in range(min(len(ds), ssl_sample)):
            try:
                g_hashes.add(str(imagehash.phash(ds[i])))
            except Exception:
                continue

    overlap = sorted(q_hashes & g_hashes)
    rep = {"n_query_hashes": len(q_hashes), "n_ssl_hashes": len(g_hashes),
           "overlap_count": len(overlap), "overlap_examples": overlap[:20]}
    out = os.path.join(os.path.expanduser(str(cfg.paths.results)), "leakage_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(rep, open(out, "w"), indent=2)
    print(f"[leakage] query={rep['n_query_hashes']} ssl={rep['n_ssl_hashes']} "
          f"overlap={rep['overlap_count']} -> {out}")
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--ssl-sample", type=int, default=20000)
    a = ap.parse_args()
    audit(load_cfg(a.config), ssl_sample=a.ssl_sample)


if __name__ == "__main__":
    main()
