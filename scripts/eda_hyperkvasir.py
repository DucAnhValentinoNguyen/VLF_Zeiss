"""Exact EDA of the HyperKvasir evaluation data (every image, no sampling).

Writes $OUT_ROOT/eda/hkv_eda.json: inventory, integrity (corrupt/truncated), image sizes,
duplicates (exact bytes and perceptual hash), cross-split leakage and what the de-dup
removed, class balance per task and split, and polyp-mask statistics.
"""
import collections
import hashlib
import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageFile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_common import redundancy  # noqa: E402
from vlfz.cfg import load_cfg  # noqa: E402
from vlfz.data import hyperkvasir as hk  # noqa: E402


def decode_status(path):
    """ok | truncated (strict decode fails, tolerant succeeds) | unreadable."""
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as im:
            im.load()
            return "ok", im.size, im.mode
    except Exception:
        pass
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        with Image.open(path) as im:
            im.load()
            return "truncated", im.size, im.mode
    except Exception:
        return "unreadable", None, None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def balance(counts):
    c = np.asarray(counts, dtype=float)
    p = c / c.sum()
    h = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    k = len(c)
    return {"n": int(c.sum()), "k": k, "majority_share": float(p.max()), "minority_share": float(p.min()),
            "imbalance_ratio": float(c.max() / max(c.min(), 1)), "norm_entropy": h / math.log(k) if k > 1 else 1.0,
            "effective_classes": float(math.exp(h)), "chance_balanced_acc": 1.0 / k}


def main():
    cfg = load_cfg()
    out_dir = os.path.join(os.path.abspath(os.path.expanduser(str(cfg.paths.out_root))), "eda")
    os.makedirs(out_dir, exist_ok=True)
    res = {}

    # ---------------- inventory
    table = hk.load_label_table(cfg)
    imgs = hk.scan_labeled_images(cfg)
    stems = [os.path.splitext(os.path.basename(p))[0] for p, _ in imgs]
    res["inventory"] = {"csv_rows": len(table), "labelled_images_found": len(imgs),
                        "min_class": int(cfg.hyperkvasir.min_class)}

    # ---------------- integrity, sizes, exact byte duplicates
    status = collections.Counter()
    sizes = collections.Counter()
    modes = collections.Counter()
    md5s = collections.defaultdict(list)
    bad = []
    for i, (p, _) in enumerate(imgs):
        st, size, mode = decode_status(p)
        status[st] += 1
        if st != "ok":
            bad.append(os.path.basename(p))
        if size:
            sizes[f"{size[0]}x{size[1]}"] += 1
            modes[mode] += 1
        with open(p, "rb") as f:
            md5s[hashlib.md5(f.read()).hexdigest()].append(stems[i])
        if (i + 1) % 2000 == 0:
            print(f"[eda-hkv] integrity {i + 1}/{len(imgs)}", flush=True)
    exact_groups = [v for v in md5s.values() if len(v) > 1]
    res["integrity"] = {"checked": len(imgs), **dict(status), "bad_files": bad[:20],
                        "exact_duplicate_groups": len(exact_groups),
                        "exact_duplicate_redundant_images": int(sum(len(v) - 1 for v in exact_groups)),
                        "n_distinct_resolutions": len(sizes),
                        "top_resolutions": sizes.most_common(12), "modes": dict(modes)}

    # ---------------- perceptual-hash duplicates (whole labelled set)
    ph = hk._phash_table(cfg)
    hashed = [s for s in stems if s in ph]
    res["phash"] = {"hashed": len(hashed), "missing_hash": len(stems) - len(hashed),
                    "dataset": redundancy(np.array([ph[s] for s in hashed], dtype=np.uint64))}

    # ---------------- cross-split leakage and what the de-dup removed
    import random
    hp = cfg.hyperkvasir.split
    by_cls = collections.defaultdict(list)
    for p, lab in imgs:
        by_cls[lab["finding"]].append(os.path.splitext(os.path.basename(p))[0])
    rng = random.Random(int(hp.seed))
    raw = {"reference": [], "cal": [], "query": []}
    for cls, ss in by_cls.items():
        rng.shuffle(ss)
        n = len(ss)
        n_ref = max(1, int(round(float(hp.ref_frac) * n)))
        n_cal = int(round(float(hp.cal_frac) * n))
        raw["reference"] += ss[:n_ref]
        raw["cal"] += ss[n_ref:n_ref + n_cal]
        raw["query"] += ss[n_ref + n_cal:]
    raw = {k: sorted(v) for k, v in raw.items()}
    dd = hk.global_split(cfg)  # the deduplicated split actually used (cached dd6)
    stem2find = {s: lab["finding"] for s, (p, lab) in zip(stems, imgs)}
    leak = {"raw_sizes": {k: len(v) for k, v in raw.items()}, "dedup_sizes": {k: len(v) for k, v in dd.items()}}
    dropped = {}
    for split in ("cal", "query"):
        kept = set(dd[split])
        dropped[split] = [s for s in raw[split] if s not in kept]
        leak[f"dropped_{split}"] = len(dropped[split])
        leak[f"dropped_{split}_pct"] = 100.0 * len(dropped[split]) / len(raw[split])
    per_cls = collections.OrderedDict()
    tot = collections.Counter(stem2find[s] for s in raw["cal"] + raw["query"])
    drp = collections.Counter(stem2find[s] for s in dropped["cal"] + dropped["query"])
    for c in sorted(tot):
        per_cls[c] = {"eligible": tot[c], "dropped": drp.get(c, 0)}
    leak["per_finding_class"] = per_cls
    # residual redundancy inside the final scoring splits (not removed by the cross-split filter)
    final_eval = [s for s in dd["cal"] + dd["query"] if s in ph]
    leak["residual_within_eval"] = redundancy(np.array([ph[s] for s in final_eval], dtype=np.uint64))
    res["leakage"] = leak

    # ---------------- class balance per task and split
    tasks = {}
    for task in hk.CLS_TASKS:
        classes = hk.canonical_classes(cfg, task)
        allc = collections.Counter(hk._task_label(lab, task) for _, lab in imgs)
        entry = {"classes": classes, "all": [int(allc.get(c, 0)) for c in classes],
                 "excluded_classes": {c: int(n) for c, n in allc.items() if c not in classes}}
        for split in ("reference", "cal", "query"):
            ys = [y for _, y in hk.build_task(cfg, task, split)]
            cc = collections.Counter(ys)
            entry[split] = [int(cc.get(i, 0)) for i in range(len(classes))]
        entry["balance_all"] = balance(entry["all"])
        entry["balance_query"] = balance(entry["query"])
        entry["majority_baseline_acc_query"] = max(entry["query"]) / sum(entry["query"])
        tasks[task] = entry
    full_f = collections.Counter(lab["finding"] for _, lab in imgs)
    res["findings_all23"] = dict(sorted(full_f.items(), key=lambda kv: kv[1]))
    res["tasks"] = tasks

    # ---------------- segmentation set
    d = hk.segmented_dir(cfg)
    pairs = hk.seg_pairs(cfg, "support") + hk.seg_pairs(cfg, "query")
    fg, empty, mismatch, sstat = [], 0, 0, collections.Counter()
    seg_hashes = []
    import imagehash
    for ip, mp in pairs:
        s1, sz1, _ = decode_status(ip)
        s2, sz2, _ = decode_status(mp)
        sstat[f"img_{s1}"] += 1
        sstat[f"mask_{s2}"] += 1
        if s1 == "unreadable" or s2 == "unreadable":
            continue
        if sz1 != sz2:
            mismatch += 1
        m = np.asarray(Image.open(mp).convert("L")) > 127
        f = float(m.mean())
        fg.append(f)
        empty += int(f == 0)
        seg_hashes.append(int(str(imagehash.phash(Image.open(ip).convert("RGB"))), 16))
    fg = np.array(fg)
    edges = [0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 1.01]
    res["segmentation"] = {"pairs": len(pairs), "support": 100, "query": len(pairs) - 100,
                           "integrity": dict(sstat), "size_mismatch": mismatch, "empty_masks": empty,
                           "fg_fraction_quantiles": {q: float(np.quantile(fg, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                           "fg_fraction_mean": float(fg.mean()),
                           "fg_hist_edges": edges, "fg_hist": np.histogram(fg, bins=edges)[0].tolist(),
                           "phash": redundancy(np.array(seg_hashes, dtype=np.uint64))}

    fp = os.path.join(out_dir, "hkv_eda.json")
    json.dump(res, open(fp, "w"), indent=1)
    print("[eda-hkv] wrote", fp)


if __name__ == "__main__":
    main()
