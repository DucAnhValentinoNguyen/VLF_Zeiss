"""HyperKvasir loader + every downstream classification task, plus zero-shot
polyp segmentation.

Staged layout (read-only):
    <root>/hyper_kvasir_labeled_images/
        image-labels.csv                       # Video file,Organ,Finding,Classification
        {upper,lower}-gi-tract/<category>/<class>/<stem>.jpg
    <root>/hyper_kvasir_unlabeled_images/images/*.jpg      # 99k (SSL corpus fallback)
    <root>/hyper_kvasir_segmented_images/{images,masks}/*.jpg + bounding-boxes.json

Classification tasks (all from the same labelled images, one shared stratified
split so pre/post and cross-task numbers use identical images):
    hkv_findings   23-way  Finding            (rare classes < min_class dropped)
    hkv_category    4-way  Classification     (anatomical / pathological / therapeutic / quality)
    hkv_tract       2-way  Organ              (upper vs lower GI)
    hkv_pathology   2-way  pathological-findings vs everything else

Segmentation task:
    hkv_seg        zero-shot polyp segmentation (dense patch features + k-NN label
                   transfer); handled by vlfz.eval.seg, not the classification path.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from glob import glob

CLS_TASKS = ("hkv_findings", "hkv_category", "hkv_tract", "hkv_pathology")
SEG_TASKS = ("hkv_seg",)
TASKS = CLS_TASKS + SEG_TASKS


def _root(cfg) -> str:
    return os.path.abspath(os.path.expanduser(str(cfg.paths.hyperkvasir)))


def labeled_dir(cfg) -> str:
    return os.path.join(_root(cfg), "hyper_kvasir_labeled_images")


def unlabeled_dir(cfg) -> str:
    return os.path.join(_root(cfg), "hyper_kvasir_unlabeled_images")


def segmented_dir(cfg) -> str:
    return os.path.join(_root(cfg), "hyper_kvasir_segmented_images")


# ------------------------------------------------------------------- label table
def _norm(s: str) -> str:
    return str(s).strip().lower().replace(" ", "-")


def load_label_table(cfg) -> dict[str, dict]:
    """stem -> {'tract','category','finding'} from image-labels.csv."""
    import csv

    p = os.path.join(labeled_dir(cfg), "image-labels.csv")
    out: dict[str, dict] = {}
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            stem = row["Video file"].strip()
            out[stem] = {
                "tract": _norm(row["Organ"]).replace("-gi", "-gi"),  # 'lower-gi' / 'upper-gi'
                "category": _norm(row["Classification"]),
                "finding": _norm(row["Finding"]),
            }
    return out


_SCAN_CACHE: dict[str, list] = {}


def scan_labeled_images(cfg) -> list[tuple[str, dict]]:
    """[(image_path, label_dict)] for every labelled image found on disk (memoised)."""
    root = labeled_dir(cfg)
    if root in _SCAN_CACHE:
        return _SCAN_CACHE[root]
    labels = load_label_table(cfg)
    out = []
    for p in glob(os.path.join(root, "**", "*.jpg"), recursive=True):
        stem = os.path.splitext(os.path.basename(p))[0]
        lab = labels.get(stem)
        if lab is None:  # fall back to the folder path
            parts = os.path.relpath(p, root).split(os.sep)
            if len(parts) >= 4:
                lab = {"tract": _norm(parts[0]).replace("-tract", ""),
                       "category": _norm(parts[1]), "finding": _norm(parts[2])}
            else:
                continue
        out.append((p, lab))
    _SCAN_CACHE[root] = out
    return out


# ------------------------------------------------------------------------ splits
def _task_label(lab: dict, task: str):
    if task == "hkv_findings":
        return lab["finding"]
    if task == "hkv_category":
        return lab["category"]
    if task == "hkv_tract":
        return "upper" if lab["tract"].startswith("upper") else "lower"
    if task == "hkv_pathology":
        return "pathological" if lab["category"] == "pathological-findings" else "other"
    raise KeyError(task)


def global_split(cfg, *, force: bool = False) -> dict[str, list[str]]:
    """One stratified (on 23-way finding) image-level 3-way split, cached.
    NOTE: HyperKvasir's image-labels.csv has no patient/procedure id, so this is
    image-level; near-duplicate procedure frames may span splits (recorded as a
    caveat in every results JSON)."""
    cache = os.path.abspath(os.path.expanduser(str(cfg.paths.cache)))
    os.makedirs(cache, exist_ok=True)
    fp = os.path.join(cache, "hkv_splits.json")
    if os.path.exists(fp) and not force:
        return json.load(open(fp))

    hp = cfg.hyperkvasir.split
    r_ref, r_cal = float(hp.ref_frac), float(hp.cal_frac)
    rng = random.Random(int(hp.seed))
    by_cls: dict[str, list[str]] = defaultdict(list)
    for p, lab in scan_labeled_images(cfg):
        by_cls[lab["finding"]].append(os.path.splitext(os.path.basename(p))[0])

    parts = {"reference": [], "cal": [], "query": []}
    for cls, stems in by_cls.items():
        rng.shuffle(stems)
        n = len(stems)
        n_ref = max(1, int(round(r_ref * n)))
        n_cal = int(round(r_cal * n))
        parts["reference"] += stems[:n_ref]
        parts["cal"] += stems[n_ref:n_ref + n_cal]
        parts["query"] += stems[n_ref + n_cal:]
    parts = {k: sorted(v) for k, v in parts.items()}
    json.dump(parts, open(fp, "w"), indent=2)
    print(f"[hkv] split -> {fp}  " + " ".join(f"{k}={len(v)}" for k, v in parts.items()))
    return parts


def canonical_classes(cfg, task: str, *, force: bool = False) -> list[str]:
    """The task's label set + ordering, derived ONCE from the global (all-splits)
    label distribution and cached. Must be split-independent: `min_class` is
    applied to global counts, not per-split, or the same string label would map
    to different integers in reference vs query (silently wrecks k-NN)."""
    if task not in CLS_TASKS:
        raise ValueError(f"{task} is not a classification task")
    cache = os.path.abspath(os.path.expanduser(str(cfg.paths.cache)))
    os.makedirs(cache, exist_ok=True)
    fp = os.path.join(cache, f"hkv_classes_{task}.json")
    if os.path.exists(fp) and not force:
        return json.load(open(fp))

    min_class = int(cfg.hyperkvasir.min_class) if task == "hkv_findings" else 0
    counts: dict[str, int] = defaultdict(int)
    for _, lab in scan_labeled_images(cfg):
        counts[_task_label(lab, task)] += 1
    classes = sorted(c for c, n in counts.items() if n >= max(1, min_class))
    json.dump(classes, open(fp, "w"))
    return classes


def build_task(cfg, task: str, split: str) -> list[tuple[str, int]]:
    """[(image_path, int_label)] for one classification task + split."""
    if task not in CLS_TASKS:
        raise ValueError(f"{task} is not a classification task")
    want = set(global_split(cfg)[split])
    per_class = int(getattr(cfg.hyperkvasir, "subset_per_class", 0) or 0)

    classes = canonical_classes(cfg, task)
    c2i = {c: i for i, c in enumerate(classes)}

    rows = []
    for p, lab in scan_labeled_images(cfg):
        stem = os.path.splitext(os.path.basename(p))[0]
        y = _task_label(lab, task)
        if stem in want and y in c2i:
            rows.append((p, y))

    if per_class:                                   # smoke: cap images per class
        seen: dict = defaultdict(int)
        capped = []
        for p, y in sorted(rows):
            if seen[y] < per_class:
                capped.append((p, y)); seen[y] += 1
        rows = capped

    samples = [(p, c2i[y]) for p, y in rows]
    print(f"[hkv:{task}/{split}] {len(samples)} imgs, {len(classes)} classes: {classes}")
    return samples


def n_classes(cfg, task: str) -> int:
    return len(canonical_classes(cfg, task))


# --------------------------------------------------------------- segmentation io
def seg_pairs(cfg, split: str) -> list[tuple[str, str]]:
    """[(image_path, mask_path)] split into support/query (seeded)."""
    d = segmented_dir(cfg)
    imgs = sorted(glob(os.path.join(d, "images", "*.jpg")))
    pairs = []
    for ip in imgs:
        mp = os.path.join(d, "masks", os.path.basename(ip))
        if os.path.exists(mp):
            pairs.append((ip, mp))
    rng = random.Random(int(cfg.hyperkvasir.split.seed))
    rng.shuffle(pairs)
    n_sup = int(cfg.hyperkvasir.seg.n_support)
    return pairs[:n_sup] if split == "support" else pairs[n_sup:]


# --------------------------------------------------------------------- unlabeled
def unlabeled_paths(cfg, cap: int = 0) -> list[str]:
    d = unlabeled_dir(cfg)
    paths = [p for p in glob(os.path.join(d, "**", "*"), recursive=True)
             if p.lower().endswith((".jpg", ".jpeg", ".png"))]
    paths.sort()
    return paths[:cap] if cap else paths


# --------------------------------------------------------------------------- CLI
def _inspect(cfg):
    imgs = scan_labeled_images(cfg)
    print(f"labeled images on disk: {len(imgs)}")
    for t in CLS_TASKS:
        cnt = defaultdict(int)
        for _, lab in imgs:
            cnt[_task_label(lab, t)] += 1
        print(f"  {t}: {dict(sorted(cnt.items(), key=lambda x: -x[1]))}")
    sp = global_split(cfg)
    print("split:", {k: len(v) for k, v in sp.items()})
    print("seg pairs:", len(seg_pairs(cfg, "support")) + len(seg_pairs(cfg, "query")))
    print("unlabeled:", len(unlabeled_paths(cfg, cap=1)) and "present")


def main():
    import argparse

    from ..cfg import load_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--make-splits", action="store_true")
    ap.add_argument("--build", choices=list(CLS_TASKS), default=None)
    ap.add_argument("--split", default="reference", choices=["reference", "cal", "query"])
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.make_splits:
        global_split(cfg, force=True)
    if a.inspect:
        _inspect(cfg)
    if a.build:
        s = build_task(cfg, a.build, a.split)
        print(f"{a.build}/{a.split}: {len(s)} samples; head={s[:3]}")


if __name__ == "__main__":
    main()
