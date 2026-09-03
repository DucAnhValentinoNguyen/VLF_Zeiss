"""REAL-Colon loader + the two binary classification tasks.

Expected on-disk layout (after the Figshare download via the cosmoimd helper):

    <root>/
      video_info.csv                 # unique_video_name, fps, num_frames, num_lesions, ...
      lesion_info.csv                # unique_object_id, unique_video_name, size [mm], site,
                                     #   histology_extended, histology_class {AD,HP,SSL,TSA,NO POLYP,OTHER}
      {SSS}-{VVV}_frames/            # {SSS}-{VVV}_{frameidx}.jpg
      {SSS}-{VVV}_annotations/       # {SSS}-{VVV}_{frameidx}.xml  (Pascal VOC; negatives may be absent)

Tasks
  rc_frame  : frame-level polyp vs no-polyp (>=1 box -> positive), clean frames
              subsampled to `neg_per_pos` x positives, taking every `frame_stride`-th frame.
  rc_lesion : per-lesion box crops -> adenoma (AD) vs non-adenoma (HP/SSL/TSA); crops
              written once under <cache>/rc_lesion_crops/.

All splits are BY VIDEO (query / reference / cal) — see video_splits().
"""
from __future__ import annotations

import json
import os
import random
import re
import xml.etree.ElementTree as ET
from glob import glob

from .folds import grouped_stratified_split

_FRAME_EXTS = (".jpg", ".jpeg", ".png")
_VIDRE = re.compile(r"^(\d{3})-(\d{3})$")


# --------------------------------------------------------------------- discovery
def _root(cfg) -> str:
    return os.path.abspath(os.path.expanduser(str(cfg.paths.realcolon)))


def discover_videos(root: str) -> list[str]:
    vids = set()
    for d in glob(os.path.join(root, "*_frames")):
        name = os.path.basename(d)[: -len("_frames")]
        if _VIDRE.match(name):
            vids.add(name)
    if not vids:  # fall back to the csv
        import pandas as pd

        p = os.path.join(root, "video_info.csv")
        if os.path.exists(p):
            vids = set(pd.read_csv(p)["unique_video_name"].astype(str))
    return sorted(vids)


def frames_dir(root: str, vid: str) -> str:
    return os.path.join(root, f"{vid}_frames")


def annos_dir(root: str, vid: str) -> str:
    return os.path.join(root, f"{vid}_annotations")


def list_frames(root: str, vid: str) -> list[tuple[int, str]]:
    out = []
    for p in glob(os.path.join(frames_dir(root, vid), "*")):
        if p.lower().endswith(_FRAME_EXTS):
            m = re.search(r"_(\d+)\.[a-zA-Z]+$", os.path.basename(p))
            if m:
                out.append((int(m.group(1)), p))
    return sorted(out)


def _anno_path(root: str, vid: str, frame_idx: int, frame_path: str) -> str | None:
    stem = os.path.splitext(os.path.basename(frame_path))[0]
    cand = os.path.join(annos_dir(root, vid), stem + ".xml")
    return cand if os.path.exists(cand) else None


def parse_voc(xml_path: str) -> dict:
    """-> {'width':int,'height':int,'objects':[(name,x0,y0,x1,y1), ...]}"""
    r = ET.parse(xml_path).getroot()
    size = r.find("size")
    w = int(float(size.findtext("width"))) if size is not None else 0
    h = int(float(size.findtext("height"))) if size is not None else 0
    objs = []
    for o in r.findall("object"):
        name = (o.findtext("name") or "").strip()
        b = o.find("bndbox")
        if b is None:
            continue
        x0 = float(b.findtext("xmin")); y0 = float(b.findtext("ymin"))
        x1 = float(b.findtext("xmax")); y1 = float(b.findtext("ymax"))
        objs.append((name, x0, y0, x1, y1))
    return {"width": w, "height": h, "objects": objs}


def frame_boxes(root: str, vid: str, frame_idx: int, frame_path: str) -> list[tuple]:
    ap = _anno_path(root, vid, frame_idx, frame_path)
    if ap is None:
        return []
    try:
        return parse_voc(ap)["objects"]
    except Exception:
        return []


# --------------------------------------------------------------------- metadata
def load_lesion_info(root: str):
    import pandas as pd

    df = pd.read_csv(os.path.join(root, "lesion_info.csv"))
    df.columns = [c.strip() for c in df.columns]
    hc = [c for c in df.columns if c.lower().startswith("histology_class")]
    if hc:
        df["histology_class"] = df[hc[0]].astype(str).str.strip().str.upper()
    df["unique_object_id"] = df["unique_object_id"].astype(str).str.strip()
    df["unique_video_name"] = df["unique_video_name"].astype(str).str.strip()
    return df


def load_video_info(root: str):
    import pandas as pd

    p = os.path.join(root, "video_info.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    df["unique_video_name"] = df["unique_video_name"].astype(str).str.strip()
    return df


# ----------------------------------------------------------------------- splits
def _prevalence_bin(root: str, vid: str, vinfo) -> int:
    if vinfo is not None and "num_lesions" in vinfo.columns:
        row = vinfo.loc[vinfo["unique_video_name"] == vid, "num_lesions"]
        if len(row):
            return 1 if float(row.iloc[0]) > 0 else 0
    return 0


def video_splits(cfg, *, force: bool = False) -> dict[str, list[str]]:
    root = _root(cfg)
    cache = os.path.abspath(os.path.expanduser(str(cfg.paths.cache)))
    os.makedirs(cache, exist_ok=True)
    fp = os.path.join(cache, "realcolon_splits.json")
    if os.path.exists(fp) and not force:
        return json.load(open(fp))

    vids = discover_videos(root)
    if not vids:
        raise FileNotFoundError(f"no REAL-Colon videos under {root}")
    vinfo = load_video_info(root)
    groups, strata = [], []
    for v in vids:
        groups.append(v)
        study = v.split("-")[0]
        strata.append(f"{study}|{_prevalence_bin(root, v, vinfo)}")

    sp = cfg.realcolon_tasks.split
    sizes = {"query": int(sp.n_query), "reference": int(sp.n_reference), "cal": int(sp.n_cal)}
    parts = grouped_stratified_split(groups, strata, sizes, seed=int(sp.seed))
    json.dump(parts, open(fp, "w"), indent=2)
    print(f"[realcolon] splits -> {fp}  " +
          " ".join(f"{k}={len(v)}" for k, v in parts.items()))
    return parts


# ------------------------------------------------------------------- task: frame
def build_frame_task(cfg, split_videos: list[str]) -> list[tuple[str, int]]:
    root = _root(cfg)
    tc = cfg.realcolon_tasks.frame
    stride = int(tc.frame_stride)
    neg_per_pos = int(tc.neg_per_pos)
    min_area = float(tc.min_box_area_frac)
    rng = random.Random(int(cfg.realcolon_tasks.split.seed))

    pos, neg = [], []
    for vid in split_videos:
        frames = list_frames(root, vid)[::stride]
        for fidx, fpath in frames:
            boxes = frame_boxes(root, vid, fidx, fpath)
            if min_area > 0 and boxes:
                fb = parse_voc(_anno_path(root, vid, fidx, fpath))
                area = max(
                    ((x1 - x0) * (y1 - y0) for _, x0, y0, x1, y1 in boxes), default=0.0
                )
                frame_area = max(1.0, fb["width"] * fb["height"])
                boxes = boxes if (area / frame_area) >= min_area else []
            (pos if boxes else neg).append((fpath, 1 if boxes else 0))

    rng.shuffle(neg)
    neg = neg[: neg_per_pos * max(1, len(pos))]
    samples = pos + neg
    rng.shuffle(samples)
    print(f"[realcolon:rc_frame] {len(split_videos)} videos -> "
          f"{len(pos)} pos / {len(neg)} neg = {len(samples)}")
    return samples


# ------------------------------------------------------------------ task: lesion
def _label_for_histology(cfg, hc: str) -> int | None:
    lc = cfg.realcolon_tasks.lesion
    if hc in set(lc.adenoma):
        return 1
    if hc in set(lc.non_adenoma):
        return 0
    return None


def _crop_and_save(frame_path: str, box, margin: float, out_path: str) -> bool:
    from PIL import Image

    try:
        im = Image.open(frame_path).convert("RGB")
    except Exception:
        return False
    W, H = im.size
    _, x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0, int(x0 - margin * bw)); y0 = max(0, int(y0 - margin * bh))
    x1 = min(W, int(x1 + margin * bw)); y1 = min(H, int(y1 + margin * bh))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return False
    im.crop((x0, y0, x1, y1)).save(out_path, quality=92)
    return True


def build_lesion_crop_task(cfg, split_videos: list[str]) -> list[tuple[str, int]]:
    root = _root(cfg)
    lc = cfg.realcolon_tasks.lesion
    margin = float(lc.crop_margin)
    cap = int(lc.max_frames_per_lesion)
    rng = random.Random(int(cfg.realcolon_tasks.split.seed))
    out_dir = os.path.join(
        os.path.abspath(os.path.expanduser(str(cfg.paths.cache))), "rc_lesion_crops"
    )
    os.makedirs(out_dir, exist_ok=True)

    linfo = load_lesion_info(root)
    linfo = linfo[linfo["unique_video_name"].isin(split_videos)]
    # lesion id -> label
    id2label = {}
    for _, r in linfo.iterrows():
        lab = _label_for_histology(cfg, str(r["histology_class"]))
        if lab is not None:
            id2label[str(r["unique_object_id"])] = lab

    def _match(name: str) -> int | None:
        if name in id2label:
            return id2label[name]
        for k, v in id2label.items():          # tolerate '001-001_1' vs '1'
            if k.endswith(name) or name.endswith(k) or k.split("_")[-1] == name:
                return v
        return None

    # gather candidate (lesion_id, frame_path, box)
    per_lesion: dict[str, list[tuple]] = {}
    for vid in split_videos:
        for fidx, fpath in list_frames(root, vid):
            for box in frame_boxes(root, vid, fidx, fpath):
                lab = _match(box[0])
                if lab is None:
                    continue
                per_lesion.setdefault(f"{vid}:{box[0]}:{lab}", []).append((fpath, box))

    samples = []
    for key, lst in per_lesion.items():
        vid, name, lab = key.split(":")
        rng.shuffle(lst)
        for j, (fpath, box) in enumerate(lst[:cap]):
            op = os.path.join(out_dir, f"{vid}_{name}_{j:03d}.jpg")
            if os.path.exists(op) or _crop_and_save(fpath, box, margin, op):
                samples.append((op, int(lab)))
    rng.shuffle(samples)
    npos = sum(l for _, l in samples)
    print(f"[realcolon:rc_lesion] {len(split_videos)} videos -> {len(per_lesion)} lesions, "
          f"{len(samples)} crops ({npos} adenoma / {len(samples) - npos} non-adenoma)")
    return samples


TASKS = {"rc_frame": build_frame_task, "rc_lesion": build_lesion_crop_task}
N_CLASSES = {"rc_frame": 2, "rc_lesion": 2}


def build_task(cfg, task: str, split_videos: list[str]) -> list[tuple[str, int]]:
    return TASKS[task](cfg, split_videos)


# --------------------------------------------------------------------------- CLI
def _inspect(cfg):
    root = _root(cfg)
    vids = discover_videos(root)
    print(f"root={root}\nvideos={len(vids)}  e.g. {vids[:3]}")
    if vids:
        fr = list_frames(root, vids[0])
        print(f"{vids[0]}: {len(fr)} frames; first={fr[0][1] if fr else None}")
        if fr:
            print("boxes on first frame:", frame_boxes(root, vids[0], *fr[0]))
    try:
        li = load_lesion_info(root)
        print("lesion_info rows:", len(li),
              "| histology_class counts:\n", li["histology_class"].value_counts().to_dict())
    except Exception as e:  # noqa: BLE001
        print("lesion_info: ", repr(e))
    print("splits:", video_splits(cfg))


def main():
    import argparse

    from ..cfg import load_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--make-splits", action="store_true")
    ap.add_argument("--build", choices=list(TASKS), default=None)
    ap.add_argument("--split", default="reference", choices=["query", "reference", "cal"])
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.make_splits:
        video_splits(cfg, force=True)
    if a.inspect:
        _inspect(cfg)
    if a.build:
        vids = video_splits(cfg)[a.split]
        s = build_task(cfg, a.build, vids)
        print(f"{a.build}/{a.split}: {len(s)} samples; head={s[:3]}")


if __name__ == "__main__":
    main()
