"""Choose which curated WebDataset shards to stage onto LRZ.

Default: every shard (the user wants the full corpus). Options narrow it:
  --global-dedup     keep one image per perceptual hash across the WHOLE corpus
                     (curation only dedups within a shard_group); a shard is kept
                     only if it still holds >= --min-keep-frac of its images.
  --max-images N      stratified cap across shard_groups.
Writes a newline-separated list of ``.tar`` basenames to ``--out`` (default
``shard_list.txt``); ``stage_in.sh`` syncs exactly those. With no narrowing it
lists every shard.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import list_keys, s3  # noqa: E402


def _load_curated(bucket: str, workdir: str) -> pd.DataFrame:
    os.makedirs(workdir, exist_ok=True)
    parts = []
    for k in list_keys(bucket, "catalog/curated/", ".parquet"):
        local = os.path.join(workdir, k.replace("/", "_"))
        s3().download_file(bucket, k, local)
        parts.append(pd.read_parquet(local))
    if not parts:
        sys.exit("no catalog/curated/ parquet — run curate first")
    df = pd.concat(parts, ignore_index=True)
    return df[df["wds_shard"] != ""].copy()  # kept images only


def _group_key(shard: pd.Series) -> pd.Series:
    return shard.str.slice(0, 4)  # "g000-0001.tar" -> "g000"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--workdir", default="/tmp/stage_sel")
    ap.add_argument("--out", default="shard_list.txt")
    ap.add_argument("--global-dedup", action="store_true")
    ap.add_argument("--min-keep-frac", type=float, default=0.5)
    ap.add_argument("--max-images", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    full = _load_curated(args.bucket, args.workdir)
    orig_counts = full["wds_shard"].value_counts()
    df = full
    n0 = len(df)

    if args.global_dedup:
        df = df.sort_values("key").drop_duplicates("phash", keep="first")
        print(f"[select] global phash dedup: {n0:,} -> {len(df):,}")

    if args.max_images and args.max_images < len(df):
        n_groups = _group_key(df["wds_shard"]).nunique()
        per = max(1, args.max_images // n_groups)
        df = (df.groupby(_group_key(df["wds_shard"]), group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per), random_state=args.seed)))
        if len(df) > args.max_images:
            df = df.sample(args.max_images, random_state=args.seed)
        print(f"[select] capped to {len(df):,} images")

    kept_counts = df["wds_shard"].value_counts()
    if args.global_dedup or args.max_images:
        keep = sorted(s for s, n in kept_counts.items()
                      if n >= args.min_keep_frac * orig_counts.get(s, n))
    else:
        keep = sorted(orig_counts.index)

    open(args.out, "w").write("\n".join(keep) + "\n")
    est_gb = df.loc[df["wds_shard"].isin(keep), "bytes"].sum() / 2**30
    print(f"[select] {len(keep):,}/{len(orig_counts):,} shards -> {args.out}  "
          f"(~{est_gb:.1f} GiB, egress ~${est_gb * 0.09:.2f})")


if __name__ == "__main__":
    main()
