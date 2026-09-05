"""Data-quality report over the Parquet catalogs.

Pulls ``catalog/manifest/`` (+ ``catalog/curated/`` if present) from S3, and
writes ``dq/data_report.md`` plus two PNGs:
  * raw image count vs the expected ~4.82M, corrupt rate, duplicate-cluster rate,
  * short-side / aspect-ratio histograms, colour-mode split,
  * curated tier: kept / dropped, shard count, approx on-disk size, est. egress.
Non-zero corrupt or a wild count is a stop-the-line signal before staging.
"""
from __future__ import annotations

import argparse
import io
import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import list_keys, s3, upload_file  # noqa: E402

EXPECTED_RAW = 4_820_653


def _read_parquet_prefix(bucket: str, prefix: str, workdir: str) -> pd.DataFrame:
    parts = []
    for k in list_keys(bucket, prefix, ".parquet"):
        local = os.path.join(workdir, k.replace("/", "_"))
        s3().download_file(bucket, k, local)
        parts.append(pd.read_parquet(local))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _hist(series, title, xlabel, path, bins=50, logy=True):
    plt.figure(figsize=(6, 3.2))
    plt.hist(series.dropna(), bins=bins)
    if logy:
        plt.yscale("log")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--workdir", default="/mnt/work/dq")
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    m = _read_parquet_prefix(args.bucket, "catalog/manifest/", args.workdir)
    if m.empty:
        sys.exit("no manifest parquet found — run catalog/build_manifest.py first")
    c = _read_parquet_prefix(args.bucket, "catalog/curated/", args.workdir)

    n = len(m)
    n_corrupt = int(m["corrupt"].sum())
    dup_groups = m.loc[m["phash"] != "", "phash"].value_counts()
    dup_clusters = int((dup_groups > 1).sum())
    dup_extra = int((dup_groups[dup_groups > 1] - 1).sum())
    short_side = m[["width", "height"]].min(axis=1)
    aspect = (m[["width", "height"]].max(axis=1) / m[["width", "height"]].min(axis=1)).replace([float("inf")], pd.NA)

    p_short = os.path.join(args.workdir, "hist_short_side.png")
    p_aspect = os.path.join(args.workdir, "hist_aspect.png")
    _hist(short_side[short_side > 0], "Raw image short side (px)", "px", p_short)
    _hist(aspect, "Raw image aspect ratio (long/short)", "ratio", p_aspect, bins=40)

    lines = [
        "# GastroNet-5M — data-quality report", "",
        f"- raw images catalogued: **{n:,}**  (expected ~{EXPECTED_RAW:,}; "
        f"delta {n - EXPECTED_RAW:+,})",
        f"- corrupt / undecodable: **{n_corrupt:,}**  ({100 * n_corrupt / n:.3f} %)",
        f"- perceptual-hash duplicate clusters: **{dup_clusters:,}**  "
        f"(**{dup_extra:,}** redundant images, {100 * dup_extra / n:.1f} %)",
        f"- colour mode: " + ", ".join(f"{k}={v:,}" for k, v in m['mode'].value_counts().items()),
        f"- short side px  min/median/max: "
        f"{int(short_side[short_side>0].min())} / {int(short_side.median())} / {int(short_side.max())}",
        "", "![short side](hist_short_side.png)", "", "![aspect](hist_aspect.png)", "",
    ]
    if not c.empty:
        kept = int((c["dup_of"] == "").sum())
        d_dup = int(((c["dup_of"] != "") & (c["dup_of"] != "CORRUPT")).sum())
        d_bad = int((c["dup_of"] == "CORRUPT").sum())
        bytes_kept = int(c.loc[c["dup_of"] == "", "bytes"].sum())
        n_tars = c.loc[c["wds_shard"] != "", "wds_shard"].nunique()
        lines += [
            "## Curated tier", "",
            f"- kept: **{kept:,}**  ({100 * kept / n:.1f} % of raw)",
            f"- dropped — duplicates {d_dup:,}, corrupt {d_bad:,}",
            f"- WebDataset shards: **{n_tars:,}**  (~{bytes_kept / 2**30:.1f} GiB on disk)",
            f"- one-time stage-in egress to LRZ ≈ **${bytes_kept / 2**30 * 0.09:.2f}** "
            f"(S3 eu-north-1 → internet @ $0.09/GB)", "",
        ]
    else:
        lines += ["## Curated tier", "", "_not built yet_", ""]

    verdict = "PASS" if (n_corrupt / n < 1e-3 and abs(n - EXPECTED_RAW) < 0.1 * EXPECTED_RAW) else "REVIEW"
    lines += [f"**verdict: {verdict}**", ""]

    report = os.path.join(args.workdir, "data_report.md")
    open(report, "w").write("\n".join(lines))
    print("\n".join(lines))

    if args.upload:
        for p in (report, p_short, p_aspect):
            upload_file(args.bucket, f"dq/{os.path.basename(p)}", p)
        print(f"[dq] -> s3://{args.bucket}/dq/")


if __name__ == "__main__":
    main()
