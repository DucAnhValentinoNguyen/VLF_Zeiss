"""Scan results/*.json -> long_results.csv (+ delta view + markdown pivot)."""
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd

from ..cfg import load_cfg
from .pivot import write_pivot

_ROW_COLS = [
    "dataset", "init", "objective", "corpus", "stage", "task", "protocol",
    "temp_scaled", "T",
    "ece_ew", "ece_adaptive", "nll", "brier",
    "accuracy", "balanced_acc", "macro_f1", "auroc",
    "dice", "miou", "nll_fg", "ece_ew_fg",
    "k", "n_query", "n_classes", "run_tag", "train_shards", "ssl_base_lr", "ssl_min_lr", "ssl_llrd",
]


def collect(results_dir: str) -> pd.DataFrame:
    recs = []
    for fp in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        try:
            d = json.load(open(fp))
        except Exception:
            continue
        if "rows" not in d:
            continue
        prov = d.get("provenance", {})
        for r in d["rows"]:
            rec = {c: r.get(c) for c in _ROW_COLS}
            rec["git_sha"] = prov.get("git_sha", "")
            rec["timestamp"] = prov.get("timestamp", "")
            rec["node"] = prov.get("slurm_nodelist", "")
            rec["gpu"] = prov.get("gpu", "")
            rec["gpu_vram_gb"] = prov.get("gpu_vram_gb", "")
            rec["source"] = os.path.basename(fp)
            for c in ("run_tag", "train_shards", "ssl_base_lr", "ssl_min_lr", "ssl_llrd"):
                if rec.get(c) is None:
                    rec[c] = prov.get(c, "")
            recs.append(rec)
    return pd.DataFrame.from_records(recs)


def delta_view(df: pd.DataFrame) -> pd.DataFrame:
    """post - pre for matched (dataset, init, objective, corpus, task, protocol, temp_scaled)."""
    keys = ["dataset", "init", "objective", "corpus", "task", "protocol", "temp_scaled", "run_tag"]
    metrics = ["ece_ew", "ece_adaptive", "nll", "brier", "accuracy",
               "balanced_acc", "auroc", "dice", "miou"]
    pre = df[df.stage == "pre"].copy()
    post = df[df.stage == "post"].copy()
    # pre has objective/corpus == "none"; match it to each post variant
    pre_any = pre.drop(columns=["objective", "corpus", "run_tag"])
    pre_any["run_tag"] = ""
    merged = post.merge(pre_any, on=[k for k in keys if k not in ("objective", "corpus", "run_tag")],
                        suffixes=("_post", "_pre"))
    for m in metrics:
        if f"{m}_post" in merged and f"{m}_pre" in merged:
            merged[f"d_{m}"] = merged[f"{m}_post"] - merged[f"{m}_pre"]
    cols = keys + [f"d_{m}" for m in metrics if f"d_{m}" in merged]
    return merged[[c for c in cols if c in merged]].drop_duplicates()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    results_dir = a.results or os.path.expanduser(str(cfg.paths.results))
    out_dir = a.out or results_dir
    os.makedirs(out_dir, exist_ok=True)

    df = collect(results_dir)
    if df.empty:
        print(f"[aggregate] no result rows under {results_dir}")
        return
    long_fp = os.path.join(out_dir, "long_results.csv")
    df.to_csv(long_fp, index=False)
    print(f"[aggregate] {len(df)} rows -> {long_fp}")

    dv = delta_view(df)
    if not dv.empty:
        dfp = os.path.join(out_dir, "delta_results.csv")
        dv.to_csv(dfp, index=False)
        print(f"[aggregate] {len(dv)} delta rows -> {dfp}")

    write_pivot(df, os.path.join(out_dir, "report.md"), dv)
    print(f"[aggregate] -> {os.path.join(out_dir, 'report.md')}")


if __name__ == "__main__":
    main()
