"""How much disk does each part of a run take? -- for planning future training.

Walks $OUT_ROOT (SSL checkpoints, feature cache, results), $GASTRONET_ROOT (the
curated tier) and $DATA_ROOT (staged HyperKvasir) and buckets every file by
component: per SSL setting (+ file kind), per eval task's feature cache, results,
W&B run dirs, data. Prints a table; optionally writes JSON and/or logs the
numbers to W&B so storage is tracked over time.

    python -m scripts.storage_report                     # print
    python -m scripts.storage_report --json store.json   # + machine-readable
    WANDB_API_KEY=... python -m scripts.storage_report --wandb   # + a W&B run

Uses os.scandir (not `du`) so it stays responsive on Lustre.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from vlfz.cfg import load_cfg


def _walk_sizes(root: str):
    """Yield (abspath, size_bytes) for every file under root, following the
    top-level symlink but not chasing symlinks deeper."""
    root = os.path.realpath(os.path.expanduser(root))
    if not os.path.isdir(root):
        return
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            yield e.path, e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue


def _kind(fname: str) -> str:
    if fname == "ema_backbone.pt":
        return "ema_backbone.pt"
    if fname == "last.ckpt":
        return "last.ckpt"
    if fname.startswith("ckpt-") and fname.endswith(".ckpt"):
        return "periodic ckpt"
    if fname == "DONE":
        return "DONE"
    return "tb / other"


def _feat_task(fname: str, meta_dir: str) -> str:
    """Attribute a feat_*.npz to its task via the sidecar .json (features.py
    writes one); fall back to 'feature cache (unattributed)'."""
    base = fname[:-4] if fname.endswith(".npz") else fname
    mp = os.path.join(meta_dir, base + ".json")
    try:
        m = json.load(open(mp))
        return f"cache/{m.get('task', '?')}"
    except Exception:
        return "cache (unattributed)"


def collect(cfg) -> dict:
    out_root = os.path.expanduser(str(cfg.paths.out_root))
    ssl_root = os.path.join(out_root, "ssl")
    cache_root = os.path.join(out_root, "cache")
    results_root = os.path.join(out_root, "results")
    gnet = os.path.expanduser(str(cfg.paths.gastronet))
    data_root = os.path.expanduser(str(cfg.paths.data_root))
    wandb_dir = os.environ.get("WANDB_DIR", os.path.join(out_root, "wandb"))

    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # bucket -> [bytes, files]
    ssl_detail: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0]))

    def add(bucket, size):
        buckets[bucket][0] += size
        buckets[bucket][1] += 1

    for p, sz in _walk_sizes(ssl_root):
        rel = os.path.relpath(p, ssl_root)
        setting = rel.split(os.sep)[0]
        k = _kind(os.path.basename(p))
        ssl_detail[setting][k][0] += sz
        ssl_detail[setting][k][1] += 1
        add(f"ssl/{setting}", sz)

    for p, sz in _walk_sizes(cache_root):
        if p.endswith(".npz"):
            add(_feat_task(os.path.basename(p), cache_root), sz)
        else:
            add("cache/misc", sz)

    for p, sz in _walk_sizes(results_root):
        add("results", sz)
    for p, sz in _walk_sizes(os.path.join(gnet, "webdataset")):
        add("gastronet/webdataset", sz)
    for p, sz in _walk_sizes(os.path.join(gnet)):
        if "/webdataset/" not in p and p.endswith(".zip"):
            add("gastronet/raw zips", sz)
    for p, sz in _walk_sizes(data_root):
        add("data (staged, read-only)", sz)
    for p, sz in _walk_sizes(wandb_dir):
        add("wandb run dirs", sz)

    return {
        "buckets": {k: {"bytes": v[0], "files": v[1]} for k, v in sorted(buckets.items())},
        "ssl_detail": {s: {k: {"bytes": v[0], "files": v[1]} for k, v in kinds.items()}
                       for s, kinds in ssl_detail.items()},
        "out_root": out_root,
    }


def _h(n: int) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f} {u}" if u != "B" else f"{n} B"
        n /= 1024


def _print(rep: dict) -> None:
    print(f"\nstorage report  ({rep['out_root']})")
    print("=" * 58)
    tot = 0
    for b, d in rep["buckets"].items():
        tot += d["bytes"]
        print(f"  {b:<34} {_h(d['bytes']):>11}  ({d['files']} files)")
    print("-" * 58)
    print(f"  {'TOTAL':<34} {_h(tot):>11}")
    if rep["ssl_detail"]:
        print("\nper SSL setting:")
        for s, kinds in rep["ssl_detail"].items():
            sub = sum(v["bytes"] for v in kinds.values())
            print(f"  {s}  ({_h(sub)})")
            for k, v in sorted(kinds.items(), key=lambda x: -x[1]["bytes"]):
                print(f"      {k:<18} {_h(v['bytes']):>11}")
    print(f"\n1 finished SSL run keeps only ema_backbone.pt (~0.33 GB); "
          f"peak during training +~3 GB (last.ckpt + 1 periodic).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--json", default=None, help="also write the report here")
    ap.add_argument("--wandb", action="store_true", help="log the numbers to a W&B run")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    rep = collect(cfg)
    _print(rep)
    if a.json:
        json.dump(rep, open(a.json, "w"), indent=2)
        print(f"\n-> {a.json}")
    if a.wandb:
        try:
            import wandb

            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "vlf-zeiss"),
                             entity=os.environ.get("WANDB_ENTITY") or None,
                             name="storage-report", job_type="monitor",
                             config={"out_root": rep["out_root"]})
            wandb.log({f"storage_gb/{b}": d["bytes"] / 2**30
                       for b, d in rep["buckets"].items()})
            tbl = wandb.Table(columns=["bucket", "GB", "files"])
            for b, d in rep["buckets"].items():
                tbl.add_data(b, round(d["bytes"] / 2**30, 3), d["files"])
            wandb.log({"storage_table": tbl})
            run.finish()
            print("\n-> logged to W&B")
        except Exception as e:  # noqa: BLE001
            print(f"\n[wandb] skipped: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
