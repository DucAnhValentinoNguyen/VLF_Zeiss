"""Push already-finished SSL runs into W&B so all settings live in one place.

Runs 1-3 of the 4-cell matrix trained before W&B was wired in. This replays each
finished run's TensorBoard history (loss, lr, effective_rank, timing) + its full
hyperparameter set into a W&B run, using the SAME run id scheme as
vlfz.ssl.pretrain (md5(out_dir)[:16]) so a later live re-run continues it rather
than forking.

    WANDB_API_KEY=... python -m scripts.wandb_backfill              # all DONE runs
    python -m scripts.wandb_backfill --only lejepa_siglip2 --dry-run
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import os
import types

from vlfz.cfg import load_cfg
from vlfz.ssl.pretrain import _hparams


def _tb_scalars(tb_dir: str) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    evs = sorted(glob.glob(os.path.join(tb_dir, "**", "events.out.*"), recursive=True))
    if not evs:
        return {}
    ea = EventAccumulator(os.path.dirname(evs[-1]),
                          size_guidance={"scalars": 0})  # 0 = keep all
    ea.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(s.step, s.value) for s in ea.Scalars(tag)]
    return out


def _replay(run, scalars: dict) -> None:
    # merge every tag onto a single step axis, log in step order
    by_step: dict[int, dict] = {}
    for tag, series in scalars.items():
        for step, val in series:
            by_step.setdefault(step, {})[tag] = val
    for step in sorted(by_step):
        run.log(by_step[step], step=step)


def backfill(cfg, out_dir: str, *, dry: bool) -> None:
    name = os.path.basename(out_dir)                       # obj_init_corpus_stage
    parts = name.split("_")
    obj, init, corpus, stage = parts[0], parts[1], parts[-2], parts[-1]
    tb = _tb_scalars(os.path.join(out_dir, "tb"))
    if not tb:
        print(f"  {name}: no TB scalars, skip")
        return
    steps = max((s for series in tb.values() for s, _ in series), default=0)
    n_epoch = len(tb.get("collapse/effective_rank", [])) or 1
    spe = steps // n_epoch if n_epoch else steps
    args = types.SimpleNamespace(objective=obj, init=init, corpus=corpus, stage=stage,
                                 source="local_webdataset", lora=False)
    hp = _hparams(cfg, args, bs=int(cfg.ssl.batch_size), num_workers=int(cfg.ssl.num_workers),
                  epochs=n_epoch, steps_per_epoch=spe, total_steps=steps,
                  device="cuda", precision="bf16-mixed", resumed=True, backfilled=True)
    last = {t: series[-1][1] for t, series in tb.items() if series}
    print(f"  {name}: {sum(len(v) for v in tb.values())} points over {len(tb)} tags, "
          f"~{n_epoch} epoch(s), {steps} steps | final loss={last.get('loss'):.4f} "
          f"eff_rank={last.get('collapse/effective_rank', float('nan')):.1f}")
    if dry:
        return
    import wandb

    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "vlf-zeiss"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=name, id=hashlib.md5(out_dir.encode()).hexdigest()[:16], resume="allow",
        tags=[obj, init, corpus, stage, "backfill"], config=hp,
    )
    _replay(run, tb)
    for k, v in last.items():
        run.summary[k] = v
    run.summary["backfilled"] = True
    run.finish()
    print(f"    -> {run.url if hasattr(run, 'url') else 'logged'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", default=None, help="setting name substring filter")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    ssl_root = os.path.join(os.path.expanduser(str(cfg.paths.out_root)), "ssl")
    dirs = sorted(d for d in glob.glob(os.path.join(ssl_root, "*"))
                  if os.path.isfile(os.path.join(d, "DONE")))
    if a.only:
        dirs = [d for d in dirs if a.only in os.path.basename(d)]
    if not dirs:
        print(f"no finished (DONE) SSL runs under {ssl_root}")
        return
    print(f"backfilling {len(dirs)} run(s){' (dry run)' if a.dry_run else ''}:")
    for d in dirs:
        backfill(cfg, d, dry=a.dry_run)


if __name__ == "__main__":
    main()
