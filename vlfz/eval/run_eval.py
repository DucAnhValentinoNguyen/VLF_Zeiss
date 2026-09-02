"""Evaluation orchestrator: one (setting x stage x task) -> results/*.json.

setting  = (init, objective)     objective ignored when stage == "pre"
stage    = pre | post            post loads <ckpts>/{objective}_{init}_full/ema_backbone.pt
task     = rc_frame | rc_lesion
protocol = knn | eknn | elin(opt-in)

Each protocol yields pre/post-calibration (temperature for knn/elin, evidence-scale
lambda for eknn) rows with ECE/NLL/Brier + context metrics (+ vacuity/err-AUROC for
the evidential protocols).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..cfg import ensure_dirs, load_cfg, provenance, set_seed
from ..data import realcolon as RC
from ..models.vit_backbone import build_vit_b16
from .calibration import calibration_report
from .edl import (
    evidential_knn,
    fit_evidence_scale,
    train_evidential_head,
    uncertainty_block,
)
from .features import extract_features
from .knn import knn_k_sweep, knn_vote
from .metrics import context_metrics, flatten_report


def _backbone_and_tag(cfg, init, objective, stage):
    if stage == "pre":
        return build_vit_b16(init, cfg, pretrained_init=True), f"{init}_pre"
    ckpt = os.path.join(
        os.path.expanduser(str(cfg.paths.ckpts)),
        f"{objective}_{init}_full", "ema_backbone.pt",
    )
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no SSL checkpoint: {ckpt}")
    return build_vit_b16(init, cfg, ckpt=ckpt), f"{init}_{objective}_post"


def _rows_from_report(rep: dict, *, base: dict, protocol: str, extra: dict | None = None):
    rows = []
    for r in flatten_report(rep):
        row = {**base, "protocol": protocol}
        row.update(r)
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def evaluate(cfg, *, init, objective, stage, task, protocols, edl_head=False,
             recompute=False) -> dict:
    set_seed(int(cfg.seed))
    C = RC.N_CLASSES[task]
    cache_dir = os.path.expanduser(str(cfg.paths.cache))
    results_dir = os.path.expanduser(str(cfg.paths.results))
    ensure_dirs(cache_dir, results_dir)

    backbone, tag = _backbone_and_tag(cfg, init, objective, stage)

    splits = RC.video_splits(cfg)
    feats = {}
    for split in ("reference", "cal", "query"):
        samples = RC.build_task(cfg, task, splits[split])
        f = extract_features(backbone, samples, cfg, backbone_tag=f"{tag}|{task}|{split}",
                             cache_dir=cache_dir, recompute=recompute)
        feats[split] = f
        print(f"[eval] {tag} {task}/{split}: {len(samples)} imgs feat{f['X'].shape} hit={f['hit']}")

    Xr, yr = feats["reference"]["X"], feats["reference"]["y"]
    Xc, yc = feats["cal"]["X"], feats["cal"]["y"]
    Xq, yq = feats["query"]["X"], feats["query"]["y"]

    base = {"init": init, "stage": stage, "task": task,
            "objective": ("none" if stage == "pre" else objective),
            "n_query": int(len(yq))}
    rows: list[dict] = []
    aux: dict = {}

    if "knn" in protocols:
        k = int(cfg.eval.knn_k); tau = float(cfg.eval.knn_tau)
        q = knn_vote(Xr, yr, Xq, n_classes=C, k=k, tau=tau, l2norm=bool(cfg.eval.feature_l2norm))
        c = knn_vote(Xr, yr, Xc, n_classes=C, k=k, tau=tau, l2norm=bool(cfg.eval.feature_l2norm))
        rep = calibration_report(q["logits"], yq, c["logits"], yc,
                                 n_bins=int(cfg.calibration.n_bins),
                                 adaptive_bins=int(cfg.calibration.adaptive_bins),
                                 temp_max_iter=int(cfg.calibration.temp_max_iter))
        ctx = context_metrics(yq, q["probs"])
        rows += _rows_from_report(rep, base={**base, **ctx, "k": q["k"]}, protocol="knn")
        aux["knn_k_sweep"] = knn_k_sweep(Xr, yr, Xq, yq, list(cfg.eval.knn_k_sweep),
                                         n_classes=C, tau=tau)

    if "eknn" in protocols:
        k = int(cfg.eval.knn_k); tau = float(cfg.eval.knn_tau)
        q = knn_vote(Xr, yr, Xq, n_classes=C, k=k, tau=tau)
        c = knn_vote(Xr, yr, Xc, n_classes=C, k=k, tau=tau)
        lam = fit_evidence_scale(c["vote_mass"], yc)
        pre = evidential_knn(q["vote_mass"], scale=1.0)
        post = evidential_knn(q["vote_mass"], scale=lam)
        ctx = context_metrics(yq, pre["prob"])
        ub_pre = uncertainty_block(pre["prob"], pre["vacuity"], yq)
        ub_post = uncertainty_block(post["prob"], post["vacuity"], yq)
        from .calibration import metric_block

        nb, ab = int(cfg.calibration.n_bins), int(cfg.calibration.adaptive_bins)
        for scaled, s, ub in ((False, pre, ub_pre), (True, post, ub_post)):
            mb = metric_block(s["prob"], yq, nb, ab)
            rows.append({**base, **ctx, "protocol": "eknn", "temp_scaled": scaled,
                         "T": (lam if scaled else None), "k": q["k"],
                         "ece_ew": mb["ece_ew"], "ece_adaptive": mb["ece_adaptive"],
                         "nll": mb["nll"], "brier": mb["brier"], **ub})

    if edl_head and "elin" in protocols:
        out = train_evidential_head(Xr, yr, Xq, Xc, n_classes=C, seed=int(cfg.seed))
        qh, ch = out["query"], out["cal"]
        rep = calibration_report(qh["logits"], yq, ch["logits"], yc,
                                 n_bins=int(cfg.calibration.n_bins),
                                 adaptive_bins=int(cfg.calibration.adaptive_bins))
        ctx = context_metrics(yq, qh["prob"])
        ub = uncertainty_block(qh["prob"], qh["vacuity"], yq)
        rows += _rows_from_report(rep, base={**base, **ctx}, protocol="elin", extra=ub)

    out = {
        "tag": tag, "init": init, "objective": base["objective"], "stage": stage,
        "task": task, "rows": rows, "aux": aux,
        "splits": {k: len(v) for k, v in splits.items()},
        "provenance": provenance(int(cfg.seed), init=init, objective=objective, stage=stage,
                                 task=task, feature_cache=feats["query"]["cache"]),
    }
    fp = os.path.join(results_dir, f"{tag}__{task}.json")
    json.dump(out, open(fp, "w"), indent=2, default=float)
    print(f"[eval] wrote {fp}  ({len(rows)} rows)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--init", choices=["siglip2", "imagenet"], required=True)
    ap.add_argument("--objective", choices=["lejepa", "dino"], default="lejepa")
    ap.add_argument("--stage", choices=["pre", "post"], required=True)
    ap.add_argument("--task", choices=list(RC.TASKS) + ["all"], default="all")
    ap.add_argument("--protocols", default="knn,eknn")
    ap.add_argument("--edl-head", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    protocols = [p.strip() for p in a.protocols.split(",") if p.strip()]
    if a.edl_head and "elin" not in protocols:
        protocols.append("elin")
    tasks = list(RC.TASKS) if a.task == "all" else [a.task]
    for t in tasks:
        evaluate(cfg, init=a.init, objective=a.objective, stage=a.stage, task=t,
                 protocols=protocols, edl_head=a.edl_head, recompute=a.recompute)


if __name__ == "__main__":
    main()
