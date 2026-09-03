"""Evaluation orchestrator: one (dataset x setting x stage x task) -> results/*.json.

dataset  = hyperkvasir | realcolon
setting  = (init, objective)      objective ignored when stage == "pre"
stage    = pre | post             post loads <ckpts>/{objective}_{init}_{corpus}_full/ema_backbone.pt
task     = a classification task of the dataset, or "all"
protocol = knn | eknn | elin(opt-in)

Each protocol yields pre/post-calibration rows (temperature for knn/elin,
evidence-scale lambda for eknn) with ECE/NLL/Brier + context metrics (+ vacuity /
err-AUROC for the evidential protocols).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..cfg import ensure_dirs, load_cfg, provenance, set_seed
from ..data.registry import get_dataset
from ..models.vit_backbone import build_vit_b16
from .calibration import calibration_report, metric_block
from .edl import evidential_knn, fit_evidence_scale, train_evidential_head, uncertainty_block
from .features import extract_features
from .knn import knn_k_sweep, knn_vote
from .metrics import context_metrics, flatten_report


def _backbone_and_tag(cfg, init, objective, stage, corpus):
    if stage == "pre":
        return build_vit_b16(init, cfg, pretrained_init=True), f"{init}_pre"
    ckpt = os.path.join(
        os.path.expanduser(str(cfg.paths.ckpts)),
        f"{objective}_{init}_{corpus}_full", "ema_backbone.pt",
    )
    if not os.path.exists(ckpt):  # tolerate the older name without a corpus tag
        alt = os.path.join(os.path.expanduser(str(cfg.paths.ckpts)),
                           f"{objective}_{init}_full", "ema_backbone.pt")
        ckpt = alt if os.path.exists(alt) else ckpt
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no SSL checkpoint: {ckpt}")
    return build_vit_b16(init, cfg, ckpt=ckpt), f"{init}_{objective}_{corpus}_post"


def _rows_from_report(rep, *, base, protocol, extra=None):
    rows = []
    for r in flatten_report(rep):
        row = {**base, "protocol": protocol, **r}
        if extra:
            row.update(extra)
        rows.append(row)
    return rows


def evaluate(cfg, *, dataset, init, objective, stage, task, protocols, corpus="gastronet",
             edl_head=False, recompute=False) -> dict:
    set_seed(int(cfg.seed))
    ds = get_dataset(dataset)
    C = ds.n_classes(cfg, task)
    cache_dir = os.path.expanduser(str(cfg.paths.cache))
    results_dir = os.path.expanduser(str(cfg.paths.results))
    ensure_dirs(cache_dir, results_dir)

    backbone, tag = _backbone_and_tag(cfg, init, objective, stage, corpus)

    feats = {}
    for split in ("reference", "cal", "query"):
        samples = ds.build(cfg, task, split)
        f = extract_features(backbone, samples, cfg,
                             backbone_tag=f"{tag}|{dataset}|{task}|{split}",
                             cache_dir=cache_dir, recompute=recompute)
        feats[split] = f
        print(f"[eval] {tag} {dataset}/{task}/{split}: {len(samples)} imgs "
              f"feat{f['X'].shape} hit={f['hit']}")

    Xr, yr = feats["reference"]["X"], feats["reference"]["y"]
    Xc, yc = feats["cal"]["X"], feats["cal"]["y"]
    Xq, yq = feats["query"]["X"], feats["query"]["y"]

    nb, ab = int(cfg.calibration.n_bins), int(cfg.calibration.adaptive_bins)
    base = {"dataset": dataset, "init": init, "stage": stage, "task": task,
            "objective": ("none" if stage == "pre" else objective),
            "corpus": ("none" if stage == "pre" else corpus),
            "n_query": int(len(yq)), "n_classes": int(C)}
    rows: list[dict] = []
    aux: dict = {}

    if "knn" in protocols:
        k, tau = int(cfg.eval.knn_k), float(cfg.eval.knn_tau)
        l2 = bool(cfg.eval.feature_l2norm)
        q = knn_vote(Xr, yr, Xq, n_classes=C, k=k, tau=tau, l2norm=l2)
        c = knn_vote(Xr, yr, Xc, n_classes=C, k=k, tau=tau, l2norm=l2)
        rep = calibration_report(q["logits"], yq, c["logits"], yc, n_bins=nb,
                                 adaptive_bins=ab, temp_max_iter=int(cfg.calibration.temp_max_iter))
        ctx = context_metrics(yq, q["probs"])
        rows += _rows_from_report(rep, base={**base, **ctx, "k": q["k"]}, protocol="knn")
        aux["knn_k_sweep"] = knn_k_sweep(Xr, yr, Xq, yq, list(cfg.eval.knn_k_sweep),
                                         n_classes=C, tau=tau)

    if "eknn" in protocols:
        k, tau = int(cfg.eval.knn_k), float(cfg.eval.knn_tau)
        q = knn_vote(Xr, yr, Xq, n_classes=C, k=k, tau=tau)
        c = knn_vote(Xr, yr, Xc, n_classes=C, k=k, tau=tau)
        lam = fit_evidence_scale(c["vote_mass"], yc)
        pre = evidential_knn(q["vote_mass"], scale=1.0)
        post = evidential_knn(q["vote_mass"], scale=lam)
        ctx = context_metrics(yq, pre["prob"])
        for scaled, s in ((False, pre), (True, post)):
            mb = metric_block(s["prob"], yq, nb, ab)
            ub = uncertainty_block(s["prob"], s["vacuity"], yq)
            rows.append({**base, **ctx, "protocol": "eknn", "temp_scaled": scaled,
                         "T": (lam if scaled else None), "k": q["k"],
                         "ece_ew": mb["ece_ew"], "ece_adaptive": mb["ece_adaptive"],
                         "nll": mb["nll"], "brier": mb["brier"], **ub})

    if edl_head and "elin" in protocols:
        out = train_evidential_head(Xr, yr, Xq, Xc, n_classes=C, seed=int(cfg.seed))
        qh, ch = out["query"], out["cal"]
        rep = calibration_report(qh["logits"], yq, ch["logits"], yc, n_bins=nb, adaptive_bins=ab)
        ctx = context_metrics(yq, qh["prob"])
        ub = uncertainty_block(qh["prob"], qh["vacuity"], yq)
        rows += _rows_from_report(rep, base={**base, **ctx}, protocol="elin", extra=ub)

    out = {"tag": tag, "dataset": dataset, "init": init, "objective": base["objective"],
           "corpus": base["corpus"], "stage": stage, "task": task, "rows": rows, "aux": aux,
           "splits": ds.split_sizes(cfg),
           "provenance": provenance(int(cfg.seed), dataset=dataset, init=init,
                                    objective=objective, stage=stage, task=task, corpus=corpus)}
    fp = os.path.join(results_dir, f"{dataset}__{tag}__{task}.json")
    json.dump(out, open(fp, "w"), indent=2, default=float)
    print(f"[eval] wrote {fp}  ({len(rows)} rows)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dataset", default="hyperkvasir", choices=["hyperkvasir", "realcolon"])
    ap.add_argument("--init", choices=["siglip2", "imagenet"], required=True)
    ap.add_argument("--objective", choices=["lejepa", "dino"], default="lejepa")
    ap.add_argument("--corpus", default=os.environ.get("CORPUS", "gastronet"),
                    choices=["gastronet", "hkv_unlabeled"])
    ap.add_argument("--stage", choices=["pre", "post"], required=True)
    ap.add_argument("--task", default="all")
    ap.add_argument("--protocols", default="knn,eknn")
    ap.add_argument("--edl-head", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    ds = get_dataset(a.dataset)
    protocols = [p.strip() for p in a.protocols.split(",") if p.strip()]
    if a.edl_head and "elin" not in protocols:
        protocols.append("elin")

    if a.task == "all":
        tasks = list(ds.cls_tasks)
        if a.dataset == "hyperkvasir":               # honour the configured task subset
            want = [t.strip() for t in str(cfg.hyperkvasir.tasks).split(",") if t.strip()]
            tasks = [t for t in want if t in ds.cls_tasks] or tasks
    elif a.task in ds.seg_tasks:
        from .seg import run_seg

        run_seg(cfg, dataset=a.dataset, init=a.init, objective=a.objective,
                stage=a.stage, corpus=a.corpus)
        return
    else:
        tasks = [a.task]

    for t in tasks:
        evaluate(cfg, dataset=a.dataset, init=a.init, objective=a.objective, stage=a.stage,
                 task=t, protocols=protocols, corpus=a.corpus, edl_head=a.edl_head,
                 recompute=a.recompute)


if __name__ == "__main__":
    main()
