"""Calibration metrics: ECE (equal-width + adaptive), NLL, Brier, temperature scaling.

Everything is numpy in / numpy out except ``fit_temperature`` (a tiny torch LBFGS).
``calibration_report`` is the single entry point used by run_eval: it takes query
logits + a held-out calibration split, fits a scalar T on the cal split, and
returns pre/post-temperature metrics.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-7


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _as_probs(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    row = x.sum(axis=1)
    if np.allclose(row, 1.0, atol=1e-3) and (x >= 0).all():
        return np.clip(x, _EPS, 1.0)
    return softmax_np(x)


def nll(probs: np.ndarray, y: np.ndarray, eps: float = _EPS) -> float:
    p = _as_probs(probs)
    py = p[np.arange(len(y)), y]
    return float(-np.log(np.clip(py, eps, 1.0)).mean())


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    p = _as_probs(probs)
    oh = np.zeros_like(p)
    oh[np.arange(len(y)), y] = 1.0
    return float(((p - oh) ** 2).sum(axis=1).mean())


def _reliability(conf: np.ndarray, correct: np.ndarray, edges: np.ndarray):
    bins, ece, n = [], 0.0, len(conf)
    for lo, hi, last in zip(edges[:-1], edges[1:], range(len(edges) - 1)):
        m = (conf > lo) & (conf <= hi) if last else (conf >= lo) & (conf <= hi)
        c = int(m.sum())
        if c == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "count": 0,
                         "conf": 0.0, "acc": 0.0})
            continue
        bc, ba = float(conf[m].mean()), float(correct[m].mean())
        ece += c / n * abs(ba - bc)
        bins.append({"lo": float(lo), "hi": float(hi), "count": c,
                     "conf": bc, "acc": ba})
    return float(ece), bins


def ece_equal_width(probs: np.ndarray, y: np.ndarray, n_bins: int = 15):
    p = _as_probs(probs)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return _reliability(conf, correct, edges)


def ece_adaptive(probs: np.ndarray, y: np.ndarray, n_bins: int = 15):
    """Equal-mass (confidence-quantile) bins; robust to skewed confidence."""
    p = _as_probs(probs)
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == y).astype(np.float64)
    q = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(conf, q))
    if len(edges) < 2:
        edges = np.array([0.0, 1.0])
    edges[0], edges[-1] = 0.0, 1.0
    return _reliability(conf, correct, edges)


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 100) -> float:
    """Scalar T > 0 minimising CE(logits / T, y). Optimises log T (Guo et al. 2017)."""
    import torch

    lg = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    ty = torch.as_tensor(np.asarray(y), dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(lg / log_t.exp(), ty)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_t.exp().detach().item())
    if not np.isfinite(T):
        return 1.0
    return float(min(max(T, 1e-2), 1e3))          # guard degenerate small cal sets


def metric_block(probs: np.ndarray, y: np.ndarray, n_bins: int, adaptive_bins: int) -> dict:
    ece_ew, rel_ew = ece_equal_width(probs, y, n_bins)
    ece_ad, _ = ece_adaptive(probs, y, adaptive_bins)
    return {
        "ece_ew": ece_ew,
        "ece_adaptive": ece_ad,
        "nll": nll(probs, y),
        "brier": brier(probs, y),
        "reliability": rel_ew,
    }


def calibration_report(
    logits_q: np.ndarray,
    y_q: np.ndarray,
    logits_cal: np.ndarray | None = None,
    y_cal: np.ndarray | None = None,
    *,
    n_bins: int = 15,
    adaptive_bins: int = 15,
    temp_max_iter: int = 100,
) -> dict:
    logits_q = np.asarray(logits_q, dtype=np.float64)
    pre = metric_block(logits_q, y_q, n_bins, adaptive_bins)
    out = {"pre": pre, "T": None, "post": None}
    if logits_cal is not None and y_cal is not None and len(y_cal) > 1:
        T = fit_temperature(logits_cal, y_cal, temp_max_iter)
        out["T"] = T
        out["post"] = metric_block(logits_q / T, y_q, n_bins, adaptive_bins)
    return out
