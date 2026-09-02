"""Evidential Deep Learning (Sensoy et al. 2018, arXiv:1806.01768).

Two uses in this pipeline:

  1. Evidential k-NN (training-free, default ON) — the weighted neighbour
     vote mass IS Dirichlet evidence: ``alpha_c = lambda * vote_mass_c + 1``.
     Gives a predictive distribution (Dirichlet mean), an epistemic
     "vacuity" u = C / S, and — after fitting the evidence scale ``lambda`` on
     the calibration split — a smoothed, better-calibrated probability than raw
     normalised votes. ``lambda`` plays the role temperature plays for softmax.

  2. Evidential linear probe (opt-in) — a Linear(d -> C) head on frozen features
     with softplus evidence and the Sensoy Type-II MLE loss (digamma form) plus an
     annealed KL to the uniform Dirichlet on "misleading" evidence.

Deep Evidential *Regression* (Amini et al. 2019, arXiv:1910.02600) is the
Normal-Inverse-Gamma analogue for continuous targets; the two REAL-Colon tasks
here are binary classification, so DER is out of scope. If a regression target is
later added (e.g. polyp size in mm from lesion_info.csv), NIG regression drops in
here as a sibling module.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-7


# --------------------------------------------------------------------- Dirichlet
def dirichlet_stats(alpha: np.ndarray) -> dict:
    alpha = np.asarray(alpha, dtype=np.float64)
    S = alpha.sum(axis=1, keepdims=True)
    C = alpha.shape[1]
    mean = alpha / S
    vacuity = (C / S[:, 0])                       # epistemic uncertainty in [0,1]
    return {"prob": mean, "S": S[:, 0], "vacuity": vacuity, "alpha": alpha}


def evidential_knn(vote_mass: np.ndarray, scale: float = 1.0) -> dict:
    """alpha = scale * vote_mass + 1."""
    alpha = scale * np.asarray(vote_mass, dtype=np.float64) + 1.0
    out = dirichlet_stats(alpha)
    out["logits"] = np.log(alpha)                 # for temperature scaling if desired
    out["scale"] = float(scale)
    return out


def fit_evidence_scale(vote_mass_cal: np.ndarray, y_cal: np.ndarray,
                       max_iter: int = 100) -> float:
    """lambda > 0 minimising NLL of the Dirichlet mean on the cal split."""
    import torch

    vm = torch.as_tensor(np.asarray(vote_mass_cal), dtype=torch.float64)
    ty = torch.as_tensor(np.asarray(y_cal), dtype=torch.long)
    C = vm.shape[1]
    log_lam = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_lam], lr=0.1, max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        alpha = log_lam.exp() * vm + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        p_y = (alpha[torch.arange(len(ty)), ty] / S[:, 0]).clamp_min(_EPS)
        loss = -(p_y.log().mean())
        loss.backward()
        return loss

    opt.step(closure)
    lam = float(log_lam.exp().detach().item())
    if not np.isfinite(lam):
        return 1.0
    return float(min(max(lam, 1e-3), 1e3))        # guard degenerate small cal sets


# --------------------------------------------------------- uncertainty reporting
def error_detection_auroc(uncertainty: np.ndarray, correct: np.ndarray) -> float:
    """AUROC of `uncertainty` predicting a wrong prediction (higher u -> error)."""
    from sklearn.metrics import roc_auc_score

    wrong = 1 - np.asarray(correct).astype(int)
    if wrong.sum() == 0 or wrong.sum() == len(wrong):
        return float("nan")
    return float(roc_auc_score(wrong, np.asarray(uncertainty, dtype=np.float64)))


def uncertainty_block(prob: np.ndarray, vacuity: np.ndarray, y: np.ndarray) -> dict:
    correct = (prob.argmax(axis=1) == y).astype(int)
    ent = -(prob * np.log(np.clip(prob, _EPS, 1))).sum(axis=1)
    return {
        "vacuity_mean": float(np.mean(vacuity)),
        "entropy_mean": float(np.mean(ent)),
        "err_auroc_vacuity": error_detection_auroc(vacuity, correct),
        "err_auroc_entropy": error_detection_auroc(ent, correct),
    }


# ---------------------------------------------------- evidential linear probe
def _edl_loss(alpha, y_onehot, t_frac, kind="digamma"):
    import torch

    S = alpha.sum(dim=1, keepdim=True)
    if kind == "digamma":
        err = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)
    else:  # expected Brier
        p = alpha / S
        err = ((y_onehot - p) ** 2 + p * (1 - p) / (S + 1)).sum(dim=1)
    # KL(Dir(alpha_tilde) || Dir(1)) on misleading evidence, annealed
    alpha_t = y_onehot + (1 - y_onehot) * alpha
    C = alpha.shape[1]
    S_t = alpha_t.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(S_t).squeeze(1) - torch.lgamma(alpha_t).sum(dim=1)
        - (torch.lgamma(torch.tensor(float(C))) - C * torch.lgamma(torch.tensor(1.0)))
        + ((alpha_t - 1) * (torch.digamma(alpha_t) - torch.digamma(S_t))).sum(dim=1)
    )
    lam = min(1.0, float(t_frac))
    return (err + lam * kl).mean()


def train_evidential_head(
    x_tr: np.ndarray, y_tr: np.ndarray,
    x_q: np.ndarray,
    x_cal: np.ndarray | None = None,
    *,
    n_classes: int | None = None,
    epochs: int = 200, lr: float = 1e-2, wd: float = 1e-4,
    loss_kind: str = "digamma", seed: int = 0, device: str | None = None,
) -> dict:
    """Fit Linear(d->C) evidential head on cached features; predict on x_q (and x_cal).

    Returns dict with alpha/prob/vacuity/logits for the query set (and cal set if given).
    """
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    C = int(n_classes or (int(y_tr.max()) + 1))
    Xtr = torch.as_tensor(np.asarray(x_tr), dtype=torch.float32, device=dev)
    ytr = torch.as_tensor(np.asarray(y_tr), dtype=torch.long, device=dev)
    yoh = torch.zeros(len(ytr), C, device=dev)
    yoh[torch.arange(len(ytr)), ytr] = 1.0
    # standardise features (fit on train)
    mu, sd = Xtr.mean(0, keepdim=True), Xtr.std(0, keepdim=True).clamp_min(1e-6)
    Xtr = (Xtr - mu) / sd

    head = torch.nn.Linear(Xtr.shape[1], C).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=wd)
    for ep in range(epochs):
        opt.zero_grad()
        ev = torch.nn.functional.softplus(head(Xtr))
        loss = _edl_loss(ev + 1.0, yoh, (ep + 1) / min(epochs, 10), loss_kind)
        loss.backward()
        opt.step()

    head.eval()

    def _predict(x):
        X = (torch.as_tensor(np.asarray(x), dtype=torch.float32, device=dev) - mu) / sd
        with torch.no_grad():
            alpha = (torch.nn.functional.softplus(head(X)) + 1.0).cpu().numpy().astype(np.float64)
        s = dirichlet_stats(alpha)
        s["logits"] = np.log(alpha)
        return s

    out = {"query": _predict(x_q)}
    if x_cal is not None:
        out["cal"] = _predict(x_cal)
    return out
