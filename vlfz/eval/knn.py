"""Weighted k-NN on frozen features (DINO protocol).

For each query, cosine-similarity to all reference features, take the top-k,
weight by ``softmax(sim / tau)``, and accumulate per-class vote mass.

Returns BOTH:
  * ``probs``      : vote mass normalised to sum 1        (softmax-style predictive)
  * ``logits``     : ``log(vote_mass + eps)``              (input to temperature scaling)
  * ``vote_mass``  : raw per-class weight sums (>= 0)      (evidence for evidential k-NN)

GPU torch; falls back to CPU. Features are L2-normalised here if not already.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-7


def _l2(x, dim=-1):
    import torch

    return x / (x.norm(dim=dim, keepdim=True) + 1e-12)


def knn_vote(
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    x_q: np.ndarray,
    *,
    n_classes: int | None = None,
    k: int = 20,
    tau: float = 0.07,
    l2norm: bool = True,
    device: str | None = None,
    q_chunk: int = 4096,
) -> dict:
    import torch

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    C = int(n_classes or (int(y_ref.max()) + 1))
    Xr = torch.as_tensor(np.asarray(x_ref), dtype=torch.float32, device=dev)
    Xq = torch.as_tensor(np.asarray(x_q), dtype=torch.float32, device=dev)
    yr = torch.as_tensor(np.asarray(y_ref), dtype=torch.long, device=dev)
    if l2norm:
        Xr, Xq = _l2(Xr), _l2(Xq)
    k = min(k, Xr.shape[0])

    onehot = torch.zeros(Xr.shape[0], C, device=dev)
    onehot[torch.arange(Xr.shape[0], device=dev), yr] = 1.0

    vote_mass = torch.empty(Xq.shape[0], C, device=dev)
    for i in range(0, Xq.shape[0], q_chunk):
        sims = Xq[i : i + q_chunk] @ Xr.T                       # (q, R)
        top_s, top_i = sims.topk(k, dim=1)                      # (q, k)
        w = torch.softmax(top_s / tau, dim=1)                   # (q, k)
        nb = onehot[top_i]                                      # (q, k, C)
        vote_mass[i : i + q_chunk] = (w.unsqueeze(-1) * nb).sum(dim=1)

    vm = vote_mass.cpu().numpy().astype(np.float64)
    probs = vm / np.clip(vm.sum(axis=1, keepdims=True), _EPS, None)
    # logits for temperature scaling: log of the *normalised* votes with a bounded
    # floor -> well-conditioned range (~[-14, 0]); raw log(vote_mass+1e-7) has a
    # pathological dynamic range that makes LBFGS drive T -> huge on small cal sets.
    logits = np.log(np.clip(probs, 1e-6, 1.0))
    return {"probs": probs, "logits": logits, "vote_mass": vm, "k": k, "tau": tau}


def knn_k_sweep(x_ref, y_ref, x_q, y_q, ks, *, n_classes=None, tau=0.07, l2norm=True):
    """Accuracy vs k, for the k-sensitivity table."""
    from sklearn.metrics import accuracy_score

    rows = []
    for k in ks:
        out = knn_vote(x_ref, y_ref, x_q, n_classes=n_classes, k=k, tau=tau, l2norm=l2norm)
        rows.append({"k": int(out["k"]),
                     "acc": float(accuracy_score(y_q, out["probs"].argmax(1)))})
    return rows
