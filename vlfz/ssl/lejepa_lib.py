"""LeJEPA objective (Balestriero & LeCun 2025, arXiv:2511.08544).

    loss = prediction_loss + sigreg_lambda * SIGReg + var_weight * variance_hinge

- prediction_loss : two augmented views embedded by the SAME encoder (no EMA
  teacher). A small predictor maps view A's embedding toward view B's, and
  symmetrically. The TARGET is stop-gradient'd so the invariance term cannot be
  minimised by shrinking embeddings toward a constant.

- SIGReg (Sketched Isotropic Gaussian Regularization) : project the standardized
  batch onto many random 1-D directions and push each 1-D marginal toward a
  standard Gaussian via the Epps-Pulley characteristic-function test. By
  Cramer-Wold, matching all 1-D projections matches the full isotropic Gaussian.
  Cost is linear in dim and batch. Vectorized over slices/frequencies.

- variance_hinge : an explicit, numerically-stable VICReg-style floor on
  per-feature std (belt-and-braces anti-collapse; SIGReg shapes, this guards).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _standardize(z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    std = z.std(dim=0, keepdim=True).clamp_min(eps)
    return (z - z.mean(dim=0, keepdim=True)) / std


def variance_hinge(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.relu(target_std - std).mean()


def sigreg(z: torch.Tensor, n_slices: int = 256, n_freq: int = 16,
          freq_max: float = 5.0) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization. 0 iff every sampled 1-D
    projection of the standardized batch looks standard-Gaussian."""
    N, D = z.shape
    zc = _standardize(z)
    dirs = torch.randn(D, n_slices, device=z.device, dtype=z.dtype)
    dirs = dirs / (dirs.norm(dim=0, keepdim=True) + 1e-8)
    proj = zc @ dirs                                            # (N, S)
    ts = torch.linspace(0.1, freq_max, n_freq, device=z.device, dtype=z.dtype)
    tx = proj[:, :, None] * ts[None, None, :]                   # (N, S, T)
    ecf_re = torch.cos(tx).mean(dim=0)                          # (S, T)
    ecf_im = torch.sin(tx).mean(dim=0)                          # (S, T)
    cf_re = torch.exp(-0.5 * ts ** 2)[None, :]                  # std-normal CF (real)
    weight = torch.exp(-0.5 * ts ** 2)[None, :]                 # Gaussian integration weight
    diff = (ecf_re - cf_re) ** 2 + ecf_im ** 2
    return (diff * weight).mean()


def prediction_loss(emb_a: torch.Tensor, emb_b: torch.Tensor, predictor) -> torch.Tensor:
    pa = predictor(emb_a)
    pb = predictor(emb_b)
    return 0.5 * (
        F.smooth_l1_loss(pa, emb_b.detach()) + F.smooth_l1_loss(pb, emb_a.detach())
    )


def lejepa_loss(emb_a, emb_b, predictor, *, sigreg_lambda=1.0, var_weight=1.0,
                n_slices=256, n_freq=16):
    pred = prediction_loss(emb_a, emb_b, predictor)
    reg = 0.5 * (sigreg(emb_a, n_slices, n_freq) + sigreg(emb_b, n_slices, n_freq))
    var = 0.5 * (variance_hinge(emb_a) + variance_hinge(emb_b))
    total = pred + sigreg_lambda * reg + var_weight * var
    return total, {"pred": float(pred.detach()), "sigreg": float(reg.detach()),
                   "var": float(var.detach())}


@torch.no_grad()
def effective_rank(z: torch.Tensor) -> float:
    """Collapse proxy: exp(entropy of normalized singular values). Full rank -> ~D."""
    z = z - z.mean(0, keepdim=True)
    s = torch.linalg.svdvals(z.float())
    p = s / s.sum().clamp_min(1e-12)
    return float(torch.exp(-(p * (p.clamp_min(1e-12)).log()).sum()))
