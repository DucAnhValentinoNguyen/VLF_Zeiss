"""Projection / prediction heads for SSL.

Predictor + Projector: LeJEPA (Balestriero & LeCun 2025, arXiv:2511.08544).
DINOHead: DINO v1 (Caron et al. 2021) — GELU MLP + L2-norm + weight-normed
linear, with the last layer's magnitude frozen (weight_g = 1).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """Backbone feature -> SIGReg space (LeJEPA operates the Gaussian test here)."""

    def __init__(self, dim: int, out_dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or max(dim, out_dim) * 2
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class Predictor(nn.Module):
    """LeJEPA latent predictor (lightweight, no EMA target)."""

    def __init__(self, dim: int, hidden_mult: int = 2):
        super().__init__()
        h = dim * hidden_mult
        self.net = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        return self.net(x)


class DINOHead(nn.Module):
    """GELU MLP -> L2-norm -> linear with unit-norm rows (equivalent to DINO's
    weight-normed last layer with the magnitude ``weight_g`` frozen to 1).
    Avoids ``torch.nn.utils.weight_norm``, which is not deepcopy-safe (the EMA
    teacher is a deepcopy)."""

    def __init__(self, in_dim: int, out_dim: int = 8192, bottleneck: int = 256,
                 hidden: int = 2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.last = nn.Linear(bottleneck, out_dim, bias=False)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        w = F.normalize(self.last.weight, dim=1, p=2)
        return F.linear(x, w)
