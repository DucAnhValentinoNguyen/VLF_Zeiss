"""Weight-EMA of the backbone — the artefact used for all downstream evaluation.

DINO needs an EMA *teacher* during training; LeJEPA does not, but an EMA of the
weights is still the more stable thing to evaluate. We keep one EMA copy for both
objectives so the eval path is identical.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, base_decay: float = 0.9995):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.base_decay = base_decay

    @torch.no_grad()
    def update(self, model: nn.Module, decay: float | None = None) -> None:
        d = self.base_decay if decay is None else decay
        for pe, pm in zip(self.ema.parameters(), model.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1.0 - d)
        for be, bm in zip(self.ema.buffers(), model.buffers()):
            be.copy_(bm)

    def state_dict(self):
        return self.ema.state_dict()

    def to(self, device):
        self.ema.to(device)
        return self


def cosine_momentum(step: int, total: int, base: float) -> float:
    """base -> 1.0 on a cosine schedule (DINO teacher-momentum style)."""
    import math

    frac = min(1.0, step / max(1, total))
    return 1.0 - (1.0 - base) * (0.5 * (1 + math.cos(math.pi * frac)))
