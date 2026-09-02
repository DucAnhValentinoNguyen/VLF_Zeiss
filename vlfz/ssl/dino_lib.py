"""DINO v1 self-distillation objective + schedules (Caron et al. 2021).

A student matches an EMA teacher's sharpened, centered output distribution across
multiple augmented crops of the same image (2 global + n_local). No labels, no
negatives; centering + teacher sharpening prevent collapse.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def dino_loss(student_out: list[torch.Tensor], teacher_out: list[torch.Tensor],
              center: torch.Tensor, student_temp: float, teacher_temp: float) -> torch.Tensor:
    """student_out: over ALL crops; teacher_out: over the 2 GLOBAL crops. CE of
    every student crop vs every teacher crop, skipping the identical view."""
    t = [F.softmax((tt - center) / teacher_temp, dim=-1).detach() for tt in teacher_out]
    total, n = 0.0, 0
    for ti, tq in enumerate(t):
        for si, sq in enumerate(student_out):
            if si == ti:
                continue
            total = total + (-(tq * F.log_softmax(sq / student_temp, dim=-1)).sum(-1).mean())
            n += 1
    return total / max(1, n)


@torch.no_grad()
def update_center(center: torch.Tensor, teacher_out: list[torch.Tensor],
                 momentum: float) -> torch.Tensor:
    batch_center = torch.cat(teacher_out, dim=0).mean(dim=0)
    center.mul_(momentum).add_(batch_center, alpha=1.0 - momentum)
    return center


def cosine_lr(step: int, total: int, base_lr: float, min_lr: float, warmup: int) -> float:
    if step < warmup:
        return base_lr * step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * p))


def teacher_temp_at(step: int, total: int, t0: float, t1: float, warmup_frac: float) -> float:
    frac = step / max(1, total)
    return t0 + (t1 - t0) * min(1.0, frac / max(1e-6, warmup_frac))
