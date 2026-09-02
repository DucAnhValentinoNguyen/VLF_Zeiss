"""SSL objective + schedule unit tests (no ViT-B; those run on a GPU node)."""
import torch

from vlfz.models.ema import EMA, cosine_momentum
from vlfz.models.heads import DINOHead, Predictor
from vlfz.ssl.dino_lib import (
    cosine_lr,
    dino_loss,
    teacher_temp_at,
    update_center,
)
from vlfz.ssl.lejepa_lib import effective_rank, lejepa_loss, sigreg, variance_hinge


def test_lejepa_loss_finite_and_backprops():
    torch.manual_seed(0)
    za = torch.randn(32, 64, requires_grad=True)
    zb = torch.randn(32, 64, requires_grad=True)
    loss, parts = lejepa_loss(za, zb, Predictor(64), n_slices=32, n_freq=8)
    loss.backward()
    assert torch.isfinite(loss) and za.grad is not None
    assert set(parts) == {"pred", "sigreg", "var"}


def test_sigreg_small_for_gaussian_large_for_collapse():
    torch.manual_seed(0)
    g = sigreg(torch.randn(256, 48))
    c = sigreg(torch.zeros(256, 48) + torch.randn(1, 48))
    assert g < c and g < 1e-2


def test_variance_hinge_zero_when_unit_std():
    z = torch.randn(4096, 16)
    assert variance_hinge(z) < 0.05
    assert variance_hinge(z * 0.01) > 0.5


def test_effective_rank_range():
    assert effective_rank(torch.randn(256, 32)) > 20
    z = torch.randn(256, 1) @ torch.randn(1, 32)          # rank 1
    assert effective_rank(z) < 3


def test_dino_loss_and_center():
    torch.manual_seed(0)
    head = DINOHead(16, out_dim=64)
    crops = [torch.randn(8, 16) for _ in range(4)]
    s_out = [head(c) for c in crops]
    t_out = [head(crops[0]).detach(), head(crops[1]).detach()]
    center = torch.zeros(64)
    loss = dino_loss(s_out, t_out, center, student_temp=0.1, teacher_temp=0.04)
    assert torch.isfinite(loss) and loss > 0
    c2 = update_center(center.clone(), t_out, 0.9)
    assert c2.shape == (64,) and not torch.allclose(c2, center)


def test_schedules_monotone_bounds():
    lrs = [cosine_lr(s, 100, 1.0, 0.0, warmup=10) for s in range(100)]
    assert abs(lrs[0]) < 1e-6 and lrs[10] > lrs[50] > lrs[95]
    assert teacher_temp_at(0, 100, 0.04, 0.07, 0.3) == 0.04
    assert abs(teacher_temp_at(100, 100, 0.04, 0.07, 0.3) - 0.07) < 1e-9
    assert 0.996 <= cosine_momentum(0, 100, 0.996) <= cosine_momentum(100, 100, 0.996) <= 1.0


def test_ema_update_moves_toward_model():
    a = torch.nn.Linear(8, 8)
    e = EMA(a, base_decay=0.5)
    with torch.no_grad():
        for p in a.parameters():
            p.add_(1.0)
    before = next(e.ema.parameters()).clone()
    e.update(a, decay=0.5)
    after = next(e.ema.parameters())
    assert not torch.allclose(before, after)
