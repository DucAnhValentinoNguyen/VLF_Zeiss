"""Image transforms. Endoscopy-safe SSL views: NO horizontal/vertical flip
(laterality is diagnostic) and NO grayscale. ImageNet normalisation is used
everywhere (eval + SSL) so there is a single feature-extraction path.
"""
from __future__ import annotations

from PIL import Image


def _norm(cfg):
    from torchvision import transforms as T

    return T.Normalize(list(cfg.backbones.norm_mean), list(cfg.backbones.norm_std))


def eval_transform(cfg):
    from torchvision import transforms as T

    s = int(cfg.eval.img_size)
    return T.Compose([
        T.Resize((s, s), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        _norm(cfg),
    ])


def _ssl_photometric(cfg):
    from torchvision import transforms as T

    cj = list(cfg.aug.color_jitter)
    return [
        T.RandomRotation(float(cfg.aug.rotation_deg)),
        T.ColorJitter(*cj),
        T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=float(cfg.aug.blur_p)),
    ]


def _one_view(cfg):
    from torchvision import transforms as T

    s = int(cfg.backbones.img_size)
    return T.Compose([
        T.RandomResizedCrop(s, scale=tuple(cfg.aug.rrc_scale),
                            interpolation=T.InterpolationMode.BICUBIC),
        *_ssl_photometric(cfg),
        T.ToTensor(),
        _norm(cfg),
    ])


class TwoViewTransform:
    """LeJEPA: two i.i.d. stochastic views at the model resolution -> [v1, v2]."""

    def __init__(self, cfg):
        self.t = _one_view(cfg)

    def __call__(self, img: Image.Image):
        return [self.t(img), self.t(img)]


def two_view_transform(cfg):
    return TwoViewTransform(cfg)


class MultiCropTransform:
    """DINO: 2 global crops (large area) + n_local small-area crops, all at model res."""

    def __init__(self, cfg):
        from torchvision import transforms as T

        s = int(cfg.backbones.img_size)
        g = tuple(cfg.ssl.dino.global_scale)
        loc = tuple(cfg.ssl.dino.local_scale)
        self.n_local = int(cfg.ssl.dino.n_local)
        common = [*_ssl_photometric(cfg), T.ToTensor(), _norm(cfg)]
        self.g = T.Compose([T.RandomResizedCrop(s, scale=g,
                            interpolation=T.InterpolationMode.BICUBIC), *common])
        self.l = T.Compose([T.RandomResizedCrop(s, scale=loc,
                            interpolation=T.InterpolationMode.BICUBIC), *common])

    def __call__(self, img: Image.Image):
        return [self.g(img), self.g(img)] + [self.l(img) for _ in range(self.n_local)]


def multicrop_collate(batch):
    """batch = list of [crop0..cropK] -> list of K stacked tensors (crop-major)."""
    import torch

    k = len(batch[0])
    return [torch.stack([b[c] for b in batch]) for c in range(k)]
