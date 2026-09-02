"""Backbone-agnostic ViT-B/16 factory with a uniform feature API.

Every init ends up as a timm ``VisionTransformer``; the wrapper hides the
architectural differences (CLS token vs GAP, attn-pool head vs none) behind:

    .forward_tokens(x) -> (B, N_patch, 768)   post-trunk patch tokens
    .feature(x)        -> (B, 768)            LayerNorm(mean over patch tokens), non-affine
    .forward(x)        == .feature(x)

Feature = non-parametric LN of the mean patch token: identical, init-agnostic
pooling for k-NN / linear-probe / EDL, and a stable SSL target.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .load_weights import (
    openclip_siglip2_trunk,
    strict_load,
    torchvision_vit_b16_timm_sd,
)

VALID_INITS = ("siglip2", "imagenet")


class ViTBackbone(nn.Module):
    def __init__(self, trunk: nn.Module, init: str):
        super().__init__()
        self.trunk = trunk
        self.init = init
        self.num_prefix = int(getattr(trunk, "num_prefix_tokens", 0))
        self.embed_dim = int(getattr(trunk, "embed_dim", getattr(trunk, "num_features", 768)))

    def set_grad_checkpointing(self, enable: bool = True) -> None:
        try:
            self.trunk.set_grad_checkpointing(enable)
        except Exception:
            pass

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        t = self.trunk.forward_features(x)          # (B, T, C)
        if t.ndim != 3:
            raise RuntimeError(f"expected token output (B,T,C), got {tuple(t.shape)}")
        return t[:, self.num_prefix:, :]            # drop cls/reg prefix tokens

    def feature(self, x: torch.Tensor) -> torch.Tensor:
        patch = self.forward_tokens(x)             # (B, N, C)
        pooled = patch.mean(dim=1)                 # (B, C)
        return F.layer_norm(pooled, (pooled.shape[-1],))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(x)


def _timm_trunk(tag_or_arch: str, *, pretrained: bool, img_size: int):
    import timm

    return timm.create_model(
        tag_or_arch,
        pretrained=pretrained,
        num_classes=0,
        img_size=img_size,
        dynamic_img_size=True,
    )


def build_vit_b16(
    init: str,
    cfg,
    *,
    ckpt: str | None = None,
    pretrained_init: bool = True,
    map_location: str = "cpu",
) -> ViTBackbone:
    """Build a ViT-B/16 backbone.

    init            : "siglip2" | "imagenet" (base weight source)
    ckpt            : optional path to a vlfz SSL checkpoint; overlays its
                      ``backbone``/``ema_backbone`` weights on top of the base arch.
    pretrained_init : load the base pretrained weights (set False for a fresh init,
                      e.g. when a ckpt fully specifies the weights anyway).
    """
    if init not in VALID_INITS:
        raise ValueError(f"init must be one of {VALID_INITS}, got {init!r}")
    img_size = int(cfg.backbones.img_size)

    if init == "imagenet":
        trunk = _timm_trunk(cfg.backbones.arch, pretrained=False, img_size=img_size)
        if pretrained_init:
            sd = torchvision_vit_b16_timm_sd(str(cfg.backbones.imagenet_tv_weights))
            rep = strict_load(trunk, sd, "torchvision->timm imagenet")
            assert rep.ok(allow_missing=("head.weight", "head.bias")), (
                f"imagenet weight load incomplete:\n{rep}"
            )
    else:  # siglip2
        trunk = None
        if pretrained_init:
            for tag in (cfg.backbones.siglip2_timm_tag, cfg.backbones.siglip2_timm_tag_alt):
                try:
                    trunk = _timm_trunk(str(tag), pretrained=True, img_size=img_size)
                    print(f"[backbone] siglip2 via timm tag {tag}")
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"[backbone] timm tag {tag} failed ({e!r}); trying next")
            if trunk is None:
                name, pt = list(cfg.backbones.siglip2_openclip)
                print(f"[backbone] siglip2 via open_clip {name}:{pt}")
                trunk = openclip_siglip2_trunk(str(name), str(pt))
        else:
            trunk = _timm_trunk(
                str(cfg.backbones.siglip2_timm_tag), pretrained=False, img_size=img_size
            )

    bb = ViTBackbone(trunk, init)

    if ckpt:
        state = torch.load(ckpt, map_location=map_location)
        sd = state.get("ema_backbone") or state.get("backbone") or state
        sd = {k.replace("trunk.", "", 1) if k.startswith("trunk.") else k: v
              for k, v in sd.items()}
        rep = strict_load(bb.trunk, sd, f"ssl-ckpt {ckpt}")
        assert rep.ok(allow_unexpected_prefix=("head.", "attn_pool.", "fc_norm.")), (
            f"SSL checkpoint load incomplete:\n{rep}"
        )
        if "init" in state and state["init"] != init:
            print(f"[backbone] WARNING ckpt init={state['init']} != requested {init}")

    return bb


def feature_dim(bb: ViTBackbone) -> int:
    return bb.embed_dim
