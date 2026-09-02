"""Weight sources for the ViT-B/16 backbone + a strict-load reporter.

Three sources:
  * imagenet : torchvision ``vit_b_16(weights=IMAGENET1K_V1)`` -> converted to the
               timm ``vit_base_patch16_224`` layout (see convert_tv_vit.py).
  * siglip2  : timm ``vit_base_patch16_siglip_gap_224.v2_webli`` (pretrained), with an
               open_clip ``ViT-B-16-SigLIP2`` fallback whose ``.visual.trunk`` is a
               timm VisionTransformer.
  * ssl ckpt : a checkpoint written by vlfz.ssl.pretrain (records its base ``init``).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LoadReport:
    source: str
    missing: list[str]
    unexpected: list[str]

    def ok(self, allow_missing=(), allow_unexpected_prefix=()) -> bool:
        miss = [k for k in self.missing if k not in allow_missing]
        unexp = [
            k for k in self.unexpected
            if not k.startswith(tuple(allow_unexpected_prefix))
        ]
        return not miss and not unexp

    def __str__(self) -> str:
        return (
            f"[{self.source}] missing={len(self.missing)} unexpected={len(self.unexpected)}"
            + (f"\n  missing[:8]={self.missing[:8]}" if self.missing else "")
            + (f"\n  unexpected[:8]={self.unexpected[:8]}" if self.unexpected else "")
        )


def strict_load(model, state_dict, source: str) -> LoadReport:
    res = model.load_state_dict(state_dict, strict=False)
    rep = LoadReport(source, list(res.missing_keys), list(res.unexpected_keys))
    print(rep)
    return rep


def torchvision_vit_b16_timm_sd(weights: str = "IMAGENET1K_V1") -> dict:
    """torchvision vit_b_16 weights, remapped to timm vit_base_patch16_224 layout."""
    import torchvision
    from .convert_tv_vit import sanity_check, tv_to_timm

    wenum = torchvision.models.get_weight(f"ViT_B_16_Weights.{weights}")
    tv_sd = torchvision.models.vit_b_16(weights=wenum).state_dict()
    sanity_check(tv_sd)
    return tv_to_timm(tv_sd)


def openclip_siglip2_trunk(name: str, pretrained: str):
    """open_clip SigLIP-2 vision trunk (a timm VisionTransformer) as a fallback."""
    import open_clip

    model = open_clip.create_model(name, pretrained=pretrained)
    visual = model.visual
    trunk = getattr(visual, "trunk", None)
    if trunk is None:  # non-timm open_clip visual — not supported here
        raise RuntimeError(
            f"open_clip {name}:{pretrained} visual has no .trunk (not a timm ViT)"
        )
    return trunk
