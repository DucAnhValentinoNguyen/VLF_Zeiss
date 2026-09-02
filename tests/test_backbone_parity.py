"""Backbone build + torchvision->timm conversion parity."""
import torch
import torch.nn.functional as F

from vlfz.cfg import load_cfg
from vlfz.models.vit_backbone import build_vit_b16

CFG = load_cfg()


def _rand(n=3):
    g = torch.Generator().manual_seed(0)
    return torch.randn(n, 3, 224, 224, generator=g)


def test_both_inits_shapes_finite():
    x = _rand()
    for init in ("imagenet", "siglip2"):
        bb = build_vit_b16(init, CFG).eval()
        with torch.no_grad():
            tok = bb.forward_tokens(x)
            feat = bb.feature(x)
        assert tok.shape == (x.shape[0], 196, 768)
        assert feat.shape == (x.shape[0], 768)
        assert torch.isfinite(feat).all()


def test_imagenet_matches_torchvision_reference():
    import torchvision

    x = _rand()
    bb = build_vit_b16("imagenet", CFG).eval()
    with torch.no_grad():
        ours = bb.feature(x)

    m = torchvision.models.vit_b_16(weights="IMAGENET1K_V1").eval()
    with torch.no_grad():
        z = m._process_input(x)
        cls = m.class_token.expand(z.shape[0], -1, -1)
        z = torch.cat([cls, z], dim=1)
        z = m.encoder(z)                      # pos-embed + blocks + final LN
        ref = F.layer_norm(z[:, 1:, :].mean(1), (768,))

    cos = F.cosine_similarity(ours, ref, dim=1)
    assert (cos > 0.999).all(), f"min cosine {cos.min().item():.5f}"


def test_convert_sanity():
    import torchvision

    from vlfz.models.convert_tv_vit import sanity_check, tv_to_timm

    tv = torchvision.models.vit_b_16(weights="IMAGENET1K_V1").state_dict()
    sanity_check(tv)
    timm_sd = tv_to_timm(tv)
    assert timm_sd["pos_embed"].shape == (1, 197, 768)
    assert timm_sd["blocks.11.attn.qkv.weight"].shape == (2304, 768)
