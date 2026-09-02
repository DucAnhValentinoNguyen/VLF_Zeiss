"""torchvision ``vit_b_16`` state_dict  ->  timm ``vit_base_patch16_224`` state_dict.

Both are the standard ViT-B/16 (CLS token, learned 197-pos embed, fused QKV in
[q;k;v] row order, 12 blocks, GELU MLP). The mapping is 1:1 apart from names, so a
straight tensor copy is correct. Verified by tests/test_backbone_parity.py
(cosine of pre-head features vs a reference torchvision forward > 0.999).
"""
from __future__ import annotations

import re

_N_BLOCKS = 12


def tv_to_timm(tv_sd: dict) -> dict:
    """Return a timm-layout state_dict. Drops the 1000-way classification head."""
    out: dict = {}

    def put(dst: str, src: str):
        if src in tv_sd:
            out[dst] = tv_sd[src]

    put("cls_token", "class_token")
    put("pos_embed", "encoder.pos_embedding")
    put("patch_embed.proj.weight", "conv_proj.weight")
    put("patch_embed.proj.bias", "conv_proj.bias")
    put("norm.weight", "encoder.ln.weight")
    put("norm.bias", "encoder.ln.bias")

    for i in range(_N_BLOCKS):
        s = f"encoder.layers.encoder_layer_{i}."
        d = f"blocks.{i}."
        put(d + "norm1.weight", s + "ln_1.weight")
        put(d + "norm1.bias", s + "ln_1.bias")
        put(d + "norm2.weight", s + "ln_2.weight")
        put(d + "norm2.bias", s + "ln_2.bias")
        # fused qkv: torchvision in_proj_{weight,bias} is already [q;k;v] rows
        put(d + "attn.qkv.weight", s + "self_attention.in_proj_weight")
        put(d + "attn.qkv.bias", s + "self_attention.in_proj_bias")
        put(d + "attn.proj.weight", s + "self_attention.out_proj.weight")
        put(d + "attn.proj.bias", s + "self_attention.out_proj.bias")
        # torchvision MLPBlock is a Sequential: 0=Linear, 1=GELU, 2=Dropout, 3=Linear, 4=Dropout
        put(d + "mlp.fc1.weight", s + "mlp.0.weight")
        put(d + "mlp.fc1.bias", s + "mlp.0.bias")
        put(d + "mlp.fc2.weight", s + "mlp.3.weight")
        put(d + "mlp.fc2.bias", s + "mlp.3.bias")

    return out


def unmapped_tv_keys(tv_sd: dict) -> list[str]:
    """torchvision keys we deliberately ignore (should only be the head)."""
    keep_prefixes = ("class_token", "conv_proj", "encoder.")
    ignored = []
    for k in tv_sd:
        if not k.startswith(keep_prefixes):
            ignored.append(k)
    return ignored


def sanity_check(tv_sd: dict) -> None:
    got = tv_to_timm(tv_sd)
    assert "cls_token" in got and "pos_embed" in got, "missing embed tensors"
    for i in range(_N_BLOCKS):
        assert f"blocks.{i}.attn.qkv.weight" in got, f"block {i} qkv not mapped"
    leftover = [k for k in unmapped_tv_keys(tv_sd) if not re.match(r"heads?\.", k)]
    assert not leftover, f"unexpected unmapped torchvision keys: {leftover}"
