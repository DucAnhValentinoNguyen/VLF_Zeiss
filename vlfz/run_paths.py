"""Stable names for canonical and tagged SSL/evaluation runs."""
from __future__ import annotations
import os

def tagged_name(name: str, run_tag: str = "") -> str:
    return f"{name}__{run_tag}" if run_tag else name

def ssl_dir(ckpts: str, objective: str, init: str, corpus: str,
            stage: str = "full", run_tag: str = "") -> str:
    return os.path.join(os.path.expanduser(str(ckpts)),
                        tagged_name(f"{objective}_{init}_{corpus}_{stage}", run_tag))

def result_tag(base: str, run_tag: str = "") -> str:
    return tagged_name(base, run_tag)


def checkpoint_metadata(cfg, objective, init, corpus, run_tag):
    import torch
    path = os.path.join(ssl_dir(str(cfg.paths.ckpts), objective, init, corpus, run_tag=run_tag),
                        'ema_backbone.pt')
    if not os.path.isfile(path):
        return {'run_tag': run_tag}
    return torch.load(path, map_location='cpu', weights_only=False).get('training_metadata',
                                                                      {'run_tag': run_tag})
