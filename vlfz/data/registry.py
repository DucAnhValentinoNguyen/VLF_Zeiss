"""Uniform interface over the eval datasets so run_eval is dataset-agnostic.

Each provider exposes:
    cls_tasks              -> tuple[str]         classification task names
    seg_tasks              -> tuple[str]         segmentation task names (may be empty)
    n_classes(cfg, task)   -> int
    build(cfg, task, split)-> list[(path, int)]  split in {"reference","cal","query"}
    split_sizes(cfg)       -> dict[str,int]
"""
from __future__ import annotations


class _HyperKvasir:
    name = "hyperkvasir"

    @property
    def cls_tasks(self):
        from . import hyperkvasir as H

        return H.CLS_TASKS

    @property
    def seg_tasks(self):
        from . import hyperkvasir as H

        return H.SEG_TASKS

    def n_classes(self, cfg, task):
        from . import hyperkvasir as H

        return H.n_classes(cfg, task)

    def build(self, cfg, task, split):
        from . import hyperkvasir as H

        return H.build_task(cfg, task, split)

    def split_sizes(self, cfg):
        from . import hyperkvasir as H

        return {k: len(v) for k, v in H.global_split(cfg).items()}


class _RealColon:
    name = "realcolon"

    @property
    def cls_tasks(self):
        from . import realcolon as R

        return tuple(R.TASKS)

    seg_tasks = ()

    def n_classes(self, cfg, task):
        from . import realcolon as R

        return R.N_CLASSES[task]

    def build(self, cfg, task, split):
        from . import realcolon as R

        return R.build_task(cfg, task, R.video_splits(cfg)[split])

    def split_sizes(self, cfg):
        from . import realcolon as R

        return {k: len(v) for k, v in R.video_splits(cfg).items()}


_PROVIDERS = {"hyperkvasir": _HyperKvasir(), "realcolon": _RealColon()}


def get_dataset(name: str):
    if name not in _PROVIDERS:
        raise ValueError(f"unknown dataset {name!r}; have {list(_PROVIDERS)}")
    return _PROVIDERS[name]
