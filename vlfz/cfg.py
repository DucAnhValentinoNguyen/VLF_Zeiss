"""Config loading (OmegaConf) + run provenance.

`load_cfg` resolves `${oc.env:...}` interpolations against the current environment
(lrz/job_env.sh exports DATA_ROOT / OUT_ROOT / STAGE / OBJ / INIT before jobs run).
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

_DEFAULT_CFG = Path(__file__).resolve().parents[1] / "config.yaml"


def load_cfg(path: str | os.PathLike | None = None):
    from omegaconf import OmegaConf

    p = Path(path) if path else _DEFAULT_CFG
    cfg = OmegaConf.load(p)
    # eagerly resolve so downstream code gets plain str/int, not interpolation nodes
    OmegaConf.resolve(cfg)
    return cfg


def expand(p: str | os.PathLike) -> str:
    """~ and $VARS -> absolute path string."""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(str(p))))


def ensure_dirs(*paths: str | os.PathLike) -> None:
    for p in paths:
        Path(expand(p)).mkdir(parents=True, exist_ok=True)


def _pkg_version(mod: str) -> str:
    try:
        return __import__(mod).__version__
    except Exception:
        return "n/a"


def git_sha(repo: str | os.PathLike | None = None) -> str:
    repo = str(repo) if repo else str(Path(__file__).resolve().parents[1])
    try:
        return (
            subprocess.check_output(
                ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "n/a"


def provenance(seed: int, **extra: Any) -> dict:
    """Reproducibility stamp embedded in every results JSON."""
    cuda, gpu, gpu_count, gpu_vram_gb = "cpu", "cpu", 0, 0.0
    try:
        import torch

        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            cuda = torch.version.cuda
            gpu = p.name                                   # e.g. "NVIDIA H100 94GB" / "NVIDIA A100-SXM4-80GB"
            gpu_count = torch.cuda.device_count()          # visible to this job (--gres=gpu:N)
            gpu_vram_gb = round(p.total_memory / 2**30, 1)
    except Exception:
        cuda, gpu = "n/a", "n/a"
    out = {
        "seed": seed,
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "torch": _pkg_version("torch"),
        "timm": _pkg_version("timm"),
        "open_clip": _pkg_version("open_clip"),
        "numpy": _pkg_version("numpy"),
        "sklearn": _pkg_version("sklearn"),
        "cuda": cuda,
        "gpu": gpu,
        "gpu_count": gpu_count,
        "gpu_vram_gb": gpu_vram_gb,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION", ""),
        "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST", ""),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out.update(extra)
    return out


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
