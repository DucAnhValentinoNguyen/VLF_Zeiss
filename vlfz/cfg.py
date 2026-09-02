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
    try:
        import torch

        cuda = torch.version.cuda if torch.cuda.is_available() else "cpu"
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
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
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
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
