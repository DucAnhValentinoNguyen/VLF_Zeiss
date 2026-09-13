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
