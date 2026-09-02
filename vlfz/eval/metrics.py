"""Context metrics (accuracy / balanced-acc / macro-F1 / AUROC) + row assembly."""
from __future__ import annotations

import numpy as np


def context_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(probs).argmax(axis=1)
    C = probs.shape[1]
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auroc": float("nan"),
        "n": int(len(y_true)),
        "n_classes": int(C),
    }
    try:
        if C == 2:
            out["auroc"] = float(roc_auc_score(y_true, probs[:, 1]))
        else:
            out["auroc"] = float(
                roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
            )
    except Exception:
        pass
    return out


def flatten_report(cal_rep: dict, prefix_pre="pre", prefix_post="post") -> list[dict]:
    """calibration_report dict -> list of flat rows, one per temp_scaled in {False,True}."""
    rows = []
    pre = cal_rep["pre"]
    rows.append({
        "temp_scaled": False, "T": None,
        "ece_ew": pre["ece_ew"], "ece_adaptive": pre["ece_adaptive"],
        "nll": pre["nll"], "brier": pre["brier"],
    })
    if cal_rep.get("post") is not None:
        po = cal_rep["post"]
        rows.append({
            "temp_scaled": True, "T": cal_rep["T"],
            "ece_ew": po["ece_ew"], "ece_adaptive": po["ece_adaptive"],
            "nll": po["nll"], "brier": po["brier"],
        })
    return rows
