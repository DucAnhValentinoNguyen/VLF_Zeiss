"""long_results.csv -> a scannable markdown report."""
from __future__ import annotations

import pandas as pd

_PRIMARY = ["ece_ew", "ece_adaptive", "nll"]
_CONTEXT = ["accuracy", "balanced_acc", "auroc"]


def _fmt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "–"
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, (int,)):
        return str(v)
    try:
        return f"{float(v):.4f}"
    except Exception:
        return str(v)


def _suspect(row) -> str:
    if row.get("dataset") == "hyperkvasir" and row.get("task") == "hkv_tract":
        return " †"
    a, b = row.get("accuracy"), row.get("balanced_acc")
    try:
        if a is not None and a > 0.98:
            return " ⚠"
        if a is not None and b is not None and b > a + 0.02:
            return " ⚠"
    except Exception:
        pass
    return ""


def _table(df: pd.DataFrame) -> str:
    cols = ["dataset", "init", "objective", "corpus", "stage", "task", "protocol",
            "temp_scaled", *_PRIMARY, "dice", "miou", *_CONTEXT, "node"]
    cols = [c for c in cols if c in df.columns]
    df = df.sort_values([c for c in ["dataset", "task", "protocol", "init", "objective",
                                     "stage", "temp_scaled"] if c in df.columns])
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = [_fmt(r[c]) for c in cols]
        lines.append("| " + " | ".join(cells) + " |" + _suspect(r))
    return "\n".join(lines)


def write_pivot(df: pd.DataFrame, out_path: str, delta: pd.DataFrame | None = None):
    dsets = "/".join(sorted(df["dataset"].dropna().unique())) if "dataset" in df else "?"
    parts = [f"# VLF_Zeiss — zero-shot calibration on {dsets}\n",
             "Primary metrics: **ECE** (equal-width & adaptive) and **NLL**. "
             "`temp_scaled=T` rows are after scalar temperature scaling. "
             "⚠ = accuracy > 0.98 or balanced_acc > accuracy + 0.02 (inspect for leakage).\n",
             "† `hyperkvasir/hkv_tract` is structurally easy: its finding ontology is tract-exclusive; high accuracy is a caveat, not an automatic leakage finding.\n",
             "## All results\n", _table(df), "\n"]
    if delta is not None and not delta.empty:
        parts += ["\n## Δ (post-SSL − pre-SSL)\n",
                  "Negative Δ on ece_ew / nll = SSL improved calibration.\n",
                  delta.round(4).to_markdown(index=False), "\n"]
    with open(out_path, "w") as f:
        f.write("\n".join(str(p) for p in parts))
