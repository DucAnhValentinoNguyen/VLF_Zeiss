"""Audit the HyperKvasir tract warning across stricter pHash splits."""
from __future__ import annotations
import argparse, json, os
from collections import Counter
from copy import deepcopy

from ..cfg import load_cfg
from ..data import hyperkvasir as H

def _cfg_for(cfg, threshold, dedup=True):
    from omegaconf import OmegaConf
    out = deepcopy(cfg)
    OmegaConf.set_struct(out, False)
    out.hyperkvasir.split.phash_dedup = dedup
    out.hyperkvasir.split.phash_max_dist = threshold
    return out

def build_audit(cfg, results_dir):
    labels = H.load_label_table(cfg)
    ontology = Counter(v["tract"] for v in labels.values())
    findings = {}
    for finding in sorted({v["finding"] for v in labels.values()}):
        tracts = {v["tract"] for v in labels.values() if v["finding"] == finding}
        findings[finding] = sorted(tracts)
    splits = {}
    for name, threshold, dedup in (("raw", 0, False), ("dd6", 6, True), ("dd8", 8, True), ("dd10", 10, True)):
        sp = H.global_split(_cfg_for(cfg, threshold, dedup), force=False)
        splits[name] = {k: len(v) for k, v in sp.items()}
    rows = []
    for fp in sorted(os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".json")):
        try: d = json.load(open(fp))
        except Exception: continue
        if d.get("task") != "hkv_tract": continue
        for r in d.get("rows", []):
            rows.append({k: r.get(k) for k in ("init", "stage", "objective", "corpus", "run_tag", "accuracy", "auroc", "n_query")})
    return {"ontology": {"tract_counts": dict(ontology), "finding_tracts": findings,
                          "single_tract_findings": sum(len(x) == 1 for x in findings.values())},
            "splits": splits, "rows": rows}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None); ap.add_argument("--results", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args(); cfg = load_cfg(a.config)
    results = a.results or os.path.expanduser(str(cfg.paths.results)); out = a.out or os.path.join(results, "hkv_tract_audit.json")
    report = build_audit(cfg, results); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(report, open(out, "w"), indent=2, default=float)
    md = os.path.splitext(out)[0] + ".md"
    with open(md, "w") as f:
        f.write("# HyperKvasir tract audit\n\n")
        f.write(f"Single-tract findings: **{report['ontology']['single_tract_findings']}/{len(report['ontology']['finding_tracts'])}**.\n\n")
        f.write("| split | reference | cal | query |\n|---|---:|---:|---:|\n")
        for n, s in report["splits"].items(): f.write(f"| {n} | {s.get('reference', 0)} | {s.get('cal', 0)} | {s.get('query', 0)} |\n")
        f.write("\nExisting hkv_tract evaluation rows:\n\n")
        for r in report["rows"]: f.write(f"- `{r}`\n")
    print(f"[hkv-audit] wrote {out} and {md}")

if __name__ == "__main__": main()
