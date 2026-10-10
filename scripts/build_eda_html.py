"""Builds the <details> EDA block for full-corpus-results.html from the measured JSON/NPZ files."""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eda_svg import card, fmt_int, hbar, vbars  # noqa: E402

E = os.path.expanduser("~/vlf_zeiss_fullstream/eda/")
R = os.path.expanduser("~/vlf_zeiss_fullstream/results/")
H = json.load(open(E + "hkv_eda.json"))
X = json.load(open(E + "hkv_extra.json"))
G = json.load(open(E + "gastronet_sample.json"))
M = json.load(open(os.path.expanduser("~/vlf_zeiss_fullstream/eda_manifest_stats.json")))
pa = G["per_archive"]
N_CORPUS, N_ARCH = 4823974, 506
pct = lambda a, b, d=1: f"{100 * a / b:.{d}f}%"
f0 = fmt_int

# ------------------------------------------------------------------ HyperKvasir numbers
inv, itg, ph, lk = H["inventory"], H["integrity"], H["phash"]["dataset"], H["leakage"]
n_files = inv["labelled_images_found"]
used = sum(lk["dedup_sizes"].values())
find = H["tasks"]["hkv_findings"]
used_find = sum(sum(find[s]) for s in ("reference", "cal", "query"))
res_top = itg["top_resolutions"][0]
seg = H["segmentation"]
pre = {t: json.load(open(R + f"hyperkvasir__imagenet_pre__{t}.json"))["rows"][0]["balanced_acc"]
       for t in ("hkv_findings", "hkv_category", "hkv_pathology", "hkv_tract")}

# ------------------------------------------------------------------ GastroNet sample numbers
n_s = sum(a["n_members"] for a in pa)
red = {t: sum(a["redundancy"][f"redundant_le{t}"] for a in pa) for t in (0, 2, 6)}
nbr6 = sum(a["redundancy"]["with_neighbor_le6"] for a in pa)
nn = np.sum([a["redundancy"]["nn_hist"] for a in pa], axis=0)
cols = {k: [] for k in ("w", "h", "bytes", "mean", "color", "dark")}
for f in sorted(glob.glob(E + "gastronet_sample/*.npz")):
    z = np.load(f)
    for k in cols:
        cols[k].append(z[k])
C = {k: np.concatenate(v) for k, v in cols.items()}
res = {}
for w, h in zip(C["w"].astype(int), C["h"].astype(int)):
    res[(w, h)] = res.get((w, h), 0) + 1
top_res = sorted(res.items(), key=lambda kv: -kv[1])[:8]
h512 = float((C["h"] == 512).mean())
exact = G["exact_byte_dups_across_sample"]
CI6 = (3.22, 10.62)  # archive-level bootstrap (5,000 resamples), computed in the analysis step
CI0, CI2 = (0.05, 0.14), (0.12, 0.52)
ub = 3 / n_s

# ================================================================== HTML
out = []
A = out.append
A('<details class="data-details">\n<summary>Dataset exploration &amp; data processing</summary>\n<div class="inside">')
A(f'''<p>Two datasets with different levels of certainty. <strong>HyperKvasir</strong> (the evaluation data) was
audited <em>exhaustively</em>: every one of its {f0(n_files)} images was decoded, hashed and counted.
<strong>GastroNet-5M</strong> (the self-supervised training data, {f0(N_CORPUS)} images) is far too large to
decode twice, so its integrity and redundancy were <em>measured on a random sample</em> of {len(pa)} of the
{G["zip_archives"]} zip archives ({f0(n_s)} images, {pct(n_s, N_CORPUS)} of the corpus; seed&nbsp;0) and extrapolated
with uncertainty ranges; the archive and image counts themselves are exact, read from the full manifest.
Redundancy is measured with a 64-bit perceptual hash (pHash): two frames are &ldquo;near-duplicates&rdquo; if their
hashes differ in at most 6 of 64 bits &mdash; the same threshold the evaluation split uses, so the two datasets are
directly comparable. A pHash match means visually near-identical frames; there are no patient or procedure IDs, so it
cannot prove they come from the same procedure.</p>''')

# ---------------- accounting table
A('<h3>1 &middot; Frames actually used</h3>')
A('<div class="tbl-wrap"><table><thead><tr><th>HyperKvasir step</th><th>images</th><th>share of 10,619</th><th>what happened</th></tr></thead><tbody>')
rows = [
    ("Rows in <code>image-labels.csv</code>", inv["csv_rows"], None, "the published label table"),
    ("Image files present locally", n_files, pct(n_files, n_files, 1),
     f'{inv["csv_rows"] - n_files} labelled rows have no image file (28 <code>esophagitis-b-d</code>, 15 <code>esophagitis-a</code>): the local copy is incomplete; they cannot be used'),
    ("Decodable (strict decoder)", itg["ok"], pct(itg["ok"], n_files), "0 corrupt, 0 truncated, all RGB; nothing to repair"),
    ("Exact byte duplicates", 0, "0.0%", "none found"),
    ("Near-duplicate frames (pHash &le; 6 bits)", ph["redundant_le6"], pct(ph["redundant_le6"], n_files),
     f'{f0(ph["with_neighbor_le6"])} images ({pct(ph["with_neighbor_le6"], n_files)}) have at least one near-duplicate; kept in the dataset, but see next row'),
    ("Removed from calibration/query as near-duplicates of a reference image",
     lk["dropped_cal"] + lk["dropped_query"], pct(lk["dropped_cal"] + lk["dropped_query"], n_files),
     f'calibration &minus;{lk["dropped_cal"]} of {f0(lk["raw_sizes"]["cal"])} ({lk["dropped_cal_pct"]:.1f}%), query &minus;{lk["dropped_query"]} of {f0(lk["raw_sizes"]["query"])} ({lk["dropped_query_pct"]:.1f}%): leakage control'),
    ("<strong>Scored in the 4 classification tasks</strong>", used, pct(used, n_files),
     f'reference {f0(lk["dedup_sizes"]["reference"])} + calibration {f0(lk["dedup_sizes"]["cal"])} + query {f0(lk["dedup_sizes"]["query"])}'),
    ("&hellip; of which in the 20-class findings task", used_find, pct(used_find, n_files),
     f'3 classes with &lt; 12 images ({sum(find["excluded_classes"].values())} images in total) are excluded'),
    ("Polyp segmentation set (separate files)", seg["pairs"], None, f'{seg["support"]} support + {seg["query"]} query pairs; all decodable, 0 empty masks, 0 image/mask size mismatches'),
]
for lab, n, sh, note in rows:
    A(f'<tr><td>{lab}</td><td class="num">{f0(n)}</td><td class="num">{sh or "&ndash;"}</td><td>{note}</td></tr>')
A('</tbody></table></div>')

A('<div class="tbl-wrap"><table><thead><tr><th>GastroNet-5M</th><th>count</th><th>share</th><th>how it was established</th></tr></thead><tbody>')
rows = [
    ("Portal archives", f"{N_ARCH}", "100%", "all used; the manifest build fails if any archive is missing or unreadable (exact)"),
    ("Images (frames) per epoch", f0(N_CORPUS), "100%", f'exact manifest count; per-archive min {f0(min(M["n"]))}, median {f0(int(np.median(M["n"])))}, max {f0(max(M["n"]))}; 3 epochs = {f0(3 * N_CORPUS)} image presentations'),
    ("Frames removed before training", "0", "0%", "none: the raw stream is used as stored (no resize, re-encode or de-duplication)"),
    ("Sample inspected", f0(n_s), pct(n_s, N_CORPUS), f'{len(pa)} random zip archives; the 13 <code>.7z</code> archives ({pct(13, N_ARCH)}) were not sampled'),
]
for lab, n, sh, note in rows:
    A(f'<tr><td>{lab}</td><td class="num">{n}</td><td class="num">{sh}</td><td>{note}</td></tr>')
A('</tbody></table></div>')

# ---------------- integrity table
A('<h3>2 &middot; Corrupt, truncated and duplicate frames: how many, and what was done</h3>')
A('<div class="tbl-wrap"><table><thead><tr><th>check</th><th>HyperKvasir (all 10,619)</th><th>GastroNet-5M (sample of 226,256)</th><th>treatment</th></tr></thead><tbody>')
A(f'<tr><td>Unreadable / corrupt</td><td class="num">0 (0.00%)</td><td class="num">0 (0.0000%)<br><span class="note">95% upper bound {100 * ub:.4f}% &asymp; {f0(round(ub * N_CORPUS))} of {f0(N_CORPUS)} frames</span></td><td>If a frame cannot be decoded, or its archive member is missing, a deterministic blank 224&times;224 sample takes its place, so step counts and checkpoint resume stay exact. Never triggered: 0 failures in about 13.4M decode attempts of the winning run, and 0 in the sample.</td></tr>')
A('<tr><td>Truncated PNG/JPEG</td><td class="num">0</td><td class="num">0</td><td>Decoded with truncation tolerance. <em>Caveat:</em> this means the &ldquo;0 failures&rdquo; above does not prove there are none. Before this fix, both training runs crashed on &ldquo;image file is truncated&rdquo; right after each resume, which indicates a truncated PNG exists somewhere in the full corpus (probably one image hit repeatedly; inferred from the logs, not verified). None was in the sample.</td></tr>')
A('<tr><td>Not RGB / grayscale-like</td><td class="num">0 non-RGB</td>'
  f'<td class="num">0 non-RGB; {int((C["color"] < 2).sum())} near-grayscale ({pct(int((C["color"] < 2).sum()), n_s, 3)})</td><td>Converted to RGB; nothing removed.</td></tr>')
A(f'<tr><td>Exact byte duplicates</td><td class="num">0</td><td class="num">{exact} ({100 * exact / n_s:.3f}%)</td><td>HyperKvasir: none to remove. GastroNet: kept.</td></tr>')
A(f'<tr><td>Identical pHash</td><td class="num">{ph["redundant_le0"]} ({pct(ph["redundant_le0"], n_files)})</td><td class="num">{red[0]} ({100 * red[0] / n_s:.2f}%)<br><span class="note">95% CI {CI0[0]}&ndash;{CI0[1]}%</span></td><td rowspan="3">HyperKvasir evaluation: a calibration/query image within 6 bits of any reference image is dropped (leakage control, see above); near-duplicates <em>inside</em> the scoring splits remain ({lk["residual_within_eval"]["redundant_le6"]} of {f0(lk["residual_within_eval"]["n"])}, {pct(lk["residual_within_eval"]["redundant_le6"], lk["residual_within_eval"]["n"])}). GastroNet: <strong>deliberately kept</strong>. The stream is unlabelled, so near-duplicates cannot leak a label, but they do mean the effective number of distinct frames is lower than 4.82M by roughly the percentage shown, and the most redundant archives are visited as often as the rest.</td></tr>')
A(f'<tr><td>Near-duplicate, &le; 2 bits</td><td class="num">{ph["redundant_le2"]} ({pct(ph["redundant_le2"], n_files)})</td><td class="num">{red[2]} ({100 * red[2] / n_s:.2f}%)<br><span class="note">95% CI {CI2[0]}&ndash;{CI2[1]}%</span></td></tr>')
A(f'<tr><td>Near-duplicate, &le; 6 bits</td><td class="num">{f0(ph["redundant_le6"])} ({pct(ph["redundant_le6"], n_files)})</td><td class="num">{f0(red[6])} ({100 * red[6] / n_s:.2f}%)<br><span class="note">95% CI {CI6[0]}&ndash;{CI6[1]}% &rarr; about {f0(round(0.0597 * N_CORPUS, -3))} frames of the corpus (range {f0(155000)}&ndash;{f0(512000)})</span></td></tr>')
A(f'<tr><td>Polyp-mask set (1,000 pairs)</td><td class="num">near-dup. {seg["phash"]["redundant_le6"]} ({pct(seg["phash"]["redundant_le6"], seg["pairs"])})</td><td class="num">&ndash;</td><td>Only {X["seg_query_near_support"]["le2"]} of {X["seg_query_near_support"]["n_query"]} query images is within 2 bits of a support image, so support-to-query leakage is negligible; no filtering applied.</td></tr>')
A('</tbody></table></div>')

# ---------------- HyperKvasir charts
A('<h3>3 &middot; HyperKvasir: how unbalanced, and therefore how hard</h3>')
all23 = H["findings_all23"]
rows_c = sorted(((k, v, "dim" if k in find["excluded_classes"] else None) for k, v in all23.items()), key=lambda r: -r[1])
A(card("Images per finding class (all 23 classes, local copy)",
       hbar(rows_c, "findings", "images", label_w=190, marks=[(12, "min 12")], aria="Images per finding class"),
       f'Greyed classes are excluded from the benchmark (fewer than 12 images). The largest class has {f0(max(all23.values()))} images, the smallest kept class {f0(min(v for k, v in all23.items() if k not in find["excluded_classes"]))}: an imbalance ratio of {find["balance_all"]["imbalance_ratio"]:.0f}:1.'))
q = sorted(zip(find["classes"], find["query"]), key=lambda r: -r[1])
A(card("Query images per class after de-duplication (20-class findings task)",
       hbar([(c, n, "bad" if n < 20 else None) for c, n in q], "query", "query images", label_w=190, marks=[(20, "n = 20")], aria="Query images per class"),
       f'Red: classes with fewer than 20 query images ({sum(1 for _, n in q if n < 20)} of 20; the smallest has {min(n for _, n in q)}). Per-class accuracy on these is very noisy, so the benchmark reports <em>balanced accuracy over all classes</em> with the caveat that a handful of images can move a class by tens of points.'))
dd = [(c, v["dropped"], v["eligible"]) for c, v in lk["per_finding_class"].items() if c in find["classes"]]
dd = sorted(dd, key=lambda r: -(r[1] / r[2]))
A(card("Share of calibration + query images removed by the near-duplicate filter, per class",
       hbar([(f"{c}  ({d}/{e})", 100 * d / e, "bad" if 100 * d / e > 50 else None) for c, d, e in dd], "removed", "% removed",
            label_w=235, vmax=100, value_fmt=lambda v: f"{v:.0f}%", aria="Removed share per class"),
       'The filter is concentrated in three classes: bowel-prep scores (<code>bbps-2-3</code>, <code>bbps-0-1</code>) and <code>impacted-stool</code>. Most of their held-out images were near-copies of reference images and were removed (probably because they come from runs of very similar consecutive frames; the dataset has no procedure IDs to confirm this). Near-copies of reference frames inflate k-NN accuracy, so keeping them would likely have made these classes look easier than they are.'))
A('<div class="tbl-wrap"><table><thead><tr><th>task</th><th>classes</th><th>images</th><th>largest class</th><th>imbalance max:min</th><th>effective classes</th><th>chance bal-acc</th><th>always-majority accuracy (query)</th><th>pre-SSL bal-acc</th></tr></thead><tbody>')
for t, lab in (("hkv_findings", "findings"), ("hkv_category", "category"), ("hkv_pathology", "pathology"), ("hkv_tract", "tract")):
    e = H["tasks"][t]; b = e["balance_all"]
    A(f'<tr><td>{lab}</td><td class="num">{b["k"]}</td><td class="num">{f0(b["n"])}</td><td class="num">{100 * b["majority_share"]:.1f}%</td>'
      f'<td class="num">{b["imbalance_ratio"]:.1f}:1</td><td class="num">{b["effective_classes"]:.1f}</td><td class="num">{100 * b["chance_balanced_acc"]:.0f}%</td>'
      f'<td class="num">{100 * e["majority_baseline_acc_query"]:.1f}%</td><td class="num">{100 * pre[t]:.1f}%</td></tr>')
A('</tbody><caption>&ldquo;Effective classes&rdquo; = exp(entropy of the class shares); equal to the class count only for perfectly balanced data. '
  '&ldquo;Always-majority accuracy&rdquo; is what a model that ignores the image would score on the query set. Reading: the 20-way task has 41:1 imbalance and a strong frozen baseline of 60.6% against 5% chance, '
  'so it has the most headroom and is where SSL moves things; <code>tract</code> starts at 99.4% against a 62.2% majority baseline, so at most 0.6 points remain and its near-zero deltas are a ceiling effect, not evidence of no change.</caption></table></div>')
edges = ["&lt;0.5%", "0.5&ndash;1%", "1&ndash;2%", "2&ndash;5%", "5&ndash;10%", "10&ndash;20%", "20&ndash;40%", "&ge;40%"]
A(card("Polyp size in the segmentation set (fraction of the frame covered by the mask)",
       vbars(seg["fg_hist"], [e.replace("&lt;", "<").replace("&ge;", "≥").replace("&ndash;", "–") for e in edges], "polyp size", "masks", "mask area / frame area", label_every=1),
       f'Median polyp covers {100 * seg["fg_fraction_quantiles"]["0.5"]:.1f}% of the frame (middle half {100 * seg["fg_fraction_quantiles"]["0.25"]:.1f}&ndash;{100 * seg["fg_fraction_quantiles"]["0.75"]:.1f}%, 5th&ndash;95th percentile {100 * seg["fg_fraction_quantiles"]["0.05"]:.1f}&ndash;{100 * seg["fg_fraction_quantiles"]["0.95"]:.1f}%). '
       f'{sum(seg["fg_hist"][:4])} of 1,000 masks are below 5% of the frame, where a few misplaced patches move Dice a lot: a hard regime for 14&times;14 patch-token transfer.'))

# ---------------- GastroNet charts
A('<h3>4 &middot; GastroNet-5M: what the training stream looks like</h3>')
cnt = np.histogram(M["n"], bins=[0, 4000, 6000, 8000, 9000, 9500, 10000, 14000])[0].tolist()
A(card("Images per archive (all 506 archives, exact)",
       vbars(cnt, ["<4k", "4–6k", "6–8k", "8–9k", "9–9.5k", "9.5–10k", "≥10k"], "archive sizes", "archives", "images per archive"),
       f'{cnt[5]} of 506 archives hold 9.5&ndash;10k images; a small number are much smaller ({sum(1 for m in M["n"] if m < 8000)} archives hold under 8,000 images, together {sum(m for m in M["n"] if m < 8000) / N_CORPUS * 100:.1f}% of all frames). All {N_ARCH} are used in every epoch.'))
tr = [(f"{w}×{h}", 100 * c / n_s, None) for (w, h), c in top_res]
other = 100 - sum(v for _, v, _ in tr)
A(card("Frame size in the sample (8 most common of {} distinct sizes)".format(f0(len(res))),
       hbar(tr + [("all other sizes", other, "dim")], "sizes", "% of sampled frames", label_w=110, value_fmt=lambda v: f"{v:.0f}%" if v >= 10 else f"{v:.1f}%", aria="Frame sizes"),
       f'Frames are not square and not a fixed size ({100 * h512:.0f}% are 512 pixels high, widths vary), stored as RGB PNG (mean {np.mean(C["bytes"]) / 1e3:.0f}&nbsp;KB per frame). The training transform resizes and crops them to the network input; no re-encoding is applied to the stored data.'))
hist, _ = np.histogram(C["mean"], bins=range(0, 257, 16))
A(card("Brightness of the sampled frames (mean grey level, 0 = black, 255 = white)",
       vbars([100 * h / n_s for h in hist], [str(b) for b in range(0, 256, 16)], "brightness", "% of frames", "mean grey level (bin start)", value_fmt=lambda v: f"{v:.0f}%", show_values=False),
       f'Central 90% of frames lie between grey level {np.percentile(C["mean"], 5):.0f} and {np.percentile(C["mean"], 95):.0f}. Practically no unusable frames: {int((C["mean"] < 15).sum())} near-black and {int((C["mean"] > 240).sum())} near-white frame in {f0(n_s)}; {int((C["dark"] > 0.5).sum())} frames are more than half dark pixels (borders, dark views). So blank-filling as a fallback has almost nothing to compensate for.'))
nnb = [int(nn[i]) for i in range(0, 22, 2)]
nnb[-1] += int(nn[22:].sum())
A(card("How redundant is the stream? Distance from each frame to its nearest other frame in the same archive",
       vbars([100 * v / n_s for v in nnb], [str(i) for i in range(0, 20, 2)] + ["20+"], "nearest-neighbour distance", "% of frames", "pHash Hamming distance (bits)", highlight={0, 1, 2, 3}, value_fmt=lambda v: f"{v:.0f}%", show_values=False),
       f'Red = within 6 bits, i.e. near-duplicate: {100 * nbr6 / n_s:.1f}% of sampled frames have such a neighbour (HyperKvasir, for comparison, {100 * ph["with_neighbor_le6"] / n_files:.1f}%). Most frames sit 10&ndash;14 bits from their nearest neighbour; a typical frame is genuinely different from every other in its archive.'))
per = sorted(((100 * a["redundancy"]["redundant_le6"] / a["n_members"], a["archive"].replace(".zip", "")) for a in pa), reverse=True)
A(card("Near-duplicate rate varies by archive (24 sampled archives, share of frames removable at ≤ 6 bits)",
       vbars([v for v, _ in per], [a for _, a in per], "per-archive redundancy", "% redundant", "archive", highlight={i for i, (v, _) in enumerate(per) if v > 10}, value_fmt=lambda v: f"{v:.0f}%", show_values=False, label_every=2),
       f'Typical archives have {np.median([v for v, _ in per]):.1f}% (median); archive <code>{per[0][1]}</code> has {per[0][0]:.0f}%. This heterogeneity is why the corpus-level estimate is a range ({CI6[0]}&ndash;{CI6[1]}%, bootstrap over archives), not a point.'))

# ---------------- limits
A('<h3>5 &middot; What this analysis does not show</h3><ul>')
A(f'<li><strong>GastroNet numbers are sample estimates</strong> (24 of {G["zip_archives"]} zip archives); the 13 <code>.7z</code> archives were not sampled, and a whole-archive defect could be missed. The zero-corruption bound assumes frames are independent.</li>')
A('<li><strong>No labels, no class balance:</strong> GastroNet is unlabelled, so no class or anatomical-site distribution (including the reported upper-GI skew) was measured. Non-endoscopic frames, blur and glare were not detected, only integrity, size, brightness and redundancy.</li>')
A('<li><strong>pHash is a proxy:</strong> it finds visually near-identical frames, not same-patient frames. Different thresholds give different counts (0.09% at 0 bits, 0.28% at 2, 5.97% at 6 in GastroNet), so the 6-bit figure is a choice, applied identically to both datasets.</li>')
A('<li><strong>HyperKvasir has no patient or procedure ID,</strong> so splits are image-level; the near-duplicate filter reduces but cannot remove leakage between splits.</li>')
A('<li>The earlier text of this section quoted 10,662 labelled images; that is the label-table row count. 10,619 image files exist locally, and 10,619 were used.</li></ul>')
A("</div>\n</details>")
HTML = "\n".join(out)
if __name__ == "__main__":
    open(os.path.expanduser("~/vlf_zeiss_fullstream/eda/eda_block.html"), "w").write(HTML)
    print("block chars", len(HTML), "| svgs", HTML.count("<svg"))
