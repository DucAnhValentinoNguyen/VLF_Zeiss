"""Tiny inline-SVG chart helpers for the results page (theme tokens only, no dependencies)."""
from html import escape

W = 560
FONT = 'font-family="inherit"'


def _t(x, y, s, size=10, anchor="start", fill="var(--muted)", weight=None, rot=None):
    w = f' font-weight="{weight}"' if weight else ""
    r = f' transform="rotate({rot} {x} {y})"' if rot else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"{w}{r}>{escape(str(s))}</text>'


def nice_top(v):
    """Smallest 4-tick axis maximum >= v whose tick step is a round number."""
    import math
    for k in range(-2, 9):
        for m in (1, 2, 2.5, 5):
            step = m * 10 ** k
            if m == 2.5 and k < 1:
                continue
            if 4 * step >= v * 1.04:
                return 4 * step
    return v


def fmt_int(n):
    return f"{int(round(n)):,}"


def hbar(rows, title, xlabel, label_w=170, bar_h=15, gap=5, colors=None, vmax=None, value_fmt=fmt_int,
         marks=None, aria=None):
    """Horizontal bar chart. rows = [(label, value, tag)] ; tag in {None,'dim'} -> accent / muted fill.
    marks = [(value, text)] vertical reference lines."""
    n = len(rows)
    top, bottom, right = 8, 30, 70
    h = top + n * (bar_h + gap) + bottom
    vmax = vmax or nice_top(max(v for _, v, _ in rows))
    pw = W - label_w - right
    sx = lambda v: label_w + pw * v / vmax
    out = [f'<svg viewBox="0 0 {W} {h}" width="100%" role="img" aria-label="{escape(aria or title)}">']
    ticks = 4
    for i in range(ticks + 1):
        v = vmax * i / ticks
        x = sx(v)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - bottom + 2}" stroke="var(--rule)" stroke-width="1"/>')
        out.append(_t(x, h - bottom + 14, value_fmt(v), 9, "middle"))
    for i, (lab, v, tag) in enumerate(rows):
        y = top + i * (bar_h + gap)
        fill = {"dim": "var(--muted)", "bad": "var(--bad)"}.get(tag, "var(--accent)")
        op = ' opacity=".55"' if tag == "dim" else ""
        out.append(_t(label_w - 6, y + bar_h - 4, lab, 10, "end", "var(--fg)"))
        out.append(f'<rect x="{label_w}" y="{y}" width="{max(sx(v) - label_w, 0.5):.1f}" height="{bar_h}" rx="2" fill="{fill}"{op}/>')
        out.append(_t(sx(v) + 4, y + bar_h - 4, value_fmt(v), 9, "start", "var(--fg)"))
    for v, text in (marks or []):
        x = sx(v)
        out.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{h - bottom + 2}" stroke="var(--bad)" stroke-width="1.5" stroke-dasharray="4 3"/>')
        out.append(_t(x + 4, h - bottom - 4, text, 9, "start", "var(--bad)"))
    out.append(_t(label_w + pw / 2, h - 3, xlabel, 10, "middle"))
    out.append("</svg>")
    return "".join(out)


def vbars(vals, labels, title, ylabel, xlabel, color="var(--accent)", height=190, value_fmt=fmt_int, highlight=None,
          label_every=1, aria=None, ymax=None, show_values=True, rot=None):
    """Vertical bar chart / histogram. highlight = set of indices drawn in --bad."""
    n = len(vals)
    left, right, top, bottom = 52, 10, 10, 44 if rot else 34
    pw, ph = W - left - right, height - top - bottom
    ymax = ymax or nice_top(max(vals)) or 1
    bw = pw / n
    out = [f'<svg viewBox="0 0 {W} {height}" width="100%" role="img" aria-label="{escape(aria or title)}">']
    for i in range(5):
        v = ymax * i / 4
        y = top + ph - ph * v / ymax
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{W - right}" y2="{y:.1f}" stroke="var(--rule)" stroke-width="1"/>')
        out.append(_t(left - 5, y + 3, value_fmt(v), 9, "end"))
    for i, v in enumerate(vals):
        x = left + i * bw
        hh = ph * v / ymax
        c = "var(--bad)" if highlight and i in highlight else color
        out.append(f'<rect x="{x + 1:.1f}" y="{top + ph - hh:.1f}" width="{max(bw - 2, 1):.1f}" height="{hh:.1f}" fill="{c}" rx="1.5"/>')
        if show_values and n <= 14 and v > 0:
            out.append(_t(x + bw / 2, top + ph - hh - 3, value_fmt(v), 8.5, "middle", "var(--fg)"))
        if i % label_every == 0:
            ly = top + ph + (14 if not rot else 10)
            out.append(_t(x + bw / 2, ly, labels[i], 9, "end" if rot else "middle", rot=rot) if not rot else
                       _t(x + bw / 2 + 3, ly, labels[i], 9, "end", rot=rot))
    out.append(_t(left + pw / 2, height - 3, xlabel, 10, "middle"))
    out.append(_t(11, top + ph / 2, ylabel, 10, "middle", rot=-90))
    out.append("</svg>")
    return "".join(out)


def card(title, svg, note):
    return (f'<div class="chart-card"><h4>{escape(title)}</h4>{svg}'
            f'<p class="chart-note">{note}</p></div>')
