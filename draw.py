"""draw.py -- render reconstructed contours to SVG.

The token NLL is the metric, but it hides the thing that matters most here:
whether the curve the model draws is actually a closed curve. A completion can
score well per-token and still leave a visible gap where the free arc fails to
meet the observed one, because closure is a global condition and per-token loss
never sees it. So the shapes are drawn.
"""

from __future__ import annotations

import numpy as np


def polyline(d, close_gap=False):
    """Vertices of the curve with turning angles d, unit edge length.

    The curve is drawn as the model actually specifies it: start at the origin,
    step one unit in the direction given by the accumulated turning. If the
    angles do not close, the polyline does not close, and the gap between the
    last vertex and the first is exactly the closure error times n.
    """
    phi = np.cumsum(d, -1) - d[..., :1]
    step = np.stack([np.cos(phi), np.sin(phi)], -1)
    pts = np.concatenate([np.zeros_like(step[..., :1, :]),
                          np.cumsum(step, -2)], -2)
    return pts if close_gap else pts[..., :-1, :]


def _fit(pts, size, pad):
    lo, hi = pts.min(0), pts.max(0)
    sc = (size - 2 * pad) / max((hi - lo).max(), 1e-9)
    off = (size - (hi - lo) * sc) / 2 - lo * sc
    return pts * sc + off


def _path(pts, close=True):
    s = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
    return f"M {s.replace(',', ' ', 0)}" if False else \
        "M " + " L ".join(f"{x:.2f} {y:.2f}" for x, y in pts) + (" Z" if close else "")


def panel(true_d, pred_d, mask, size=150, pad=12):
    """One cell: the true curve behind, the reconstruction in front, and the
    occluded arc picked out. Both curves are fitted with the SAME transform so
    the comparison is geometric and not an artefact of rescaling."""
    tp = polyline(true_d, close_gap=True)
    pp = polyline(pred_d, close_gap=True)
    both = np.concatenate([tp, pp], 0)
    lo, hi = both.min(0), both.max(0)
    sc = (size - 2 * pad) / max((hi - lo).max(), 1e-9)
    tp = (tp - lo) * sc + pad
    pp = (pp - lo) * sc + pad
    free = np.where(mask == 0)[0]
    out = [f'<rect width="{size}" height="{size}" fill="none" '
           f'stroke="#e5e5e5" stroke-width="1"/>']
    out.append(f'<path d="{_path(tp[:-1])}" fill="none" stroke="#c9c9c9" '
               f'stroke-width="2.4"/>')
    # the reconstruction, drawn open so a failure to close is visible
    out.append(f'<path d="{_path(pp, close=False)}" fill="none" '
               f'stroke="#2563eb" stroke-width="1.8"/>')
    if len(free):
        seg = pp[free.min():free.max() + 2]
        out.append(f'<path d="{_path(seg, close=False)}" fill="none" '
                   f'stroke="#dc2626" stroke-width="2.6"/>')
    # the closure gap: last vertex back to the first
    out.append(f'<line x1="{pp[-1,0]:.2f}" y1="{pp[-1,1]:.2f}" '
               f'x2="{pp[0,0]:.2f}" y2="{pp[0,1]:.2f}" stroke="#dc2626" '
               f'stroke-width="1" stroke-dasharray="3 3"/>')
    return "".join(out)


def grid(rows, size=150, gap=8, labels=None):
    """rows: list of lists of (true_d, pred_d, mask)."""
    ncol = max(len(r) for r in rows)
    lab = 18 if labels else 0
    W = ncol * (size + gap) + gap
    H = len(rows) * (size + gap + lab) + gap
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}"><rect width="{W}" height="{H}" fill="white"/>']
    for r, row in enumerate(rows):
        y = gap + r * (size + gap + lab)
        if labels:
            out.append(f'<text x="{gap}" y="{y + 12}" font-family="ui-monospace,'
                       f'monospace" font-size="11" fill="#444">{labels[r]}</text>')
        for c, (td, pd, mk) in enumerate(row):
            x = gap + c * (size + gap)
            out.append(f'<g transform="translate({x},{y + lab})">'
                       f'{panel(td, pd, mk, size)}</g>')
    out.append("</svg>")
    return "".join(out)
