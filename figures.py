"""figures.py -- the README figures, as dependency-free SVG.

No matplotlib in this project and no reason to add it: three hand-drawn SVGs are
smaller, sharper at any zoom, and diff as text. Colours are mid-tone throughout
so the figures read on both light and dark GitHub themes -- GitHub does not let
embedded SVG inherit the page's text colour reliably, so nothing here depends on
that.

    python figures.py            # writes docs/*.svg

  fig1  what the task is: a braid word, its closure, the invariant
  fig2  the dose-response curve and its turnover
  fig3  in-distribution versus extrapolation, per model
"""

from __future__ import annotations

import os

import numpy as np

import braids as B

OUT = "docs"
INK = "#5b6470"      # axes, text
GRID = "#c8ced6"
BRAID = "#4a6fa5"    # strands
HOT = "#c0562e"      # the highlighted series
GOOD = "#2f7d5d"
MUTE = "#8c95a1"


def _svg(w, h, body, title):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="Helvetica,Arial,sans-serif">'
            f'<title>{title}</title>{body}</svg>\n')


def _text(x, y, s, size=13, fill=INK, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>')


# ---------------------------------------------------------------- figure 1

def braid_diagram(word, s, x0, y0, dx, dy, closure=True):
    """Draw a braid word left-to-right, optionally with its Markov closure.

    Strand j runs at height y0 + j*dy. A letter (i, sign) swaps tracks i and
    i+1 over one dx step; the UNDER strand is drawn with a gap so the crossing
    reads correctly, which is the whole point -- the sign is the only thing
    distinguishing a knot from its mirror.
    """
    out, L = [], len(word)
    pos = list(range(s))                     # pos[track] = strand id
    paths = {j: [(x0, y0 + j * dy)] for j in range(s)}
    for t, (i, sgn) in enumerate(word):
        x = x0 + t * dx
        a, b = pos[i], pos[i + 1]
        paths[a].append((x + dx, y0 + (i + 1) * dy))
        paths[b].append((x + dx, y0 + i * dy))
        for j in range(s):
            if j not in (i, i + 1):
                paths[pos[j]].append((x + dx, y0 + j * dy))
        pos[i], pos[i + 1] = pos[i + 1], pos[i]
        out.append(("over" if sgn > 0 else "under", x, i))
    xe = x0 + L * dx
    if closure:
        for j in range(s):
            yj = y0 + j * dy
            r = 14 + 9 * j
            out.append(("arc", xe, yj, r))

    body = []
    for j in range(s):
        pts = paths[j]
        d = f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"
        for k in range(1, len(pts)):
            x1, y1 = pts[k - 1]
            x2, y2 = pts[k]
            d += f" C {(x1+x2)/2:.1f} {y1:.1f} {(x1+x2)/2:.1f} {y2:.1f} {x2:.1f} {y2:.1f}"
        body.append(f'<path d="{d}" fill="none" stroke="{BRAID}" '
                    f'stroke-width="2.4" stroke-linecap="round"/>')
    # redraw a short white gap on the under-strand of each crossing
    for item in out:
        if item[0] == "under":
            _, x, i = item
            body.append(f'<circle cx="{x+dx/2:.1f}" cy="{y0+(i+0.5)*dy:.1f}" '
                        f'r="5.5" fill="#ffffff" opacity="0.92"/>')
    if closure:
        for item in out:
            if item[0] == "arc":
                _, xe_, yj, r = item
                body.append(
                    f'<path d="M {xe_:.1f} {yj:.1f} C {xe_+r:.1f} {yj:.1f} '
                    f'{xe_+r:.1f} {y0-14:.1f} {xe_-r*0.1:.1f} {y0-14:.1f} '
                    f'L {x0:.1f} {y0-14:.1f} C {x0-r:.1f} {y0-14:.1f} '
                    f'{x0-r:.1f} {yj:.1f} {x0:.1f} {yj:.1f}" fill="none" '
                    f'stroke="{MUTE}" stroke-width="1.6" stroke-dasharray="4 3"/>')
    return "".join(body)


def fig1():
    W, H = 720, 320
    b = [f'<rect width="{W}" height="{H}" fill="none"/>']
    b.append(_text(24, 30, "A braid word is a token sequence; its closure is a knot",
                   16, INK, weight="600"))
    b.append(_text(24, 52, "The model input is the token sequence — integers, "
                           "not pixels. The drawing is for the reader.", 13, MUTE))
    b.append(_text(24, 72, "encode(σ₁σ₁σ₁) → [2, 2, 2, 0, 0, …]     "
                           "target: the Jones polynomial of the closure",
                   12, GOOD))

    tre = [(0, 1)] * 3
    b.append(braid_diagram(tre, 2, 90, 128, 46, 40))
    b.append(_text(90, 196, "σ₁ σ₁ σ₁", 14, BRAID))
    b.append(_text(90, 215, "trefoil", 12, MUTE))
    v = B.jones(tre, 2, np.exp(1j * 0.255))
    b.append(_text(90, 234, f"V = {v.real:+.3f} {v.imag:+.3f}i", 11, INK))

    fig8 = [(0, 1), (1, -1), (0, 1), (1, -1)]
    b.append(braid_diagram(fig8, 3, 330, 128, 46, 40))
    b.append(_text(330, 236, "σ₁ σ₂⁻¹ σ₁ "
                             "σ₂⁻¹", 14, BRAID))
    b.append(_text(330, 255, "figure-eight", 12, MUTE))
    v8 = B.jones(fig8, 3, np.exp(1j * 0.255))
    b.append(_text(330, 274, f"V = {v8.real:+.3f} {v8.imag:+.3f}i", 11, INK))

    b.append(_text(560, 138, "Two words can present", 12, MUTE))
    b.append(_text(560, 155, "the same knot while", 12, MUTE))
    b.append(_text(560, 172, "looking nothing alike.", 12, MUTE))
    b.append(_text(560, 196, "Telling them apart has", 12, MUTE))
    b.append(_text(560, 213, "no local shortcut —", 12, MUTE))
    b.append(_text(560, 230, "that is why the symmetry", 12, MUTE))
    b.append(_text(560, 247, "is worth building in.", 12, MUTE))
    return _svg(W, H, "".join(b), "Braid words and their closures")


# ---------------------------------------------------------------- figure 2

def _axes(x0, y0, w, h, xlab, ylab, xticks, yticks, xlog=False):
    b = [f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0-h}" stroke="{INK}" '
         f'stroke-width="1.4"/>',
         f'<line x1="{x0}" y1="{y0}" x2="{x0+w}" y2="{y0}" stroke="{INK}" '
         f'stroke-width="1.4"/>']
    for v, px in xticks:
        b.append(f'<line x1="{px:.1f}" y1="{y0}" x2="{px:.1f}" y2="{y0+5}" '
                 f'stroke="{INK}" stroke-width="1.2"/>')
        b.append(_text(px, y0 + 20, v, 11, INK, "middle"))
    for v, py in yticks:
        b.append(f'<line x1="{x0-5}" y1="{py:.1f}" x2="{x0+w}" y2="{py:.1f}" '
                 f'stroke="{GRID}" stroke-width="0.8"/>')
        b.append(_text(x0 - 9, py + 4, v, 11, INK, "end"))
    b.append(_text(x0 + w / 2, y0 + 40, xlab, 12, INK, "middle"))
    b.append(f'<g transform="translate({x0-46},{y0-h/2}) rotate(-90)">'
             + _text(0, 0, ylab, 12, INK, "middle") + '</g>')
    return "".join(b)


def fig2(sweep):
    """sweep: list of (w_ybe, residual, extrapolation R2)."""
    W, H = 620, 380
    x0, y0, pw, ph = 90, 300, 470, 220
    res = np.array([r for _, r, _ in sweep])
    ex = np.array([e for _, _, e in sweep])
    lx = np.log10(res)
    xmin, xmax = lx.min() - 0.2, lx.max() + 0.2
    ymin, ymax = 0.0, 0.26
    X = lambda v: x0 + (np.log10(v) - xmin) / (xmax - xmin) * pw
    Y = lambda v: y0 - (v - ymin) / (ymax - ymin) * ph

    b = [_text(24, 30, "Extrapolation tracks how closely the braid relation holds",
               16, INK, weight="600"),
         _text(24, 52, "Same network, same data — only the Yang–Baxter "
                       "penalty weight varies", 13, MUTE)]
    xt = [(f"10{'⁻' if e < 0 else ''}{'¹²³⁴'[abs(e)-1] if abs(e) in (1,2,3,4) else abs(e)}", X(10.0 ** e))
          for e in (-4, -3, -2, -1)]
    yt = [(f"{v:.2f}", Y(v)) for v in (0.05, 0.10, 0.15, 0.20, 0.25)]
    b.append(_axes(x0, y0, pw, ph, "Yang–Baxter residual  (lower = closer "
                                   "to a braid representation)",
                   "extrapolation R²", xt, yt))

    d = " ".join(f"{'M' if k == 0 else 'L'} {X(res[k]):.1f} {Y(ex[k]):.1f}"
                 for k in range(len(res)))
    b.append(f'<path d="{d}" fill="none" stroke="{HOT}" stroke-width="2.4"/>')
    best = int(np.argmax(ex))
    for k in range(len(res)):
        r = 6.5 if k == best else 4.5
        b.append(f'<circle cx="{X(res[k]):.1f}" cy="{Y(ex[k]):.1f}" r="{r}" '
                 f'fill="{HOT if k == best else "#ffffff"}" stroke="{HOT}" '
                 f'stroke-width="2"/>')
    b.append(_text(X(res[best]), Y(ex[best]) - 16,
                   f"peak  w={sweep[best][0]:g}", 12, HOT, "middle", "600"))
    b.append(_text(X(res[0]) + 6, Y(ex[0]) + 20, "no penalty", 11, MUTE))
    b.append(_text(X(res[-1]), Y(ex[-1]) + 24, "hardest enforcement", 11, MUTE,
                   "middle"))
    b.append(f'<line x1="{x0}" y1="{Y(0.174):.1f}" x2="{x0+pw}" '
             f'y2="{Y(0.174):.1f}" stroke="{MUTE}" stroke-width="1.6" '
             f'stroke-dasharray="6 4"/>')
    b.append(_text(x0 + pw, Y(0.174) - 8, "tuned transformer baseline", 11,
                   MUTE, "end"))
    b.append(_text(x0 + 10, Y(0.245), "approximate invariance beats exact",
                   12, INK, "start", "600"))
    return _svg(W, H, "".join(b), "Yang-Baxter dose-response")


# ---------------------------------------------------------------- figure 3

def fig3(rows):
    """rows: list of (label, test R2, extrapolation R2, is_braid)."""
    W, H = 620, 330
    x0, y0, pw, ph = 130, 250, 420, 175
    ymax = 0.75
    Y = lambda v: y0 - v / ymax * ph
    b = [_text(24, 30, "The gap opens where the data runs out", 16, INK,
               weight="600"),
         _text(24, 52, "Trained on braid words of length 4–10, tested on "
                       "12–16", 13, MUTE)]
    yt = [(f"{v:.1f}", Y(v)) for v in (0.2, 0.4, 0.6)]
    b.append(_axes(x0, y0, pw, ph, "", "R²", [], yt))
    n = len(rows)
    slot = pw / n
    for k, (lab, te, ex, isb) in enumerate(rows):
        cx = x0 + slot * (k + 0.5)
        col = GOOD if isb else MUTE
        b.append(f'<rect x="{cx-26:.1f}" y="{Y(te):.1f}" width="24" '
                 f'height="{y0-Y(te):.1f}" fill="{col}" opacity="0.35"/>')
        b.append(f'<rect x="{cx+2:.1f}" y="{Y(ex):.1f}" width="24" '
                 f'height="{y0-Y(ex):.1f}" fill="{col}"/>')
        for li, part in enumerate(lab.split("\n")):
            b.append(_text(cx, y0 + 18 + li * 13, part, 11, INK, "middle"))
        b.append(_text(cx + 14, Y(ex) - 6, f"{ex:.3f}", 10, col, "middle", "600"))
    b.append(f'<rect x="{x0+pw-150}" y="{y0-ph-24}" width="12" height="12" '
             f'fill="{MUTE}" opacity="0.35"/>')
    b.append(_text(x0 + pw - 132, y0 - ph - 14, "in distribution", 11, INK))
    b.append(f'<rect x="{x0+pw-46}" y="{y0-ph-24}" width="12" height="12" '
             f'fill="{MUTE}"/>')
    b.append(_text(x0 + pw - 28, y0 - ph - 14, "extrapolation", 11, INK))
    return _svg(W, H, "".join(b), "In-distribution versus extrapolation")


def fig4(chain):
    """Enforcement -> invariance -> extrapolation, with invariance measured on
    synthetic Reidemeister-III word pairs that never appear in training."""
    W, H = 620, 330
    x0, y0, pw, ph = 110, 250, 400, 170
    b = [_text(24, 30, "The gain is traceable to invariance, not to the "
                       "architecture", 16, INK, weight="600"),
         _text(24, 52, "R-III ratio measured on word pairs that never appear "
                       "in training  (r = −0.994)", 13, MUTE)]
    rmin, rmax = 0.48, 0.96
    emin, emax = 0.10, 0.30
    X = lambda r: x0 + (rmax - r) / (rmax - rmin) * pw
    Y = lambda e: y0 - (e - emin) / (emax - emin) * ph
    xt = [(f"{v:.1f}", X(v)) for v in (0.9, 0.8, 0.7, 0.6, 0.5)]
    yt = [(f"{v:.2f}", Y(v)) for v in (0.10, 0.15, 0.20, 0.25, 0.30)]
    b.append(_axes(x0, y0, pw, ph,
                   "more invariant  →        (Reidemeister-III ratio)",
                   "extrapolation R²", xt, yt))
    d = " ".join(f"{'M' if k == 0 else 'L'} {X(r):.1f} {Y(e):.1f}"
                 for k, (_l, r, e) in enumerate(chain))
    b.append(f'<path d="{d}" fill="none" stroke="{GOOD}" stroke-width="2.4"/>')
    for k, (lab, r, e) in enumerate(chain):
        b.append(f'<circle cx="{X(r):.1f}" cy="{Y(e):.1f}" r="6" fill="{GOOD}"/>')
        for li, part in enumerate(lab.split("\n")):
            b.append(_text(X(r), Y(e) - 26 + li * 13, part, 11, INK, "middle"))
    b.append(f'<line x1="{x0}" y1="{Y(0.174):.1f}" x2="{x0+pw}" '
             f'y2="{Y(0.174):.1f}" stroke="{MUTE}" stroke-width="1.6" '
             f'stroke-dasharray="6 4"/>')
    b.append(_text(x0 + 6, Y(0.174) - 7, "tuned transformer baseline", 11, MUTE))
    return _svg(W, H, "".join(b), "Invariance versus extrapolation")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    SWEEP = [(0, 1.18e-01, 0.141), (0.1, 6.27e-02, 0.153), (1, 1.68e-02, 0.199),
             (3, 7.89e-03, 0.213), (10, 3.35e-03, 0.225), (30, 1.15e-03, 0.208),
             (100, 2.90e-04, 0.190)]
    ROWS = [("tf-rope", 0.537, 0.174, False), ("braid", 0.631, 0.128, True),
            ("braid-ybe\nuntied", 0.681, 0.186, True),
            ("braid-ybe\ntied", 0.760, 0.269, True)]
    CHAIN = [("braid\nno penalty", 0.912, 0.128),
             ("untied\nw=10", 0.723, 0.186),
             ("tied\nw=10", 0.543, 0.267)]
    for name, svg in (("fig1_braids", fig1()), ("fig2_doseresponse", fig2(SWEEP)),
                      ("fig3_extrapolation", fig3(ROWS)),
                      ("fig4_mechanism", fig4(CHAIN))):
        path = os.path.join(OUT, name + ".svg")
        with open(path, "w") as f:
            f.write(svg)
        print(f"wrote {path} ({len(svg)} bytes)")
