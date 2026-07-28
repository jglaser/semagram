"""contours.py -- real closed curves as turning-angle sequences on Z_n.

Why this representation, and why it is the honest test of the Semagram claims.

`semagram.py` reports that the architecture LOSES to a parameter-matched
bidirectional transformer on Shakespeare. That is the correct result: text is
not a loop. A 48-character window of a novel has a first character and a last
character, they are not neighbours, and the model's central commitment -- that
position lives on S^1 and there is no index 0 -- is simply false of the data.
Every one of the three commitments is a PRIOR, and a prior that is false costs
you.

So the question this file exists to answer is what data the priors are TRUE of.

A closed planar curve resampled to n points of equal arc length is that data,
and not by analogy:

  * It is literally a function on Z_n. Where you start tracing a closed curve
    is a gauge choice with no geometric content, so cyclic shift is an exact
    symmetry of the data, not an approximation. This is commitment 1, and here
    it is a fact about coastlines rather than a modelling convenience.

  * With equal arc length every edge has the same length L/n, so the entire
    shape is carried by the TURNING ANGLE d_i at each vertex. The state at a
    vertex is an element of SO(2). This is commitment 3's per-edge transport,
    except that here the transports are the data.

  * Going once around the loop must return you to yourself. Two closure
    identities hold exactly, for every simple closed curve:

        sum_i d_i          = 2*pi        (Hopf's Umlaufsatz: turning number 1)
        sum_i exp(i Phi_i) = 0           (the curve actually closes)

    The first IS the holonomy that `semagram.holonomy` computes and that the
    README admits "earns nothing measurable" on text. On text there is no such
    identity to earn anything with. Here there is.

The second identity is the interesting one. It is global, it couples every
vertex to every other, and no token-wise decoder can enforce it: a transformer
emits n independent distributions and the curve it draws simply does not close.
An energy-based layer can have it added as a term at inference time. That is
what task_shape.py measures.

DATA. Three real sources, no synthetic shapes anywhere in this file.

  ne_lakes   Natural Earth 10m lake polygons -- real measured shorelines.
  ne_admin1  Natural Earth 10m admin-1 boundaries -- real coast/border rings.
  mnist      Outer contours of MNIST handwritten digits -- real handwriting.

Geography and handwriting are deliberately unlike each other. If the result
only held on one of them it would be a fact about that dataset.
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np

DATA_DIR = "data"

SOURCES = {
    "ne_lakes": ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
                 "/master/geojson/ne_10m_lakes.geojson", "ne_10m_lakes.geojson"),
    "ne_admin1": ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector"
                  "/master/geojson/ne_10m_admin_1_states_provinces.geojson",
                  "ne_10m_admin_1_states_provinces.geojson"),
    "mnist": ("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
              "mnist.npz"),
    "fashion:train": ("https://storage.googleapis.com/tensorflow/tf-keras-datasets"
                      "/train-images-idx3-ubyte.gz",
                      "fashion-train-images-idx3-ubyte.gz"),
    "fashion:test": ("https://storage.googleapis.com/tensorflow/tf-keras-datasets"
                     "/t10k-images-idx3-ubyte.gz",
                     "fashion-t10k-images-idx3-ubyte.gz"),
}


def _fetch(name):
    url, fn = SOURCES[name]
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, fn)
    if not os.path.exists(path):
        print(f"  downloading {name} ...")
        urllib.request.urlretrieve(url, path)
    return path


# ----------------------------------------------------------------------------
# raw rings

def _geojson_rings(name):
    """Every linear ring in a GeoJSON file, as (m, 2) float arrays.

    Longitude is scaled by cos(mean latitude) so that an equirectangular ring is
    not stretched east-west; without it every polygon near the poles turns into
    a horizontal smear and the turning-angle distribution measures the map
    projection instead of the coastline.
    """
    d = json.load(open(_fetch(name)))
    out = []
    for f in d["features"]:
        g = f.get("geometry")
        if g is None:
            continue
        polys = ([g["coordinates"]] if g["type"] == "Polygon"
                 else g["coordinates"] if g["type"] == "MultiPolygon" else [])
        for poly in polys:
            for ring in poly:
                a = np.asarray(ring, dtype=np.float64)[:, :2]
                if len(a) < 4:
                    continue
                if np.allclose(a[0], a[-1]):
                    a = a[:-1]
                a[:, 0] *= np.cos(np.deg2rad(np.clip(a[:, 1].mean(), -85, 85)))
                out.append(a)
    return out


def _mnist_rings(split):
    """Outer contour of each MNIST digit, via marching squares at the mid level.

    The image is padded so a digit touching the border still yields a closed
    curve, and only the longest contour is kept -- the outer boundary. Holes
    (the inside of a 0 or an 8) are separate contours and are discarded, since
    the task is one closed curve per example.

    The level is 128, not 0.5. MNIST is greyscale with antialiased strokes, so
    the half-intensity isocontour is interpolated to subpixel accuracy and is a
    smooth curve; thresholding just above zero instead returns the staircase
    boundary of the pixel grid, whose turning angles are a property of the
    raster and not of the handwriting.
    """
    from skimage import measure
    z = np.load(_fetch("mnist"))
    imgs = z["x_train" if split == "train" else "x_test"]
    return _raster_rings(imgs)


def _fashion_rings(split):
    """Outer silhouette of each Fashion-MNIST garment photo.

    Deliberately a different world from handwriting: these are outlines of
    photographed objects (shoes, bags, shirts), so they are convex-ish and
    smooth where digits are thin and looping. A result that held on one and not
    the other would be a fact about the dataset rather than about the geometry.
    """
    import gzip
    path = _fetch(f"fashion:{'train' if split == 'train' else 'test'}")
    with gzip.open(path, "rb") as f:
        buf = f.read()
    cnt = int.from_bytes(buf[4:8], "big")
    imgs = np.frombuffer(buf[16:], dtype=np.uint8).reshape(cnt, 28, 28)
    return _raster_rings(imgs)


def _raster_rings(imgs):
    from skimage import measure
    out = []
    for im in imgs:
        pad = np.pad(im.astype(np.float64), 2)
        cs = measure.find_contours(pad, 128.0)
        if not cs:
            continue
        c = max(cs, key=len)
        if len(c) < 20 or not np.allclose(c[0], c[-1]):
            continue
        out.append(c[:-1, ::-1].copy())   # (row, col) -> (x, y)
    return out


# ----------------------------------------------------------------------------
# geometry

def signed_area(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def resample_closed(p, n):
    """n points at equal arc length around the closed polyline p.

    Equal spacing is what makes the representation pure: every edge then has the
    same length, so the shape is carried entirely by the turning angles and
    nothing leaks into an edge-length channel.
    """
    q = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    if L <= 0 or not np.isfinite(L):
        return None
    t = np.linspace(0.0, L, n, endpoint=False)
    return np.stack([np.interp(t, s, q[:, 0]), np.interp(t, s, q[:, 1])], 1), L


def lowpass_closed(p, keep, work=512):
    """Band-limit a closed curve to `keep` harmonics before sampling it at n
    points. This is anti-aliasing, and leaving it out is a bug, not a choice.

    A coastline is rough at every scale. Dropping n equally spaced points onto
    one without a prefilter aliases all of that roughness into the turning
    angles: measured on Natural Earth at n=48, |d| had mean 0.44 rad against a
    smooth-curve expectation of 2*pi/48 = 0.13, and a first-order Markov table
    beat the unigram by 0.15 nats. That is a sampled fractal, and a benchmark on
    it would have measured almost nothing, for every model equally.

    So the curve is resampled finely, low-passed as a complex signal
    z = x + iy (the standard elliptic-Fourier-descriptor treatment of a closed
    curve), and only then sampled at n points. `keep` is set to the Nyquist
    limit n//2, so the data retains every mode the n-point grid can represent.
    This is a property of the SAMPLING, not of any model: the same tokens are
    handed to the transformers.
    """
    rs = resample_closed(p, work)
    if rs is None:
        return None
    q, _ = rs
    z = q[:, 0] + 1j * q[:, 1]
    F = np.fft.fft(z)
    f = np.fft.fftfreq(work, d=1.0 / work)
    F[np.abs(f) > keep] = 0.0
    w = np.fft.ifft(F)
    return np.stack([w.real, w.imag], 1)


def turning_angles(p):
    """d_i = wrap(Phi_i - Phi_{i-1}), where Phi_i is the direction of edge i.

    For a simple closed curve traversed counter-clockwise, sum(d) == 2*pi
    exactly (Hopf). That identity is the ground truth the model's holonomy is
    supposed to be tracking, so it is checked rather than assumed.
    """
    e = np.roll(p, -1, axis=0) - p
    phi = np.arctan2(e[:, 1], e[:, 0])
    d = phi - np.roll(phi, 1)
    return (d + np.pi) % (2 * np.pi) - np.pi


def phases_from(d, phi0=0.0):
    """Phi from turning angles: Phi_i = phi0 + cumsum(d)_i - d_0."""
    return phi0 + np.cumsum(d, axis=-1) - d[..., :1]


def closure_error(d):
    """||sum_i exp(i Phi_i)|| / n -- 0 iff the curve closes. Scale-free."""
    phi = phases_from(d)
    return np.abs(np.exp(1j * phi).sum(-1)) / d.shape[-1]


# ----------------------------------------------------------------------------
# build

RASTER = {"mnist": _mnist_rings, "fashion": _fashion_rings}


def raw_turnings(source, n, split="train", max_items=None):
    rings = (RASTER[source](split) if source in RASTER
             else _geojson_rings(source))
    out = []
    for r in rings:
        if len(r) < max(12, n // 4):
            continue
        if signed_area(r) < 0:          # force counter-clockwise: turning +2pi
            r = r[::-1]
        r = lowpass_closed(r, keep=n // 2)   # anti-alias; see lowpass_closed
        if r is None:
            continue
        rs = resample_closed(r, n)
        if rs is None:
            continue
        p, L = rs
        if L <= 0:
            continue
        d = turning_angles(p)
        # A ring that fails Hopf after resampling is self-intersecting or
        # degenerate at this resolution. Drop it rather than model it: the
        # closure identity is the thing under test and it must be exact in the
        # data before it can be asked of a model.
        if abs(d.sum() - 2 * np.pi) > 1e-6:
            continue
        out.append(d)
        if max_items and len(out) >= max_items:
            break
    return np.asarray(out)


def quantile_bins(d, vocab):
    """Equal-frequency bins over the training turning angles.

    Turning angles pile up near 2*pi/n and have long tails at corners, so
    uniform bins would spend most of the vocabulary on angles that never occur.
    Equal-frequency bins make the marginal token distribution uniform, which
    fixes the unigram reference at exactly log(vocab) and makes "beats the
    unigram" mean the same thing for every dataset.
    """
    qs = np.quantile(d.ravel(), np.linspace(0, 1, vocab + 1)[1:-1])
    edges = np.unique(qs)
    centres = np.zeros(len(edges) + 1)
    idx = np.digitize(d.ravel(), edges)
    for b in range(len(centres)):
        m = idx == b
        centres[b] = np.median(d.ravel()[m]) if m.any() else 0.0
    return edges, centres


def tokenize(d, edges):
    return np.digitize(d, edges).astype(np.int32)


def build(source, n=48, vocab=32, seed=0):
    """Tokenised train/test splits plus the bin table needed to draw shapes."""
    if source in RASTER:
        tr = raw_turnings(source, n, "train")
        te = raw_turnings(source, n, "test")
    else:
        allr = raw_turnings(source, n)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(allr))
        cut = int(0.85 * len(allr))
        tr, te = allr[perm[:cut]], allr[perm[cut:]]
    edges, centres = quantile_bins(tr, vocab)
    return dict(train=tokenize(tr, edges), test=tokenize(te, edges),
                train_raw=tr, test_raw=te, edges=edges, centres=centres)


def cached(source, n=48, vocab=32):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{source}_n{n}_v{vocab}.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    d = build(source, n, vocab)
    np.savez_compressed(path, **d)
    return d


# ----------------------------------------------------------------------------

def _report(source, n=48, vocab=32):
    d = cached(source, n, vocab)
    tr, te = d["train"], d["test"]
    raw = d["train_raw"]
    cent = d["centres"]
    rec = cent[tr]                       # what the tokeniser can express
    p = np.bincount(tr.ravel(), minlength=vocab).astype(float)
    p /= p.sum()
    H = -(p * np.log(p + 1e-12)).sum()
    print(f"{source:10s} n={n} vocab={vocab} | train {len(tr):6d} test {len(te):5d} "
          f"| unigram NLL {H:.3f} (log V = {np.log(vocab):.3f})")
    print(f"           turning: |d| mean {np.abs(raw).mean():.3f} rad, "
          f"p95 {np.quantile(np.abs(raw), .95):.3f} | sum(d)/2pi = "
          f"{raw.sum(1).mean() / (2 * np.pi):.6f}")
    print(f"           closure err: exact {closure_error(raw).mean():.4f} | "
          f"after tokenising {closure_error(rec).mean():.4f}")
    # How much structure is there to model at all? A first-order Markov table on
    # the token sequence is the cheap lower bound; if this does not beat the
    # unigram the dataset is noise at this resolution and the benchmark on it
    # would measure nothing.
    c = np.ones((vocab, vocab))
    np.add.at(c, (tr[:, :-1].ravel(), tr[:, 1:].ravel()), 1.0)
    lp = np.log(c / c.sum(1, keepdims=True))
    print(f"           neighbour-Markov NLL {-lp[te[:, :-1], te[:, 1:]].mean():.3f} "
          f"(vs unigram {H:.3f})")


if __name__ == "__main__":
    import sys
    v = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    for s in ("mnist", "fashion", "ne_lakes", "ne_admin1"):
        _report(s, 48, v)
