"""task_cont.py -- continuous edge-vector head: does a compliance channel rescue
the test-time constraint?

Part II reaches the same recommendation from three directions -- the closure
metric is 94% quantisation noise, the constraint moves a categorical's mean but
never its mode, and Result 8 shows the constraint's only lever IS the content
prediction. All three say: stop tokenising.

The representation here is the edge VECTOR rather than a binned turning angle.
Sampling at uniform parameter instead of equal arc length, each vertex carries

    (d_i, log L_i)      turning angle, and log edge length

so a closed curve is n complex numbers `L_i exp(i Phi_i)` and closure is

    sum_i L_i exp(i Phi_i) = 0

Two things change, and they are different changes:

  * TYPE. The angle stops being one of 32 bins and becomes a real number with a
    predicted spread. That removes a quantisation error of 0.030 rad, 70.6% of
    which sits in the sharpest curvature quintile -- exactly where the layer
    trails the transformer worst.

  * CHANNEL. Length comes back. Equal-arc-length resampling deleted it by
    construction ("so nothing leaks into an edge-length channel"), and it is the
    literal analogue of stroke weight varying around a logogram. Measured on
    this data the channel is thin -- coefficient of variation 0.057 -- so it is
    not expected to buy accuracy.

The reason to want it anyway is Result 8. The constraint failed there because
the only quantity it could move, the expected turning angle, is also the answer:
no direction changes the geometry without changing the prediction. With
`(L, Phi)` per vertex, closure is still two real constraints but there is now a
COMPLIANCE channel -- adjust lengths to make the curve meet while leaving the
angles alone. That is a genuine null direction, and its absence is the standing
explanation for why every constraint weight made things worse.

Prediction, recorded before running: this helps the constraint mechanism much
more than it helps NLL.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import contours as C
import loop_layer as L
import semagram as S
import task_shape as T

TWO_PI = 2.0 * np.pi
LOG2PI = float(np.log(2 * np.pi))


# ----------------------------------------------------------------------------
# data: uniform-PARAMETER sampling, so edge length varies and carries the ink

def build(source="mnist", n=48, split="train", cache=True):
    path = os.path.join(C.DATA_DIR, f"{source}_cont_n{n}_{split}.npz")
    if cache and os.path.exists(path):
        z = np.load(path)
        return z["d"], z["logl"]
    rings = (C.RASTER[source](split) if source in C.RASTER
             else C._geojson_rings(source))
    D, LG = [], []
    for r in rings:
        if len(r) < max(12, n // 4):
            continue
        if C.signed_area(r) < 0:
            r = r[::-1]
        r = C.lowpass_closed(r, keep=n // 2)
        if r is None:
            continue
        idx = np.linspace(0, len(r), n, endpoint=False).astype(int)
        p = r[idx]                                  # UNIFORM PARAMETER
        e = np.roll(p, -1, axis=0) - p
        Ln = np.linalg.norm(e, axis=1)
        if Ln.min() <= 1e-9:
            continue
        phi = np.arctan2(e[:, 1], e[:, 0])
        d = (phi - np.roll(phi, 1) + np.pi) % TWO_PI - np.pi
        if abs(d.sum() - TWO_PI) > 1e-6:
            continue
        D.append(d)
        LG.append(np.log(Ln / Ln.mean()))           # scale-free
    D, LG = np.asarray(D, np.float32), np.asarray(LG, np.float32)
    if cache:
        os.makedirs(C.DATA_DIR, exist_ok=True)
        np.savez_compressed(path, d=D, logl=LG)
    return D, LG


def bin_nll(mu, sd, tok, edges):
    """Score a continuous model in the OLD units.

    The Part II numbers are discrete NLL over 32 equal-frequency turning-angle
    bins; a Gaussian density in radians is not comparable to that. The honest
    conversion is to push the predicted density through the SAME quantiser --
    integrate it over each bin interval -- and score the resulting categorical
    against the same label. That is exact, and it is the only way these two
    families cross-compare.

    It is not a free win for the continuous model: it still has to place its
    mass in the right bin, and a Gaussian fits the heavy tails at corners badly.
    """
    from jax.scipy.stats import norm
    e = jnp.concatenate([jnp.array([-1e3]), jnp.asarray(edges),
                         jnp.array([1e3])])
    z = (e[None, None, :] - mu[..., None]) / sd[..., None]
    p = jnp.clip(jnp.diff(norm.cdf(z), axis=-1), 1e-12, 1.0)
    return -jnp.log(jnp.take_along_axis(p, tok[..., None], -1)[..., 0])


def feats(d, logl):
    """Inputs the model sees: (cos d, sin d, log L). Angle enters as a point on
    the circle so that wrap-around is not a discontinuity in the input."""
    return jnp.stack([jnp.cos(d), jnp.sin(d), logl], -1)


# ----------------------------------------------------------------------------
# head and loss

def init_cont(key, cfg, width=None):
    p = L.init(key, cfg)
    k = jax.random.split(key, 3)
    w = cfg.d
    p["inp"] = jax.random.normal(k[0], (3, w)) * (1.0 / np.sqrt(3))
    p["out"] = jax.random.normal(k[1], (w, 4)) * (1.0 / np.sqrt(w))
    p["bout"] = jnp.zeros((4,))
    return p


def head(p, x, cfg):
    """-> (mu_d, log sd_d, mu_logl, log sd_logl)."""
    o = S.rmsnorm(x) @ p["out"] + p["bout"]
    return o[..., 0], jnp.clip(o[..., 1], -4.0, 2.0), o[..., 2], \
        jnp.clip(o[..., 3], -4.0, 2.0)


def cont_nll(p, x, cfg, d, logl):
    """Gaussian NLL on both channels, in nats per vertex.

    A Gaussian on the angle rather than a von Mises: |d| has mean 0.36 and p95
    1.07, so wrap-around is rare and the approximation is cheap. It is an
    approximation and it is the reason these numbers are not comparable to the
    32-way categorical NLL elsewhere in Part II -- the geometric metrics are.
    """
    mu, ls, ml, lsl = head(p, x, cfg)
    e1 = 0.5 * (((d - mu) / jnp.exp(ls)) ** 2) + ls + 0.5 * LOG2PI
    e2 = 0.5 * (((logl - ml) / jnp.exp(lsl)) ** 2) + lsl + 0.5 * LOG2PI
    return e1 + e2


def solve_cont(p, d, logl, clamp, cfg, extra=None):
    c = feats(d, logl) @ p["inp"]
    return L.solve(p, c, clamp, cfg, extra=extra)


def loss_cont(p, d, logl, clamp, cfg):
    x = solve_cont(p, d, logl, clamp, cfg)
    free = 1.0 - clamp
    nll = jnp.sum(cont_nll(p, x, cfg, d, logl) * free) / (jnp.sum(free) + 1e-6)
    resid, _ = L.holonomy(p, x, cfg)
    return nll + cfg.w_holo * resid, (nll, resid)


def _step(p, st, d, logl, clamp, opt_update, cfg):
    (_, aux), g = jax.value_and_grad(loss_cont, has_aux=True)(
        p, d, logl, clamp, cfg)
    u, st = opt_update(g, st, p)
    return optax.apply_updates(p, u), st, aux


@functools.partial(jax.jit, static_argnames=("opt_update","cfg"))
def train_chunk(p, st, ds, lgs, cls, opt_update, cfg):
    def body(carry, xs):
        p, st = carry
        p, st, aux = _step(p, st, xs[0], xs[1], xs[2], opt_update, cfg)
        return (p, st), aux
    (p, st), aux = jax.lax.scan(body, (p, st), (ds, lgs, cls))
    return p, st, jax.tree.map(lambda a: a[-1], aux)


# ----------------------------------------------------------------------------
# geometry

def closure_cont(d, logl):
    """||sum_i L_i exp(i Phi_i)|| / sum_i L_i -- scale-free, 0 iff closed."""
    phi = jnp.cumsum(d, -1) - d[..., :1]
    Ln = jnp.exp(logl)
    cx = jnp.sum(Ln * jnp.cos(phi), -1)
    cy = jnp.sum(Ln * jnp.sin(phi), -1)
    return jnp.sqrt(cx ** 2 + cy ** 2) / jnp.sum(Ln, -1)


def decode(p, x, cfg, d, logl, clamp):
    mu, _, ml, _ = head(p, x, cfg)
    return (jnp.where(clamp > 0, d, mu), jnp.where(clamp > 0, logl, ml))


def closure_energy_cont(p, cfg, d, logl, clamp, w, lock_angle=False):
    """The same constraint as Result 4, now with a length channel to move.

    `lock_angle` freezes the predicted turning angles at their unconstrained
    values, so the ONLY way to close the curve is through the lengths. That
    isolates the compliance channel: if closure improves under a locked angle,
    the null direction is real and is what Result 8 was missing.
    """
    def energy(x):
        mu, _, ml, _ = head(p, x, cfg)
        dd = jnp.where(clamp > 0, d, mu)
        ll = jnp.where(clamp > 0, logl, ml)
        if lock_angle:
            dd = jax.lax.stop_gradient(dd)
        return w * jnp.sum(closure_cont(dd, ll) ** 2)
    return energy


# ----------------------------------------------------------------------------

def run(a):
    tokd = C.cached(a.dataset, a.n, 32)          # for scoring in the old units
    if a.repr == "arc":
        # Identical data to the tokenised benchmark -- the SAME equal-arc-length
        # turning angles, pre-quantisation -- with length held constant. Only
        # the head differs, so this isolates the TYPE change from the CHANNEL.
        dtr = np.asarray(tokd["train_raw"]); dte0 = np.asarray(tokd["test_raw"])
        ltr = np.zeros_like(dtr); lte0 = np.zeros_like(dte0)
        dte, lte = dte0, lte0
    else:
        dtr, ltr = build(a.dataset, a.n, "train")
        dte, lte = build(a.dataset, a.n, "test")
    te_tok = jnp.asarray(np.digitize(dte[:a.eval_n], tokd["edges"]))
    edges = tokd["edges"]
    dte, lte = jnp.asarray(dte[:a.eval_n]), jnp.asarray(lte[:a.eval_n])
    dtr, ltr = jnp.asarray(dtr), jnp.asarray(ltr)
    mask = T.eval_masks(dte.shape[0], a.n, a.gap)
    cfg = L.LoopCfg(n=a.n, d=a.d, heads=4, k_steps=a.k, modes=12, vocab=4,
                    phi_dev=0.5, w_stat=a.w_stat)
    print(f"continuous edge-vector task | train {dtr.shape[0]} test {dte.shape[0]}"
          f" | log-length CV {float(jnp.std(ltr)):.3f}")
    print(f"closure of the TRUE curves: {float(jnp.mean(closure_cont(dte, lte))):.5f}"
          f"   (tokenised baseline floor was 0.0714)")

    key = jax.random.PRNGKey(a.seed)
    p = init_cont(key, cfg)
    opt = S.make_opt(a.lr, a.steps)
    st = opt.init(p)
    t0 = time.time()
    ck = 25
    for it in range(0, a.steps, ck):
        key, k1 = jax.random.split(key)
        idx = jax.random.randint(k1, (ck, a.batch), 0, dtr.shape[0])
        ks = jax.random.split(jax.random.fold_in(k1, 1), ck)
        cls = jnp.stack([T.occlusion(kk, a.batch, a.n) for kk in ks])
        p, st, (nll, res) = train_chunk(p, st, dtr[idx], ltr[idx], cls,
                                        opt.update, cfg)
        if (it // ck) % 20 == 0 or it + ck >= a.steps:
            x = solve_cont(p, dte, lte, mask, cfg)
            dd, ll = decode(p, x, cfg, dte, lte, mask)
            print(f"  {it+ck:5d}/{a.steps} train {float(nll):7.3f} | "
                  f"closure {float(jnp.mean(closure_cont(dd, ll))):.4f} | "
                  f"{time.time()-t0:5.0f}s", flush=True)
    os.makedirs("ckpt", exist_ok=True)
    import pickle
    pickle.dump(jax.tree.map(np.asarray, p),
                open(f"ckpt/{a.dataset}_cont_s{a.seed}.pkl", "wb"))

    x = solve_cont(p, dte, lte, mask, cfg)
    dd, ll = decode(p, x, cfg, dte, lte, mask)
    base = float(jnp.mean(closure_cont(dd, ll)))
    free = 1.0 - mask
    nll0 = float(jnp.sum(cont_nll(p, x, cfg, dte, lte) * free) / jnp.sum(free))
    mu, ls, _, _ = head(p, x, cfg)
    bn = bin_nll(mu, jnp.exp(ls), te_tok, edges)
    bnll = float(jnp.sum(bn * free) / jnp.sum(free))
    print(f"\nunconstrained: cont-NLL {nll0:.4f}  closure {base:.4f}")
    print(f"SAME UNITS AS PART II (density integrated over the 32 quantile "
          f"bins): {bnll:.4f}")
    print(f"  references: unigram 3.466 | Markov two-sided infill 3.445 | "
          f"sema-so2 3.397 | tf-abs 3.167")
    print(f"\n{'w':>8s} {'cont-NLL':>10s} {'closure':>9s}   {'closure (angles locked)':>24s}")
    for w in a.con_w:
        out = []
        for lock in (False, True):
            xs = solve_cont(p, dte, lte, mask, cfg,
                            extra=closure_energy_cont(p, cfg, dte, lte, mask, w,
                                                      lock_angle=lock))
            d2, l2 = decode(p, xs, cfg, dte, lte, mask)
            out.append((float(jnp.sum(cont_nll(p, xs, cfg, dte, lte) * free)
                              / jnp.sum(free)),
                        float(jnp.mean(closure_cont(d2, l2)))))
        print(f"{w:8g} {out[0][0]:10.4f} {out[0][1]:9.4f}   {out[1][1]:24.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--w-stat", type=float, default=0.0)
    ap.add_argument("--repr", default="param", choices=["param", "arc"],
                    help="param: uniform-parameter, angle+length (adds the "
                         "stroke channel). arc: equal arc length, angle only "
                         "-- identical data to the tokenised benchmark, so it "
                         "isolates the continuous-head change.")
    ap.add_argument("--con-w", type=float, nargs="*",
                    default=[0.3, 1, 3, 10, 30, 100])
    run(ap.parse_args())
