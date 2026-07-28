"""commute_dose.py -- intervene on the symmetry, at fixed capacity.

WHY THIS FILE EXISTS. `tf_capacity.py` reported
`corr(far-commutation ratio, depth extrapolation) = -0.812` over nine
transformers and the README read it as "the models that accidentally learn more
of the symmetry extrapolate better". That reading does not survive a control.
The sweep varied CAPACITY and observed the commute ratio as a downstream
consequence, and capacity predicts both:

    commute vs extrapolation          r = -0.812   p = 0.008
    log(params) vs extrapolation      r = +0.799   p = 0.010
    log(params) vs commute            r = -0.852   p = 0.004
    PARTIAL commute vs extrapolation
                  given log(params)   r = -0.418   p = 0.263

Control for capacity and half the relationship disappears and significance goes
with it. Bigger models are both more invariant and better at extrapolating, and
an observational sweep cannot say which causes which.

The knot result did not have this problem because it was an INTERVENTION: vary
`w_ybe` with capacity held fixed, and the symmetry is the only thing moving. So
do the same thing here. The transformer is held at one architecture and one
parameter count, and the only thing that changes is a penalty pushing it toward
far commutation:

    L = mse + w * mean || f(A) - f(B) ||^2

over pairs `(A, B)` differing by ONE DISJOINT gate swap, which cannot change
the true answer. The pairs are synthetic, generated on the fly, and carry no
label -- this adds no information about the target, only the symmetry.

WHAT EACH OUTCOME MEANS. If raising `w` lowers the commute ratio AND raises
depth extrapolation, the causal claim is earned on this task the way it was
earned on knots. If invariance improves and extrapolation does not, then far
commutation is not what the braid layer's advantage is made of, and the
capacity correlation was capacity all along -- which would leave the honest
summary as the one the review proposed: the transformer never reaches the
invariance the architecture has for free, full stop, no mechanism story.

The braid layer is not in this sweep. It cannot be: its commute ratio is 0.000
by construction and no penalty can move it.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import circuits as C
import commute_probe as P
import task_circuit as T
import task_knot as K


def swap_pairs(rng, n_qubits, depth_range, n_pairs, n_max, l_max):
    """Circuit pairs differing by one DISJOINT gate swap. No labels needed."""
    A, B = [], []
    while len(A) < n_pairs:
        d = int(rng.integers(depth_range[0], depth_range[1] + 1))
        w = C.random_circuit(rng, n_qubits, d)
        t = int(rng.integers(0, max(d - 1, 1)))
        i = int(rng.integers(0, n_qubits - 1))
        cand = [j for j in range(n_qubits - 1) if abs(j - i) >= 2]
        if not cand or t + 2 > d:
            continue
        j = int(rng.choice(cand))
        gi, gj = (int(rng.integers(0, len(C.GATES))) for _ in range(2))
        A.append(w[:t] + [(i, gi), (j, gj)] + w[t + 2:])
        B.append(w[:t] + [(j, gj), (i, gi)] + w[t + 2:])
    Xa, Ma = C.encode(A, n_max, l_max)
    Xb, Mb = C.encode(B, n_max, l_max)
    return tuple(jnp.asarray(v) for v in (Xa, Ma, Xb, Mb))


def run(a):
    n_max, l_max = 7, a.dmax + 8
    vocab = C.VOCAB(n_max)
    tr = T.dataset(a.train_n, a.train_q, (4, a.dmax), 1, n_max, l_max)
    packs = {"depth 12-18": T.dataset(a.test_n, a.train_q,
                                      (a.dmax + 2, a.dmax + 8), 2, n_max, l_max),
             "in dist": T.dataset(a.test_n, a.train_q, (4, a.dmax), 3,
                                  n_max, l_max)}
    rng = np.random.default_rng(7)
    pairs = swap_pairs(rng, a.train_q, (4, a.dmax), a.pair_n, n_max, l_max)
    print(f"{a.pair_n} disjoint-swap pairs, unlabelled: they carry the "
          f"symmetry and no information about <Z_i>\n", flush=True)

    ref = T.init_layer(jax.random.PRNGKey(0), a.d, len(C.GATES))
    target = sum(v.size for v in jax.tree.leaves(ref))
    def _size(w):
        sh = jax.eval_shape(lambda: K.init_tf(jax.random.PRNGKey(0), vocab,
                                              l_max, a.tf_layers, w, a.heads))
        return sum(v.size for v in jax.tree.leaves(sh))
    width = min(range(8, 520, 8), key=lambda w: abs(_size(w) - target * a.mult))

    rows = []
    for w_com in a.weights:
        for seed in a.seeds:
            key = jax.random.PRNGKey(seed)
            p = K.init_tf(key, vocab, l_max, a.tf_layers, width, a.heads)
            p["out"] = jax.random.normal(key, (width, n_max)) / np.sqrt(width)
            p["bo"] = jnp.zeros((n_max,))
            fwd = lambda p, x, m: T.tf_pool_forward(p, x, m, a.heads,
                                                    a.rope_base)
            npar = sum(v.size for v in jax.tree.leaves(p))

            def loss(p, x, m, y, qm, k):
                l = jnp.sum(((fwd(p, x, m) - y) ** 2) * qm) / jnp.sum(qm)
                if w_com > 0:
                    i = jax.random.randint(k, (a.pair_batch,), 0,
                                           pairs[0].shape[0])
                    d = fwd(p, pairs[0][i], pairs[1][i]) \
                        - fwd(p, pairs[2][i], pairs[3][i])
                    l = l + w_com * jnp.mean(d ** 2)
                return l

            opt = optax.adamw(optax.warmup_cosine_decay_schedule(
                0.0, a.lr, a.steps // 20, a.steps, end_value=a.lr * 0.05),
                weight_decay=1e-4)
            st = opt.init(p)

            @jax.jit
            def step(p, st, x, m, y, qm, k):
                l, g = jax.value_and_grad(loss)(p, x, m, y, qm, k)
                u, st = opt.update(g, st, p)
                return optax.apply_updates(p, u), st, l

            t0 = time.time()
            for it in range(1, a.steps + 1):
                key, k1, k2 = jax.random.split(key, 3)
                i = jax.random.randint(k1, (a.batch,), 0, tr[0].shape[0])
                p, st, l = step(p, st, tr[0][i], tr[1][i], tr[2][i], tr[3][i], k2)
            pr = P.probe(lambda X, M: fwd(p, X, M), a.train_q, n_max, l_max,
                         depth=a.dmax, n_pairs=a.probe_n, with_truth=False)
            r = {t: T.masked_r2(fwd(p, X, M), Y, Q)
                 for t, (X, M, Y, Q) in packs.items()}
            rows.append((w_com, seed, npar, pr["ratio"], r["depth 12-18"],
                         r["in dist"]))
            print(f"  w={w_com:<6g} seed {seed}: commute {pr['ratio']:.3f} "
                  f"depth {r['depth 12-18']:+.3f} in-dist {r['in dist']:.3f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    print("\n" + "=" * 78)
    print(f"Far-commutation DOSE-RESPONSE, capacity fixed at {rows[0][2]} params")
    print("=" * 78)
    print(f"{'w_commute':>10s} {'commute ratio':>16s} {'depth 12-18':>16s} "
          f"{'in dist':>14s}")
    agg = {}
    for w_com in a.weights:
        sel = [r for r in rows if r[0] == w_com]
        c = np.array([r[3] for r in sel]); e = np.array([r[4] for r in sel])
        d = np.array([r[5] for r in sel])
        agg[w_com] = (c.mean(), e.mean())
        f = lambda v: (f"{v.mean():.3f} +/- {v.std(ddof=1):.3f}"
                       if len(v) > 1 else f"{v.mean():.3f}")
        print(f"{w_com:10g} {f(c):>16s} {f(e):>16s} {f(d):>14s}")
    ws = list(agg)
    if len(ws) > 2:
        cc = np.array([agg[w][0] for w in ws]); ee = np.array([agg[w][1] for w in ws])
        print(f"\ncorr(commute ratio, depth extrapolation) across the dose "
              f"= {np.corrcoef(cc, ee)[0, 1]:+.3f}")
        print("Capacity is IDENTICAL across these rows, so this correlation "
              "cannot be\nconfounded by it the way the capacity sweep's was.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=float, nargs="*",
                    default=[0.0, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--seeds", type=int, nargs="*", default=[1, 2])
    ap.add_argument("--tf-layers", type=int, default=4)
    ap.add_argument("--mult", type=float, default=4.0)
    ap.add_argument("--train-q", type=int, default=4)
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--dmax", type=int, default=10)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--pair-n", type=int, default=8000)
    ap.add_argument("--pair-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rope-base", type=float, default=8.0)
    ap.add_argument("--probe-n", type=int, default=256)
    run(ap.parse_args())
