"""tf_capacity.py -- is the transformer losing on symmetry, or on depth?

THE ALTERNATIVE EXPLANATION, stated so it can win. `task_circuit.py` shows the
braid layer at 0.89 depth-extrapolation R2 against the transformer's 0.19, and
the natural reading is far commutation: the layer has it exactly and the
transformer does not. But there is a second explanation that has nothing to do
with symmetry.

Predicting <Z_i> means tracking a state through a sequence of gates. That is a
SEQUENTIAL COMPUTATION of depth equal to the circuit depth, and the braid layer
is a sequential state machine of exactly that depth -- it runs one step per
gate. A 2-layer transformer has two rounds of attention to simulate a depth-10
circuit, no matter how wide it is. If that is the binding constraint, then the
result is "the architecture matches the computation", which is a real finding
but a much weaker and more ordinary one than "the symmetry pays".

These are separable, because they predict different things about capacity.
If it is depth, adding transformer LAYERS should close the gap and adding
parameters at fixed depth should not. If it is symmetry, neither should,
because no amount of capacity makes a model exactly invariant.

So sweep both: layers in {2, 4, 8, 12} at matched parameters, and a
deliberately GENEROUS budget at 4x and 16x the braid layer's parameters. The
dataset is built once and shared, which is the whole reason this file exists
separately -- rebuilding 26,000 exact simulations per configuration made the
sweep cost more than the experiment.

Whatever this shows goes in the README. A transformer that closes the gap at
8 layers would make the symmetry claim unsupported, and that is a result.
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


def train(fwd, p, tr, steps, lr, batch, seed):
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(
        0.0, lr, steps // 20, steps, end_value=lr * 0.05), weight_decay=1e-4)
    st = opt.init(p)

    def loss(p, x, m, y, qm):
        return jnp.sum(((fwd(p, x, m) - y) ** 2) * qm) / jnp.sum(qm)

    @jax.jit
    def step(p, st, x, m, y, qm):
        l, g = jax.value_and_grad(loss)(p, x, m, y, qm)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    key = jax.random.PRNGKey(seed)
    for it in range(1, steps + 1):
        key, k1 = jax.random.split(key)
        i = jax.random.randint(k1, (batch,), 0, tr[0].shape[0])
        p, st, l = step(p, st, tr[0][i], tr[1][i], tr[2][i], tr[3][i])
    return p, float(l)


def run(a):
    n_max, l_max = 7, a.dmax + 8
    vocab = C.VOCAB(n_max)
    t0 = time.time()
    tr = T.dataset(a.train_n, a.train_q, (4, a.dmax), 1, n_max, l_max)
    packs = {"depth 12-18": T.dataset(a.test_n, a.train_q,
                                      (a.dmax + 2, a.dmax + 8), 2, n_max, l_max),
             "in dist": T.dataset(a.test_n, a.train_q, (4, a.dmax), 3,
                                  n_max, l_max)}
    print(f"dataset built once in {time.time()-t0:.0f}s "
          f"({a.train_n} train circuits, exact statevector)\n", flush=True)

    ref = T.init_layer(jax.random.PRNGKey(0), a.d, len(C.GATES))
    braid_par = sum(v.size for v in jax.tree.leaves(ref))
    rows = []

    # the braid layer itself, as the reference line
    fwd_b = lambda p, x, m: T.forward(p, x, m, n_max, a.d)
    if a.skip_braid:
        rows.append(("braid (see main run)", braid_par, 0.0,
                     {"depth 12-18": float("nan"), "in dist": float("nan")}))
    else:
        p, _ = train(fwd_b, ref, tr, a.steps, a.lr, a.batch, a.seed)
        pr = P.probe(lambda X, M: fwd_b(p, X, M), a.train_q, n_max, l_max,
                     depth=a.dmax, n_pairs=a.probe_n, with_truth=False)
        rows.append(("braid (1 step/gate)", braid_par, pr["ratio"],
                     {t: T.masked_r2(fwd_b(p, X, M), Y, Q)
                      for t, (X, M, Y, Q) in packs.items()}))
        print(f"  braid: {rows[-1][3]}", flush=True)

    for layers, mult in a.configs:
        nv = vocab
        def _size(w):
            # shapes without materialising the arrays -- see task_circuit._size
            sh = jax.eval_shape(lambda: K.init_tf(jax.random.PRNGKey(0), nv,
                                                  l_max, layers, w, a.heads))
            return sum(v.size for v in jax.tree.leaves(sh))
        tgt = braid_par * mult
        width = min(range(8, 520, 8), key=lambda w: abs(_size(w) - tgt))
        key = jax.random.PRNGKey(a.seed)
        p = K.init_tf(key, nv, l_max, layers, width, a.heads)
        p["out"] = jax.random.normal(key, (width, n_max)) / np.sqrt(width)
        p["bo"] = jnp.zeros((n_max,))
        fwd = lambda p, x, m: T.tf_pool_forward(p, x, m, a.heads, a.rope_base)
        npar = sum(v.size for v in jax.tree.leaves(p))
        t1 = time.time()
        p, _ = train(fwd, p, tr, a.steps, a.lr, a.batch, a.seed)
        pr = P.probe(lambda X, M: fwd(p, X, M), a.train_q, n_max, l_max,
                     depth=a.dmax, n_pairs=a.probe_n, with_truth=False)
        r = {t: T.masked_r2(fwd(p, X, M), Y, Q)
             for t, (X, M, Y, Q) in packs.items()}
        rows.append((f"tf {layers}L w{width} ({mult:g}x)", npar, pr["ratio"], r))
        print(f"  tf {layers}L width {width} ({npar} par, {mult:g}x): "
              f"{r}  [{time.time()-t1:.0f}s]", flush=True)

    print("\n" + "=" * 88)
    print("Does transformer CAPACITY close the gap? (depth extrapolation R2)")
    print("=" * 88)
    print(f"{'model':24s} {'params':>9s} {'commute':>8s} "
          f"{'depth 12-18':>12s} {'in dist':>9s}")
    for nm, npar, cr, r in rows:
        print(f"{nm:24s} {npar:9d} {cr:8.3f} "
              f"{r['depth 12-18']:12.3f} {r['in dist']:9.3f}")
    print("\nIf depth were the binding constraint, more LAYERS would close the")
    print("gap and more width would not. If far commutation is, neither will --")
    print("no amount of capacity makes a model exactly invariant.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-q", type=int, default=4)
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--dmax", type=int, default=10)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--rope-base", type=float, default=8.0)
    ap.add_argument("--probe-n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="layers:param_mult pairs, e.g. 4:4 8:16. Used to "
                         "re-run the big configurations for longer, since "
                         "'the large model was undertrained' is the obvious "
                         "objection to a capacity sweep that stalls")
    ap.add_argument("--skip-braid", action="store_true")
    a = ap.parse_args()
    if a.configs:
        a.configs = [(int(c.split(":")[0]), float(c.split(":")[1]))
                     for c in a.configs]
    else:
        a.configs = [(2, 1.0), (4, 1.0), (8, 1.0), (12, 1.0),
                     (4, 4.0), (8, 4.0), (8, 16.0)]
    run(a)
