"""task_perm.py -- does the braided layer's advantage survive without topology?

The knot benchmark shows the braided layer extrapolating 3.4x better than a
parameter-matched transformer, with the gain tracking Yang-Baxter enforcement.
That establishes the advantage for one task whose target is a topological
invariant. The obvious worry is that this is a knot result rather than a
structural one.

This is the cheapest discriminating test. Same layer, same token format, same
train-short/test-long protocol -- and no topology anywhere. The word is a
sequence of ADJACENT TRANSPOSITIONS and the target is the permutation they
compose to.

Why it is the right control. The braid relation

    sigma_i sigma_{i+1} sigma_i = sigma_{i+1} sigma_i sigma_{i+1}

holds in the symmetric group too; it is the Coxeter relation, and S_n is the
quotient of B_n that forgets which strand went over. So the Yang-Baxter penalty
is enforcing a relation that is EXACTLY true of this task, while the target --
where each element ends up -- has nothing topological about it. If the advantage
survives, the claim is "this works for sequences of local operations with known
commutation relations", which covers quantum circuits, trace monoids,
instruction scheduling and sorting networks. If it does not, the honest claim
shrinks to "this works when the target is a topological invariant".

Prediction, recorded before running: the advantage survives, because nothing in
the braid layer's inductive bias is about knots -- it is about composing local
operations on adjacent tracks. If that is wrong, the knot result was narrower
than it looked.
"""

from __future__ import annotations

import argparse
import math
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import braids as B
import task_knot as K


def build(n_items, s, len_range, seed):
    """Adjacent-transposition words and the permutation they compose to."""
    rng = np.random.default_rng(seed)
    W, Y = [], []
    seen = set()
    while len(W) < n_items:
        L = int(rng.integers(len_range[0], len_range[1] + 1))
        w = [(int(rng.integers(0, s - 1)), int(rng.choice([-1, 1])))
             for _ in range(L)]
        key = tuple(w)
        if key in seen:
            continue
        seen.add(key)
        perm = np.arange(s)
        for (i, _sgn) in w:                       # sign is irrelevant in S_n
            perm[i], perm[i + 1] = perm[i + 1], perm[i]
        W.append(w)
        Y.append(perm.copy())
    return W, np.asarray(Y)


def run(a):
    s, lmax = 4, a.lmax + 6
    vocab = B.VOCAB(s)
    Wtr, Ytr = build(a.train_n, s, (a.lmin, a.lmax), 1)
    Wte, Yte = build(a.test_n, s, (a.lmin, a.lmax), 2)
    Wex, Yex = build(a.test_n, s, (a.lmax + 2, a.lmax + 6), 3)
    enc = lambda W: B.encode(W, s, lmax)
    Xtr, Mtr = enc(Wtr); Xte, Mte = enc(Wte); Xex, Mex = enc(Wex)
    Xtr, Mtr, Ytr = jnp.asarray(Xtr), jnp.asarray(Mtr), jnp.asarray(Ytr)
    packs = {"test (same lengths)": (jnp.asarray(Xte), jnp.asarray(Mte),
                                     jnp.asarray(Yte)),
             f"EXTRAPOLATION (len {a.lmax+2}-{a.lmax+6})":
                 (jnp.asarray(Xex), jnp.asarray(Mex), jnp.asarray(Yex))}
    print(f"permutation composition | train {len(Wtr)} (len {a.lmin}-{a.lmax}) "
          f"| test {len(Wte)} | extrapolation {len(Wex)}")
    print(f"chance exact-match = 1/{math.factorial(s)} = "
          f"{1/math.factorial(s):.3f}\n", flush=True)

    results = {}
    for name in a.models:
        key = jax.random.PRNGKey(a.seed)
        if name.startswith("braid"):
            p = K.init_braid(key, s, a.d, vocab)
            # the head predicts s slots of s classes instead of 6 reals
            p["out2"] = jax.random.normal(key, (4 * a.d, s * s)) / np.sqrt(4 * a.d)
            p["bo2"] = jnp.zeros((s * s,))
            fwd = lambda p, x, m: K.braid_forward(p, x, m, s, a.d).reshape(-1, s, s)
            wy = a.w_ybe if name == "braid-ybe" else 0.0
            def loss(p, x, m, y, wy=wy):
                l = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                    fwd(p, x, m), y))
                return l + wy * K.ybe_residual(p, a.d) if wy > 0 else l
        else:
            target = 32 * a.d ** 2
            width = min(range(8, 200, 4), key=lambda w: abs(24 * w ** 2 - target))
            p = K.init_tf(key, vocab, lmax, 2, width, a.heads)
            p["out"] = jax.random.normal(key, (width, s * s)) / np.sqrt(width)
            p["bo"] = jnp.zeros((s * s,))
            fwd = lambda p, x, m: K._pool(p, x, m, a.heads).reshape(-1, s, s)
            def loss(p, x, m, y):
                return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(
                    fwd(p, x, m), y))
        npar = sum(v.size for v in jax.tree.leaves(p))
        opt = optax.adamw(optax.warmup_cosine_decay_schedule(
            0.0, a.lr, a.steps // 20, a.steps, end_value=a.lr * 0.05),
            weight_decay=1e-4)
        st = opt.init(p)

        @jax.jit
        def step(p, st, x, m, y):
            l, g = jax.value_and_grad(loss)(p, x, m, y)
            u, st = opt.update(g, st, p)
            return optax.apply_updates(p, u), st, l

        t0 = time.time()
        for it in range(1, a.steps + 1):
            key, k1 = jax.random.split(key)
            idx = jax.random.randint(k1, (a.batch,), 0, Xtr.shape[0])
            p, st, l = step(p, st, Xtr[idx], Mtr[idx], Ytr[idx])
            if it % max(a.steps // 4, 1) == 0:
                print(f"  {name:10s} {it:5d}/{a.steps} loss {float(l):.4f} "
                      f"| {time.time()-t0:4.0f}s", flush=True)
        row = {"params": npar,
               "ybe": float(K.ybe_residual(p, a.d)) if name.startswith("braid")
               else float("nan")}
        for tag, (x, m, y) in packs.items():
            pred = jnp.argmax(fwd(p, x, m), -1)
            row[tag] = float(jnp.mean(jnp.all(pred == y, -1)))     # exact match
        results[name] = row

    print("\n" + "=" * 78)
    print("Permutation composition from adjacent-transposition words "
          "(exact-match accuracy)")
    print("=" * 78)
    hdr = f"{'model':11s} {'params':>8s} {'YBE resid':>11s}"
    for tag in packs:
        hdr += f" | {tag[:26]:>26s}"
    print(hdr)
    for nm, r in results.items():
        line = f"{nm:11s} {r['params']:8d} {r['ybe']:11.2e}"
        for tag in packs:
            line += f" | {r[tag]:26.3f}"
        print(line)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["braid", "braid-ybe", "tf"])
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--lmin", type=int, default=4)
    ap.add_argument("--lmax", type=int, default=10)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-ybe", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())
