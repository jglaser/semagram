"""rIII_probe.py -- is the penalty enforcing Reidemeister III, or just shrinking?

The review's sharpest point. `ybe_residual` lifts ONE matrix to two adjacent
strand pairs, but the untied model applies a DIFFERENT matrix at each generator
index, so the penalty enforces "each map is independently a constant-R
Yang-Baxter solution" rather than the braid relation between generators.
Measured directly: two matrices that each satisfy YBE to 1.3e-15 have a cross
braid-relation residual of 11.15.

If that reading is right, the penalty is a strong structural prior wearing a
topology costume, and its benefit has nothing to do with invariance.

This settles it behaviourally, with no training. Generate word pairs that differ
by an actual Reidemeister III move,

    ... sigma_i sigma_{i+1} sigma_i ...   <->   ... sigma_{i+1} sigma_i sigma_{i+1} ...

which present the SAME braid, and measure how far apart the model's outputs are.
An R-III-invariant model must give identical predictions. The prediction is
sharp and it discriminates:

  * TIED, where the penalty really is the braid relation: the distance should
    collapse as w_ybe rises.
  * UNTIED, where it is not: the distance should be roughly flat, because
    nothing ever tied R[sigma_i] to R[sigma_{i+1}].

A control pair -- same length, same multiset of letters, but NOT related by
R-III, so a genuinely different braid -- keeps the measurement honest. A model
that has simply collapsed toward a constant would score zero on both, and the
ratio is what matters.

    python rIII_probe.py            # trains small models across w_ybe
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np
import optax

import braids as B
import task_knot as K


def rIII_pairs(n_pairs, s, length, rng):
    """Word pairs differing by one Reidemeister III move (same braid)."""
    A, C = [], []
    while len(A) < n_pairs:
        i = int(rng.integers(0, s - 2))          # need i and i+1 both valid
        sg = int(rng.choice([-1, 1]))
        pre = [(int(rng.integers(0, s - 1)), int(rng.choice([-1, 1])))
               for _ in range(length)]
        post = [(int(rng.integers(0, s - 1)), int(rng.choice([-1, 1])))
                for _ in range(length)]
        left = pre + [(i, sg), (i + 1, sg), (i, sg)] + post
        right = pre + [(i + 1, sg), (i, sg), (i + 1, sg)] + post
        A.append(left)
        C.append(right)
    return A, C


def control_pairs(n_pairs, s, length, rng):
    """Same shape, same letter multiset, but NOT an R-III move -- a different
    braid. Without this a model that collapsed to a constant would look
    perfectly invariant."""
    A, C = [], []
    while len(A) < n_pairs:
        i = int(rng.integers(0, s - 2))
        sg = int(rng.choice([-1, 1]))
        pre = [(int(rng.integers(0, s - 1)), int(rng.choice([-1, 1])))
               for _ in range(length)]
        post = [(int(rng.integers(0, s - 1)), int(rng.choice([-1, 1])))
                for _ in range(length)]
        left = pre + [(i, sg), (i + 1, sg), (i, sg)] + post
        # same three letters, order scrambled so it is NOT the braid relation
        right = pre + [(i, sg), (i, sg), (i + 1, sg)] + post
        A.append(left)
        C.append(right)
    return A, C


def train(key, tie_r, w_ybe, a, vocab, s):
    Wtr, Ytr, _, _ = B.build(a.train_n, len_range=(4, 10), seed=1)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-8
    X, M = B.encode(Wtr, s, a.lmax)
    X, M = jnp.asarray(X), jnp.asarray(M)
    Y = jnp.asarray((Ytr - mu) / sd)
    p = K.init_braid(key, s, a.d, vocab, tie_r=tie_r)
    fwd = lambda p, x, m: K.braid_forward(p, x, m, s, a.d)

    def loss(p, x, m, y):
        l = jnp.mean((fwd(p, x, m) - y) ** 2)
        return l + w_ybe * K.ybe_residual(p, a.d) if w_ybe > 0 else l

    opt = optax.adamw(optax.warmup_cosine_decay_schedule(
        0.0, 2e-3, a.steps // 20, a.steps, end_value=1e-4), weight_decay=1e-4)
    st = opt.init(p)

    @jax.jit
    def step(p, st, x, m, y):
        l, g = jax.value_and_grad(loss)(p, x, m, y)
        u, st = opt.update(g, st, p)
        return optax.apply_updates(p, u), st, l

    for it in range(a.steps):
        key, k1 = jax.random.split(key)
        idx = jax.random.randint(k1, (128,), 0, X.shape[0])
        p, st, _ = step(p, st, X[idx], M[idx], Y[idx])
    return p, fwd


def main(a):
    s = 4
    vocab = B.VOCAB(s)
    rng = np.random.default_rng(0)
    PA, PB = rIII_pairs(a.n_pairs, s, 3, rng)
    CA, CB = control_pairs(a.n_pairs, s, 3, rng)
    enc = lambda W: tuple(jnp.asarray(v) for v in B.encode(W, s, a.lmax))
    pa, ma = enc(PA); pb, mb = enc(PB)
    ca, mca = enc(CA); cb, mcb = enc(CB)

    print("Reidemeister III invariance probe")
    print("  R-III pairs differ by sigma_i sigma_{i+1} sigma_i <-> "
          "sigma_{i+1} sigma_i sigma_{i+1}  (SAME braid)")
    print("  control pairs use the same letters in a non-braid order "
          "(DIFFERENT braid)\n")
    print(f"{'model':8s} {'w_ybe':>7s} {'YBE resid':>11s} "
          f"{'d(R-III pair)':>14s} {'d(control)':>11s} {'ratio':>8s}")
    for tie in (False, True):
        for w in a.weights:
            p, fwd = train(jax.random.PRNGKey(0), tie, w, a, vocab, s)
            d_r3 = float(jnp.mean(jnp.linalg.norm(
                fwd(p, pa, ma) - fwd(p, pb, mb), axis=-1)))
            d_ct = float(jnp.mean(jnp.linalg.norm(
                fwd(p, ca, mca) - fwd(p, cb, mcb), axis=-1)))
            print(f"{'tied' if tie else 'untied':8s} {w:7g} "
                  f"{float(K.ybe_residual(p, a.d)):11.2e} {d_r3:14.4f} "
                  f"{d_ct:11.4f} {d_r3/max(d_ct,1e-9):8.3f}", flush=True)
    print("\nA model that is R-III invariant has d(R-III pair) -> 0 while "
          "d(control) stays up,")
    print("so the RATIO is the measurement. Flat ratio = the penalty is not "
          "buying invariance.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=float, nargs="*", default=[0, 1, 10, 100])
    ap.add_argument("--train-n", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--lmax", type=int, default=16)
    ap.add_argument("--n-pairs", type=int, default=512)
    main(ap.parse_args())
