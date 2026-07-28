"""trace_layer.py -- a readout with a real cyclic property.

The conjugation penalty failed, monotonically: 0.267 -> 0.252 -> 0.210 as the
weight rose. The reason is structural, not a tuning failure. The readout
mean-pools the final strand state, mean pooling has no cyclic property, so
`f(a b a^-1) = f(b)` is a property that architecture CANNOT have. The only way
to reduce the penalty is to become insensitive to the added letters -- that is,
more constant. A penalty cannot install a property the architecture forbids; it
can only buy it with capacity.

So install it instead of charging for it.

THE CONSTRUCTION. Accumulate a matrix along the word rather than a vector,

    M <- G(i, sign) @ M ,        M_0 = I

and read out traces of powers, `tr(M), tr(M^2), ...`. Then

    tr(rho(a) rho(b) rho(a)^-1) = tr(rho(b))

holds EXACTLY, because a trace is cyclic. Conjugation invariance stops being
something to hope for and becomes something the readout cannot violate.

Two conditions make it exact rather than approximate, and both are things this
project was missing:

  * `G(i, -1)` must be the inverse of `G(i, +1)`. That is Reidemeister II,
    which had no representation at all until now. Here it is enforced by
    CONSTRUCTION -- the inverse block is computed, not learned and penalised --
    so `sigma sigma^-1 = 1` holds to machine precision.

  * `G` must act at the position the letter names while being the same map
    everywhere, which is what tying R already achieved. Each `G(i, .)` is the
    identity except for a learned `2k x 2k` block at strands `(i, i+1)`, so the
    layer stays strand-count agnostic and one learned block serves every
    position.

This is the shape of the reduced Burau representation, with a learned block in
place of the standard one. Burau is a genuine braid-group representation and its
traces are genuine conjugation invariants; nothing here is analogy.

WHAT IT COSTS. Traces of a product of near-orthogonal matrices are numerically
delicate over long words, and a trace discards everything except the conjugacy
class -- which is the point, but it is also a hard information bottleneck. The
prediction is therefore NOT that this beats the tanh model outright; it is that
it is conjugation-invariant to machine precision where the tanh model is not,
and that the invariance is worth something at extrapolation. If it is worth
nothing, then conjugation is simply not a useful symmetry for this target, and
the failed penalty was measuring that rather than its own impossibility.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import braids as B


def init(key, s_max, k, n_out=6):
    kk = jax.random.split(key, 4)
    return {
        # one learned 2k x 2k block, applied at whichever pair a letter names
        "blk": jnp.eye(2 * k) + jax.random.normal(kk[0], (2 * k, 2 * k)) * 0.05,
        "feat": jax.random.normal(kk[1], (4, 32)) * 0.5,
        "bf": jnp.zeros((32,)),
        "out": jax.random.normal(kk[2], (32, n_out)) * (1 / np.sqrt(32)),
        "bo": jnp.zeros((n_out,)),
    }


def _embed(blk, i, s, k):
    """Lift the 2k x 2k block to an sk x sk matrix acting at strands (i, i+1)."""
    n = s * k
    E = jnp.eye(n)
    rows = jnp.concatenate([jnp.arange(k) + i * k, jnp.arange(k) + (i + 1) * k])
    sub = E[rows][:, rows] * 0.0 + blk
    E = E.at[jnp.ix_(rows, rows)].set(sub)
    return E


def forward(p, toks, mask, s, k):
    """Accumulate the matrix product, read out traces of powers.

    `tr(M)`, `tr(M^2)` and `log|det M|` are class functions: invariant under
    M -> A M A^-1 for any invertible A. Conjugating the braid word conjugates
    the accumulated matrix, so these features cannot see the difference.
    """
    b, L = toks.shape
    n = s * k
    inv_blk = jnp.linalg.inv(p["blk"])          # Reidemeister II, by construction

    def step(M, t):
        tok = toks[:, t]
        live = mask[:, t][:, None, None]
        i = jnp.clip((tok - 1) // 2, 0, s - 2)
        sign = (tok - 1) % 2                     # 0 = sigma, 1 = sigma^-1
        blk = jnp.where(sign[:, None, None] == 0, p["blk"], inv_blk)
        G = jax.vmap(lambda ii, bb: _embed(bb, ii, s, k))(i, blk)
        return jnp.where(live > 0, G @ M, M), None

    M = jnp.broadcast_to(jnp.eye(n), (b, n, n))
    M, _ = jax.lax.scan(step, M, jnp.arange(L))
    t1 = jnp.trace(M, axis1=1, axis2=2) / n
    t2 = jnp.trace(M @ M, axis1=1, axis2=2) / n
    sgn, ld = jnp.linalg.slogdet(M)
    f = jnp.stack([t1, t2, ld / n, sgn], -1)
    return jax.nn.gelu(f @ p["feat"] + p["bf"]) @ p["out"] + p["bo"]


def conj_error(p, s, k, key, n_pairs=256, s_gen=3):
    """How far the readout is from f(a b a^-1) = f(b). Should be ~machine eps."""
    rng = np.random.default_rng(0)
    base, conj = [], []
    for _ in range(n_pairs):
        w = [(int(rng.integers(0, s_gen - 1)), int(rng.choice([-1, 1])))
             for _ in range(int(rng.integers(4, 9)))]
        al = [(int(rng.integers(0, s_gen - 1)), int(rng.choice([-1, 1])))
              for _ in range(2)]
        base.append(w)
        conj.append(al + w + [(i, -g) for (i, g) in reversed(al)])
    Xb, Mb = B.encode(base, s, 20)
    Xc, Mc = B.encode(conj, s, 20)
    fb = forward(p, jnp.asarray(Xb), jnp.asarray(Mb), s, k)
    fc = forward(p, jnp.asarray(Xc), jnp.asarray(Mc), s, k)
    return float(jnp.mean(jnp.linalg.norm(fb - fc, axis=-1)))


def run(a):
    s, l_max = 4, a.lmax + 6
    Wtr, Ytr, _, _ = B.build(a.train_n, len_range=(4, a.lmax), seed=1)
    Wte, Yte, _, _ = B.build(a.test_n, len_range=(4, a.lmax), seed=2)
    Wex, Yex, _, _ = B.build(a.test_n, len_range=(a.lmax + 2, a.lmax + 6), seed=3)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-8
    enc = lambda W: tuple(jnp.asarray(v) for v in B.encode(W, s, l_max))
    Xtr, Mtr = enc(Wtr); Xte, Mte = enc(Wte); Xex, Mex = enc(Wex)
    Ytr_ = jnp.asarray((Ytr - mu) / sd)
    packs = {"test": (Xte, Mte, jnp.asarray((Yte - mu) / sd)),
             "EXTRAPOLATION": (Xex, Mex, jnp.asarray((Yex - mu) / sd))}

    key = jax.random.PRNGKey(a.seed)
    p = init(key, s, a.k)
    print(f"trace readout | params {sum(v.size for v in jax.tree.leaves(p))} "
          f"| matrix {s*a.k}x{s*a.k}")
    print(f"conjugation error BEFORE training: "
          f"{conj_error(p, s, a.k, key):.3e}", flush=True)

    def loss(p, x, m, y):
        return jnp.mean((forward(p, x, m, s, a.k) - y) ** 2)

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
        i = jax.random.randint(k1, (a.batch,), 0, Xtr.shape[0])
        p, st, l = step(p, st, Xtr[i], Mtr[i], Ytr_[i])
        if it % max(a.steps // 4, 1) == 0:
            print(f"  {it:5d}/{a.steps} loss {float(l):.4f} "
                  f"| {time.time()-t0:4.0f}s", flush=True)

    print("\n" + "=" * 60)
    for tag, (X, M, Y) in packs.items():
        r2 = float(1 - jnp.mean((forward(p, X, M, s, a.k) - Y) ** 2)
                   / jnp.mean(Y ** 2))
        print(f"{tag:16s} R2 {r2:7.3f}")
    print(f"\nconjugation error AFTER training: "
          f"{conj_error(p, s, a.k, key):.3e}")
    print("(the tanh model cannot make this small at all -- mean pooling has no "
          "cyclic property)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--lmax", type=int, default=10)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())
