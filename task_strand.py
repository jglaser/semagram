"""task_strand.py -- train on 3 strands, run on 5. Can the baseline even try?

Every result so far is a score on a fixed problem size. This one is a
capability: with `--tie-r` the layer holds ONE map per sign and lifts it to
whichever adjacent pair a letter names, so nothing in it is indexed by strand
count. Train it on 3-strand braids and it will run on 5-strand braids without a
new parameter.

A transformer cannot attempt this in any natural way. Its input is a token per
generator, `sigma_1 .. sigma_{s-1}`, so a 5-strand word contains symbols whose
embedding rows never received a gradient. That is the same failure that made the
absolute-position baseline unfair earlier in this project -- but here it is not
a fixable artefact of my setup, it is what "a symbol the model has never seen"
means. Reported as such, not as a beaten baseline.

TWO DESIGN POINTS, both learned the hard way from the position-embedding bug.

`x0` is a SINGLE shared vector broadcast to every strand, not a table indexed by
strand. A per-strand table would leave rows 3 and 4 untrained after training on
3 strands, and the measurement would then be about my initialisation rather than
about the architecture -- exactly the confound that invalidated the first
transformer comparison.

The readout pools over ACTIVE strands only. Padding a 3-strand example into a
5-wide state and averaging over all five would dilute the pooled vector by a
factor that changes with strand count, which is a scale artefact that would look
like generalisation failure.

Prediction, recorded before running: the tied layer degrades gracefully with
strand count and the transformer collapses to chance. If the tied layer ALSO
collapses, then nothing about it is really strand-agnostic and the tying result
is narrower than the extrapolation numbers suggest.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import braids as B
import task_knot as K


def init_shared(key, d, vocab, tie_r=True):
    """Like `K.init_braid` but with a strand-count-agnostic start state."""
    p = K.init_braid(key, 1, d, vocab, tie_r=tie_r)
    p["x0"] = jax.random.normal(jax.random.fold_in(key, 7), (d,)) * 0.5
    return p


def forward(p, toks, mask, s, d, strand_mask):
    """Apply the tied map along the word on `s` strands, pool over active ones."""
    b, Lw = toks.shape
    x = jnp.broadcast_to(p["x0"], (b, s, d))
    idx = jnp.arange(s)

    def step(x, t):
        tok = toks[:, t]
        live = mask[:, t][:, None, None]
        i = jnp.clip((tok - 1) // 2, 0, s - 2)
        gi = jnp.take_along_axis(
            x, i[:, None, None] * jnp.ones((1, 1, d), jnp.int32), 1)[:, 0]
        gj = jnp.take_along_axis(
            x, (i + 1)[:, None, None] * jnp.ones((1, 1, d), jnp.int32), 1)[:, 0]
        flat = jnp.concatenate([gi, gj], -1)
        # tied: index by sign, so an UNSEEN generator still hits a trained map.
        # untied: index by token, so sigma_3 and sigma_4 hit rows that never
        # received a gradient -- which is the whole point of the control.
        ridx = ((tok - 1) % 2 if p["R"].shape[0] == 2
                else jnp.clip(tok, 0, p["R"].shape[0] - 1))
        out = jnp.tanh(jnp.einsum("bij,bj->bi", p["R"][ridx], flat)
                       + p["bR"][ridx]).reshape(b, 2, d)
        si = (idx[None, :] == i[:, None])[..., None]
        sj = (idx[None, :] == (i + 1)[:, None])[..., None]
        xn = x * (1.0 - (si | sj).astype(x.dtype)) \
            + jnp.where(si, out[:, 0:1], 0.0) + jnp.where(sj, out[:, 1:2], 0.0)
        return x * (1 - live) + xn * live, None

    x, _ = jax.lax.scan(step, x, jnp.arange(Lw))
    sm = strand_mask[..., None]
    h = jnp.sum(x * sm, 1) / jnp.maximum(jnp.sum(sm, 1), 1.0)
    return jax.nn.gelu(h @ p["out1"] + p["bo1"]) @ p["out2"] + p["bo2"]


def dataset(n, s, len_range, seed, s_max, l_max):
    W, Y, _, _ = B.build(n, s_range=(s, s), len_range=len_range, seed=seed,
                         cache=False)
    X, M = B.encode(W, s_max, l_max)
    sm = np.zeros((len(W), s_max), np.float32)
    sm[:, :s] = 1.0
    return jnp.asarray(X), jnp.asarray(M), Y, jnp.asarray(sm)


def run(a):
    s_max, l_max = 5, a.lmax
    vocab = B.VOCAB(s_max)
    Xtr, Mtr, Ytr, Str = dataset(a.train_n, a.train_s, (4, a.lmax), 1,
                                 s_max, l_max)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-8
    Ytr_ = jnp.asarray((Ytr - mu) / sd)
    packs = {}
    for s in a.test_s:
        X, M, Y, S = dataset(a.test_n, s, (4, a.lmax), 2 + s, s_max, l_max)
        packs[s] = (X, M, jnp.asarray((Y - mu) / sd), S)
    print(f"train on {a.train_s} strands ({Xtr.shape[0]} words), "
          f"test on {a.test_s}\n", flush=True)

    results = {}
    for tie in ([True, False] if a.control else [True]):
      key = jax.random.PRNGKey(a.seed)
      p = init_shared(key, a.d, vocab, tie_r=tie)
      fwd = lambda p, x, m, sm: forward(p, x, m, s_max, a.d, sm)

      def loss(p, x, m, y, sm):
        return (jnp.mean((fwd(p, x, m, sm) - y) ** 2)
                + a.w_ybe * K.ybe_residual(p, a.d))

      opt = optax.adamw(optax.warmup_cosine_decay_schedule(
          0.0, a.lr, a.steps // 20, a.steps, end_value=a.lr * 0.05),
          weight_decay=1e-4)
      st = opt.init(p)

      @jax.jit
      def step(p, st, x, m, y, sm):
          l, g = jax.value_and_grad(loss)(p, x, m, y, sm)
          u, st = opt.update(g, st, p)
          return optax.apply_updates(p, u), st, l

      t0 = time.time()
      for it in range(1, a.steps + 1):
          key, k1 = jax.random.split(key)
          i = jax.random.randint(k1, (a.batch,), 0, Xtr.shape[0])
          p, st, l = step(p, st, Xtr[i], Mtr[i], Ytr_[i], Str[i])
          if it % max(a.steps // 4, 1) == 0:
              print(f"  {'tied  ' if tie else 'untied'} {it:5d}/{a.steps} "
                    f"loss {float(l):.4f} | {time.time()-t0:4.0f}s", flush=True)
      results[tie] = ({s: float(1 - jnp.mean((fwd(p, X, M, S) - Y) ** 2)
                                / jnp.mean(Y ** 2))
                       for s, (X, M, Y, S) in packs.items()},
                      float(K.ybe_residual(p, a.d)),
                      sum(v.size for v in jax.tree.leaves(p)))

    print("\n" + "=" * 72)
    print(f"Strand-count extrapolation  (trained on {a.train_s} strands only)")
    print("=" * 72)
    hdr = f"{'model':8s} {'params':>8s} {'YBE':>9s}"
    for s in packs:
        hdr += f" | {str(s)+' strands':>12s}"
    print(hdr)
    for tie, (r2s, yb, npar) in results.items():
        line = f"{'tied' if tie else 'untied':8s} {npar:8d} {yb:9.1e}"
        for s in packs:
            line += f" | {r2s[s]:12.3f}"
        print(line)
    print(f"\n{'':8s} {'':8s} {'':9s}" + "".join(
        f" | {'in dist' if s == a.train_s else 'NEVER SEEN':>12s}" for s in packs))
    print("A transformer cannot be run in this column at all: a 5-strand word "
          "contains\ngenerator tokens whose embedding rows never received a "
          "gradient.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-s", type=int, default=3)
    ap.add_argument("--test-s", type=int, nargs="*", default=[3, 4, 5])
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--lmax", type=int, default=10)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-ybe", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control", action="store_true", default=True,
                    help="also train an UNTIED model, whose R rows for unseen "
                         "generators never get a gradient")
    run(ap.parse_args())
