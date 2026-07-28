"""task_knot.py -- braided attention: does building in the braid relation pay?

The hypothesis this file tests, stated so it can fail.

Part II's verdict was that Semagram's symmetries are exactly true and worth
nothing, because a transformer with absolute positions -- provably wrong about
the circle -- beat every model that was right about it. The proposed reason:
cyclic shift is too easy a symmetry to matter. A flexible model learns it more
cheaply than a rigid one imposes it.

Reidemeister equivalence is the opposite kind of symmetry. Deciding whether two
braid words close to the same knot has no cheap local statistic; the Jones
polynomial is #P-hard in general. If ANY symmetry is worth building in, it is
this one, and if this one does not pay either then "build the symmetry in" is
simply the wrong move for sequence models and Part II's result was not about
circles at all.

ARCHITECTURE. A braid acts on strands, so the state is one vector per strand,
`(batch, s, d)`, and a letter `sigma_i^{+-}` applies a learned map to the two
strands `i, i+1` and leaves the rest alone. Reading the word left to right IS
the braid representation, with learned matrices in place of the R-matrix. The
closure is a trace, so the readout pools over strands.

That gives the exact ablation the question needs:

  `braid`      the strand-structured map, learned freely
  `braid-ybe`  the same, plus a penalty forcing the learned map to satisfy the
               Yang-Baxter equation, which is precisely the statement that the
               layer is invariant under Reidemeister III
  `tf`         a parameter-matched transformer over the same token sequence,
               which must learn all of this from data

EXTRAPOLATION IS THE POINT. All three are trained on short braids and tested on
longer ones. An exactly-invariant model should degrade gracefully in crossing
number; a model that has fitted local statistics should not. If the built-in
symmetry buys anything anywhere, it buys it there.
"""

from __future__ import annotations

import argparse
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import braids as B
import loop_layer as L
import semagram as S


# ----------------------------------------------------------------------------
# the braided layer

def init_braid(key, s_max, d, vocab, tie_r=False):
    """`tie_r` uses ONE map per sign, applied at every generator position.

    This is what a braid representation actually is -- a single R lifted to each
    adjacent pair -- and it is also what makes `ybe_residual` mean what the
    README claimed. Untied, `R[sigma_1]` and `R[sigma_2]` are different matrices
    that are never applied at the same strand pair, so enforcing YBE on each
    INDEPENDENTLY does not enforce the braid relation BETWEEN them: two matrices
    that each satisfy YBE to 1.3e-15 have a cross braid-relation residual of
    11.15. Tying them makes the two lifts genuinely sigma_i sigma_{i+1} sigma_i
    against sigma_{i+1} sigma_i sigma_{i+1}, i.e. Reidemeister III.

    It also cuts parameters ~3x and makes the layer strand-count agnostic.
    """
    k = jax.random.split(key, 6)
    sc = 1.0 / np.sqrt(2 * d)
    nR = 2 if tie_r else vocab
    return {
        "R": jax.random.normal(k[0], (nR, 2 * d, 2 * d)) * sc,
        "bR": jnp.zeros((nR, 2 * d)),
        "x0": jax.random.normal(k[1], (s_max, d)) * 0.5,
        "out1": jax.random.normal(k[2], (d, 4 * d)) * (1 / np.sqrt(d)),
        "bo1": jnp.zeros((4 * d,)),
        "out2": jax.random.normal(k[3], (4 * d, 6)) * (1 / np.sqrt(4 * d)),
        "bo2": jnp.zeros((6,)),
    }


def braid_forward(p, toks, mask, s_max, d):
    """Apply the learned 2-strand maps along the word, in order.

    `toks[:, t]` names which generator acts at step t (0 = padding, no-op).
    Because a generator only touches strands (i, i+1), the update is a masked
    write into the strand state -- which is what makes this the braid
    representation rather than a generic sequence model.
    """
    b, Lw = toks.shape
    x = jnp.broadcast_to(p["x0"], (b, s_max, d))
    idx = jnp.arange(s_max)

    def step(x, t):
        tok = toks[:, t]                          # (b,)
        live = mask[:, t][:, None, None]
        i = jnp.maximum((tok - 1) // 2, 0)        # generator index
        pair = jnp.stack([jnp.take_along_axis(x, i[:, None, None]
                                              * jnp.ones((1, 1, d), jnp.int32), 1),
                          jnp.take_along_axis(x, (i + 1)[:, None, None]
                                              * jnp.ones((1, 1, d), jnp.int32), 1)],
                         axis=1)[:, :, 0, :]      # (b, 2, d)
        flat = pair.reshape(b, 2 * d)
        # tied: index by SIGN only, so the same map acts at every position
        ridx = jnp.where(p["R"].shape[0] == 2, (tok - 1) % 2, tok)
        ridx = jnp.clip(ridx, 0, p["R"].shape[0] - 1)
        R = p["R"][ridx]                          # (b, 2d, 2d)
        out = jnp.tanh(jnp.einsum("bij,bj->bi", R, flat) + p["bR"][ridx])
        out = out.reshape(b, 2, d)
        sel_i = (idx[None, :] == i[:, None])[..., None]
        sel_j = (idx[None, :] == (i + 1)[:, None])[..., None]
        upd = jnp.where(sel_i, out[:, 0:1, :], 0.0) + \
            jnp.where(sel_j, out[:, 1:2, :], 0.0)
        keep = 1.0 - (sel_i | sel_j).astype(x.dtype)
        xn = x * keep + upd
        return x * (1 - live) + xn * live, None

    x, _ = jax.lax.scan(step, x, jnp.arange(Lw))
    h = jnp.mean(x, axis=1)                       # closure = trace -> pool
    return jax.nn.gelu(h @ p["out1"] + p["bo1"]) @ p["out2"] + p["bo2"]


def ybe_residual(p, d):
    """How far the learned maps are from satisfying the braid relation.

    R_i R_{i+1} R_i = R_{i+1} R_i R_{i+1} on three strands. Enforcing it is
    exactly the statement that the layer cannot tell two diagrams apart when
    they differ by a Reidemeister III move, so this doubles as the training
    penalty and as the measurement of how invariant the model actually is.
    """
    v = p["R"].shape[0]
    I = jnp.eye(d)
    def lift(R, first):                      # 2-strand map -> 3-strand map
        Z = jnp.zeros((d, d))
        top = jnp.concatenate([R[:d, :d], R[:d, d:], Z], -1)
        mid = jnp.concatenate([R[d:, :d], R[d:, d:], Z], -1)
        bot = jnp.concatenate([Z, Z, I], -1)
        A = jnp.concatenate([top, mid, bot], 0)
        if first:
            return A
        P = jnp.concatenate([
            jnp.concatenate([I, Z, Z], -1),
            jnp.concatenate([Z, R[:d, :d], R[:d, d:]], -1),
            jnp.concatenate([Z, R[d:, :d], R[d:, d:]], -1)], 0)
        return P
    tot, cnt = 0.0, 0
    lo = 0 if v == 2 else 1
    for a in range(lo, v):
        R1, R2 = lift(p["R"][a], True), lift(p["R"][a], False)
        tot = tot + jnp.mean((R1 @ R2 @ R1 - R2 @ R1 @ R2) ** 2)
        cnt += 1
        if v == 2:                      # tied: also braid sigma against sigma^-1
            for b in range(lo, v):
                if b == a:
                    continue
                S1, S2 = lift(p["R"][a], True), lift(p["R"][b], False)
                tot = tot + jnp.mean((S1 @ S2 @ S1 - S2 @ S1 @ S2) ** 2)
                cnt += 1
    return tot / max(cnt, 1)


# ----------------------------------------------------------------------------
# baseline

def init_tf(key, vocab, n, layers, width, heads):
    p = L.tf_init(key, vocab, n, layers, width, heads, ring=False)
    k = jax.random.split(key, 2)
    p["out"] = jax.random.normal(k[0], (width, 6)) * (1 / np.sqrt(width))
    p["bo"] = jnp.zeros((6,))
    return p


def tf_forward(p, toks, mask, heads):
    clamp = jnp.ones_like(mask)
    h = L.tf_forward(p, toks, clamp, heads, ring=False)   # (b, n, vocab) logits
    return h[:, 0] * 0.0 + _pool(p, toks, mask, heads)


def _rope(x, n, base=10000.0):
    """Rotary positions: attention depends on i - j, so unseen absolute
    positions are not a problem. This is the FAIR baseline.

    NoPE was the first attempt and it is not fair in the other direction: a
    braid word is order-dependent, so removing position entirely drops the
    transformer to R2 0.417 in distribution against 0.643 with (broken)
    absolute positions. Dropping information is not the same as fixing a
    confound.
    """
    dh = x.shape[-1]
    half = dh // 2
    if half == 0:
        return x
    # `base` must suit the sequence length. The transformer default of 10000 is
    # tuned for contexts of thousands: over n = 16 it gives total rotations of
    # 15, 1.5, 0.15 and 0.015 radians, so three of four bands are nearly static
    # and the encoding runs on one usable frequency. Comparing a braid layer
    # against that is the same class of unfairness as the untrained
    # absolute-position rows it replaced.
    freq = 1.0 / (base ** (np.arange(half) / half))
    ang = np.arange(n)[:, None] * freq[None, :]
    c = jnp.asarray(np.cos(ang), x.dtype)[None, None]
    s = jnp.asarray(np.sin(ang), x.dtype)[None, None]
    x1, x2, rest = x[..., :half], x[..., half:2 * half], x[..., 2 * half:]
    return jnp.concatenate([x1 * c - x2 * s, x1 * s + x2 * c, rest], -1)


def _pool(p, toks, mask, heads, use_pos=True, rope=False, rope_base=10000.0):
    """`use_pos=False` is the NoPE baseline, and it is not cosmetic.

    Training words are length 4-10 and extrapolation words 12-16, so rows
    pos[10:16] are ALWAYS masked out of attention and out of the pool during
    training and receive exactly zero gradient -- measured. At extrapolation the
    model then reads 37.5% of its positional signal off random init. Any
    transformer-versus-braid number using learned absolute positions is
    confounded by that, and the braid layer is a scan, so length generalisation
    is nearly free for it.
    """
    b, n = toks.shape
    e = p["emb"][toks] + (p["pos"][:n] if use_pos else 0.0)
    h = e
    for blk in p["blocks"]:
        z = S.layernorm(h)
        q, k, v = jnp.split(z @ blk["qkv"], 3, -1)
        d = q.shape[-1]
        dh = d // heads
        rs = lambda t: t.reshape(b, n, heads, dh).transpose(0, 2, 1, 3)
        qh, kh = rs(q), rs(k)
        if rope:
            qh, kh = _rope(qh, n, rope_base), _rope(kh, n, rope_base)
        att = jnp.einsum("bhid,bhjd->bhij", qh, kh) / np.sqrt(dh)
        att = jnp.where(mask[:, None, None, :] > 0, att, -1e9)
        o = jnp.einsum("bhij,bhjd->bhid", jax.nn.softmax(att, -1), rs(v))
        h = h + o.transpose(0, 2, 1, 3).reshape(b, n, d) @ blk["proj"]
        z = S.layernorm(h)
        h = h + jax.nn.gelu(z @ blk["fc1"] + blk["b1"]) @ blk["fc2"] + blk["b2"]
    h = S.layernorm(h)
    pooled = jnp.sum(h * mask[..., None], 1) / (jnp.sum(mask, 1, keepdims=True) + 1e-6)
    return pooled @ p["out"] + p["bo"]


# ----------------------------------------------------------------------------

def mse(pred, y):
    return jnp.mean((pred - y) ** 2)


def run(a):
    print("building exact Jones targets (state sum, verified against the "
          "literature) ...", flush=True)
    Wtr, Ytr, Str, Ltr = B.build(a.train_n, len_range=(a.lmin, a.lmax), seed=1,
                                 pure=a.pure)
    Wte, Yte, Ste, Lte = B.build(a.test_n, len_range=(a.lmin, a.lmax), seed=2,
                                 pure=a.pure)
    Wex, Yex, Sex, Lex = B.build(a.test_n, len_range=(a.lmax + 2, a.lmax + 6),
                                 seed=3, pure=a.pure)
    if a.pure:
        print("PURE BRAIDS ONLY: every word has the identity permutation, so "
              "strand tracking carries no information", flush=True)
    smax, lmax = 4, a.lmax + 6
    vocab = B.VOCAB(smax)
    mu, sd = Ytr.mean(0), Ytr.std(0) + 1e-8
    nz = lambda Y: (Y - mu) / sd
    enc = lambda W: B.encode(W, smax, lmax)
    Xtr, Mtr = enc(Wtr); Xte, Mte = enc(Wte); Xex, Mex = enc(Wex)
    Xtr, Ytr_ = jnp.asarray(Xtr), jnp.asarray(nz(Ytr))
    Mtr = jnp.asarray(Mtr)
    packs = {"test (same lengths)": (jnp.asarray(Xte), jnp.asarray(Mte),
                                     jnp.asarray(nz(Yte))),
             f"EXTRAPOLATION (len {a.lmax+2}-{a.lmax+6})":
                 (jnp.asarray(Xex), jnp.asarray(Mex), jnp.asarray(nz(Yex)))}
    print(f"train {len(Wtr)} (len {a.lmin}-{a.lmax}) | test {len(Wte)} | "
          f"extrapolation {len(Wex)} (len {a.lmax+2}-{a.lmax+6})\n", flush=True)

    results = {}
    for name in a.models:
        key = jax.random.PRNGKey(a.seed)
        if name.startswith("braid"):
            p = init_braid(key, smax, a.d, vocab, tie_r=a.tie_r)
            fwd = lambda p, x, m: braid_forward(p, x, m, smax, a.d)
            wy = a.w_ybe if name == "braid-ybe" else 0.0
            def loss(p, x, m, y, wy=wy):
                l = mse(fwd(p, x, m), y)
                return l + wy * ybe_residual(p, a.d) if wy > 0 else l
        else:
            # parameter-match the transformer to the braided model rather than
            # picking a width. Part II's comparisons were all matched and this
            # one has to be too.
            target = 32 * a.d ** 2
            layers = 2
            # step 8 so that width/heads is even and rotary pairs line up
            width = min(range(8, 200, 8),
                        key=lambda w: abs(24 * w ** 2 - target))
            p = init_tf(key, vocab, lmax, layers, width, a.heads)
            use_pos = name not in ("tf-nope", "tf-rope")
            rope = (name == "tf-rope")
            fwd = (lambda p, x, m, u=use_pos, r=rope:
                   _pool(p, x, m, a.heads, u, r, a.rope_base))
            def loss(p, x, m, y):
                return mse(fwd(p, x, m), y)
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
            p, st, l = step(p, st, Xtr[idx], Mtr[idx], Ytr_[idx])
            if it % max(a.steps // 4, 1) == 0:
                print(f"  {name:10s} {it:5d}/{a.steps} loss {float(l):.4f} "
                      f"| {time.time()-t0:4.0f}s", flush=True)
        row = {"params": npar,
               "ybe": float(ybe_residual(p, a.d)) if name.startswith("braid")
               else float("nan")}
        for tag, (x, m, y) in packs.items():
            pr = fwd(p, x, m)
            row[tag] = float(mse(pr, y))
            row[tag + " R2"] = float(1 - mse(pr, y) / jnp.mean(y ** 2))
        results[name] = row

    print("\n" + "=" * 78)
    print("Jones-polynomial regression from braid words (MSE on standardised "
          "targets)")
    print("=" * 78)
    hdr = f"{'model':11s} {'params':>8s} {'YBE resid':>11s}"
    for tag in packs:
        hdr += f" | {tag[:26]:>26s}"
    print(hdr)
    for nm, r in results.items():
        line = f"{nm:11s} {r['params']:8d} {r['ybe']:11.2e}"
        for tag in packs:
            line += f" | {r[tag]:10.4f} (R2 {r[tag+' R2']:6.3f})"
        print(line)
    print("\nR2 <= 0 means the model does no better than predicting the mean.")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["braid", "braid-ybe", "tf"])
    ap.add_argument("--train-n", type=int, default=6000)
    ap.add_argument("--test-n", type=int, default=1000)
    ap.add_argument("--lmin", type=int, default=4)
    ap.add_argument("--lmax", type=int, default=10)
    ap.add_argument("--d", type=int, default=24)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--w-ybe", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rope-base", type=float, default=8.0,
                    help="rotary base; 10000 is tuned for long contexts and "
                         "wastes 3 of 4 bands at length 16")
    ap.add_argument("--tie-r", action="store_true",
                    help="one R per sign, applied at every position -- makes "
                         "ybe_residual genuinely Reidemeister III")
    ap.add_argument("--pure", action="store_true",
                    help="restrict to pure braids (identity permutation)")
    run(ap.parse_args())
