"""
semagram_d.py -- Semagram as masked discrete diffusion. Single file, JAX.

The variational principle moves from configurations to PATHS. Instead of solving
for a stationary point of a scalar action S[X] -- which forced tied Q=K, no V
matrix, and one weight-tied block, and capped the model at unigram entropy --
the reverse process is a Doob h-transform of an absorbing-state forward process.
That is still "the path is determined by its endpoints", but posed over path
measures (a Schrodinger bridge) rather than per-configuration energies. So the
denoiser is unconstrained: multi-head, independent Q/K/V/O, FFN, real depth.

Absorbing (masked) diffusion, not Gaussian diffusion on embeddings. Gaussian
diffusion decodes per-position marginals, which is the same mode-averaging that
produced walls of 'e' and spaces. Unmasking commits tokens progressively, so
later positions condition on actual samples -- effectively any-order
autoregression, which keeps mode commitment alongside bidirectional context.

Kept from the energy version, because these worked in every run:
  * circular topology: real, band-limited spectral multiplier applied by rfft.
    Real == even kernel == reflection-symmetric token mixing, O(n log n).
  * holonomy: learned SO(2) transports per 2-plane, cumulative phase rotating
    q and k (generalised RoPE), plus a loop-closure penalty and winding readout.
  * boundary-value decoding: to condition on an arc, never mask it. Free.

Honest caveat: attention with independent Q,K is NOT reflection-equivariant, so
exact reversal symmetry is now only a property of the spectral path, not of the
whole model. It is a soft bias here, not a guarantee.

    pip install jax optax && python semagram_d.py
"""
import functools
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import semagram as S  # data loading, gap_clamp, unigram_nll

N, D, LAYERS, HEADS, FF = 48, 64, 2, 4, 128
M = 12          # spectral kernel modes
T_MIN = 0.10    # min mask rate; the 1/t ELBO weight is high-variance below this
W_HOLO = 0.05
BATCH, LR, STEPS, EVAL = 32, 3e-3, 3000, 250
UNMASK_STEPS = 8
USE_CIRCMIX = True   # circular spectral token mixing
USE_GAUGE = True     # holonomy transports (generalised RoPE)
USE_POSEMB = False   # fall back to learned absolute positions


def init(key, vocab):
    k = jax.random.split(key, 12)
    s = 1.0 / np.sqrt(D)
    p = dict(
        emb=jax.random.normal(k[0], (vocab, D)) * 0.02,
        mask=jax.random.normal(k[1], (D,)) * 0.02,
        temb=jax.random.normal(k[2], (2, D)) * s,   # linear-in-t conditioning
        head=jax.random.normal(k[3], (D, vocab)) * s,
        wphi=jax.random.normal(k[4], (D, D // 2)) * s * 0.1,
        pos=jax.random.normal(k[11], (N, D)) * 0.02,
        layers=[],
    )
    for i in range(LAYERS):
        kk = jax.random.split(k[5 + i], 6)
        p["layers"].append(dict(
            wq=jax.random.normal(kk[0], (D, D)) * s,
            wk=jax.random.normal(kk[1], (D, D)) * s,
            wv=jax.random.normal(kk[2], (D, D)) * s,   # independent V: the thing
            wo=jax.random.normal(kk[3], (D, D)) * s,   # the energy form forbade
            w1=jax.random.normal(kk[4], (D, FF)) * s,
            w2=jax.random.normal(kk[5], (FF, D)) / np.sqrt(FF),
            ghat=jnp.zeros((M + 1, D)),                # real => even => symmetric
        ))
    return p


def ln(x):
    return (x - x.mean(-1, keepdims=True)) / jnp.sqrt(x.var(-1, keepdims=True) + 1e-5)


def circmix(x, ghat):
    """Symmetric circulant mixing on the loop. Real multiplier, band-limited."""
    g = jnp.pad(jnp.tanh(ghat), ((0, N // 2 + 1 - (M + 1)), (0, 0)))
    return jnp.fft.irfft(jnp.fft.rfft(x, axis=1) * g[None], n=N, axis=1)


def gauge(q, phi_cum):
    c, s = jnp.cos(phi_cum), jnp.sin(phi_cum)
    a, b = q[..., 0::2], q[..., 1::2]
    return jnp.stack([a * c - b * s, a * s + b * c], -1).reshape(q.shape)


def denoise(p, tokens, mask, t):
    """mask==1 means the position is [MASK] and must be predicted."""
    m = mask[..., None]
    x = (1 - m) * p["emb"][tokens] + m * p["mask"]
    x = x + (t[:, None, None] * p["temb"][0] + p["temb"][1])
    if USE_POSEMB:
        x = x + p["pos"]

    phi = (2 * np.pi / N) * jnp.tanh(ln(x) @ p["wphi"])
    phi_cum = jnp.cumsum(phi, axis=1)

    for L in p["layers"]:
        h = ln(x)
        gq = (lambda z: gauge(z, phi_cum)) if USE_GAUGE else (lambda z: z)
        q = gq(h @ L["wq"]).reshape(*h.shape[:2], HEADS, -1)
        k = gq(h @ L["wk"]).reshape(*h.shape[:2], HEADS, -1)
        v = (h @ L["wv"]).reshape(*h.shape[:2], HEADS, -1)
        a = jax.nn.softmax(jnp.einsum("bihd,bjhd->bhij", q, k)
                           / np.sqrt(D // HEADS), -1)
        o = jnp.einsum("bhij,bjhd->bihd", a, v).reshape(h.shape)
        x = x + o @ L["wo"]
        if USE_CIRCMIX:
            x = x + circmix(h, L["ghat"])
        h = ln(x)
        x = x + jax.nn.relu(h @ L["w1"]) @ L["w2"]
    return ln(x) @ p["head"], phi


def loss_fn(p, tokens, key):
    """MDLM objective: mask each token w.p. t, weight the CE by 1/t."""
    kt, km = jax.random.split(key)
    t = jax.random.uniform(kt, (tokens.shape[0],), minval=T_MIN, maxval=1.0)
    mask = (jax.random.uniform(km, tokens.shape) < t[:, None]).astype(jnp.float32)
    logits, phi = denoise(p, tokens, mask, t)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, tokens)
    nll = jnp.sum(ce * mask / t[:, None]) / (jnp.sum(mask) + 1e-6)
    total = jnp.sum(phi, axis=1)
    closure = jnp.mean(jnp.abs(jnp.arctan2(jnp.sin(total), jnp.cos(total))))
    return nll + W_HOLO * closure, (nll, closure)


@functools.partial(jax.jit, static_argnames=("upd",))
def train_step(p, st, tokens, key, upd):
    (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, tokens, key)
    u, st = upd(g, st, p)
    return optax.apply_updates(p, u), st, aux


@jax.jit
def one_shot(p, tokens, mask):
    """Predict every masked position independently, in a single pass."""
    t = jnp.full((tokens.shape[0],), 1.0)
    logits, _ = denoise(p, tokens, mask, t)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, tokens)
    acc = jnp.sum((jnp.argmax(logits, -1) == tokens) * mask) / jnp.sum(mask)
    return jnp.sum(ce * mask) / jnp.sum(mask), acc


@functools.partial(jax.jit, static_argnames=("steps",))
def iterative(p, tokens, mask, steps=UNMASK_STEPS):
    """Unmask confidence-first. Progressive commitment is what breaks
    mode averaging: each round conditions on tokens already sampled."""
    def body(carry, i):
        cur, mk = carry
        t = jnp.sum(mk, 1) / N
        logits, _ = denoise(p, cur, mk, t)
        pred = jnp.argmax(logits, -1)
        conf = jnp.max(jax.nn.softmax(logits, -1), -1) - 1e9 * (1 - mk)
        n_keep = jnp.maximum(1, (jnp.sum(mk, 1) / (steps - i)).astype(jnp.int32))
        thresh = jnp.sort(conf, 1)[:, ::-1]
        cut = jnp.take_along_axis(thresh, (n_keep - 1)[:, None], 1)
        reveal = mk * (conf >= cut).astype(jnp.float32)
        return (jnp.where(reveal > 0, pred, cur), mk * (1 - reveal)), None

    (final, _), _ = jax.lax.scan(body, (tokens * (1 - mask).astype(jnp.int32),
                                        mask), jnp.arange(steps))
    return jnp.sum((final == tokens) * mask) / jnp.sum(mask)


def main():
    S.N = N
    tr, va, vocab, chars = S.load_data()
    tr, va = jnp.asarray(tr), jnp.asarray(va)
    key = jax.random.PRNGKey(0)
    p = init(key, vocab)
    n_par = sum(x.size for x in jax.tree.leaves(p))
    print(f"masked-diffusion Semagram | params {n_par / 1e6:.3f}M | "
          f"n={N} d={D} layers={LAYERS} heads={HEADS} modes={M}")
    print(f"unigram NLL reference = {S.unigram_nll(tr, vocab):.3f}\n", flush=True)

    opt = optax.adamw(LR, weight_decay=0.01)
    st = opt.init(p)
    step = functools.partial(train_step, upd=opt.update)
    t0 = time.time()
    for it in range(1, STEPS + 1):
        key, k1, k2 = jax.random.split(key, 3)
        tokens, _ = S.get_batch(tr, k1, BATCH)
        p, st, (nll, clo) = step(p, st, tokens, k2)
        if it % EVAL == 0 or it == 1:
            key, k3, k4 = jax.random.split(key, 3)
            vt, _ = S.get_batch(va, k3, 64)
            gap = 1.0 - S.gap_clamp(k4, 64, 3, 4)      # 1 == masked
            v_nll, v_1shot = one_shot(p, vt, gap)
            v_iter = iterative(p, vt, gap)
            brg = 1.0 - S.bridge_clamp(64, ends=6)
            print(f"{it:5d} | train nll {nll:.3f} | gap nll {v_nll:.3f} "
                  f"| 1-shot {v_1shot:.3f} | iterative {v_iter:.3f} "
                  f"| endpoint {iterative(p, vt, brg):.3f} "
                  f"| closure {clo:.3f}", flush=True)
    print(f"wall {time.time() - t0:.0f}s")

    vt, _ = S.get_batch(va, jax.random.PRNGKey(5), 4)
    brg = 1.0 - S.bridge_clamp(4, ends=6)
    dec = lambda r: "".join(chars[i] for i in r).replace("\n", " ")
    print("\nendpoint-clamped, iterative unmasking:")
    for i in range(4):
        cur = vt * (1 - brg).astype(jnp.int32)
        out = jnp.where(brg[i] > 0, cur[i], vt[i])
        print(f"  true {dec(vt[i][:6])}|{dec(vt[i][6:-6])}|{dec(vt[i][-6:])}")
    print("  (per-sample decode omitted; see iterative() metric above)")


if __name__ == "__main__":
    main()
