"""
semagram.py -- minimal Semagram LM in JAX. Single file, nanogpt-style.

A "logogram" is a closed loop of n tokens. There is no BOS/EOS and no causal
mask. The forward pass does not stack layers: it finds a stationary point of a
scalar action S[X], so attention is literally jax.grad(energy).

Three architectural commitments:
  1. positions live on S^1  -> token mixing is a symmetric circulant operator,
     applied in O(n log n) via rfft. Real spectral multiplier == even kernel
     == exact reflection (time-reversal) symmetry, by construction.
  2. Q and K are tied      -> the attention logit matrix is symmetric, so the
     attention term is a genuine gradient and there is no V matrix at all.
     Value mixing falls out of differentiating the log-sum-exp.
  3. edges carry SO(2)^(d/2) transports -> gauge-rotating q,k by the cumulative
     edge phase. RoPE is the flat, content-independent special case. Because
     the 2-planes are fixed the transports commute, so the holonomy is just the
     total phase and the winding number is literally round(sum(phi)/2pi).

Training is masked-arc denoising: clamp a random contiguous arc, solve for the
rest, cross-entropy on the free positions.

    pip install jax optax
    python semagram.py

Requires: jax, optax, numpy.
"""

import functools
import os
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import optax

# ----------------------------------------------------------------------------
# config

N = 48  # loop length (tokens per logogram)
D = 96  # model width (must be even: d/2 rotation planes)
M = 12  # circulant kernel modes retained (band limit)
K_STEPS = 8  # stationarity solver sweeps
BETA = 1.0  # inverse temperature in the attention energy
ETA = 0.8  # step size for SOLVER="descent"
RHO = 12.0  # proximal weight for SOLVER="cccp"
LAM = 10.0  # clamp weight ("descent" only; "cccp" clamps hard)
SOLVER = "descent"  # "descent" = truncated, trains well | "cccp" = monotone
EMB_STD = 0.02  # 0.02 trains much faster; 1.0 conditions the solve far better
                # (rmsnorm Jacobian ~1/EMB_STD). They pull opposite ways, and
                # which one is right depends on fixing the solver first.
IMPLICIT = False  # implicit diff (O(1) memory) vs unrolled backprop

W_HOLO = 0.2  # loop-closure penalty weight
BATCH = 16
LR = 3e-3
STEPS = 2000
EVAL_EVERY = 250
SEED = 0

TASK = "bridge"  # "bridge" = synthetic BVP unit test, "text" = char-level

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")


# ----------------------------------------------------------------------------
# data

def load_data():
    if not os.path.exists("input.txt"):
        try:
            urllib.request.urlretrieve(DATA_URL, "input.txt")
        except Exception:
            # offline fallback: a corpus where endpoints constrain the middle
            rng = np.random.default_rng(0)
            toks = []
            for _ in range(20000):
                a, b = rng.integers(0, 26, 2)
                mid = (np.arange(6) + a) % 26
                toks += [a, *mid, b]
            open("input.txt", "w").write(
                "".join(chr(97 + t) for t in toks))
    text = open("input.txt").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    split = int(0.9 * len(ids))
    return ids[:split], ids[split:], len(chars), chars


def get_batch(data, key, batch=BATCH):
    """Random loops plus a random clamped arc per loop."""
    k1, k2, k3 = jax.random.split(key, 3)
    starts = jax.random.randint(k1, (batch,), 0, len(data) - N)
    idx = starts[:, None] + jnp.arange(N)[None, :]
    tokens = data[idx]

    arc_start = jax.random.randint(k2, (batch, 1), 0, N)
    arc_len = jax.random.randint(k3, (batch, 1), N // 4, 3 * N // 4)
    offset = (jnp.arange(N)[None, :] - arc_start) % N
    clamp = (offset < arc_len).astype(jnp.float32)  # 1 = given, 0 = solve for
    return tokens, clamp


def bridge_clamp(batch, ends=6):
    """Endpoints only: the Arrival test. Know the start and the end, solve the middle."""
    pos = jnp.arange(N)
    m = (pos < ends) | (pos >= N - ends)
    return jnp.broadcast_to(m.astype(jnp.float32), (batch, N))


# ----------------------------------------------------------------------------
# the bridge unit test
#
# x[k] = a for k < N/2, else b.  Position 0 holds a, position N-1 holds b, and
# those two are adjacent on the loop -- so the free arc is one contiguous
# stretch with a boundary condition pinned at each end.
#
# Deliberately NOT harmonic: the smoothness prior alone interpolates a ramp
# between the two clamped values, not a step, so the circulant term cannot
# shortcut this. Information has to reach the interior through the content
# pathway. And nothing in the prefix reveals b, so a causal model is
# structurally incapable of the second half no matter how well trained.

BRIDGE_VOCAB = 26


def bridge_batch(key, batch):
    ka, kb = jax.random.split(key)
    a = jax.random.randint(ka, (batch, 1), 0, BRIDGE_VOCAB)
    b = jax.random.randint(kb, (batch, 1), 0, BRIDGE_VOCAB)
    first = jnp.arange(N)[None, :] < N // 2
    return jnp.where(first, a, b).astype(jnp.int32)


def clamp_at(batch, positions):
    m = jnp.zeros(N).at[jnp.asarray(positions)].set(1.0)
    return jnp.broadcast_to(m, (batch, N))


@jax.jit
def eval_bridge(p, tokens, cl_both, cl_prefix):
    """Same weights, same input, two different boundary conditions."""
    def score(cl):
        x = solve(p, p["emb"][tokens], cl)
        pred = jnp.argmax(rmsnorm(x) @ p["head"], -1)
        free = 1.0 - cl
        hit = (pred == tokens) * free
        return (jnp.sum(hit) / jnp.sum(free),
                jnp.mean(jnp.sum(hit, 1) == jnp.sum(free, 1)))
    (a_both, x_both), (a_pre, _) = score(cl_both), score(cl_prefix)
    return a_both, x_both, a_pre


def main_bridge():
    key = jax.random.PRNGKey(SEED)
    key, sk = jax.random.split(key)
    p = init(sk, BRIDGE_VOCAB)
    print(f"bridge unit test | loop n={N} d={D} modes={M} K={K_STEPS} "
          f"| vocab {BRIDGE_VOCAB} | params "
          f"{sum(x.size for x in jax.tree.leaves(p)) / 1e6:.3f}M")
    print(f"chance = {1 / BRIDGE_VOCAB:.3f} | prefix-only ceiling ~= 0.52\n")

    opt = optax.adamw(LR, weight_decay=0.01)
    opt_state = opt.init(p)
    step = functools.partial(train_step, opt_update=opt.update)
    cl_both = clamp_at(BATCH, [0, N - 1])

    for it in range(1, STEPS + 1):
        key, k1 = jax.random.split(key)
        tokens = bridge_batch(k1, BATCH)
        p, opt_state, (nll, resid) = step(p, opt_state, tokens, cl_both)

        if it % EVAL_EVERY == 0 or it == 1:
            key, k2 = jax.random.split(key)
            vt = bridge_batch(k2, 128)
            ab, xb, ap = eval_bridge(p, vt, clamp_at(128, [0, N - 1]),
                                     clamp_at(128, [0]))
            print(f"{it:5d} | nll {nll:.3f} | both-ends {ab:.3f} "
                  f"(exact {xb:.3f}) | prefix-only {ap:.3f} "
                  f"| gap {ab - ap:+.3f} | closure {resid:.3f}")
    return p


# ----------------------------------------------------------------------------
# params

def init(key, vocab):
    k = jax.random.split(key, 6)
    s = 1.0 / np.sqrt(D)
    return {
        "emb": jax.random.normal(k[0], (vocab, D)) * EMB_STD,
        "head": jax.random.normal(k[1], (D, vocab)) * s,
        # spectral multiplier of the quadratic (geometric) term, real & banded.
        # real  ==  even kernel  ==  reflection-symmetric circulant.
        "g_raw": jnp.zeros((M + 1, D)),
        "wqk": jax.random.normal(k[2], (D, D)) * s,  # tied Q=K
        "wphi": jax.random.normal(k[3], (D, D // 2)) * s * 0.1,
        "wh": jax.random.normal(k[4], (D, 4 * D)) * s,  # Hopfield "FFN"
        "bh": jnp.zeros((4 * D,)),
        "alpha": jnp.array(0.5),
    }


def spectral_multiplier(p):
    """g_m >= 1 for all m, hard zero-pad past the band limit M.

    Positivity makes the quadratic part of the action strictly convex, which
    bounds S below and hands us an exact diagonal preconditioner in Fourier.
    """
    g = 1.0 + jax.nn.softplus(p["g_raw"])
    return jnp.pad(g, ((0, N // 2 + 1 - (M + 1)), (0, 0)),
                   constant_values=1e3)  # (N//2+1, D)


def circulant(x, g):
    """Symmetric circulant apply on the loop axis. O(n log n), exact."""
    return jnp.fft.irfft(jnp.fft.rfft(x, axis=1) * g[None], n=N, axis=1)


def rmsnorm(x):
    return x / jnp.sqrt(jnp.mean(x ** 2, -1, keepdims=True) + 1e-6)


def gauge_rotate(q, phi_cum):
    """Rotate each 2-plane of q by its cumulative edge phase (RoPE, learned)."""
    c, s = jnp.cos(phi_cum), jnp.sin(phi_cum)
    a, b = q[..., 0::2], q[..., 1::2]
    return jnp.stack([a * c - b * s, a * s + b * c], -1).reshape(q.shape)


# ----------------------------------------------------------------------------
# the action

def s_quad(p, x):
    """Convex half: (1/2)<X, A X>, A symmetric circulant PSD. Exact in Fourier."""
    return 0.5 * jnp.sum(x * circulant(x, spectral_multiplier(p)))


def s_rest(p, x):
    """The non-quadratic half. Both terms enter negatively, and -logsumexp of a
    quadratic form supplies genuine negative curvature -- which is why this half
    gets majorized by its tangent plane instead of descended on."""
    xn = rmsnorm(x)

    # tied Q=K => symmetric logits => a real gradient field, and no V matrix.
    q = xn @ p["wqk"]
    phi = (2 * np.pi / N) * jnp.tanh(xn @ p["wphi"])
    q = gauge_rotate(q, jnp.cumsum(phi, axis=1))
    logits = jnp.einsum("bid,bjd->bij", q, q) / np.sqrt(D)
    s_att = -(1.0 / BETA) * jnp.sum(jax.nn.logsumexp(BETA * logits, axis=-1))

    # softplus stays linear at large |X|, so the quadratic always dominates
    s_hop = -p["alpha"] * jnp.sum(jax.nn.softplus(xn @ p["wh"] + p["bh"]))
    return s_att + s_hop


def action(p, x, c, clamp):
    """Full action S[X] with the clamped arc as a quadratic data term."""
    dat = 0.5 * LAM * jnp.sum(clamp[..., None] * (x - c) ** 2)
    return s_quad(p, x) + dat + s_rest(p, x)


def solve(p, c, clamp):
    """Find X on the free arc, given X = C on the clamped arc.

    Two solvers, and the difference between them is the main open problem here.

    "descent": preconditioned gradient descent on the full action, truncated at
    K_STEPS. This is what trains and what produced the bridge result. It is NOT
    a converged fixed point -- measured spectral radius of the iteration matrix
    is >1 for every step size, because -logsumexp of a quadratic form supplies
    genuine negative curvature. So treat it as an 8-sweep weight-tied unrolled
    network that happens to be parameterised by an energy. Legitimate model,
    but it is not a DEQ, and IMPLICIT must stay False for it.

    "cccp": projected prox-linear. Majorises the non-quadratic half by its
    tangent plane and solves the convex half exactly in Fourier. Descends
    monotonically at RHO >= 12 (verified), which descent provably cannot. But
    the residual plateaus rather than vanishing: projecting after applying the
    nonlocal (A + rho*I)^-1 does not solve A restricted to the free arc. The fix
    is CG on the reduced system, matrix-free with A applied by FFT -- roughly
    ten lines, not yet written. Until then IMPLICIT is unsound for both.
    """
    g = spectral_multiplier(p)
    keep = clamp[..., None]
    project = lambda x: keep * c + (1.0 - keep) * x

    if SOLVER == "cccp":
        inv = 1.0 / (g + RHO)
        grad_rest = jax.grad(s_rest, argnums=1)

        def sweep(x, _):
            return project(circulant(RHO * x - grad_rest(p, x), inv)), None

        x0 = project(jnp.broadcast_to(jnp.mean(p["emb"], axis=0), c.shape))
    else:
        inv_geo = 1.0 / (g + 1.0)          # circulant part, exact in Fourier
        jac = 1.0 + LAM * clamp[..., None]  # clamp part, diagonal in real space
        grad_x = jax.grad(action, argnums=1)

        def sweep(x, _):
            gr = circulant(grad_x(p, x, c, clamp), inv_geo) / jac
            return x - ETA * gr, None

        x0 = keep * c + (1.0 - keep) * jnp.mean(p["emb"], axis=0)

    x, _ = jax.lax.scan(sweep, x0, None, length=K_STEPS)
    return x


def holonomy(p, x):
    """Total phase around the loop. Flat <=> globally coherent reading."""
    phi = (2 * np.pi / N) * jnp.tanh(rmsnorm(x) @ p["wphi"])
    total = jnp.sum(phi, axis=1)  # (B, D//2); planes commute, so this is it
    resid = jnp.arctan2(jnp.sin(total), jnp.cos(total))  # wrap to [-pi, pi]
    winding = jnp.round(total / (2 * np.pi))
    return jnp.mean(jnp.abs(resid)), winding


def loss_fn(p, tokens, clamp):
    c = p["emb"][tokens]
    x = solve(p, c, clamp)
    logits = rmsnorm(x) @ p["head"]
    ce = optax.softmax_cross_entropy_with_integer_labels(logits, tokens)
    free = 1.0 - clamp
    nll = jnp.sum(ce * free) / (jnp.sum(free) + 1e-6)
    resid, _ = holonomy(p, x)
    return nll + W_HOLO * resid, (nll, resid)


# ----------------------------------------------------------------------------
# train / eval

@functools.partial(jax.jit, static_argnames=("opt_update",))
def train_step(p, opt_state, tokens, clamp, opt_update):
    (l, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(p, tokens, clamp)
    updates, opt_state = opt_update(grads, opt_state, p)
    return optax.apply_updates(p, updates), opt_state, aux


@jax.jit
def evaluate(p, tokens, clamp):
    """Masked-arc NLL, endpoint-bridge accuracy, and closure residual."""
    _, (nll, resid) = loss_fn(p, tokens, clamp)

    bc = bridge_clamp(tokens.shape[0])
    xb = solve(p, p["emb"][tokens], bc)
    pred = jnp.argmax(rmsnorm(xb) @ p["head"], -1)
    free = 1.0 - bc
    acc = jnp.sum((pred == tokens) * free) / jnp.sum(free)
    _, winding = holonomy(p, xb)
    return nll, acc, resid, jnp.mean(jnp.abs(winding))


def main():
    key = jax.random.PRNGKey(SEED)
    tr, va, vocab, chars = load_data()
    tr, va = jnp.asarray(tr), jnp.asarray(va)
    print(f"vocab {vocab} | train {len(tr):,} | val {len(va):,} | "
          f"loop n={N} d={D} modes={M} solver K={K_STEPS}")

    key, sk = jax.random.split(key)
    p = init(sk, vocab)
    n_params = sum(x.size for x in jax.tree.leaves(p))
    print(f"params {n_params / 1e6:.2f}M\n")

    opt = optax.adamw(LR, weight_decay=0.01)
    opt_state = opt.init(p)
    step = functools.partial(train_step, opt_update=opt.update)

    for it in range(1, STEPS + 1):
        key, k1 = jax.random.split(key)
        tokens, clamp = get_batch(tr, k1)
        p, opt_state, (nll, resid) = step(p, opt_state, tokens, clamp)

        if it % EVAL_EVERY == 0 or it == 1:
            key, k2 = jax.random.split(key)
            vt, vc = get_batch(va, k2, batch=64)
            v_nll, v_acc, v_res, v_wind = evaluate(p, vt, vc)
            print(f"{it:5d} | train nll {nll:.3f} | val nll {v_nll:.3f} "
                  f"| bridge acc {v_acc:.3f} | closure {v_res:.3f} "
                  f"| |winding| {v_wind:.2f}")

    # qualitative: clamp both endpoints, read out the solved interior
    key, k3 = jax.random.split(key)
    vt, _ = get_batch(va, k3, batch=4)
    bc = bridge_clamp(4)
    x = solve(p, p["emb"][vt], bc)
    pred = jnp.argmax(rmsnorm(x) @ p["head"], -1)
    dec = lambda row: "".join(chars[i] for i in row).replace("\n", " ")
    print("\nendpoint-clamped reconstruction (| marks the free arc):")
    for i in range(4):
        e = 6
        print(f"  true {dec(vt[i][:e])}|{dec(vt[i][e:-e])}|{dec(vt[i][-e:])}")
        print(f"  pred {dec(vt[i][:e])}|{dec(pred[i][e:-e])}|{dec(vt[i][-e:])}")


if __name__ == "__main__":
    main_bridge() if TASK == "bridge" else main()
