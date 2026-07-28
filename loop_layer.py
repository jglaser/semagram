"""loop_layer.py -- the Semagram solve as a reusable layer, with a non-abelian
gauge option and constraints you can add after training.

`semagram.py` is a demo that happens to contain a layer. This file is the layer,
pulled out and given the two things that make it worth using rather than
admiring:

  (A) TEST-TIME ENERGY TERMS.  The forward pass is argmin_X S[X], so a new
      constraint is a new term in S. `solve(..., extra=f)` adds f(X) to the
      action and descends on the sum, using the same sweeps and the same
      preconditioner. Nothing is retrained. A transformer has no analogue: its
      forward pass is not the minimiser of anything, so a constraint has to be
      bolted on outside as search, rejection, or a fine-tune.

      This is the property the Arrival framing is actually about. A logogram is
      written all at once, with the ending known before the beginning; the
      engineering content of that is not "attention on a circle" but "state
      anything you know, anywhere on the loop, and solve for a whole that is
      consistent with all of it".

  (B) NON-ABELIAN HOLONOMY.  semagram.py transports along each edge by
      SO(2)^(d/2). Those commute, so the holonomy of a loop is the SUM of the
      edge phases and nothing else: two loops carrying the same multiset of
      edge phases in different orders are indistinguishable to it. For a
      structure whose whole claim is that going around the loop in order is
      what produces meaning, an order-blind holonomy is a strange thing to
      have.

      `gauge="su2"` transports by SU(2)^(d/4) instead -- unit quaternions
      acting on R^4 = C^2 by left multiplication -- composed as a PATH-ORDERED
      product. These do not commute, so the holonomy sees the arrangement and
      not just the total. SO(2) is recovered exactly when every edge rotates
      about the same axis, which makes `gauge` a clean ablation rather than a
      different model.

      The ordered product is computed with `lax.associative_scan`, so it costs
      O(n log n) -- the same budget the circulant attention is justified by.

      There is a specific reason to expect this to matter on closed curves.
      A closed curve's constraint is that the rigid motions along its edges
      compose to the identity, and rigid motions of the plane are SE(2), which
      is NOT abelian. The abelian holonomy can only ever express the part of
      that constraint which is a sum -- sum(turning) = 2*pi, the tangent
      winding. The part that says the curve actually joins up,
      sum_i exp(i Phi_i) = 0, is exactly the non-commuting part. So the
      prediction is sharp: on this data su2 should buy something and, on data
      with no such constraint, it should not.

Everything numeric that is not about the gauge -- the circulant operator, the
spectral multiplier, rmsnorm, the preconditioner -- is imported from
semagram.py rather than copied, so the two files cannot drift apart.
"""

from __future__ import annotations

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import optax

import semagram as S

TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------------
# config

@dataclasses.dataclass(frozen=True)
class LoopCfg:
    """Frozen so it can be a static jit argument; see semagram.Cfg."""
    n: int = 48
    d: int = 96
    heads: int = 4
    modes: int = 12
    k_steps: int = 8
    beta: float = 1.0
    eta: float = 0.8
    lam: float = 10.0
    g_floor: float = 1.0
    g_init: float = 0.0
    band_pad: float = 1.0
    tied: bool = False
    flat: bool = True
    phi_dev: float = 0.5
    hard_clamp: bool = True
    tie_head: bool = True
    head_scale: float = 1.0
    emb_std: float = 1.0
    norm_eps: float = 1e-6
    w_holo: float = 0.2
    vocab: int = 32
    gauge: str = "so2"        # "so2" (abelian, as shipped) | "su2" | "none"
    gauge_close: bool = True  # renormalise the connection so the loop holonomy
                              # is exactly trivial. False reproduces
                              # semagram.py, whose branch cut at index 0 breaks
                              # the cyclic-shift equivariance that is the whole
                              # premise. See node_gauge.

    def __post_init__(self):
        assert self.d % 4 == 0, "d must be divisible by 4 (quaternion blocks)"
        assert (self.d // self.heads) % 4 == 0, "d/heads must be divisible by 4"
        assert self.modes <= self.n // 2, "kernel band exceeds Nyquist"
        assert self.gauge in ("so2", "su2", "none")

    @property
    def units(self):
        """Number of independent transports: 2-planes for so2, 4-blocks for su2."""
        return self.d // (2 if self.gauge == "so2" else 4)


@functools.lru_cache(maxsize=None)
def windings(cfg):
    """Integer winding per transport unit: RoPE quantised to the ring.

    Integers are what make the FLAT part of the holonomy exactly trivial -- the
    base transport composes to a full 2*pi*m turn, hence the identity -- so the
    measured holonomy is the deviation from flatness and nothing else. Same
    geometric spread as semagram.windings, over `units` rather than d/2.
    """
    u = cfg.units
    if not cfg.flat:
        return np.zeros(u, dtype=np.float64)
    j = np.arange(u)
    # float64, not float32. These are exact integers, but the base phase they
    # generate is 2*pi*m with m up to n/2, and a float32 round-trip there leaves
    # a residual of ~1e-5 radians in the loop closure -- which is six orders of
    # magnitude above the float64 noise floor and was, on its own, enough to
    # stop the closed connection from being exactly closed.
    return np.round((cfg.n // 2) ** (1.0 - j / max(u - 1, 1))).astype(np.float64)


# ----------------------------------------------------------------------------
# quaternions -- SU(2) acting on R^4 = C^2 by left multiplication.
#
# Left multiplication by a UNIT quaternion is an orthogonal map of R^4, which is
# the property the whole gauge construction rests on: attention logits are inner
# products, so transporting q and k by Q_i and Q_j leaves the logit depending
# only on the relative transport Q_i^{-1} Q_j along the path between them. That
# is RoPE's defining property, and it survives the move to a non-abelian group
# unchanged. What does NOT survive is Q_i^{-1} Q_j depending only on (i - j):
# with non-commuting transports it depends on the content in between, which is
# the entire point.

def qmul(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], axis=-1)


def qexp(w):
    """exp of an su(2) element given as a 3-vector. Safe and smooth at w = 0.

    The doubled `where` is the standard trick for a gradient that would
    otherwise be nan at the origin: the branch that divides by |w| must not be
    evaluated with |w| = 0 even on the untaken side.
    """
    n2 = jnp.sum(w * w, axis=-1, keepdims=True)
    th = jnp.sqrt(jnp.maximum(n2, 1e-24))
    safe = jnp.where(n2 > 1e-16, th, 1.0)
    sinc = jnp.where(n2 > 1e-16, jnp.sin(safe) / safe, 1.0 - n2 / 6.0)
    return jnp.concatenate([jnp.cos(th), sinc * w], axis=-1)


def qprefix(q):
    """Exclusive path-ordered product along the loop axis, in O(n log n).

    Q_i = q_0 q_1 ... q_{i-1}, with Q_0 = 1. Inclusive first via
    associative_scan (qmul is associative but not commutative, which is exactly
    what associative_scan requires and exactly what makes this interesting),
    then shifted by one so node i carries the transport of the edges STRICTLY
    before it -- the same exclusive convention as semagram.cumulative_phase,
    and for the same reason: the phase belongs to the edge, not the node.
    """
    inc = jax.lax.associative_scan(qmul, q, axis=1)
    ident = jnp.zeros_like(inc[:, :1]).at[..., 0].set(1.0)
    return jnp.concatenate([ident, inc[:, :-1]], axis=1), inc[:, -1]


def qapply(Q, v):
    """Left-multiply the 4-blocks of v (b, n, d) by the transports Q."""
    b, n, d = v.shape
    return qmul(Q, v.reshape(b, n, d // 4, 4)).reshape(b, n, d)


# ----------------------------------------------------------------------------
# params

def init(key, cfg):
    k = jax.random.split(key, 8)
    s = 1.0 / np.sqrt(cfg.d)
    dev_out = cfg.units * (3 if cfg.gauge == "su2" else 1)
    return {
        "emb": jax.random.normal(k[0], (cfg.vocab, cfg.d)) * cfg.emb_std,
        "head": jax.random.normal(k[1], (cfg.d, cfg.vocab)) * s,
        "bhead": jnp.zeros((cfg.vocab,)),
        "hscale": jnp.array(cfg.head_scale),
        "g_raw": jnp.full((cfg.modes + 1, cfg.d), cfg.g_init),
        "wqk": jax.random.normal(k[2], (cfg.d, cfg.d)) * s,
        "wk": jax.random.normal(k[3], (cfg.d, cfg.d)) * s,
        "wphi": jax.random.normal(k[4], (cfg.d, dev_out)) * s * 0.1,
        "wh": jax.random.normal(k[5], (cfg.d, 4 * cfg.d)) * s,
        "bh": jnp.zeros((4 * cfg.d,)),
        "alpha": jnp.array(0.5),
        "prior": jax.random.normal(k[6], (cfg.d,)),
    }


# ----------------------------------------------------------------------------
# the connection

def edge_generators(p, xn, cfg):
    """Per-EDGE Lie-algebra element per transport unit: flat base + content.

    Reflection bookkeeping is identical to semagram.edge_phases and holds for
    the same reasons: `dx` is antisymmetric in the edge's endpoints and tanh is
    odd, so the CONTENT term reverses with the loop and cannot break reflection
    however it is trained; the flat base is attached to the ring, does not
    reverse, and is the only symmetry-breaking term in the model.
    """
    base = jnp.asarray(windings(cfg))
    dx = xn - jnp.roll(xn, -1, axis=1)
    dev = cfg.phi_dev * jnp.tanh(dx @ p["wphi"])
    b, n, _ = xn.shape
    if cfg.gauge == "su2":
        dev = dev.reshape(b, n, cfg.units, 3)
        # The base turns about a fixed axis (e_z), which is precisely the
        # abelian sub-case; the content term tilts that axis, and a tilted
        # rotation composed with an untilted one does not commute. That is
        # where the non-abelian behaviour comes from -- not from the magnitude
        # of the deviation but from its direction varying along the loop.
        axis = jnp.zeros((cfg.units, 3)).at[:, 2].set(base)
        return (TWO_PI / cfg.n) * (axis[None, None] + dev)
    return (TWO_PI / cfg.n) * (base[None, None] + dev)


@functools.lru_cache(maxsize=None)
def _base_angles(cfg):
    """Flat-base angle at each node, reduced mod 2*pi in exact integer
    arithmetic before it is ever a float.

    2*pi*m*i/n reaches 148 radians at m = n/2 = 24, and in float32 that carries
    an absolute error of ~1e-5 rad -- which would put a floor under the
    shift-equivariance two orders of magnitude above float32 epsilon, on the one
    number this layer is supposed to be exact about. Since m and i are integers,
    (m*i mod n)/n is exact, and the angle never leaves [0, 2*pi).
    """
    m = windings(cfg).astype(np.int64)
    k = (np.arange(cfg.n)[:, None] * m[None, :]) % cfg.n
    return (TWO_PI * k / cfg.n).astype(np.float64)      # (n, units)


def node_gauge(p, xn, cfg):
    """The pure-gauge transport: flat base times a content-dependent rotation.

    This is not a heuristic, it is the only construction that can work, and the
    reason is topology rather than numerics.

    A connection on a circle is classified, up to gauge, by ONE invariant: its
    holonomy. Everything else about it is pure gauge. Equivalently: parallel
    transport is single-valued around the loop if and only if the holonomy is
    trivial, and a layer built from a multi-valued transport has to cut the
    circle somewhere to define itself. semagram.py cuts it at index 0, via the
    exclusive cumsum in `cumulative_phase`.

    So the two things Semagram wants are in direct conflict:

        "ambiguity is holonomy"   wants the holonomy NON-trivial
        "there is no index 0"     requires the holonomy trivial

    and no amount of implementation care reconciles them. Measured in float64
    on an untrained model, the relative change in output logits under a cyclic
    shift of the loop is exactly proportional to the holonomy, and vanishes to
    1.7e-15 exactly when the holonomy does:

        gauge  phi_dev   shift error   holonomy
        so2      0.00      2.7e-15      3.6e-15
        so2      0.10      2.0e-05      1.1e-04
        so2      1.00      2.0e-04      1.1e-03
        su2      0.00      1.7e-15      3.9e-08
        su2      1.00      1.1e-02      1.5e-01

    That reframes the README's finding that the holonomy mechanism "earns
    nothing measurable" on text. It was not inert; it was quietly spending
    commitment 1 to pay for commitment 3, on a task where neither could be
    seen.

    The resolution is to stop making the transport carry both. The holonomy is
    computed from the edge generators and kept as a readout and a loss term --
    it is still the coherence measure it was meant to be -- while the transport
    used by attention is the pure-gauge part, which is exactly periodic and
    therefore exactly shift-equivariant:

        Q_i = B_i G_i,   B_i = flat base (integer winding, exactly periodic)
                         G_i = exp(phi_dev * tanh(x_i W))

    Both factors are the same for either group, so `gauge` remains a clean
    ablation. What changes with the group is the relative transport a pair of
    positions can see: an SO(2) phase difference, or an SU(2) element, which
    does not commute and so can encode an ordering that a phase cannot.

    One honest cost: G is pointwise in the content, whereas the path-ordered
    prefix was nonlocal. The nonlocality survives only in the holonomy readout.
    """
    b, n, _ = xn.shape
    ang = jnp.asarray(_base_angles(cfg))
    g = cfg.phi_dev * jnp.tanh(xn @ p["wphi"])
    if cfg.gauge != "su2":
        return ang[None] + g
    B = qexp(jnp.stack([jnp.zeros_like(ang), jnp.zeros_like(ang), ang], -1))
    G = qexp(g.reshape(b, n, cfg.units, 3))
    return qmul(B[None], G)


def transports(p, xn, cfg):
    """Node transports Q_i, and the RAW path-ordered loop holonomy.

    The holonomy is always measured the original way -- the ordered product of
    the per-edge generators -- so it means the same thing in both modes. Only
    the transport handed to attention differs. See node_gauge.
    """
    w = edge_generators(p, xn, cfg)
    if cfg.gauge == "su2":
        Qraw, raw = qprefix(qexp(w))
    else:
        Qraw, raw = jnp.cumsum(w, axis=1) - w, jnp.sum(w, axis=1)
    return (node_gauge(p, xn, cfg) if cfg.gauge_close else Qraw), raw


def apply_transport(v, Q, cfg):
    if cfg.gauge == "su2":
        return qapply(Q, v)
    if cfg.gauge == "none":
        return v
    return S.gauge_rotate(v, Q)


def holonomy(p, x, cfg):
    """Deviation of the loop holonomy from the identity, and a winding count.

    so2: the residual is the total content phase wrapped to [-pi, pi], exactly
    as in semagram.holonomy -- the flat base contributes 2*pi*m and is
    invisible by construction.

    su2: the residual is the geodesic angle of the ORDERED product from the
    identity, 2*arccos|Re Q|. This is strictly more information: it is zero
    only if the transports actually undo each other around the loop, whereas
    the abelian residual is zero whenever the phases merely sum to zero.
    """
    xn = S.rmsnorm(x, cfg.norm_eps)
    _, loop = transports(p, xn, cfg)
    if cfg.gauge == "su2":
        # 2*atan2(|Im|, |Re|), NOT 2*arccos|Re|. They agree in value, but
        # arccos has infinite derivative at 1 and the holonomy of a
        # freshly-initialised model sits essentially at the identity, so the
        # loss term differentiates to nan on the very first step. atan2 is
        # smooth there. This was the entire su2 nan.
        re = jnp.abs(loop[..., 0])
        im = jnp.sqrt(jnp.maximum(jnp.sum(loop[..., 1:] ** 2, -1), 1e-24))
        ang = 2.0 * jnp.arctan2(im, re)
        return jnp.mean(ang), jnp.mean(ang / TWO_PI)
    resid = jnp.arctan2(jnp.sin(loop), jnp.cos(loop))
    return jnp.mean(jnp.abs(resid)), jnp.mean(jnp.abs(jnp.round(loop / TWO_PI)))


# ----------------------------------------------------------------------------
# the action

def s_rest(p, x, cfg):
    xn = S.rmsnorm(x, cfg.norm_eps)
    Q, _ = transports(p, xn, cfg)
    q = apply_transport(xn @ p["wqk"], Q, cfg)
    k = q if cfg.tied else apply_transport(xn @ p["wk"], Q, cfg)
    b, n, d = q.shape
    dh = d // cfg.heads
    logits = jnp.einsum("bihe,bjhe->bhij", q.reshape(b, n, cfg.heads, dh),
                        k.reshape(b, n, cfg.heads, dh)) / np.sqrt(dh)
    s_att = -(1.0 / cfg.beta) * jnp.sum(
        jax.nn.logsumexp(cfg.beta * logits, axis=-1))
    s_hop = -p["alpha"] * jnp.sum(jax.nn.softplus(xn @ p["wh"] + p["bh"]))
    return s_att + s_hop


def action(p, x, c, clamp, cfg, extra=None):
    dat = 0.5 * cfg.lam * jnp.sum(clamp[..., None] * (x - c) ** 2)
    tot = S.s_quad(p, x, cfg) + dat + s_rest(p, x, cfg)
    return tot if extra is None else tot + extra(x)


def solve(p, c, clamp, cfg, extra=None, x0=None, trace=False):
    """Solve the boundary-value problem on the ring.

    `extra` is an arbitrary scalar function of X added to the action. It is the
    whole point of the file: a constraint discovered after training is a term
    here, needs no gradient of its own w.r.t. parameters, and is descended on by
    the same preconditioned sweeps as everything else. Pass it as a static
    argument when jitting.
    """
    g = S.spectral_multiplier(p, cfg)
    keep = clamp[..., None]
    free = 1.0 - keep
    inv_geo = 1.0 / (g + 1.0)
    jac = 1.0 + cfg.lam * keep
    grad_x = jax.grad(lambda q, x, cc, cl: action(q, x, cc, cl, cfg, extra),
                      argnums=1)

    def sweep(x, _):
        gr = S.circulant(grad_x(p, x, c, clamp), inv_geo, cfg.n) / jac
        x = x - cfg.eta * gr
        if cfg.hard_clamp:
            x = keep * c + free * x
        return x, (jnp.sqrt(jnp.sum((free * gr) ** 2)) if trace else None)

    x0 = p["prior"] if x0 is None else x0
    x, res = jax.lax.scan(sweep, keep * c + free * x0, None, length=cfg.k_steps)
    return (x, res) if trace else x


def logits_of(p, x, cfg):
    w = (p["hscale"] * p["emb"].T) if cfg.tie_head else p["head"]
    return S.rmsnorm(x) @ w + p["bhead"]


def loss_fn(p, tokens, clamp, cfg):
    x = solve(p, p["emb"][tokens], clamp, cfg)
    ce = optax.softmax_cross_entropy_with_integer_labels(
        logits_of(p, x, cfg), tokens)
    free = 1.0 - clamp
    nll = jnp.sum(ce * free) / (jnp.sum(free) + 1e-6)
    resid, _ = holonomy(p, x, cfg)
    return nll + cfg.w_holo * resid, (nll, resid)


# ----------------------------------------------------------------------------
# train / eval

def _step(p, opt_state, tokens, clamp, opt_update, cfg):
    (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        p, tokens, clamp, cfg)
    updates, opt_state = opt_update(grads, opt_state, p)
    return optax.apply_updates(p, updates), opt_state, aux


@functools.partial(jax.jit, static_argnames=("opt_update", "cfg"))
def train_chunk(p, opt_state, toks, clamps, opt_update, cfg):
    """Many optimiser steps inside one jit, via scan over the chunk axis.

    Not a nicety. On CPU this model is dispatch-bound rather than flop-bound --
    the action's gradient is a long chain of small ops -- and stepping it from
    Python costs about as much as computing it. Fusing 25 steps into one traced
    scan roughly halves wall-clock at identical arithmetic, which is the
    difference between running three seeds and running one.
    """
    def body(carry, xs):
        p, st = carry
        p, st, aux = _step(p, st, xs[0], xs[1], opt_update, cfg)
        return (p, st), aux
    (p, opt_state), aux = jax.lax.scan(body, (p, opt_state), (toks, clamps))
    return p, opt_state, jax.tree.map(lambda a: a[-1], aux)


@functools.partial(jax.jit, static_argnames=("cfg",))
def score(p, tokens, clamp, cfg, weight=None):
    x = solve(p, p["emb"][tokens], clamp, cfg)
    lg = logits_of(p, x, cfg)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, tokens)
    w = (1.0 - clamp) if weight is None else weight
    hit = (jnp.argmax(lg, -1) == tokens).astype(jnp.float32)
    tot = jnp.sum(w) + 1e-6
    return jnp.sum(ce * w) / tot, jnp.sum(hit * w) / tot


def n_params(p):
    return sum(x.size for x in jax.tree.leaves(p))


# ----------------------------------------------------------------------------
# baseline: a transformer that is ALSO exactly circular.
#
# The obvious baseline -- learned absolute position embeddings -- is what people
# actually reach for, and it is genuinely mis-specified on a closed curve, since
# there is no first vertex. Reporting only that would be beating up a strawman.
# So there is a second baseline with rotary positions quantised to the ring:
# angles 2*pi*m*i/n with integer m, which makes q_i . k_j a function of
# (i - j) mod n exactly, and with the absolute embedding removed makes the whole
# model exactly equivariant to cyclic shift. That baseline has commitment 1
# built into it and is the honest comparison for everything else.

def tf_init(key, vocab, n, layers, width, heads, ring):
    ks = jax.random.split(key, 4 + 4 * layers)
    s = 1.0 / np.sqrt(width)
    p = {"emb": jax.random.normal(ks[0], (vocab, width)) * 0.02,
         "mask": jax.random.normal(ks[2], (width,)) * 0.02,
         "head": jax.random.normal(ks[3], (width, vocab)) * s,
         "bhead": jnp.zeros((vocab,)), "blocks": []}
    if not ring:
        p["pos"] = jax.random.normal(ks[1], (n, width)) * 0.02
    for i in range(layers):
        k = ks[4 + 4 * i: 8 + 4 * i]
        p["blocks"].append({
            "qkv": jax.random.normal(k[0], (width, 3 * width)) * s,
            "proj": jax.random.normal(k[1], (width, width)) * s,
            "fc1": jax.random.normal(k[2], (width, 4 * width)) * s,
            "fc2": jax.random.normal(k[3], (4 * width, width)) * s,
            "b1": jnp.zeros((4 * width,)), "b2": jnp.zeros((width,)),
        })
    return p


@functools.lru_cache(maxsize=None)
def _ring_rope(n, dh):
    """Cached as NUMPY, converted at the use site.

    Caching jnp arrays here instead leaks a tracer: the first call may happen
    inside a jit/scan trace, and the cache then hands that trace's intermediate
    to every later call. Numpy constants are inlined by XLA and have no such
    problem.
    """
    j = np.arange(dh // 2)
    m = np.round((n // 2) ** (1.0 - j / max(dh // 2 - 1, 1)))
    ang = TWO_PI * np.outer(np.arange(n), m) / n
    return np.cos(ang), np.sin(ang)


def _rope(x, n):
    cn, sn = _ring_rope(n, x.shape[-1])
    c, s = jnp.asarray(cn), jnp.asarray(sn)
    a, b = x[..., 0::2], x[..., 1::2]
    return jnp.stack([a * c - b * s, a * s + b * c], -1).reshape(x.shape)


def tf_forward(p, tokens, clamp, heads, ring):
    b, n = tokens.shape
    e = p["emb"][tokens]
    h = jnp.where(clamp[..., None] > 0, e, p["mask"])
    if not ring:
        h = h + p["pos"][:n]
    for blk in p["blocks"]:
        z = S.layernorm(h)
        q, k, v = jnp.split(z @ blk["qkv"], 3, -1)
        d = q.shape[-1]
        dh = d // heads
        rs = lambda t: t.reshape(b, n, heads, dh).transpose(0, 2, 1, 3)
        qh, kh = rs(q), rs(k)
        if ring:
            qh, kh = _rope(qh, n), _rope(kh, n)
        att = jnp.einsum("bhid,bhjd->bhij", qh, kh) / np.sqrt(dh)
        o = jnp.einsum("bhij,bhjd->bhid", jax.nn.softmax(att, -1), rs(v))
        h = h + o.transpose(0, 2, 1, 3).reshape(b, n, d) @ blk["proj"]
        z = S.layernorm(h)
        h = h + jax.nn.gelu(z @ blk["fc1"] + blk["b1"]) @ blk["fc2"] + blk["b2"]
    return S.layernorm(h) @ p["head"] + p["bhead"]


def tf_loss(p, tokens, clamp, heads, ring):
    ce = optax.softmax_cross_entropy_with_integer_labels(
        tf_forward(p, tokens, clamp, heads, ring), tokens)
    free = 1.0 - clamp
    return jnp.sum(ce * free) / (jnp.sum(free) + 1e-6)


@functools.partial(jax.jit, static_argnames=("heads", "ring", "opt_update"))
def tf_chunk(p, opt_state, toks, clamps, heads, ring, opt_update):
    def body(carry, xs):
        p, st = carry
        l, g = jax.value_and_grad(tf_loss)(p, xs[0], xs[1], heads, ring)
        u, st = opt_update(g, st, p)
        return (optax.apply_updates(p, u), st), l
    (p, opt_state), ls = jax.lax.scan(body, (p, opt_state), (toks, clamps))
    return p, opt_state, ls[-1]


@functools.partial(jax.jit, static_argnames=("heads", "ring"))
def tf_score(p, tokens, clamp, weight, heads, ring):
    lg = tf_forward(p, tokens, clamp, heads, ring)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, tokens)
    hit = (jnp.argmax(lg, -1) == tokens).astype(jnp.float32)
    tot = jnp.sum(weight) + 1e-6
    return jnp.sum(ce * weight) / tot, jnp.sum(hit * weight) / tot


def tf_match(target, vocab, n, heads, ring, layer_opts=(2, 3, 4)):
    """Smallest (layers, width) whose parameter count is closest to `target`."""
    best = None
    for layers in layer_opts:
        for width in range(16, 256, 4 * heads):
            if (width // heads) % 2:
                continue
            cnt = (vocab * width + width + width * vocab + vocab
                   + (0 if ring else n * width)
                   + layers * (4 * width * width + 8 * width * width + 5 * width))
            err = abs(cnt - target) / target
            if best is None or err < best[0]:
                best = (err, layers, width, cnt)
    return best[1], best[2], best[3]
