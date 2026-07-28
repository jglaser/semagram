"""
semagram.py -- minimal Semagram LM in JAX. Single file, nanogpt-style.

A "logogram" is a closed loop of n tokens. There is no BOS/EOS and no causal
mask. The forward pass does not stack layers: it finds a stationary point of a
scalar action S[X], so attention is literally jax.grad(energy).

Three architectural commitments:
  1. positions live on S^1  -> token mixing is a symmetric circulant operator,
     applied in O(n log n) via rfft. Real spectral multiplier == even kernel
     == exact reflection (time-reversal) symmetry of the GEOMETRIC prior.
  2. the action is a scalar -> the layer Jacobian is symmetric and there is no
     separate V matrix; value mixing falls out of differentiating log-sum-exp.
     This is what the energy formulation buys, and it does NOT require tying
     Q and K -- any scalar has a conservative gradient. Tying is off by default
     and kept as an ablation; see note (1).
  3. edges carry SO(2)^(d/2) transports -> gauge-rotating q,k by the cumulative
     edge phase. RoPE is the flat, content-independent special case, and we now
     initialise AT that special case rather than at the trivial connection.
     Because the 2-planes are fixed the transports commute, so the holonomy is
     the total phase and the winding number is round(sum(phi)/2pi). This flat
     base is the ONLY term in the model that breaks reflection symmetry, which
     makes it the difference between modelling text and not.

Training is masked denoising on the loop: clamp part of the ring, solve for the
rest, cross-entropy on the free positions.

    pip install jax optax
    python semagram.py --task text
    python semagram.py --task bridge      # fast synthetic BVP unit test
    python semagram.py --diagnose         # the four structural diagnostics
    python semagram.py --ablate --seeds 3 # leave-one-out over the structural
                                          # flags, plus the tied x flat 2x2

Requires: jax, optax, numpy.

WHAT CHANGED, AND WHY (each of these was a bug, not a hyperparameter)

  (1) Nothing in the model broke reflection symmetry, so a bigram was
      structurally unrepresentable and the unigram-entropy plateau was forced.

      This was NOT, as first supposed, caused by tying Q=K. Tying does make the
      logit matrix exactly symmetric (measured: 0.0), but a symmetric L does not
      imply an equivariant model, and the inference from one to the other is
      simply invalid. Write the tied channel map out per 2-plane:

          W^T R(dth) W = cos(dth) * (W^T W)  +  sin(dth) * (W^T J W)

      The second term is antisymmetric and generically nonzero, and it is odd in
      dth -- so orientation survives tying perfectly well. Measured: tied+flat
      has reflection residual 1.9e-1, i.e. thoroughly directed. Claim refuted by
      d_reflection before any of it was built on.

      What actually happened is that the model had no symmetry-breaking term at
      all. Reflection about the origin sends the loop i -> -i, and:
        - the circulant kernel is even, hence equivariant, by commitment 1;
        - rmsnorm and the Hopfield term are pointwise, hence equivariant;
        - the content-dependent connection is COVARIANT -- reversing the loop
          reverses each transport -- so it cannot break the symmetry either,
          however it is trained.
      With the flat base switched off, the whole solve is therefore equivariant
      to 6e-15 in float64, whatever W_q and W_k are. It could not tell "th" from
      "ht" because there was nothing there to tell them apart with.

      The one term that can break it is a connection attached to the RING rather
      than to the content, because that one does not reverse. That is the flat
      base of fix (3), which is thereby promoted from a helpful initialisation
      to the load-bearing fix. Tying remains a question about the expressivity
      of the channel map -- W_q^T W_k arbitrary vs W^T W symmetric PSD -- which
      is an empirical question, settled by the ablation grid, not by a theorem.

      Untying costs the energy formulation nothing either way: any scalar has a
      conservative gradient, so W_q != W_k still gives a valid action and a
      symmetric Hessian. The claim in the original docstring that tying is
      NEEDED for the gradient property is false.

  (2) The band limit was applied to the SIGNAL, not the kernel. Padding the
      spectral multiplier with 1e3 past mode M meant the preconditioned descent
      recursion multiplied those modes by (1 - eta) every sweep, so after
      K_STEPS they were gone: the free arc was confined to ~M of n/2 modes.
      Text is broadband, so this was a hard capacity ceiling. The kernel is
      band-limited to M learned modes as specified; the state is not.
      Measured by d_spectrum; ablated by cfg.band_pad.

  (3) The connection was initialised trivially, so attention began completely
      position-blind and had to discover relative position through cumsum
      gradients. The docstring already named the fix -- RoPE is the flat case --
      so we start from a non-trivial flat connection with integer winding m_j
      per 2-plane. Closure stays exact (total phase 2*pi*m_j), and holonomy now
      measures deviation FROM that flat section, which is the quantity the
      "distinct flat sections are distinct readings" story actually wants.
      Ablated by cfg.flat.

  (4) The endpoint metric on text measured itself. Predicting 36 Shakespeare
      characters from 12 has enormous conditional entropy and emitting the
      unigram mode is near CE-optimal there. Training and eval now use short
      multi-gap masks as well; the endpoint reconstruction is kept as a
      qualitative display only.

  (5) The readout could not express a constant. logits = rmsnorm(x) @ head is a
      linear map of a unit-norm vector with no bias, so the model could not emit
      the unigram distribution and was losing to a constant predictor: train NLL
      3.28 against a unigram entropy of 3.31. A bias term and a tied readout
      (cfg.tie_head, reading out through emb^T, which under hard clamping
      decodes the given tokens for free) took gap NLL 3.35 -> 2.75 by
      themselves. Least glamorous fix here and the largest single one.

  (6) Weight decay was applied to the tied readout, which collapses long runs.
      See decay_mask: decaying emb and hscale shrinks the logit scale ~0.55x
      over 20k steps and walks the model into the flat-logit unigram basin.
      Every run short enough to finish first looked fine, which is why this hid
      behind three other hypotheses.

STABILITY. The unigram is an absorbing state -- flat readout, uninformative
solved state, no gradient back out -- and at lr 3e-3 the model is bistable
about it: the SAME seed with byte-identical code reached gap NLL 2.356 in one
run and 3.333 in another, diverging only through floating-point
nondeterminism, with the split occurring between steps 4000 and 5000. The
default lr is 2e-3, which has not been observed to collapse and scores better
anyway (2.370). main_text prints a COLLAPSED banner rather than tabulating a
degenerate run as an architecture result.

WHAT WAS TRIED AND DID NOT WORK, all measured, none of them kept:
  - gradient clipping (worse everywhere, and collapses sooner: 4.415 vs 2.639)
  - more solver sweeps, k_steps 10 -> 16 (2.683 vs 2.584)
  - weaker/stronger quadratic conditioning, g_floor and eta (all worse)
  - soft clamping, to let clamped positions contextualise (indistinguishable
    where it works, catastrophic above lam ~ 1)
  - scaling the tied readout down at init (collapses immediately)
  - bounding the content-dependent gauge phase, cfg.phi_dev (neutral 0.0..1.0)

SOLVER STATUS. "descent" is a truncated, preconditioned unroll and is NOT a
converged fixed point -- the measured spectral radius of the iteration matrix
exceeds 1 at every step size, because -logsumexp of a quadratic form supplies
genuine negative curvature. It is an honest weight-tied K-sweep network that
happens to be parameterised by an energy. It is not a DEQ, and implicit
differentiation would be unsound for it. d_residual records this rather than
hiding it.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import time
import urllib.request

import jax
import jax.numpy as jnp
import numpy as np
import optax

DATA_URL = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt")
ABLATION_JSON = "ablation.json"


# ----------------------------------------------------------------------------
# config
#
# A frozen dataclass rather than module globals, so it can be a hashable static
# argument to jit: changing a flag then re-tracing is automatic, which is what
# the ablation grid needs. Module globals would be silently cached.

@dataclasses.dataclass(frozen=True)
class Cfg:
    n: int = 48            # loop length (tokens per logogram)
    d: int = 160           # model width (must be even: d/2 rotation planes)
    heads: int = 4         # attention heads (d/heads must be even)
    modes: int = 12        # circulant KERNEL modes learned (the band limit)
    k_steps: int = 10      # stationarity solver sweeps
    beta: float = 1.0      # inverse temperature in the attention energy
    eta: float = 0.8       # descent step size
    lam: float = 10.0      # soft clamp weight (inert when hard_clamp)
    g_floor: float = 1.0   # g_m >= g_floor; makes the quadratic part convex
    g_init: float = 0.0    # g_raw init. The quadratic term is a smoothness
                           # PRIOR, but it also sets how much state survives a
                           # sweep: retention is 1 - eta*g/(g+1) per sweep, so
                           # g=1.69 (g_floor=1, g_raw=0) retains 0.50 and ten
                           # sweeps collapse to about one effective layer.
                           # Any g_floor > 0 suffices for coercivity because
                           # rmsnorm already bounds s_rest, so this is free.
    band_pad: float = 1.0  # multiplier past mode `modes`. g_floor == neutral.
                           # 1e3 reproduces the capacity ceiling of bug (2).
    tied: bool = False     # tie W_q = W_k. True reproduces bug (1).
    flat: bool = True      # non-trivial flat connection (RoPE) as the base
    phi_dev: float = 0.1   # how far the CONTENT-dependent phase may bend the
                           # connection away from the flat base, per edge, in
                           # units of the base quantum 2*pi/n.
                           #
                           # Does NOT affect accuracy. Measured at 8k steps:
                           #   0.0 -> 2.472   0.1 -> 2.477
                           #   0.3 -> 2.468   1.0 -> 2.462
                           # all within seed noise, and 0.0 removes the learned
                           # connection entirely. So the content-dependent gauge
                           # -- the "ambiguity is holonomy" mechanism -- earns
                           # nothing here; the fixed flat base does the work.
                           #
                           # It was introduced on the theory that a saturating
                           # phase was causing the late-training collapse, since
                           # the closure residual blew up 0.056 -> 0.364 exactly
                           # as gap NLL went 2.618 -> 3.361. That was a symptom:
                           # the collapse survived phi_dev=0.1 (merely moving
                           # from step 7000 to 9000) and was actually caused by
                           # weight decay on the tied readout -- see decay_mask.
                           #
                           # Kept, at 0.1, for the one thing it does do: bound
                           # the deviation so the holonomy stays a deviation
                           # from a flat section rather than a competitor to it
                           # (closure 0.015 vs 0.38), which keeps the winding
                           # numbers interpretable at no measured cost.
    hard_clamp: bool = True  # project clamped positions exactly, every sweep
    tie_head: bool = True  # read out through emb^T. Under hard clamping a
                           # clamped position holds exactly emb[t], so a tied
                           # head decodes the given tokens for free and the
                           # solved arc is scored in the same basis it is
                           # written in. Untied, the head only ever sees SOLVED
                           # states during training and never learns to decode
                           # an embedding.
    head_scale: float = 1.0  # readout temperature at init; see init(). Small
                             # values collapse the model onto the unigram.
    emb_std: float = 1.0   # 1.0 conditions the rmsnorm Jacobian far better
    norm_eps: float = 1e-6
    w_holo: float = 0.2    # loop-closure penalty weight
    vocab: int = 65

    def __post_init__(self):
        assert self.d % 2 == 0, "d must be even (2-planes)"
        assert (self.d // self.heads) % 2 == 0, "d/heads must be even"
        assert self.modes <= self.n // 2, "kernel band exceeds Nyquist"


TEXT = Cfg()
BRIDGE = Cfg(d=96, heads=2, k_steps=8, vocab=26)

# The flat connection's winding numbers must be integers for the loop to close
# exactly. Spread them geometrically over [1, n/2], which is RoPE quantised to
# the ring: plane j turns m_j times per circuit, so the holonomy of the base
# connection is exactly 2*pi*m_j and the closure residual is exactly zero.


@functools.lru_cache(maxsize=None)
def windings(cfg: Cfg) -> np.ndarray:
    planes = cfg.d // 2
    if not cfg.flat:
        return np.zeros(planes, dtype=np.float32)
    j = np.arange(planes)
    top = cfg.n // 2
    m = np.round(top ** (1.0 - j / max(planes - 1, 1)))
    return m.astype(np.float32)


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
            open("input.txt", "w").write("".join(chr(97 + t) for t in toks))
    text = open("input.txt").read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.int32)
    split = int(0.9 * len(ids))
    return ids[:split], ids[split:], len(chars), chars


def sample_loops(data, key, batch, n):
    starts = jax.random.randint(key, (batch,), 0, len(data) - n)
    return data[starts[:, None] + jnp.arange(n)[None, :]]


def arc_clamp(key, batch, n):
    """One contiguous given arc. The Arrival regime: long free stretch."""
    k1, k2 = jax.random.split(key)
    start = jax.random.randint(k1, (batch, 1), 0, n)
    length = jax.random.randint(k2, (batch, 1), n // 4, 3 * n // 4)
    offset = (jnp.arange(n)[None, :] - start) % n
    return (offset < length).astype(jnp.float32)


def gap_clamp(key, batch, n, n_gaps=3, max_len=6):
    """Clamp everything except a few short gaps -- realistic infilling.

    The endpoint task (clamp 12 of 48, solve the rest) is near
    information-theoretically impossible on natural text: predicting 36 chars of
    Shakespeare from 6 either side has enormous conditional entropy, and
    emitting the unigram mode at every free position is close to CE-optimal. So
    a low score there measures the metric, not the model. Short gaps have
    genuinely low conditional entropy and are what infilling actually means.
    """
    k1, k2 = jax.random.split(key)
    starts = jax.random.randint(k1, (batch, n_gaps), 0, n)
    lens = jax.random.randint(k2, (batch, n_gaps), 1, max_len + 1)
    pos = jnp.arange(n)[None, :, None]
    inside = ((pos - starts[:, None, :]) % n) < lens[:, None, :]
    return 1.0 - jnp.any(inside, axis=-1).astype(jnp.float32)


def mix_clamp(key, batch, n):
    """Half short gaps (learnable signal), half long arcs (the actual claim)."""
    ka, kg, kc = jax.random.split(key, 3)
    coin = jax.random.bernoulli(kc, 0.5, (batch, 1))
    return jnp.where(coin, gap_clamp(kg, batch, n), arc_clamp(ka, batch, n))


def fixed_gaps(batch, n, starts=(8, 22, 36), length=4):
    """Deterministic eval mask, identical for every model and every seed.

    Gaps are placed away from index 0 so that every free position has fully
    observed LEFT context on the linear reading of the slice. That is what makes
    the causal baseline scorable on exactly these positions.
    """
    pos = np.arange(n)[:, None]
    inside = ((pos - np.array(starts)[None, :]) % n) < length
    m = 1.0 - inside.any(-1).astype(np.float32)
    return jnp.broadcast_to(jnp.asarray(m), (batch, n))


def gap_starts_mask(batch, n, starts=(8, 22, 36)):
    """The subset of free positions whose entire left context is given."""
    m = np.zeros(n, dtype=np.float32)
    m[list(starts)] = 1.0
    return jnp.broadcast_to(jnp.asarray(m), (batch, n))


def boundary_masks(batch, n, start=24, length=4):
    """The Arrival claim on real text, isolated to one bit of information.

    Two masks and one scored position. Both give positions [0, start) and
    neither gives [start, start+length); they differ ONLY in whether the text to
    the RIGHT of the gap is available:

        two-sided : clamp everything except [start, start+length)
        left-only : clamp [0, start), leave all of [start, n) free

    Score position `start` alone. Its left context is identical under both, so
    the difference is exactly what the right-hand boundary condition is worth --
    the text analogue of bridge's both-ends vs prefix-only split, measured
    inside a single model rather than across two architectures.

    This is the comparison the causal baseline cannot participate in, which is
    why it is kept separate from the causal numbers: a causal model scored on
    the same position is *by construction* the left-only case, so pitting it
    against the two-sided case confounds the boundary condition with the
    objective and the parameterisation at once.
    """
    two = np.ones(n, dtype=np.float32)
    two[start:start + length] = 0.0
    left = np.ones(n, dtype=np.float32)
    left[start:] = 0.0
    w = np.zeros(n, dtype=np.float32)
    w[start] = 1.0
    bc = lambda a: jnp.broadcast_to(jnp.asarray(a), (batch, n))
    return bc(two), bc(left), bc(w)


def bridge_clamp(batch, n, ends=6):
    """Endpoints only: the Arrival test. Know the start and end, solve between."""
    pos = jnp.arange(n)
    m = (pos < ends) | (pos >= n - ends)
    return jnp.broadcast_to(m.astype(jnp.float32), (batch, n))


def unigram_nll(data, vocab):
    """Reference point: NLL of the best context-free predictor."""
    counts = np.bincount(np.asarray(data), minlength=vocab).astype(np.float64)
    pr = counts / counts.sum()
    return float(-(pr * np.log(pr + 1e-12)).sum())


def bigram_nll(train, val, vocab):
    """Reference point: add-1 smoothed bigram table, next-token NLL on val."""
    tr, va = np.asarray(train), np.asarray(val)
    counts = np.ones((vocab, vocab), dtype=np.float64)
    np.add.at(counts, (tr[:-1], tr[1:]), 1.0)
    logp = np.log(counts / counts.sum(1, keepdims=True))
    return float(-logp[va[:-1], va[1:]].mean())


# ----------------------------------------------------------------------------
# the bridge unit test
#
# x[k] = a for k < n/2, else b.  Position 0 holds a, position n-1 holds b, and
# those two are adjacent on the loop -- so the free arc is one contiguous
# stretch with a boundary condition pinned at each end.
#
# Deliberately NOT harmonic: the smoothness prior alone interpolates a ramp
# between the two clamped values, not a step, so the circulant term cannot
# shortcut this. Information has to reach the interior through the content
# pathway. And nothing in the prefix reveals b, so a causal model is
# structurally incapable of the second half no matter how well trained.

def bridge_batch(key, batch, cfg):
    ka, kb = jax.random.split(key)
    a = jax.random.randint(ka, (batch, 1), 0, cfg.vocab)
    b = jax.random.randint(kb, (batch, 1), 0, cfg.vocab)
    first = jnp.arange(cfg.n)[None, :] < cfg.n // 2
    return jnp.where(first, a, b).astype(jnp.int32)


def clamp_at(batch, n, positions):
    m = jnp.zeros(n).at[jnp.asarray(positions)].set(1.0)
    return jnp.broadcast_to(m, (batch, n))


# ----------------------------------------------------------------------------
# params

def init(key, cfg):
    k = jax.random.split(key, 8)
    s = 1.0 / np.sqrt(cfg.d)
    return {
        "emb": jax.random.normal(k[0], (cfg.vocab, cfg.d)) * cfg.emb_std,
        "head": jax.random.normal(k[1], (cfg.d, cfg.vocab)) * s,
        # A readout bias is not a detail: without it the logits are a pure
        # linear map of a unit-RMS vector, so the model cannot emit a constant
        # distribution and cannot even represent the unigram. Measured: train
        # NLL sat at 3.28 against a unigram entropy of 2.90, i.e. the model was
        # losing to a constant. This is the floor everything else builds on.
        "bhead": jnp.zeros((cfg.vocab,)),
        # Readout temperature, learned. Left at 1.0 despite the tied readout
        # starting with logit std ~ sqrt(d)*emb_std ~ 12.6, i.e. a near-one-hot
        # softmax. "Fixing" that by initialising hscale to 1/sqrt(d) collapses
        # the model onto the unigram (measured: 3.336 vs 2.584, identical to 3
        # decimals across 3 seeds), and the mechanism is a chicken-and-egg:
        # small hscale makes the logits essentially bhead alone, the loss then
        # supplies no gradient to X, and without an informative X there is no
        # gradient to grow hscale back. A peaked initial softmax costs a few
        # steps; a flat one costs the whole run.
        "hscale": jnp.array(cfg.head_scale),
        # spectral multiplier of the quadratic (geometric) term, real & banded.
        # real  ==  even kernel  ==  reflection-symmetric circulant.
        "g_raw": jnp.full((cfg.modes + 1, cfg.d), cfg.g_init),
        "wqk": jax.random.normal(k[2], (cfg.d, cfg.d)) * s,
        "wk": jax.random.normal(k[3], (cfg.d, cfg.d)) * s,  # unused iff tied
        "wphi": jax.random.normal(k[4], (cfg.d, cfg.d // 2)) * s * 0.1,
        "wh": jax.random.normal(k[5], (cfg.d, 4 * cfg.d)) * s,  # Hopfield "FFN"
        "bh": jnp.zeros((4 * cfg.d,)),
        "alpha": jnp.array(0.5),
        "prior": jax.random.normal(k[6], (cfg.d,)),  # init for free positions
    }


def spectral_multiplier(p, cfg):
    """g_m >= g_floor, learned for m <= modes, neutral past the band limit.

    Positivity makes the quadratic part of the action strictly convex, which
    bounds S below and hands us an exact diagonal preconditioner in Fourier.
    The band limit is on the KERNEL: past `modes` the multiplier is the constant
    `band_pad`, i.e. a uniform ridge that does not attenuate any mode relative
    to any other. Setting band_pad >> g_floor instead makes the state itself
    band-limited, which is bug (2).
    """
    g = cfg.g_floor + jax.nn.softplus(p["g_raw"])
    pad = cfg.n // 2 + 1 - (cfg.modes + 1)
    return jnp.pad(g, ((0, pad), (0, 0)), constant_values=cfg.band_pad)


def circulant(x, g, n):
    """Symmetric circulant apply on the loop axis. O(n log n), exact."""
    return jnp.fft.irfft(jnp.fft.rfft(x, axis=1) * g[None], n=n, axis=1)


def rmsnorm(x, eps=1e-6):
    return x / jnp.sqrt(jnp.mean(x ** 2, -1, keepdims=True) + eps)


def gauge_rotate(q, phi_cum):
    """Rotate each 2-plane of q by its cumulative edge phase (RoPE, learned)."""
    c, s = jnp.cos(phi_cum), jnp.sin(phi_cum)
    a, b = q[..., 0::2], q[..., 1::2]
    return jnp.stack([a * c - b * s, a * s + b * c], -1).reshape(q.shape)


def edge_phases(p, xn, cfg):
    """Per-EDGE SO(2) phase per 2-plane: flat base + content-dependent part.

    The phase must be a property of the edge (i, i+1), not of the node i. The
    original code accumulated a node quantity with an inclusive cumsum, which is
    an off-by-one against the stated design and leaves the connection not
    covariant under reflection -- enough to mask the tying effect entirely (the
    residual sat at 1e-1 for every config, so d_reflection could not separate
    anything). Two consequences of doing it properly:

      * the content term is ANTISYMMETRIC in the edge's endpoints, tanh being
        odd, so reversing the loop reverses each transport exactly. The
        content-dependent connection is therefore reflection-COVARIANT by
        construction and cannot break the symmetry, whatever W_q and W_k are.
      * the flat base is attached to the RING, not to the content, so it does
        not reverse. It is a background field, and it is the only thing in the
        model that breaks reflection.

    So the arrow of time enters exactly once, as the flat connection -- and only
    an untied (hence non-PSD) channel map can read it. Direction needs both.
    """
    base = jnp.asarray(windings(cfg))
    dx = xn - jnp.roll(xn, -1, axis=1)   # edge (i, i+1); antisymmetric
    dev = cfg.phi_dev * jnp.tanh(dx @ p["wphi"])
    return (2 * np.pi / cfg.n) * (base + dev)


def cumulative_phase(phi):
    """Exclusive cumsum: the transport from the origin to node i is the product
    over edges strictly before i. Inclusive would put node i's own edge in."""
    return jnp.cumsum(phi, axis=1) - phi


# ----------------------------------------------------------------------------
# the action

def s_quad(p, x, cfg):
    """Convex half: (1/2)<X, A X>, A symmetric circulant PSD. Exact in Fourier."""
    return 0.5 * jnp.sum(x * circulant(x, spectral_multiplier(p, cfg), cfg.n))


def s_rest(p, x, cfg):
    """The non-quadratic half. Both terms enter negatively, and -logsumexp of a
    quadratic form supplies genuine negative curvature -- which is why this half
    is what makes the truncated descent not a fixed point."""
    xn = rmsnorm(x, cfg.norm_eps)
    cum = cumulative_phase(edge_phases(p, xn, cfg))

    q = gauge_rotate(xn @ p["wqk"], cum)
    k = q if cfg.tied else gauge_rotate(xn @ p["wk"], cum)
    b, n, d = q.shape
    dh = d // cfg.heads
    qh = q.reshape(b, n, cfg.heads, dh)
    kh = k.reshape(b, n, cfg.heads, dh)
    logits = jnp.einsum("bihe,bjhe->bhij", qh, kh) / np.sqrt(dh)
    s_att = -(1.0 / cfg.beta) * jnp.sum(
        jax.nn.logsumexp(cfg.beta * logits, axis=-1))

    # softplus stays linear at large |X|, so the quadratic always dominates
    s_hop = -p["alpha"] * jnp.sum(jax.nn.softplus(xn @ p["wh"] + p["bh"]))
    return s_att + s_hop


def action(p, x, c, clamp, cfg):
    """Full action S[X] with the clamped arc as a quadratic data term."""
    dat = 0.5 * cfg.lam * jnp.sum(clamp[..., None] * (x - c) ** 2)
    return s_quad(p, x, cfg) + dat + s_rest(p, x, cfg)


def solve(p, c, clamp, cfg, trace=False, x0=None):
    """Find X on the free arc, given X = C on the clamped arc.

    Preconditioned gradient descent on the action, truncated at k_steps, with
    the clamped positions projected exactly each sweep. The preconditioner is
    the exact inverse of the circulant part in Fourier plus the diagonal clamp
    part in real space, which is the Newton preconditioner for the quadratic
    half of S. Truncated, not converged -- see SOLVER STATUS in the header.
    """
    g = spectral_multiplier(p, cfg)
    keep = clamp[..., None]
    free = 1.0 - keep
    inv_geo = 1.0 / (g + 1.0)           # circulant part, exact in Fourier
    jac = 1.0 + cfg.lam * keep          # clamp part, diagonal in real space
    grad_x = jax.grad(action, argnums=1)

    def sweep(x, _):
        gr = circulant(grad_x(p, x, c, clamp, cfg), inv_geo, cfg.n) / jac
        x = x - cfg.eta * gr
        if cfg.hard_clamp:
            x = keep * c + free * x
        return x, (jnp.sqrt(jnp.sum((free * gr) ** 2)) if trace else None)

    if x0 is None:
        x0 = p["prior"]
    x0 = keep * c + free * x0
    x, res = jax.lax.scan(sweep, x0, None, length=cfg.k_steps)
    return (x, res) if trace else x


def holonomy(p, x, cfg):
    """Total phase around the loop, relative to the flat section.

    The base connection contributes exactly 2*pi*m_j and so is invisible here by
    construction; what is measured is the deviation from flatness, which is the
    quantity that distinguishes readings.
    """
    xn = rmsnorm(x, cfg.norm_eps)
    dx = xn - jnp.roll(xn, -1, axis=1)
    dev = (2 * np.pi / cfg.n) * cfg.phi_dev * jnp.sum(
        jnp.tanh(dx @ p["wphi"]), axis=1)
    resid = jnp.arctan2(jnp.sin(dev), jnp.cos(dev))  # wrap to [-pi, pi]
    winding = jnp.round(dev / (2 * np.pi))
    return jnp.mean(jnp.abs(resid)), winding


def logits_of(p, x, cfg):
    w = (p["hscale"] * p["emb"].T) if cfg.tie_head else p["head"]
    return rmsnorm(x) @ w + p["bhead"]


def loss_fn(p, tokens, clamp, cfg):
    c = p["emb"][tokens]
    x = solve(p, c, clamp, cfg)
    ce = optax.softmax_cross_entropy_with_integer_labels(logits_of(p, x, cfg), tokens)
    free = 1.0 - clamp
    nll = jnp.sum(ce * free) / (jnp.sum(free) + 1e-6)
    resid, _ = holonomy(p, x, cfg)
    return nll + cfg.w_holo * resid, (nll, resid)


# ----------------------------------------------------------------------------
# train / eval

@functools.partial(jax.jit, static_argnames=("opt_update", "cfg"))
def train_step(p, opt_state, tokens, clamp, opt_update, cfg):
    (_, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        p, tokens, clamp, cfg)
    updates, opt_state = opt_update(grads, opt_state, p)
    return optax.apply_updates(p, updates), opt_state, aux


@functools.partial(jax.jit, static_argnames=("cfg",))
def score(p, tokens, clamp, cfg, weight=None):
    """Mean NLL and accuracy over the free positions (or a weighted subset)."""
    x = solve(p, p["emb"][tokens], clamp, cfg)
    lg = logits_of(p, x, cfg)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, tokens)
    w = (1.0 - clamp) if weight is None else weight
    hit = (jnp.argmax(lg, -1) == tokens).astype(jnp.float32)
    tot = jnp.sum(w) + 1e-6
    return jnp.sum(ce * w) / tot, jnp.sum(hit * w) / tot


@functools.partial(jax.jit, static_argnames=("cfg",))
def eval_text(p, tokens, mix, gaps, starts, cfg):
    mix_nll, _ = score(p, tokens, mix, cfg)
    gap_nll, gap_acc = score(p, tokens, gaps, cfg)
    st_nll, st_acc = score(p, tokens, gaps, cfg, weight=starts)
    br = bridge_clamp(tokens.shape[0], cfg.n)
    br_nll, br_acc = score(p, tokens, br, cfg)
    # what the right-hand boundary condition is worth, same model, same
    # position, same left context -- see boundary_masks
    two, left, w = boundary_masks(tokens.shape[0], cfg.n)
    bc_two, _ = score(p, tokens, two, cfg, weight=w)
    bc_left, _ = score(p, tokens, left, cfg, weight=w)
    x = solve(p, p["emb"][tokens], gaps, cfg)
    resid, winding = holonomy(p, x, cfg)
    # RMS of the solved state on the FREE arc, and the attention weight norm.
    # The action is only coercive while the quadratic term dominates the
    # content term (old/semagram.py:144 gives the condition as roughly
    # g_floor/2 > c*||W||^2*sqrt(n)/sqrt(d)); if ||W|| grows past that, the
    # truncated solve stops being a descent and the state can run away. These
    # two numbers are what distinguish "the model got worse" from "the solver
    # came apart", which is the difference between a tuning problem and a
    # structural one.
    free = 1.0 - gaps
    xr = jnp.sqrt(jnp.sum((x * free[..., None]) ** 2) /
                  (jnp.sum(free) * cfg.d + 1e-6))
    wn = jnp.linalg.norm(p["wqk"]) * (1.0 if cfg.tied
                                      else jnp.linalg.norm(p["wk"])) ** 0.5
    return dict(mix_nll=mix_nll, gap_nll=gap_nll, gap_acc=gap_acc,
                start_nll=st_nll, start_acc=st_acc, bridge_nll=br_nll,
                bridge_acc=br_acc, bc_two=bc_two, bc_left=bc_left,
                closure=resid, winding=jnp.mean(jnp.abs(winding)),
                xrms=xr, wnorm=wn)


def n_params(p):
    return sum(x.size for x in jax.tree.leaves(p))


def schedule(lr, steps):
    return optax.warmup_cosine_decay_schedule(
        0.0, lr, max(steps // 20, 1), steps, end_value=lr * 0.1)


GRAD_CLIP = 0.0  # global-norm clip; <= 0 disables. Off by default, measured.


def make_opt(lr, steps, clip=None):
    """AdamW, by default with NO gradient clipping.

    Clipping is the reflex fix for what this model does -- train to ~2.6 and
    then fall off a cliff (gap NLL 2.618 at step 6000 -> 3.361 at step 7000).
    It is the wrong fix twice over.

    It does not work: at global norm 1.0 the same run is *worse everywhere*,
    4.415 vs 2.639 at step 4000, and collapses sooner rather than later. The
    likely reason is that clip_by_global_norm rescales the entire parameter tree
    by one factor, and here that factor swings by orders of magnitude from step
    to step (initial loss is ~28 nats because the tied readout starts peaked).
    Adam is invariant to a CONSTANT rescale, not to a violently varying one --
    m/sqrt(v) stops estimating anything.

    And it is not needed, because the collapse has a cause rather than being
    generic bad luck: the content-dependent gauge phase saturates and swamps the
    flat connection. Bounding that at source with cfg.phi_dev fixes it without
    touching the optimiser. See the note on phi_dev.

    Left here, off, because "we tried clipping" is worth recording.
    """
    clip = GRAD_CLIP if clip is None else clip
    adam = optax.adamw(schedule(lr, steps), weight_decay=0.01, mask=decay_mask)
    return optax.chain(optax.clip_by_global_norm(clip), adam) if clip > 0 else adam


NO_DECAY = ("emb", "pos", "mask", "bhead", "hscale", "prior", "alpha", "g_raw")


def decay_mask(params):
    """Decay the weight matrices only -- not embeddings, biases, or scales.

    Standard practice, and here it is the difference between training and
    collapsing. Decaying `emb` and `hscale` is not neutral under a TIED readout,
    because both multiply the logits: at lr 3e-3, wd 0.01 over 20k steps each
    shrinks by exp(-lr*wd*steps) ~ 0.55, so the logit scale drifts to ~0.3 of
    where it started. That walks the model into exactly the flat-logit unigram
    basin that head_scale=0.079 falls into from the start -- and it explains why
    the collapse is late (step ~9000), why higher lr reaches it sooner, and why
    every run short enough to finish first looked perfectly healthy.
    """
    def keep(path, leaf):
        name = getattr(path[-1], "key", None)
        return leaf.ndim >= 2 and name not in NO_DECAY
    return jax.tree_util.tree_map_with_path(keep, params)


def train_semagram(key, tr, cfg, steps, batch, lr, log=None, eval_fn=None,
                   every=0):
    p = init(key, cfg)
    opt = make_opt(lr, steps)
    opt_state = opt.init(p)
    for it in range(1, steps + 1):
        key, k1, k2 = jax.random.split(key, 3)
        tokens = sample_loops(tr, k1, batch, cfg.n)
        clamp = mix_clamp(k2, batch, cfg.n)
        p, opt_state, (nll, resid) = train_step(
            p, opt_state, tokens, clamp, opt.update, cfg)
        if every and (it % every == 0 or it == 1) and log:
            log(it, float(nll), float(resid), p, eval_fn)
    return p


# ----------------------------------------------------------------------------
# Phase 0 diagnostics.  Each one targets exactly one structural claim, and each
# can refute it.

def d_spectrum(p, tokens, cfg, key):
    """Per-mode transfer function of the solver ON THE FREE ARC.

    The question is what bandwidth a free position can actually carry, so the
    probe has to live on free coordinates: leave the whole loop free, perturb
    the initial state by white noise, and measure how much of each Fourier mode
    of that perturbation survives K sweeps. (An earlier version clamped every
    position instead, which is vacuous -- the clamp Jacobian 1 + lam damps the
    recursion uniformly and the free arc is never probed. It reported a factor
    of 3 where the truth is five orders of magnitude.)

    Prediction: band_pad=1e3 annihilates every mode past M, since those modes
    are multiplied by 1 - eta*g/(g+1) ~ 1 - eta each sweep. A neutral pad leaves
    the transfer flat. Gains are reported relative to the largest, because
    truncated descent contracts everything somewhat and only the SHAPE matters.
    """
    c = p["emb"][tokens]
    clamp = jnp.zeros(tokens.shape)
    base = jnp.broadcast_to(p["prior"], c.shape)
    delta = jax.random.normal(key, c.shape)
    xa = solve(p, c, clamp, cfg, x0=base)
    xb = solve(p, c, clamp, cfg, x0=base + delta)
    f = lambda z: (np.abs(np.asarray(jnp.fft.rfft(z, axis=1))) ** 2).sum((0, 2))
    gain = f(xb - xa) / (f(delta) + 1e-30)
    rel = gain / (gain.max() + 1e-30)
    return dict(gain=rel, bandwidth=int((rel > 0.05).sum()))


def d_reflection(p, tokens, cfg):
    """Is the solve equivariant to reflection of the loop?

    Reflection about index 0 is i -> (-i) mod n. If the solve is equivariant
    then reversing the loop reverses the solution exactly, so "th" and "ht" are
    the same object and no amount of training can separate them.

    Reported twice, because the two numbers answer different questions.

    `full` uses the model as-is and is NOT expected to vanish even when the
    model is reflection-blind. A connection with holonomy cannot be globally
    trivialised, and cumulative_phase trivialises it anyway by fixing a gauge at
    the origin -- so there is a branch cut there, of size exactly the holonomy.
    That residual is the loop refusing to be flattened, which is the phenomenon
    the architecture is about, not an asymmetry in the content pathway.

    `flat` zeroes the content phase, removing the holonomy and hence the branch
    cut, and isolates the structural question. There the result is sharp, and it
    is not the one this file originally predicted:

        reflection is broken iff (flat base), regardless of tying

    Every other term is equivariant -- even kernel, pointwise nonlinearities,
    and a content connection that reverses with the loop -- so with flat=False
    the residual is float-epsilon (6e-15 in float64) for tied and untied alike.
    Tying makes L exactly symmetric but that does NOT make the model
    reflection-blind: the sin(dth) * W^T J W part of the channel map is
    antisymmetric, nonzero, and odd in the phase difference. See note (1).
    """
    rev = lambda z: jnp.concatenate([z[:, :1], z[:, 1:][:, ::-1]], axis=1)
    clamp = fixed_gaps(tokens.shape[0], cfg.n)

    def residual(q):
        x = solve(q, q["emb"][tokens], clamp, cfg)
        xr = solve(q, q["emb"][rev(tokens)], rev(clamp), cfg)
        return float(jnp.linalg.norm(rev(x) - xr) /
                     (jnp.linalg.norm(x) + 1e-30))

    return residual(p), residual({**p, "wphi": jnp.zeros_like(p["wphi"])})


def d_clamp_echo(p, tokens, cfg):
    """Can the model reproduce a token it was GIVEN, and does that token's
    representation ever become contextual?

    Two numbers. Decode accuracy on the clamped arc is a floor check: if a given
    token does not survive the solve, nothing downstream can work.

    Drift, ||x - c|| / ||c|| on the clamped arc, is exactly 0 under hard_clamp
    by construction. That looks alarming: a position pinned to emb[t] for every
    sweep never contextualises, so free positions read frozen raw embeddings no
    matter how many sweeps run, and the model should collapse to one layer of
    attention over static features -- about a bigram, which is roughly what it
    scores. Soft clamping is the obvious fix, since the data term re-injects
    lam*(x - c) every sweep and so keeps the observation while letting the state
    move (input injection, in the DEQ sense).

    Measured, and it is NOT the explanation. At 4000 steps, 3 seeds:

        hard clamp          gap 2.560 +/- 0.020
        soft, lam = 0.5     gap 2.580 +/- 0.034
        soft, lam = 2.0     gap 3.336   (collapsed to unigram)
        soft, lam = 10.0    gap 3.338   (collapsed to unigram)

    Indistinguishable where it works at all, and catastrophic above lam ~ 1:
    with a large restoring force the nonlocal preconditioner smears lam*(x - c)
    from the clamped arc across the free one and swamps it, which hard clamping
    prevents by construction. Enforcing the Dirichlet condition exactly is the
    better engineering choice as well as the one the BVP framing asks for.

    Worth recording that soft clamping DID rescue a broken configuration
    (head_scale=0.079, where hard clamp sits at unigram and lam=0.5 reaches
    2.443). A knob that helps only when something else is misconfigured is
    measuring the misconfiguration.
    """
    clamp = fixed_gaps(tokens.shape[0], cfg.n)
    c = p["emb"][tokens]
    x = solve(p, c, clamp, cfg)
    keep = clamp[..., None]
    pred = jnp.argmax(logits_of(p, x, cfg), -1)
    acc = jnp.sum((pred == tokens) * clamp) / jnp.sum(clamp)
    drift = jnp.linalg.norm(keep * (x - c)) / (jnp.linalg.norm(keep * c) + 1e-30)
    return float(acc), float(drift)


def d_residual(p, tokens, cfg):
    """Free-arc stationarity residual per sweep, and realised rank of X.

    "descent" is expected NOT to converge. This measures how far from a fixed
    point it is instead of assuming either way.
    """
    clamp = fixed_gaps(tokens.shape[0], cfg.n)
    x, res = solve(p, p["emb"][tokens], clamp, cfg, trace=True)
    sv = np.linalg.svd(np.asarray(x).reshape(-1, cfg.d), compute_uv=False)
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    return np.asarray(res), int(np.searchsorted(energy, 0.95) + 1)


def run_diagnostics(p, tokens, cfg, tag, key=None):
    print(f"\n--- diagnostics [{tag}] "
          f"tied={cfg.tied} band_pad={cfg.band_pad:g} flat={cfg.flat} ---")

    spec = d_spectrum(p, tokens, cfg,
                      jax.random.PRNGKey(99) if key is None else key)
    gain = spec["gain"]
    lo = gain[:cfg.modes + 1].mean()
    hi = gain[cfg.modes + 1:].mean() if len(gain) > cfg.modes + 1 else float("nan")
    print(f"  d_spectrum   rel gain m<=M {lo:9.3e} | m>M {hi:9.3e} "
          f"| ratio {lo / (hi + 1e-30):8.2e} | usable modes {spec['bandwidth']}"
          f"/{cfg.n // 2 + 1}")

    r_full, r_flat = d_reflection(p, tokens, cfg)
    # float32 through k sweeps floors the residual near 1e-3; in float64 the
    # equivariant configs sit at 6e-15, so 1e-2 separates cleanly here.
    verdict = "REFLECTION-BLIND" if r_flat < 1e-2 else "directed"
    print(f"  d_reflection full {r_full:9.3e} (holonomy branch cut) | "
          f"flat-only {r_flat:9.3e}  -> {verdict}")

    acc, drift = d_clamp_echo(p, tokens, cfg)
    print(f"  d_clamp_echo given-token decode acc {acc:.3f} | drift {drift:.3e}")

    res, rank = d_residual(p, tokens, cfg)
    ratio = res[-1] / (res[0] + 1e-30)
    print(f"  d_residual   |grad S| {res[0]:.3e} -> {res[-1]:.3e} "
          f"(x{ratio:.2e} over {cfg.k_steps} sweeps) | 95% rank {rank}/{cfg.d}")
    return dict(gain_ratio=float(lo / (hi + 1e-30)), bandwidth=spec["bandwidth"],
                reflection=r_flat, reflection_full=r_full, echo_acc=acc,
                rank=rank)


# ----------------------------------------------------------------------------
# Phase 2 baselines: pre-LN transformers, parameter-matched, same objective.

def tf_init(key, vocab, n, layers, width, heads):
    ks = jax.random.split(key, 3 + 4 * layers)
    s = 1.0 / np.sqrt(width)
    p = {"emb": jax.random.normal(ks[0], (vocab, width)) * 0.02,
         "pos": jax.random.normal(ks[1], (n, width)) * 0.02,
         "mask": jax.random.normal(ks[2], (width,)) * 0.02,
         "head": jax.random.normal(ks[3], (width, vocab)) * s,
         "blocks": []}
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


def layernorm(x):
    mu = jnp.mean(x, -1, keepdims=True)
    return (x - mu) / jnp.sqrt(jnp.var(x, -1, keepdims=True) + 1e-5)


def tf_forward(p, tokens, clamp, heads, causal):
    b, n = tokens.shape
    pos = p["pos"][:n]  # causal is fed tokens[:, :-1], so pos must be sliced
    if causal:
        h = p["emb"][tokens] + pos
    else:
        e = p["emb"][tokens]
        h = jnp.where(clamp[..., None] > 0, e, p["mask"]) + pos
    for blk in p["blocks"]:
        z = layernorm(h)
        qkv = z @ blk["qkv"]
        d = qkv.shape[-1] // 3
        q, k, v = jnp.split(qkv, 3, -1)
        dh = d // heads
        rs = lambda t: t.reshape(b, n, heads, dh).transpose(0, 2, 1, 3)
        att = jnp.einsum("bhid,bhjd->bhij", rs(q), rs(k)) / np.sqrt(dh)
        if causal:
            att = jnp.where(np.tril(np.ones((n, n)))[None, None] > 0, att, -1e9)
        o = jnp.einsum("bhij,bhjd->bhid", jax.nn.softmax(att, -1), rs(v))
        h = h + o.transpose(0, 2, 1, 3).reshape(b, n, d) @ blk["proj"]
        z = layernorm(h)
        h = h + jax.nn.gelu(z @ blk["fc1"] + blk["b1"]) @ blk["fc2"] + blk["b2"]
    return layernorm(h) @ p["head"]


@functools.partial(jax.jit, static_argnames=("heads", "causal"))
def tf_loss(p, tokens, clamp, heads, causal):
    if causal:
        lg = tf_forward(p, tokens[:, :-1], None, heads, True)
        ce = optax.softmax_cross_entropy_with_integer_labels(lg, tokens[:, 1:])
        return jnp.mean(ce)
    lg = tf_forward(p, tokens, clamp, heads, False)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, tokens)
    free = 1.0 - clamp
    return jnp.sum(ce * free) / (jnp.sum(free) + 1e-6)


@functools.partial(jax.jit, static_argnames=("heads", "causal", "opt_update"))
def tf_step(p, opt_state, tokens, clamp, heads, causal, opt_update):
    l, grads = jax.value_and_grad(tf_loss)(p, tokens, clamp, heads, causal)
    updates, opt_state = opt_update(grads, opt_state, p)
    return optax.apply_updates(p, updates), opt_state, l


@functools.partial(jax.jit, static_argnames=("heads", "causal"))
def tf_score(p, tokens, clamp, weight, heads, causal):
    """Score on an arbitrary weighted position set.

    Causal is teacher-forced and shifted, so position t is predicted from the
    true tokens strictly to its left. Restricting `weight` to gap-START
    positions makes that the honest left-context-only infill number; scoring it
    on every gap position instead leaks the gap's own prefix and is a generous
    upper bound on what a causal model could actually do.
    """
    if causal:
        lg = tf_forward(p, tokens[:, :-1], None, heads, True)
        ce = optax.softmax_cross_entropy_with_integer_labels(lg, tokens[:, 1:])
        hit = (jnp.argmax(lg, -1) == tokens[:, 1:]).astype(jnp.float32)
        w = weight[:, 1:]
    else:
        lg = tf_forward(p, tokens, clamp, heads, False)
        ce = optax.softmax_cross_entropy_with_integer_labels(lg, tokens)
        hit = (jnp.argmax(lg, -1) == tokens).astype(jnp.float32)
        w = weight
    tot = jnp.sum(w) + 1e-6
    return jnp.sum(ce * w) / tot, jnp.sum(hit * w) / tot


def match_params(target, vocab, n, heads, layer_opts=(2, 3, 4)):
    """Smallest (layers, width) whose param count is closest to `target`."""
    best = None
    for layers in layer_opts:
        for width in range(32, 512, heads * 2):
            if (width // heads) % 2:
                continue
            cnt = vocab * width + n * width + width + width * vocab + \
                layers * (3 * width * width + width * width +
                          4 * width * width + 4 * width * width +
                          4 * width + width)
            err = abs(cnt - target) / target
            if best is None or err < best[0]:
                best = (err, layers, width, cnt)
    return best[1], best[2], best[3]


def train_transformer(key, tr, vocab, cfg, steps, batch, lr, layers, width,
                      heads, causal, label):
    p = tf_init(key, vocab, cfg.n, layers, width, heads)
    opt = make_opt(lr, steps)
    opt_state = opt.init(p)
    t0 = time.time()
    for it in range(1, steps + 1):
        key, k1, k2 = jax.random.split(key, 3)
        tokens = sample_loops(tr, k1, batch, cfg.n)
        clamp = mix_clamp(k2, batch, cfg.n)
        p, opt_state, l = tf_step(p, opt_state, tokens, clamp, heads, causal,
                                  opt.update)
        if it % max(steps // 4, 1) == 0:
            print(f"    {label} {it:6d}/{steps} | loss {float(l):.3f} "
                  f"| {time.time() - t0:5.0f}s")
    return p


# ----------------------------------------------------------------------------
# runners

def main_bridge(args):
    cfg = dataclasses.replace(
        BRIDGE, tied=args.tied, band_pad=args.band_pad, flat=args.flat,
        g_floor=args.g_floor, g_init=args.g_init, eta=args.eta,
        tie_head=args.tie_head)
    key = jax.random.PRNGKey(args.seed)
    key, sk = jax.random.split(key)
    p = init(sk, cfg)
    print(f"bridge unit test | loop n={cfg.n} d={cfg.d} modes={cfg.modes} "
          f"K={cfg.k_steps} | vocab {cfg.vocab} | params {n_params(p) / 1e6:.3f}M")
    print(f"tied={cfg.tied} band_pad={cfg.band_pad:g} flat={cfg.flat}")
    print(f"chance = {1 / cfg.vocab:.3f} | prefix-only ceiling ~= 0.52\n")

    steps = args.steps or 2000
    opt = make_opt(args.lr, steps)
    opt_state = opt.init(p)
    cl_both = clamp_at(args.batch, cfg.n, [0, cfg.n - 1])

    for it in range(1, steps + 1):
        key, k1 = jax.random.split(key)
        tokens = bridge_batch(k1, args.batch, cfg)
        p, opt_state, (nll, resid) = train_step(
            p, opt_state, tokens, cl_both, opt.update, cfg)
        if it % max(steps // 8, 1) == 0 or it == 1:
            key, k2 = jax.random.split(key)
            vt = bridge_batch(k2, 128, cfg)
            _, ab = score(p, vt, clamp_at(128, cfg.n, [0, cfg.n - 1]), cfg)
            _, ap = score(p, vt, clamp_at(128, cfg.n, [0]), cfg)
            print(f"{it:5d} | nll {nll:.3f} | both-ends {ab:.3f} "
                  f"| prefix-only {ap:.3f} | gap {ab - ap:+.3f} "
                  f"| closure {resid:.3f}")
    return p, float(ab), float(ap)


def main_text(args):
    cfg = cfg_from(args, TEXT)
    tr, va, vocab, chars = load_data()
    cfg = dataclasses.replace(cfg, vocab=vocab)
    tr, va = jnp.asarray(tr), jnp.asarray(va)
    steps = args.steps or 20000

    uni, big = unigram_nll(tr, vocab), bigram_nll(tr, va, vocab)
    print(f"vocab {vocab} | train {len(tr):,} | val {len(va):,} | "
          f"loop n={cfg.n} d={cfg.d} heads={cfg.heads} modes={cfg.modes} "
          f"K={cfg.k_steps}")
    print(f"tied={cfg.tied} band_pad={cfg.band_pad:g} flat={cfg.flat} "
          f"hard_clamp={cfg.hard_clamp}")
    print(f"references: unigram NLL {uni:.3f} | bigram next-token NLL {big:.3f}")

    key = jax.random.PRNGKey(args.seed)
    key, sk, kv = jax.random.split(key, 3)
    p = init(sk, cfg)
    npar = n_params(p)
    print(f"semagram params {npar / 1e6:.3f}M\n")

    # one fixed val set, shared by every model so comparisons are exact
    vt = sample_loops(va, kv, 512, cfg.n)
    v_mix = mix_clamp(jax.random.PRNGKey(1234), 512, cfg.n)
    v_gap = fixed_gaps(512, cfg.n)
    v_start = gap_starts_mask(512, cfg.n)

    opt = make_opt(args.lr, steps)
    opt_state = opt.init(p)
    t0 = time.time()
    for it in range(1, steps + 1):
        key, k1, k2 = jax.random.split(key, 3)
        tokens = sample_loops(tr, k1, args.batch, cfg.n)
        clamp = mix_clamp(k2, args.batch, cfg.n)
        p, opt_state, (nll, resid) = train_step(
            p, opt_state, tokens, clamp, opt.update, cfg)
        if it % max(steps // 20, 1) == 0 or it == 1:
            m = eval_text(p, vt, v_mix, v_gap, v_start, cfg)
            print(f"{it:6d} | train {nll:.3f} | mix {m['mix_nll']:.3f} "
                  f"| gap {m['gap_nll']:.3f} (acc {m['gap_acc']:.3f}) "
                  f"| start {m['start_nll']:.3f} | closure {m['closure']:.3f} "
                  f"| xrms {m['xrms']:.2f} | |W| {m['wnorm']:.1f} "
                  f"| {time.time() - t0:5.0f}s")
    m = {k: float(v) for k, v in eval_text(p, vt, v_mix, v_gap, v_start,
                                           cfg).items()}

    results = dict(semagram=m, params=npar, unigram=uni, bigram=big)

    # The unigram is an ABSORBING state for this model: once the readout goes
    # flat the solved state carries no information, and there is then no
    # gradient pointing back out. Runs that fall in never recover, so a final
    # score sitting on the unigram means "this run collapsed", not "this
    # architecture scores 3.33". Say so rather than tabulating it as a result.
    if m["gap_nll"] > uni - 0.05:
        print(f"\n*** COLLAPSED: gap NLL {m['gap_nll']:.3f} is at the unigram "
              f"({uni:.3f}).\n*** This run fell into the degenerate basin; the "
              f"numbers below measure that,\n*** not the architecture. See the "
              f"stability note in the README. Re-run,\n*** or use the lower "
              f"learning rate that has not been observed to collapse.")

    if args.baselines:
        layers, width, cnt = match_params(npar, vocab, cfg.n, args.heads)
        print(f"\nbaselines: {layers} layers, width {width}, {args.heads} heads "
              f"| params {cnt / 1e6:.3f}M ({100 * (cnt - npar) / npar:+.1f}% vs "
              f"semagram)")
        key, kb, kc = jax.random.split(key, 3)

        pb = train_transformer(kb, tr, vocab, cfg, steps, args.batch, args.lr,
                               layers, width, args.heads, False, "bidir")
        b_gap = tf_score(pb, vt, v_gap, 1.0 - v_gap, args.heads, False)
        b_mix = tf_score(pb, vt, v_mix, 1.0 - v_mix, args.heads, False)
        b_start = tf_score(pb, vt, v_gap, v_start, args.heads, False)

        pc = train_transformer(kc, tr, vocab, cfg, steps, args.batch, args.lr,
                               layers, width, args.heads, True, "causal")
        c_next = tf_score(pc, vt, None, jnp.ones((512, cfg.n)), args.heads, True)
        c_start = tf_score(pc, vt, None, v_start, args.heads, True)
        c_all = tf_score(pc, vt, None, 1.0 - v_gap, args.heads, True)

        results["bidir"] = dict(gap_nll=float(b_gap[0]), gap_acc=float(b_gap[1]),
                                mix_nll=float(b_mix[0]),
                                start_nll=float(b_start[0]),
                                start_acc=float(b_start[1]), params=cnt)
        results["causal"] = dict(next_nll=float(c_next[0]),
                                 start_nll=float(c_start[0]),
                                 start_acc=float(c_start[1]),
                                 all_gap_nll=float(c_all[0]), params=cnt)
        report(results)

    # qualitative: clamp both endpoints, read out the solved interior
    key, k3 = jax.random.split(key)
    qt = sample_loops(va, k3, 4, cfg.n)
    bc = bridge_clamp(4, cfg.n)
    pred = jnp.argmax(logits_of(p, solve(p, p["emb"][qt], bc, cfg), cfg), -1)
    gp = fixed_gaps(4, cfg.n)
    gpred = jnp.argmax(logits_of(p, solve(p, p["emb"][qt], gp, cfg), cfg), -1)
    dec = lambda row: "".join(chars[i] for i in row).replace("\n", " ")
    e = 6
    print("\nendpoint-clamped reconstruction (| marks the free arc):")
    for i in range(4):
        print(f"  true {dec(qt[i][:e])}|{dec(qt[i][e:-e])}|{dec(qt[i][-e:])}")
        print(f"  pred {dec(qt[i][:e])}|{dec(pred[i][e:-e])}|{dec(qt[i][-e:])}")
    print("\ngap infill ([] marks the 3 four-char gaps):")
    for i in range(4):
        show = lambda row: "".join(
            (f"[{chars[t]}]" if g == 0 else chars[t]).replace("\n", " ")
            for t, g in zip(row, np.asarray(gp[i])))
        print(f"  true {show(qt[i])}")
        print(f"  pred {show(gpred[i])}")
    return results


def report(r):
    s, b, c = r["semagram"], r["bidir"], r["causal"]
    print("\n" + "=" * 74)
    print("SUCCESS CRITERIA".center(74))
    print("=" * 74)
    print(f"  references         unigram {r['unigram']:.3f} | "
          f"bigram(next-token) {r['bigram']:.3f}")
    print(f"  params             semagram {r['params']:,} | "
          f"transformers {b['params']:,}")
    print()
    print("  4-char gap infill, all 12 free positions, two-sided context")
    print(f"    semagram         NLL {s['gap_nll']:.3f}  acc {s['gap_acc']:.3f}")
    print(f"    bidir baseline   NLL {b['gap_nll']:.3f}  acc {b['gap_acc']:.3f}")
    d1 = s["gap_nll"] - b["gap_nll"]
    print(f"    [1] within 0.1 nats of bidir: {d1:+.3f}  "
          f"{'PASS' if d1 <= 0.1 else 'FAIL'}")
    print()
    print("  [2] what the RIGHT-hand boundary condition is worth. One model,")
    print("  one position, identical left context; only the right side differs.")
    d2 = s["bc_left"] - s["bc_two"]
    print(f"    semagram two-sided   NLL {s['bc_two']:.3f}")
    print(f"    semagram left-only   NLL {s['bc_left']:.3f}")
    print(f"    right context buys {d2:+.3f} nats  {'PASS' if d2 > 0 else 'FAIL'}")
    print()
    print("  gap-START positions -- left context fully observed. This is the")
    print("  only position set a causal model can honestly be scored on, but")
    print("  the contrast below is NOT the boundary-condition question: it")
    print("  varies objective and architecture at the same time, so read it")
    print("  against the bidir row, which holds the objective fixed.")
    print(f"    semagram (2-sided)   NLL {s['start_nll']:.3f}  "
          f"acc {s['start_acc']:.3f}")
    print(f"    bidir    (2-sided)   NLL {b['start_nll']:.3f}  "
          f"acc {b['start_acc']:.3f}")
    print(f"    causal   (left-only) NLL {c['start_nll']:.3f}  "
          f"acc {c['start_acc']:.3f}")
    print(f"    causal next-token over all positions {c['next_nll']:.3f}; "
          f"teacher-forced on all")
    print(f"    gap positions {c['all_gap_nll']:.3f} -- generous, it leaks the "
          f"gap's own prefix.")
    print()
    # Calibrated against the measured references rather than magic constants.
    # Gap infill has two-sided context and so is strictly easier than the
    # next-token bigram task; beating the bigram table is the weakest result
    # worth calling "uses context".
    ok3 = s["gap_nll"] < r["bigram"] and s["mix_nll"] < r["unigram"]
    print(f"  [3] uses context: gap {s['gap_nll']:.3f} < bigram "
          f"{r['bigram']:.3f} and mix {s['mix_nll']:.3f} < unigram "
          f"{r['unigram']:.3f}  {'PASS' if ok3 else 'FAIL'}")
    print("=" * 74)


def main_ablate(args):
    """Which fix was load-bearing? Resumes from JSON, skipping finished cells.

    Named configs rather than a full factorial. A 2x2x2x2 product spends most of
    its cells on combinations nobody reads, and at ~90s each that is the
    difference between a grid you run and one you don't. What is kept:

      * the full tied x flat 2x2, because that is the one place an INTERACTION
        is plausible -- flat is the only reflection-breaking term and tying
        changes the channel map that has to read it. exp_crossed.py in this repo
        exists to test exactly this shape of claim, so it gets the same
        treatment: a main effect would be a much weaker result than a crossing.
      * single flips of band_pad and hard_clamp off the best config, which is
        all a leave-one-out needs to attribute those two.
    """
    tr, va, vocab, chars = load_data()
    tr, va = jnp.asarray(tr), jnp.asarray(va)
    steps = args.steps or 4000
    store = json.load(open(ABLATION_JSON)) if os.path.exists(ABLATION_JSON) else {}

    configs = {
        "best":              dict(),
        "tied":              dict(tied=True),
        "no-flat":           dict(flat=False),
        "tied+no-flat":      dict(tied=True, flat=False),
        "band_pad=1e3":      dict(band_pad=1e3),
        # hard_clamp is already the default, so flipping it ON would be a no-op
        # duplicate of `best`. The ablation is the soft clamp, at the only lam
        # that trains at all -- above ~1 the nonlocal preconditioner smears the
        # restoring force across the free arc and the run collapses.
        "soft_clamp":        dict(hard_clamp=False, lam=0.5),
        "as-shipped":        dict(tied=True, flat=False, band_pad=1e3,
                                  hard_clamp=True, tie_head=False),
    }
    vt = sample_loops(va, jax.random.PRNGKey(7), 512, TEXT.n)
    v_mix = mix_clamp(jax.random.PRNGKey(1234), 512, TEXT.n)
    v_gap = fixed_gaps(512, TEXT.n)
    v_start = gap_starts_mask(512, TEXT.n)

    for name, over in configs.items():
        for seed in range(args.seeds):
            cell = f"{name}|seed={seed}"
            if cell in store:
                print(f"skip {cell}")
                continue
            cfg = cfg_from(args, TEXT, vocab=vocab, **over)
            t0 = time.time()
            p = train_semagram(jax.random.PRNGKey(seed), tr, cfg, steps,
                               args.batch, args.lr)
            m = {k: float(v) for k, v in
                 eval_text(p, vt, v_mix, v_gap, v_start, cfg).items()}
            m["seconds"] = time.time() - t0
            m["reflection"] = d_reflection(p, vt[:64], cfg)[1]
            store[cell] = m
            json.dump(store, open(ABLATION_JSON, "w"), indent=1)
            print(f"{cell:24s} gap {m['gap_nll']:.3f} acc {m['gap_acc']:.3f} "
                  f"mix {m['mix_nll']:.3f} refl {m['reflection']:.1e} "
                  f"({m['seconds']:.0f}s)")

    print("\n" + "=" * 74)
    print(f"{'config':18s} {'gap NLL':>15s} {'gap acc':>15s} {'refl':>9s}"
          f"  {'d(gap) vs best':>14s}")
    base = None
    for name in configs:
        vals = [store[f"{name}|seed={s}"] for s in range(args.seeds)
                if f"{name}|seed={s}" in store]
        if not vals:
            continue
        g = np.array([v["gap_nll"] for v in vals])
        a = np.array([v["gap_acc"] for v in vals])
        r = np.mean([v["reflection"] for v in vals])
        if base is None:
            base = g.mean()
        print(f"{name:18s} {g.mean():7.3f} +/-{g.std():5.3f} "
              f"{a.mean():7.3f} +/-{a.std():5.3f} {r:9.1e}  "
              f"{g.mean() - base:+14.3f}")
    print("=" * 74)
    print(f"seeds={args.seeds}, steps={steps}. The README records ~4pp of")
    print("fixed-seed run-to-run variance on this hardware, so treat any gap")
    print("smaller than the printed std as unresolved rather than as zero.")
    return store


def main_diagnose(args):
    tr, va, vocab, chars = load_data()
    tr, va = jnp.asarray(tr), jnp.asarray(va)
    base = cfg_from(args, TEXT, vocab=vocab)
    key = jax.random.PRNGKey(args.seed)
    key, kv = jax.random.split(key)
    vt = sample_loops(va, kv, 64, base.n)

    print("Each diagnostic targets one structural claim and can refute it.")
    print("Predictions: band_pad=1e3 annihilates every mode past M; reflection")
    print("is broken by flat=True and by nothing else, tied or untied, since")
    print("every other term is equivariant; descent does not converge.")
    print("NOTE float32 through 10 sweeps puts the equivariance floor near")
    print("1e-3, not 1e-7; the 6e-15 figure quoted above is float64.")

    for tied, bp, flat, tag in [
            (True, 1e3, False, "as-shipped (the failing config)"),
            (False, 1.0, False, "untied, no flat base"),
            (True, 1.0, True, "flat base, but tied"),
            (False, 1.0, True, "untied + flat")]:
        cfg = dataclasses.replace(base, tied=tied, band_pad=bp, flat=flat)
        key, sk, dk = jax.random.split(key, 3)
        run_diagnostics(init(sk, cfg), vt, cfg, tag, key=dk)

    if args.steps:
        print(f"\n\n=== after {args.steps} training steps, all fixes on ===")
        cfg = dataclasses.replace(base, tied=False, band_pad=1.0, flat=True)
        key, sk = jax.random.split(key)
        p = train_semagram(sk, tr, cfg, args.steps, args.batch, args.lr)
        run_diagnostics(p, vt, cfg, f"trained {args.steps}")


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--task", choices=["text", "bridge"], default="text")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    # 3e-3 trains faster but is bistable at this horizon: the same seed and
    # byte-identical code reached 2.356 in one run and collapsed to the unigram
    # in another, decided by floating-point nondeterminism alone (the repo's
    # own reproducibility warning, further down this README, describes the same
    # chaos). 2e-3 has not been observed to collapse and ends up better anyway.
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--d", type=int, default=TEXT.d)
    ap.add_argument("--heads", type=int, default=TEXT.heads)
    ap.add_argument("--k", type=int, default=TEXT.k_steps)
    ap.add_argument("--band-pad", type=float, default=TEXT.band_pad)
    ap.add_argument("--g-floor", type=float, default=TEXT.g_floor)
    ap.add_argument("--g-init", type=float, default=TEXT.g_init)
    ap.add_argument("--eta", type=float, default=TEXT.eta)
    ap.add_argument("--tied", action="store_true")
    ap.add_argument("--no-flat", dest="flat", action="store_false")
    ap.add_argument("--no-tie-head", dest="tie_head", action="store_false")
    ap.add_argument("--no-baselines", dest="baselines", action="store_false")
    return ap.parse_args()


def cfg_from(args, base, **over):
    """One place where CLI overrides become a Cfg, so every runner agrees."""
    fields = dict(d=args.d, heads=args.heads, k_steps=args.k, tied=args.tied,
                  band_pad=args.band_pad, flat=args.flat, g_floor=args.g_floor,
                  g_init=args.g_init, eta=args.eta, tie_head=args.tie_head)
    fields.update(over)   # explicit overrides win over CLI defaults
    return dataclasses.replace(base, **fields)


if __name__ == "__main__":
    a = parse()
    if a.diagnose:
        main_diagnose(a)
    elif a.ablate:
        main_ablate(a)
    elif a.task == "bridge":
        main_bridge(a)
    else:
        main_text(a)
