"""Evaluation under three conditioning patterns, plus the halting certificate.

An autoregressive model trained on one ordering can only answer a query whose
*given* positions form a prefix of that ordering.  That is checked explicitly
rather than assumed, and cells it cannot reach are reported as unavailable
instead of being silently scored as failures.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from .data import query_positions
from .model import Config, forward


# ----------------------------------------------------------------- orderings

def ordering(seq_len: int, kind: str) -> np.ndarray:
    if kind == "std":
        return np.arange(seq_len)
    if kind == "endpoints_first":
        mid = np.arange(1, seq_len - 1)
        return np.concatenate([[0], [seq_len - 1], mid])
    raise ValueError(kind)


def ar_is_native(perm: np.ndarray, given: np.ndarray) -> bool:
    """True iff the given positions are exactly a prefix of the ordering."""
    return set(perm[: len(given)].tolist()) == set(given.tolist())


# -------------------------------------------------------------- rank helpers

def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a correct example > score of an incorrect one)."""
    labels = labels.astype(bool)
    npos, nneg = int(labels.sum()), int((~labels).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = _rankdata(scores)
    return float((r[labels].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


# --------------------------------------------------------- AR greedy decoding

def ar_eval(params, cfg: Config, phi, seqs: np.ndarray, mode: str,
            order_kind: str) -> Optional[Dict[str, float]]:
    L = cfg.seq_len
    perm = ordering(L, order_kind)
    q = query_positions(L, mode)
    given = np.array(sorted(set(range(L)) - set(q.tolist())))

    if not ar_is_native(perm, given):
        return None

    ordered = seqs[:, perm]                       # (N, L) in model order
    n_given = len(given)
    cur = jnp.asarray(ordered[:, :n_given])

    fwd = jax.jit(lambda t: forward(params, cfg, t, phi))
    for _ in range(n_given, L):
        logits = fwd(cur)
        nxt = jnp.argmax(logits[:, -1, :], axis=-1)
        cur = jnp.concatenate([cur, nxt[:, None]], axis=1)

    pred_ordered = np.asarray(cur)
    inv = np.argsort(perm)
    pred = pred_ordered[:, inv]                   # back to original positions

    tok = (pred[:, q] == seqs[:, q]).mean()
    exact = (pred[:, q] == seqs[:, q]).all(axis=1).mean()
    return {"token_acc": float(tok), "exact_acc": float(exact)}


# ------------------------------------------------------- BVP iterative solve

def bvp_eval(params, cfg: Config, phi, seqs: np.ndarray, mode: str,
             n_iter: int = 4) -> Dict[str, float]:
    L = cfg.seq_len
    q = query_positions(L, mode)
    qj = jnp.asarray(q)

    tokens = jnp.asarray(seqs)
    inp = tokens.at[:, qj].set(cfg.p)             # clamp everything but Q

    fwd = jax.jit(lambda t: forward(params, cfg, t, phi))

    prev_probs = None
    accs: List[float] = []
    residuals = np.zeros(seqs.shape[0])
    for _ in range(n_iter):
        logits = fwd(inp)
        lq = logits[:, qj, :]
        probs = jax.nn.softmax(lq, axis=-1)
        pred = jnp.argmax(lq, axis=-1)
        accs.append(float((np.asarray(pred) == seqs[:, q]).all(axis=1).mean()))
        if prev_probs is not None:
            residuals = np.asarray(jnp.abs(probs - prev_probs).sum(-1).mean(-1))
        prev_probs = probs
        inp = inp.at[:, qj].set(pred)

    pred = np.asarray(pred)
    correct = (pred == seqs[:, q]).all(axis=1)
    lse = np.asarray(jax.scipy.special.logsumexp(lq, axis=-1).mean(-1))
    maxp = np.asarray(probs.max(-1).mean(-1))

    return {
        "token_acc": float((pred == seqs[:, q]).mean()),
        "exact_acc": float(correct.mean()),
        "acc_iter1": accs[0],
        "acc_iterT": accs[-1],
        # halting certificate: does a scalar read off the forward pass predict
        # whether this particular example came out right?
        "auroc_residual": auroc(-residuals, correct),
        "auroc_energy": auroc(lse, correct),          # low energy == high LSE
        "auroc_maxprob": auroc(maxp, correct),        # the control
    }
