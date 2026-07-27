"""Sequence families over Z_p, all of which are two-point boundary value problems.

additive        x_i = (x_0 + i*s) mod p            group: (Z_p, +)
multiplicative  x_i = (x_0 * s^i) mod p            group: (Z_p*, x)
permuted        additive, then the alphabet is relabelled by a random bijection

The point of `multiplicative` is that it is exactly as learnable as `additive`
-- same alphabet, same sequence length, same "two endpoints determine the
interior" structure, a cyclic group underneath -- but the cyclic group is
Z_{p-1} acting through discrete logarithm, not Z_p acting by translation.  An
additive-character embedding is therefore in the *wrong coordinates* for it.

That is what makes it a better control than `permuted`.  Permuting the alphabet
destroys learnability for every fiber at once, so the comparison comes back
void (which is what happened in the first grid).  The multiplicative task keeps
the task learnable and moves only the coordinate system.
"""

from __future__ import annotations

import math

import numpy as np


def _require_unique_interior(gap: int, order: int, name: str) -> None:
    """The interior is recoverable from the endpoints iff gcd(L-1, order) == 1."""
    g = math.gcd(gap, order)
    if g != 1:
        raise ValueError(
            f"{name}: gcd(L-1={gap}, group order={order}) = {g} != 1, so the "
            f"endpoints do not determine a unique interior. Pick another "
            f"(p, seq_len)."
        )


def additive(p: int, seq_len: int) -> np.ndarray:
    _require_unique_interior(seq_len - 1, p, "additive")
    x0 = np.arange(p, dtype=np.int64)[:, None, None]
    s = np.arange(p, dtype=np.int64)[None, :, None]
    i = np.arange(seq_len, dtype=np.int64)[None, None, :]
    return ((x0 + s * i) % p).reshape(-1, seq_len)


def multiplicative(p: int, seq_len: int) -> np.ndarray:
    """Values live in Z_p* = {1..p-1}; 0 never occurs."""
    _require_unique_interior(seq_len - 1, p - 1, "multiplicative")
    vals = np.arange(1, p, dtype=np.int64)
    cur = np.repeat(vals[:, None], p - 1, axis=1)          # rows: x0, cols: s
    s = vals[None, :]
    out = []
    for _ in range(seq_len):
        out.append(cur.copy())
        cur = (cur * s) % p
    return np.stack(out, axis=-1).reshape(-1, seq_len)


def permuted(p: int, seq_len: int, seed: int = 0) -> np.ndarray:
    perm = np.random.default_rng(seed).permutation(p)
    return perm[additive(p, seq_len)]


FAMILIES = {"additive": additive, "multiplicative": multiplicative,
            "permuted": permuted}


def build(task: str, p: int, seq_len: int, train_frac: float, seed: int):
    """-> (train, test) arrays of shape (N, seq_len)."""
    if task == "permuted":
        seqs = permuted(p, seq_len, seed=seed)
    else:
        seqs = FAMILIES[task](p, seq_len)
    rng = np.random.default_rng(seed)
    order = rng.permutation(seqs.shape[0])
    n_train = int(round(train_frac * seqs.shape[0]))
    return seqs[order[:n_train]], seqs[order[n_train:]]
