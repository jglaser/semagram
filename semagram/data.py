"""Arithmetic progressions mod p.

A sequence is x_i = (x0 + i*s) mod p for i = 0..L-1.

This is deliberately the most favourable possible test bed for the claims:

  * The interior is a two-point boundary value problem.  Given x_0 and x_{L-1},
    the step is s = (x_{L-1} - x_0) * (L-1)^{-1} mod p, which exists and is
    unique whenever gcd(L-1, p) = 1.  So "know the endpoints, recover the path"
    is literally, exactly true here -- Fermat's principle with no metaphor.

  * The token alphabet Z_p is a cyclic group and the generating operation is
    translation, so a product-of-circles embedding makes the task linear.

Both of those are hardcoded gifts to the architecture under test.  A win here
is necessary, not sufficient.  See `permute_vocab` for the control.
"""

from __future__ import annotations

import numpy as np


class ModArithData:
    def __init__(
        self,
        p: int = 97,
        seq_len: int = 8,
        train_frac: float = 0.4,
        permute_vocab: bool = False,
        seed: int = 0,
    ):
        assert seq_len >= 3
        self.p = p
        self.seq_len = seq_len
        self.mask_id = p          # MASK is one past the value alphabet
        self.vocab_size = p + 1
        self.permute_vocab = permute_vocab

        rng = np.random.default_rng(seed)

        # Every (x0, s) pair.  s = 0 gives a constant sequence; that is fine.
        x0 = np.arange(p, dtype=np.int64)
        s = np.arange(p, dtype=np.int64)
        X0, S = np.meshgrid(x0, s, indexing="ij")
        idx = np.arange(seq_len, dtype=np.int64)
        seqs = (X0[..., None] + S[..., None] * idx[None, None, :]) % p
        seqs = seqs.reshape(-1, seq_len)

        # Control: relabel the alphabet with a fixed random permutation.  The
        # sequence family is structurally identical but the cyclic embedding's
        # circle no longer lines up with the group, so a geometry prior that
        # only works because it matched the task should collapse here.
        if permute_vocab:
            perm = rng.permutation(p)
            seqs = perm[seqs]
            self.perm = perm
        else:
            self.perm = np.arange(p)

        order = rng.permutation(seqs.shape[0])
        n_train = int(round(train_frac * seqs.shape[0]))
        self.train = seqs[order[:n_train]]
        self.test = seqs[order[n_train:]]

    def __repr__(self) -> str:
        return (
            f"ModArithData(p={self.p}, L={self.seq_len}, "
            f"train={len(self.train)}, test={len(self.test)}, "
            f"permuted={self.permute_vocab})"
        )

    # ---------------------------------------------------------------- orderings

    def reorder_endpoints_first(self, seqs: np.ndarray) -> np.ndarray:
        """Present as [x_0, x_{L-1}, x_1, ..., x_{L-2}].

        This is the strong autoregressive baseline: an AR model trained on this
        factorisation sees both boundary conditions before it has to produce any
        interior token, so it can do the BVP task natively.  Including it keeps
        the comparison from being a straw man.
        """
        first = seqs[:, :1]
        last = seqs[:, -1:]
        mid = seqs[:, 1:-1]
        return np.concatenate([first, last, mid], axis=1)

    # ---------------------------------------------------------------- batching

    def batches(self, split: str, batch_size: int, steps: int, seed: int = 0):
        data = self.train if split == "train" else self.test
        rng = np.random.default_rng(seed)
        for _ in range(steps):
            sel = rng.integers(0, data.shape[0], size=batch_size)
            yield data[sel]


# ------------------------------------------------------------------ eval specs

def query_positions(seq_len: int, mode: str) -> np.ndarray:
    """Which positions the model is asked to produce, given all others.

    next    : the final token, from the whole prefix     (AR's home turf)
    infill  : every interior token, from both endpoints  (the BVP / Fermat task)
    reverse : the first token, from the whole suffix     (the reversal task)
    """
    if mode == "next":
        return np.array([seq_len - 1])
    if mode == "infill":
        return np.arange(1, seq_len - 1)
    if mode == "reverse":
        return np.array([0])
    raise ValueError(mode)


EVAL_MODES = ("next", "infill", "reverse")
