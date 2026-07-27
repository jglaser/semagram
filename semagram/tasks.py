"""
Tasks.  Each returns (tokens, visible) with visible=1 where the site is clamped.

`bridge` is the decisive one: the masked span is a deterministic function of
*both* the material before it and the material after it.  It is constructed
precisely to separate a boundary-value solver from a left-to-right one, and it
is reported as such -- an autoregressive model cannot see the right boundary,
so it is at chance on the span no matter how well it fits the data.

`antipode` tests the ring topology instead: each site is determined by the site
diametrically opposite, so the required information is always at maximum
distance along the loop.

`palindrome` is the reflection control, and the reason the odd-phase fix
matters -- a model with exact reflection symmetry gets this for free while
being unable to tell "dog bites man" from "man bites dog".
"""

import numpy as np


def bridge(rng, batch, k=8, vocab=16, op="revcopy"):
    """[ P | M | S ] on a ring of length 3k; M is a function of BOTH ends.

    op="revcopy"  M = reverse(S)        -- needs the right boundary, easy to fit
    op="mix"      M = S where P even    -- needs both boundaries
    op="modsum"   M = (P + S) mod vocab -- also needs both, but modular
                  arithmetic is a grokking-scale problem and will sit at chance
                  for thousands of steps; it is here as the hard setting, not
                  as a smoke test.
    """
    n = 3 * k
    P = rng.integers(0, vocab, size=(batch, k))
    S = rng.integers(0, vocab, size=(batch, k))
    if op == "revcopy":
        M = S[:, ::-1]
    elif op == "mix":
        M = np.where(P % 2 == 0, S, P)
    else:
        M = (P + S) % vocab
    tokens = np.concatenate([P, M, S], axis=1)
    visible = np.ones((batch, n), dtype=np.float32)
    visible[:, k : 2 * k] = 0.0
    return tokens.astype(np.int32), visible


def antipode(rng, batch, half=12, vocab=16, span=6, seed=0):
    """x_{i + n/2} = perm[x_i].  Mask a contiguous arc of length `span`."""
    n = 2 * half
    perm = np.random.default_rng(seed).permutation(vocab)
    A = rng.integers(0, vocab, size=(batch, half))
    tokens = np.concatenate([A, perm[A]], axis=1)
    visible = np.ones((batch, n), dtype=np.float32)
    starts = rng.integers(0, n, size=batch)
    idx = (starts[:, None] + np.arange(span)[None, :]) % n
    np.put_along_axis(visible, idx, 0.0, axis=1)
    return tokens.astype(np.int32), visible


def palindrome(rng, batch, half=12, vocab=16, span=6):
    n = 2 * half
    A = rng.integers(0, vocab, size=(batch, half))
    tokens = np.concatenate([A, A[:, ::-1]], axis=1)
    visible = np.ones((batch, n), dtype=np.float32)
    starts = rng.integers(0, n, size=batch)
    idx = (starts[:, None] + np.arange(span)[None, :]) % n
    np.put_along_axis(visible, idx, 0.0, axis=1)
    return tokens.astype(np.int32), visible


class CharText:
    """Char-level loops carved out of a corpus, with a random arc masked."""

    def __init__(self, path, n=48, span=12):
        with open(path) as f:
            text = f.read()
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.data = np.array([self.stoi[c] for c in text], dtype=np.int32)
        self.vocab = len(self.chars)
        self.n, self.span = n, span

    def __call__(self, rng, batch):
        starts = rng.integers(0, len(self.data) - self.n, size=batch)
        tokens = np.stack([self.data[s : s + self.n] for s in starts])
        visible = np.ones((batch, self.n), dtype=np.float32)
        ms = rng.integers(0, self.n, size=batch)
        idx = (ms[:, None] + np.arange(self.span)[None, :]) % self.n
        np.put_along_axis(visible, idx, 0.0, axis=1)
        return tokens, visible


REGISTRY = {
    "bridge": (bridge, dict(vocab=16, n=24, span=8)),
    "antipode": (antipode, dict(vocab=16, n=24, span=6)),
    "palindrome": (palindrome, dict(vocab=16, n=24, span=6)),
}
