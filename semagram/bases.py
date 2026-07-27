"""Two ways to put Z_p on a product of circles.

`additive`       angle proportional to t         -- diagonalises translation
`multiplicative` angle proportional to log_g(t)  -- diagonalises multiplication

The model code does not care which one it gets: the fiber parameterization takes
`phi` as an argument, so swapping the basis is a one-line change at the call
site.  That is what makes the crossed experiment cheap.
"""

from __future__ import annotations

import numpy as np


def primitive_root(p: int) -> int:
    """Smallest generator of Z_p*."""
    if p == 2:
        return 1
    # factor p-1
    n, f = p - 1, []
    d = 2
    while d * d <= n:
        if n % d == 0:
            f.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        f.append(n)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in f):
            return g
    raise ValueError(f"no primitive root found for p={p}")


def _from_angles(ang: np.ndarray) -> np.ndarray:
    """ang: (p, K) -> (p, 2K) interleaved cos/sin."""
    p, k = ang.shape
    phi = np.empty((p, 2 * k), dtype=np.float32)
    phi[:, 0::2] = np.cos(ang)
    phi[:, 1::2] = np.sin(ang)
    return phi


def additive_basis(p: int, n_freq: int) -> np.ndarray:
    t = np.arange(p)[:, None]
    k = np.arange(1, n_freq + 1)[None, :]
    return _from_angles(2.0 * np.pi * k * t / p)


def multiplicative_basis(p: int, n_freq: int) -> np.ndarray:
    """Angle proportional to the discrete log.  t = 0 has no log; its row is
    zeroed, and in multiplicative tasks the value 0 never occurs anyway."""
    g = primitive_root(p)
    ind = np.zeros(p, dtype=np.int64)
    x = 1
    for e in range(p - 1):
        ind[x] = e
        x = (x * g) % p
    t = ind[:, None].astype(np.float64)
    k = np.arange(1, n_freq + 1)[None, :]
    phi = _from_angles(2.0 * np.pi * k * t / (p - 1))
    phi[0, :] = 0.0
    return phi


BASES = {"additive": additive_basis, "multiplicative": multiplicative_basis}


def build_basis(kind: str, p: int, n_freq: int):
    """kind in {additive, multiplicative}; returns a jnp-ready float32 array."""
    import jax.numpy as jnp
    return jnp.asarray(BASES[kind](p, n_freq))
