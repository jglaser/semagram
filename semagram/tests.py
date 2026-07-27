"""Turn the claims into assertions.  `python tests.py`"""

import jax
import jax.numpy as jnp
import numpy as np

import semagram as sg

jax.config.update("jax_enable_x64", True)


def setup(n=10, d=8, heads=2, vocab=7, modes=4, seed=0):
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    P = sg.init_params(k[0], vocab, d, heads, modes)
    P = jax.tree.map(lambda x: x.astype(jnp.float64), P)
    X = jax.random.normal(k[1], (n, d), dtype=jnp.float64)
    C = jax.random.normal(k[2], (n, d), dtype=jnp.float64)
    vis = jnp.array(np.r_[np.ones(n - 3), np.zeros(3)], dtype=jnp.float64)
    return P, X, C, vis


def t1_scores_symmetric():
    P, X, _, _ = setup()
    s = sg.scores(X, P)
    err = jnp.abs(s - jnp.swapaxes(s, -1, -2)).max()
    print(f"  1. scores symmetric              max|s - s^T| = {err:.3e}")
    assert err < 1e-11


def t2_attention_is_a_gradient():
    """-grad(associative term) must equal the symmetrised attention readout."""
    P, X, _, _ = setup()
    beta = 1.7
    g = jax.grad(sg.associative)(X, P, beta, False)
    readout = sg.attention_readout(X, P, beta)
    err = jnp.abs(-g - readout).max() / jnp.abs(readout).max()
    print(f"  2. attention == -grad(E)         rel err      = {err:.3e}")
    assert err < 1e-10


def t3_hessian_symmetric():
    P, X, C, vis = setup(n=6, d=8, heads=2)
    f = lambda x: sg.action(x.reshape(6, 8), P, C, vis, 1.3)
    H = jax.hessian(f)(X.reshape(-1))
    err = jnp.abs(H - H.T).max() / jnp.abs(H).max()
    print(f"  3. Hessian symmetric (=> CG ok)  rel err      = {err:.3e}")
    assert err < 1e-12


def t4_solver_converges():
    P, X, C, vis = setup()
    _, hist = sg._solve_scan(C, P, C, vis, 1.5, 60, 0.7, "fp", 6)
    print(f"  4. residual |gradS| decay        {hist[0]:.2e} -> {hist[-1]:.2e}")
    assert hist[-1] < hist[0] * 1e-3, f"{hist[0]} -> {hist[-1]}"


def t5_implicit_matches_unrolled():
    """Implicit-diff gradient vs backprop-through-the-solver."""
    P, _, C, vis = setup()
    beta, iters, relax = 1.5, 120, 0.7

    def unrolled(p):
        X, _ = sg._solve_scan(C, p, C, vis, beta, iters, relax, "fp", 6)
        return jnp.sum(jnp.sin(X))

    def implicit(p):
        X = sg.solve_implicit(p, C, vis, C, beta, iters, relax, 40, "fp", 6)
        return jnp.sum(jnp.sin(X))

    gu = jax.grad(unrolled)(P)
    gi = jax.grad(implicit)(P)
    num = jnp.sqrt(sum(jnp.sum((a - b) ** 2)
                       for a, b in zip(jax.tree.leaves(gu), jax.tree.leaves(gi))))
    den = jnp.sqrt(sum(jnp.sum(a ** 2) for a in jax.tree.leaves(gu)))
    print(f"  5. implicit vs unrolled grad     rel err      = {num/den:.3e}")
    assert num / den < 5e-3


def t6_not_reflection_blind():
    """The odd-phase fix: reversing the loop must change the representation."""
    P, X, C, vis = setup(n=12)
    rev = lambda A: A[::-1]
    s = sg.scores(X, P)
    s_rev = sg.scores(rev(X), P)
    # if the model were mirror-blind, s_rev would equal s reversed on both axes
    mirrored = s[:, ::-1, :][:, :, ::-1]
    err = jnp.abs(s_rev - mirrored).max() / jnp.abs(s).max()
    print(f"  6. reversal is NOT a symmetry    rel diff     = {err:.3e}")
    assert err > 1e-2, "model is mirror-blind -- the odd phase is missing"


if __name__ == "__main__":
    print("semagram claim tests (float64)")
    for t in [t1_scores_symmetric, t2_attention_is_a_gradient, t3_hessian_symmetric,
              t4_solver_converges, t5_implicit_matches_unrolled,
              t6_not_reflection_blind]:
        t()
    print("all passed")
