"""variational_gap.py -- is the model at the minimum of its own action?

The one cheap, general test for whether a variational forward pass is doing work
rather than bookkeeping. Minimise the action properly and score the result:

    gap = NLL(argmin S) - NLL(unroll)

    gap > 0   the minimiser is WORSE than what the network computes. The energy
              did not define the model; it generated a recurrence and training
              shaped the recurrence. Every downstream promise of the framing --
              a convergence certificate, constraints added at inference,
              composing two energies -- is unavailable, because the object those
              promises are about is not the object being used.

    gap ~ 0   the network's output IS the stationary point. The certificate
              certifies something true, and the rest becomes available.

Measured on `sema-so2` as trained, the gap is +0.329 nats with ||grad S|| at 137
against 0.003 at the true minimum. `loop_layer.LoopCfg.w_stat` is the attempt to
close it by putting the gap in the training loss.

    python variational_gap.py mnist|sema-so2|s0 mnist|sema-stat0.3|s0
"""

from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

import contours as C          # noqa: E402
import loop_layer as L        # noqa: E402
import task_shape as T        # noqa: E402


def gap(cell, n=48, vocab=32, d=64, batch=8, occl=12, maxiter=4000):
    ds = cell.split("|")[0]
    name = cell.split("|")[1]
    data = C.cached(ds, n, vocab)
    te = jnp.asarray(data["test"][:batch])
    mask = T.eval_masks(batch, n, occl)
    p = T.load_ckpt(cell)
    if p is None:
        return None
    p = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), p)
    cfg = L.LoopCfg(n=n, d=d, heads=4, k_steps=8, modes=12, vocab=vocab,
                    phi_dev=0.5, gauge="su2" if "su2" in name else "so2",
                    odd_conv="odd" in name)
    c = p["emb"][te]
    keep = mask[..., None]
    freem = jnp.broadcast_to(1.0 - keep, c.shape)
    shape = c.shape

    def full(z):
        return keep * c + (1.0 - keep) * jnp.asarray(z).reshape(shape)

    @jax.jit
    def vg(z):
        v, g = jax.value_and_grad(
            lambda zz: L.action(p, full(zz), c, mask, cfg))(z)
        return v, g * freem.ravel()

    def nll(x):
        lp = jax.nn.log_softmax(L.logits_of(p, x, cfg), -1)
        pick = jnp.take_along_axis(lp, te[..., None], -1)[..., 0]
        return float(-jnp.sum(pick * (1 - mask)) / jnp.sum(1 - mask))

    z8 = jnp.asarray(np.asarray(L.solve(p, c, mask, cfg)).ravel())
    s8, g8 = vg(z8)
    f = lambda z: (lambda o: (float(o[0]), np.asarray(o[1], np.float64)))(
        vg(jnp.asarray(z)))
    r = minimize(f, np.asarray(z8), jac=True, method="L-BFGS-B",
                 options=dict(maxiter=maxiter, maxfun=maxiter + 2000,
                              ftol=1e-14, gtol=1e-10))
    sc, gc = vg(jnp.asarray(r.x))
    n8, nc = nll(full(z8)), nll(full(jnp.asarray(r.x)))
    return dict(nll_unroll=n8, nll_min=nc, gap=nc - n8,
                grad_unroll=float(jnp.linalg.norm(g8)),
                grad_min=float(jnp.linalg.norm(gc)),
                S_unroll=float(s8), S_min=float(sc), iters=int(r.nit))


if __name__ == "__main__":
    cells = sys.argv[1:] or ["mnist|sema-so2|s0"]
    print(f"{'model':16s} {'NLL unroll':>11s} {'NLL argmin':>11s} {'GAP':>8s} "
          f"{'|gS| unroll':>12s} {'|gS| min':>10s}")
    print("-" * 74)
    for cell in cells:
        r = gap(cell)
        if r is None:
            print(f"{cell:16s} (no checkpoint)")
            continue
        print(f"{cell.split('|')[1]:16s} {r['nll_unroll']:11.4f} "
              f"{r['nll_min']:11.4f} {r['gap']:+8.4f} {r['grad_unroll']:12.3f} "
              f"{r['grad_min']:10.4f}")
