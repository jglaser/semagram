"""solver_probe.py -- is the trained model an approximation to the minimiser of
its own action? Answer: no. See Result 5 in the README.

Minimises the SAME action over the free coordinates with L-BFGS, from two very
different starts, and compares against the K-sweep unroll: value of S, gradient
norm, task NLL, and the Hessian spectrum. Also checks the concavity that a
concave-convex (CCCP) splitting would need, which does not hold once queries and
keys come from the same state."""

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
import task_shape as T, loop_layer as L, contours as C
from scipy.optimize import minimize

n, vocab, B = 48, 32, 8
d = C.cached("mnist", n, vocab)
te = jnp.asarray(d["test"][:B]); centres = jnp.asarray(d["centres"])
mask = T.eval_masks(B, n, 12)
p = jax.tree.map(lambda a: jnp.asarray(a, jnp.float64), T.load_ckpt("mnist|sema-so2|s0"))
cfg = L.LoopCfg(n=n, d=64, heads=4, k_steps=8, modes=12, vocab=vocab, phi_dev=0.5)
c = p["emb"][te]; keep = mask[...,None]; freem = 1.0-keep
shape = c.shape

def full(z): return keep*c + freem*jnp.asarray(z).reshape(shape)
@jax.jit
def vg(z):
    v,g = jax.value_and_grad(lambda zz: L.action(p, full(zz), c, mask, cfg))(z)
    return v, g*jnp.broadcast_to(freem, shape).ravel()
def nll_of(x):
    lg = L.logits_of(p, x, cfg)
    lp = jax.nn.log_softmax(lg,-1)
    pick = jnp.take_along_axis(lp, te[...,None], -1)[...,0]
    return float(-jnp.sum(pick*(1-mask))/jnp.sum(1-mask))

def row(tag, z):
    v,g = vg(z); x = full(z)
    print(f"{tag:26s} S {float(v):13.2f}  |grad_free S| {float(jnp.linalg.norm(g)):10.4f}  NLL {nll_of(x):7.4f}")
    return float(v)

x8 = L.solve(p, c, mask, cfg)
z8 = jnp.asarray(np.asarray(x8).ravel())
row("unroll K=8 (as trained)", z8)
zp = jnp.asarray(np.asarray(keep*c + freem*p["prior"]).ravel())
row("init (prior)", zp)

f = lambda z: (lambda o: (float(o[0]), np.asarray(o[1], np.float64)))(vg(jnp.asarray(z)))
for tag, z0 in (("L-BFGS from unroll", np.asarray(z8)), ("L-BFGS from init", np.asarray(zp))):
    r = minimize(f, z0, jac=True, method="L-BFGS-B",
                 options=dict(maxiter=4000, maxfun=6000, ftol=1e-14, gtol=1e-10))
    row(f"{tag} ({r.nit} it)", jnp.asarray(r.x))

# ---- curvature: is the attention term concave in x, as CCCP would need?
print("\nHessian of the action on the FREE coordinates (one example, 768 dims):")
idx = np.where(np.asarray(jnp.broadcast_to(freem, shape))[0].ravel() > 0)[0]
def S1(zf):
    x0 = (keep*c)[0].ravel().at[idx].set(zf)
    return L.action(p, x0.reshape(1,n,cfg.d), c[:1], mask[:1], cfg)
H = np.asarray(jax.hessian(S1)(jnp.asarray(x8)[0].ravel()[idx]))
w = np.linalg.eigvalsh((H+H.T)/2)
print(f"  full action : min {w.min():+.3f}  max {w.max():+.3f}  "
      f"negative {100*(w<0).mean():.1f}%  |  indefinite = {bool((w<0).any() and (w>0).any())}")

def Satt(zf):
    x0 = (keep*c)[0].ravel().at[idx].set(zf)
    return L.s_rest(p, x0.reshape(1,n,cfg.d), cfg)
Ha = np.asarray(jax.hessian(Satt)(jnp.asarray(x8)[0].ravel()[idx]))
wa = np.linalg.eigvalsh((Ha+Ha.T)/2)
print(f"  attention+Hopfield term alone (CCCP assumes this is CONCAVE, i.e. all eigenvalues <= 0):")
print(f"    min {wa.min():+.3f}  max {wa.max():+.3f}  POSITIVE eigenvalues {100*(wa>0).mean():.1f}%")

# curvature at the CONVERGED point, and what the extra sweeps are doing to S
r = minimize(f, np.asarray(z8), jac=True, method="L-BFGS-B",
             options=dict(maxiter=4000, maxfun=6000, ftol=1e-14, gtol=1e-10))
zc = jnp.asarray(r.x)
Hc = np.asarray(jax.hessian(S1)(zc.reshape(shape)[0].ravel()[idx]))
wc = np.linalg.eigvalsh((Hc+Hc.T)/2)
print(f"  full action AT THE MINIMUM: min {wc.min():+.4f} max {wc.max():+.4f} "
      f"negative {100*(wc<0).mean():.1f}%")
print("\nS and NLL along the unroll (is it descending on its own action?):")
for K in (1,2,4,8,12,16,32,64):
    xk = L.solve(p, c, mask, jnp.asarray(0) if False else __import__('dataclasses').replace(cfg, k_steps=K))
    zk = jnp.asarray(np.asarray(xk).ravel())
    v,g = vg(zk)
    print(f"  K={K:3d}  S {float(v):13.2f}  |grad_free S| {float(jnp.linalg.norm(g)):9.3f}  NLL {nll_of(xk):7.4f}")
print(f"  L-BFGS S {float(r.fun):13.2f}  |grad_free S|     ~0.003  NLL {nll_of(full(zc)):7.4f}")
