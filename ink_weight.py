"""Does the error concentrate where the 'ink' is heavy? Stratify per-position
NLL by LOCAL CURVATURE (|turning angle|), which is the intrinsic, origin-free
analogue of stroke weight."""
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import task_shape as T, loop_layer as L, contours as C, optax

n, vocab, B = 48, 32, 512
d = C.cached("mnist", n, vocab)
te = jnp.asarray(d["test"][:B]); centres = np.asarray(d["centres"])
mask = T.eval_masks(B, n, 12)
truth = centres[np.asarray(te)]                       # true turning per position
q = np.quantile(np.abs(truth), [0.2,0.4,0.6,0.8])
base = L.LoopCfg(n=n,d=64,heads=4,k_steps=8,modes=12,vocab=vocab,phi_dev=0.5)
print(f"|turning| quintile edges: {np.round(q,3)}   (mean {np.abs(truth).mean():.3f})")
print(f"{'model':10s}" + "".join(f"{'Q'+str(i+1):>9s}" for i in range(5)) + f"{'Q5-Q1':>9s}")
for name, over in [("sema-so2",dict(gauge="so2")),("sema-su2",dict(gauge="su2")),
                   ("tf-abs",None),("tf-ring",None)]:
    p = T.load_ckpt(f"mnist|{name}|s0")
    if p is None: continue
    if over is not None:
        cfg = dataclasses.replace(base, **over)
        lg = L.logits_of(p, L.solve(p, p["emb"][te], mask, cfg), cfg)
    else:
        lg = L.tf_forward(p, te, mask, 4, name=="tf-ring")
    ce = np.asarray(optax.softmax_cross_entropy_with_integer_labels(lg, te))
    fr = np.asarray(1-mask)
    b = np.digitize(np.abs(truth), q)
    row=f"{name:10s}"; vals=[]
    for k in range(5):
        m = (b==k)&(fr>0)
        vals.append(ce[m].mean()); row += f"{vals[-1]:9.3f}"
    print(row + f"{vals[-1]-vals[0]:+9.3f}")
