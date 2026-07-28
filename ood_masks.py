"""Every model here was trained on ONE conditioning family: a single contiguous
occluded arc of length 6..18. An energy model defines a joint and should let you
condition on an arbitrary subset; a feedforward masked model has to have seen
the mask family. So: hold the number of hidden vertices at 12 and vary only the
SHAPE of the conditioning set, out of distribution."""
import dataclasses

import numpy as np
import jax
import jax.numpy as jnp
import task_shape as T, loop_layer as L, contours as C

n, vocab, B = 48, 32, 512
d = C.cached("mnist", n, vocab)
te = jnp.asarray(d["test"][:B]); centres = jnp.asarray(d["centres"])
rng = np.random.default_rng(0)

def contiguous(k=12):
    s = rng.integers(0,n,B); off=(np.arange(n)[None,:]-s[:,None])%n
    return (off>=k).astype(np.float32)
def scattered(k=12):
    m=np.ones((B,n),np.float32)
    for i in range(B): m[i, rng.choice(n,k,replace=False)]=0.
    return m
def periodic(k=12):
    m=np.ones((B,n),np.float32); ph=rng.integers(0,4,B)
    for i in range(B): m[i, (np.arange(k)*4+ph[i])%n]=0.
    return m
def two_arcs(k=12):
    m=np.ones((B,n),np.float32)
    a=rng.integers(0,n,B); b=rng.integers(0,n,B)
    for i in range(B):
        m[i,(np.arange(k//2)+a[i])%n]=0.; m[i,(np.arange(k//2)+b[i])%n]=0.
    return m
def long_arc(k=30):
    s=rng.integers(0,n,B); off=(np.arange(n)[None,:]-s[:,None])%n
    return (off>=k).astype(np.float32)

FAM=[("contiguous 12 (IN-DIST)",contiguous),("two arcs of 6",two_arcs),
     ("scattered 12",scattered),("periodic every-4th",periodic),
     ("contiguous 30 (2x train max)",long_arc)]
base=L.LoopCfg(n=n,d=64,heads=4,k_steps=8,modes=12,vocab=vocab,phi_dev=0.5)
MODELS=[("sema-so2",dict(gauge="so2")),("sema-su2",dict(gauge="su2")),
        ("tf-abs",None),("tf-ring",None)]
res={}
for name,over in MODELS:
    for fam,mk in FAM:
        vals=[]
        for s in (0,1,2):
            p=T.load_ckpt(f"mnist|{name}|s{s}")
            if p is None: continue
            m=jnp.asarray(mk())
            if over is not None:
                cfg=dataclasses.replace(base,**over)
                vals.append(T.eval_semagram(p,cfg,te,m,centres)["nll"])
            else:
                vals.append(T.eval_tf(p,te,m,centres,4,name=="tf-ring")["nll"])
        if vals: res[(name,fam)]=(float(np.mean(vals)),float(np.std(vals)))
hdr=f"{'model':10s}" + "".join(f"{f[:22]:>24s}" for f,_ in FAM)
print(hdr); print("-"*len(hdr))
for name,_ in MODELS:
    row=f"{name:10s}"
    for fam,_ in FAM:
        v=res.get((name,fam))
        row += f"{v[0]:>17.3f}+-{v[1]:.3f}" if v else f"{'-':>24s}"
    print(row)
print("\ndegradation vs its own in-distribution score:")
for name,_ in MODELS:
    b=res.get((name,FAM[0][0]))
    if not b: continue
    print(f"  {name:10s}" + "".join(
        f"{res[(name,f)][0]-b[0]:+9.3f}" for f,_ in FAM[1:] if (name,f) in res))
