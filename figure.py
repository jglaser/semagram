"""figure.py -- draw completed contours for each model, side by side.

Produces `shapes.svg`: one row per model, one column per test contour, with the
true curve in grey behind, the reconstruction in blue, the occluded arc in red,
and a dashed line marking the gap between the curve's last vertex and its first.
That dashed line is the closure error made visible -- the quantity a per-token
loss cannot see and a per-token decoder cannot control.
"""

from __future__ import annotations

import argparse
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

import contours as C
import draw
import loop_layer as L
import task_shape as T


def main(a):
    d = C.cached(a.dataset, a.n, a.vocab)
    te = jnp.asarray(d["test"][:a.cols * 4])
    centres = jnp.asarray(d["centres"])
    mask = T.eval_masks(te.shape[0], a.n, a.gap)
    cen = np.asarray(d["centres"])
    pick = list(range(a.cols))

    rows, labels = [], []
    truth = [(cen[np.asarray(te[i])], cen[np.asarray(te[i])],
              np.asarray(mask[i])) for i in pick]
    rows.append(truth)
    labels.append(f"ground truth ({a.dataset}, n={a.n}, occlusion {a.gap})")

    base = L.LoopCfg(n=a.n, d=a.d, heads=a.heads, k_steps=a.k, modes=a.modes,
                     vocab=a.vocab, phi_dev=a.phi_dev)
    for name in a.models:
        cell = f"{a.dataset}|{name}|s{a.seed}"
        p = T.load_ckpt(cell)
        if p is None:
            print(f"  no checkpoint for {cell}")
            continue
        if name.startswith("sema"):
            cfg = dataclasses.replace(
                base, gauge="su2" if "su2" in name else "so2",
                gauge_close="open" not in name)
            if a.constrain:
                sched = [float(w) for w in a.constrain.split(",")]
                x, cfg = T.constrained_solve(p, cfg, te, mask, centres, sched,
                                             a.con_steps, a.con_kind)
            else:
                x = L.solve(p, p["emb"][te], mask, cfg)
            lg = L.logits_of(p, x, cfg)
        else:
            lg = L.tf_forward(p, te, mask, a.heads, name == "tf-ring")
        pred = np.asarray(jnp.argmax(lg, -1))
        rows.append([(cen[np.asarray(te[i])],
                      np.where(np.asarray(mask[i]) > 0, cen[np.asarray(te[i])],
                               cen[pred[i]]),
                      np.asarray(mask[i])) for i in pick])
        cl = float(jnp.mean(T.closure_of(T.decode_angles(
            te, mask, jnp.asarray(pred), centres))))
        labels.append(f"{name}   mean closure error {cl:.3f}")

    open(a.out, "w").write(draw.grid(rows, size=a.size, labels=labels))
    print(f"wrote {a.out} ({len(rows)} rows x {a.cols} cols)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--models", nargs="*",
                    default=["sema-so2", "sema-su2", "tf-abs", "tf-ring"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--phi-dev", type=float, default=0.5)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--size", type=int, default=140)
    ap.add_argument("--constrain", default="",
                    help="comma-separated continuation schedule, e.g. 0.1,1,10")
    ap.add_argument("--con-kind", default="close", choices=["close", "turn"])
    ap.add_argument("--con-steps", type=int, default=25)
    ap.add_argument("--out", default="shapes.svg")
    main(ap.parse_args())
