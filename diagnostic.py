"""Diagnostic for the failed bvp arm.

The any-order objective has to learn every conditional, which for this task means
modular inversion at every index gap.  That is a far larger function class than
the single factorisation an AR model commits to, and the loss also averages over
masks where the answer is information-theoretically undetermined.

So: train on the target conditioning pattern *only* -- clamp both endpoints, mask
the interior -- and see whether two-sided conditioning is learnable at all here.
If this works, the earlier failure was breadth, not the boundary-value idea.
"""

from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from semagram.data import ModArithData
from semagram.evaluate import bvp_eval
from semagram.model import Config, circle_basis, forward, init_params
from semagram.train import TrainConfig, make_schedule


def infill_loss(params, cfg, phi, tokens):
    """Mask exactly the interior; clamp x_0 and x_{L-1}."""
    L = tokens.shape[1]
    q = jnp.arange(1, L - 1)
    inp = tokens.at[:, q].set(cfg.p)
    logits = forward(params, cfg, inp, phi)
    lp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(lp, tokens[..., None], axis=-1)[..., 0]
    return nll[:, 1:-1].mean()


def run(fiber, steps, data, tc, cfg_kw):
    cfg = Config(causal=False, fiber=fiber, **cfg_kw)
    phi = circle_basis(cfg.p, cfg.n_freq)
    key = jax.random.PRNGKey(tc.seed)
    key, k = jax.random.split(key)
    params = init_params(cfg, k)

    opt = optax.adamw(make_schedule(tc), weight_decay=tc.weight_decay)
    opt_state = opt.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        loss, g = jax.value_and_grad(infill_loss)(params, cfg, phi, batch)
        upd, opt_state = opt.update(g, opt_state, params)
        return optax.apply_updates(params, upd), opt_state, loss

    rng = np.random.default_rng(tc.seed)
    t0 = time.time()
    curve = []
    for i in range(steps):
        sel = rng.integers(0, data.train.shape[0], size=tc.batch_size)
        params, opt_state, loss = step(params, opt_state, jnp.asarray(data.train[sel]))
        if (i + 1) % 1000 == 0:
            curve.append((i + 1, float(loss)))
    res = bvp_eval(params, cfg, phi, data.test, "infill")
    res["train_curve"] = curve
    res["wall_s"] = time.time() - t0
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fiber", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    args = ap.parse_args()

    data = ModArithData(p=97, seq_len=8, train_frac=0.4, seed=0)
    tc = TrainConfig(steps=args.steps, batch_size=128, seed=0)
    r = run(args.fiber, args.steps, data, tc, dict(p=97, seq_len=8, d_model=64, n_freq=8))

    try:
        allr = json.load(open("mine_diagnostic.json"))
    except (FileNotFoundError, json.JSONDecodeError):
        allr = {}
    allr[f"{args.fiber}@{args.steps}"] = r
    json.dump(allr, open("mine_diagnostic.json", "w"), indent=2)

    print(f"{args.fiber} infill-only, {args.steps} steps ({r['wall_s']:.0f}s)")
    print(f"  train curve: {[f'{s}:{l:.3f}' for s, l in r['train_curve'][-4:]]}")
    print(f"  test token acc {100*r['token_acc']:.1f}%   exact {100*r['exact_acc']:.1f}%")
    print(f"  acc@1 {100*r['acc_iter1']:.1f}%  acc@4 {100*r['acc_iterT']:.1f}%")
    print(f"  AUROC residual {r['auroc_residual']:.3f}  energy {r['auroc_energy']:.3f}"
          f"  maxprob {r['auroc_maxprob']:.3f}")
