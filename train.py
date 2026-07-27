"""Two objectives on one backbone.

ar  : causal next-token cross-entropy.  The model commits to a factorization.
bvp : bidirectional any-order denoising.  A random subset of positions is
      replaced by MASK and the loss is the cross-entropy on exactly those.
      At inference any subset can be clamped and the rest solved, which is the
      two-sided conditioning the whole design was arguing for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from .model import Config, forward, init_params


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 5000
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.5      # high wd: the standard accelerant for grokking
    warmup: int = 200
    seed: int = 0
    mask_lo: float = 0.15
    mask_hi: float = 0.90


def make_schedule(tc: TrainConfig):
    warmup = min(tc.warmup, max(tc.steps // 10, 1))
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=tc.lr, warmup_steps=warmup,
        decay_steps=tc.steps, end_value=tc.lr * 0.1,
    )


def ar_loss(params, cfg: Config, phi, tokens):
    inp, tgt = tokens[:, :-1], tokens[:, 1:]
    logits = forward(params, cfg, inp, phi)
    lp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(lp, tgt[..., None], axis=-1)[..., 0]
    return nll.mean()


def bvp_loss(params, cfg: Config, phi, tokens, key):
    B, L = tokens.shape
    k_rate, k_bern, k_force = jax.random.split(key, 3)

    rate = jax.random.uniform(k_rate, (B, 1), minval=0.15, maxval=0.90)
    mask = jax.random.uniform(k_bern, (B, L)) < rate
    # guarantee at least one masked position per example
    forced = jax.random.randint(k_force, (B,), 0, L)
    mask = mask | (jnp.arange(L)[None, :] == forced[:, None])

    inp = jnp.where(mask, cfg.p, tokens)          # cfg.p == MASK id
    logits = forward(params, cfg, inp, phi)
    lp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(lp, tokens[..., None], axis=-1)[..., 0]
    return (nll * mask).sum() / jnp.maximum(mask.sum(), 1)


def train(cfg: Config, tc: TrainConfig, phi, train_seqs: np.ndarray,
          objective: str, verbose: bool = False) -> Tuple[Dict[str, Any], list]:
    key = jax.random.PRNGKey(tc.seed)
    key, k_init = jax.random.split(key)
    params = init_params(cfg, k_init)

    opt = optax.adamw(make_schedule(tc), weight_decay=tc.weight_decay)
    opt_state = opt.init(params)

    if objective == "ar":
        def loss_fn(p, batch, k):
            return ar_loss(p, cfg, phi, batch)
    elif objective == "bvp":
        def loss_fn(p, batch, k):
            return bvp_loss(p, cfg, phi, batch, k)
    else:
        raise ValueError(objective)

    @jax.jit
    def step(params, opt_state, batch, k):
        loss, grads = jax.value_and_grad(loss_fn)(params, batch, k)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    rng = np.random.default_rng(tc.seed)
    history = []
    for i in range(tc.steps):
        sel = rng.integers(0, train_seqs.shape[0], size=tc.batch_size)
        batch = jnp.asarray(train_seqs[sel])
        key, k_step = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, batch, k_step)
        if (i + 1) % 250 == 0:
            history.append((i + 1, float(loss)))
            if verbose:
                print(f"    step {i+1:5d}  loss {float(loss):.4f}", flush=True)

    return params, history
