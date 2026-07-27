"""A small nanoGPT-style transformer in raw JAX, with three token-space
("fiber") parameterizations and a switchable attention mask.

The factorial design is deliberate.  Two independent claims are on trial:

  factorization : causal / autoregressive   vs.  bidirectional boundary-value
  fiber         : free  vs.  tied  vs.  cyclic

`free`   : separate input embedding and output head.  No closure.
`tied`   : one matrix reads types in and reads them back out.  This is the
           closure that already exists in every real language model.
`cyclic` : token values live on a product of K circles,
               e(t) = concat_j [ a_j cos(2 pi j t / p), a_j sin(2 pi j t / p) ]
           lifted to d_model by a learned map, with the readout tied through
           the same map.  Translation of the token value is then a *rotation*
           of the representation, one 2-plane per frequency -- the transport
           operators from the design, acting on the fiber rather than on
           position, which is where the argument ended up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

Params = Dict[str, Any]


@dataclass(frozen=True)
class Config:
    p: int = 97                 # size of the value alphabet
    seq_len: int = 8
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 2
    n_freq: int = 8             # K, only used by the cyclic fiber
    fiber: str = "tied"         # free | tied | cyclic
    causal: bool = False
    init_scale: float = 0.02

    @property
    def vocab_size(self) -> int:
        return self.p + 1       # + MASK

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


def circle_basis(p: int, n_freq: int) -> jnp.ndarray:
    """Phi[t] = the point of Z_p on a product of n_freq circles."""
    t = np.arange(p)[:, None]
    k = np.arange(1, n_freq + 1)[None, :]
    ang = 2.0 * np.pi * k * t / p
    phi = np.empty((p, 2 * n_freq), dtype=np.float32)
    phi[:, 0::2] = np.cos(ang)
    phi[:, 1::2] = np.sin(ang)
    return jnp.asarray(phi)


# ------------------------------------------------------------------------ init

def init_params(cfg: Config, key: jax.Array) -> Params:
    s = cfg.init_scale
    keys = list(jax.random.split(key, 8 + 8 * cfg.n_layer))
    kp = iter(keys)

    def nrm(shape, scale):
        return scale * jax.random.normal(next(kp), shape, dtype=jnp.float32)

    params: Params = {}

    if cfg.fiber == "cyclic":
        # Match the expected embedding norm of the free/tied case: a Phi row has
        # norm sqrt(K), so shrink the lift accordingly.
        params["emb"] = {
            "amp": jnp.ones((cfg.n_freq,), dtype=jnp.float32),
            "w_up": nrm((2 * cfg.n_freq, cfg.d_model), s / np.sqrt(cfg.n_freq)),
            "mask": nrm((cfg.d_model,), s),
        }
    else:
        params["emb"] = {"tok": nrm((cfg.vocab_size, cfg.d_model), s)}

    params["pos"] = nrm((cfg.seq_len, cfg.d_model), s)

    blocks = []
    for _ in range(cfg.n_layer):
        blocks.append(
            {
                "ln1_g": jnp.ones((cfg.d_model,)), "ln1_b": jnp.zeros((cfg.d_model,)),
                "wq": nrm((cfg.d_model, cfg.d_model), s),
                "wk": nrm((cfg.d_model, cfg.d_model), s),
                "wv": nrm((cfg.d_model, cfg.d_model), s),
                "wo": nrm((cfg.d_model, cfg.d_model), s),
                "ln2_g": jnp.ones((cfg.d_model,)), "ln2_b": jnp.zeros((cfg.d_model,)),
                "w1": nrm((cfg.d_model, 4 * cfg.d_model), s),
                "b1": jnp.zeros((4 * cfg.d_model,)),
                "w2": nrm((4 * cfg.d_model, cfg.d_model), s),
                "b2": jnp.zeros((cfg.d_model,)),
            }
        )
    params["blocks"] = blocks

    params["lnf_g"] = jnp.ones((cfg.d_model,))
    params["lnf_b"] = jnp.zeros((cfg.d_model,))
    params["out_bias"] = jnp.zeros((cfg.p,))
    params["logit_scale"] = jnp.asarray(1.0, dtype=jnp.float32)

    if cfg.fiber == "free":
        params["out_w"] = nrm((cfg.d_model, cfg.p), s)

    return params


# --------------------------------------------------------------------- forward

def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + eps) * g + b


def embed(params: Params, cfg: Config, tokens, phi):
    """tokens: (B, L) int32 in [0, p] where p == MASK."""
    if cfg.fiber == "cyclic":
        amp = jnp.repeat(params["emb"]["amp"], 2)             # (2K,)
        e_val = (phi * amp[None, :]) @ params["emb"]["w_up"]  # (p, d)
        table = jnp.concatenate([e_val, params["emb"]["mask"][None, :]], axis=0)
    else:
        table = params["emb"]["tok"]
    return table[tokens]


def readout(params: Params, cfg: Config, h, phi):
    if cfg.fiber == "free":
        logits = h @ params["out_w"]
    elif cfg.fiber == "tied":
        logits = h @ params["emb"]["tok"][: cfg.p].T
    elif cfg.fiber == "cyclic":
        amp = jnp.repeat(params["emb"]["amp"], 2)
        z = (h @ params["emb"]["w_up"].T) * amp[None, None, :]
        logits = z @ phi.T
    else:
        raise ValueError(cfg.fiber)
    return logits * params["logit_scale"] + params["out_bias"]


def attention(block, cfg: Config, x, causal_mask):
    B, L, D = x.shape
    H, dh = cfg.n_head, cfg.d_head

    def split(w):
        return (x @ w).reshape(B, L, H, dh).transpose(0, 2, 1, 3)

    q, k, v = split(block["wq"]), split(block["wk"]), split(block["wv"])
    att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(dh)
    if causal_mask is not None:
        att = jnp.where(causal_mask, att, -1e9)
    att = jax.nn.softmax(att, axis=-1)
    out = (att @ v).transpose(0, 2, 1, 3).reshape(B, L, D)
    return out @ block["wo"]


def forward(params: Params, cfg: Config, tokens, phi):
    """-> logits (B, L, p)"""
    L = tokens.shape[1]
    x = embed(params, cfg, tokens, phi) + params["pos"][:L][None]

    causal_mask = None
    if cfg.causal:
        causal_mask = jnp.tril(jnp.ones((L, L), dtype=bool))[None, None]

    for block in params["blocks"]:
        x = x + attention(block, cfg, layer_norm(x, block["ln1_g"], block["ln1_b"]), causal_mask)
        h = layer_norm(x, block["ln2_g"], block["ln2_b"])
        x = x + (jax.nn.gelu(h @ block["w1"] + block["b1"]) @ block["w2"] + block["b2"])

    x = layer_norm(x, params["lnf_g"], params["lnf_b"])
    return readout(params, cfg, x, phi)


def n_params(params: Params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))
