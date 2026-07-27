"""One place that trains a cell, evaluates it, and persists it.

Every experiment script here is a loop over cells plus a table.  Results are
written after each cell and reloaded on startup, so an interrupted sweep costs
at most one cell.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

import numpy as np

from . import tasks
from .bases import build_basis
from .evaluate import ar_eval, bvp_eval
from .model import Config, n_params
from .train import TrainConfig, train


def run_cell(task: str, basis: str, fiber: str, objective: str, seed: int,
             steps: int, p: int = 97, seq_len: int = 8, d_model: int = 64,
             n_freq: int = 8, batch_size: int = 128,
             train_frac: float = 0.4) -> Dict:
    """Train and evaluate one configuration.

    objective: 'infill' (bidirectional, endpoints clamped) or
               'ar'      (causal, endpoints-first ordering -- the matched
                          autoregressive baseline for the same inversion)
    """
    tr, te = tasks.build(task, p, seq_len, train_frac, seed)
    cfg = Config(p=p, seq_len=seq_len, d_model=d_model, n_freq=n_freq,
                 fiber=fiber, causal=(objective == "ar"))
    phi = build_basis(basis, p, n_freq)
    tc = TrainConfig(steps=steps, batch_size=batch_size, seed=seed)

    train_seqs = tr
    if objective == "ar":
        # [x_0, x_{L-1}, x_1, ..., x_{L-2}] so the AR model sees both boundary
        # conditions before it must produce any interior token.
        train_seqs = np.concatenate([tr[:, :1], tr[:, -1:], tr[:, 1:-1]], axis=1)

    t0 = time.time()
    params, hist = train(cfg, tc, phi, train_seqs, objective)
    wall = time.time() - t0

    if objective == "ar":
        ev = ar_eval(params, cfg, phi, te, "infill", "endpoints_first")
        if ev is None:
            raise RuntimeError("ar/endpoints_first should be native for infill")
    else:
        ev = bvp_eval(params, cfg, phi, te, "infill")

    return {
        "task": task, "basis": basis, "fiber": fiber, "objective": objective,
        "seed": seed, "steps": steps, "n_params": n_params(params),
        "train_loss": hist[-1][1], "curve": hist, "wall_s": wall,
        "token_acc": ev["token_acc"], "exact_acc": ev["exact_acc"],
        "acc_iter1": ev.get("acc_iter1"), "acc_iterT": ev.get("acc_iterT"),
        "auroc_residual": ev.get("auroc_residual"),
        "auroc_energy": ev.get("auroc_energy"),
        "auroc_maxprob": ev.get("auroc_maxprob"),
    }


class Store:
    """Append-only JSON store keyed by cell name."""

    def __init__(self, path: str):
        self.path = path
        self.data: Dict[str, Dict] = {}
        if os.path.exists(path):
            try:
                with open(path) as f:
                    self.data = json.load(f)
                print(f"resuming: {len(self.data)} cells in {path}", flush=True)
            except json.JSONDecodeError:
                print(f"warning: {path} unreadable, starting fresh", flush=True)

    def has(self, key: str) -> bool:
        return key in self.data

    def put(self, key: str, value: Dict) -> None:
        self.data[key] = value
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def select(self, **match) -> List[Dict]:
        out = []
        for v in self.data.values():
            if all(v.get(k) == val for k, val in match.items()):
                out.append(v)
        return out


def agg(rows: List[Dict], key: str = "token_acc") -> Optional[Dict]:
    if not rows:
        return None
    x = np.array([r[key] for r in rows], dtype=float)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max()), "n": len(x)}


def fmt(a: Optional[Dict]) -> str:
    if a is None:
        return "       --      "
    return f"{100*a['mean']:5.1f} +/- {100*a['std']:4.1f}%"


def welch(a: List[float], b: List[float]) -> Optional[float]:
    """Welch t statistic.  Reported as a rough effect indicator only -- with 3-5
    seeds this is not a p-value, it is a sanity check that a gap is larger than
    the seed noise."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    va, vb = a.var(ddof=1), b.var(ddof=1)
    denom = np.sqrt(va / len(a) + vb / len(b))
    if denom == 0:
        return float("inf") if a.mean() != b.mean() else 0.0
    return float((a.mean() - b.mean()) / denom)
