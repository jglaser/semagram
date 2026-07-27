"""Run the factorial experiment and print the tables.

    python run_experiment.py                 # full grid + control
    python run_experiment.py --steps 1500    # fast smoke test

Two claims are separated:

  1. factorization -- does two-sided conditioning beat a committed
     autoregressive ordering when the evidence sits at both ends?
  2. fiber        -- does a closed (cyclic) token space beat free or tied
     embeddings?

The permuted-vocab control relabels Z_p with a fixed random permutation.  The
sequence family is structurally identical, but the cyclic embedding's circles no
longer align with the group.  If `cyclic` keeps its advantage there, the win was
not coming from the geometry and my analysis is wrong.
"""

from __future__ import annotations

import argparse
import json
import time

import jax

from semagram.data import EVAL_MODES, ModArithData
from semagram.evaluate import ar_eval, bvp_eval
from semagram.model import Config, circle_basis, n_params
from semagram.train import TrainConfig, train

FIBERS = ("free", "tied", "cyclic")
ARMS = (("ar", "std"), ("ar", "endpoints_first"), ("bvp", None))


def arm_name(objective, order):
    return f"ar/{order}" if objective == "ar" else "bvp"


def run_one(data, cfg_kw, objective, order, tc, verbose=False):
    cfg = Config(causal=(objective == "ar"), **cfg_kw)
    phi = circle_basis(cfg.p, cfg.n_freq)

    train_seqs = data.train
    if objective == "ar" and order == "endpoints_first":
        train_seqs = data.reorder_endpoints_first(train_seqs)

    t0 = time.time()
    params, hist = train(cfg, tc, phi, train_seqs, objective, verbose=verbose)
    wall = time.time() - t0

    res = {"n_params": n_params(params), "final_loss": hist[-1][1], "wall_s": wall}
    for mode in EVAL_MODES:
        if objective == "ar":
            r = ar_eval(params, cfg, phi, data.test, mode, order)
            res[mode] = r if r is not None else "unavailable"
        else:
            res[mode] = bvp_eval(params, cfg, phi, data.test, mode)
    return res


def fmt(cell, key="exact_acc"):
    if cell == "unavailable":
        return "    n/a"
    return f"{100 * cell[key]:6.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.4)
    ap.add_argument("--n-freq", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0)
    args = ap.parse_args()

    tc = TrainConfig(steps=args.steps, batch_size=args.batch, seed=args.seed)
    cfg_kw = dict(p=args.p, seq_len=args.seq_len, d_model=args.d_model,
                  n_freq=args.n_freq)

    # Resume: reload anything already finished so a killed job costs at most
    # one run of compute instead of all of them.
    try:
        with open("results.json") as f:
            out = json.load(f)
        out.setdefault("main", {})
        out.setdefault("control", {})
        print(f"resuming: {len(out['main']) + len(out['control'])} runs on disk",
              flush=True)
    except (FileNotFoundError, json.JSONDecodeError):
        out = {"args": vars(args), "main": {}, "control": {}}

    n_new = 0
    for permuted, bucket, arms in (
        (False, "main", ARMS),
        (True, "control", (("bvp", None),)),
    ):
        if permuted and args.skip_control:
            break
        data = None
        for objective, order in arms:
            for fiber in FIBERS:
                name = f"{arm_name(objective, order)} | {fiber}"
                if name in out[bucket]:
                    continue
                if args.max_runs and n_new >= args.max_runs:
                    break
                if data is None:
                    data = ModArithData(p=args.p, seq_len=args.seq_len,
                                        train_frac=args.train_frac,
                                        permute_vocab=permuted, seed=args.seed)
                    print(f"\n=== {'CONTROL (permuted vocab)' if permuted else 'MAIN'}"
                          f" === {data}", flush=True)
                print(f"  training {name} ...", flush=True)
                r = run_one(data, dict(cfg_kw, fiber=fiber), objective, order, tc)
                out[bucket][name] = r
                n_new += 1
                with open("results.json", "w") as f:      # persist immediately
                    json.dump(out, f, indent=2)
                print(f"    done in {r['wall_s']:.0f}s  loss={r['final_loss']:.4f}"
                      f"  [saved]", flush=True)

    total = len(out["main"]) + len(out["control"])
    expected = len(ARMS) * len(FIBERS) + (0 if args.skip_control else len(FIBERS))
    if total < expected:
        print(f"\n{total}/{expected} runs complete -- rerun to continue")
        return

    # -------------------------------------------------------------- tables
    print("\n\nExact-match accuracy on held-out sequences")
    print("(all queried positions correct; chance = 1/p per token)\n")
    hdr = f"{'model':28s}" + "".join(f"{m:>10s}" for m in EVAL_MODES)
    print(hdr)
    print("-" * len(hdr))
    for name, r in out["main"].items():
        print(f"{name:28s}" + "".join(fmt(r[m]) + "    " for m in EVAL_MODES))

    if not args.skip_control:
        print("\nControl: same task, vocabulary randomly relabelled\n")
        print(hdr)
        print("-" * len(hdr))
        for name, r in out["control"].items():
            print(f"{name:28s}" + "".join(fmt(r[m]) + "    " for m in EVAL_MODES))

    print("\n\nPer-token accuracy (chance = 1.03%)\n")
    print(hdr)
    print("-" * len(hdr))
    for name, r in out["main"].items():
        print(f"{name:28s}" + "".join(fmt(r[m], "token_acc") + "    " for m in EVAL_MODES))
    if not args.skip_control:
        print("  -- control (permuted vocab) --")
        for name, r in out["control"].items():
            print(f"{name:28s}" + "".join(fmt(r[m], "token_acc") + "    " for m in EVAL_MODES))

    print("\n\nHalting certificate (bvp only), infill mode")
    print("AUROC for predicting per-example correctness\n")
    print(f"{'model':28s}{'residual':>10s}{'energy':>10s}{'maxprob':>10s}"
          f"{'acc@1':>9s}{'acc@4':>9s}")
    for bucket in ("main", "control"):
        for name, r in out[bucket].items():
            if not name.startswith("bvp"):
                continue
            c = r["infill"]
            tag = name if bucket == "main" else name + " (perm)"
            print(f"{tag:28s}{c['auroc_residual']:10.3f}{c['auroc_energy']:10.3f}"
                  f"{c['auroc_maxprob']:10.3f}{100*c['acc_iter1']:8.1f}%"
                  f"{100*c['acc_iterT']:8.1f}%")

    print(f"\nwrote results.json   (jax backend: {jax.default_backend()})")


if __name__ == "__main__":
    main()
