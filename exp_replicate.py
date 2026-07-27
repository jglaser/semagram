#!/usr/bin/env python3
"""Experiment 1 -- does the cyclic/tied gap survive multiple seeds?

The single-seed observation was 50.9% (cyclic) vs 5.9% (tied) token accuracy on
infill-only training of the additive task.  That is a large gap but n=1, so it
is not yet a result.  This runs S seeds per fiber and reports mean +/- std.

    python exp_replicate.py --seeds 5 --steps 6000

Runtime: seeds x 3 fibers x ~180 s on CPU.  5 seeds is about 45 minutes.
Use --steps 2000 --seeds 3 for a quick look (~9 min).
"""

from __future__ import annotations

import argparse

from semagram.runner import Store, agg, fmt, run_cell, welch

FIBERS = ("free", "tied", "cyclic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-freq", type=int, default=8)
    ap.add_argument("--out", default="results_replicate.json")
    args = ap.parse_args()

    store = Store(args.out)

    for seed in range(args.seeds):
        for fiber in FIBERS:
            key = f"additive|additive|{fiber}|infill|s{seed}|{args.steps}"
            if store.has(key):
                continue
            print(f"  {key} ...", flush=True)
            r = run_cell(task="additive", basis="additive", fiber=fiber,
                         objective="infill", seed=seed, steps=args.steps,
                         p=args.p, seq_len=args.seq_len, d_model=args.d_model,
                         n_freq=args.n_freq)
            store.put(key, r)
            print(f"    token_acc {100*r['token_acc']:.1f}%  "
                  f"exact {100*r['exact_acc']:.1f}%  "
                  f"loss {r['train_loss']:.3f}  {r['wall_s']:.0f}s", flush=True)

    print(f"\n\nAdditive task, infill-only training, {args.steps} steps, "
          f"{args.seeds} seeds")
    print("token accuracy on held-out sequences (chance = "
          f"{100/args.p:.2f}%)\n")
    print(f"{'fiber':10s}{'token acc':>18s}{'exact acc':>18s}"
          f"{'train loss':>13s}{'n':>4s}")
    print("-" * 63)

    per_fiber = {}
    for fiber in FIBERS:
        rows = store.select(fiber=fiber, task="additive", objective="infill",
                            steps=args.steps)
        per_fiber[fiber] = rows
        print(f"{fiber:10s}{fmt(agg(rows, 'token_acc')):>18s}"
              f"{fmt(agg(rows, 'exact_acc')):>18s}"
              f"{(agg(rows, 'train_loss') or {'mean': float('nan')})['mean']:13.3f}"
              f"{len(rows):4d}")

    print("\nWelch t (rough effect indicator, not a p-value):")
    for a, b in (("cyclic", "tied"), ("cyclic", "free"), ("tied", "free")):
        ta = [r["token_acc"] for r in per_fiber[a]]
        tb = [r["token_acc"] for r in per_fiber[b]]
        t = welch(ta, tb)
        print(f"  {a:7s} vs {b:7s}: t = {t:.2f}" if t is not None
              else f"  {a:7s} vs {b:7s}: t = n/a (need >=2 seeds each)")

    print("\nAlso worth reading off the halting-certificate columns, which were")
    print("at chance (0.50) in the single-seed run:\n")
    print(f"{'fiber':10s}{'AUROC resid':>14s}{'AUROC energy':>14s}"
          f"{'AUROC maxprob':>15s}{'acc@1':>9s}{'acc@4':>9s}")
    for fiber in FIBERS:
        rows = [r for r in per_fiber[fiber] if r.get("auroc_residual") is not None]
        if not rows:
            continue
        def m(k):
            a = agg(rows, k)
            return a["mean"] if a else float("nan")
        print(f"{fiber:10s}{m('auroc_residual'):14.3f}{m('auroc_energy'):14.3f}"
              f"{m('auroc_maxprob'):15.3f}{100*m('acc_iter1'):8.1f}%"
              f"{100*m('acc_iterT'):8.1f}%")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
