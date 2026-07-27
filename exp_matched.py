#!/usr/bin/env python3
"""Experiment 3 -- matched-difficulty AR vs boundary-value.

The first grid compared an AR model doing repeated *addition* against a
bidirectional model doing modular *inversion*, then reported that AR won.  That
was a confound, not a result: the two arms were not solving the same problem.

Here both arms solve the same problem -- recover the interior from the two
endpoints, which requires inverting (L-1) mod the group order:

    ar      causal, sequence reordered to [x_0, x_{L-1}, x_1, ..., x_{L-2}]
            so both boundary conditions precede every interior token
    infill  bidirectional, endpoints clamped, interior masked

Same backbone, same fiber, same data, same step budget.  Now the only difference
is the factorization, which is the thing that was supposed to be under test.

Both arms need to get past memorization, so the default budget is long.  Watch
the printed curves: train loss falling while test accuracy stays at chance is
the pre-grokking regime and means the budget is still too small.

    python exp_matched.py --seeds 3 --steps 40000 --fiber cyclic

Runtime: seeds x 2 arms x (steps/6000 x ~180 s).  40k steps x 3 seeds is
roughly 4 hours on CPU -- run it on a GPU, or start with --steps 12000.
"""

from __future__ import annotations

import argparse

from semagram.runner import Store, agg, fmt, run_cell, welch

ARMS = ("ar", "infill")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--fiber", default="cyclic", choices=["free", "tied", "cyclic"])
    ap.add_argument("--task", default="additive",
                    choices=["additive", "multiplicative"])
    ap.add_argument("--basis", default="additive",
                    choices=["additive", "multiplicative"])
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-freq", type=int, default=8)
    ap.add_argument("--out", default="results_matched.json")
    args = ap.parse_args()

    store = Store(args.out)

    for seed in range(args.seeds):
        for arm in ARMS:
            key = (f"{args.task}|{args.basis}|{args.fiber}|{arm}"
                   f"|s{seed}|{args.steps}")
            if store.has(key):
                continue
            print(f"  {arm:7s} seed {seed} ({args.steps} steps) ...", flush=True)
            r = run_cell(task=args.task, basis=args.basis, fiber=args.fiber,
                         objective=arm, seed=seed, steps=args.steps,
                         p=args.p, seq_len=args.seq_len,
                         d_model=args.d_model, n_freq=args.n_freq)
            store.put(key, r)
            print(f"    token_acc {100*r['token_acc']:.1f}%  "
                  f"exact {100*r['exact_acc']:.1f}%  "
                  f"loss {r['train_loss']:.3f}  {r['wall_s']:.0f}s", flush=True)

    def rows(arm):
        return [r for r in store.data.values()
                if r["objective"] == arm and r["fiber"] == args.fiber
                and r["task"] == args.task and r["steps"] == args.steps]

    print(f"\n\nMatched difficulty: both arms recover the interior from the two")
    print(f"endpoints.  task={args.task} fiber={args.fiber} basis={args.basis}")
    print(f"{args.steps} steps, {args.seeds} seeds, "
          f"chance = {100/args.p:.2f}%\n")
    print(f"{'arm':10s}{'token acc':>18s}{'exact acc':>18s}{'train loss':>13s}{'n':>4s}")
    print("-" * 63)
    for arm in ARMS:
        rs = rows(arm)
        tl = agg(rs, "train_loss")
        print(f"{arm:10s}{fmt(agg(rs, 'token_acc')):>18s}"
              f"{fmt(agg(rs, 'exact_acc')):>18s}"
              f"{(tl or {'mean': float('nan')})['mean']:13.3f}{len(rs):4d}")

    t = welch([r["token_acc"] for r in rows("infill")],
              [r["token_acc"] for r in rows("ar")])
    if t is not None:
        print(f"\nWelch t (infill - ar): {t:+.2f}")

    a_inf, a_ar = agg(rows("infill")), agg(rows("ar"))
    if a_inf and a_ar:
        gap = 100 * (a_inf["mean"] - a_ar["mean"])
        print(f"gap: {gap:+.1f}pp in favour of "
              f"{'boundary-value' if gap > 0 else 'autoregressive'}")
        if max(a_inf["mean"], a_ar["mean"]) < 5 / args.p:
            print("\nWARNING: both arms near chance -- budget too small to")
            print("conclude anything.  Increase --steps and rerun.")

    print("\nLast four points of each training curve (watch for train loss")
    print("falling while test accuracy stays at chance = memorization):\n")
    for arm in ARMS:
        for r in sorted(rows(arm), key=lambda r: r["seed"]):
            tail = ", ".join(f"{s}:{l:.3f}" for s, l in r["curve"][-4:])
            print(f"  {arm:7s} seed {r['seed']}  test {100*r['token_acc']:5.1f}%"
                  f"  [{tail}]")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
