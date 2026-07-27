#!/usr/bin/env python3
"""Experiment 2 -- the falsification test.

Claimed mechanism: a cyclic embedding helps because it puts the task's group in
coordinates where the group action is linear.  If that is right, the benefit
must move when the group moves.

    task            group underneath        basis that should win
    additive        (Z_p, +)                additive characters
    multiplicative  (Z_p*, x)               multiplicative characters (disc. log)

So this is a 2 x 3 crossed design and the prediction is an *interaction*, not a
main effect:

    cyclic@additive        wins on additive,       not on multiplicative
    cyclic@multiplicative  wins on multiplicative, not on additive
    tied                   wins on neither

A crossed interaction is much harder to get by accident than a single main
effect.  Outcomes and what they mean:

  interaction present        -> mechanism story supported
  cyclic@additive wins both  -> mechanism story WRONG; the benefit is coming
                                from something else (conditioning, smoothness,
                                effective vocabulary size, ...)
  neither cyclic wins        -> the original single-seed result was noise or
                                specific to the any-order/infill difference

    python exp_crossed.py --seeds 3 --steps 6000

Runtime: seeds x 2 tasks x 3 fibers x ~180 s.  3 seeds is about 55 minutes.
Quick look: --seeds 2 --steps 2000  (~13 min).
"""

from __future__ import annotations

import argparse

from semagram.runner import Store, agg, fmt, run_cell, welch

TASKS = ("additive", "multiplicative")
# (label, fiber, basis)
CELLS = (
    ("tied", "tied", "additive"),                    # basis unused by tied
    ("cyclic@add", "cyclic", "additive"),
    ("cyclic@mult", "cyclic", "multiplicative"),
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-freq", type=int, default=8)
    ap.add_argument("--out", default="results_crossed.json")
    args = ap.parse_args()

    # gcd checks happen inside tasks.build; surface them early and clearly
    from semagram import tasks as T
    for t in TASKS:
        T.build(t, args.p, args.seq_len, 0.4, 0)

    store = Store(args.out)

    for seed in range(args.seeds):
        for task in TASKS:
            for label, fiber, basis in CELLS:
                key = f"{task}|{basis}|{fiber}|infill|s{seed}|{args.steps}"
                if store.has(key):
                    continue
                print(f"  {task:15s} {label:12s} seed {seed} ...", flush=True)
                r = run_cell(task=task, basis=basis, fiber=fiber,
                             objective="infill", seed=seed, steps=args.steps,
                             p=args.p, seq_len=args.seq_len,
                             d_model=args.d_model, n_freq=args.n_freq)
                r["label"] = label
                store.put(key, r)
                print(f"    token_acc {100*r['token_acc']:.1f}%  "
                      f"loss {r['train_loss']:.3f}  {r['wall_s']:.0f}s",
                      flush=True)

    def rows(task, label):
        return [r for r in store.data.values()
                if r["task"] == task and r.get("label") == label
                and r["steps"] == args.steps]

    print(f"\n\nCrossed design, infill-only, {args.steps} steps, "
          f"{args.seeds} seeds")
    print(f"token accuracy on held-out sequences (chance = {100/args.p:.2f}%)\n")
    hdr = f"{'task':16s}" + "".join(f"{l:>19s}" for l, _, _ in CELLS)
    print(hdr)
    print("-" * len(hdr))
    for task in TASKS:
        line = f"{task:16s}"
        for label, _, _ in CELLS:
            line += f"{fmt(agg(rows(task, label))):>19s}"
        print(line)

    print("\nInteraction contrast (token accuracy, percentage points):")
    print("  matched basis minus tied, within each task\n")
    verdict = {}
    for task, matched in (("additive", "cyclic@add"),
                          ("multiplicative", "cyclic@mult")):
        mismatched = "cyclic@mult" if matched == "cyclic@add" else "cyclic@add"
        am, at, ax = (agg(rows(task, matched)), agg(rows(task, "tied")),
                      agg(rows(task, mismatched)))
        if not (am and at and ax):
            print(f"  {task}: incomplete")
            continue
        d_matched = 100 * (am["mean"] - at["mean"])
        d_mismatch = 100 * (ax["mean"] - at["mean"])
        t = welch([r["token_acc"] for r in rows(task, matched)],
                  [r["token_acc"] for r in rows(task, "tied")])
        verdict[task] = (d_matched, d_mismatch)
        print(f"  {task:16s} matched({matched:11s}) {d_matched:+6.1f}pp"
              f"   mismatched({mismatched:11s}) {d_mismatch:+6.1f}pp"
              + (f"   [Welch t {t:.2f}]" if t is not None else ""))

    if len(verdict) == 2:
        ok = all(dm > dx for dm, dx in verdict.values())
        print("\n  ==> matched basis beats mismatched basis in "
              f"{'BOTH' if ok else 'NOT both'} tasks: "
              f"{'interaction present, mechanism supported' if ok else 'mechanism NOT supported'}")

    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
