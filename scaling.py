"""scaling.py -- does the advantage survive scale, or does scale eat it?

THE STANDARD OBJECTION, and it is a good one. Every architectural-prior result
in the literature has the same failure mode: the prior wins in the small-data,
small-model corner and the gap closes as either axis grows, because a flexible
model eventually learns from data what the rigid one was given for free. If
that happens here, then "1.54x at one third the parameters" is a statement
about a budget I happened to pick, not about braids.

So measure it. Two axes, one at a time, with everything else held fixed:

  DATA   1.5k -> 45k training braids at fixed width
  PARAMS d = 8 -> 64 at fixed data, with the transformer re-matched at each
         point (task_knot MEASURES both models rather than assuming a formula)

The quantity plotted is EXTRAPOLATION R2 -- trained on words of length 4-10,
tested on 12-16 -- because that is where a symmetry can pay at all. In-
distribution numbers are reported alongside, since a prior that only helps
in distribution is just a regulariser.

WHAT WOULD FALSIFY THE CLAIM. If the braid/transformer ratio at extrapolation
declines monotonically along either axis and is heading for 1.0, the prior is
being learned rather than needed, and the honest summary is "helps in the
small-data regime" -- which is a much weaker claim than the README makes. If
the ratio is flat or grows, the prior is buying something data does not supply.

The transformer is `tf-rope` with base 8 throughout. That baseline was
repaired twice (untrained absolute-position rows; a rotary base tuned for long
contexts) and both repairs made it stronger, so it is the strongest baseline
this project has, not a convenient one.
"""

from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import task_knot as K

BASE = dict(
    models=["braid-ybe", "tf-rope"], test_n=2000, lmin=4, lmax=10, heads=4,
    batch=128, lr=2e-3, w_ybe=10.0, seed=0, conj_n=0, w_inv=0.0, w_conj=0.0,
    rope_base=8.0, tie_r=True, pure=False,
)

EXTRA = "EXTRAPOLATION (len 12-16)"
TEST = "test (same lengths)"


def one(train_n, d, steps):
    a = SimpleNamespace(**BASE, train_n=train_n, d=d, steps=steps)
    t0 = time.time()
    r = K.run(a)
    out = {}
    for m in BASE["models"]:
        out[m] = dict(params=r[m]["params"],
                      test=r[m][TEST + " R2"],
                      extra=r[m][EXTRA + " R2"])
    out["_secs"] = round(time.time() - t0)
    return out


def table(title, axis_name, rows):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"{axis_name:>10s} | {'braid par':>9s} {'braid ext':>9s} | "
          f"{'tf par':>9s} {'tf ext':>9s} | {'ratio':>6s} | "
          f"{'braid test':>10s} {'tf test':>8s}")
    for v, o in rows:
        b, t = o["braid-ybe"], o["tf-rope"]
        # a ratio against a baseline at chance is meaningless, not infinite
        rs = (f"{b['extra'] / t['extra']:6.2f}" if t["extra"] > 0.02
              else "   n/a")
        print(f"{v:>10} | {b['params']:9d} {b['extra']:9.3f} | "
              f"{t['params']:9d} {t['extra']:9.3f} | {rs:>6s} | "
              f"{b['test']:10.3f} {t['test']:8.3f}")
    print("\nratio = braid extrapolation R2 / transformer extrapolation R2.")
    print("A ratio trending to 1.0 along the axis means scale is eating the "
          "prior.")


def run(a):
    log = {}
    if a.axis in ("data", "both"):
        rows = []
        for n in a.data_sizes:
            print(f"\n########## DATA {n} ##########", flush=True)
            rows.append((n, one(n, a.d, a.steps)))
            log[f"data{n}"] = rows[-1][1]
            table("SCALING IN DATA (width fixed)", "train n", rows)
    if a.axis in ("params", "both"):
        rows = []
        for d in a.widths:
            print(f"\n########## WIDTH {d} ##########", flush=True)
            rows.append((d, one(a.train_n, d, a.steps)))
            log[f"d{d}"] = rows[-1][1]
            table("SCALING IN PARAMETERS (data fixed)", "d", rows)
    with open(a.out, "w") as f:
        json.dump(log, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", choices=["data", "params", "both"], default="both")
    ap.add_argument("--data-sizes", type=int, nargs="*",
                    default=[1500, 5000, 15000, 45000])
    ap.add_argument("--widths", type=int, nargs="*", default=[8, 16, 32, 64])
    ap.add_argument("--train-n", type=int, default=15000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", default="logs/scaling.json")
    run(ap.parse_args())
