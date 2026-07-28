"""commute_probe.py -- does the model actually respect far commutation?

The braid work's most persuasive result was not a score, it was
`rIII_probe.py`: measure the symmetry DIRECTLY on synthetic pairs that never
appear in training, show it tracks extrapolation, and the mechanism claim stops
being a story about why the number moved. This is the same instrument for
circuits.

THE SYMMETRY. Two-qubit gates acting on disjoint pairs commute exactly:
`G_i G_j = G_j G_i` whenever `|i - j| >= 2`. `circuits.py` verifies this on the
simulator at 0.0e+00. So swapping two adjacent-in-time, disjoint-in-space gates
leaves the circuit -- and therefore every `<Z_i>` -- completely unchanged.

THE MEASUREMENT, and why it needs a control. Generate pairs of circuits
differing by exactly one such swap and measure `||f(A) - f(B)||`. On its own
that number is worthless: a model that has collapsed toward a constant scores
perfectly. So generate CONTROL pairs too -- the same two gates swapped, but
OVERLAPPING (`|i - j| == 1`), which is a genuinely different circuit that the
model should distinguish. The ratio

    mean ||f(A) - f(B)|| over commuting swaps
    -----------------------------------------
    mean ||f(A) - f(B)|| over overlapping swaps

is the measurement. 0 means the model is exactly invariant where it should be
while still telling different circuits apart; 1 means it cannot tell the
difference between a swap that matters and one that does not.

WHAT TO EXPECT, recorded before running. The braid layer should score EXACTLY
zero, not approximately: a gate at `(i, i+1)` reads and writes only those two
slots of the state, so two gates touching disjoint slots commute as operations
on the state whatever the learned maps are. That is a claim about the
architecture rather than about training, which is precisely why it is worth
checking rather than asserting -- masking, padding, or the scan could break it.

The overlapping control also has to be checked for a floor. If overlapping
swaps barely change the true answer, the denominator is small and the ratio is
uninformative; so the probe reports the ground-truth distances for both classes
alongside the model's.
"""

from __future__ import annotations

import argparse

import jax.numpy as jnp
import numpy as np

import circuits as C


def make_pairs(rng, n_qubits, depth, n_pairs, overlapping):
    """Circuit pairs differing by one adjacent transposition of two gates.

    `overlapping=False` swaps gates on disjoint pairs (must not matter);
    `overlapping=True` swaps gates that share a qubit (must matter).
    """
    A, B = [], []
    while len(A) < n_pairs:
        w = C.random_circuit(rng, n_qubits, depth)
        # pick a slot t and choose the two gates' positions to order
        t = int(rng.integers(0, depth - 1))
        i = int(rng.integers(0, n_qubits - 1))
        if overlapping:
            cand = [j for j in (i - 1, i + 1) if 0 <= j <= n_qubits - 2]
        else:
            cand = [j for j in range(n_qubits - 1) if abs(j - i) >= 2]
        if not cand:
            continue
        j = int(rng.choice(cand))
        gi = int(rng.integers(0, len(C.GATES)))
        gj = int(rng.integers(0, len(C.GATES)))
        a = w[:t] + [(i, gi), (j, gj)] + w[t + 2:]
        b = w[:t] + [(j, gj), (i, gi)] + w[t + 2:]
        A.append(a)
        B.append(b)
    return A, B


def truth_gap(A, B, n_qubits):
    """How much the swap changes the REAL answer. Zero for disjoint gates."""
    ya = np.array([C.simulate(w, n_qubits) for w in A])
    yb = np.array([C.simulate(w, n_qubits) for w in B])
    return float(np.mean(np.linalg.norm(ya - yb, axis=-1)))


def probe(fwd, n_qubits, n_max, l_max, depth=10, n_pairs=512, seed=0,
          with_truth=True):
    """Return (commuting distance, overlapping distance, ratio)."""
    rng = np.random.default_rng(seed)
    out = {}
    for tag, ov in (("commuting", False), ("overlapping", True)):
        A, B = make_pairs(rng, n_qubits, depth, n_pairs, ov)
        Xa, Ma = C.encode(A, n_max, l_max)
        Xb, Mb = C.encode(B, n_max, l_max)
        fa = fwd(jnp.asarray(Xa), jnp.asarray(Ma))
        fb = fwd(jnp.asarray(Xb), jnp.asarray(Mb))
        # compare only the live qubits
        d = jnp.linalg.norm((fa - fb)[:, :n_qubits], axis=-1)
        out[tag] = float(jnp.mean(d))
        if with_truth:
            out[tag + "_truth"] = truth_gap(A, B, n_qubits)
    out["ratio"] = out["commuting"] / (out["overlapping"] + 1e-12)
    return out


def _selfcheck(a):
    """Run the probe on the ground truth itself, as a sanity check.

    The simulator must score exactly 0 on commuting swaps and clearly non-zero
    on overlapping ones. If the control class does NOT move the answer, the
    denominator is a floor and every ratio computed here is meaningless -- so
    this check gates the whole file.
    """
    rng = np.random.default_rng(0)
    for tag, ov in (("commuting", False), ("overlapping", True)):
        A, B = make_pairs(rng, a.qubits, a.depth, 256, ov)
        g = truth_gap(A, B, a.qubits)
        print(f"  ground truth, {tag:12s} swaps: mean |dY| = {g:.3e}")
        if not ov and g > 1e-12:
            print("  FAIL: disjoint gates changed the answer")
            return False
        if ov and g < 1e-3:
            print("  FAIL: overlapping swaps barely matter, control is a floor")
            return False
    print("  control is well separated from the invariant class")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", type=int, default=4)
    ap.add_argument("--depth", type=int, default=10)
    args = ap.parse_args()
    print("checking the probe against the simulator before trusting it:")
    print("all checks passed" if _selfcheck(args) else "FAILED")
