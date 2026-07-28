# Braided attention: a symmetry that is worth building in

A sequence layer whose forward pass **is** a braid representation, and a
benchmark where that pays off: predicting the Jones polynomial of a braid
closure from its word. Against a parameter-matched transformer it generalises
**3.4x better** past its training range, and the advantage tracks how closely
the learned maps satisfy the Yang-Baxter equation.

```bash
python -m venv .venv && . .venv/bin/activate
pip install "jax[cpu]" optax numpy scipy scikit-image

python braids.py                                   # verify the invariants
python task_knot.py                                # the benchmark, ~12 min CPU
python task_knot.py --models braid-ybe --w-ybe 10  # the best setting
```

CPU only, no GPU used anywhere.

## The question, and why this is the right test of it

Building a symmetry into an architecture is supposed to beat learning it from
data. [`lessons.md`](lessons.md) is a long record of that failing: an earlier
layer here imposed cyclic-shift equivariance **exactly** -- measured at 1.3e-15
-- on data where the symmetry is literally true, and lost to a transformer with
learned absolute position embeddings that is provably *wrong* about the geometry
(its output moves 71% under a rotation with no geometric content).

The diagnosis was that **cyclic shift is too easy a symmetry to matter**. A
flexible model learns it from data more cheaply than a rigid one imposes it. If
that is right it makes a prediction: find a symmetry with no cheap local
statistic, and building it in should pay.

Braid closures are that symmetry. Two braid words can present the same knot
while looking nothing alike, deciding whether they do has no local shortcut, and
the Jones polynomial is #P-hard in general -- but it is exactly what a
Yang-Baxter R-matrix respects for free.

## The layer

A braid on `s` strands is a word in generators `sigma_1 .. sigma_{s-1}` and
their inverses. So the state is **one vector per strand**, `(batch, s, d)`, and
a letter applies a learned map to the two strands it touches, leaving the rest
alone. Reading the word left to right *is* the braid representation, with
learned matrices where a physicist would put the R-matrix. The closure of a
braid is a trace, so the readout pools over strands.

```python
x = broadcast(p["x0"], (b, s, d))              # one vector per strand
for (i, sign) in word:                         # each letter of the braid
    pair   = x[:, i:i+2]                       # the two strands it touches
    x[:, i:i+2] = tanh(R[token] @ pair.flat)   # a learned 2-strand map
return mlp(mean(x, axis=strands))              # closure = trace = pool
```

The braid relation `R_i R_{i+1} R_i = R_{i+1} R_i R_{i+1}` is exactly the
statement that the layer cannot distinguish two diagrams differing by a
Reidemeister III move. `ybe_residual` measures the violation, and adding it to
the loss with weight `w_ybe` is how the symmetry is imposed -- softly, which
turns out to matter.

## Results

Three seeds, parameter-matched, trained on braid words of length 4-10 and tested
on 12-16.

| model | params | YBE residual | test R2 | extrapolation R2 |
|---|---|---|---|---|
| `braid` | 34.2k | 1.15e-01 | 0.631 +/- 0.010 | 0.128 +/- 0.009 |
| **`braid-ybe`** | 34.2k | **1.66e-02** | **0.681 +/- 0.005** | **0.186 +/- 0.012** |
| `tf` (transformer) | 32.8k | -- | 0.643 +/- 0.008 | 0.055 +/- 0.028 |

Seed-matched extrapolation deltas against the transformer are +0.170, +0.148,
+0.076: positive on every seed, mean **+0.131**, a 3.4x ratio.

**The advantage splits in two and both halves are measured.** `tf` -> `braid` is
+0.073, the architecture matching the generative process. `braid` ->
`braid-ybe` is +0.058, the symmetry itself -- those two are the same network and
differ only by the penalty. So about 44% of the gain is Reidemeister invariance
specifically.

**It is causal, by dose-response.** Varying only `w_ybe`, seed 0:

| `w_ybe` | YBE residual | test R2 | extrapolation R2 |
|---|---|---|---|
| 0 | 1.18e-01 | 0.644 | 0.141 |
| 0.1 | 6.27e-02 | 0.651 | 0.153 |
| 1 | 1.68e-02 | 0.683 | 0.199 |
| 3 | 7.89e-03 | 0.699 | 0.213 |
| **10** | 3.35e-03 | **0.700** | **0.225** |
| 30 | 1.15e-03 | 0.682 | 0.208 |
| 100 | 2.90e-04 | 0.664 | 0.190 |

Up to `w = 10` this is monotone across a 35x range of residual,
`corr(log residual, extrapolation R2) = -0.987`, with nothing else varying. At
the peak it is **7.8x** the transformer.

**Then it turns over, which is the more interesting half.**

```
soft (0.225)  >  exact (0.190)  >  none (0.141)  >>  transformer (0.029)
```

Every level of the symmetry beats having none and all of them beat the
transformer, but **approximate invariance beats exact invariance**. Yang-Baxter
solutions are a measure-zero variety; forcing the learned maps exactly onto it
removes capacity the model needs for fitting.

And that is the sentence the whole project has been circling. The earlier layer
imposed its symmetries *exactly* and they were worth nothing. Exactness was
never the axis. A symmetry has to be **hard enough to be worth having** --
Reidemeister, not cyclic shift -- and **imposed loosely enough to leave capacity
behind**.

## The ground truth is verified two independent ways

Because the previous benchmark's lesson was that the benchmark is usually where
the bug is. `braids.py` computes the Jones polynomial by

- **state sum**: all `2^L` Kauffman resolutions, loops counted by union-find.
  Obviously correct, and 6.5 s per braid at length 16.
- **Temperley-Lieb transfer**: a vector over non-crossing matchings of `2s`
  points (Catalan-many -- verified at 2, 5, 14, 42 for `s` = 2..5), so a word
  becomes a product of small matrices.

**0 mismatches in 150 comparisons, 27712x faster** (0.23 ms at length 16), and
both agree with the published trefoil and figure-eight polynomials.

Three bugs surfaced before any of it was used: `sigma_1^3` closes to the
*left*-handed trefoil under this sign convention (the figure-eight is
amphichiral, so it passed either way and hid this); evaluation points off the
unit circle are unusable, `|V|` reaching 1123 against a median of 1.0; and at
`t = exp(2 pi i/3)` the Jones polynomial has modulus 1 for **every** knot, which
makes it a degenerate regression target.

## Honest limits

- **Absolute extrapolation R2 is 0.225.** Every model here is poor at
  generalising in crossing number; this compares degrees of failure, not a
  solved task.
- **Three to four strands, lengths 4-16, one invariant, one architecture
  family.** The braid-vs-transformer result is replicated over three seeds; the
  **dose-response sweep is single-seed**, and the peak-versus-endpoint
  differences (0.017-0.035) sit against a three-seed spread of 0.012, so the
  turnover is suggestive rather than established.
- **The optimum is one number on one setup.** Whether the best residual is fixed
  or scales with capacity or data is untested, and a fixed optimum would be a
  far stronger claim than this.

## Files

| file | what it is |
|---|---|
| `braids.py` | braid words, knot closures, exact Jones polynomials (two ways) |
| `task_knot.py` | the braided layer, the Yang-Baxter penalty, the transformer baseline |
| `lessons.md` | the two earlier architectures and why they failed |
| `semagram.py`, `loop_layer.py` | the earlier circular-attention layer |
| `contours.py`, `task_shape.py`, `task_cont.py` | the earlier closed-contour benchmark |
| `variational_gap.py`, `ood_masks.py`, `ink_weight.py`, `solver_probe.py` | diagnostics from that work, several of them reusable |

The single most portable thing to come out of the earlier work is
`variational_gap.py`: **minimise your action and see whether the model gets
worse.** If it does, the variational structure is bookkeeping -- a recurrence
that an energy happened to generate, rather than a model that solves a
variational problem.
