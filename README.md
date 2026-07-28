# Braided attention: a symmetry that is worth building in

A sequence layer whose forward pass **is** a braid representation, and a
benchmark where that pays off: predicting the Jones polynomial of a braid
closure from its word. At **one third** the parameters of a tuned transformer
baseline it generalises **1.54x better** past its training range -- and the gain
is traceable, link by link, to how nearly the learned maps satisfy the braid
relation.

```bash
python -m venv .venv && . .venv/bin/activate
pip install "jax[cpu]" optax numpy scipy scikit-image

python braids.py                                   # verify the invariants
python task_knot.py                                # the benchmark, ~12 min CPU
python task_knot.py --models braid-ybe --w-ybe 10  # the best setting
```

CPU only, no GPU used anywhere.

![Braid words and their closures](docs/fig1_braids.svg)

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

Trained on braid words of length 4-10, tested on 12-16. Transformer baselines
use **rotary** positions with the base tuned to the sequence length; see
"Getting the baseline right" below for why the obvious alternatives are not
fair comparisons.

| model | params | test R2 | extrapolation R2 |
|---|---|---|---|
| `tf-rope` (baseline) | 40.3k | 0.537 +/- 0.010 | 0.174 +/- 0.011 |
| `braid`, no penalty | 34.2k | 0.631 +/- 0.010 | 0.128 +/- 0.009 |
| `braid-ybe`, untied R | 34.2k | 0.681 +/- 0.005 | 0.186 +/- 0.012 |
| **`braid-ybe`, tied R** | 29.4k | **0.760 +/- 0.005** | **0.269 +/- 0.015** |
| **`braid-ybe`, tied R** | **13.4k** | 0.730 | **0.267** |

![In-distribution versus extrapolation](docs/fig3_extrapolation.svg)

Two things to read off this before anything else.

**The architecture on its own loses.** `braid` without the penalty scores 0.128
against the baseline's 0.174. A strand-structured state and a scan over letters
buy nothing by themselves. Everything below is about the penalty.

**Tying R is what makes the penalty mean anything.** With one map per sign
applied at every position -- which is what a braid representation is -- the
model reaches 0.269, and the `d = 32` version does the same at **13.4k
parameters against the baseline's 40.3k**. The two widths agree to 0.002, so
this is tying rather than capacity.

## The mechanism, measured at every link

The obvious worry about any penalty like this is that it is a generic
regulariser wearing a topology costume: push a matrix toward a measure-zero
variety and things often improve for reasons having nothing to do with the
variety. `rIII_probe.py` settles it without retraining anything for the task.

Generate word pairs differing by a real Reidemeister III move --
`sigma_i sigma_{i+1} sigma_i` against `sigma_{i+1} sigma_i sigma_{i+1}`, the
same braid -- and measure how far apart the model's outputs are. Control pairs
use the same three letters in a non-braid order, so they are a *different*
braid; without them a model that had collapsed toward a constant would look
perfectly invariant. The ratio is the measurement.

| model | `w_ybe` | YBE residual | R-III ratio |
|---|---|---|---|
| untied | 0 | 6.54e-02 | 0.912 |
| untied | 1 | 2.03e-02 | 0.834 |
| untied | 10 | 3.68e-03 | 0.723 |
| untied | 100 | 4.23e-04 | 0.626 |
| tied | 0 | 5.94e-02 | 0.795 |
| tied | 1 | 1.73e-02 | 0.652 |
| tied | 10 | 3.74e-03 | 0.543 |
| **tied** | **100** | 4.79e-04 | **0.468** |

At matched residual the tied model is always more invariant -- 0.834 against
0.652 at ~2e-02, 0.723 against 0.543 at ~3.7e-03 -- and it starts lower with no
penalty at all. So tying is the correct formulation, exactly as the algebra
says: untied, `R[sigma_i]` and `R[sigma_{i+1}]` are different matrices never
applied at the same strand pair, and two matrices that each satisfy Yang-Baxter
to 1.3e-15 have a **cross braid-relation residual of 11.15**.

What the algebra did *not* predict is that the untied penalty gains invariance
anyway, 0.912 to 0.626. Constraining each map onto the variety evidently makes
different solutions behave alike enough that the relation approximately holds.
Part generic prior, part genuine invariance -- and tying converts the rest.

**The chain closes end to end:**

![Invariance versus extrapolation](docs/fig4_mechanism.svg)

| | R-III ratio | extrapolation R2 |
|---|---|---|
| `braid`, no penalty | 0.912 | 0.128 |
| untied, `w = 10` | 0.723 | 0.186 |
| tied, `w = 10` | 0.543 | **0.267** |

`corr(R-III ratio, extrapolation R2) = -0.994`, with invariance measured on
synthetic word pairs that never appear in training and have nothing to do with
the Jones polynomial. Enforcement raises invariance; invariance raises
extrapolation.

## The dose-response, and where it turns over

Varying only the penalty weight on the untied model (seed 0 for the sweep,
three seeds at the endpoints):

| `w_ybe` | YBE residual | test R2 | extrapolation R2 |
|---|---|---|---|
| 0 | 1.18e-01 | 0.644 | 0.141 |
| 0.1 | 6.27e-02 | 0.651 | 0.153 |
| 1 | 1.68e-02 | 0.683 | 0.199 |
| 3 | 7.89e-03 | 0.699 | 0.213 |
| **10** | 3.35e-03 | **0.700** | **0.225** |
| 30 | 1.15e-03 | 0.682 | 0.208 |
| 100 | 2.90e-04 | 0.664 | 0.190 |

![Yang-Baxter dose-response](docs/fig2_doseresponse.svg)

Monotone up to `w = 10` across a 35x range of residual, then it turns over.
Three seeds at the endpoints confirm it: `w = 10` gives 0.224 +/- 0.003 against
`w = 100` at 0.180 +/- 0.010, seed-matched +0.035, +0.054, +0.045.

So **approximate invariance beats exact invariance** on the untied model, even
though the probe shows `w = 100` is the *more* invariant setting (ratio 0.626
against 0.723). Yang-Baxter solutions are a measure-zero variety; past a point,
buying more invariance costs more capacity than it returns.

That is the sentence the whole project has been circling. The earlier layer in
[`lessons.md`](lessons.md) imposed its symmetries *exactly* -- shift
equivariance measured at 1.3e-15 -- and they were worth nothing. Exactness was
never the axis. A symmetry has to be **hard enough to be worth having** --
Reidemeister, not cyclic shift -- and **imposed loosely enough to leave capacity
behind**.

## Getting the baseline right

Two baselines had to be discarded first, and both flattered the layer.

**Learned absolute positions are broken at exactly the lengths tested.**
Training words are length 4-10 and extrapolation words 12-16, while `pos` has 16
rows; padded positions are masked out of attention and out of the pool, so rows
`pos[10:16]` receive **exactly zero gradient**. Measured: `1.334e+01` at row 0,
`0.000e+00` at rows 10 through 15. At extrapolation that model reads 37.5% of
its positional signal off random init. An earlier version of this README
reported a 3.4x advantage against it and explained the gap as the transformer
"fitting length-specific features that mislead". That was wrong.

**Removing positions entirely is not the fix either.** `tf-nope` scores 0.417
in distribution against 0.643, because a braid word is order-dependent. Dropping
information is not the same as removing a confound.

**Rotary positions with a tuned base are the fair comparison.** The logit
depends on `i - j`, so unseen absolute indices never arise. But the usual base
of 10000 is tuned for contexts of thousands and over 16 tokens gives total
rotations of 15, 1.5, 0.15, 0.02 radians -- three of four bands nearly static.
Base 8 gives 15, 8.9, 5.3, 3.2, and lifts the baseline from 0.145 to 0.174.

## Does it work away from knots?

`task_perm.py` is the cheapest control: same layer, same tokens, same
train-short/test-long protocol, no topology. The word is a sequence of adjacent
transpositions and the target is the permutation they compose to. The braid
relation is the Coxeter relation, so Yang-Baxter is exactly true here too, while
the target has nothing topological in it.

| model | YBE residual | test | extrapolation |
|---|---|---|---|
| `braid` | 3.27e-02 | **1.000** | **1.000** |
| `braid-ybe` | **1.26e-07** | **1.000** | **1.000** |
| `tf` | -- | 0.702 | 0.018 |

Both braid variants solve it exactly and length-generalise perfectly; the
transformer reaches 0.702 in distribution and then falls **below chance**
(0.042) on longer words, having fitted length-specific features that mislead.

**Read this as weaker than it looks.** The task is an almost perfect
architectural match -- the layer's state *is* the permutation state, so it needs
only to learn "swap these two vectors", after which length-generalisation is
free because the layer is a recurrence. Both variants sit at 100%, so there is
no headroom to separate the symmetry from the architecture, which was the
question worth asking. It confirms that the advantage is not knot-specific; it
does not establish that the SYMMETRY generalises, and the knot benchmark remains
the only measurement where that contribution is isolated.

One detail worth keeping: `braid-ybe` drove its Yang-Baxter residual to 1.26e-07
here, against 1.66e-02 on knots. The learned maps became near-exact braid
representations on their own when the task permitted it.

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
| `task_knot.py` | the braided layer, the Yang-Baxter penalty, the transformer baselines |
| `rIII_probe.py` | measures Reidemeister-III invariance directly, independent of the task |
| `figures.py` | the figures above, as dependency-free SVG |
| `lessons.md` | the two earlier architectures and why they failed |
| `semagram.py`, `loop_layer.py` | the earlier circular-attention layer |
| `contours.py`, `task_shape.py`, `task_cont.py` | the earlier closed-contour benchmark |
| `variational_gap.py`, `ood_masks.py`, `ink_weight.py`, `solver_probe.py` | diagnostics from that work, several of them reusable |

The single most portable thing to come out of the earlier work is
`variational_gap.py`: **minimise your action and see whether the model gets
worse.** If it does, the variational structure is bookkeeping -- a recurrence
that an energy happened to generate, rather than a model that solves a
variational problem.
