# Semagram: circular attention as a boundary-value problem

`semagram.py` is a single-file JAX demo of an attention architecture with three
commitments, taken from the logograms in *Arrival*:

1. **Tokens live on a circle.** No BOS, no EOS, no position 0. Attention that
   depends only on angular difference is circulant, hence diagonal in the DFT
   basis and applied in `O(n log n)` by `rfft`. A real spectral multiplier is an
   even kernel, so reflection symmetry of the geometric prior is exact by
   construction rather than learned.
2. **The forward pass is a stationary point, not a stack.** An action `S[X]` is
   defined over the whole loop and the layer solves `dS/dX = 0`, so attention is
   literally `jax.grad(energy)`, the Jacobian is symmetric, and there is no
   separate `V` matrix — value mixing falls out of differentiating log-sum-exp.
3. **Ambiguity is holonomy.** Each edge carries an `SO(2)` transport per
   2-plane; going once around the loop should return you to yourself, and the
   closure residual measures how far a reading is from globally coherent.

```bash
python semagram.py --task bridge      # synthetic BVP unit test, ~1 min
python semagram.py --task text        # headline run + both baselines
python semagram.py --diagnose         # four structural diagnostics
python semagram.py --ablate --seeds 3 # leave-one-out over the structural flags
```

**[Part II](#part-ii-what-the-layer-is-for)** takes the architecture off text and
onto data its three commitments are exactly true of -- real closed contours,
where the origin genuinely is a gauge choice and the loop genuinely must close.
Short version: two of the commitments were not doing what this file claims, and
fixing that is worth reading; the layer still loses to a parameter-matched
transformer, and the transformer that wins is the one that is *wrong* about the
geometry.

## Results

`bridge` works, and for the reason the design predicts. `text` went from *worse
than a unigram predictor* to a real character model, but does **not** match a
parameter-matched bidirectional transformer.

4-char gap infill, 12 free positions, two-sided context, 20k steps, ~0.19M
params (baselines +5.5%). References: unigram 3.309, bigram 2.482.

| model | gap NLL | acc | gap-start NLL |
|---|---|---|---|
| **semagram** | **2.373** | 0.320 | 2.126 |
| bidirectional transformer, same objective | 2.202 | 0.356 | 1.949 |
| causal transformer, left context only | — | — | 1.514 |

- **Uses context, clearly.** 2.373 beats the bigram table (2.482) and the
  masked-arc score 2.959 beats unigram (3.309). The original code did neither.
- **The right-hand boundary condition is worth +0.057 nats**, measured inside
  one model at one position with identical left context (`boundary_masks`) —
  the text analogue of `bridge`'s both-ends vs prefix-only split.
- **It does not match the bidirectional baseline** (+0.171, target was 0.1).
  Reported rather than tuned away.
- The causal column varies objective *and* architecture at once and is not an
  architecture result; read it against the bidir row, which holds the objective
  fixed. Both masked models trail causal by ~0.5 nats, which is the masked
  objective's sample-inefficiency, not the circular geometry.

`bridge` (26-symbol step function, clamp the two adjacent endpoints, solve the
interior): **both-ends 1.000, prefix-only 0.247, gap +0.753**. Nothing in the
prefix reveals the second half, so a causal model is structurally incapable of
it. This is the claim working.

Qualitatively, gap infill went from blanks to English-shaped text:

```
true  And you,[ ][s][i][r]! you are [w][e][l][c]ome. Trave[l][ ][y][o]u far on
pred  And you,[ ][a][o][d]! you are [a][e][ ][c]ome. Trave[ ][ ][y][o]u far on
```

The *endpoint* reconstruction is still mostly blank, and that is the metric
rather than the model: predicting 36 Shakespeare characters from 12 has enormous
conditional entropy and emitting the unigram mode is near CE-optimal there.

## The failure, and what actually caused it

Starting point: validation NLL stuck at **3.42**, worse than a context-free
unigram predictor, decoding to blanks. Six causes, in descending order of how
much they mattered:

| cause | effect |
|---|---|
| readout had no bias and was untied — could not express a constant | gap 3.35 → 2.75 |
| band limit applied to the signal, not the kernel | 1 usable Fourier mode of 25 |
| weight decay on the tied readout | collapse at ~step 5000 |
| nothing in the model broke reflection symmetry | bigram unrepresentable |
| endpoint metric measured itself | unigram mode near-optimal |
| connection initialised trivially | no positional signal at init |

### The band limit was on the signal

`spectral_multiplier` padded modes above `M` with `1e3`. In the preconditioned
descent those modes are multiplied by `1 - eta*g/(g+1) ~ 1 - eta` every sweep, so
after `K` sweeps they are gone. `d_spectrum` perturbs the free arc with white
noise and reports the per-mode transfer: in-band to out-of-band gain ratio
**1.3e6**, leaving **1 usable mode out of 25**. A free position could carry DC
and nothing else — which is precisely a wall of spaces. The spec says band-limit
the *kernel*; the state should keep full bandwidth.

### Nothing broke reflection symmetry — and it was not Q=K tying

The original tied `W_q = W_k`, justified as what makes the gradient
conservative. That justification is false (any scalar has a conservative
gradient). The obvious replacement claim — that tying makes the model
reflection-blind — is *also* false. Tying does make the logit matrix exactly
symmetric, but symmetric logits do not imply an equivariant model. Per 2-plane,

```
W^T R(dth) W = cos(dth) * (W^T W)  +  sin(dth) * (W^T J W)
```

and `W^T J W` is antisymmetric, nonzero, and odd in `dth`, so orientation
survives tying — measured, `tied + flat` has reflection residual 0.19, and the
ablation below shows tied models stay *directed* while still failing.

What was actually wrong: *no term broke the symmetry*. The circulant kernel is
even; `rmsnorm` and the Hopfield term are pointwise; and the content-dependent
connection is covariant — reversing the loop reverses each transport — so it
cannot break reflection however it is trained. With the flat base off, the whole
solve is equivariant to **6e-15** in float64 regardless of `W_q`, `W_k`. It could
not tell `th` from `ht` because there was nothing there to tell them apart with.

The only term that can break it is a connection attached to the **ring** rather
than the content, because that one does not reverse. That is the flat RoPE base,
promoted from "nicer initialisation" to load-bearing.

## Ablation

Leave-one-out from the best config, plus the full `tied × flat` 2×2 because
that is the one place an interaction was plausible. 3 seeds, 4000 steps,
`python semagram.py --ablate --seeds 3`. `refl` is the isolated reflection
residual: **~1e-4 means the model is provably blind to the direction of the
loop**, ~1e-2 and above means it is directed.

| config | gap NLL | gap acc | refl | Δ vs best |
|---|---|---|---|---|
| **best** | **2.625 ± 0.013** | 0.278 | 3.3e-02 | — |
| `soft_clamp` (λ=0.5) | 2.648 ± 0.018 | 0.274 | 4.0e-01 | +0.023 |
| `band_pad=1e3` | 2.828 ± 0.018 | 0.242 | 1.2e-01 | +0.204 |
| `no-flat` | 3.091 ± 0.027 | 0.179 | **1.6e-04** | +0.466 |
| `tied` | 3.121 ± 0.035 | 0.186 | 4.6e-02 | +0.496 |
| `as-shipped` | 3.219 ± 0.191 | 0.162 | 1.1e-02 | +0.594 |
| `tied+no-flat` | 3.220 ± 0.017 | 0.161 | **4.5e-04** | +0.595 |

Reading the table:

- **`tied` and `no-flat` cost about the same (+0.50, +0.47) for different
  reasons**, and the `refl` column separates them. `no-flat` is reflection-blind
  (1.6e-04); `tied` stays firmly **directed** (4.6e-02) and fails anyway. So
  untying was the right call for the reason `old/semagram.py:132` gave — the
  channel map `W_q^T W_k` vs a symmetric `W^T W` — and *not* for the
  symmetry reason this file originally claimed. The two effects are separable
  and roughly additive (`tied+no-flat`, +0.595).
- **No interaction worth the name.** +0.496 and +0.466 individually, +0.595
  together, well short of additive. These are two independent handicaps, not a
  crossing.
- `soft_clamp` is inside noise (+0.023) but drives the closure residual to 0.40,
  i.e. it buys nothing and destroys the holonomy's interpretability.
- `as-shipped` has by far the **largest seed variance** (±0.191): one seed
  reached 2.949 and two collapsed outright. The original configuration was not
  merely bad, it was unstable.

## Stability

The unigram is an **absorbing state** — flat readout, uninformative solved
state, no gradient back out. At `lr=3e-3` the model is bistable about it: the
same seed with byte-identical code reached gap NLL **2.356** in one run and
**3.333** in another, separating between steps 4000 and 5000 through
floating-point nondeterminism alone. That is the reproducibility warning in the
modular-arithmetic section below, in a sharper form: not 4 percentage points,
but the difference between working and not.

The nondeterminism is strictly **cross-process**: the ablation grid ran two
configurations that turned out to be identical (`hard_clamp=True` is already the
default) and they agreed to every printed digit, 2.643/2.643 and 2.611/2.611,
seed for seed. Within one process this is reproducible; across processes, XLA
reduction order is enough to change the outcome. `as-shipped` shows the same
bistability at 4000 steps (±0.191 across seeds: one run at 2.949, two
collapsed), so this is a property of the original design, not something the
fixes introduced.

The default `lr` is therefore `2e-3`, which has not been observed to collapse
and scores better anyway. `main_text` prints a `COLLAPSED` banner rather than
tabulating a degenerate run as an architecture result.

## Result 7: when would the variational principle be relevant?

Every model here saw exactly one conditioning family in training: a single
contiguous occluded arc of length 6..18. An energy defines a *joint*, so
conditioning on an arbitrary subset should be free; a feedforward masked model
has to have seen the mask family. That is the most-cited practical advantage of
the framing, so it is worth testing directly. `ood_masks.py` holds the number of
hidden vertices at 12 and varies only the SHAPE of the conditioning set.

| model | contiguous 12 (in-dist) | two arcs of 6 | scattered 12 | periodic every-4th | contiguous 30 |
|---|---|---|---|---|---|
| `sema-so2` | 3.397 | +0.002 | **+0.091** | +0.013 | +0.117 |
| `sema-su2` | 3.383 | +0.003 | **+0.151** | +0.009 | +0.111 |
| `tf-abs` | 3.172 | -0.009 | **-0.013** | -0.013 | +0.159 |
| `tf-ring` | 3.184 | +0.067 | +0.087 | +0.091 | +0.279 |

The scattered and periodic columns are the diagnostic, and they are *easier*
problems than the one trained on: every hidden vertex sits between two observed
ones. A model that genuinely solves the boundary-value problem should improve
there. `tf-abs` does, 3.172 -> 3.160. Semagram gets **worse**, 3.397 -> 3.488.
On the most favourable conditioning geometry available, the boundary-value
framing moves the wrong way.

That is the fourth predicted advantage to fail, after accuracy on matched priors
(Result 3), constraint composition (Result 4), and the directed-convolution
prohibition being worth 0.002 nats (Result 6). So the question is worth asking
properly: under what conditions *would* a variational forward pass earn its
keep? Three, and the first two are prerequisites rather than benefits.

**1. The forward pass has to actually be `argmin S`.** Result 5 shows it is not:
training backprops through a truncated unroll, the network learns the
truncation, and the energy ends up as the generator of an update rule rather
than an objective. Every downstream property -- the `‖∇S‖` certificate, the CG
backward pass, constraint composition -- is unavailable until this holds, and
the fix is training through a converged fixed point, not a better inference-time
solver.

**2. The energy's minimum has to be what you want.** Even fully converged,
minimising this action costs 0.33 nats against the unroll. Implicit
differentiation is what would make the minimiser good, because it is the only
setup in which the gradient reaches the energy *as* an energy.

**3. The varying part of the inference problem has to be inexpressible as a
mask.** This is where the usual pitch is weakest, and the table above is why.
"Condition on an arbitrary subset of the same variables" is precisely what
masked training already teaches, which is why `tf-abs` shrugs off mask shapes it
never saw. It is not a moat.

What has no feedforward analogue, and would justify the machinery:

- **Composing independently-trained energies.** `S_1 + S_2` from two separately
  trained models is a valid joint; two masked transformers cannot be added. This
  is the actual EBM superpower -- product of experts, compositional generation
  -- and it requires a converged solve.
- **Constraints with no mask expression at all**: hard physical laws, symbolic
  conditions, conservation statements, anything not of the form "observe these
  variables".
- **Instance-adaptive compute**, spending solver iterations where a particular
  input is hard.

Multimodality can be struck off the list for this energy specifically: L-BFGS
from two very different starts reached the same minimum to seven digits, so
there are no basins here for "ambiguity is holonomy" to bifurcate between.

The portable result is a diagnostic, and it is cheap: **minimise your action
properly and see whether the model gets better or worse.** If worse, the
variational structure is bookkeeping -- a recurrence that an energy happened to
generate, rather than a model that solves a variational problem. Running that
one test first would have saved this architecture a great deal of theory.

## What was tried and did not work

Each was a plausible mechanism, measured and dropped. The negative results were
most of the work.

| hypothesis | measured |
|---|---|
| Q=K tying makes the model reflection-blind | residual 0.19 — directed; refuted |
| gradient clipping fixes the collapse | 4.415 vs 2.639, and collapses *sooner* |
| the solver contracts too hard, wasting the unroll | `g_floor` 0.1: 2.642 vs 2.584 |
| the unroll needs more depth | `k_steps` 16: 2.683 vs 2.584 |
| hard clamping blocks contextualisation | soft clamp 2.580 vs 2.560; catastrophic above `lam~1` |
| the peaked tied readout needs scaling down | collapses to unigram immediately |
| a saturating gauge phase causes the collapse | neutral across `phi_dev` 0.0–1.0 |

The last row deserves its own note: `phi_dev=0.0` removes the content-dependent
connection *entirely* and costs nothing (2.472 vs 2.462 at `1.0`). **The
"ambiguity is holonomy" mechanism earns nothing measurable on this task**; the
fixed flat base does the work. It is kept, bounded at 0.1, because it keeps the
winding numbers interpretable at no measured cost — not because it was shown to
help.

## Diagnostics

`python semagram.py --diagnose` runs four checks against untrained models in
four configurations. Each targets one structural claim and each can refute it.

- `d_spectrum` — per-mode transfer of the solver on the free arc.
- `d_reflection` — is the solve equivariant to reversing the loop? Reported
  twice: as-is (nonzero even when blind, because a connection with holonomy has
  a branch cut at the gauge origin) and with the content phase zeroed, which
  isolates the structural question.
- `d_clamp_echo` — can a given token survive the solve, and does it ever become
  contextual?
- `d_residual` — stationarity residual per sweep.

## Honest limits

- **The solver is not a solver.** `SOLVER="descent"` is a truncated,
  preconditioned unroll whose iteration matrix has spectral radius > 1 for every
  step size, because `-logsumexp` of a quadratic form supplies genuine negative
  curvature. It is a weight-tied K-sweep network parameterised by an energy —
  not a DEQ, and implicit differentiation would be unsound for it.
- **The energy formulation forces a tied FFN.** A conservative gradient needs a
  symmetric Hessian, so the Hopfield term's `w2` must equal `w1^T`. That is a
  real expressivity cost of the framing, not an implementation choice.
- Headline numbers are single-seed; the ablation uses 3 seeds.
- `n=48`, `d=160`, 20k steps. Everything here is small.



---

# Part II: what the layer is for

Everything above is a report on a language model that does not work as well as a
transformer. That result is correct and it is also the wrong question, because
the three commitments are **priors**, and text satisfies none of them. A
48-character window of Shakespeare has a first character and a last character
and they are not neighbours. Imposing a circle on it costs accuracy, and the
0.171 nats are the bill.

So Part II asks what data the priors are true of, and measures the layer there
instead. Files: `contours.py` (data), `loop_layer.py` (the layer),
`task_shape.py` (the benchmark), `figure.py` (pictures).

```bash
python contours.py                                  # dataset statistics
python task_shape.py --dataset mnist --seeds 3      # the benchmark
python task_shape.py --report                       # tables
python figure.py --out shapes.svg                   # completions, drawn
```

## The data the priors are literally true of

A closed planar curve, resampled to `n` points of **equal arc length**. Then:

- It **is** a function on `Z_n`. Where you start tracing a closed curve is a
  gauge choice with no geometric content, so cyclic shift is an exact symmetry
  of the data. Commitment 1 stops being a modelling convenience and becomes a
  fact about coastlines.
- Every edge has the same length, so the whole shape is carried by the
  **turning angle** at each vertex, which is an element of `SO(2)`. Commitment
  3's per-edge transport is not a mechanism imposed on the data; it is the
  data.
- Going around the loop must return you to yourself, and does so twice over,
  exactly, for every simple closed curve:

  ```
  sum_i d_i          = 2*pi      Hopf's Umlaufsatz -- the tangent winds once
  sum_i exp(i Phi_i) = 0         the curve actually joins up
  ```

  The first identity **is** the holonomy that `semagram.holonomy` computes and
  that Part I found "earns nothing measurable". On text there was no such
  identity for it to earn anything with.

Four real sources, no synthetic shapes. Turning angles are tokenised into 32
equal-frequency bins, which pins the unigram reference at exactly `log 32 =
3.466` for every dataset.

| dataset | what it is | contours (train/test) | Markov two-sided infill | closure of truth | after tokenising |
|---|---|---|---|---|---|
| `mnist` | handwritten digit outlines | 59994 / 9997 | 3.445 | 0.0041 | 0.0694 |
| `fashion` | garment silhouettes | 57920 / 9678 | 3.438 | 0.0105 | 0.1073 |
| `ne_lakes` | Natural Earth lake shorelines | 1242 / 220 | -- | 0.0111 | 0.0756 |
| `ne_admin1` | Natural Earth admin-1 boundaries | 6217 / 1098 | -- | 0.0085 | 0.0792 |

**The baseline column is the one to argue about, and getting it wrong was the
easiest mistake available here.** The obvious reference is a next-token bigram
(3.298 on `mnist`), and it is the wrong one: it scores every position with the
entire left context observed, which is not this task. The task hides a
contiguous run of 12 vertices between two observed ones, so the right classical
answer is the posterior of a first-order chain conditioned on *both* ends,

```
p(x_k) ~ (P^k)[a, :] * (P^(L+1-k))[:, b]
```

exact by forward-backward, computed in `markov_infill`. That is 3.445, not
3.298. The difference matters enormously: against 3.298 the layer looks like it
fails to use context, and against 3.445 it does not. A 12-vertex hole simply has
very little pairwise information in it -- the pairwise-optimal two-sided
predictor beats the unigram by only 0.021 nats -- so the headroom this benchmark
is fighting over is genuinely narrow, and any claim in it needs the right
denominator.

Two things in that table matter later. `sum(d)/2pi` is `1.000000` on every
dataset, so the closure identities hold in the data to the precision they are
measured at. And **tokenising costs an order of magnitude of closure** --
0.0041 becomes 0.0694 on `mnist` -- so 0.069 is a floor no model can beat, and
the constraint experiment has to be read against it rather than against zero.

Anti-aliasing is not optional here and is done identically for every model.
A coastline is rough at every scale; dropping 48 equally spaced points on one
without a prefilter aliases that roughness straight into the turning angles.
Measured before the fix, `|d|` had mean 0.44 rad against a smooth-curve
expectation of `2*pi/48 = 0.13`, and a first-order Markov table beat the unigram
by 0.15 nats -- a sampled fractal, on which a benchmark would have measured
nothing. Curves are therefore low-passed to the Nyquist limit `n/2` as complex
signals `z = x + iy` before sampling, which is the standard
elliptic-Fourier-descriptor treatment.

Even after that, the two geographic sets stay close to noise: `ne_admin1` keeps
24% of its turning-angle power in the top band and has lag-1 autocorrelation
0.04, against 0.37 in the lowest band and 0.45 for `mnist`. Coastlines are
fractal, which is a fact about coastlines. The headline runs use `mnist` and
`fashion`, and that choice is made on measured predictability rather than on
results.

## Result 1: the layer was not on a circle

This one is structural, it was found by a diagnostic rather than a benchmark,
and it invalidates a claim in Part I.

`cumulative_phase` trivialises the connection by declaring node 0 to carry the
identity. If the holonomy `H` is not the identity, that declaration is a
**branch cut sitting at index 0**: for a pair `(i, j)` straddling it the
relative transport is conjugated by `H`, and for a pair on the same side it is
not. Cyclically shifting the loop moves the cut, so the model's output changes
-- and "there is no index 0" is commitment 1, the headline claim.

Part I already names this branch cut, in `d_reflection`, and treats it as a
nuisance specific to the reflection diagnostic. It is not specific to it. The
same cut breaks **translation** equivariance, which is the commitment the whole
architecture is built around.

Measured in float64 on an untrained model: the relative change in output logits
under a cyclic shift of the loop is proportional to the holonomy, and vanishes
to float64 epsilon exactly when the holonomy does.

| gauge | `phi_dev` | shift-equivariance error | holonomy |
|---|---|---|---|
| `so2` | 0.00 | **2.7e-15** | 3.6e-15 |
| `so2` | 0.10 | 2.0e-05 | 1.1e-04 |
| `so2` | 1.00 | 2.0e-04 | 1.1e-03 |
| `su2` | 0.00 | **1.7e-15** | 3.9e-08 |
| `su2` | 0.10 | 1.1e-03 | 1.5e-02 |
| `su2` | 1.00 | 1.1e-02 | 1.5e-01 |

The reason no amount of care fixes this is topology, not implementation. **A
connection on a circle is classified up to gauge by exactly one invariant, its
holonomy**; everything else about it is pure gauge. Equivalently, parallel
transport is single-valued around the loop if and only if the holonomy is
trivial. So the two things Semagram wants,

```
"ambiguity is holonomy"   wants the holonomy NON-trivial
"there is no index 0"     requires the holonomy trivial
```

are in direct conflict, and the layer as shipped was quietly paying for
commitment 3 with commitment 1.

The resolution is to stop making the transport carry both. The holonomy is
still computed from the edge generators, and is still the loss term and the
coherence readout it was meant to be; the transport handed to attention is the
pure-gauge part, `Q_i = B_i G_i` with `B_i` the flat base at integer winding
and `G_i` a content-dependent rotation, which is exactly periodic and therefore
exactly equivariant. `gauge_close=False` reproduces the old behaviour.

| model | shift-equivariance | holonomy still measured |
|---|---|---|
| `so2`, as shipped | 1.97e-04 | 1.09e-03 |
| `so2`, closed | **1.28e-15** | 1.09e-03 |
| `su2`, as shipped | 1.11e-02 | 1.49e-01 |
| `su2`, closed | **1.30e-15** | 1.49e-01 |
| transformer, learned absolute positions | 7.7e-01 | -- |
| transformer, rotary on the ring | 4.2e-15 | -- |

This reframes both holonomy knobs, and Part I's verdict on them is right about
accuracy and wrong about cost.

`phi_dev` is what *creates* a non-trivial holonomy. Part I measures it as
neutral -- "0.0 -> 2.472, 1.0 -> 2.462", all within seed noise -- and keeps it
"at no measured cost". The cost was real and simply never measured: at
`phi_dev=1.0` the layer is 2e-04 away from equivariant in `so2` and 1e-02 in
`su2`, against zero at `phi_dev=0`. It was buying nothing with commitment 3 and
paying for it with commitment 1.

`w_holo` is what *shrinks* the holonomy again, and is therefore the only thing
that was holding the model near the circle it claims to live on. Part I keeps
it at 0.2 as a loop-closure penalty for interpretability. It was load-bearing
for a different reason than the one given.

With the connection split, both knobs become honest: `phi_dev` buys
content-dependent transport, `w_holo` shapes the coherence readout, and neither
can spend the equivariance any more.

The last two rows are the honest framing of the equivariance claim. A rotary
transformer on the ring is *also* exactly equivariant. Circularity has to be
built in, and there is more than one way to build it in; what Semagram uniquely
has is Result 3.


## Using the layer

The reusable object is `loop_layer.solve`: values on a ring, a mask saying which
are given, and it returns the stationary state. `extra` is any scalar function
of that state, added to the action at solve time.

```python
import dataclasses, jax.numpy as jnp, loop_layer as L

cfg = L.LoopCfg(n=48, d=64, vocab=32, gauge="su2")   # so2 | su2 | none
p   = L.init(key, cfg)

x = L.solve(p, values, clamp, cfg)                   # (batch, n, d)

def must_close(x):                                   # a constraint as energy
    d   = jax.nn.softmax(L.logits_of(p, x, cfg), -1) @ centres
    phi = jnp.cumsum(d, -1) - d[..., :1]
    return 0.3 * jnp.sum(jnp.cos(phi).sum(-1)**2 + jnp.sin(phi).sum(-1)**2)

x = L.solve(p, values, clamp, cfg, extra=must_close)  # no retraining
```

Read Result 4 before relying on that last line. Keep `k_steps` at its trained
value and the constraint weight below ~1; outside that band the solve diverges,
and the reason is architectural rather than a tuning failure.

Two properties are worth knowing:

- **The origin is free.** With `gauge_close=True` the layer is equivariant to
  cyclic shift to machine precision, so nothing has to decide where the loop
  starts. Measured here, that is worth nothing in accuracy -- a transformer with
  absolute positions beat it -- so reach for this when you need the invariance
  itself, not when you want a better score.
- **It is not a converged fixed point.** `SOLVER STATUS` in Part I still
  applies, and Result 4 quantifies it: four extra sweeps cost 0.385 nats.

## Result 2: non-abelian holonomy, and what it does not buy

`semagram.py` transports by `SO(2)^(d/2)`. Those commute, so the holonomy of a
loop is the **sum** of the edge phases and nothing else: two loops carrying the
same multiset of phases in different orders are indistinguishable to it. For an
architecture whose claim is that going around the loop *in order* is what
produces meaning, an order-blind holonomy is a strange thing to have.

`gauge="su2"` transports by `SU(2)^(d/4)` -- unit quaternions acting on
`R^4 = C^2` by left multiplication -- composed as a **path-ordered** product via
`lax.associative_scan`, so it stays `O(n log n)`, the same budget the circulant
attention is justified by. Left multiplication by a unit quaternion is
orthogonal, so attention logits still depend only on the relative transport
between two positions, which is the property RoPE is built on; what no longer
holds is that the relative transport depends only on `i - j`. `SO(2)` is
recovered exactly when every edge turns about one axis, so `gauge` is a clean
ablation rather than a different model.

The machinery is correct to machine precision -- associativity 1.8e-15, unit
norm 1.1e-16, orthogonality of the action on `R^4` 4.4e-16, the scan against a
naive loop 2.2e-16 -- and it has the property it was built for: permuting the
edges of a loop leaves the abelian holonomy **exactly** unchanged by
construction, and moves the `su2` holonomy by 1.09 radians.

The prediction was that this should matter *here*, because a closed curve's
edges compose to the identity in `SE(2)`, which is not abelian; the part of that
condition which is a plain sum, `sum d_i = 2*pi`, is all an abelian holonomy can
express, and the part that says the curve joins up is exactly the part that does
not commute.

**The prediction is refuted.** On `mnist` at three seeds, `su2` scores
3.388 +/- 0.017 against `so2` at 3.397 +/- 0.013 -- inside seed noise -- and on
`fashion` the order reverses (3.455 against 3.448). Closure is no better either
(0.2274 against 0.2258). The non-abelian gauge earns nothing measurable on this
task, which is the same verdict Part I reached for the abelian one, now reached
for the more expressive version on data that actually has the structure.

One `su2`-specific bug is worth recording because it would silently sink any
reimplementation: the holonomy angle was `2*arccos|Re H|`, whose derivative is
infinite at the identity -- which is exactly where an initialised model sits --
so `su2` produced `nan` on the first optimiser step. `2*atan2(|Im|, |Re|)` is
the same angle and is smooth there.

## Result 3: the benchmark, and it is a loss

Occluded-arc completion, 12 of 48 vertices hidden at a uniformly random
position, identical evaluation masks for every model, parameter-matched. `mnist`
at three seeds, `fashion` at one. `n=48`, `d=64`, `vocab=32`, 3000 steps.

**mnist**

| model | NLL | acc | closure | shift-equivariance | params |
|---|---|---|---|---|---|
| unigram | 3.466 | -- | -- | -- | -- |
| Markov two-sided infill | 3.445 | -- | -- | -- | -- |
| `sema-so2` | 3.397 +/- 0.013 | 0.057 | 0.2258 | **5.4e-07** | 32k |
| `sema-su2` | 3.388 +/- 0.017 | 0.059 | 0.2274 | **5.6e-07** | 32k |
| **`tf-abs`** | **3.167 +/- 0.008** | 0.092 | 0.2071 | 7.1e-01 | 29k |
| `tf-ring` | 3.186 +/- 0.011 | 0.091 | **0.1942** | **2.8e-07** | 27k |

**fashion** (single seed)

| model | NLL | closure |
|---|---|---|
| Markov two-sided infill | 3.438 | -- |
| `sema-so2` | 3.448 | 0.2514 |
| `sema-su2` | 3.455 | 0.2563 |
| **`tf-abs`** | **3.264** | 0.2478 |
| `tf-ring` | 3.342 | 0.2380 |

Three things, in descending order of how much they should change your mind.

**The layer loses, on the data its priors are exactly true of.** 0.23 nats
behind on `mnist`, 0.18 behind on `fashion`, at matched parameters, matched
steps, matched masks. This is the same verdict as Part I's Shakespeare result,
reached after removing the excuse that Part I offered for it. The priors being
true of the data was the hypothesis; it is not sufficient.

**Being right about the symmetry buys nothing.** `tf-abs` has learned absolute
position embeddings, which are meaningless on a closed curve -- its output
changes by 71% under a cyclic shift of an input whose shift is physically
meaningless. It still **beats** `tf-ring`, which is exactly equivariant
(2.8e-07), on both datasets. So on this task the correct symmetry is not worth
having, and the model that is provably wrong about the geometry wins anyway.
That is the most uncomfortable number here and it is not one the architecture
can be defended against: it says the symmetry argument, which is the entire
motivation for commitment 1, does not cash out.

**The closure story inverts.** `tf-ring` produces the curves that close best
(0.1942), better than either Semagram variant (0.2258, 0.2274), despite having
no holonomy, no closure penalty, and no notion that a curve should close. The
architecture built around loop closure is worse at loop closure than a
transformer that has never heard of it.

The one place the boundary-value framing does show up is in how the models
degrade as the hole grows:

| occluded arc (of 48) | 6 | 12 | 18 | 24 |
|---|---|---|---|---|
| `sema-so2` | 3.364 | 3.397 | 3.416 | 3.458 |
| `tf-abs` | 3.123 | 3.167 | 3.222 | 3.270 |
| `tf-ring` | 3.136 | 3.186 | 3.257 | 3.352 |
| gap to `tf-ring` | 0.228 | 0.211 | 0.159 | **0.106** |

Semagram degrades most slowly of the three -- 0.094 nats from a 6-vertex hole to
a 24-vertex one, against 0.147 for `tf-abs` and 0.216 for `tf-ring` -- so its
deficit more than halves as the problem becomes genuinely two-sided. On
`fashion` the same trend appears (gap to `tf-ring` 0.145 -> 0.077). It is a real
trend in the direction the architecture predicts, and it is not close to
overturning the ranking at any hole size tested. Reported because extrapolating
it would be exactly the kind of thing this repo exists not to do.

## Result 4: constraints after training -- why the energy framing does not cash out

This is the capability the energy formulation is *for*, and the one thing no
transformer can imitate. The forward pass is `argmin_X S[X]`, so a constraint
discovered after training should be one more term in `S`, solved by the same
sweeps, nothing retrained. A closed curve supplies the ideal test case: a
constraint that is global, exact, and impossible for a token-wise decoder to
enforce, `sum_i exp(i Phi_i) = 0`.

It does not work. The effect is real but tiny where it exists, and the reason it
cannot be pushed further is structural.

**At the trained operating point it does the right thing, weakly.** `eta=0.8`,
`K=8`, `mnist`, `sema-so2` seed 0. The energy term *is* the soft closure, so
that is the quantity it should reduce:

| `w` | NLL | soft closure | argmax closure |
|---|---|---|---|
| 0 | 3.385 | 0.1754 | 0.2207 |
| 0.1 | 3.384 | 0.1733 | 0.2251 |
| **0.3** | **3.385** | **0.1697** | 0.2246 |
| 1 | 3.390 | 0.1722 | 0.2304 |
| 3 | 3.458 | 0.2304 | 0.2535 |

At `w=0.3` the constraint reduces what it penalises by 3.2% for 0.000 nats.
Above `w~1` the solve diverges.

**The gain never reaches the drawn shape.** The argmax closure moves the wrong
way throughout. The constraint shifts the *mean* of a 32-way categorical, and a
small shift in a mean does not move a mode, so the polygon actually drawn is no
better. That is a tokenisation problem and the clearest argument for a
continuous-valued head.

**And there is no room to push, because the forward pass is not a solve.** The
obvious response -- run the solver longer -- is unavailable. With *no constraint
at all*, changing only the sweep schedule of an already-trained model:

| `eta` | `K` | NLL | argmax closure |
|---|---|---|---|
| 0.8 | **8** (as trained) | **3.385** | 0.2207 |
| 0.8 | 12 | 3.770 | 0.2719 |
| 0.8 | 16 | 5.076 | 0.2753 |
| 0.8 | 32 | 6.700 | 0.2657 |
| 0.1 | 100 | 4.295 | 0.2997 |

Four extra sweeps cost 0.385 nats. Twenty-four cost 3.3. The state is only
meaningful at exactly the `K` it was trained at, and adding a term to `S`
changes what the sweeps descend on, so there is no budget to absorb it.

*The first version of this section blamed that on negative curvature making the
stationary point unreachable. That explanation is wrong, and Result 5 is the
measurement that refutes it. The stationary point is perfectly reachable. It is
just not where the model lives.*

Hence one-shot at a useful weight is a kick and not a descent (`w=30` sends NLL
3.385 -> 11.7 and makes closure *worse*), continuation does not rescue it, and
lowering `eta` to buy stability costs more than the constraint gains because the
`w=0` control has already fallen apart. The table printed by
`task_shape.py --report` uses a deliberately over-large sweep budget and shows
the failure at full size (NLL 3.4 -> 15.5).

**"Compose constraints at inference" is a property of energy-based layers in
general and not of this one, and the blocker is the same negative curvature that
stops the solve converging.** Two things would change it: a continuous output
head, so the constraint acts on geometry rather than on a categorical's mean;
and an energy whose stationary point is actually reachable, which the
`-logsumexp` attention term rules out by construction.

## Result 5: does the solver matter? Yes -- a better one makes it worse

The natural response to Result 4 is that the solver is the problem: swap the
fixed-step preconditioned unroll for something that actually converges -- CCCP,
or L-BFGS, or Newton-CG exploiting the symmetric Hessian -- and you get a
convergence certificate, `‖∇S‖` as a number that says the forward pass
succeeded, and implicit differentiation via conjugate gradients instead of
GMRES-and-hope. All of that is correct in principle. It was measured. `mnist`,
`sema-so2` seed 0, float64, 8 test contours, minimising the *same* action over
the free coordinates with the clamped ones held at `emb[t]`.

**The action has a genuine minimum and a real solver finds it easily.**

| solve | `S` | `‖∇S‖` on free coords | NLL |
|---|---|---|---|
| unroll, `K=8` (as trained) | -44074.90 | 136.75 | **3.3880** |
| L-BFGS from the unroll (121 it) | **-49306.05** | 0.0028 | 3.7172 |
| L-BFGS from the prior init (137 it) | **-49306.05** | 0.0004 | 3.7172 |

Two very different starts reach the same value to seven digits, with the
gradient driven to `3e-3`. The Hessian on the free coordinates is positive
definite at that point -- eigenvalues `+5.69` to `+2304`, none negative -- so it
is a strong local minimum and, from the evidence of two starts, the only one
worth reaching. Nothing about this landscape is pathological, and CG would work
on it exactly as advertised.

**The unroll is not descending on it, and never was.**

| `K` | 1 | 2 | 4 | **8** | 12 | 16 | 32 | 64 | L-BFGS |
|---|---|---|---|---|---|---|---|---|---|
| `S` | -42110 | -46354 | -41616 | **-44075** | -46404 | -46875 | -44095 | -44932 | **-49306** |
| `‖∇S‖` | 177 | 115 | 180 | **137** | 107 | 158 | 172 | 161 | **0.003** |
| NLL | 9.231 | 3.872 | 4.412 | **3.388** | 3.905 | 4.931 | 6.706 | 6.614 | 3.717 |

`S` oscillates in a band and the gradient norm never leaves 107-180. This is not
slow convergence, it is not divergence either -- it is a cycle. The sweeps are
not a descent on the action in any regime.

**And the true minimiser is worse at the task than the unroll's output**: 3.7172
against 3.3880, a third of a nat. The NLL row has its minimum at exactly `K=8`,
which is exactly the `K` the model was trained with, and degrades in both
directions.

So the answer to "does the solver matter" is yes, decisively, and in the
opposite of the hoped direction. **The trained model is not an approximation to
the minimiser of its own action.** Training shaped a weight-tied 8-step
recurrence; the energy is the thing that generated the recurrence, not an
objective the network is trying to reach. Switching to CCCP or Newton-CG buys
every guarantee on the list -- and delivers you, with a certificate, to a point
the task does not want. The certificate would be honest and the model would be
worse.

**The CCCP argument has a specific gap.** It needs the non-quadratic part to be
concave in the variable being solved for. That holds for `E(ξ) = -(1/β) log Σ_j
exp(β ⟨ξ, k_j⟩)` with the keys **fixed** -- Hopfield retrieval, or cross
attention -- where `E` is a concave function of `ξ` and `attention = -∇E` is
exact. It does not hold for self-attention, where the same state supplies both
queries and keys, so the argument of the log-sum-exp is *quadratic* in `x` and
concavity is lost in the composition. Measured on the free coordinates here, the
attention-plus-Hopfield term alone has **98.4% positive eigenvalues** (range
`-0.425` to `+3.441`). It is not concave, the concave-convex splitting does not
apply, and CCCP's monotone-decrease guarantee does not hold for this action.

A related correction to Part I, which the same measurement settles: tying
`V = K` is required to make the *standard attention formula* a gradient field,
and that derivation is right. It is not required here, because the layer is not
the standard formula -- it is `jax.grad` of a scalar, so its Jacobian is a
Hessian and therefore symmetric for any `W_q`, `W_k`. The two statements answer
different questions and do not conflict. Part I's ablation is the empirical half
of it: tying costs `+0.496` nats.

## Result 6: the energy framing forbids a directed convolution -- and it is cheap

The sharpest argument against conservativity is two lines and it is correct. A
quadratic form only ever sees the symmetric part of its operator,

```
S = 0.5 x^T A x   =>   grad S = 0.5 (A + A^T) x
```

and for a real circulant `A` the transpose conjugates the symbol, so the
symmetrised multiplier is `Re g(m)`: real symbol, even kernel,
reflection-symmetric. Verified in this code path -- `‖grad S - circ(Re g) x‖`
is **2e-16** relative, and for a generic kernel the odd part the gradient throws
away is **65%** of its norm. **A scalar action cannot produce a directed
convolution.** Commitment 1's `O(n log n)` prior is permanently even: a free
model can learn "look three to the left" and this one structurally cannot.

Two questions follow, and they have different answers.

**Does that make the layer undirected? No.** Direction does not enter Semagram
through the convolution; it enters through the connection, and that survives
inside a scalar action. The gauge rotates `q` and `k` by the cumulative edge
phase, and per 2-plane

```
W^T R(dth) W = cos(dth) * (W^T W)  +  sin(dth) * (W^T J W)
```

whose second term is antisymmetric, nonzero, and **odd in dth**. No
symmetrisation removes it, because the gradient of a log-sum-exp is a softmax
and not the symmetric part of anything -- the attention term is not a quadratic
form. Part I's ablation is the evidence: dropping the flat base costs +0.466
nats and sends the reflection residual to 1.6e-04 (provably blind), while
keeping it leaves the model directed at 3.3e-02 *and* exactly a gradient field.

**Is the prohibition expensive? Measured: no.** `odd_apply` adds the forbidden
antisymmetric circulant -- purely imaginary symbol, DC and Nyquist zeroed --
straight to the update field, bypassing `jax.grad`, so the layer deliberately
stops being a gradient field. `mnist`, three seeds, +832 params on 31906.

| model | NLL | seed-matched delta |
|---|---|---|
| `sema-so2` (conservative) | 3.3970 +/- 0.0131 | -- |
| `sema-odd` (+ directed convolution) | 3.3953 +/- 0.0130 | **-0.002, -0.001, -0.002** |

The effect is `-0.0017` nats against a seed spread of `0.0131` -- the noise is
**7.9x** the signal. And the model was not prevented from using the term: `g_odd`
is initialised at zero and grows on its own to `‖g_odd‖_rms ~ 0.028` against
`‖g_even‖_rms ~ 1.66`, i.e. it takes about **1.7%** as much directed kernel as
even kernel, spread evenly over modes 1-12, and gets 0.002 nats for it.

So conservativity forbids exactly what the argument says it forbids, the model
would use a little more if allowed, and it is worth nothing here. The honest
scope: this is one task, on data where orientation matters less than it does in
language. But Part I's evidence points the same way even on text -- direction
there is worth +0.466 nats and it comes from the flat connection, not from the
convolution. On the evidence available, the even-kernel constraint is not where
this architecture is losing.

## What was tried and did not work

Same convention as Part I: each was a plausible mechanism, measured, and either
dropped or replaced.

| hypothesis | measured |
|---|---|
| the priors being true of the data is enough to win | 0.23 nats behind a parameter-matched transformer on `mnist` |
| exact circular equivariance helps on circular data | `tf-abs` (equivariance 0.71) beats `tf-ring` (2.8e-07) on both datasets |
| non-abelian holonomy captures the `SE(2)` closure an abelian one cannot | `su2` 3.388 +/- 0.017 vs `so2` 3.397 +/- 0.013 -- inside noise, and reversed on `fashion` |
| a closure-aware architecture draws curves that close | `tf-ring` closes best (0.1942) with no closure machinery at all |
| add the whole constraint at one large weight | a kick, not a descent: NLL 3.4 -> 11.7, closure worse |
| continuation rescues the constraint | it does not; same failure, NLL 11.7 |
| run the solver longer to make room for the constraint | +4 sweeps costs 0.385 nats with no constraint present |
| a proper solver (CCCP / L-BFGS / Newton-CG) fixes the forward pass | it converges (`‖∇S‖` 137 -> 0.003) and the minimiser is 0.33 nats WORSE than the 8-sweep unroll |
| the concave-convex splitting CCCP needs | the attention term has 98.4% positive eigenvalues in `x`; concave only for FIXED keys |
| negative curvature is what stops the solve converging (my own Result 4 claim) | refuted -- the Hessian at the minimum is PD, +5.69 to +2304 |
| the even-kernel prohibition is what the layer needs most | a scalar action provably cannot express a directed convolution (verified 2e-16), and handing it one anyway is worth -0.002 nats against a 0.013 seed spread |
| an energy model conditions on arbitrary subsets better than a masked one | refuted: on scattered/periodic masks `tf-abs` improves (-0.013) and Semagram degrades (+0.091) |
| sample the contour at `n` points directly | a sampled fractal: `\|d\|` mean 0.44 rad vs 0.13 expected |
| close the `su2` holonomy by subtracting `log(H)/n` per edge | does not converge; the correction does not commute with what it corrects |
| `2*arccos\|Re H\|` for the `su2` holonomy | `nan` on step 1 -- infinite derivative at the identity, where init sits |
| `windings` as float32 | 1e-5 rad of base-angle error, two orders above float32 epsilon on the equivariance |
| next-token bigram as the baseline | wrong task; the right one is two-sided Markov infill, 3.445 not 3.298 |

## Honest limits

- **Small, and short.** `n=48`, `d=64`, `vocab=32`, 3000 steps, CPU. Semagram
  starts at NLL 12.6 because of the peaked tied readout while the transformers
  start at 3.47, and it is still descending at 3000 steps. A longer horizon
  would narrow the gap by an unknown amount; it would have to narrow it by 0.23
  nats to change the conclusion.
- **`fashion` is a single seed** and `ne_lakes` / `ne_admin1` were not run at
  all, because they are near-noise at `n=48` (24% of turning power in the top
  band, lag-1 autocorrelation 0.04). That is a property of coastlines.
- **Tokenising dominates the closure metric.** The floor is 0.069 on `mnist`
  against 0.004 for the underlying curves. Every closure number here is mostly
  quantisation, which is why the constraint experiment has so little room.
- **The equivariance result is not unique to this layer.** A ring-rotary
  transformer is exactly equivariant too, and is in the table for that reason.
- **The solver is still not a solver**, and Result 4 is what that costs.

## Verdict

The question was whether this is a genuinely useful layer, proved on real task
data. On this evidence: **not yet, and the reasons are specific rather than
vague.**

What survives is real and exact:

- a structural bug and its fix -- the layer was not on a circle, because a
  non-trivial holonomy puts a branch cut at index 0, and the shift-equivariance
  error is exactly proportional to the holonomy. `phi_dev` was spending
  commitment 1 to buy nothing; `w_holo` was the only thing paying it back. Split
  the connection and equivariance is exact at 1.3e-15 with the holonomy
  preserved as a readout;
- a working non-abelian gauge at `O(n log n)`, verified to 1e-15, which detects
  loop orderings the abelian one is provably blind to;
- a precise account of why test-time constraint composition -- the whole point
  of an energy-based forward pass -- is not available on an unroll whose
  iteration matrix has spectral radius above 1.

What does not survive is the motivating claim. Every distinctive property was
verified exactly and none of them converted into accuracy, on data chosen
specifically because the priors are true of it. The most direct statement of the
result is that `tf-abs`, which is provably wrong about the geometry, beat every
model here that is provably right about it.

Result 5 removes the change I would have suggested first. Making the forward
pass an actual solve is *easy* -- L-BFGS converges on this action in ~120
iterations to `‖∇S‖ = 3e-3` -- and it costs 0.33 nats, because the trained model
is not an approximation to the minimiser of its own action. The energy generated
a good 8-step recurrence; it is not an objective the network is trying to reach.
Any argument for this architecture that runs through the variational structure
has to survive that measurement first, and the honest reading is that the
structure is doing bookkeeping rather than work.

That leaves a continuous-valued output head as the one change likely to help,
since it would make the closure metric and the constraint mechanism mean
something. It is not a hyperparameter, and it would not close 0.23 nats on its
own.
