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

