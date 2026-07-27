# Cyclic fibers vs. weight tying on modular arithmetic

A small factorial experiment in raw JAX testing two claims from a design
conversation about boundary-value ("Fermat-style") sequence models:

1. **factorization** — does two-sided conditioning beat a committed
   autoregressive ordering when evidence sits at both ends?
2. **fiber** — does a *closed* (cyclic) token-embedding space beat free or tied
   embeddings?

The headline: claim 2 has a real, narrow, mechanistic effect. Claim 1 is
**not established** by this code, and a third claim (an energy-based halting
certificate) is **refuted**.

---

## Task

`x_i = (x_0 + i·s) mod p`, for `i = 0..L-1`, with `p = 97`, `L = 8`.
All `p² = 9409` sequences enumerated; 40% train, 60% held out.

Chosen because it is maximally favourable and I want that on the record:

- The interior is genuinely a two-point boundary value problem. Given `x_0` and
  `x_{L-1}`, the step is `s = (x_{L-1} − x_0) · (L−1)⁻¹ mod p`, unique whenever
  `gcd(L−1, p) = 1`. "Know the endpoints, recover the path" is literally true.
- The alphabet `Z_p` is a cyclic group and the generating operation is
  translation, so a product-of-circles embedding makes the task linear.

Both are hardcoded gifts to the architecture under test. A win here is
necessary, not sufficient.

## Design

Backbone is a 2-layer pre-LN transformer, `d_model=64`, 4 heads, ~100k params.
Shared across all arms; only the attention mask and the token-space
parameterization change.

**Fibers**

| | |
|---|---|
| `free` | separate input embedding and output head. No closure. |
| `tied` | one matrix reads types in and reads them back out. The closure that already exists in every real LM. |
| `cyclic` | `e(t) = concat_j [a_j cos(2πjt/p), a_j sin(2πjt/p)]` over `K=8` frequencies, lifted to `d_model` by a learned map, readout tied through the same map. Translation of the token value becomes a *rotation*, one 2-plane per frequency. |

**Arms** — `{ar/std, ar/endpoints_first, bvp} × {free, tied, cyclic}`, plus a
permuted-vocabulary control on the `bvp` arm.

`ar/endpoints_first` presents the sequence as `[x_0, x_{L-1}, x_1, ..., x_{L-2}]`
so an AR model sees both boundary conditions before producing any interior
token. It exists so the comparison isn't a straw man.

`evaluate.ar_is_native` checks whether the *given* positions form a prefix of a
model's ordering. Cells an AR model structurally cannot reach are reported
`n/a` rather than scored as failures.

## Results

### Main grid, per-token accuracy on held-out sequences (chance = 1.03%)

```
model                          next     infill   reverse
ar/std | free                 61.6%      n/a      n/a
ar/std | tied                100.0%      n/a      n/a
ar/std | cyclic               99.9%      n/a      n/a
ar/endpoints_first | free      n/a       0.9%     n/a
ar/endpoints_first | tied      n/a       0.9%     n/a
ar/endpoints_first | cyclic    n/a       0.9%     n/a
bvp | free                     0.8%      1.0%     0.8%
bvp | tied                     0.9%      0.8%     1.1%
bvp | cyclic                   1.4%      1.0%     1.2%
  control (permuted vocab)
bvp | free                     0.9%      0.9%     0.8%
bvp | tied                     1.3%      0.8%     1.4%
bvp | cyclic                   0.9%      0.9%     0.9%
```

Three things to read off:

- The `bvp` arm failed entirely (final losses 4.55–4.57 vs `ln 97 = 4.575`).
- `ar/endpoints_first` reached train loss 0.93–1.06 while sitting at chance on
  test: memorization without generalization, the pre-grok regime.
- The permuted-vocab control is **void**. It can only falsify anything if the
  unpermuted models learned, and in the `bvp` arm none did. Right control,
  wrong arm.

### Diagnostic: infill-only training

The `bvp` failure is plausibly *breadth* — the any-order objective must learn
modular inversion at every index gap at once, and averages its loss over masks
where the target is information-theoretically undetermined. `diagnostic.py`
trains on the infill mask alone (clamp both endpoints, mask the interior), so
the model needs one inverse instead of all of them.

```
fiber    train loss (6k)  test token acc  exact  acc@1  acc@4   AUROC res/energy/maxprob
cyclic        1.572           50.9%       4.6%   4.6%   4.6%     0.500 / 0.532 / 0.487
tied          2.346            5.9%       0.0%   0.0%   0.0%       nan / nan  / nan
```

## What this supports and what it kills

**The one positive result.** Cyclic geometry is *redundant* where tying already
suffices and *load-bearing* where it doesn't:

- repeated **addition** (`next` task): tied 100.0% vs cyclic 99.9% — no effect.
- modular **inversion** (`infill` task): cyclic 50.9% vs tied 5.9% — ~9×.

Mechanism: inversion is a rotation by a scaled angle, which the cyclic basis
makes linear, while a free or tied table has to memorize 97 inverses. This is
essentially the grokking-Fourier result arriving at initialization rather than
after a phase transition, so it is not new — but it is a clean statement of
*when* the prior pays, and it is reproducible in three minutes on a CPU.

**The halting certificate is refuted.** AUROC 0.500 (fixed-point residual) and
0.532 (log-sum-exp energy) against a 0.487 max-probability control, on a model
at 50.9% token accuracy with both correct and incorrect examples present. And
`acc@1 == acc@4`: four refinement steps do exactly what one does. There is no
fixed-point structure for a residual to measure. A single-shot denoiser with a
symmetric readout is not a solver.

**The factorization claim is unestablished, not refuted.** `ar/std` needs
`s = x_1 − x_0` plus repeated addition; `bvp` infill needs
`s = (x_7 − x_0)·7⁻¹ mod 97`. Those are not the same task differently factored —
the second is strictly harder. The confound is now identified but the clean
experiment has not been run.

## Next experiments, in priority order

1. **Matched difficulty.** Give AR the inversion task too (present
   `[x_0, x_{L-1}]` and require the interior) at 10⁴–10⁵ steps so both arms get
   past grokking. Only then is "BVP vs AR" a real comparison.
2. **Make the solve a solve.** The refinement loop currently converges to
   nothing, which means the variational framing is ornamental in this code.
   Either implement the actual CCCP iteration with the clamp term inside the
   objective so a fixed point exists, or drop the energy vocabulary.
3. **Frequency ablation.** Sweep `K` and inspect learned `amp`. If a single
   frequency suffices, the fiber prior is even cheaper than advertised.
4. **Rescue the control.** Re-run permuted-vocab on the infill-only arm, where
   models actually learn, so it can do the falsification job it was built for.
5. **Non-group task.** Replace `Z_p` translation with something lacking group
   structure. The cyclic prior should collapse; if it doesn't, the mechanism
   story above is wrong.

## Reproduce

```bash
pip install jax jaxlib optax
python run_experiment.py --steps 5000       # full grid, ~20 min CPU, resumes from results.json
python run_experiment.py --steps 1500       # smoke test
python diagnostic.py --fiber cyclic --steps 6000   # ~3 min
python diagnostic.py --fiber tied   --steps 6000
```

`run_experiment.py` persists after every run and skips completed cells, so an
interrupted grid costs at most one run.

## Files

```
semagram/data.py       task, splits, orderings, eval specs
semagram/model.py      transformer + the three fiber parameterizations
semagram/train.py      AR and any-order-denoising objectives
semagram/evaluate.py   three conditioning modes, AR native-ordering check,
                       iterative solve, AUROC
run_experiment.py      factorial grid + control
diagnostic.py          infill-only training
results.json           main grid output
mine_diagnostic.json   diagnostic output
```

`unverified_diagnostic.json` is a `cyclic@6000` result found in the working
directory that I did not launch (the container held state from another session).
My own rerun reproduced it exactly — same config, same seed, deterministic — but
it is kept separate because I cannot vouch for its provenance.

## Caveats

- Single seed (0) throughout. No error bars. The 100.0% / 99.9% and 50.9% / 5.9%
  gaps are large relative to plausible seed variance, but this is not measured.
- `d_model=64`, 2 layers, 5–6k steps, CPU. Everything here is small.
- Positional encoding is learned-absolute for all arms. At `L=8` the positional
  prior is irrelevant, so none of the interval/Dirichlet/DST argument that
  motivated this design is tested at all.
- Any-order AR training (XLNet-style) would cover all three conditioning modes,
  and any-order AR is close to the masked objective. So the AR-vs-BVP axis is
  really about the training distribution over factorizations, not architecture.
