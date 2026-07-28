"""task_circuit.py -- does the braid layer transfer to quantum circuits?

THE ANALOGY, precisely. A circuit on n qubits is a sequence of two-qubit gates
on adjacent pairs; a braid word on n strands is a sequence of generators on
adjacent pairs. Qubits are strands, gates are generators, depth is word length,
qubit count is strand count. The layer holds one map PER GATE TYPE and lifts it
to wherever the gate acts, exactly as tying held one map per sign.

WHAT CARRIES OVER AND WHAT DOES NOT. Braid generators obey
sigma_i sigma_{i+1} sigma_i = sigma_{i+1} sigma_i sigma_{i+1}. Quantum gates do
not, so the Yang-Baxter penalty is not imported here and `--w-ybe` defaults to
zero. What both structures share, exactly and universally, is FAR COMMUTATION:
gates on disjoint pairs commute, verified in `circuits.py` to 0.0e+00. That is
the symmetry the architecture provides for free -- a gate at (i, i+1) writes
only those two strand slots, so reordering two disjoint gates cannot change the
result. It is structural, not penalised.

TWO COLUMNS, and only one of them is a comparison.

  DEPTH extrapolation at fixed 4 qubits, braid against a rotary transformer.
  Both models can attempt this, so it is a fair fight.

  QUBIT extrapolation, tied against untied. The transformer is RUN here but the
  column is not a fair comparison and its numbers should not be read as one: a
  6-qubit circuit contains gate tokens whose embedding rows never received a
  gradient in 4-qubit training, so it is reading its input off random init. That
  is the same defect that invalidated the absolute-position baseline earlier in
  this project, except here it is not fixable -- it is what "a symbol the model
  has never seen" means. Its output width is NOT the obstacle; `out` is built
  `n_max` wide precisely so that the fixed-width excuse is not doing any work
  and the token problem is visible on its own.

THE HONEST LIMITATION. The layer carries one vector per qubit, and a real
quantum state is entangled across 2^n dimensions and does not factor that way.
This is a surrogate for a measurable quantity, not a simulator, and no result
here bears on whether it could replace simulation. It could not.

Prediction, recorded before running: the braid layer beats the transformer on
depth extrapolation by less than it did on knots, because far commutation is a
weaker and more learnable symmetry than the braid relation. Tied should beat
untied on qubit extrapolation for the same structural reason as before.
"""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import circuits as C
import commute_probe as P
import semagram as S
import task_knot as K


def init_layer(key, d, n_gate, n_out=1):
    """One map per GATE TYPE, plus a shared start state and a per-qubit head."""
    k = jax.random.split(key, 5)
    sc = 1.0 / np.sqrt(2 * d)
    return {
        "R": jax.random.normal(k[0], (n_gate, 2 * d, 2 * d)) * sc,
        "bR": jnp.zeros((n_gate, 2 * d)),
        "x0": jax.random.normal(k[1], (d,)) * 0.5,
        "h1": jax.random.normal(k[2], (d, 4 * d)) * (1 / np.sqrt(d)),
        "b1": jnp.zeros((4 * d,)),
        "h2": jax.random.normal(k[3], (4 * d, n_out)) * (1 / np.sqrt(4 * d)),
        "b2": jnp.zeros((n_out,)),
    }


def forward(p, toks, mask, n, d, untied_vocab=0):
    """Run the circuit through the layer; read out one value per qubit.

    `untied_vocab > 0` is the control: index R by the full token (qubit AND gate
    type) instead of by gate type alone, so a gate acting on a qubit pair never
    seen in training hits a row that never received a gradient.
    """
    b, L = toks.shape
    x = jnp.broadcast_to(p["x0"], (b, n, d))
    idx = jnp.arange(n)
    ng = len(C.GATES)

    def step(x, t):
        tok = toks[:, t]
        live = mask[:, t][:, None, None]
        i = jnp.clip((tok - 1) // ng, 0, n - 2)
        ridx = (jnp.clip(tok, 0, untied_vocab - 1) if untied_vocab
                else jnp.clip((tok - 1) % ng, 0, ng - 1))
        gi = jnp.take_along_axis(
            x, i[:, None, None] * jnp.ones((1, 1, d), jnp.int32), 1)[:, 0]
        gj = jnp.take_along_axis(
            x, (i + 1)[:, None, None] * jnp.ones((1, 1, d), jnp.int32), 1)[:, 0]
        out = jnp.tanh(jnp.einsum("bij,bj->bi", p["R"][ridx],
                                  jnp.concatenate([gi, gj], -1))
                       + p["bR"][ridx]).reshape(b, 2, d)
        si = (idx[None, :] == i[:, None])[..., None]
        sj = (idx[None, :] == (i + 1)[:, None])[..., None]
        xn = x * (1.0 - (si | sj).astype(x.dtype)) \
            + jnp.where(si, out[:, 0:1], 0.0) + jnp.where(sj, out[:, 1:2], 0.0)
        return x * (1 - live) + xn * live, None

    x, _ = jax.lax.scan(step, x, jnp.arange(L))
    # per-QUBIT readout: the target is one number per qubit, so no pooling
    return (jax.nn.gelu(x @ p["h1"] + p["b1"]) @ p["h2"] + p["b2"])[..., 0]


def tf_forward(p, toks, mask, heads, n_max, rope_base, vocab):
    """Transformer with a PER-QUBIT readout -- built to remove a confound that
    turned out not to exist.

    THE HYPOTHESIS. The first version of this benchmark pooled the gate
    sequence and emitted all `n_max` numbers from the pooled vector, and I
    flagged that as unfair: the braid layer carries a slot per qubit and the
    target is one number per qubit, so it had an architectural match to the
    answer's SHAPE that the transformer did not. The in-distribution gap
    (0.99 against 0.62), in a column where extrapolation is not in play,
    looked like the fingerprint of exactly that.

    THE FIX. Append `n_max` QUERY TOKENS to the gate sequence, one per qubit,
    with their own embedding rows. Attention lets each query read whatever it
    needs from the gates, and qubit `q`'s answer is read from query `q`'s
    position through a head shared across qubits. Both models now emit one
    number per qubit from a per-qubit state.

    THE RESULT: it makes the transformer WORSE, consistently over three seeds
    -- 0.529 +/- 0.045 in distribution against the pooled readout's
    0.618 +/- 0.011, and -0.023 against 0.186 on depth extrapolation. So
    output shape was not what the pooled model was losing on, and the confound
    I confessed to was not there. Both readouts are kept and reported, and the
    STRONGER of the two is the baseline, which is the same rule that made
    `tf-rope` rather than `tf-nope` the baseline in the braid work. Quoting
    the weaker one would inflate the result by an arbitrary factor.
    """
    b, L = toks.shape
    # queries occupy vocab .. vocab+n_max-1, appended after the gate sequence
    q_ids = jnp.broadcast_to(vocab + jnp.arange(n_max), (b, n_max))
    seq = jnp.concatenate([toks, q_ids], 1)
    msk = jnp.concatenate([mask, jnp.ones((b, n_max), mask.dtype)], 1)
    h = _tf_hidden(p, seq, msk, heads, rope_base)
    return (h[:, L:] @ p["qout"] + p["qbo"])[..., 0]


def _tf_hidden(p, toks, mask, heads, rope_base):
    """`K._pool` without the pool: per-position hidden states.

    Rotary positions only (no learned `pos` table), for the reason established
    in the braid work -- absolute rows past the training length never receive a
    gradient, so any extrapolation number using them is confounded.
    """
    b, n = toks.shape
    h = p["emb"][toks]
    for blk in p["blocks"]:
        z = S.layernorm(h)
        q, k, v = jnp.split(z @ blk["qkv"], 3, -1)
        d = q.shape[-1]
        dh = d // heads
        rs = lambda t: t.reshape(b, n, heads, dh).transpose(0, 2, 1, 3)
        qh, kh = K._rope(rs(q), n, rope_base), K._rope(rs(k), n, rope_base)
        att = jnp.einsum("bhid,bhjd->bhij", qh, kh) / np.sqrt(dh)
        att = jnp.where(mask[:, None, None, :] > 0, att, -1e9)
        o = jnp.einsum("bhij,bhjd->bhid", jax.nn.softmax(att, -1), rs(v))
        h = h + o.transpose(0, 2, 1, 3).reshape(b, n, d) @ blk["proj"]
        z = S.layernorm(h)
        h = h + jax.nn.gelu(z @ blk["fc1"] + blk["b1"]) @ blk["fc2"] + blk["b2"]
    return S.layernorm(h)


def tf_pool_forward(p, toks, mask, heads, rope_base):
    """The original confounded readout, kept as an ablation."""
    return K._pool(p, toks, mask, heads, use_pos=False, rope=True,
                   rope_base=rope_base)


def dataset(n_items, n_qubits, depth_range, seed, n_max, l_max):
    W, Y = C.build(n_items, n_qubits, depth_range, seed)
    X, M = C.encode(W, n_max, l_max)
    Yp = np.zeros((len(W), n_max), np.float32)
    Yp[:, :n_qubits] = Y
    qm = np.zeros((len(W), n_max), np.float32)
    qm[:, :n_qubits] = 1.0
    return (jnp.asarray(X), jnp.asarray(M), jnp.asarray(Yp), jnp.asarray(qm))


def masked_r2(pred, y, qm):
    num = jnp.sum(((pred - y) ** 2) * qm)
    mu = jnp.sum(y * qm) / jnp.sum(qm)
    den = jnp.sum(((y - mu) ** 2) * qm)
    return float(1 - num / den)


def run(a):
    n_max = 7
    l_max = a.dmax + 8
    if a.scale_depth:
        l_max = max(l_max, int(round(a.dmax * max(a.test_q) / a.train_q)) + 2)
    vocab = C.VOCAB(n_max)
    tr = dataset(a.train_n, a.train_q, (4, a.dmax), 1, n_max, l_max)
    packs = {f"depth {a.dmax+2}-{a.dmax+8}, {a.train_q}q":
             dataset(a.test_n, a.train_q, (a.dmax + 2, a.dmax + 8), 2,
                     n_max, l_max),
             f"in dist, {a.train_q}q":
             dataset(a.test_n, a.train_q, (4, a.dmax), 3, n_max, l_max)}
    for q in a.test_q:
        if q != a.train_q:
            # Scale depth with qubit count so GATES PER QUBIT stays constant.
            # Without this the extrapolation column is an easier task, not a
            # harder one: a depth-10 circuit spread over 6 qubits touches each
            # qubit less than over 4, so <Z_i> stays nearer its starting value.
            # The first run showed exactly that -- 0.808 at 6 qubits against
            # 0.637 in distribution, extrapolation beating interpolation, which
            # is the signature of a benchmark measuring the wrong thing.
            f = q / a.train_q if a.scale_depth else 1.0
            lo, hi = int(round(4 * f)), int(round(a.dmax * f))
            packs[f"{q} QUBITS (never seen)"] = dataset(
                a.test_n, q, (lo, hi), 4 + q, n_max, l_max)
    print(f"train: {a.train_q} qubits, depth 4-{a.dmax}, {a.train_n} circuits")
    print(f"far commutation holds exactly (verified in circuits.py)\n",
          flush=True)

    rows = {}
    for name in a.models:
        key = jax.random.PRNGKey(a.seed)
        if name.startswith("braid"):
            untied = vocab if name == "braid-untied" else 0
            p = init_layer(key, a.d, vocab if untied else len(C.GATES))
            fwd = lambda p, x, m, u=untied: forward(p, x, m, n_max, a.d, u)
        else:
            per_qubit = (name != "tf-pool")
            # Match parameters by MEASURING the braid model, as in task_knot --
            # the query-token variant carries an extra n_max embedding rows and
            # a (width, 1) head, so the count has to include them.
            ref = init_layer(jax.random.PRNGKey(0), a.d, len(C.GATES))
            target = sum(v.size for v in jax.tree.leaves(ref))
            nv = vocab + (n_max if per_qubit else 0)

            def _size(w):
                # jax.eval_shape gets the true shapes WITHOUT materialising the
                # arrays. That still measures the real model rather than
                # trusting a formula -- the distinction that mattered when a
                # stale `32*d**2` aimed the baseline at a number that no longer
                # existed -- but it costs nothing. Materialising every candidate
                # meant generating ~200M random normals across the sweep, which
                # took 240s per model here and read as a hang.
                sh = jax.eval_shape(
                    lambda: K.init_tf(jax.random.PRNGKey(0), nv,
                                      l_max + n_max, a.tf_layers, w, a.heads))
                s = sum(v.size for v in jax.tree.leaves(sh))
                return s + (w + 1 if per_qubit else 0)

            tgt = target * a.param_mult
            width = min(range(8, 400, 8), key=lambda w: abs(_size(w) - tgt))
            p = K.init_tf(key, nv, l_max + n_max, a.tf_layers,
                          width, a.heads)
            if per_qubit:
                p["qout"] = jax.random.normal(key, (width, 1)) / np.sqrt(width)
                p["qbo"] = jnp.zeros((1,))
                fwd = lambda p, x, m: tf_forward(p, x, m, a.heads, n_max,
                                                 a.rope_base, vocab)
            else:
                p["out"] = jax.random.normal(key, (width, n_max)) / np.sqrt(width)
                p["bo"] = jnp.zeros((n_max,))
                fwd = lambda p, x, m: tf_pool_forward(p, x, m, a.heads,
                                                      a.rope_base)
        npar = sum(v.size for v in jax.tree.leaves(p))

        def loss(p, x, m, y, qm):
            return jnp.sum(((fwd(p, x, m) - y) ** 2) * qm) / jnp.sum(qm)

        opt = optax.adamw(optax.warmup_cosine_decay_schedule(
            0.0, a.lr, a.steps // 20, a.steps, end_value=a.lr * 0.05),
            weight_decay=1e-4)
        st = opt.init(p)

        @jax.jit
        def step(p, st, x, m, y, qm):
            l, g = jax.value_and_grad(loss)(p, x, m, y, qm)
            u, st = opt.update(g, st, p)
            return optax.apply_updates(p, u), st, l

        t0 = time.time()
        for it in range(1, a.steps + 1):
            key, k1 = jax.random.split(key)
            i = jax.random.randint(k1, (a.batch,), 0, tr[0].shape[0])
            p, st, l = step(p, st, tr[0][i], tr[1][i], tr[2][i], tr[3][i])
            if it % max(a.steps // 4, 1) == 0:
                print(f"  {name:13s} {it:5d}/{a.steps} loss {float(l):.5f} "
                      f"| {time.time()-t0:4.0f}s", flush=True)
        # Far commutation, measured directly on synthetic pairs that never
        # appear in training -- the mechanism link, not another score.
        pr = P.probe(lambda X, M: fwd(p, X, M), a.train_q, n_max, l_max,
                     depth=a.dmax, n_pairs=a.probe_n, with_truth=False)
        rows[name] = (npar, {tag: masked_r2(fwd(p, X, M), Y, Q)
                             for tag, (X, M, Y, Q) in packs.items()}, pr)

    print("\n" + "=" * 104)
    print("Quantum circuit surrogate: predict <Z_i> for every qubit (R2)")
    print("=" * 104)
    hdr = f"{'model':14s} {'params':>8s} {'commute':>8s}"
    for tag in packs:
        hdr += f" | {tag[:22]:>22s}"
    print(hdr)
    for nm, (npar, r, pr) in rows.items():
        print(f"{nm:14s} {npar:8d} {pr['ratio']:8.4f}"
              + "".join(f" | {r[t]:22.3f}" for t in packs))
    print("\ncommute = far-commutation ratio from commute_probe.py: the model's"
          "\nsensitivity to swapping two DISJOINT gates (which cannot matter),"
          "\ndivided by its sensitivity to swapping two OVERLAPPING gates"
          "\n(which must). 0 = exactly invariant where it should be while still"
          "\ntelling different circuits apart. 1 = cannot tell the two apart.")
    print("\nThe qubit columns are NOT a fair transformer comparison: gate "
          "tokens for unseen\nqubit pairs never received a gradient, so those "
          "entries measure random init.\nThe depth column is the fair fight.")
    if "tf-pool" in rows and "tf-rope" in rows:
        print("\ntf-rope reads one number per qubit from a per-qubit query "
              "token; tf-pool emits\nall of them from a pooled vector. The gap "
              "between those two rows is output\nshape, not symmetry -- it is "
              "the confound this benchmark used to contain.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*",
                    default=["braid", "braid-untied", "tf-rope", "tf-pool"])
    ap.add_argument("--tf-layers", type=int, default=2,
                    help="transformer depth. The braid layer is a sequential "
                         "state machine and so is the task, so 'the "
                         "transformer is too shallow to simulate a depth-10 "
                         "circuit' is a live alternative to any symmetry "
                         "explanation -- and it is testable by turning this up")
    ap.add_argument("--param-mult", type=float, default=1.0,
                    help="give the transformer this multiple of the braid "
                         "layer's parameter budget. >1 is a deliberately "
                         "GENEROUS baseline")
    ap.add_argument("--probe-n", type=int, default=512,
                    help="circuit pairs for the far-commutation probe")
    ap.add_argument("--train-q", type=int, default=4)
    ap.add_argument("--test-q", type=int, nargs="*", default=[4, 5, 6])
    ap.add_argument("--train-n", type=int, default=20000)
    ap.add_argument("--test-n", type=int, default=2000)
    ap.add_argument("--dmax", type=int, default=10)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--scale-depth", action="store_true", default=True,
                    help="scale depth with qubit count so gates-per-qubit is "
                         "constant across the extrapolation columns")
    ap.add_argument("--rope-base", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())
