"""task_shape.py -- does the layer earn its priors on data that really is a loop?

semagram.py's own conclusion on Shakespeare is that it loses to a
parameter-matched bidirectional transformer by 0.171 nats. That is the right
answer to the wrong question. Its three commitments are PRIORS -- positions on
a circle, a two-sided boundary-value solve, transport with holonomy -- and text
satisfies none of them. A 48-character window has a first character and a last
character and they are not neighbours.

This file asks the question the architecture is actually a candidate answer to,
on real measured data, with three experiments.

  [1] MODELLING.  Complete an occluded arc of a real closed contour. Against a
      parameter-matched bidirectional transformer with learned absolute
      positions -- what people reach for -- and against one with rotary
      positions quantised to the ring, which has commitment 1 built in and is
      the honest baseline.

  [2] EQUIVARIANCE.  A closed curve has no first vertex, so where the loop is
      cut is a gauge choice with no geometric content. Rotate it and see what
      survives. This is not a tuning result: it is exact or it is not.

  [3] CONSTRAINT COMPOSITION.  A closed curve must close. It is a global
      condition -- sum_i exp(i Phi_i) = 0 -- and a token-wise decoder emits n
      independent distributions with no way to enforce it. Because this layer's
      forward pass is argmin_X S[X], the constraint is one more term in S,
      added AFTER training, solved by the same sweeps. That is the capability
      the Arrival framing is really about: state what must be true anywhere on
      the loop, and solve for a whole that satisfies all of it at once.

Run:
    python task_shape.py --quick                  # ~5 min, sanity
    python task_shape.py --dataset mnist --seeds 3
    python task_shape.py --report                 # tabulate results.json
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import contours as C
import loop_layer as L
import semagram as S

RESULTS = "results.json"
CKPT_DIR = "ckpt"
TWO_PI = 2.0 * np.pi


def save_ckpt(cell, p):
    """Keep trained weights so the inference-time experiments can be iterated
    on without paying for training again -- on CPU a run is ~25 minutes."""
    import pickle
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(os.path.join(CKPT_DIR, cell.replace("|", "_") + ".pkl"), "wb") as f:
        pickle.dump(jax.tree.map(np.asarray, p), f)


def load_ckpt(cell):
    import pickle
    path = os.path.join(CKPT_DIR, cell.replace("|", "_") + ".pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        print(f"  [ckpt] {cell}")
        return jax.tree.map(jnp.asarray, pickle.load(f))


# ----------------------------------------------------------------------------
# task definition

def occlusion(key, batch, n, lo=6, hi=18):
    """One contiguous occluded arc per example, random position and length.

    Contiguous and randomly placed because that is what an occlusion IS -- a
    piece of the object hidden behind something else. Random placement also
    keeps the absolute-position baseline honest: it never gets to memorise
    where the hole goes.
    """
    k1, k2 = jax.random.split(key)
    start = jax.random.randint(k1, (batch, 1), 0, n)
    length = jax.random.randint(k2, (batch, 1), lo, hi + 1)
    off = (jnp.arange(n)[None, :] - start) % n
    return (off >= length).astype(jnp.float32)


def eval_masks(batch, n, gap=12, seed=0):
    """A fixed occlusion set, identical for every model and every seed."""
    rng = np.random.default_rng(seed)
    start = rng.integers(0, n, batch)
    off = (np.arange(n)[None, :] - start[:, None]) % n
    return jnp.asarray((off >= gap).astype(np.float32))


# ----------------------------------------------------------------------------
# geometry of a decoded shape

def decode_angles(tokens, clamp, pred, centres):
    """Turning angles of the reconstruction: truth where given, model elsewhere."""
    return jnp.where(clamp > 0, centres[tokens], centres[pred])


def closure_of(d):
    """||sum_i exp(i Phi_i)|| / n. Zero iff the reconstructed curve closes."""
    phi = jnp.cumsum(d, -1) - d[..., :1]
    return jnp.sqrt(jnp.sum(jnp.stack(
        [jnp.cos(phi).sum(-1), jnp.sin(phi).sum(-1)], -1) ** 2, -1)) / d.shape[-1]


def turn_defect(d):
    """|sum_i d_i - 2*pi| -- the tangent-winding half of the closure condition."""
    return jnp.abs(jnp.sum(d, -1) - TWO_PI)


def turn_energy(p, cfg, centres, tokens, clamp, weight):
    """The tangent half of closure: sum_i d_i = 2*pi (Hopf's Umlaufsatz).

    Worth separating from the positional half, because the two are exactly the
    abelian and non-abelian parts of the same statement. A closed curve's edges
    compose to the identity in SE(2); the part of that condition which is a
    plain SUM of turning angles is what an abelian holonomy can express, and it
    is linear in the decoded angles, hence well conditioned. The part that says
    the curve actually joins up is the part that does not commute.
    """
    true_d = centres[tokens]

    def energy(x):
        pr = jax.nn.softmax(L.logits_of(p, x, cfg), -1)
        d = jnp.where(clamp > 0, true_d, pr @ centres)
        return weight * jnp.sum((jnp.sum(d, -1) - TWO_PI) ** 2)
    return energy


def closure_energy(p, cfg, centres, tokens, clamp, weight):
    """The test-time constraint: a term added to the action after training.

    The map from the solved state X to the curve is differentiable all the way
    through -- logits, softmax, expected turning angle, cumulative phase, the
    closing vector -- so `jax.grad` of this is just another force in the same
    preconditioned descent. Observed vertices contribute their TRUE angle,
    because they are given; free vertices contribute the model's expectation.

    Nothing here was available at training time and nothing is retrained. This
    function is the entire mechanism.
    """
    true_d = centres[tokens]

    def energy(x):
        pr = jax.nn.softmax(L.logits_of(p, x, cfg), -1)
        d = jnp.where(clamp > 0, true_d, pr @ centres)
        phi = jnp.cumsum(d, -1) - d[..., :1]
        cx = jnp.sum(jnp.cos(phi), -1) / cfg.n
        cy = jnp.sum(jnp.sin(phi), -1) / cfg.n
        return weight * jnp.sum(cx ** 2 + cy ** 2)
    return energy


ENERGIES.update(close=closure_energy, turn=turn_energy)


# ----------------------------------------------------------------------------
# training

def batches(key, data, batch, chunk, n):
    idx = jax.random.randint(key, (chunk, batch), 0, data.shape[0])
    ks = jax.random.split(jax.random.fold_in(key, 1), chunk)
    toks = data[idx]
    cls = jnp.stack([occlusion(k, batch, n) for k in ks])
    return toks, cls


def train_semagram(key, data, cfg, steps, batch, lr, chunk=25, log=None):
    p = L.init(key, cfg)
    opt = S.make_opt(lr, steps)
    st = opt.init(p)
    t0 = time.time()
    for it in range(0, steps, chunk):
        key, k = jax.random.split(key)
        toks, cls = batches(k, data, batch, chunk, cfg.n)
        p, st, (nll, res) = L.train_chunk(p, st, toks, cls, opt.update, cfg)
        if log and ((it // chunk) % 20 == 0 or it + chunk >= steps):
            log(it + chunk, float(nll), float(res), time.time() - t0)
    return p


def train_tf(key, data, vocab, n, steps, batch, lr, layers, width, heads, ring,
             chunk=25, log=None):
    p = L.tf_init(key, vocab, n, layers, width, heads, ring)
    opt = S.make_opt(lr, steps)
    st = opt.init(p)
    t0 = time.time()
    for it in range(0, steps, chunk):
        key, k = jax.random.split(key)
        toks, cls = batches(k, data, batch, chunk, n)
        p, st, l = L.tf_chunk(p, st, toks, cls, heads, ring, opt.update)
        if log and ((it // chunk) % 20 == 0 or it + chunk >= steps):
            log(it + chunk, float(l), 0.0, time.time() - t0)
    return p


# ----------------------------------------------------------------------------
# evaluation

ENERGIES = {}   # filled in below; keyed by constraint name


def constrained_solve(p, cfg, toks, mask, centres, weights, sweeps, kind,
                      eta=None):
    """Descend on the constrained action by continuation: raise the weight in
    stages, warm-starting each solve from the previous one.

    Adding the whole constraint at once does not work, and the failure is
    informative rather than fatal. The preconditioner is built for the
    quadratic part of the original action and knows nothing about the new term,
    so a large weight applied cold is a kick rather than a descent: measured on
    a lightly-trained model, w=10 in one shot moved the closure error from
    0.285 to 0.267 while sending NLL from 3.5 to 13.7 -- the state was thrown
    off the manifold where the readout means anything, without the constraint
    being satisfied.

    Continuation is the standard remedy for exactly this and costs nothing
    extra structurally: `solve` already accepts `x0`, so each stage is the same
    layer called again. It also makes the trade explicit, since every
    intermediate weight is a point on the curve of "how much NLL does this much
    closure cost".
    """
    cfg = dataclasses.replace(cfg, k_steps=sweeps)
    if eta is not None:
        cfg = dataclasses.replace(cfg, eta=eta)
    mk = ENERGIES[kind]
    x = None
    for w in weights:
        x = L.solve(p, p["emb"][toks], mask, cfg,
                    extra=(mk(p, cfg, centres, toks, mask, w) if w > 0 else None),
                    x0=x)
    return x, cfg


def measure(p, cfg, toks, mask, centres, x):
    lg = L.logits_of(p, x, cfg)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, toks)
    free = 1.0 - mask
    pred = jnp.argmax(lg, -1)
    d = decode_angles(toks, mask, pred, centres)
    soft = jnp.where(mask > 0, centres[toks], jax.nn.softmax(lg, -1) @ centres)
    return dict(
        nll=float(jnp.sum(ce * free) / jnp.sum(free)),
        acc=float(jnp.sum((pred == toks) * free) / jnp.sum(free)),
        closure=float(jnp.mean(closure_of(d))),
        soft_closure=float(jnp.mean(closure_of(soft))),
        turn=float(jnp.mean(turn_defect(d))))


def eval_semagram(p, cfg, toks, mask, centres, con_w=0.0, sweeps=None,
                  kind="close", eta=None):
    """Score the layer, optionally with a constraint term added to the action.

    `sweeps` overrides k_steps at inference. That is legitimate and worth
    stating plainly: the forward pass is a descent on an energy, so when a term
    is added to that energy there is more to descend and the solver may be run
    longer. Nothing is retrained and no parameter changes. To keep the
    comparison honest the unconstrained baseline is re-scored at the SAME
    sweep count, so the constraint's effect is never confused with the effect
    of solving harder.
    """
    if sweeps is not None:
        cfg = dataclasses.replace(cfg, k_steps=sweeps)
    if eta is not None:
        cfg = dataclasses.replace(cfg, eta=eta)
    extra = None
    if con_w > 0:
        extra = ENERGIES[kind](p, cfg, centres, toks, mask, con_w)
    x = L.solve(p, p["emb"][toks], mask, cfg, extra=extra)
    lg = L.logits_of(p, x, cfg)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, toks)
    free = 1.0 - mask
    pred = jnp.argmax(lg, -1)
    d = decode_angles(toks, mask, pred, centres)
    soft = jnp.where(mask > 0, centres[toks], jax.nn.softmax(lg, -1) @ centres)
    hol, _ = L.holonomy(p, x, cfg)
    return dict(
        nll=float(jnp.sum(ce * free) / jnp.sum(free)),
        acc=float(jnp.sum((pred == toks) * free) / jnp.sum(free)),
        closure=float(jnp.mean(closure_of(d))),
        soft_closure=float(jnp.mean(closure_of(soft))),
        turn=float(jnp.mean(turn_defect(d))),
        holonomy=float(hol))


def eval_tf(p, toks, mask, centres, heads, ring):
    lg = L.tf_forward(p, toks, mask, heads, ring)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, toks)
    free = 1.0 - mask
    pred = jnp.argmax(lg, -1)
    d = decode_angles(toks, mask, pred, centres)
    return dict(
        nll=float(jnp.sum(ce * free) / jnp.sum(free)),
        acc=float(jnp.sum((pred == toks) * free) / jnp.sum(free)),
        closure=float(jnp.mean(closure_of(d))),
        turn=float(jnp.mean(turn_defect(d))))


def equivariance(fn, toks, mask, shifts=(1, 7, 23)):
    """Relative change in output logits when the loop's origin is moved.

    A closed curve has no first vertex. Any dependence on where the sequence was
    cut is the model inventing structure the data does not have.
    """
    base = fn(toks, mask)
    out = []
    for s in shifts:
        r = lambda z: jnp.roll(z, s, axis=1)
        out.append(float(jnp.linalg.norm(r(base) - fn(r(toks), r(mask)))
                         / (jnp.linalg.norm(base) + 1e-30)))
    return float(np.mean(out))


# ----------------------------------------------------------------------------
# trivial references, so "beats the unigram" means something concrete

def references(d, te_tok, mask, centres):
    vocab = len(centres)
    tr = d["train"]
    cnt = np.bincount(tr.ravel(), minlength=vocab).astype(np.float64)
    pr = cnt / cnt.sum()
    uni = float(-(pr * np.log(pr + 1e-12)).sum())

    c = np.ones((vocab, vocab))
    np.add.at(c, (tr[:, :-1].ravel(), tr[:, 1:].ravel()), 1.0)
    lp = np.log(c / c.sum(1, keepdims=True))
    tt = np.asarray(te_tok)
    big = float(-lp[tt[:, :-1], tt[:, 1:]].mean())

    # The geometric floor: the closure error of the TRUE shape once it has been
    # pushed through the tokeniser. No model can do better than this, so it is
    # the number the constraint experiment has to be read against.
    floor = float(jnp.mean(closure_of(jnp.asarray(centres)[te_tok])))
    return dict(unigram=uni, bigram=big, closure_floor=floor)


# ----------------------------------------------------------------------------
# the run

def run(args):
    n, vocab = args.n, args.vocab
    d = C.cached(args.dataset, n, vocab)
    tr = jnp.asarray(d["train"])
    te = jnp.asarray(d["test"][:args.eval_n])
    centres = jnp.asarray(d["centres"])
    mask = eval_masks(te.shape[0], n, args.gap)
    steps = args.steps

    res = args.results
    store = json.load(open(res)) if os.path.exists(res) else {}
    refs = references(d, te, mask, centres)
    store.setdefault("_refs", {})[args.dataset] = refs
    print(f"\n=== {args.dataset} | n={n} vocab={vocab} | train {tr.shape[0]} "
          f"test {te.shape[0]} | occlusion {args.gap}/{n} ===")
    print(f"references: unigram {refs['unigram']:.3f} | neighbour-bigram "
          f"{refs['bigram']:.3f} | closure floor {refs['closure_floor']:.4f}")

    base = L.LoopCfg(n=n, d=args.d, heads=args.heads, k_steps=args.k,
                     modes=args.modes, vocab=vocab, phi_dev=args.phi_dev)
    npar = L.n_params(L.init(jax.random.PRNGKey(0), base))

    for seed in (args.seed_list or list(range(args.seeds))):
        for name, over in [("sema-so2", dict(gauge="so2")),
                           ("sema-su2", dict(gauge="su2")),
                           ("sema-so2-open", dict(gauge="so2", gauge_close=False)),
                           ("sema-su2-open", dict(gauge="su2", gauge_close=False))]:
            if name not in args.models:
                continue
            cell = f"{args.dataset}|{name}|s{seed}"
            if cell in store and not args.force:
                print(f"skip {cell}")
                continue
            cfg = dataclasses.replace(base, **over)
            t0 = time.time()
            lg = lambda i, l, r, t: print(
                f"  {name} s{seed} {i:5d}/{steps} loss {l:.3f} holo {r:.3f} "
                f"| {t:5.0f}s")
            p = load_ckpt(cell)
            if p is None:
                p = train_semagram(jax.random.PRNGKey(seed), tr, cfg, steps,
                                   args.batch, args.lr, log=lg)
                save_ckpt(cell, p)
            m = eval_semagram(p, cfg, te, mask, centres)
            m["equivariance"] = equivariance(
                lambda t, c: L.logits_of(p, L.solve(p, p["emb"][t], c, cfg), cfg),
                te[:64], mask[:64])
            # test-time constraint: same weights, extra energy term. The w=0
            # row at the same sweep count is the control.
            m0 = eval_semagram(p, cfg, te, mask, centres, sweeps=args.con_steps)
            m["con0_nll"], m["con0_closure"] = m0["nll"], m0["closure"]
            m["con0_soft"] = m0["soft_closure"]
            for w in args.con_w:
                mc = eval_semagram(p, cfg, te, mask, centres, con_w=w,
                                   sweeps=args.con_steps)
                m[f"con{w}_nll"] = mc["nll"]
                m[f"con{w}_closure"] = mc["closure"]
                m[f"con{w}_soft"] = mc["soft_closure"]
                m[f"con{w}_acc"] = mc["acc"]
            # How the completion degrades as the hole grows. This is where a
            # boundary-value solve should separate from a token-wise decoder:
            # at 6 vertices the neighbours almost determine the answer and any
            # model can interpolate, while at 24 the interior is reachable only
            # through a global constraint.
            for g in args.gaps:
                mg = eval_semagram(p, cfg, te, eval_masks(te.shape[0], n, g),
                                   centres)
                m[f"gap{g}_nll"] = mg["nll"]
                m[f"gap{g}_closure"] = mg["closure"]
            m["params"] = int(npar)
            m["seconds"] = time.time() - t0
            store[cell] = m
            json.dump(store, open(res, "w"), indent=1)
            print(f"  -> {cell}: nll {m['nll']:.3f} acc {m['acc']:.3f} "
                  f"closure {m['closure']:.4f} equiv {m['equivariance']:.1e} "
                  f"({m['seconds']:.0f}s)")

        for name, ring in [("tf-abs", False), ("tf-ring", True)]:
            if name not in args.models:
                continue
            cell = f"{args.dataset}|{name}|s{seed}"
            if cell in store and not args.force:
                print(f"skip {cell}")
                continue
            layers, width, cnt = L.tf_match(npar, vocab, n, args.heads, ring)
            t0 = time.time()
            lg = lambda i, l, r, t: print(
                f"  {name} s{seed} {i:5d}/{steps} loss {l:.3f} | {t:5.0f}s")
            p = load_ckpt(cell)
            if p is None:
                p = train_tf(jax.random.PRNGKey(seed), tr, vocab, n, steps,
                             args.batch, args.lr, layers, width, args.heads,
                             ring, log=lg)
                save_ckpt(cell, p)
            m = eval_tf(p, te, mask, centres, args.heads, ring)
            m["equivariance"] = equivariance(
                lambda t, c: L.tf_forward(p, t, c, args.heads, ring),
                te[:64], mask[:64])
            for g in args.gaps:
                mg = eval_tf(p, te, eval_masks(te.shape[0], n, g), centres,
                             args.heads, ring)
                m[f"gap{g}_nll"] = mg["nll"]
                m[f"gap{g}_closure"] = mg["closure"]
            m["params"] = int(cnt)
            m["layers"], m["width"] = layers, width
            m["seconds"] = time.time() - t0
            store[cell] = m
            json.dump(store, open(res, "w"), indent=1)
            print(f"  -> {cell}: nll {m['nll']:.3f} acc {m['acc']:.3f} "
                  f"closure {m['closure']:.4f} equiv {m['equivariance']:.1e} "
                  f"| {layers}L w{width} {cnt/1e3:.0f}k ({m['seconds']:.0f}s)")
    return store


# ----------------------------------------------------------------------------

def agg(store, dataset, name, field):
    v = [m[field] for k, m in store.items()
         if k.startswith(f"{dataset}|{name}|") and field in m]
    return (float(np.mean(v)), float(np.std(v)), len(v)) if v else (None, 0, 0)


def load_all():
    """Merge every results shard.

    Runs are sharded across processes -- on a 4-core box the layer is
    dispatch-bound, so several single-process runs in parallel finish sooner
    than one after another -- and a shared JSON file would race.
    """
    import glob
    store = {}
    for f in sorted(glob.glob("results*.json")):
        d = json.load(open(f))
        for k, v in d.items():
            if k == "_refs":
                store.setdefault("_refs", {}).update(v)
            else:
                store[k] = v
    return store


def report(args):
    store = load_all()
    datasets = sorted({k.split("|")[0] for k in store if not k.startswith("_")})
    names = ["sema-so2", "sema-su2", "sema-so2-open", "sema-su2-open",
             "tf-abs", "tf-ring"]
    for ds in datasets:
        r = store["_refs"][ds]
        print("\n" + "=" * 78)
        print(f"{ds}  --  occluded-arc completion on real closed contours")
        print("=" * 78)
        print(f"  references: unigram {r['unigram']:.3f} | neighbour-bigram "
              f"{r['bigram']:.3f} | closure floor {r['closure_floor']:.4f}")
        print(f"\n  {'model':16s} {'NLL':>14s} {'acc':>7s} {'closure':>9s} "
              f"{'shift-equiv':>12s} {'params':>8s}")
        for nm in names:
            mu, sd, k = agg(store, ds, nm, "nll")
            if mu is None:
                continue
            ac, _, _ = agg(store, ds, nm, "acc")
            cl, _, _ = agg(store, ds, nm, "closure")
            eq, _, _ = agg(store, ds, nm, "equivariance")
            pa, _, _ = agg(store, ds, nm, "params")
            print(f"  {nm:16s} {mu:7.3f} +/-{sd:5.3f} {ac:7.3f} {cl:9.4f} "
                  f"{eq:12.1e} {pa/1e3:7.0f}k  (n={k})")

        gaps = sorted({int(k[3:k.index("_")]) for m in store.values()
                       if isinstance(m, dict) for k in m
                       if k.startswith("gap") and k.endswith("_nll")})
        if gaps:
            print(f"\n  completion NLL vs occluded-arc length (of {48})")
            print(f"  {'model':16s}" + "".join(f"{g:>9d}" for g in gaps))
            for nm in names:
                row = [agg(store, ds, nm, f"gap{g}_nll")[0] for g in gaps]
                if row[0] is None:
                    continue
                print(f"  {nm:16s}" + "".join(
                    f"{v:9.3f}" if v is not None else f"{'-':>9s}" for v in row))

        print(f"\n  test-time closure constraint (same weights, extra energy "
              f"term in S; floor {r['closure_floor']:.4f})")
        print(f"  {'model':16s} {'weight':>8s} {'NLL':>8s} {'closure':>9s}")
        for nm in names:
            if not nm.startswith("sema"):
                continue
            base_n, _, k = agg(store, ds, nm, "nll")
            if base_n is None:
                continue
            base_c, _, _ = agg(store, ds, nm, "closure")
            print(f"  {nm:16s} {'0':>8s} {base_n:8.3f} {base_c:9.4f}")
            for w in (1.0, 10.0, 100.0):
                mu, _, kk = agg(store, ds, nm, f"con{w}_nll")
                if mu is None:
                    continue
                cc, _, _ = agg(store, ds, nm, f"con{w}_closure")
                print(f"  {'':16s} {w:8.0f} {mu:8.3f} {cc:9.4f}")


def parse():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dataset", default="mnist",
                    choices=["mnist", "fashion", "ne_lakes", "ne_admin1"])
    ap.add_argument("--models", nargs="*", default=[
        "sema-so2", "sema-su2", "sema-so2-open", "tf-abs", "tf-ring"])
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--gap", type=int, default=12)
    ap.add_argument("--gaps", type=int, nargs="*", default=[6, 12, 18, 24])
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--modes", type=int, default=12)
    ap.add_argument("--phi-dev", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--seed-list", type=int, nargs="*", default=None,
                    help="explicit seeds, so runs can be sharded over processes")
    ap.add_argument("--eval-n", type=int, default=512)
    ap.add_argument("--con-w", type=float, nargs="*", default=[1.0, 10.0, 100.0])
    ap.add_argument("--con-steps", type=int, default=40,
                    help="solver sweeps for the constrained solve (and for its "
                         "own w=0 control)")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.steps, a.eval_n, a.d = 300, 128, 64
    return a


if __name__ == "__main__":
    a = parse()
    if a.report:
        report(a)
    else:
        run(a)
        report(a)
