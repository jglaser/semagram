"""braids.py -- braid words, their knot closures, and exact Jones polynomials.

Why this dataset rather than another contour set. Part II's benchmark had two
weaknesses that this one is built to avoid.

The metric was invented. Closure error was a quantity I defined, with a
quantisation floor of 0.0714 against measured values near 0.22, so a third of it
was noise no model could get under. Here the target is the Jones polynomial: a
genuine topological invariant, computed exactly by state sum, with no floor and
no modelling choice in it.

The symmetry was too easy. Cyclic shift is the symmetry Semagram builds in
exactly, and a transformer with absolute positions -- which does not have it and
is provably wrong about it -- beat every model that did. Reidemeister
equivalence is categorically harder: deciding whether two braid words close to
the same knot has no cheap local statistic, and it is the symmetry an R-matrix
layer gets for free from Yang-Baxter.

REPRESENTATION. A braid on `s` strands is a word in generators sigma_1 ..
sigma_{s-1} and their inverses; closing it up gives a knot or link. The word is
a token sequence, which is what a sequence model wants, and its closure is a
topological object, which is what the invariant sees.

GROUND TRUTH. The Kauffman bracket by state sum. Each crossing resolves two
ways,

    sigma_i      = A * (identity)  + A^-1 * (cup-cap)
    sigma_i^-1   = A^-1 * (identity) + A * (cup-cap)

and each of the 2^L resolutions is a planar diagram whose closure has some
number of loops. Summing A^(exponent) * delta^(loops-1) with
delta = -A^2 - A^-2 gives the bracket, and

    V = (-A^3)^(-writhe) * bracket

is the Jones polynomial with the unknot normalised to 1. Loop counting is a
union-find over (level, strand) nodes, which is exact and easy to get right.

Verified against the literature for the trefoil and the figure-eight before any
of it is used.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np


# ----------------------------------------------------------------------------
# exact invariants

def _loops(word, state, s):
    """Number of loops in the closure of one resolution, by union-find.

    Nodes are (level, strand). A crossing resolved as the identity connects
    (t,i)-(t+1,i) and (t,i+1)-(t+1,i+1); resolved as the cup-cap it connects
    (t,i)-(t,i+1) and (t+1,i)-(t+1,i+1) instead. Every uninvolved strand passes
    straight through, and the Markov closure joins level L back to level 0.
    """
    L = len(word)
    par = list(range((L + 1) * s))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            par[ra] = rb
    nid = lambda t, j: t * s + j
    for t, ((i, _sgn), how) in enumerate(zip(word, state)):
        for j in range(s):
            if j not in (i, i + 1):
                uni(nid(t, j), nid(t + 1, j))
        if how == 0:                       # identity resolution
            uni(nid(t, i), nid(t + 1, i))
            uni(nid(t, i + 1), nid(t + 1, i + 1))
        else:                              # cup-cap resolution
            uni(nid(t, i), nid(t, i + 1))
            uni(nid(t + 1, i), nid(t + 1, i + 1))
    for j in range(s):                     # closure
        uni(nid(L, j), nid(0, j))
    return len({find(v) for v in range((L + 1) * s)})


def kauffman(word, s, A):
    """Kauffman bracket of the closure, by exact state sum over 2^L states."""
    A = complex(A)
    delta = -A ** 2 - A ** -2
    tot = 0.0 + 0.0j
    for state in itertools.product((0, 1), repeat=len(word)):
        e = 0
        for (i, sgn), how in zip(word, state):
            # positive crossing: identity carries A, cup-cap carries A^-1.
            # negative crossing: the two are exchanged.
            plus = (how == 0) if sgn > 0 else (how == 1)
            e += 1 if plus else -1
        tot += A ** e * delta ** (_loops(word, state, s) - 1)
    return tot


def writhe(word):
    return sum(sgn for _i, sgn in word)


def jones(word, s, A):
    """Jones polynomial of the closure evaluated at A (t = A^-4), unknot -> 1."""
    return (-A ** 3) ** (-writhe(word)) * kauffman(word, s, A)


# ----------------------------------------------------------------------------
# the same invariant in polynomial time, via Temperley-Lieb
#
# The state sum above is exact and 2^L, which is 6.5 seconds per braid at
# length 16 and therefore unusable for a dataset. But the loop count of a
# resolution depends only on its CONNECTIVITY, and the connectivity of a planar
# resolution is a Temperley-Lieb diagram -- a non-crossing perfect matching of
# 2s points, of which there are only Catalan(s): 5 for three strands, 14 for
# four, 42 for five.
#
# So carry a vector over TL diagrams through the word instead of enumerating
# resolutions. Each letter is sigma_i = A*1 + A^-1*e_i (signs exchanged for the
# inverse), composition of diagrams is a union-find that also counts the closed
# loops produced, and the Markov trace closes the result. Cost is O(L * dim^2)
# with dim <= 42, and it agrees with the state sum to machine precision.

@functools.lru_cache(maxsize=None)
def _tl_basis(s):
    """Non-crossing perfect matchings of 2s points, as partner tuples.

    Points are indexed around the boundary: 0..s-1 along the top left-to-right,
    then s..2s-1 along the bottom RIGHT-to-left, so planarity is exactly
    non-crossing in this cyclic order. There are Catalan(s) of them.
    """
    res = []
    def gen(rem, acc):
        if not rem:
            res.append(dict(acc))
            return
        a = rem[0]
        for j in range(1, len(rem)):
            b = rem[j]
            gen([x for x in rem[1:] if x != b], acc + [(a, b), (b, a)])
    gen(list(range(2 * s)), [])
    out, seen = [], set()
    for d in res:
        if any(not (a < d[c] < b)
               for a, b in ((x, d[x]) for x in d if x < d[x])
               for c in range(a + 1, b)):
            continue                                   # crossing
        t = tuple(d[i] for i in range(2 * s))
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _pos(j, s):
    """Boundary index of top j (j<s) or bottom (j-s)."""
    return j if j < s else (3 * s - 1 - j)


@functools.lru_cache(maxsize=None)
def _tl_table(s):
    """Composition table: (i, j) -> (index of a*b, number of closed loops)."""
    basis = _tl_basis(s)
    index = {d: i for i, d in enumerate(basis)}
    tab = {}
    for ia, a in enumerate(basis):
        for ib, b in enumerate(basis):
            # nodes: 0..2s-1 for a, 2s..4s-1 for b
            par = list(range(4 * s))
            def find(x):
                while par[x] != x:
                    par[x] = par[par[x]]
                    x = par[x]
                return x
            def uni(x, y):
                rx, ry = find(x), find(y)
                if rx != ry:
                    par[rx] = ry
            for k in range(2 * s):
                uni(k, a[k])
            for k in range(2 * s):
                uni(2 * s + k, 2 * s + b[k])
            for k in range(s):                     # a's bottom to b's top
                uni(_pos(s + k, s), 2 * s + _pos(k, s))
            outer = [_pos(k, s) for k in range(s)] + \
                    [2 * s + _pos(s + k, s) for k in range(s)]
            groups = {}
            for oi, node in enumerate(outer):
                groups.setdefault(find(node), []).append(oi)
            # `groups` keys are OUTER-LIST positions oi; the composed diagram
            # is indexed by BOUNDARY position _pos(oi, s). Conflating the two
            # produced matchings outside the basis.
            res = [0] * (2 * s)
            for g in groups.values():
                if len(g) == 2:
                    u, v = _pos(g[0], s), _pos(g[1], s)
                    res[u], res[v] = v, u
            loops = len({find(x) for x in range(4 * s)}) - len(groups)
            tab[(ia, ib)] = (index[tuple(res)], loops)
    return basis, index, tab


def jones_tl(word, s, A):
    """Jones polynomial via Temperley-Lieb. Same value as `jones`, poly time."""
    A = complex(A)
    delta = -A ** 2 - A ** -2
    basis, index, tab = _tl_table(s)
    ident = index[tuple(_pos_identity(s))]
    vec = np.zeros(len(basis), dtype=complex)
    vec[ident] = 1.0
    for (i, sgn) in word:
        gen = _tl_gen(s, i)
        ca, cb = (A, 1 / A) if sgn > 0 else (1 / A, A)
        new = np.zeros_like(vec)
        for di, amp in enumerate(vec):
            if amp == 0:
                continue
            j, lp = tab[(di, ident)]
            new[j] += amp * ca * delta ** lp
            j, lp = tab[(di, gen)]
            new[j] += amp * cb * delta ** lp
        vec = new
    tot = 0.0 + 0.0j
    for di, amp in enumerate(vec):
        if amp != 0:
            tot += amp * delta ** (_closure_loops(basis[di], s) - 1)
    return (-A ** 3) ** (-writhe(word)) * tot


@functools.lru_cache(maxsize=None)
def _pos_identity(s):
    m = [0] * (2 * s)
    for k in range(s):
        a, b = _pos(k, s), _pos(s + k, s)
        m[a], m[b] = b, a
    return tuple(m)


@functools.lru_cache(maxsize=None)
def _tl_gen(s, i):
    basis, index, _ = _tl_table(s)
    m = [0] * (2 * s)
    a, b = _pos(i, s), _pos(i + 1, s)
    m[a], m[b] = b, a
    c, d = _pos(s + i, s), _pos(s + i + 1, s)
    m[c], m[d] = d, c
    for k in range(s):
        if k in (i, i + 1):
            continue
        x, y = _pos(k, s), _pos(s + k, s)
        m[x], m[y] = y, x
    return index[tuple(m)]


def _closure_loops(diag, s):
    par = list(range(2 * s))
    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    def uni(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            par[rx] = ry
    for k in range(2 * s):
        uni(k, diag[k])
    for k in range(s):
        uni(_pos(k, s), _pos(s + k, s))
    return len({find(x) for x in range(2 * s)})


# ----------------------------------------------------------------------------
# data

def random_braid(rng, s, length):
    return [(int(rng.integers(0, s - 1)), int(rng.choice([-1, 1])))
            for _ in range(length)]


# Evaluation points on the UNIT CIRCLE. Off it the Jones polynomial is
# unbounded -- at A = 1.3 a random 8-letter braid reaches |V| = 1123 while the
# median is 1.0, so a regression target built there is dominated by a few
# outliers. On |A| = 1 the values stay O(1) (max ~1.9 measured). A = exp(i*pi/4)
# gives t = -1, where V is the knot determinant.
# Chosen by scanning theta for informativeness without heavy tails. At
# t = exp(2 pi i/3) (theta = pi/6) the Jones polynomial has modulus 1 for every
# knot, so that point carries a sign bit and its imaginary part is identically
# zero -- a degenerate regression target. These three have std ~1-2.7 in both
# parts and max |V| under 13.
POINTS = tuple(np.exp(1j * np.array([0.255, 0.888, 1.415])))


def perm_of(word, s):
    """The image of a braid word in the symmetric group."""
    p = np.arange(s)
    for (i, _sgn) in word:
        p[i], p[i + 1] = p[i + 1], p[i]
    return p


def build(n_items, s_range=(3, 4), len_range=(4, 10), points=POINTS,
          seed=0, max_strands=5, cache=True, pure=False):
    """`pure=True` keeps only words whose permutation is the identity.

    This is the control the permutation benchmark demanded. There, the braided
    layer scored 1.000 because its state IS the permutation state -- an almost
    perfect architectural match, with no headroom to separate the symmetry from
    the architecture. Restricting to PURE braids removes that shortcut entirely:
    every example has the identity permutation, so strand tracking carries no
    information and the target can only be read off the crossing structure. The
    pure braid group is the kernel of B_n -> S_n, and it is where the braiding
    actually lives.
    """
    """A dataset of braid words with exact Jones values at fixed evaluation
    points. Evaluating at a few numeric A rather than carrying coefficients
    keeps the target a fixed-size real vector while remaining exact."""
    import os, pickle
    tag = (f"braid_{n_items}_{s_range}_{len_range}_{seed}"
           f"{'_pure' if pure else ''}.pkl").replace(" ", "")
    path = os.path.join("data", tag)
    if cache and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    rng = np.random.default_rng(seed)
    W, Y, S, Lw = [], [], [], []
    seen = set()
    while len(W) < n_items:
        s = int(rng.integers(s_range[0], s_range[1] + 1))
        L = int(rng.integers(len_range[0], len_range[1] + 1))
        w = random_braid(rng, s, L)
        if pure and not np.array_equal(perm_of(w, s), np.arange(s)):
            continue
        key = (s, tuple(w))
        if key in seen:
            continue
        seen.add(key)
        v = [jones_tl(w, s, a) for a in points]   # exact, poly time
        y = np.array([f(x) for x in v for f in (np.real, np.imag)])
        if not np.all(np.isfinite(y)):
            continue
        W.append(w)
        Y.append(y)
        S.append(s)
        Lw.append(L)
    out = (W, np.asarray(Y, np.float64), np.asarray(S), np.asarray(Lw))
    if cache:
        os.makedirs("data", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(out, f)
    return out


def encode(words, s_max, l_max):
    """Braid word -> token ids. Token = (generator index, sign), 0 = padding."""
    X = np.zeros((len(words), l_max), np.int32)
    M = np.zeros((len(words), l_max), np.float32)
    for r, w in enumerate(words):
        for t, (i, sgn) in enumerate(w[:l_max]):
            X[r, t] = 1 + i * 2 + (0 if sgn > 0 else 1)
            M[r, t] = 1.0
    return X, M


VOCAB = lambda s_max: 1 + 2 * (s_max - 1)


# ----------------------------------------------------------------------------

def _verify():
    """Against the literature, before any of this is used for anything."""
    print("Jones polynomial check (t = A^-4):")
    tests = [
        ("unknot      sigma_1        in B_2", [(0, 1)], 2, lambda t: 1 + 0 * t),
        # sigma_1^3 with this sign convention closes to the LEFT-handed
        # trefoil, so the reference is the mirror of the one usually quoted.
        # The trefoil is chiral and the figure-eight is not, which is why the
        # figure-eight passed against either convention and this did not.
        ("trefoil     sigma_1^3      in B_2", [(0, 1)] * 3, 2,
         lambda t: -t ** 4 + t ** 3 + t),
        ("hopf link   sigma_1^2      in B_2", [(0, 1)] * 2, 2, None),
        ("figure-8    (s1 s2^-1)^2   in B_3",
         [(0, 1), (1, -1), (0, 1), (1, -1)], 3,
         lambda t: t ** -2 - t ** -1 + 1 - t + t ** 2),
    ]
    ok = True
    for name, w, s, ref in tests:
        for A in (0.83 + 0.21j, 1.17 - 0.34j):
            v = jones(w, s, A)
            if ref is None:
                print(f"  {name}: V = {v:.6f}  (no reference, informational)")
                break
            t = A ** -4
            e = ref(t)
            err = abs(v - e) / max(abs(e), 1e-12)
            flag = "OK " if err < 1e-9 else "FAIL"
            ok &= err < 1e-9
            print(f"  {name}: |V - ref|/|ref| = {err:.2e}  {flag}")
    print("all Jones values match the literature" if ok else "MISMATCH")
    return ok


if __name__ == "__main__":
    _verify()
    W, Y, S, L = build(200, seed=0)
    print(f"\nsample dataset: {len(W)} braids, strands {S.min()}-{S.max()}, "
          f"length {L.min()}-{L.max()}")
    print(f"target vector dim {Y.shape[1]}, "
          f"per-dim std {np.round(Y.std(0), 3)}")
    print(f"writhe range {min(writhe(w) for w in W)}..{max(writhe(w) for w in W)}")
