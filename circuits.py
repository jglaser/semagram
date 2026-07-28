"""circuits.py -- quantum circuits as braid-shaped words, with exact targets.

WHY THIS DATASET. A quantum circuit on n qubits is a sequence of two-qubit gates
acting on adjacent pairs. A braid word on n strands is a sequence of generators
acting on adjacent pairs. Same shape: qubits are strands, gates are generators,
circuit depth is word length, and qubit count is strand count. The tied layer
holds one map per gate type and lifts it wherever the gate acts, so it can be
trained on 4-qubit circuits and run on 6 -- which is the thing a transformer
cannot attempt, because a 6-qubit circuit contains gate tokens whose embedding
rows do not exist in a 4-qubit model.

WHAT IS THE SAME AND WHAT IS NOT. Braid generators obey
sigma_i sigma_{i+1} sigma_i = sigma_{i+1} sigma_i sigma_{i+1}; quantum gates in
general do not, so the Yang-Baxter penalty is NOT automatically appropriate
here. What both structures share is FAR COMMUTATION: operations on disjoint
pairs commute, `G_i G_j = G_j G_i` for |i - j| >= 2, exactly and universally.
That is the symmetry a circuit actually has, and it is what should be built in.

THE HONEST LIMITATION, stated before any number. The layer carries one vector
per qubit. A real quantum state is entangled and lives in 2^n dimensions; it
generally does not factor into per-qubit pieces. So this layer CANNOT represent
an arbitrary quantum state and is not a simulator. It is a surrogate: something
that predicts measurable quantities cheaply. Whether a strand-structured
inductive bias helps a surrogate is a real question; whether it can replace
simulation is not, and the answer to that is no.

GROUND TRUTH. Exact statevector simulation. For n <= 10 the state is 1024
complex numbers and there is no approximation anywhere -- the target is what the
circuit really does, computed directly, the same standard the Jones benchmark
was held to.

THE TASK. Each qubit starts in Ry(pi/4)|0>, a superposition, so nothing is
sitting in a computational basis state where the answer would be trivially +-1.
The circuit runs, and the target is <Z_i> for every qubit: the expectation value
of a Z measurement. That is a per-qubit real number in [-1, 1], which matches
the layer's per-strand readout while NOT being locally readable -- <Z_i> after
an entangling circuit depends on what happened to every qubit it interacted
with, directly or indirectly.
"""

from __future__ import annotations

import numpy as np

# Two-qubit gate set.
#
# The first version of this file used {CZ, iSWAP} and produced a CONSTANT
# target: <Z_i> = 0.7071 for every qubit of every circuit, std exactly 0. The
# simulator was correct -- its own verification checks passed -- and the task
# was empty. CZ is diagonal, so it cannot change any single-qubit <Z>; and
# iSWAP exchanges two qubits' amplitudes, which does nothing when every qubit
# starts in the SAME state. Two gates, both invisible to the observable.
#
# CNOT fixes it because it is neither diagonal nor a permutation of identical
# things: it flips the target conditioned on the control, which correlates the
# two qubits and moves <Z> on both.
GATES = ["cnot", "cz", "iswap"]


def _two_qubit(gate):
    """4x4 unitary in the basis |00>, |01>, |10>, |11>."""
    if gate == "cnot":
        U = np.eye(4, dtype=complex)
        U[[2, 3]] = U[[3, 2]]           # |10> <-> |11>: flip target if control
        return U
    if gate == "cz":
        return np.diag([1, 1, 1, -1]).astype(complex)
    if gate == "iswap":
        U = np.zeros((4, 4), complex)
        U[0, 0] = 1
        U[3, 3] = 1
        U[1, 2] = 1j
        U[2, 1] = 1j
        return U
    raise ValueError(gate)


def _apply(psi, U, i, n):
    """Apply a two-qubit gate to adjacent qubits (i, i+1) of an n-qubit state.

    The state is stored as a length-2^n vector. Reshaping it to
    (2^i, 2, 2, 2^(n-i-2)) puts the two target qubits in the middle two axes,
    so the gate is a plain matrix multiply on those axes and everything else is
    carried along untouched. That is the whole trick of statevector simulation.
    """
    psi = psi.reshape(2 ** i, 4, 2 ** (n - i - 2))
    psi = np.einsum("ab,ibj->iaj", U, psi)
    return psi.reshape(-1)


def initial_state(n, theta=np.pi / 4):
    """Ry(theta)|0> on every qubit, as a product state.

    Starting from |0...0> would make <Z_i> = 1 everywhere until a gate acts,
    and with CZ (which is diagonal) it would stay there -- a degenerate target.
    A superposition start means every qubit has something to lose.
    """
    one = np.array([np.cos(theta / 2), np.sin(theta / 2)], complex)
    psi = np.array([1.0 + 0j])
    for _ in range(n):
        psi = np.kron(psi, one)
    return psi


def simulate(word, n):
    """Run the circuit and return <Z_i> for every qubit. Exact."""
    psi = initial_state(n)
    for (i, g) in word:
        psi = _apply(psi, _two_qubit(GATES[g]), i, n)
    p = np.abs(psi) ** 2
    idx = np.arange(2 ** n)
    return np.array([float(np.sum(p * (1 - 2 * ((idx >> (n - 1 - q)) & 1))))
                     for q in range(n)])


def random_circuit(rng, n, depth):
    return [(int(rng.integers(0, n - 1)), int(rng.integers(0, len(GATES))))
            for _ in range(depth)]


def encode(words, n_max, l_max):
    """Gate sequence -> tokens. token = 1 + qubit*len(GATES) + gate_type."""
    X = np.zeros((len(words), l_max), np.int32)
    M = np.zeros((len(words), l_max), np.float32)
    for r, w in enumerate(words):
        for t, (i, g) in enumerate(w[:l_max]):
            X[r, t] = 1 + i * len(GATES) + g
            M[r, t] = 1.0
    return X, M


VOCAB = lambda n_max: 1 + len(GATES) * (n_max - 1)


def build(n_items, n_qubits, depth_range, seed):
    rng = np.random.default_rng(seed)
    W, Y = [], []
    seen = set()
    while len(W) < n_items:
        d = int(rng.integers(depth_range[0], depth_range[1] + 1))
        w = random_circuit(rng, n_qubits, d)
        if tuple(w) in seen:
            continue
        seen.add(tuple(w))
        W.append(w)
        Y.append(simulate(w, n_qubits))
    return W, np.asarray(Y)


# ----------------------------------------------------------------------------

def _verify():
    """Check the simulator against cases with a known answer, before use."""
    ok = True
    n = 3
    psi = initial_state(n)
    z = np.cos(np.pi / 4)
    got = simulate([], n)
    err = np.abs(got - z).max()
    print(f"  empty circuit: <Z_i> = {got.round(4)}, expected {z:.4f} "
          f"(err {err:.1e})")
    ok &= err < 1e-12

    # CZ is diagonal, so it cannot change any single-qubit <Z>.
    # Index by NAME, not position: inserting CNOT at index 0 silently made this
    # check apply the wrong gate, and it failed loudly. That is the test working.
    got = simulate([(0, GATES.index("cz"))], n)
    err = np.abs(got - z).max()
    print(f"  one CZ:        <Z_i> unchanged? err {err:.1e}   "
          f"(CZ is diagonal, so it must be)")
    ok &= err < 1e-12

    # iSWAP exchanges the two qubits' amplitudes, so on a symmetric product
    # state it also leaves <Z> alone -- but on an asymmetric one it swaps them.
    one = np.array([1.0, 0.0], complex)
    other = np.array([0.0, 1.0], complex)
    psi = np.kron(np.kron(one, other), one)
    p0 = psi.copy()
    p1 = _apply(p0.copy(), _two_qubit("iswap"), 0, 3)
    pr = np.abs(p1) ** 2
    idx = np.arange(8)
    zs = [float(np.sum(pr * (1 - 2 * ((idx >> (2 - q)) & 1)))) for q in range(3)]
    print(f"  iSWAP on |010>: <Z> = {np.round(zs, 4)}  "
          f"(expected [-1, 1, 1] -- the excitation moved)")
    ok &= abs(zs[0] + 1) < 1e-12 and abs(zs[1] - 1) < 1e-12

    # unitarity: the norm must be preserved exactly
    rng = np.random.default_rng(0)
    w = random_circuit(rng, 5, 20)
    psi = initial_state(5)
    for (i, g) in w:
        psi = _apply(psi, _two_qubit(GATES[g]), i, 5)
    print(f"  norm after 20 gates: {np.linalg.norm(psi):.12f} (must be 1)")
    ok &= abs(np.linalg.norm(psi) - 1) < 1e-12

    # FAR COMMUTATION -- the symmetry this benchmark is really about. Gates on
    # disjoint qubit pairs commute exactly, so swapping them leaves the circuit
    # unchanged. This is what a model has to learn and what the layer gets by
    # construction, so it had better actually hold.
    n = 6
    for trial in range(3):
        i, j = 0, 3                       # |i - j| >= 2, so disjoint
        gi, gj = int(rng.integers(0, len(GATES))), int(rng.integers(0, len(GATES)))
        pre = random_circuit(rng, n, 4)
        a = simulate(pre + [(i, gi), (j, gj)], n)
        b = simulate(pre + [(j, gj), (i, gi)], n)
        e = np.abs(a - b).max()
        ok &= e < 1e-12
    print(f"  far commutation (disjoint gates swap freely): err {e:.1e}")
    return ok


if __name__ == "__main__":
    print("verifying the simulator against known cases:")
    ok = _verify()
    print("all checks passed" if ok else "FAILED")
    for n in (4, 5, 6):
        W, Y = build(300, n, (4, 12), seed=n)
        print(f"\n{n} qubits: <Z> mean {Y.mean():+.3f}  std {Y.std():.3f}  "
              f"range [{Y.min():+.2f}, {Y.max():+.2f}]")
