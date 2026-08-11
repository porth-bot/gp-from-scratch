"""Blocked triangular solves: against the general solver, and against the
structure they are supposed to exploit."""

import numpy as np
import pytest

from gp.linalg import cho_solve, solve_lower, solve_upper


def _factor(n, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    K = A @ A.T + n * np.eye(n)
    return np.linalg.cholesky(K), K, rng


# -- agreement with the solver being replaced ---------------------------------


@pytest.mark.parametrize("n", [1, 2, 7, 64, 65, 200])
@pytest.mark.parametrize("k", [1, 5])
def test_lower_matches_the_general_solver_at_every_size(n, k):
    """Including sizes that are not multiples of the block, and n < block --
    off-by-one in the last partial block is the obvious way to write this
    wrong, and it would show up only at those sizes."""
    L, _, rng = _factor(n)
    B = rng.standard_normal((n, k))
    assert np.allclose(solve_lower(L, B), np.linalg.solve(L, B), atol=1e-11)


@pytest.mark.parametrize("n", [1, 2, 7, 64, 65, 200])
def test_upper_matches_the_general_solver_at_every_size(n):
    L, _, rng = _factor(n)
    B = rng.standard_normal((n, 3))
    U = L.T
    assert np.allclose(solve_upper(U, B), np.linalg.solve(U, B), atol=1e-11)


@pytest.mark.parametrize("block", [1, 3, 16, 64, 512])
def test_the_block_size_changes_the_arithmetic_but_not_the_answer(block):
    """Block size is a performance knob and nothing else. If a block size ever
    changes an answer by more than float noise, the loop bookkeeping is wrong."""
    L, _, rng = _factor(129)
    B = rng.standard_normal((129, 4))
    reference = np.linalg.solve(L, B)
    assert np.abs(solve_lower(L, B, block=block) - reference).max() < 1e-11
    assert np.abs(solve_upper(L.T, B, block=block)
                  - np.linalg.solve(L.T, B)).max() < 1e-11


def test_cho_solve_inverts_the_matrix_it_factorized():
    """The residual, which is the property substitution actually guarantees."""
    L, K, rng = _factor(150)
    B = rng.standard_normal((150, 2))
    X = cho_solve(L, B)
    assert np.abs(K @ X - B).max() < 1e-9


def test_vector_in_vector_out():
    """A 1-D right-hand side comes back 1-D, matching np.linalg.solve's
    convention -- callers pass alpha = K^-1 y and index it as a vector."""
    L, K, rng = _factor(40)
    y = rng.standard_normal(40)
    x = cho_solve(L, y)
    assert x.shape == (40,)
    assert np.allclose(K @ x, y, atol=1e-10)
    assert solve_lower(L, y).shape == (40,)
    assert solve_upper(L.T, y).shape == (40,)


# -- the structural claims -----------------------------------------------------


def test_only_the_triangle_is_read():
    """The strictly-upper entries of a lower factor are never touched. This is
    what licenses passing a Cholesky factor straight in without masking it, and
    it is checked by filling the unused triangle with garbage: an
    implementation that read it would produce a different answer."""
    L, _, rng = _factor(100)
    B = rng.standard_normal((100, 2))
    clean = solve_lower(L, B)
    dirty = L.copy()
    dirty[np.triu_indices(100, 1)] = 1e6
    assert np.array_equal(solve_lower(dirty, B), clean)

    upper_clean = solve_upper(L.T, B)
    dirty_u = L.T.copy()
    dirty_u[np.tril_indices(100, -1)] = 1e6
    assert np.array_equal(solve_upper(dirty_u, B), upper_clean)


def test_forward_substitution_is_actually_forward():
    """Row i of the solution depends only on rows <= i of B. Perturbing the
    last entry of the right-hand side must leave every earlier entry of the
    solution bit-identical -- the property that makes it substitution rather
    than a general solve."""
    L, _, rng = _factor(80)
    y = rng.standard_normal(80)
    x = solve_lower(L, y)
    y2 = y.copy()
    y2[-1] += 3.0
    assert np.array_equal(solve_lower(L, y2)[:-1], x[:-1])


def test_solves_the_identity_to_the_inverse():
    """Many right-hand sides at once: L^-1 as a matrix, checked against L."""
    L, _, _ = _factor(70)
    L_inv = solve_lower(L, np.eye(70))
    assert np.abs(L @ L_inv - np.eye(70)).max() < 1e-10


def test_ill_conditioned_factor_is_no_worse_than_the_solver_it_replaces():
    """What substitution guarantees is *backward* stability: the residual is
    small relative to ||L|| ||x||, not relative to ||b||. On a factor whose
    inverse overflows into 1e44, the forward error is enormous and the absolute
    residual is enormous with it -- and neither is a defect. The measure that
    means something is the backward error, and the standard to hold it to is
    the LAPACK path this replaces.
    """
    d = np.geomspace(1.0, 1e-8, 120)
    rng = np.random.default_rng(3)
    L = np.tril(rng.standard_normal((120, 120)) * 0.01, -1) + np.diag(d)
    b = rng.standard_normal(120)

    def backward_error(x):
        scale = np.abs(L).sum(axis=1).max() * np.abs(x).max()
        return float(np.abs(L @ x - b).max() / scale)

    ours = backward_error(solve_lower(L, b))
    lapack = backward_error(np.linalg.solve(L, b))
    assert ours < 1e-14                       # a few eps, as substitution gives
    assert ours <= 10 * lapack                # and no worse than what it replaces


# -- guards --------------------------------------------------------------------


def test_shape_mismatches_are_refused():
    L, _, _ = _factor(10)
    with pytest.raises(ValueError, match="square"):
        solve_lower(np.ones((3, 4)), np.ones(3))
    with pytest.raises(ValueError, match="does not match"):
        solve_lower(L, np.ones(9))
    with pytest.raises(ValueError, match="does not match"):
        cho_solve(L, np.ones((9, 2)))
