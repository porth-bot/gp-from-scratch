"""Triangular solves, blocked, in NumPy -- because ``np.linalg.solve`` is not one.

Everything in this repo that conditions on data ends in the same two lines: a
Cholesky ``K = L L^T``, then solves against ``L`` and ``L^T``. The Cholesky is
the expensive step in the complexity table, at n^3/3 flops, and the solves are
supposed to be the cheap ones, at n^2 each.

They were not. ``np.linalg.solve(L, B)`` does not know that ``L`` is
triangular -- LAPACK's ``gesv`` runs a general LU factorization with partial
pivoting first, at 2 n^3/3 flops, and only then substitutes. Calling it on an
already-triangular factor throws away the factorization that was just computed
and pays *twice the Cholesky* to compute a worse one. Measured at n = 8000 on
this machine: Cholesky 1.00 s, and each ``np.linalg.solve`` against its factor
2.07 s. An exact GP fit + predict spent about 6 of its 7.7 seconds re-doing
work it already had.

NumPy exposes no triangular solve (SciPy does, as ``solve_triangular``, and
this repo is NumPy-only by design), so here is one. Blocked forward
substitution, which is the standard formulation and is almost all GEMM:

    partition L into b x b diagonal blocks L_ii and the panels below them.
    For each block row i:  X_i = L_ii^{-1} (B_i - L_{i,<i} X_{<i}).

The panel product is a matrix multiply into BLAS at n^2 k / 2 flops total; the
only non-GEMM work is a b x b solve per block row, and with b small that costs
2 n b^2 / 3, which is negligible next to it. The result agrees with the LU path
to 3.6e-16 while being 313x faster on one right-hand side and 34x faster on 200.

Block size: 64 by default, chosen by measurement, not taste (n = 8000, this
machine):

    b        32      64     128     256     512    1024
    1 rhs   6.6ms   6.6ms   7.4ms  10.4ms  24.6ms  70.9ms
    200     63ms    61ms    63ms    57ms    67ms    97ms

Flat from 32 to 128 and then degrading, because the b x b LU on each diagonal
block grows as b^3 while the number of Python-level iterations it saves falls
only as 1/b. Anywhere in the flat region is fine; the top of the table shows
what happens if the block grows until the diagonal solves become the cost.

These are the standard algorithms, and the backward-stability results for
substitution (Higham 2002, Ch. 8) apply as usual: the computed solution is the
exact solution of a nearby triangular system, with the usual caveat that "the
error is small" is a statement about the residual and not about ill-conditioned
inverses.
"""

from __future__ import annotations

import numpy as np

BLOCK = 64


def _prepare(A: np.ndarray, B: np.ndarray) -> "tuple[np.ndarray, np.ndarray, bool]":
    """Validate shapes; promote a vector right-hand side to one column."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"triangular factor must be square, got {A.shape}")
    vector = B.ndim == 1
    if vector:
        B = B.reshape(-1, 1)
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError(
            f"right-hand side of shape {B.shape} does not match a "
            f"{A.shape[0]} x {A.shape[0]} factor"
        )
    return A, B, vector


def solve_lower(L: np.ndarray, B: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Solve ``L X = B`` for lower-triangular ``L``, by blocked substitution.

    The strictly-upper entries of ``L`` are never read, so a factor stored in a
    full array alongside junk above the diagonal is fine -- which is what
    ``np.linalg.cholesky`` returns anyway (zeros).

    Examples
    --------
    Agrees with the general solver, at a fraction of the work:

    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> A = rng.standard_normal((60, 60))
    >>> L = np.linalg.cholesky(A @ A.T + 60 * np.eye(60))
    >>> B = rng.standard_normal((60, 3))
    >>> bool(np.allclose(solve_lower(L, B), np.linalg.solve(L, B), atol=1e-12))
    True

    A vector in, a vector out:

    >>> solve_lower(L, B[:, 0]).shape
    (60,)
    """
    L, B, vector = _prepare(L, B)
    n = L.shape[0]
    X = np.empty_like(B)
    for start in range(0, n, block):
        stop = min(start + block, n)
        rhs = B[start:stop] - L[start:stop, :start] @ X[:start]
        # np.tril on the diagonal block, not on all of L: the panel to the
        # left is used whole, but the block itself would otherwise have its
        # strictly-upper entries read by the general solver, and then "only the
        # triangle is read" would be false exactly where it is easiest to miss.
        # b x b per block row, so O(n b) elements copied in total.
        X[start:stop] = np.linalg.solve(np.tril(L[start:stop, start:stop]), rhs)
    return X[:, 0] if vector else X


def solve_upper(U: np.ndarray, B: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Solve ``U X = B`` for upper-triangular ``U``, by blocked back-substitution.

    The mirror image of `solve_lower`: sweep the block rows bottom to top, and
    subtract the panel to the *right* of each diagonal block.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(1)
    >>> A = rng.standard_normal((50, 50))
    >>> U = np.linalg.cholesky(A @ A.T + 50 * np.eye(50)).T
    >>> B = rng.standard_normal((50, 2))
    >>> bool(np.allclose(solve_upper(U, B), np.linalg.solve(U, B), atol=1e-12))
    True
    """
    U, B, vector = _prepare(U, B)
    n = U.shape[0]
    X = np.empty_like(B)
    for stop in range(n, 0, -block):
        start = max(stop - block, 0)
        rhs = B[start:stop] - U[start:stop, stop:] @ X[stop:]
        X[start:stop] = np.linalg.solve(np.triu(U[start:stop, start:stop]), rhs)
    return X[:, 0] if vector else X


def cho_solve(L: np.ndarray, B: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """Solve ``(L L^T) X = B`` given the Cholesky factor: two triangular solves.

    The operation every GP posterior is made of. Forward through ``L``, then
    backward through ``L^T`` -- and the inverse is never formed, which matters
    for accuracy as well as cost: ``K^{-1}`` computed explicitly and then
    multiplied loses the backward stability that substitution has.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(2)
    >>> A = rng.standard_normal((40, 40))
    >>> K = A @ A.T + 40 * np.eye(40)
    >>> L = np.linalg.cholesky(K)
    >>> y = rng.standard_normal(40)
    >>> bool(np.allclose(K @ cho_solve(L, y), y, atol=1e-10))
    True
    """
    return solve_upper(L.T, solve_lower(L, B, block), block)
