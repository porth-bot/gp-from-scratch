"""Sparse GP regression via inducing points and the Titsias variational bound.

An exact GP costs O(n^3) because the model lives in the n x n Gram matrix.
Section 8 bought O(n) scaling by replacing the *kernel* with a randomized
finite-rank surrogate, and paid variance starvation for it (gp/rff.py). This
module takes the other route: keep the model exactly as it is, and approximate
the *posterior* by summarizing f through M inducing variables u = f(Z) at
inducing inputs Z (Titsias 2009; derived in theory/derivations.md, Sec. 9).

Augmenting the model with u changes nothing -- u is the same GP evaluated at M
more inputs -- so the marginal likelihood being approximated is the real one.
With the variational family q(f, u) = p(f | u) q(u), and q(u) collapsed to its
optimum, the evidence lower bound is

    F = log N(y | 0, Qff + sigma^2 I) - (1 / (2 sigma^2)) tr(Kff - Qff),

    Qff = Kfu Kuu^{-1} Kuf     (the Nystrom approximation to Kff).

The two terms are worth separating. The first is the log evidence of a
different, low-rank model (DTC); on its own it is not a bound on anything and
can sit *above* the true evidence, because Qff is a smaller covariance than
Kff. The second term is what makes F a bound: it is a sum of n non-negative
terms k(x_i, x_i) - q(x_i, x_i), each the posterior variance of f(x_i) given
u -- the part of f the inducing set fails to explain. It is the penalty on a
bad inducing set, and it is measured in units of sigma^2, so the same
geometric shortfall costs more when the data are precise.

Because Z appears only in q and never in the model, log p(y) does not depend on
it: moving Z cannot change what is being bounded, only how tight the bound is.
That is why the inducing locations can be optimized like any other parameter
without the overfitting risk a model parameter would carry -- and it is the
structural difference from FITC, where Z do enter the model.

Exact-recovery limit: with Z = X, Qff = Kff, the trace term vanishes, and both
F and the predictive equations reduce to the exact GP of gp/gp.py. That is
tests/test_sparse.py's first test, and it pins the algebra to an already-
trusted implementation before anything approximate is measured.

Numerics: no inverse is formed and nothing larger than M x M is factorized.
With Luu = chol(Kuu + jitter I), A = Luu^{-1} Kuf, B = I_M + sigma^{-2} A A^T
and LB = chol(B), the matrix determinant lemma and Woodbury give the bound in
O(n M^2) time and O(n M) memory (Sec. 9.5). The trace term uses only diag(Kff),
never the full matrix.

Gradients (Sec. 9.6). All three blocks -- kernel hyperparameters, log sigma^2,
and the inducing locations Z -- contract against the same two matrices

    R = P H     (M, n),      S = P H P^T   (M, M),
    P = Kuu^-1 Kuf,          H = alpha alpha^T - W^-1 + sigma^-2 I,

with W = Qff + sigma^2 I and alpha = W^-1 y. H is n x n and is never formed;
Woodbury turns both into O(n M^2). The +sigma^-2 I in H is the trace penalty's
contribution folded in, so the penalty is not a separate gradient but a
rank-correction to the matrix everything else already contracts against. Kff
survives differentiation only through its diagonal, which is why the trace
gradient goes through `kernel_grad_diag` and not `kernel.grads(X)`.

Honest limitations of what is here:

- The bound is *collapsed*, so it needs all of y at once: no minibatching.
  SVGP (Hensman et al. 2013) is the uncollapsed version that does.
- Gaussian likelihood only. The collapsed form is a Gaussian-conjugacy result.
- The gradients exist but nothing drives them yet: joint ML-II over theta and Z
  is the next piece, and until it lands the caller still chooses Z.
- Z-gradients need a kernel that is differentiable in its inputs, which rules
  out Gibbs (non-stationary, derivative not derived) and Matern nu=0.5 (a real
  cusp at r=0). Both raise from `kernel.dK_dX1` rather than returning a number.
- The gradient is of the *loaded* bound, jitter included -- see `_grad_params`
  for why ignoring a 1e-10 loading turned out to be a 1e-4 error. The Z-block
  needs no such correction only because every kernel with an input derivative
  here is stationary, so mean(diag Kuu) does not depend on Z; a non-stationary
  kernel with a dK_dX1 would need the term added.

References
----------
Titsias (2009), Variational learning of inducing variables in sparse GPs.
Quinonero-Candela & Rasmussen (2005), A unifying view of sparse approximate GPs.
Hensman, Fusi & Lawrence (2013), Gaussian processes for big data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from gp.kernels import Kernel


def kernel_diag(kernel: "Kernel", X: np.ndarray, block: int = 256) -> np.ndarray:
    """diag(k(X, X)) without ever materializing the n x n matrix.

    `np.diag(kernel(X, X))` is the obvious spelling and it costs O(n^2) memory,
    which defeats the point of an O(n M) method -- the trace term and the
    predictive variance both need only the diagonal. Evaluating in b x b blocks
    keeps the peak at O(b^2) and stays vectorized, unlike a per-point loop.

    Kernel-agnostic on purpose: stationary kernels have a constant diagonal
    k(0) = s2, but ``Gibbs`` and sums/products of anything non-stationary do
    not, and this has to be right for all of them.

    Examples
    --------
    >>> import numpy as np
    >>> from gp.kernels import RBF, Gibbs
    >>> X = np.linspace(-2, 2, 300).reshape(-1, 1)
    >>> for k in (RBF(s2=1.7, l=0.4), Gibbs(s2=1.1, a=0.5, b=0.3)):
    ...     bool(np.allclose(kernel_diag(k, X, block=32), np.diag(k(X, X))))
    True
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    out = np.empty(n)
    for start in range(0, n, block):
        stop = min(start + block, n)
        chunk = X[start:stop]
        out[start:stop] = np.diag(kernel(chunk, chunk))
    return out


def kernel_grad_diag(kernel: "Kernel", X: np.ndarray,
                     block: int = 256) -> "list[np.ndarray]":
    """diag(dk(X, X)/dtheta_i) for each free parameter, without the n x n.

    The trace penalty's hyperparameter gradient needs sum_i dk(x_i,x_i)/dtheta,
    the exact analogue of what `kernel_diag` does for the penalty itself. The
    obvious spelling, ``[np.diag(g) for g in kernel.grads(X)]``, allocates one
    n x n matrix *per parameter* -- worse than the O(n^2) `kernel_diag` avoids,
    by a factor of the parameter count. Blocking keeps the peak at O(b^2).

    Returns one (n,) array per free parameter, in ``theta`` order.

    Examples
    --------
    >>> import numpy as np
    >>> from gp.kernels import RBF, Periodic
    >>> X = np.linspace(-2, 2, 50).reshape(-1, 1)
    >>> k = RBF(s2=1.3, l=0.7) * Periodic(s2=1.0, l=0.9, p=2.0)
    >>> blocked = kernel_grad_diag(k, X, block=16)
    >>> full = [np.diag(g) for g in k.grads(X)]
    >>> [bool(np.allclose(a, b)) for a, b in zip(blocked, full)]
    [True, True, True, True, True]
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    out: "list[np.ndarray] | None" = None
    for start in range(0, n, block):
        stop = min(start + block, n)
        chunk = X[start:stop]
        grads = kernel.grads(chunk)
        if out is None:
            out = [np.empty(n) for _ in grads]
        for dst, g in zip(out, grads):
            dst[start:stop] = np.diag(g)
    assert out is not None                      # n >= 1 is enforced upstream
    return out


class SGPR:
    """Sparse variational GP regression (Titsias' collapsed bound).

    Parameters
    ----------
    kernel : gp.kernels.Kernel
    Z : array of shape (M, d)
        Inducing inputs. Copied on construction, so later mutation of the
        caller's array does not silently invalidate a fit.
    noise_var : float
        Observation-noise variance sigma^2, stored in log space to match
        ``GPRegressor``.
    jitter : float
        Diagonal loading on ``Kuu``, relative to its mean diagonal. Unlike the
        jitter on ``K_y`` in ``GPRegressor``, this one is not hygiene: ``Kuu``
        carries no noise term and a good inducing set is a nearly-collinear
        one, so ``Kuu`` really does approach singularity. It also sets the
        floor on how exactly ``Z = X`` can reproduce the exact GP -- the
        recovery error is O(jitter), which ``tests/test_sparse.py`` measures
        rather than assumes.

    Attributes set by ``fit``: ``L_uu``, ``A``, ``L_B``, ``c`` (the factors of
    Sec. 9.5), plus ``trace_term`` and ``elbo``.

    Examples
    --------
    With the inducing points placed on the data (M = n) the approximation is
    not an approximation: the bound equals the exact log marginal likelihood
    and the trace penalty is zero, up to the jitter on ``Kuu``.

    >>> import numpy as np
    >>> from gp.gp import GPRegressor
    >>> from gp.kernels import RBF
    >>> rng = np.random.default_rng(0)
    >>> X = np.sort(rng.uniform(-3, 3, 40)).reshape(-1, 1)
    >>> y = np.sin(X).ravel() + 0.1 * rng.standard_normal(40)
    >>> exact = GPRegressor(RBF(s2=1.0, l=1.0), noise_var=0.05).fit(X, y)
    >>> sparse = SGPR(RBF(s2=1.0, l=1.0), Z=X, noise_var=0.05).fit(X, y)
    >>> bool(abs(sparse.elbo() - exact.log_marginal_likelihood()) < 1e-7)
    True
    >>> bool(sparse.trace_term() < 1e-8)
    True

    Move the inducing points off the data and the bound drops strictly below
    the exact evidence, paying the trace penalty for what u no longer explains:

    >>> few = SGPR(RBF(s2=1.0, l=1.0), Z=np.linspace(-3, 3, 5).reshape(-1, 1),
    ...            noise_var=0.05).fit(X, y)
    >>> bool(few.elbo() < exact.log_marginal_likelihood())
    True
    >>> bool(few.trace_term() > 0.0)
    True
    """

    JITTER = 1e-10

    def __init__(self, kernel: "Kernel", Z: np.ndarray, noise_var: float = 0.1,
                 jitter: float = JITTER):
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        if Z.shape[0] < 1:
            raise ValueError("need at least one inducing point")
        if jitter < 0:
            raise ValueError("jitter must be non-negative")
        self.kernel = kernel
        self._Z = Z.copy()
        self.jitter = float(jitter)
        self.log_noise = float(np.log(noise_var))
        self._fitted = False

    # -- parameter vector: kernel theta plus log noise -----------------------
    #
    # Z is deliberately NOT in here. It is a variational parameter, not a model
    # parameter (Sec. 9.2), so it is optimized against its own gradient
    # (``elbo_and_grads`` returns it separately) rather than being concatenated
    # into ``params`` -- keeping the distinction visible in the interface.

    @property
    def Z(self) -> np.ndarray:
        return self._Z

    @Z.setter
    def Z(self, value: np.ndarray) -> None:
        value = np.atleast_2d(np.asarray(value, dtype=float))
        if value.shape != self._Z.shape:
            raise ValueError(
                f"Z has shape {value.shape}, expected {self._Z.shape}"
            )
        self._Z = value.copy()
        self._fitted = False

    @property
    def params(self) -> np.ndarray:
        return np.concatenate([self.kernel.theta, [self.log_noise]])

    @params.setter
    def params(self, value: np.ndarray) -> None:
        self.kernel.theta = value[:-1]
        self.log_noise = float(value[-1])
        self._fitted = False

    @property
    def noise_var(self) -> float:
        return float(np.exp(self.log_noise))

    @property
    def n_inducing(self) -> int:
        return int(self.Z.shape[0])

    # -- fitting ---------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SGPR":
        """Build the O(n M^2) factorization the bound and predictions share."""
        self.X = np.atleast_2d(np.asarray(X, dtype=float))
        self.y = np.asarray(y, dtype=float)
        if self.X.shape[1] != self.Z.shape[1]:
            raise ValueError(
                f"inducing inputs have {self.Z.shape[1]} dims, data has "
                f"{self.X.shape[1]}"
            )
        n = len(self.y)
        M = self.n_inducing
        sn2 = self.noise_var

        Kuu = self.kernel(self.Z, self.Z)
        Kuf = self.kernel(self.Z, self.X)                       # (M, n)
        kff_diag = kernel_diag(self.kernel, self.X)   # never builds the n x n Kff

        # Kuu carries no noise term and the informative inducing sets are the
        # nearly-collinear ones, so this jitter is doing real work, not hygiene.
        Kuu = Kuu + self.jitter * float(np.mean(np.diag(Kuu))) * np.eye(M)

        self.L_uu = np.linalg.cholesky(Kuu)
        self.A = np.linalg.solve(self.L_uu, Kuf)                # (M, n), Qff = A^T A
        B = np.eye(M) + (self.A @ self.A.T) / sn2
        self.L_B = np.linalg.cholesky(B)
        self.c = np.linalg.solve(self.L_B, self.A @ self.y) / sn2   # (M,)

        # tr(Kff - Qff) = sum_i k(x_i, x_i) - ||A||_F^2, never forming Kff.
        # Clipped at 0: it is provably non-negative (a conditional covariance
        # has non-negative diagonal), so any negative value is float noise.
        self._trace_term = float(max(np.sum(kff_diag) - np.sum(self.A**2), 0.0))
        self._n = n
        self._yTy = float(self.y @ self.y)
        self._fitted = True
        return self

    # -- the bound -------------------------------------------------------------

    def trace_term(self) -> float:
        """tr(Kff - Qff): the part of f the inducing set does not explain.

        Non-negative, zero iff u determines f at the data (e.g. Z = X). This is
        the quantity that penalizes a bad inducing set; the bound subtracts
        ``trace_term / (2 sigma^2)``.
        """
        assert self._fitted
        return self._trace_term

    def dtc_log_evidence(self) -> float:
        """log N(y | 0, Qff + sigma^2 I) -- the bound's first term alone.

        This is the DTC / projected-process approximation's log evidence. It is
        NOT a lower bound on anything: Qff <= Kff in the PSD order, so a model
        with too few inducing points can report an evidence above the exact GP's
        by claiming the data are less surprising than they are. Exposed
        separately because seeing the two terms move against each other is the
        whole content of Sec. 9.3.
        """
        assert self._fitted
        sn2 = self.noise_var
        log_det = self._n * np.log(sn2) + 2.0 * float(
            np.sum(np.log(np.diag(self.L_B)))
        )
        quad = self._yTy / sn2 - float(self.c @ self.c)
        return float(-0.5 * (log_det + quad + self._n * np.log(2.0 * np.pi)))

    def elbo(self) -> float:
        """The Titsias free energy F -- a lower bound on the exact log evidence.

        F = log N(y | 0, Qff + sigma^2 I) - tr(Kff - Qff) / (2 sigma^2).

        Because it bounds the log marginal likelihood of the *unaugmented*
        model, it is comparable across different M and different Z: a larger F
        means a tighter approximation to the same fixed quantity.
        """
        assert self._fitted
        return self.dtc_log_evidence() - self._trace_term / (2.0 * self.noise_var)

    # -- gradients of the bound -------------------------------------------------

    def _contractors(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
        """R = P H, S = P H P^T, alpha = W^-1 y, and tr(W^-1) -- Sec. 9.6.

        H = alpha alpha^T - W^-1 + sigma^-2 I is n x n and is never formed.
        Two Woodbury identities collapse it. With A = Luu^-1 Kuf and
        B = I + sigma^-2 A A^T (so that A A^T / sigma^2 = B - I):

            A W^-1 = sigma^-2 (A - A A^T B^-1 A / sigma^2) = sigma^-2 B^-1 A
            alpha  = (y - A^T L_B^-T c) / sigma^2

        so P H = Luu^-T [ (A alpha) alpha^T + sigma^-2 (A - B^-1 A) ], which is
        M x n throughout. S is symmetric in exact arithmetic (H is); it is
        symmetrized explicitly because the two triangular solves that build it
        do not enforce that to the last bit, and an asymmetric S would leak an
        antisymmetric part into the Z-gradient's Kuu term.
        """
        assert self._fitted
        sn2 = self.noise_var
        A, L_B = self.A, self.L_B

        alpha = (self.y - A.T @ np.linalg.solve(L_B.T, self.c)) / sn2   # (n,)
        BinvA = np.linalg.solve(L_B.T, np.linalg.solve(L_B, A))         # (M, n)

        PH = np.linalg.solve(
            self.L_uu.T, np.outer(A @ alpha, alpha) + (A - BinvA) / sn2
        )                                                               # (M, n)
        P = np.linalg.solve(self.L_uu.T, A)                             # (M, n)
        S = PH @ P.T
        S = 0.5 * (S + S.T)

        # tr W^-1 = sigma^-2 (n - sigma^-2 tr(A^T B^-1 A)), same identity.
        tr_W_inv = (self._n - float(np.sum(A * BinvA)) / sn2) / sn2
        return PH, S, alpha, tr_W_inv

    def elbo_and_grads(self) -> "tuple[float, np.ndarray, np.ndarray]":
        """The bound, d F/d params, and d F/d Z (Sec. 9.6).

        Returns ``(F, grad_params, grad_Z)`` with ``grad_params`` matching
        ``self.params`` entry for entry (kernel theta, then log sigma^2) and
        ``grad_Z`` shaped like ``Z``.

        The Z-gradient is two opposing forces, (9.20): ``R`` pulls each
        inducing point toward data the current set explains badly, ``S`` pushes
        it off the other inducing points. The Kuu diagonal needs no special
        case -- k(z, z) is constant for a stationary kernel, so its input
        derivative is zero and the j = m term drops out on its own.

        Raises ``NotImplementedError`` (from the kernel) if the kernel has no
        input-space derivative; ``elbo_grad_params`` still works in that case,
        since only the Z block needs one.
        """
        PH, S, alpha, tr_W_inv = self._contractors()
        grad_params = self._grad_params(PH, S, alpha, tr_W_inv)
        Z, X = self.Z, self.X

        dKuf = self.kernel.dK_dX1(Z, X)          # (M, n, d)
        dKuu = self.kernel.dK_dX1(Z, Z)          # (M, M, d)
        grad_Z = (np.einsum("mj,mja->ma", PH, dKuf)
                  - np.einsum("mj,mja->ma", S, dKuu))
        return self.elbo(), grad_params, grad_Z

    def elbo_grad_params(self) -> "tuple[float, np.ndarray]":
        """The bound and its gradient w.r.t. ``params`` only (no Z block).

        The signature ``adam_maximize`` and friends expect. Separate from
        ``elbo_and_grads`` because it works for every kernel, including the
        ones with no input-space derivative.
        """
        PH, S, alpha, tr_W_inv = self._contractors()
        return self.elbo(), self._grad_params(PH, S, alpha, tr_W_inv)

    def _grad_params(self, PH: np.ndarray, S: np.ndarray,
                     alpha: np.ndarray, tr_W_inv: float) -> np.ndarray:
        """(9.18) for the kernel block, (9.19) for the noise.

        The jitter is differentiated, not ignored. ``fit`` factorizes
        ``Kuu + j * mean(diag Kuu) I``, and that loading is itself a function of
        theta (through the signal variance), so the honest derivative carries
        ``+ j * mean(diag dKuu) I``. Dropping it looks harmless -- j is 1e-10 --
        and is not: the term enters contracted against S = P H P^T, and P
        carries a factor of Kuu^{-1}, so the error scales like j * ||P||^2 and
        grows with the *square* of Kuu's conditioning. Measured on a Periodic
        kernel whose inducing points sit near-duplicate under its own period,
        omitting it put the log-s2 gradient 1.3e-4 out -- five orders of
        magnitude worse than j, and enough to fail a finite-difference check
        that RBF passes comfortably. Including it is one line.
        """
        sn2 = self.noise_var
        dKuf = self.kernel.grads(self.Z, self.X)       # cross-covariance grads
        dKuu = self.kernel.grads(self.Z, self.Z)
        dkff = kernel_grad_diag(self.kernel, self.X)
        eye = np.eye(self.n_inducing)

        grads = [
            float(np.sum(PH * guf))
            - 0.5 * float(np.sum(
                S * (guu + self.jitter * float(np.mean(np.diag(guu))) * eye)
            ))
            - float(np.sum(gff)) / (2.0 * sn2)
            for guf, guu, gff in zip(dKuf, dKuu, dkff)
        ]

        # Noise: the exact-GP gradient plus the trace penalty's own sigma^2
        # dependence, which always pushes the fitted noise UP -- variation the
        # inducing set cannot explain is cheaper to call noise (Sec. 9.6).
        grads.append(sn2 * (
            0.5 * (float(alpha @ alpha) - tr_W_inv)
            + self._trace_term / (2.0 * sn2**2)
        ))
        return np.array(grads)

    # -- prediction ------------------------------------------------------------

    def predict(self, Xs: np.ndarray,
                include_noise: bool = False) -> "tuple[np.ndarray, np.ndarray]":
        """Posterior mean and pointwise variance at Xs (latent f by default).

        mean = tmp^T c and var = diag(K**) - colsum(As^2) + colsum(tmp^2),
        with As = Luu^{-1} Ku* and tmp = LB^{-1} As (Sec. 9.5). The two variance
        corrections cancel far from every inducing point, so the band relaxes to
        the prior there -- the property random Fourier features lose.
        """
        assert self._fitted
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Kus = self.kernel(self.Z, Xs)                            # (M, n*)
        As = np.linalg.solve(self.L_uu, Kus)                     # (M, n*)
        tmp = np.linalg.solve(self.L_B, As)                      # (M, n*)

        mean = tmp.T @ self.c
        kss = kernel_diag(self.kernel, Xs)
        var = kss - np.sum(As**2, axis=0) + np.sum(tmp**2, axis=0)
        var = np.maximum(var, 0.0)
        if include_noise:
            var = var + self.noise_var
        return mean, var
