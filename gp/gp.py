"""Exact GP regression via Cholesky, with the marginal likelihood and its gradient.

Model: f ~ GP(0, k), y = f(X) + eps, eps ~ N(0, sigma^2 I). Everything below
follows from one fact -- (f(X*), y) is jointly Gaussian -- and the Gaussian
conditioning formula (derived via the Schur complement in
theory/derivations.md, Sec. 1):

    mean = K*^T (K + sigma^2 I)^{-1} y
    cov  = K** - K*^T (K + sigma^2 I)^{-1} K*

Numerics: the inverse is never formed. With L = chol(K + sigma^2 I):

    alpha = L^{-T} L^{-1} y          (two triangular solves)
    mean  = K*^T alpha
    v = L^{-1} K*,  var = diag(K**) - sum(v^2, axis=0)
    log|K + sigma^2 I| = 2 sum_i log L_ii

Cholesky is the right factorization because the matrix is SPD, it is
backward-stable, costs n^3/3 (half an LU), and exposes the log-determinant
for free. A tiny "jitter" is added to the diagonal before factorizing --
finitely-sampled smooth kernels are often numerically semidefinite (nearby
points give nearly identical rows), and jitter of 1e-10 * mean(diag) restores
positive definiteness at noise levels far below anything statistical.

Type-II maximum likelihood ("ML-II", "empirical Bayes"): choose kernel
hyperparameters and noise by maximizing

    log p(y | theta) = -1/2 y^T alpha - sum_i log L_ii - n/2 log 2pi.

The three terms are data fit, complexity penalty, and a constant -- the
built-in Occam's razor: a kernel flexible enough to fit anything pays
through log|K|. The gradient, derived in theory/derivations.md Sec. 2:

    d/dtheta_j log p(y | theta) = 1/2 tr[ (alpha alpha^T - K^{-1}) dK/dtheta_j ].

Noise enters as theta_noise = log sigma^2 with dK/d(log sigma^2) = sigma^2 I.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from gp.kernels import Kernel


def _chol_solve(L: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve (L L^T) x = B via two triangular solves (never form the inverse)."""
    return np.linalg.solve(L.T, np.linalg.solve(L, B))


def sample_prior(
    kernel: "Kernel",
    X: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    jitter: float = 1e-8,
) -> np.ndarray:
    """Draw sample paths f ~ GP(0, k) evaluated at inputs ``X``.

    The GP prior is, by definition, a zero-mean Gaussian with covariance
    ``K = k(X, X)`` over any finite set of inputs. Factor ``K = L L^T`` and
    push standard-normal draws through ``L``: if ``z ~ N(0, I)`` then
    ``L z ~ N(0, K)``. ``jitter`` (relative to the mean diagonal) is added so
    the Cholesky succeeds on smooth kernels, whose Gram matrices are only
    numerically semidefinite when inputs are dense.

    Returns an ``(n_samples, len(X))`` array -- one prior function per row.
    These are draws from the *prior* (no data conditioned on); the visual
    point is how kernel choice alone sets sample-path smoothness and structure.

    Examples
    --------
    The draws really do have covariance ``K``: with enough of them, the
    empirical covariance converges to the kernel matrix.

    >>> import numpy as np
    >>> from gp.kernels import RBF
    >>> X = np.linspace(0, 1, 5).reshape(-1, 1)
    >>> rng = np.random.default_rng(0)
    >>> F = sample_prior(RBF(s2=2.0, l=0.5), X, n_samples=20000, rng=rng)
    >>> F.shape
    (20000, 5)
    >>> bool(np.allclose(np.cov(F.T), RBF(s2=2.0, l=0.5)(X, X), atol=0.1))
    True
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    n = X.shape[0]
    K = kernel(X, X)
    K = K + jitter * float(np.mean(np.diag(K))) * np.eye(n)
    L = np.linalg.cholesky(K)
    z = rng.standard_normal((n, n_samples))
    return (L @ z).T


class GPRegressor:
    """Zero-mean exact GP regression.

    Parameters
    ----------
    kernel : gp.kernels.Kernel
    noise_var : float
        Initial observation-noise variance sigma^2 (optimized in log space
        alongside the kernel's theta when using gp.optimize).

    Examples
    --------
    Fit five noiseless samples of ``sin`` and predict. With tiny noise the
    posterior mean passes through the data, and the posterior standard
    deviation collapses at an observed input but is large far away, where the
    GP falls back on the prior (here ``s2 = 1``):

    >>> import numpy as np
    >>> from gp.kernels import RBF
    >>> X = np.linspace(0, np.pi, 5).reshape(-1, 1)
    >>> y = np.sin(X).ravel()
    >>> gp = GPRegressor(RBF(s2=1.0, l=1.0), noise_var=1e-8).fit(X, y)
    >>> mean, var = gp.predict(X)
    >>> bool(np.allclose(mean, y, atol=1e-4))        # interpolates the data
    True
    >>> bool(np.sqrt(var).max() < 1e-3)              # certain where it has data
    True
    >>> _, var_far = gp.predict(np.array([[100.0]]))  # far from every input
    >>> bool(np.isclose(var_far[0], 1.0, atol=1e-6))  # relaxes to the prior s2
    True
    """

    JITTER = 1e-10

    def __init__(self, kernel: "Kernel", noise_var: float = 0.1):
        self.kernel = kernel
        self.log_noise = float(np.log(noise_var))
        self._fitted = False

    # -- parameter vector: kernel theta plus log noise -----------------------

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

    # -- fitting and prediction ----------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray,
            noise: Optional[np.ndarray] = None) -> "GPRegressor":
        """Condition on data.

        noise : optional per-point observation-noise *variance*, shape (n,).
            Default (None) puts the scalar ``self.noise_var`` on every
            diagonal entry (homoscedastic, the standard GP). A length-n vector
            enables **heteroscedastic** regression -- each point gets its own
            noise variance, e.g. the two-stage estimate in
            ``experiments/heteroscedastic.py``. The homoscedastic path is
            numerically unchanged.
        """
        self.X = np.atleast_2d(np.asarray(X, dtype=float))
        self.y = np.asarray(y, dtype=float)
        n = len(self.y)
        K = self.kernel(self.X, self.X)
        jitter = self.JITTER * float(np.mean(np.diag(K)))
        if noise is None:
            noise_diag = np.full(n, self.noise_var)
        else:
            noise_diag = np.asarray(noise, dtype=float)
            if noise_diag.shape != (n,):
                raise ValueError(f"per-point noise must have shape ({n},)")
        self._noise_diag = noise_diag
        K = K + np.diag(noise_diag + jitter)
        self.L = np.linalg.cholesky(K)
        self.alpha = _chol_solve(self.L, self.y)
        self._fitted = True
        return self

    def predict(self, Xs: np.ndarray,
                include_noise: bool = False) -> "tuple[np.ndarray, np.ndarray]":
        """Posterior mean and pointwise variance at Xs.

        include_noise=False gives the credible band for the latent f;
        True adds sigma^2 for a predictive band on new observations y*.
        """
        assert self._fitted
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        Ks = self.kernel(self.X, Xs)                 # (n, n*)
        mean = Ks.T @ self.alpha
        v = np.linalg.solve(self.L, Ks)              # (n, n*)
        kss = np.diag(self.kernel(Xs, Xs)).copy()
        var = kss - np.sum(v**2, axis=0)
        var = np.maximum(var, 0.0)                   # clip float negatives
        if include_noise:
            var = var + self.noise_var
        return mean, var

    # -- leave-one-out cross-validation --------------------------------------

    def loo(self) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Closed-form leave-one-out CV predictive distribution.

        For each training point i, the predictive mean and variance of the GP
        refit on the other n-1 points are available *without* refitting, via
        the Cholesky factor already computed (Rasmussen & Williams 2006,
        Sec. 5.4.2, eqs. 5.10-5.12):

            mu_i     = y_i - alpha_i / [K_y^{-1}]_ii
            sigma^2_i = 1 / [K_y^{-1}]_ii

        where ``alpha = K_y^{-1} y`` (cached from ``fit``) and ``K_y`` is the
        noisy covariance ``k(X, X) + diag(noise)``. Both come from the single
        factorization done in ``fit``: naive LOO refits n GPs at O(n^4); this
        is O(n^3) once, then O(n^2) for the diagonal of ``K_y^{-1}``. Because
        ``K_y`` already carries the observation noise, ``sigma^2_i`` is an
        *observation*-level predictive variance -- it matches ``predict(...,
        include_noise=True)`` on the held-out point, not the latent band.

        The per-point log predictive density is the LOO log-CV score
        (R&W eq. 5.11), a leakage-free alternative to the marginal likelihood
        for model selection:

            log p(y_i | X, y_{-i}, theta)
                = -1/2 [ log(2 pi sigma^2_i) + (y_i - mu_i)^2 / sigma^2_i ].

        Returns
        -------
        mean, var, log_pred : each shape (n,) -- the LOO predictive mean,
        variance, and log predictive density per training point.

        Examples
        --------
        The identity is worth distrusting until you have seen it beat the
        brute force it replaces. Hold out point 0 by actually refitting on the
        other n-1 points, and compare against the closed form -- which never
        refits anything:

        >>> import numpy as np
        >>> from gp.kernels import RBF
        >>> X = np.linspace(0, 3, 8).reshape(-1, 1)
        >>> y = np.sin(X).ravel()
        >>> gp = GPRegressor(RBF(s2=1.0, l=1.0), noise_var=0.1).fit(X, y)
        >>> mean, var, log_pred = gp.loo()
        >>> refit = GPRegressor(RBF(s2=1.0, l=1.0), noise_var=0.1).fit(X[1:], y[1:])
        >>> m0, v0 = refit.predict(X[:1], include_noise=True)
        >>> bool(np.isclose(mean[0], m0[0]) and np.isclose(var[0], v0[0]))
        True
        >>> log_pred.shape
        (8,)
        """
        assert self._fitted
        n = len(self.y)
        inv_diag = np.diag(_chol_solve(self.L, np.eye(n)))
        var = 1.0 / inv_diag
        mean = self.y - self.alpha / inv_diag
        log_pred = -0.5 * (np.log(2.0 * np.pi * var) + (self.y - mean) ** 2 / var)
        return mean, var, log_pred

    def loo_log_predictive(self) -> float:
        """Total LOO cross-validation log predictive density (R&W eq. 5.11).

        A single scalar model-selection score: higher is better, and unlike the
        marginal likelihood it never conditions a point on itself.
        """
        return float(np.sum(self.loo()[2]))

    # -- evidence -------------------------------------------------------------

    def log_marginal_likelihood(self) -> float:
        assert self._fitted
        n = len(self.y)
        return float(
            -0.5 * self.y @ self.alpha
            - np.sum(np.log(np.diag(self.L)))
            - 0.5 * n * np.log(2.0 * np.pi)
        )

    def lml_and_grad(self, X: np.ndarray, y: np.ndarray,
                     params: Optional[np.ndarray] = None) -> "tuple[float, np.ndarray]":
        """Log marginal likelihood and its gradient w.r.t. ``self.params``.

        Gradient: 1/2 tr[(alpha alpha^T - K^{-1}) dK/dtheta_j] for each kernel
        parameter, and the same expression with dK = sigma^2 I for the noise:
        1/2 (sigma^2) * (alpha^T alpha - tr K^{-1}).
        """
        if params is not None:
            self.params = params
        self.fit(X, y)
        n = len(self.y)
        K_inv = _chol_solve(self.L, np.eye(n))
        A = np.outer(self.alpha, self.alpha) - K_inv
        grads = [0.5 * float(np.sum(A * dK)) for dK in self.kernel.grads(self.X)]
        grads.append(0.5 * self.noise_var * float(np.trace(A)))
        return self.log_marginal_likelihood(), np.array(grads)
