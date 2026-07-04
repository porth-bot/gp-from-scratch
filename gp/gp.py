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

import numpy as np


def _chol_solve(L, B):
    """Solve (L L^T) x = B via two triangular solves (never form the inverse)."""
    return np.linalg.solve(L.T, np.linalg.solve(L, B))


class GPRegressor:
    """Zero-mean exact GP regression.

    Parameters
    ----------
    kernel : gp.kernels.Kernel
    noise_var : float
        Initial observation-noise variance sigma^2 (optimized in log space
        alongside the kernel's theta when using gp.optimize).
    """

    JITTER = 1e-10

    def __init__(self, kernel, noise_var=0.1):
        self.kernel = kernel
        self.log_noise = float(np.log(noise_var))
        self._fitted = False

    # -- parameter vector: kernel theta plus log noise -----------------------

    @property
    def params(self):
        return np.concatenate([self.kernel.theta, [self.log_noise]])

    @params.setter
    def params(self, value):
        self.kernel.theta = value[:-1]
        self.log_noise = float(value[-1])
        self._fitted = False

    @property
    def noise_var(self):
        return float(np.exp(self.log_noise))

    # -- fitting and prediction ----------------------------------------------

    def fit(self, X, y):
        self.X = np.atleast_2d(np.asarray(X, dtype=float))
        self.y = np.asarray(y, dtype=float)
        n = len(self.y)
        K = self.kernel(self.X, self.X)
        K = K + (self.noise_var + self.JITTER * float(np.mean(np.diag(K)))) * np.eye(n)
        self.L = np.linalg.cholesky(K)
        self.alpha = _chol_solve(self.L, self.y)
        self._fitted = True
        return self

    def predict(self, Xs, include_noise=False):
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

    # -- evidence -------------------------------------------------------------

    def log_marginal_likelihood(self):
        assert self._fitted
        n = len(self.y)
        return float(
            -0.5 * self.y @ self.alpha
            - np.sum(np.log(np.diag(self.L)))
            - 0.5 * n * np.log(2.0 * np.pi)
        )

    def lml_and_grad(self, X, y, params=None):
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
