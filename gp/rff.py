"""Random Fourier features: a randomized finite-rank approximation to the RBF GP.

An exact GP costs O(n^3) to fit and O(n^2) to store, because the whole model
lives in the n x n Gram matrix. Random Fourier features (Rahimi & Recht 2007)
trade that for O(n D^2 + D^3) with D features -- linear in n -- by writing the
kernel as an inner product in an explicit, finite, randomly-chosen feature
space and then doing plain Bayesian linear regression there.

Why a Fourier basis is the right one: Bochner's theorem (derived in
theory/derivations.md, Sec. 8). A continuous stationary kernel k(x - x') is
positive definite if and only if it is the Fourier transform of a finite
non-negative measure. Normalizing that measure to a probability density p(w)
(the kernel's *spectral density*, k(0) = s2 factored out) gives

    k(x - x') = s2 E_{w ~ p} [ cos( w . (x - x') ) ],

a plain expectation -- so it can be estimated by Monte Carlo. For the RBF
kernel k(r) = s2 exp(-r^2 / (2 l^2)) the spectral density is itself Gaussian,
p(w) = N(0, l^{-2} I): a *short* lengthscale means a *wide* spread of
frequencies, which is the frequency-domain statement of "wiggly".

The estimator used here draws m = D/2 frequencies w_j ~ p and pairs a cosine
with a sine at each:

    z(x) = sqrt(s2 / m) [ cos(w_1.x), ..., cos(w_m.x),
                          sin(w_1.x), ..., sin(w_m.x) ]  in R^D,

so that, by the angle-subtraction identity cos(a)cos(b) + sin(a)sin(b) =
cos(a - b),

    z(x) . z(x') = (s2 / m) sum_j cos( w_j . (x - x') )   -->   k(x, x')

as an unbiased average of m i.i.d. terms: the error decays as D^{-1/2}, with
no dependence on the input dimension.

The cos/sin pairing rather than Rahimi & Recht's original single feature
sqrt(2 s2 / D) cos(w.x + b), b ~ U(0, 2pi): both are unbiased, but the paired
version has strictly lower variance (Sutherland & Schneider 2015) and -- the
property this module leans on -- it reproduces the prior variance *exactly*,

    z(x) . z(x) = (s2 / m) sum_j [cos^2 + sin^2] = s2   for every x,

since the Pythagorean identity holds per frequency, not just in expectation.
The offset version's k(x, x) fluctuates around s2, which shows up directly as
noise in predictive error bars. ``offset=True`` is available for comparison.

Weight space vs function space. Given a fixed feature map, the model

    f(x) = z(x) . w,   w ~ N(0, I_D),   y = f(x) + eps,  eps ~ N(0, sigma^2)

is *exactly* a GP with the finite-rank kernel k_D(x, x') = z(x) . z(x') -- not
an approximation of it. The approximation is entirely in k_D ~ k; everything
after that is exact Bayesian linear regression, done in D dimensions instead
of n. ``RFFRegressor`` and ``GPRegressor(RFFMap(...))`` therefore agree to
machine precision (tests/test_rff.py), which cleanly separates the two error
sources: Monte Carlo error in the kernel, and nothing else.

Honest limitation -- variance starvation. A rank-D model has only D degrees of
freedom to spend. Once n >> D the data pins down essentially all of them, and
the posterior variance collapses *everywhere*, including in gaps where an
exact GP correctly widens (Wang et al. 2018). RFF is a mean-prediction
accelerator; its error bars are trustworthy only while D is comfortably large
relative to the number of independent things the data says. Section 10 of the
README measures exactly how badly this bites.

References
----------
Rahimi & Recht (2007), Random features for large-scale kernel machines.
Sutherland & Schneider (2015), On the error of random Fourier features.
Wang et al. (2018), Batched large-scale Bayesian optimization in
high-dimensional spaces (variance starvation).
"""

from __future__ import annotations

import numpy as np

from gp.linalg import cho_solve, solve_lower


class RFFMap:
    """Random Fourier feature map for the (isotropic or ARD) RBF kernel.

    Parameters
    ----------
    n_features : int
        The feature dimension D. Must be even: features come in cos/sin pairs
        sharing a frequency, so D = 2m draws m frequencies.
    lengthscale : float or array of shape (d,)
        RBF lengthscale l. A vector gives per-dimension lengthscales (ARD);
        the spectral density is then N(0, diag(l_i^{-2})).
    s2 : float
        Signal variance -- the prior variance k(x, x), reproduced exactly by
        the cos/sin map.
    n_dims : int
        Input dimension d.
    rng : np.random.Generator
        The frequencies are drawn once, here, and then held fixed: they are
        part of the model, not resampled per call.
    offset : bool
        False (default) uses the cos/sin pairing. True uses Rahimi & Recht's
        original sqrt(2 s2 / D) cos(w.x + b) with b ~ U(0, 2pi) -- same
        expectation, higher variance, and k(x, x) only approximately s2.

    The instance is callable as ``kernel(X1, X2)``, so it drops straight into
    ``GPRegressor`` as a fixed kernel. It deliberately exposes no ``grads``:
    the hyperparameters are baked into the sampled frequencies, so ML-II would
    have to redraw them: fit the *exact* GP's hyperparameters first (on a
    subset if n is large), then build the map at those values.

    Examples
    --------
    The map reproduces the prior variance exactly, and the RBF kernel
    approximately:

    >>> import numpy as np
    >>> from gp.kernels import RBF
    >>> rng = np.random.default_rng(0)
    >>> phi = RFFMap(n_features=4096, lengthscale=1.0, s2=2.0, n_dims=1, rng=rng)
    >>> X = np.linspace(0, 3, 40).reshape(-1, 1)
    >>> Z = phi.transform(X)
    >>> Z.shape
    (40, 4096)
    >>> bool(np.allclose(np.sum(Z**2, axis=1), 2.0))        # k(x, x) = s2 exactly
    True
    >>> bool(np.abs(phi(X, X) - RBF(s2=2.0, l=1.0)(X, X)).max() < 0.05)
    True
    """

    def __init__(
        self,
        n_features: int,
        lengthscale: "float | np.ndarray",
        s2: float,
        n_dims: int,
        rng: np.random.Generator,
        offset: bool = False,
    ):
        if n_features < 1:
            raise ValueError("n_features must be positive")
        if not offset and n_features % 2 != 0:
            raise ValueError("n_features must be even for the cos/sin map")
        l = np.asarray(lengthscale, dtype=float)
        if l.ndim == 0:
            l = np.full(n_dims, float(l))
        if l.shape != (n_dims,):
            raise ValueError(f"lengthscale must be scalar or shape ({n_dims},)")
        if np.any(l <= 0):
            raise ValueError("lengthscale must be positive")

        self.n_features = int(n_features)
        self.lengthscale = l
        self.s2 = float(s2)
        self.n_dims = int(n_dims)
        self.offset = bool(offset)

        # w ~ N(0, diag(l_i^{-2})): the RBF's spectral density (Sec. 8).
        n_freq = self.n_features if offset else self.n_features // 2
        self.W = rng.standard_normal((n_freq, n_dims)) / l          # (m, d)
        self.b = rng.uniform(0.0, 2.0 * np.pi, n_freq) if offset else None

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Feature matrix Z, shape (n, D), with Z Z^T the approximate kernel."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != self.n_dims:
            raise ValueError(f"expected {self.n_dims} input dims, got {X.shape[1]}")
        proj = X @ self.W.T                                          # (n, m)
        if self.offset:
            assert self.b is not None
            return np.sqrt(2.0 * self.s2 / self.n_features) * np.cos(proj + self.b)
        m = self.W.shape[0]
        return np.sqrt(self.s2 / m) * np.concatenate(
            [np.cos(proj), np.sin(proj)], axis=1
        )

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """The finite-rank kernel k_D(X1, X2) = Z1 Z2^T (drop-in for GPRegressor)."""
        return self.transform(X1) @ self.transform(X2).T


class RFFRegressor:
    """Bayesian linear regression in an RFF feature space = the approximate GP.

    Model: f(x) = z(x) . w with w ~ N(0, I_D) and y = f(x) + eps,
    eps ~ N(0, sigma^2). The prior scale sits in the feature map (which carries
    sqrt(s2 / m)), so a unit-variance weight prior is the right one.

    With Z the (n, D) feature matrix and A = Z^T Z + sigma^2 I_D:

        posterior over weights:  w | y ~ N( A^{-1} Z^T y,  sigma^2 A^{-1} )
        latent mean at x*:       z(x*) . A^{-1} Z^T y
        latent variance at x*:   sigma^2 z(x*)^T A^{-1} z(x*)

    This is the *weight-space* view of the same posterior the exact GP computes
    in function space (Rasmussen & Williams Sec. 2.1); the cost is
    O(n D^2 + D^3) instead of O(n^3), which is the entire point when n >> D.
    A is formed once and Cholesky-factorized; the inverse is never built.

    Examples
    --------
    Enough features and the posterior mean tracks the exact GP's:

    >>> import numpy as np
    >>> from gp.gp import GPRegressor
    >>> from gp.kernels import RBF
    >>> rng = np.random.default_rng(0)
    >>> X = np.sort(rng.uniform(-3, 3, 60)).reshape(-1, 1)
    >>> y = np.sin(X).ravel() + 0.1 * rng.standard_normal(60)
    >>> Xs = np.linspace(-3, 3, 50).reshape(-1, 1)
    >>> exact = GPRegressor(RBF(s2=1.0, l=1.0), noise_var=0.01).fit(X, y)
    >>> phi = RFFMap(4096, lengthscale=1.0, s2=1.0, n_dims=1, rng=rng)
    >>> approx = RFFRegressor(phi, noise_var=0.01).fit(X, y)
    >>> mean_exact, _ = exact.predict(Xs)
    >>> mean_rff, _ = approx.predict(Xs)
    >>> bool(np.abs(mean_exact - mean_rff).max() < 0.02)
    True
    """

    def __init__(self, feature_map: RFFMap, noise_var: float = 0.1):
        self.feature_map = feature_map
        self.noise_var = float(noise_var)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RFFRegressor":
        Z = self.feature_map.transform(X)
        self.y = np.asarray(y, dtype=float)
        D = Z.shape[1]
        A = Z.T @ Z + self.noise_var * np.eye(D)
        self.L = np.linalg.cholesky(A)
        self.w_mean = cho_solve(self.L, Z.T @ self.y)
        self._fitted = True
        return self

    def predict(self, Xs: np.ndarray,
                include_noise: bool = False) -> "tuple[np.ndarray, np.ndarray]":
        """Posterior mean and pointwise variance at Xs (latent f by default)."""
        assert self._fitted
        Zs = self.feature_map.transform(Xs)                          # (n*, D)
        mean = Zs @ self.w_mean
        v = solve_lower(self.L, Zs.T)                                # (D, n*)
        var = self.noise_var * np.sum(v**2, axis=0)
        var = np.maximum(var, 0.0)
        if include_noise:
            var = var + self.noise_var
        return mean, var
