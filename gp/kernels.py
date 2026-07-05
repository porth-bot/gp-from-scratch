"""Covariance kernels with analytic gradients in log-parameter space.

Every kernel is parameterized by the *logs* of its positive hyperparameters
(theta = log psi), for two reasons:

1. Unconstrained optimization -- gradient steps can never leave the feasible
   set, so no projections or barriers are needed.
2. Scale-free steps -- d/d(log psi) = psi * d/d(psi) (chain rule), so a step
   of 0.1 in theta means "change psi by ~10%" regardless of psi's magnitude.
   Lengthscales of 0.01 and 100 get comparable effective learning rates.

Each kernel implements:

- ``__call__(X1, X2)``: the covariance matrix K, shape (n1, n2).
- ``grads(X)``: list of dK/dtheta_i on the symmetric train matrix (X, X),
  in the same order as ``theta``. These feed the marginal-likelihood
  gradient (gp.py); all of them are verified against central finite
  differences in tests/test_kernels.py.

Composition: ``Sum`` and ``Product`` combine kernels; their gradients follow
from linearity and the product rule. Observation noise is NOT a kernel here
-- it belongs to the likelihood and lives in ``GPRegressor`` (gp.py).
"""

import numpy as np


def sqdist(X1, X2):
    """Pairwise squared Euclidean distances, (n1, n2).

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, clipped at 0 to kill the tiny
    negatives float cancellation produces (they would NaN a sqrt later).
    """
    X1 = np.atleast_2d(X1)
    X2 = np.atleast_2d(X2)
    d2 = (
        np.sum(X1**2, axis=1)[:, None]
        + np.sum(X2**2, axis=1)[None, :]
        - 2.0 * X1 @ X2.T
    )
    return np.maximum(d2, 0.0)


class Kernel:
    """Base for leaf kernels: log-params ``_theta`` plus an optional ``fixed`` mask.

    Fixed parameters. Some hyperparameters are known from the physics and
    should not be learned -- the canonical case is the CO2 seasonal period,
    which is exactly one year. Freezing such a parameter is not just a
    convenience: leaving it free can wreck the optimization (the periodic
    log-period gradient is enormous near a phase mismatch, blowing up Adam),
    so pinning it to the known value both encodes prior knowledge and
    stabilizes ML-II.

    The mechanism is a boolean mask ``_fixed`` over the log-params. The public
    interface then reports *free* parameters only:

    - ``theta`` (get/set) exposes just the free entries, so an optimizer never
      sees or touches a fixed one.
    - ``grads(X)`` returns one gradient per free entry, in ``theta`` order.
    - ``n_params`` counts free entries (used by ``Sum``/``Product`` to split a
      concatenated theta correctly).

    ``__call__`` always uses the full ``_theta``, so a fixed parameter keeps
    its constructed value in every covariance evaluation. With nothing fixed
    (the default) the behavior is identical to a plain unconstrained kernel.
    """

    names: tuple = ()

    def _mask(self, fixed):
        """Build the boolean fixed-mask from ``fixed`` (called by subclasses).

        ``fixed`` may be None (nothing fixed), an iterable of parameter names
        drawn from ``self.names``, or a boolean mask the length of ``_theta``.
        """
        n = len(self._theta)
        if fixed is None:
            return np.zeros(n, dtype=bool)
        fixed = list(fixed)
        if all(isinstance(f, str) for f in fixed):
            idx = {name: i for i, name in enumerate(self.names)}
            mask = np.zeros(n, dtype=bool)
            for name in fixed:
                if name not in idx:
                    raise ValueError(
                        f"unknown parameter {name!r}; choose from {self.names}"
                    )
                mask[idx[name]] = True
            return mask
        mask = np.asarray(fixed, dtype=bool)
        if mask.shape != (n,):
            raise ValueError(f"fixed mask must have length {n}")
        return mask

    @property
    def free(self):
        """Boolean mask of the trainable (non-fixed) parameters."""
        return ~self._fixed

    @property
    def theta(self):
        return self._theta[self.free].copy()

    @theta.setter
    def theta(self, value):
        value = np.asarray(value, dtype=float)
        assert value.shape == self._theta[self.free].shape
        self._theta[self.free] = value

    @property
    def n_params(self):
        return int(self.free.sum())

    def grads(self, X):
        """dK/dtheta_i for the free parameters, in ``theta`` order."""
        full = self._grads_full(X)
        return [g for g, fixed in zip(full, self._fixed) if not fixed]

    def __add__(self, other):
        return Sum(self, other)

    def __mul__(self, other):
        return Product(self, other)


class RBF(Kernel):
    """Squared-exponential: k(r) = s2 * exp(-r^2 / (2 l^2)).

    Sample paths are infinitely differentiable -- the smoothest standard
    choice, and often *too* smooth for physical data (cf. Matern).

    theta = (log s2, log l):
        dK/d(log s2) = K
        dK/d(log l)  = K * r^2 / l^2      [d/dl = K r^2/l^3, times l]
    """

    names = ("s2", "l")

    def __init__(self, s2=1.0, l=1.0, fixed=None):
        self._theta = np.log([s2, l])
        self._fixed = self._mask(fixed)

    def __call__(self, X1, X2):
        s2, l = np.exp(self._theta)
        return s2 * np.exp(-0.5 * sqdist(X1, X2) / l**2)

    def _grads_full(self, X):
        _, l = np.exp(self._theta)
        d2 = sqdist(X, X)
        K = self(X, X)
        return [K, K * d2 / l**2]


class Matern(Kernel):
    """Matern kernel for nu in {0.5, 1.5, 2.5} (the closed-form cases).

    With a = sqrt(2 nu) r / l:
        nu=0.5: k = s2 e^{-a}                  (Ornstein-Uhlenbeck; continuous,
                                                nowhere differentiable paths)
        nu=1.5: k = s2 (1 + a) e^{-a}          (once-differentiable paths)
        nu=2.5: k = s2 (1 + a + a^2/3) e^{-a}  (twice-differentiable paths)

    Gradients w.r.t. log l use da/d(log l) = -a:
        nu=0.5: dK/d(log l) = s2 * a e^{-a}
        nu=1.5: dK/d(log l) = s2 * a^2 e^{-a}
        nu=2.5: dK/d(log l) = s2 * (a^2 (1 + a) / 3) e^{-a}
    """

    names = ("s2", "l")

    def __init__(self, nu=1.5, s2=1.0, l=1.0, fixed=None):
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError("closed forms exist for nu in {0.5, 1.5, 2.5}")
        self.nu = nu
        self._theta = np.log([s2, l])
        self._fixed = self._mask(fixed)

    def _a(self, X1, X2):
        _, l = np.exp(self._theta)
        return np.sqrt(2.0 * self.nu) * np.sqrt(sqdist(X1, X2)) / l

    def __call__(self, X1, X2):
        s2, _ = np.exp(self._theta)
        a = self._a(X1, X2)
        if self.nu == 0.5:
            poly = 1.0
        elif self.nu == 1.5:
            poly = 1.0 + a
        else:
            poly = 1.0 + a + a**2 / 3.0
        return s2 * poly * np.exp(-a)

    def _grads_full(self, X):
        s2, _ = np.exp(self._theta)
        a = self._a(X, X)
        e = np.exp(-a)
        if self.nu == 0.5:
            dlogl = s2 * a * e
        elif self.nu == 1.5:
            dlogl = s2 * a**2 * e
        else:
            dlogl = s2 * (a**2 * (1.0 + a) / 3.0) * e
        return [self(X, X), dlogl]


class Periodic(Kernel):
    """MacKay's periodic kernel: k(r) = s2 * exp(-2 sin^2(pi r / p) / l^2).

    Construction: map x to the circle u(x) = (cos 2pi x/p, sin 2pi x/p) and
    apply an RBF there; ||u(x) - u(x')||^2 = 4 sin^2(pi r / p) gives the
    form above. Exactly periodic by construction.

    theta = (log s2, log l, log p):
        dK/d(log s2) = K
        dK/d(log l)  = K * 4 sin^2(pi r / p) / l^2
        dK/d(log p)  = K * (2 pi r / (p l^2)) * sin(2 pi r / p)
    """

    names = ("s2", "l", "p")

    def __init__(self, s2=1.0, l=1.0, p=1.0, fixed=None):
        self._theta = np.log([s2, l, p])
        self._fixed = self._mask(fixed)

    def __call__(self, X1, X2):
        s2, l, p = np.exp(self._theta)
        r = np.sqrt(sqdist(X1, X2))
        return s2 * np.exp(-2.0 * np.sin(np.pi * r / p) ** 2 / l**2)

    def _grads_full(self, X):
        _, l, p = np.exp(self._theta)
        r = np.sqrt(sqdist(X, X))
        K = self(X, X)
        s = np.sin(np.pi * r / p)
        return [
            K,
            K * 4.0 * s**2 / l**2,
            K * (2.0 * np.pi * r / (p * l**2)) * np.sin(2.0 * np.pi * r / p),
        ]


class Sum(Kernel):
    """k = k1 + k2; gradients concatenate (linearity)."""

    def __init__(self, k1, k2):
        self.k1, self.k2 = k1, k2

    @property
    def theta(self):
        return np.concatenate([self.k1.theta, self.k2.theta])

    @theta.setter
    def theta(self, value):
        n1 = self.k1.n_params
        self.k1.theta = value[:n1]
        self.k2.theta = value[n1:]

    @property
    def n_params(self):
        return self.k1.n_params + self.k2.n_params

    def __call__(self, X1, X2):
        return self.k1(X1, X2) + self.k2(X1, X2)

    def grads(self, X):
        return self.k1.grads(X) + self.k2.grads(X)


class Product(Kernel):
    """k = k1 * k2 (elementwise); product rule: dK = dK1 * K2 + K1 * dK2."""

    def __init__(self, k1, k2):
        self.k1, self.k2 = k1, k2

    @property
    def theta(self):
        return np.concatenate([self.k1.theta, self.k2.theta])

    @theta.setter
    def theta(self, value):
        n1 = self.k1.n_params
        self.k1.theta = value[:n1]
        self.k2.theta = value[n1:]

    @property
    def n_params(self):
        return self.k1.n_params + self.k2.n_params

    def __call__(self, X1, X2):
        return self.k1(X1, X2) * self.k2(X1, X2)

    def grads(self, X):
        K1, K2 = self.k1(X, X), self.k2(X, X)
        return [g * K2 for g in self.k1.grads(X)] + [K1 * g for g in self.k2.grads(X)]
