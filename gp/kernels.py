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

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

# What ``fixed=`` accepts: nothing, an iterable of parameter names, or a
# boolean mask the length of ``_theta`` (see ``Kernel._mask``).
FixedArg = Optional[Iterable]


def sqdist(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
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

    names: tuple[str, ...] = ()

    # Declared here so the base methods (``free``, ``theta``, ``grads``) type-
    # check; every leaf kernel assigns them in ``__init__``.
    _theta: np.ndarray
    _fixed: np.ndarray

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """The covariance matrix K = k(X1, X2). Overridden by every leaf."""
        raise NotImplementedError

    def _grads_full(self, X: np.ndarray) -> list[np.ndarray]:
        """dK/dtheta_i for *all* params (before the fixed mask). Overridden."""
        raise NotImplementedError

    def _mask(self, fixed: FixedArg) -> np.ndarray:
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
    def free(self) -> np.ndarray:
        """Boolean mask of the trainable (non-fixed) parameters."""
        return ~self._fixed

    @property
    def theta(self) -> np.ndarray:
        return self._theta[self.free].copy()

    @theta.setter
    def theta(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float)
        assert value.shape == self._theta[self.free].shape
        self._theta[self.free] = value

    @property
    def n_params(self) -> int:
        return int(self.free.sum())

    def grads(self, X: np.ndarray) -> list[np.ndarray]:
        """dK/dtheta_i for the free parameters, in ``theta`` order."""
        full = self._grads_full(X)
        return [g for g, fixed in zip(full, self._fixed) if not fixed]

    def __add__(self, other: "Kernel") -> "Sum":
        return Sum(self, other)

    def __mul__(self, other: "Kernel") -> "Product":
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

    def __init__(self, s2: float = 1.0, l: float = 1.0, fixed: FixedArg = None):
        self._theta = np.log([s2, l])
        self._fixed = self._mask(fixed)

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        s2, l = np.exp(self._theta)
        return s2 * np.exp(-0.5 * sqdist(X1, X2) / l**2)

    def _grads_full(self, X: np.ndarray) -> list[np.ndarray]:
        _, l = np.exp(self._theta)
        d2 = sqdist(X, X)
        K = self(X, X)
        return [K, K * d2 / l**2]


class ARD(Kernel):
    """Squared-exponential with one lengthscale per input dimension
    (Automatic Relevance Determination).

        k(x, x') = s2 * exp( -1/2 sum_d (x_d - x'_d)^2 / l_d^2 )

    The isotropic RBF uses a single l for every dimension; ARD gives each input
    its own l_d. That turns the kernel into a relevance detector: a large l_d
    flattens the covariance's dependence on dimension d (the GP becomes nearly
    constant along it), so ML-II *automatically* suppresses uninformative inputs
    by driving their lengthscales up. The inverse lengthscales 1/l_d then rank
    input relevance -- ARD doubles as feature selection (MacKay 1994; Neal 1996;
    Rasmussen & Williams 2006, Sec. 5.1). With all l_d equal it is exactly the
    isotropic RBF.

    theta = (log s2, log l_0, ..., log l_{D-1}):
        dK/d(log s2)  = K
        dK/d(log l_d) = K * (x_d - x'_d)^2 / l_d^2

    Each lengthscale gradient sees only its own dimension's squared distances,
    which is what lets ML-II move them independently.
    """

    def __init__(
        self,
        s2: float = 1.0,
        lengthscales: object = 1.0,
        dim: Optional[int] = None,
        fixed: FixedArg = None,
    ):
        ls = np.atleast_1d(np.asarray(lengthscales, dtype=float))
        if dim is not None:
            if ls.size == 1:
                ls = np.full(dim, ls.item())
            elif ls.size != dim:
                raise ValueError(
                    f"lengthscales has size {ls.size}, expected dim={dim}"
                )
        self.dim_in = ls.size
        # per-instance names so the fixed-mask machinery can address each l_d
        self.names = ("s2",) + tuple(f"l{d}" for d in range(self.dim_in))
        self._theta = np.log(np.concatenate([[s2], ls]))
        self._fixed = self._mask(fixed)

    def _scaled(self, X: np.ndarray) -> np.ndarray:
        """Divide each input column by its lengthscale (so plain sqdist gives the
        ARD-weighted squared distance sum_d (x_d - x'_d)^2 / l_d^2)."""
        l = np.exp(self._theta[1:])
        return np.atleast_2d(X) / l

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        s2 = np.exp(self._theta[0])
        return s2 * np.exp(-0.5 * sqdist(self._scaled(X1), self._scaled(X2)))

    def _grads_full(self, X: np.ndarray) -> list[np.ndarray]:
        X = np.atleast_2d(X)
        l = np.exp(self._theta[1:])
        K = self(X, X)
        grads = [K]
        for d in range(self.dim_in):
            xd = X[:, d]
            per_dim_d2 = (xd[:, None] - xd[None, :]) ** 2
            grads.append(K * per_dim_d2 / l[d] ** 2)
        return grads


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

    def __init__(self, nu: float = 1.5, s2: float = 1.0, l: float = 1.0,
                 fixed: FixedArg = None):
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError("closed forms exist for nu in {0.5, 1.5, 2.5}")
        self.nu = nu
        self._theta = np.log([s2, l])
        self._fixed = self._mask(fixed)

    def _a(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        _, l = np.exp(self._theta)
        return np.sqrt(2.0 * self.nu) * np.sqrt(sqdist(X1, X2)) / l

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        s2, _ = np.exp(self._theta)
        a = self._a(X1, X2)
        if self.nu == 0.5:
            poly = np.ones_like(a)
        elif self.nu == 1.5:
            poly = 1.0 + a
        else:
            poly = 1.0 + a + a**2 / 3.0
        return s2 * poly * np.exp(-a)

    def _grads_full(self, X: np.ndarray) -> list[np.ndarray]:
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

    def __init__(self, s2: float = 1.0, l: float = 1.0, p: float = 1.0,
                 fixed: FixedArg = None):
        self._theta = np.log([s2, l, p])
        self._fixed = self._mask(fixed)

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        s2, l, p = np.exp(self._theta)
        r = np.sqrt(sqdist(X1, X2))
        return s2 * np.exp(-2.0 * np.sin(np.pi * r / p) ** 2 / l**2)

    def _grads_full(self, X: np.ndarray) -> list[np.ndarray]:
        _, l, p = np.exp(self._theta)
        r = np.sqrt(sqdist(X, X))
        K = self(X, X)
        s = np.sin(np.pi * r / p)
        return [
            K,
            K * 4.0 * s**2 / l**2,
            K * (2.0 * np.pi * r / (p * l**2)) * np.sin(2.0 * np.pi * r / p),
        ]


class RationalQuadratic(Kernel):
    """Rational quadratic: k(r) = s2 * (1 + r^2 / (2 alpha l^2))^(-alpha).

    A scale mixture of RBFs: draw an inverse squared-lengthscale from a Gamma
    distribution (shape alpha) and average the resulting RBFs. It therefore
    models data whose correlations decay over *several* lengthscales at once,
    with a heavier-than-Gaussian tail. As alpha -> infinity the mixture
    concentrates on a single scale and the kernel recovers the RBF exactly
    (verified in tests) -- alpha is the knob between "one lengthscale" (large)
    and "many" (small). This is the medium-term flexibility the CO2 model's RBF
    trend lacked; see the README limitation it addresses.

    theta = (log s2, log l, log alpha). Writing B = 1 + r^2 / (2 alpha l^2) so
    that K = s2 B^{-alpha}:
        dK/d(log s2)    = K
        dK/d(log l)     = K * (r^2 / l^2) / B
        dK/d(log alpha) = K * ( -alpha ln B + r^2 / (2 l^2 B) )
    Both r-dependent gradients vanish on the diagonal (r = 0, B = 1) and, in
    the alpha -> infinity limit, reduce to the RBF's (the log-alpha gradient
    -> 0, log-l gradient -> K r^2/l^2).
    """

    names = ("s2", "l", "alpha")

    def __init__(self, s2: float = 1.0, l: float = 1.0, alpha: float = 1.0,
                 fixed: FixedArg = None):
        self._theta = np.log([s2, l, alpha])
        self._fixed = self._mask(fixed)

    def _B(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        _, l, alpha = np.exp(self._theta)
        return 1.0 + sqdist(X1, X2) / (2.0 * alpha * l**2)

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        s2, _, alpha = np.exp(self._theta)
        return s2 * self._B(X1, X2) ** (-alpha)

    def _grads_full(self, X: np.ndarray) -> list[np.ndarray]:
        _, l, alpha = np.exp(self._theta)
        d2 = sqdist(X, X)
        B = 1.0 + d2 / (2.0 * alpha * l**2)
        K = self(X, X)
        dlogl = K * (d2 / l**2) / B
        dlogalpha = K * (-alpha * np.log(B) + d2 / (2.0 * l**2 * B))
        return [K, dlogl, dlogalpha]


class Sum(Kernel):
    """k = k1 + k2; gradients concatenate (linearity)."""

    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1, self.k2 = k1, k2

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([self.k1.theta, self.k2.theta])

    @theta.setter
    def theta(self, value: np.ndarray) -> None:
        n1 = self.k1.n_params
        self.k1.theta = value[:n1]
        self.k2.theta = value[n1:]

    @property
    def n_params(self) -> int:
        return self.k1.n_params + self.k2.n_params

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.k1(X1, X2) + self.k2(X1, X2)

    def grads(self, X: np.ndarray) -> list[np.ndarray]:
        return self.k1.grads(X) + self.k2.grads(X)


class Product(Kernel):
    """k = k1 * k2 (elementwise); product rule: dK = dK1 * K2 + K1 * dK2."""

    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1, self.k2 = k1, k2

    @property
    def theta(self) -> np.ndarray:
        return np.concatenate([self.k1.theta, self.k2.theta])

    @theta.setter
    def theta(self, value: np.ndarray) -> None:
        n1 = self.k1.n_params
        self.k1.theta = value[:n1]
        self.k2.theta = value[n1:]

    @property
    def n_params(self) -> int:
        return self.k1.n_params + self.k2.n_params

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.k1(X1, X2) * self.k2(X1, X2)

    def grads(self, X: np.ndarray) -> list[np.ndarray]:
        K1, K2 = self.k1(X, X), self.k2(X, X)
        return [g * K2 for g in self.k1.grads(X)] + [K1 * g for g in self.k2.grads(X)]
