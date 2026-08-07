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
- ``grads(X1, X2=None)``: list of dK/dtheta_i on the covariance between two
  input sets, in the same order as ``theta``. With ``X2`` omitted this is the
  symmetric train matrix (X, X) that the marginal-likelihood gradient needs
  (gp.py); the *cross* case is what the sparse bound needs, where the
  hyperparameters have to be differentiated through ``k(Z, X)`` with Z and X
  different sets (gp/sparse.py). All of them are verified against central
  finite differences in tests/test_kernels.py.
- ``dK_dX1(X1, X2)``: derivative of each entry with respect to the *left*
  input's coordinates, shape (n1, n2, d). This is a derivative in input
  space, not parameter space, and it exists so that inducing locations can be
  optimized. It is defined on the leaves that are smooth functions of the
  squared distance; the ones that are not (Matern nu=0.5 at r=0) or that
  simply have not been derived (Gibbs) raise rather than return something
  plausible-looking.

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

    Examples
    --------
    A Periodic kernel with its period pinned to 1 (the CO2 case: a year is a
    year). The optimizer sees two free parameters instead of three, and
    ``grads`` returns one entry per free parameter, in ``theta`` order:

    >>> import numpy as np
    >>> k = Periodic(s2=1.0, l=1.0, p=1.0, fixed=["p"])
    >>> k.names
    ('s2', 'l', 'p')
    >>> k.n_params                      # p is frozen, so only s2 and l
    2
    >>> X = np.linspace(0, 2, 4).reshape(-1, 1)
    >>> len(k.grads(X))
    2

    Setting ``theta`` touches only the free entries -- the frozen period keeps
    its constructed value, and the covariance still sees it:

    >>> k.theta = np.log([4.0, 2.0])    # new s2, l (log space)
    >>> np.exp(k._theta).round(3)       # s2, l updated; p untouched
    array([4., 2., 1.])
    """

    names: tuple[str, ...] = ()

    # Declared here so the base methods (``free``, ``theta``, ``grads``) type-
    # check; every leaf kernel assigns them in ``__init__``.
    _theta: np.ndarray
    _fixed: np.ndarray

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """The covariance matrix K = k(X1, X2). Overridden by every leaf."""
        raise NotImplementedError

    def _grads_full(self, X1: np.ndarray, X2: np.ndarray) -> list[np.ndarray]:
        """dK/dtheta_i for *all* params (before the fixed mask). Overridden."""
        raise NotImplementedError

    def _dk_dsqdist(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """dk/d(r^2) elementwise, for kernels that are functions of r^2 alone.

        Supplying this is how a stationary leaf gets ``dK_dX1`` for free: the
        chain rule through ``r^2 = ||x - x'||^2`` does the rest. Leaves that
        are not functions of r^2 (ARD, whose distance is lengthscale-weighted
        per dimension; Gibbs, which is not stationary at all) override
        ``dK_dX1`` directly or decline it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define an input-space derivative"
        )

    def dK_dX1(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """d k(x1_i, x2_j) / d x1_i, shape (n1, n2, d).

        Only the *left* argument is differentiated. For a stationary kernel the
        right-hand derivative is the negative of this one, but nothing here
        relies on that -- the sparse bound's Z-gradient (gp/sparse.py) needs
        only the left slot, because it differentiates k(Z, X) and k(Z, Z) with
        respect to Z and reaches the second Z of k(Z, Z) by symmetry of the
        contracting matrix rather than by a second derivative.

        With k = f(r^2) and r^2 = ||x - x'||^2, d r^2 / d x_a = 2 (x_a - x'_a),
        so the whole thing is ``2 * f'(r^2) * (x - x')``.
        """
        X1 = np.atleast_2d(np.asarray(X1, dtype=float))
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        g = self._dk_dsqdist(X1, X2)                        # (n1, n2)
        diff = X1[:, None, :] - X2[None, :, :]              # (n1, n2, d)
        return 2.0 * g[:, :, None] * diff

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

    def grads(self, X1: np.ndarray,
              X2: Optional[np.ndarray] = None) -> list[np.ndarray]:
        """dK/dtheta_i for the free parameters, in ``theta`` order.

        ``X2`` defaults to ``X1`` (the symmetric train matrix). Passing a
        different second set gives the cross-covariance gradients dK(X1,X2)/dtheta.
        """
        full = self._grads_full(X1, X1 if X2 is None else X2)
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

    Input space: k = s2 exp(-r^2 / (2 l^2)), so dk/d(r^2) = -k / (2 l^2).
    """

    names = ("s2", "l")

    def __init__(self, s2: float = 1.0, l: float = 1.0, fixed: FixedArg = None):
        self._theta = np.log([s2, l])
        self._fixed = self._mask(fixed)

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        s2, l = np.exp(self._theta)
        return s2 * np.exp(-0.5 * sqdist(X1, X2) / l**2)

    def _grads_full(self, X1: np.ndarray, X2: np.ndarray) -> list[np.ndarray]:
        _, l = np.exp(self._theta)
        d2 = sqdist(X1, X2)
        K = self(X1, X2)
        return [K, K * d2 / l**2]

    def _dk_dsqdist(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        _, l = np.exp(self._theta)
        return -self(X1, X2) / (2.0 * l**2)


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

    Input space: the distance is lengthscale-weighted per dimension, so this is
    not a function of the plain r^2 and ``_dk_dsqdist`` does not apply.
    Differentiating the exponent directly,
    ``dk/dx_d = -k (x_d - x'_d) / l_d^2`` -- the isotropic result with each
    dimension divided by its own l_d, which also says a suppressed dimension
    (large l_d) exerts almost no pull on an inducing point's location.
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

    def _grads_full(self, X1: np.ndarray, X2: np.ndarray) -> list[np.ndarray]:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        l = np.exp(self._theta[1:])
        K = self(X1, X2)
        grads = [K]
        for d in range(self.dim_in):
            per_dim_d2 = (X1[:, d][:, None] - X2[:, d][None, :]) ** 2
            grads.append(K * per_dim_d2 / l[d] ** 2)
        return grads

    def dK_dX1(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(np.asarray(X1, dtype=float))
        X2 = np.atleast_2d(np.asarray(X2, dtype=float))
        l = np.exp(self._theta[1:])
        K = self(X1, X2)
        diff = X1[:, None, :] - X2[None, :, :]
        return -K[:, :, None] * diff / (l**2)[None, None, :]


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

    Input space. With da/d(r^2) = a / (2 r^2) and a^2 = 2 nu r^2 / l^2, the
    apparent 1/r^2 singularity cancels against the factor of a that every
    dk/da carries -- except at nu = 0.5, where it does not:

        nu=0.5: dk/da = -s2 e^{-a}            =>  dk/d(r^2) = -s2 e^{-a} a/(2 r^2)
        nu=1.5: dk/da = -s2 a e^{-a}          =>  dk/d(r^2) = -s2 e^{-a} * 3/(2 l^2)
        nu=2.5: dk/da = -s2 a (1+a) e^{-a}/3  =>  dk/d(r^2) = -s2 e^{-a}(1+a) * 5/(6 l^2)

    The nu=0.5 case really is non-differentiable at coincident points -- the
    OU process has nowhere-differentiable paths and its kernel has a cusp at
    r = 0 -- so ``dK_dX1`` raises there rather than returning a large number.
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

    def _grads_full(self, X1: np.ndarray, X2: np.ndarray) -> list[np.ndarray]:
        s2, _ = np.exp(self._theta)
        a = self._a(X1, X2)
        e = np.exp(-a)
        if self.nu == 0.5:
            dlogl = s2 * a * e
        elif self.nu == 1.5:
            dlogl = s2 * a**2 * e
        else:
            dlogl = s2 * (a**2 * (1.0 + a) / 3.0) * e
        return [self(X1, X2), dlogl]

    def _dk_dsqdist(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        if self.nu == 0.5:
            raise NotImplementedError(
                "Matern nu=0.5 has a cusp at r=0: k = s2 exp(-r/l) is not "
                "differentiable in x at coincident points, so there is no "
                "input-space gradient to hand an optimizer. Use nu>=1.5."
            )
        s2, l = np.exp(self._theta)
        a = self._a(X1, X2)
        e = np.exp(-a)
        if self.nu == 1.5:
            return -s2 * e * 1.5 / l**2
        return -s2 * e * (1.0 + a) * 5.0 / (6.0 * l**2)


class Periodic(Kernel):
    """MacKay's periodic kernel: k(r) = s2 * exp(-2 sin^2(pi r / p) / l^2).

    Construction: map x to the circle u(x) = (cos 2pi x/p, sin 2pi x/p) and
    apply an RBF there; ||u(x) - u(x')||^2 = 4 sin^2(pi r / p) gives the
    form above. Exactly periodic by construction.

    theta = (log s2, log l, log p):
        dK/d(log s2) = K
        dK/d(log l)  = K * 4 sin^2(pi r / p) / l^2
        dK/d(log p)  = K * (2 pi r / (p l^2)) * sin(2 pi r / p)

    Input space: dk/dr = -(2 pi / (p l^2)) k sin(2 pi r / p), and dividing by
    2r to get dk/d(r^2) leaves a 0/0 at coincident points whose limit is
    finite. Writing sin(2 pi r/p)/r as (2 pi/p) sinc(2r/p) with numpy's
    normalized sinc removes the branch entirely and is exact at r = 0:

        dk/d(r^2) = -(2 pi^2 / (p^2 l^2)) k sinc(2 r / p).
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

    def _grads_full(self, X1: np.ndarray, X2: np.ndarray) -> list[np.ndarray]:
        _, l, p = np.exp(self._theta)
        r = np.sqrt(sqdist(X1, X2))
        K = self(X1, X2)
        s = np.sin(np.pi * r / p)
        return [
            K,
            K * 4.0 * s**2 / l**2,
            K * (2.0 * np.pi * r / (p * l**2)) * np.sin(2.0 * np.pi * r / p),
        ]

    def _dk_dsqdist(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        _, l, p = np.exp(self._theta)
        r = np.sqrt(sqdist(X1, X2))
        return -(2.0 * np.pi**2 / (p**2 * l**2)) * self(X1, X2) * np.sinc(2.0 * r / p)


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

    Input space: dk/d(r^2) = -k / (2 l^2 B), the RBF's -k/(2 l^2) divided by B.
    Since B >= 1 and grows with r, the pull an RQ kernel exerts on a distant
    point decays polynomially rather than exponentially -- the same heavy tail,
    seen from the optimizer's side.
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

    def _grads_full(self, X1: np.ndarray, X2: np.ndarray) -> list[np.ndarray]:
        _, l, alpha = np.exp(self._theta)
        d2 = sqdist(X1, X2)
        B = 1.0 + d2 / (2.0 * alpha * l**2)
        K = self(X1, X2)
        dlogl = K * (d2 / l**2) / B
        dlogalpha = K * (-alpha * np.log(B) + d2 / (2.0 * l**2 * B))
        return [K, dlogl, dlogalpha]

    def _dk_dsqdist(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        _, l, _ = np.exp(self._theta)
        return -self(X1, X2) / (2.0 * l**2 * self._B(X1, X2))


class Gibbs(Kernel):
    """Nonstationary kernel with an input-dependent lengthscale (Gibbs 1997).

    Every other kernel here is *stationary*: k(x, x') depends only on x - x',
    so one lengthscale governs the whole input space. That is a real modeling
    assumption, and it is often wrong -- a function can be flat on the left and
    oscillate on the right, and a stationary GP must then compromise, choosing
    one lengthscale that is too short for the smooth region (bands too wide,
    posterior mean too wiggly) and too long for the rough one (structure
    smoothed away).

    Gibbs (1997) gives the fix in 1D. Let ``l(x) > 0`` vary with the input; then

        k(x, x') = s2 * sqrt( 2 l(x) l(x') / (l(x)^2 + l(x')^2) )
                      * exp( -(x - x')^2 / (l(x)^2 + l(x')^2) )

    is positive semi-definite for *any* positive ``l(.)``. (It is not a hack:
    the construction comes from a product of Gaussian basis functions whose
    widths vary with position, and the prefactor is exactly the normalizer that
    keeps it a valid covariance. Paciorek & Schervish 2004 generalize it to any
    stationary base kernel and to higher dimensions.) The prefactor is what does
    the work: it damps the covariance between two points that disagree about the
    lengthscale, which is what makes the whole thing PSD rather than merely
    plausible.

    Two properties worth checking against the formula. On the diagonal
    ``x = x'``, the prefactor is ``sqrt(2 l^2 / 2 l^2) = 1`` and the exponent is
    0, so ``k(x, x) = s2`` everywhere -- the kernel is unit-variance-normalized
    regardless of how ``l`` varies. And if ``l`` is *constant*, the prefactor is
    1 and ``l(x)^2 + l(x')^2 = 2 l^2``, so it collapses to
    ``s2 exp(-r^2 / (2 l^2))``, this repo's :class:`RBF` exactly (tested).

    Lengthscale parameterization. This prototype takes the simplest genuinely
    nonstationary choice, log-linear in the input:

        l(x) = exp(a + b x)

    which is positive by construction, has clean gradients, and -- at b = 0 --
    reduces to a constant lengthscale ``e^a``, giving the exact RBF limit above.
    ``b`` is the tilt: negative means the function gets rougher as x increases.
    A monotone lengthscale is the right shape for a chirp-like target and the
    wrong one for a bump; a richer ``l(.)`` (a second GP on log l, as in
    Paciorek & Schervish) is the natural extension and is *not* implemented
    here. 1D input only; the constructor rejects anything else rather than
    silently doing the wrong thing.

    theta = (log s2, a, b), with ``a`` already a log-lengthscale. Writing
    ``l = l(x)``, ``l' = l(x')``, ``S = l^2 + l'^2``, ``d = x - x'``:

        d(log k)/dl = 1/(2l) - l/S + 2 l d^2 / S^2

    (and symmetrically in l'), and since ``dl/da = l`` and ``dl/db = x l``, the
    two chain-rule factors

        A  = l  * d(log k)/dl  = 1/2 - l^2/S  + 2 l^2  d^2/S^2
        A' = l' * d(log k)/dl' = 1/2 - l'^2/S + 2 l'^2 d^2/S^2

    give

        dK/d(log s2) = K
        dK/da        = K * (A + A')
        dK/db        = K * (x A + x' A')

    Sanity check of that algebra against the RBF limit: at b = 0 (so l = l' = L,
    S = 2L^2) both A and A' collapse to ``d^2 / (2 L^2)``, hence
    ``dK/da = K d^2 / L^2`` -- precisely :class:`RBF`'s ``dK/d(log l)``, as it
    must be. The gradients are finite-difference-checked in the tests anyway.
    """

    names = ("s2", "a", "b")

    def __init__(self, s2: float = 1.0, a: float = 0.0, b: float = 0.0,
                 fixed: FixedArg = None):
        self._theta = np.array([np.log(s2), float(a), float(b)], dtype=float)
        self._fixed = self._mask(fixed)

    def lengthscale(self, X: np.ndarray) -> np.ndarray:
        """l(x) = exp(a + b x), shape (n,). Public so experiments can plot it."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != 1:
            raise ValueError("Gibbs kernel is 1D only: X must have shape (n, 1)")
        _, a, b = self._theta
        return np.exp(a + b * X[:, 0])

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        s2 = np.exp(self._theta[0])
        l1 = self.lengthscale(X1)[:, None]  # (n1, 1)
        l2 = self.lengthscale(X2)[None, :]  # (1, n2)
        S = l1**2 + l2**2
        d = X1[:, 0][:, None] - X2[:, 0][None, :]
        prefactor = np.sqrt(2.0 * l1 * l2 / S)
        return s2 * prefactor * np.exp(-(d**2) / S)

    def _grads_full(self, X1: np.ndarray, X2: np.ndarray) -> list[np.ndarray]:
        l1 = self.lengthscale(X1)[:, None]
        l2 = self.lengthscale(X2)[None, :]
        S = l1**2 + l2**2
        x1 = np.atleast_2d(X1)[:, 0][:, None]
        x2 = np.atleast_2d(X2)[:, 0][None, :]
        d2 = (x1 - x2) ** 2

        K = self(X1, X2)
        A = 0.5 - l1**2 / S + 2.0 * l1**2 * d2 / S**2
        A2 = 0.5 - l2**2 / S + 2.0 * l2**2 * d2 / S**2
        return [K, K * (A + A2), K * (x1 * A + x2 * A2)]

    def dK_dX1(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Gibbs is non-stationary: x enters through l(x) as well as through "
            "x - x', so the input-space derivative has an extra term and has "
            "not been derived here. Sparse GPs with a learned Z therefore do "
            "not support this kernel."
        )


class Sum(Kernel):
    """k = k1 + k2; gradients concatenate (linearity).

    Independent additive processes add their kernels, which is what lets the
    CO2 model be *read off the physics* (trend + season + short-term).
    Composition is via ``+``, and the composite's theta is the concatenation
    of its parts' free parameters -- a frozen one stays invisible to the
    optimizer through the composite too:

    >>> import numpy as np
    >>> k = RBF(s2=1.0, l=1.0) + Periodic(s2=1.0, l=1.0, p=1.0, fixed=["p"])
    >>> k.n_params                      # RBF's 2 + Periodic's free 2
    4
    >>> X = np.linspace(0, 2, 4).reshape(-1, 1)
    >>> bool(np.allclose(k(X, X), RBF()(X, X) + Periodic()(X, X)))
    True
    >>> len(k.grads(X))                 # one gradient per free parameter
    4
    """

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

    def grads(self, X1: np.ndarray,
              X2: Optional[np.ndarray] = None) -> list[np.ndarray]:
        return self.k1.grads(X1, X2) + self.k2.grads(X1, X2)

    def dK_dX1(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.k1.dK_dX1(X1, X2) + self.k2.dK_dX1(X1, X2)


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

    def grads(self, X1: np.ndarray,
              X2: Optional[np.ndarray] = None) -> list[np.ndarray]:
        X2 = X1 if X2 is None else X2
        K1, K2 = self.k1(X1, X2), self.k2(X1, X2)
        return ([g * K2 for g in self.k1.grads(X1, X2)]
                + [K1 * g for g in self.k2.grads(X1, X2)])

    def dK_dX1(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        K1, K2 = self.k1(X1, X2), self.k2(X1, X2)
        return (self.k1.dK_dX1(X1, X2) * K2[:, :, None]
                + K1[:, :, None] * self.k2.dK_dX1(X1, X2))
