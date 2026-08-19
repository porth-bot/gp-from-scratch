"""The Laplace approximation: GPs with a likelihood that is not Gaussian.

Everything else in this repo assumes ``y = f(x) + eps`` with Gaussian eps, and
that assumption is what makes the posterior closed form -- a Gaussian prior
conjugate to a Gaussian likelihood is Gaussian, and Sec. 1's Schur complement
does the rest. The README's Limitations have said since v1.1 that the natural
next gap is a likelihood that breaks that: binary labels, counts, anything
whose observation model is not a normal density. This module is that gap.

Once the likelihood is non-Gaussian the posterior over the latent values

    p(f | y) = p(y | f) N(f | 0, K) / p(y)

has no closed form and ``p(y)`` is an n-dimensional integral. The Laplace
approximation replaces it with the Gaussian that matches the posterior's *mode*
and the curvature there:

    q(f | y) = N(f | fhat, (K^-1 + W)^-1),
    fhat = argmax Psi(f),   Psi(f) = log p(y|f) - (1/2) f^T K^-1 f,
    W = -grad grad log p(y | fhat)   (diagonal, since the likelihood factorizes)

and approximates the evidence by the Gaussian integral around that mode,

    log p(y) ~= Psi(fhat) - (1/2) log |I + K W|.

Three things make this work in practice and all three are in the code below.

**The mode is unique, for the likelihoods here.** ``Psi`` is the sum of a
concave log-likelihood and a concave quadratic, so it is concave and Newton's
method converges to the single maximum from any start. Bernoulli-logit, Poisson
with a log link and Gaussian are all log-concave in f; ``tests/test_laplace.py``
checks the uniqueness by starting the iteration from several places and landing
on the same mode.

**The linear algebra is arranged so nothing ill-conditioned is inverted.**
``K^-1 + W`` is the natural object and the wrong one to form: K is
near-singular whenever the kernel is smooth, which is always. The standard
rearrangement (Rasmussen & Williams, Alg. 3.1) works with

    B = I + W^(1/2) K W^(1/2),

which is symmetric positive definite with eigenvalues in ``[1, 1 + n max(W)
max(K)]`` -- a conditioning that depends on W and K only through their sizes,
not through K's smallest eigenvalue. Every solve goes through ``chol(B)``, and
``|I + KW| = |B|`` gives the determinant for free. This repo's ``gp/linalg.py``
supplies the triangular solves (``np.linalg.solve`` on a Cholesky factor runs a
general LU -- the bug Sec. 14 found).

**The Newton step needs a line search.** The undamped step can overshoot for a
likelihood whose curvature changes quickly (Poisson's W is ``exp(f)``, so a
step that doubles f multiplies the curvature by e^f). The search here is done
in ``a = K^-1 f`` coordinates rather than in f: any convex combination of two
``a`` vectors maps to the same convex combination of the ``f`` vectors, so
``Psi`` at an intermediate point costs no extra solve.

What this module does *not* do, and the limitation is real
----------------------------------------------------------
There are no gradients of the approximate log marginal likelihood with respect
to the hyperparameters. They exist (Rasmussen & Williams, Alg. 5.1) and they
are not a small addition: ``fhat`` itself depends on theta, so the total
derivative has an *implicit* term through ``d fhat / d theta`` on top of the
explicit one, and that term needs the likelihood's third derivative. So
hyperparameter selection here is a grid search over the approximate evidence
(:func:`grid_search`), which is honest for two parameters and does not scale.
That is the next gap this module opens, and the README says so.

References: Rasmussen & Williams, *Gaussian Processes for Machine Learning*,
Chapter 3 (Algorithms 3.1 and 3.2). The derivation, including why the mode
satisfies ``fhat = K grad log p(y | fhat)`` and where the log-determinant term
comes from, is worked out in ``theory/derivations.md`` Sec. 10.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .linalg import cho_solve, solve_lower

__all__ = [
    "Likelihood", "Bernoulli", "Poisson", "GaussianLikelihood",
    "LaplaceGP", "LaplaceFit", "grid_search",
]


# ---------------------------------------------------------------------------
# Likelihoods
# ---------------------------------------------------------------------------
class Likelihood:
    """A factorizing observation model ``p(y_i | f_i)``, in log space.

    Subclasses provide the log density and its first two derivatives with
    respect to the latent value. Everything is elementwise, shape (n,).

    The second derivative must be <= 0 (log-concavity) for the Newton iteration
    below to be guaranteed to converge; ``W = -d2`` is then a nonnegative
    diagonal and ``B = I + sqrt(W) K sqrt(W)`` is positive definite. A
    likelihood that is not log-concave can still be used, but the guarantees go
    and so does ``sqrt(W)``, so the constructor of :class:`LaplaceGP` checks the
    sign at the mode rather than trusting it.
    """

    def log_pdf(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def d1(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        """d log p(y|f) / df."""
        raise NotImplementedError

    def d2(self, y: np.ndarray, f: np.ndarray) -> np.ndarray:
        """d^2 log p(y|f) / df^2, <= 0 where the likelihood is log-concave."""
        raise NotImplementedError

    def check_targets(self, y: np.ndarray) -> np.ndarray:
        """Validate and canonicalize the observations. Default: pass through."""
        return np.asarray(y, dtype=float)

    def mean(self, f: np.ndarray) -> np.ndarray:
        """E[y | f], for turning a latent prediction into an observable one."""
        raise NotImplementedError


class Bernoulli(Likelihood):
    """Binary labels with a logit link: ``p(y | f) = sigmoid(y f)``, y in {-1, +1}.

    Written with the numerically safe form of the log sigmoid,
    ``log sigmoid(z) = -softplus(-z)`` evaluated as ``-log1p(exp(-|z|)) +
    min(z, 0)``, so a confidently-correct point (z = 40) does not go through
    ``log(1 - 1e-18)`` and a confidently-wrong one (z = -40) returns -40 rather
    than -inf.

    The derivatives are the textbook ones and are worth writing out because the
    second one does not depend on y at all:

        d1 = (y+1)/2 - sigmoid(f)          (the residual: target minus probability)
        d2 = -sigmoid(f) (1 - sigmoid(f))  (<= 0 always, so W >= 0 always)

    That W does not see the labels is what makes the logit model's Newton
    iteration so well behaved: the curvature is bounded by 1/4 everywhere,
    whatever the data.
    """

    def check_targets(self, y):
        y = np.asarray(y)
        vals = set(np.unique(y).tolist())
        if vals <= {0.0, 1.0, 0, 1}:
            return 2.0 * np.asarray(y, dtype=float) - 1.0       # {0,1} -> {-1,+1}
        if not vals <= {-1.0, 1.0, -1, 1}:
            raise ValueError("Bernoulli targets must be in {0,1} or {-1,+1}")
        return np.asarray(y, dtype=float)

    @staticmethod
    def _log_sigmoid(z):
        return np.minimum(z, 0.0) - np.log1p(np.exp(-np.abs(z)))

    @staticmethod
    def _sigmoid(z):
        # branch-free stable logistic
        out = np.empty_like(z, dtype=float)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        e = np.exp(z[~pos])
        out[~pos] = e / (1.0 + e)
        return out

    def log_pdf(self, y, f):
        return self._log_sigmoid(y * f)

    def d1(self, y, f):
        return 0.5 * (y + 1.0) - self._sigmoid(f)

    def d2(self, y, f):
        p = self._sigmoid(f)
        return -p * (1.0 - p)

    def mean(self, f):
        return self._sigmoid(f)


class Poisson(Likelihood):
    """Counts with a log link: ``p(y | f) = exp(y f - e^f) / y!``.

    Included because it fails differently from Bernoulli, which is the point of
    having two. Bernoulli's curvature is bounded (``W <= 1/4``) and its mean is
    bounded in (0,1); Poisson's ``W = e^f`` is unbounded above and unbounded
    *below* toward zero, so the Newton step can overshoot badly and the line
    search below is not decorative. The log factorial uses ``gammaln`` written
    out via ``math.lgamma`` so the module stays NumPy-only.
    """

    def check_targets(self, y):
        y = np.asarray(y, dtype=float)
        if np.any(y < 0) or np.any(y != np.round(y)):
            raise ValueError("Poisson targets must be nonnegative integers")
        return y

    def log_pdf(self, y, f):
        from math import lgamma
        log_fact = np.array([lgamma(v + 1.0) for v in np.atleast_1d(y).ravel()])
        return y * f - np.exp(f) - log_fact.reshape(np.shape(y))

    def d1(self, y, f):
        return y - np.exp(f)

    def d2(self, y, f):
        return -np.exp(f)

    def mean(self, f):
        return np.exp(f)


class GaussianLikelihood(Likelihood):
    """``p(y | f) = N(y | f, sigma^2)``. Here to prove the algebra, not to use.

    With this likelihood the Laplace approximation is *exact*: the posterior is
    Gaussian, so the mode is its mean, the curvature ``W = 1/sigma^2`` is
    constant, and the log-determinant term completes the exact log marginal
    likelihood. So :class:`LaplaceGP` with this likelihood must reproduce
    :class:`gp.gp.GPRegressor` to floating point -- mode against posterior mean,
    predictive mean and variance, and ``log_marginal_likelihood`` against the
    exact one.

    That is the strongest test available for this module, because it checks the
    whole pipeline (Newton iteration, the ``B = I + sqrt(W) K sqrt(W)``
    rearrangement, the determinant, Alg. 3.2's predictive variance) against an
    answer that is known in closed form and computed by entirely separate code.
    Every other check here is against quadrature or Monte Carlo.
    """

    def __init__(self, noise_var: float = 0.1):
        self.noise_var = float(noise_var)
        if self.noise_var <= 0:
            raise ValueError("noise_var must be > 0")

    def log_pdf(self, y, f):
        s2 = self.noise_var
        return -0.5 * np.log(2.0 * np.pi * s2) - 0.5 * (y - f) ** 2 / s2

    def d1(self, y, f):
        return (y - f) / self.noise_var

    def d2(self, y, f):
        return np.full(np.shape(f), -1.0 / self.noise_var)

    def mean(self, f):
        return f


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------
class LaplaceFit:
    """What one Newton solve leaves behind. Everything prediction needs.

    Attributes
    ----------
    f : ndarray, (n,)
        The posterior mode ``fhat``.
    a : ndarray, (n,)
        ``K^-1 fhat``, which at the mode equals ``grad log p(y | fhat)``. Kept
        because it is what the predictive mean contracts against, and because
        the identity is a free consistency check (:meth:`stationarity`).
    W : ndarray, (n,)
        ``-d2 log p(y | fhat)``, the diagonal curvature.
    L : ndarray, (n, n)
        Cholesky factor of ``B = I + sqrt(W) K sqrt(W)``.
    log_ml : float
        The approximate log marginal likelihood.
    n_iter : int
        Newton iterations taken.
    objective : list of float
        ``Psi(f)`` at each iteration, so a caller can see it increase
        monotonically -- the line search's job, and the thing that would show a
        broken step immediately.
    """

    def __init__(self, f, a, W, L, log_ml, n_iter, objective):
        self.f = f
        self.a = a
        self.W = W
        self.L = L
        self.log_ml = float(log_ml)
        self.n_iter = int(n_iter)
        self.objective = list(objective)

    def stationarity(self, lik, y):
        """``max |K^-1 fhat - grad log p(y | fhat)|``: zero at a true mode.

        The two are computed by different routes -- ``a`` comes out of the
        Newton solve and the gradient straight from the likelihood -- so this is
        a real check rather than an identity of the code.
        """
        return float(np.max(np.abs(self.a - lik.d1(y, self.f))))


class LaplaceGP:
    """Zero-mean GP with a non-Gaussian likelihood, by the Laplace approximation.

    Parameters
    ----------
    kernel : gp.kernels.Kernel
    likelihood : Likelihood
    jitter : float
        Relative diagonal jitter on K, as a multiple of ``mean(diag(K))`` --
        the same convention as :class:`gp.gp.GPRegressor`.

    Examples
    --------
    Four points, two of each class, separated at the origin:

    >>> import numpy as np
    >>> from gp.kernels import RBF
    >>> X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    >>> y = np.array([-1, -1, 1, 1])
    >>> m = LaplaceGP(RBF(s2=4.0, l=1.0), Bernoulli()).fit(X, y)
    >>> Xs = np.array([[-1.5], [0.0], [1.5]])
    >>> p, _ = m.predict_prob(Xs)
    >>> bool(p[0] < 0.5 < p[2])               # the classes come out the right way
    True
    >>> bool(abs(p[1] - 0.5) < 1e-9)          # exactly undecided at the origin
    True

    And averaging over the latent is not the same as pushing its mean through
    the link. The plug-in is more confident, everywhere:

    >>> plug = Bernoulli().mean(m.predict(Xs)[0])
    >>> float(round(p[2], 3)), float(round(plug[2], 3))
    (0.747, 0.813)
    """

    JITTER = 1e-10

    def __init__(self, kernel, likelihood: Likelihood, jitter: float = JITTER):
        self.kernel = kernel
        self.likelihood = likelihood
        self.jitter = float(jitter)
        self._fitted = False

    # -- mode finding --------------------------------------------------------
    def fit(self, X, y, tol: float = 1e-10, max_iter: int = 100,
            f_init: Optional[np.ndarray] = None) -> "LaplaceGP":
        """Find the posterior mode by Newton's method (R&W Alg. 3.1).

        Each iteration solves the Newton system for ``Psi`` at the current f.
        Writing ``W = -d2`` and ``b = W f + d1``, the update is

            a_new = b - W^(1/2) B^-1 W^(1/2) K b,     f_new = K a_new,

        which is the Newton step ``f + (K^-1 + W)^-1 (d1 - K^-1 f)`` rearranged
        so that only ``B = I + W^(1/2) K W^(1/2)`` is factorized -- never K.
        The two are algebraically identical and numerically are not: K's
        condition number is unbounded, B's is ``1 + n max(W) max(K)``.

        ``tol`` is on the *increase in Psi*, not on the step size: the objective
        is what the method maximizes, and a small step with a large objective
        change means the curvature is large, which is exactly when stopping
        would be wrong.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = self.likelihood.check_targets(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        n = X.shape[0]

        K = self.kernel(X, X)
        K = K + np.eye(n) * (self.jitter * float(np.mean(np.diag(K))))

        a = np.zeros(n) if f_init is None else np.linalg.solve(K, np.asarray(f_init))
        f = K @ a
        psi = self._psi(y, f, a)
        history = [psi]

        L = None
        for it in range(1, max_iter + 1):
            W = -self.likelihood.d2(y, f)
            if np.any(W < 0):
                raise ValueError(
                    "the likelihood is not log-concave at the current iterate "
                    "(W has negative entries); the Newton scheme here assumes "
                    "W >= 0 and takes its square root")
            sW = np.sqrt(W)
            B = np.eye(n) + sW[:, None] * K * sW[None, :]
            L = np.linalg.cholesky(B)

            b = W * f + self.likelihood.d1(y, f)
            a_new = b - sW * cho_solve(L, sW * (K @ b))

            # Line search in ``a`` coordinates. f = K a is linear, so the
            # convex combination t*a_new + (1-t)*a maps to the same combination
            # of the f's and Psi costs no extra solve. Halving from the full
            # Newton step is the standard damping and terminates because Psi is
            # concave, so the Newton direction is an ascent direction.
            step = 1.0
            for _ in range(50):
                a_try = a + step * (a_new - a)
                f_try = K @ a_try
                psi_try = self._psi(y, f_try, a_try)
                if psi_try >= psi:
                    break
                step *= 0.5
            else:
                # no improvement at any damping: already at the mode to
                # floating point, so stop rather than loop
                break

            gain = psi_try - psi
            a, f, psi = a_try, f_try, psi_try
            history.append(psi)
            if gain < tol:
                break

        # Recompute the factorization at the final f so W, L and f are mutually
        # consistent -- the loop's L belongs to the *previous* iterate, and the
        # predictive variance and the determinant both read W and L.
        W = -self.likelihood.d2(y, f)
        sW = np.sqrt(W)
        B = np.eye(n) + sW[:, None] * K * sW[None, :]
        L = np.linalg.cholesky(B)
        log_ml = psi - float(np.sum(np.log(np.diag(L))))

        self.X, self.y, self.K = X, y, K
        self.fit_ = LaplaceFit(f, a, W, L, log_ml, it, history)
        self._fitted = True
        return self

    def _psi(self, y, f, a):
        """``Psi(f) = log p(y|f) - (1/2) f^T K^-1 f``, with ``a = K^-1 f`` given."""
        return float(np.sum(self.likelihood.log_pdf(y, f)) - 0.5 * float(a @ f))

    # -- prediction ----------------------------------------------------------
    def predict(self, Xs) -> "tuple[np.ndarray, np.ndarray]":
        """Latent posterior mean and variance at Xs (R&W Alg. 3.2).

        The mean is ``k_*^T grad log p(y | fhat)``, which is the same expression
        as GP regression's ``k_*^T alpha`` with the likelihood's residual in
        place of ``(K + sigma^2 I)^-1 y``. The variance subtracts

            v^T v,   v = L^-1 (W^(1/2) k_*),

        which is ``k_*^T (K + W^-1)^-1 k_*`` -- i.e. the usual reduction, with
        the likelihood's local curvature playing the role of the noise. Where
        the curvature is small (a confidently-classified point, where W -> 0)
        that observation removes almost no variance, which is the honest
        behaviour and is why a GP classifier stays uncertain in regions of
        saturated labels.
        """
        assert self._fitted, "call fit() first"
        Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
        fit = self.fit_
        Ks = self.kernel(self.X, Xs)                       # (n, m)
        mean = Ks.T @ fit.a
        v = solve_lower(fit.L, np.sqrt(fit.W)[:, None] * Ks)
        var = np.diag(self.kernel(Xs, Xs)).copy() - np.sum(v ** 2, axis=0)
        return mean, np.maximum(var, 0.0)

    def predict_prob(self, Xs, nodes: int = 64) -> "tuple[np.ndarray, np.ndarray]":
        """Posterior predictive ``E[y* | y]``, averaged over the latent Gaussian.

        Returns ``(mean, latent_var)``. The expectation
        ``int lik.mean(f) N(f | mu_*, s_*^2) df`` is done by Gauss-Hermite
        quadrature, which for a smooth bounded integrand and a Gaussian weight
        is exact to machine precision at a few tens of nodes -- checked against
        both a finer rule and plain Monte Carlo in the tests.

        Averaging matters and is not a formality: for the logit link,
        ``sigmoid(mu_*)`` ignores the latent variance entirely and is
        over-confident exactly where the GP is least sure. The gap between the
        two is measured in ``experiments/laplace.py``.
        """
        mean, var = self.predict(Xs)
        x, w = np.polynomial.hermite_e.hermegauss(int(nodes))
        w = w / np.sqrt(2.0 * np.pi)                 # probabilists' weights sum to 1
        sd = np.sqrt(var)
        grid = mean[:, None] + sd[:, None] * x[None, :]
        return (self.likelihood.mean(grid) @ w), var

    def log_marginal_likelihood(self) -> float:
        """The approximate log evidence. **Not a bound in either direction.**

        ``Psi(fhat) - (1/2) log|I + KW|`` is the Gaussian integral around the
        mode, and the error is whatever the true posterior's non-Gaussianity
        contributes -- with no sign attached, unlike the variational bound of
        Sec. 9, which is always below. ``experiments/laplace.py`` measures the
        error and its sign against an importance-sampling estimate of the exact
        evidence.
        """
        assert self._fitted, "call fit() first"
        return self.fit_.log_ml


# ---------------------------------------------------------------------------
# Hyperparameters, without gradients
# ---------------------------------------------------------------------------
def grid_search(kernel_factory, likelihood, X, y, grids, verbose=False):
    """ML-II by exhaustive search over a grid of hyperparameters.

    ``kernel_factory(*values) -> Kernel`` is called once per grid point and
    ``grids`` is a sequence of 1-D arrays, one per argument. Returns
    ``(best_values, best_log_ml, table)`` where ``table`` is the full
    ``log_ml`` array over the grid, so a caller can see the surface rather than
    just the argmax -- Sec. 9 found ML-II's own surface to be multimodal on the
    Gaussian likelihood and there is no reason to expect better here.

    This exists because the approximate evidence has no gradient in this module
    (see the module docstring), so ``gp.optimize.adam_maximize`` cannot be used.
    It is fine for two parameters and hopeless for ten, and that is the honest
    statement of where this module stops.
    """
    mesh = np.meshgrid(*grids, indexing="ij")
    table = np.full(mesh[0].shape, -np.inf)
    best = (None, -np.inf)
    for idx in np.ndindex(table.shape):
        values = tuple(float(m[idx]) for m in mesh)
        try:
            model = LaplaceGP(kernel_factory(*values), likelihood).fit(X, y)
            table[idx] = model.log_marginal_likelihood()
        except np.linalg.LinAlgError:
            continue                                  # keep -inf, report later
        if table[idx] > best[1]:
            best = (values, table[idx])
        if verbose:
            print(f"  {values} -> {table[idx]:.4f}")
    return best[0], best[1], table
