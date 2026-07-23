"""Adam ascent on the log marginal likelihood, in log-hyperparameter space.

Why Adam rather than plain gradient ascent: the LML surface mixes directions
of very different curvature (signal variance vs lengthscale vs noise), and
Adam's per-coordinate step normalization handles that without a
hand-tuned per-parameter learning rate. Why not (L-)BFGS: it would work
(and is what sklearn uses); Adam keeps the repo dependency-free and is
15 lines. ML-II surfaces are multimodal in general -- restarts from
different initializations are the standard mitigation (used in the
experiments where it matters).

Adam (Kingma & Ba 2015), ascent form:

    m_t = b1 m_{t-1} + (1-b1) g_t          (first-moment EMA)
    v_t = b2 v_{t-1} + (1-b2) g_t^2        (second-moment EMA)
    m_hat = m_t / (1 - b1^t),  v_hat = v_t / (1 - b2^t)   (bias correction)
    theta_t = theta_{t-1} + lr * m_hat / (sqrt(v_hat) + eps)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

ValueAndGrad = Callable[[np.ndarray], "tuple[float, np.ndarray]"]
Callback = Callable[[int, np.ndarray, float], None]


def adam_maximize(
    value_and_grad: ValueAndGrad,
    theta0: np.ndarray,
    lr: float = 0.05,
    steps: int = 300,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    callback: Optional[Callback] = None,
) -> "tuple[np.ndarray, list[float]]":
    """Maximize a function of theta given ``value_and_grad(theta) -> (v, g)``.

    Returns (best_theta, history) where history is the list of values per
    step and best_theta is the iterate with the highest value seen (the
    last iterate of a fixed-step run is not guaranteed to be the best).

    Examples
    --------
    Maximize a concave quadratic whose argmax is known: ``f(theta) =
    -||theta - [1, -2]||^2``, so ``grad = -2 (theta - [1, -2])``.

    >>> import numpy as np
    >>> target = np.array([1.0, -2.0])
    >>> def value_and_grad(theta):
    ...     d = theta - target
    ...     return -float(d @ d), -2.0 * d
    >>> best, history = adam_maximize(value_and_grad, np.zeros(2), lr=0.1, steps=500)
    >>> bool(np.allclose(best, target, atol=1e-3))
    True
    >>> len(history)
    500
    >>> bool(history[-1] > history[0])       # it climbed
    True
    """
    theta = np.asarray(theta0, dtype=float).copy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    b1, b2 = betas
    best_val, best_theta = -np.inf, theta.copy()
    history: list[float] = []
    for t in range(1, steps + 1):
        val, g = value_and_grad(theta)
        history.append(val)
        if val > best_val:
            best_val, best_theta = val, theta.copy()
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        theta = theta + lr * m_hat / (np.sqrt(v_hat) + eps)
        if callback is not None:
            callback(t, theta, val)
    return best_theta, history


@dataclass
class MultiStartResult:
    """Outcome of :func:`maximize_lml_multistart`.

    Attributes
    ----------
    params : the best log-space hyperparameters found (already set on the model).
    lml : the log marginal likelihood at ``params`` (the max over restarts).
    lmls : per-restart converged LML, in restart order (``-inf`` for a restart
        whose optimization failed numerically, e.g. an init so extreme the
        Cholesky is not positive-definite). ``inits[0]`` is the model's original
        parameters when ``keep_init=True``.
    inits : the log-space initializations used, in restart order.
    best_restart : index into ``lmls``/``inits`` of the winning restart.
    """

    params: np.ndarray
    lml: float
    lmls: np.ndarray
    inits: np.ndarray
    best_restart: int


def maximize_lml_multistart(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    n_restarts: int = 8,
    *,
    bounds: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
    jitter: float = 1.5,
    lr: float = 0.05,
    steps: int = 300,
    keep_init: bool = True,
) -> MultiStartResult:
    """Type-II maximum likelihood from several initializations; keep the best.

    The evidence (log marginal likelihood) is a *non-convex* function of the
    hyperparameters and is genuinely multimodal in the standard case: a short
    lengthscale with little noise ("the data is signal") and a long lengthscale
    with large noise ("the data is noise") are competing explanations, each a
    local optimum (Rasmussen & Williams 2006, Sec. 5.4.1, Fig. 5.5). A single
    gradient ascent lands in whichever basin its initialization sits in, so the
    standard mitigation is to restart from several initializations and keep the
    fit with the highest evidence.

    Each restart runs :func:`adam_maximize` on ``model.lml_and_grad`` from a
    log-space initialization, and the model is left fit at the winning
    parameters. Restart 0 is the model's current parameters when
    ``keep_init=True`` (so multi-start never does worse than the plain single
    fit it wraps). The remaining inits come from one of two schemes:

    - ``bounds`` given (recommended): sample each hyperparameter *uniformly in
      log-space* within its ``[low, high]`` box -- the same strategy scikit-learn
      uses for ``n_restarts_optimizer``. This actually explores the space, so it
      can escape a bad starting basin. ``bounds`` has shape ``(d, 2)`` matching
      ``model.params`` (log s2, log lengthscales..., log noise).
    - ``bounds=None``: perturb the kept init by ``jitter``-scaled Gaussian noise
      in log-space. This is a *local* search -- fine for polishing a decent init,
      but it cannot cross into a distant basin, so pass ``bounds`` when the point
      is to defeat multimodality.

    A restart whose init is extreme enough to break the Cholesky is scored
    ``-inf`` and skipped rather than crashing the sweep.

    Parameters
    ----------
    model : a GP exposing ``params`` (log-space vector), ``lml_and_grad(X, y,
        params)`` and ``log_marginal_likelihood()`` -- e.g. ``GPRegressor``.
    n_restarts : total number of initializations (including the kept init).
    bounds : optional ``(d, 2)`` log-space box the random inits are drawn from.
    rng : NumPy Generator for the draws (pass one for reproducibility).
    jitter : std of the log-space Gaussian perturbations when ``bounds`` is None.

    Returns
    -------
    MultiStartResult -- see that class. ``model.params`` is set to the winner.

    Examples
    --------
    A deliberately multimodal 12-point set (sparse samples of a wiggly signal):
    a bad single start from a long lengthscale gets trapped in the "it's all
    noise" mode, while multi-start over a broad log-space box recovers the far
    better signal explanation.

    >>> import numpy as np
    >>> from gp.gp import GPRegressor
    >>> from gp.kernels import RBF
    >>> rng = np.random.default_rng(0)
    >>> X = np.linspace(-4, 4, 12).reshape(-1, 1)
    >>> y = np.sin(1.7 * X).ravel() + 0.15 * rng.standard_normal(12)
    >>> stuck = GPRegressor(RBF(s2=1.0, l=30.0), noise_var=0.6)
    >>> _ = adam_maximize(lambda p: stuck.lml_and_grad(X, y, p), stuck.params,
    ...                   lr=0.05, steps=400)
    >>> stuck_lml = stuck.lml_and_grad(X, y, stuck.params)[0]
    >>> model = GPRegressor(RBF(s2=1.0, l=30.0), noise_var=0.6)
    >>> box = np.log([[1e-2, 1e2], [1e-1, 1e2], [1e-4, 1e0]])  # s2, l, noise
    >>> res = maximize_lml_multistart(model, X, y, n_restarts=8, bounds=box,
    ...                               rng=np.random.default_rng(0), steps=400)
    >>> bool(res.lml > stuck_lml + 1.0)          # escaped the bad basin
    True
    >>> bool(res.lml == np.max(res.lmls))        # returns the best restart
    True
    """
    rng = np.random.default_rng() if rng is None else rng
    theta0 = np.asarray(model.params, dtype=float).copy()
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (theta0.size, 2):
            raise ValueError(
                f"bounds must have shape ({theta0.size}, 2), got {bounds.shape}"
            )

    inits = [theta0.copy()] if keep_init else []
    while len(inits) < n_restarts:
        if bounds is not None:
            inits.append(rng.uniform(bounds[:, 0], bounds[:, 1]))
        else:
            inits.append(theta0 + jitter * rng.standard_normal(theta0.shape))

    lmls = np.full(len(inits), -np.inf)
    best_params = theta0.copy()
    best_lml = -np.inf
    for i, p0 in enumerate(inits):
        try:
            best_p, _ = adam_maximize(
                lambda p: model.lml_and_grad(X, y, p), p0, lr=lr, steps=steps
            )
            lml = model.lml_and_grad(X, y, best_p)[0]
        except (np.linalg.LinAlgError, FloatingPointError, ValueError):
            continue
        if not np.isfinite(lml):
            continue
        lmls[i] = lml
        if lml > best_lml:
            best_lml = lml
            best_params = np.asarray(best_p, dtype=float).copy()

    best_restart = int(np.argmax(lmls))
    model.params = best_params
    model.fit(X, y)  # leave the model conditioned at the winning hyperparameters
    return MultiStartResult(
        params=best_params,
        lml=float(np.max(lmls)),
        lmls=lmls,
        inits=np.array(inits),
        best_restart=best_restart,
    )
