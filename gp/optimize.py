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

from typing import Callable, Optional

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
