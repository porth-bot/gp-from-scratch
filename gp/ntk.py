"""Analytic infinite-width kernels for the one-hidden-layer ReLU network,
and the closed-form gradient-descent dynamics they induce.

Both kernels reduce to two Gaussian expectations over w ~ N(0, I) (the
arc-cosine kernels of Cho & Saul 2009; derivations in
theory/derivations.md, Sec. 6, via a polar integral in the 2D plane
spanned by u and v; theta is the angle between u and v):

    kappa1(u, v) = E[relu(w.u) relu(w.v)]
                 = (1 / 2pi) ||u|| ||v|| ( sin theta + (pi - theta) cos theta )

    kappa0(u, v) = E[ 1[w.u > 0] 1[w.v > 0] ]
                 = (pi - theta) / (2 pi)
      (an orthant probability: the fraction of directions in the plane lying
       in the intersection of two half-planes whose normals differ by theta)

For f(x) = sqrt(2/m) a . relu(W x~) with both layers trained:

    NNGP(x, x')  = E_init[f(x) f(x')] = 2 kappa1(x~, x~')
    NTK(x, x')   = <df/dparams, df/dparams'>  ->  2 kappa1 + 2 kappa0 * (x~ . x~')
                   (a-gradients give kappa1; W-gradients give kappa0 times
                    the input inner product)

Linearized full-batch GD on L = 1/2 ||f - y||^2 with step lr (derivation in
theory/derivations.md, Sec. 7): residuals contract as
r_{k+1} = (I - lr * Theta) r_k, and summing the geometric series gives, for
any test set,

    f_k(X*) = f_0(X*) + Theta(X*, X) Theta^{-1} (I - (I - lr Theta)^k) (y - f_0(X)).

As k -> infinity (with lr < 2 / lambda_max) this is exact kernel
("ridgeless") regression with the NTK, plus the transient from f_0.
"""

import numpy as np

from .nn import _augment


def _angles(U, V):
    """cos(theta) matrix between rows of U and V, clipped into [-1, 1]."""
    nu = np.linalg.norm(U, axis=1)
    nv = np.linalg.norm(V, axis=1)
    c = (U @ V.T) / np.outer(nu, nv)
    return np.clip(c, -1.0, 1.0), nu, nv


def kappa0(X1, X2):
    U, V = _augment(X1), _augment(X2)
    c, _, _ = _angles(U, V)
    theta = np.arccos(c)
    return (np.pi - theta) / (2.0 * np.pi)


def kappa1(X1, X2):
    U, V = _augment(X1), _augment(X2)
    c, nu, nv = _angles(U, V)
    theta = np.arccos(c)
    return np.outer(nu, nv) * (np.sin(theta) + (np.pi - theta) * c) / (2.0 * np.pi)


def nngp_kernel(X1, X2):
    """Covariance of the network's outputs at initialization (m -> inf)."""
    return 2.0 * kappa1(X1, X2)


def ntk_kernel(X1, X2):
    """The neural tangent kernel of the two-layer ReLU network (m -> inf)."""
    U, V = _augment(X1), _augment(X2)
    return 2.0 * kappa1(X1, X2) + 2.0 * kappa0(X1, X2) * (U @ V.T)


def gd_prediction(X_train, y_train, X_test, f0_train, f0_test, lr, steps):
    """Closed-form test-set predictions of linearized GD after ``steps`` steps.

    Requires lr < 2 / lambda_max(Theta) for the geometric series to converge
    (asserted). Computed via the eigendecomposition of the (symmetric PSD)
    train NTK; the k-th power of (I - lr Theta) costs one eigh, not k matmuls.
    """
    T_tr = ntk_kernel(X_train, X_train)
    T_te = ntk_kernel(X_test, X_train)
    lam, Q = np.linalg.eigh(T_tr)
    assert lr * lam.max() < 2.0, "lr too large: linearized GD diverges"
    decay = (1.0 - lr * lam) ** steps
    # Theta^{-1} (I - (I - lr Theta)^k) = Q diag((1 - decay)/lam) Q^T
    mid = (1.0 - decay) / lam
    M = (Q * mid) @ Q.T
    return f0_test + T_te @ M @ (y_train - f0_train)
