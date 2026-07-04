"""A width-m one-hidden-layer ReLU network with backprop done by hand.

Architecture (NTK parameterization, Jacot et al. 2018):

    f(x) = sqrt(2/m) * sum_i a_i relu(w_i . x~),    x~ = (x, 1),

with w_i ~ N(0, I_2) and a_i ~ N(0, 1) at initialization, and BOTH layers
trained. The sqrt(2/m) lives in the *function*, not the initialization --
that is what keeps the network's output O(1) and its tangent kernel
deterministic as m -> infinity, giving the two limits this repo verifies:

- At init, f is a Gaussian process with covariance 2*kappa1 (gp/ntk.py).
- Under full-batch gradient descent with small lr, the function evolves as
  kernel regression with the (constant) NTK.

Gradients (hand-derived; FD-checked in tests). With z_i = w_i . x~,
h_i = relu(z_i), s_i = 1[z_i > 0]:

    df/da_i = sqrt(2/m) h_i
    df/dw_i = sqrt(2/m) a_i s_i x~

and for the squared loss L = 1/2 sum_n (f(x_n) - y_n)^2 with residuals r_n,
sum over data of r_n times the above.

The empirical NTK is the parameter-gradient Gram matrix:

    Theta_emp(x, x') = <df(x)/dparams, df(x')/dparams>
                     = (2/m) [ sum_i h_i h_i'  +  (x~ . x~') sum_i a_i^2 s_i s_i' ].
"""

import numpy as np


def _augment(X):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.hstack([X, np.ones((X.shape[0], 1))])


class TwoLayerReLU:
    def __init__(self, width, rng):
        self.m = width
        self.W = rng.standard_normal((width, 2))  # rows w_i, acting on (x, 1)
        self.a = rng.standard_normal(width)
        self.scale = np.sqrt(2.0 / width)

    def _pre(self, X):
        """Preactivations Z (n, m) and hidden H, gate S."""
        Z = _augment(X) @ self.W.T
        return Z, np.maximum(Z, 0.0), (Z > 0).astype(float)

    def forward(self, X):
        _, H, _ = self._pre(X)
        return self.scale * H @ self.a

    def loss_grads(self, X, y):
        """Gradients of L = 1/2 ||f - y||^2 w.r.t. (a, W)."""
        Xa = _augment(X)
        Z = Xa @ self.W.T
        H = np.maximum(Z, 0.0)
        S = (Z > 0).astype(float)
        r = self.scale * H @ self.a - y                      # residuals (n,)
        g_a = self.scale * H.T @ r                           # (m,)
        g_W = self.scale * ((S * r[:, None]).T * self.a[:, None]) @ Xa  # (m, 2)
        return g_a, g_W

    def gd_step(self, X, y, lr):
        g_a, g_W = self.loss_grads(X, y)
        self.a -= lr * g_a
        self.W -= lr * g_W

    def empirical_ntk(self, X1, X2):
        """<grad_params f(x), grad_params f(x')> -- converges to the analytic
        NTK as width -> infinity (verified in tests at finite width)."""
        Xa1, Xa2 = _augment(X1), _augment(X2)
        _, H1, S1 = self._pre(X1)
        _, H2, S2 = self._pre(X2)
        term_a = H1 @ H2.T
        term_W = ((S1 * self.a) @ (S2 * self.a).T) * (Xa1 @ Xa2.T)
        return (2.0 / self.m) * (term_a + term_W)
