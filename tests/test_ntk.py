"""The NTK stack, validated layer by layer:

1. The arc-cosine formulas against brute-force Monte Carlo of the defining
   Gaussian expectations (the *math* check).
2. The network's hand-derived gradients against finite differences.
3. The empirical NTK of a wide finite network against the analytic kernel.
4. The empirical output covariance at init against the analytic NNGP.
5. Actual gradient-descent training of a wide network against the
   closed-form linearized dynamics (the theorem, observed).
"""

import numpy as np

from gp.nn import TwoLayerReLU, _augment
from gp.ntk import gd_prediction, kappa0, kappa1, nngp_kernel, ntk_kernel


def test_kappa_formulas_match_monte_carlo():
    rng = np.random.default_rng(0)
    X = np.array([[0.7], [-1.3], [2.0]])
    U = _augment(X)
    w = rng.standard_normal((2_000_000, 2))
    Zu = w @ U.T                                  # (N, 3)
    relu = np.maximum(Zu, 0.0)
    ind = (Zu > 0).astype(float)

    mc_k1 = relu.T @ relu / len(w)
    mc_k0 = ind.T @ ind / len(w)
    np.testing.assert_allclose(kappa1(X, X), mc_k1, rtol=0.01, atol=1e-3)
    np.testing.assert_allclose(kappa0(X, X), mc_k0, rtol=0.01, atol=1e-3)


def test_network_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    net = TwoLayerReLU(width=16, rng=rng)
    X = rng.uniform(-2, 2, size=(6, 1))
    y = rng.standard_normal(6)
    g_a, g_W = net.loss_grads(X, y)

    def loss():
        r = net.forward(X) - y
        return 0.5 * float(r @ r)

    eps = 1e-6
    for i in range(4):  # spot-check a few coordinates of each block
        net.a[i] += eps; lp = loss()
        net.a[i] -= 2 * eps; lm = loss()
        net.a[i] += eps
        np.testing.assert_allclose(g_a[i], (lp - lm) / (2 * eps), rtol=1e-4, atol=1e-7)
    for i in range(3):
        for j in range(2):
            net.W[i, j] += eps; lp = loss()
            net.W[i, j] -= 2 * eps; lm = loss()
            net.W[i, j] += eps
            np.testing.assert_allclose(g_W[i, j], (lp - lm) / (2 * eps), rtol=1e-4, atol=1e-7)


def test_empirical_ntk_converges_to_analytic():
    rng = np.random.default_rng(2)
    X = np.linspace(-2, 2, 6)[:, None]
    analytic = ntk_kernel(X, X)
    net = TwoLayerReLU(width=16_384, rng=rng)
    emp = net.empirical_ntk(X, X)
    rel = np.linalg.norm(emp - analytic) / np.linalg.norm(analytic)
    assert rel < 0.05, rel


def test_init_covariance_matches_nngp():
    """For one hidden layer the init covariance equals the NNGP exactly at
    ANY width (mean of i.i.d. per-neuron terms) -- the tolerance below is
    pure ensemble Monte-Carlo error, not a finite-width effect. What
    converges with width is Gaussianity (see experiments)."""
    rng = np.random.default_rng(3)
    X = np.array([[-1.5], [0.0], [0.8], [2.2]])
    Xa = _augment(X)
    n_nets, m = 6000, 256
    # vectorized ensemble of independent networks
    W = rng.standard_normal((n_nets, m, 2))
    a = rng.standard_normal((n_nets, m))
    Z = np.einsum("kmi,ni->knm", W, Xa)
    f = np.sqrt(2.0 / m) * np.einsum("knm,km->kn", np.maximum(Z, 0), a)
    emp_cov = f.T @ f / n_nets
    analytic = nngp_kernel(X, X)
    rel = np.linalg.norm(emp_cov - analytic) / np.linalg.norm(analytic)
    assert rel < 0.08, rel


def test_wide_network_training_matches_linearized_dynamics():
    """Train an actual width-4096 network by full-batch GD and compare its
    test predictions after k steps with the closed-form NTK dynamics --
    the Jacot linearization, observed at finite width."""
    rng = np.random.default_rng(4)
    X_tr = np.linspace(-2, 2, 12)[:, None]
    y = np.sin(2.0 * X_tr[:, 0])
    X_te = np.linspace(-2.5, 2.5, 40)[:, None]

    net = TwoLayerReLU(width=4096, rng=rng)
    f0_tr, f0_te = net.forward(X_tr), net.forward(X_te)
    lr, steps = 0.05, 400
    for _ in range(steps):
        net.gd_step(X_tr, y, lr)

    predicted = gd_prediction(X_tr, y, X_te, f0_tr, f0_te, lr, steps)
    err = np.max(np.abs(net.forward(X_te) - predicted))
    assert err < 0.05, err  # function values are O(1); deviation is O(1/sqrt(m))
