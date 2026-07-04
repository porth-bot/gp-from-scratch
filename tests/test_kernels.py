"""Kernels: gradient checks, positive-definiteness, composition rules."""

import numpy as np
import pytest

from gp.kernels import RBF, Matern, Periodic, Product, Sum, sqdist

RNG = np.random.default_rng(0)


def finite_diff_grads(kernel, X, eps=1e-6):
    """Central differences of K(X, X) w.r.t. each theta_i."""
    out = []
    theta0 = kernel.theta
    for i in range(kernel.n_params):
        for sign in (+1, -1):
            t = theta0.copy()
            t[i] += sign * eps
            kernel.theta = t
            if sign > 0:
                plus = kernel(X, X)
            else:
                minus = kernel(X, X)
        out.append((plus - minus) / (2 * eps))
    kernel.theta = theta0
    return out


ALL_KERNELS = [
    RBF(s2=2.0, l=0.7),
    Matern(nu=0.5, s2=1.5, l=0.9),
    Matern(nu=1.5, s2=0.8, l=1.3),
    Matern(nu=2.5, s2=1.1, l=0.5),
    Periodic(s2=1.2, l=0.8, p=2.3),
    Sum(RBF(s2=1.0, l=0.5), Matern(nu=1.5, s2=0.5, l=2.0)),
    Product(RBF(s2=1.0, l=1.5), Periodic(s2=0.9, l=1.1, p=1.7)),
]
IDS = ["rbf", "matern12", "matern32", "matern52", "periodic", "sum", "product"]


@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=IDS)
def test_gradients_match_finite_differences(kernel):
    X = RNG.uniform(-2, 2, size=(7, 1))
    analytic = kernel.grads(X)
    numeric = finite_diff_grads(kernel, X)
    assert len(analytic) == kernel.n_params
    for a, n in zip(analytic, numeric):
        np.testing.assert_allclose(a, n, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=IDS)
def test_kernel_matrices_are_positive_semidefinite(kernel):
    X = RNG.uniform(-3, 3, size=(40, 1))
    K = kernel(X, X)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(K)
    assert eigs.min() > -1e-9 * eigs.max()


def test_sqdist_matches_direct_computation():
    A = RNG.standard_normal((5, 3))
    B = RNG.standard_normal((4, 3))
    direct = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    np.testing.assert_allclose(sqdist(A, B), direct, atol=1e-10)


def test_periodic_kernel_is_exactly_periodic():
    k = Periodic(s2=1.0, l=0.7, p=1.9)
    x = np.array([[0.3]])
    shifted = np.array([[0.3 + 3 * 1.9]])  # three full periods away
    np.testing.assert_allclose(k(x, shifted), k(x, x), rtol=1e-12)


def test_theta_roundtrip_through_composites():
    k = Sum(RBF(), Product(Matern(nu=2.5), Periodic()))
    theta = RNG.standard_normal(k.n_params)
    k.theta = theta
    np.testing.assert_allclose(k.theta, theta)
