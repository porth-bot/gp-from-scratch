"""Kernels: gradient checks, positive-definiteness, composition rules."""

import numpy as np
import pytest

from gp.kernels import (
    ARD,
    RBF,
    Matern,
    Periodic,
    Product,
    RationalQuadratic,
    Sum,
    sqdist,
)

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
    RationalQuadratic(s2=1.3, l=0.9, alpha=0.5),
    RationalQuadratic(s2=0.7, l=1.4, alpha=5.0),
    ARD(s2=1.4, lengthscales=[0.6]),
    Sum(RBF(s2=1.0, l=0.5), Matern(nu=1.5, s2=0.5, l=2.0)),
    Product(RBF(s2=1.0, l=1.5), Periodic(s2=0.9, l=1.1, p=1.7)),
]
IDS = ["rbf", "matern12", "matern32", "matern52", "periodic",
       "rq_alpha0.5", "rq_alpha5", "ard_1d", "sum", "product"]


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


def test_rational_quadratic_recovers_rbf_as_alpha_grows():
    """The RQ kernel is a scale mixture of RBFs; as alpha -> infinity the
    mixture collapses to a single scale and RQ -> RBF with the same s2, l.
    Checked both on the covariance and (since the same limit governs the
    optimizer) on the shared log-s2/log-l gradient blocks."""
    X = RNG.uniform(-2, 2, size=(8, 1))
    rbf = RBF(s2=1.3, l=0.9)
    for alpha, tol in [(50.0, 5e-2), (1e4, 3e-4)]:
        rq = RationalQuadratic(s2=1.3, l=0.9, alpha=alpha)
        np.testing.assert_allclose(rq(X, X), rbf(X, X), atol=tol)
        # log-s2 and log-l gradients converge to the RBF's two gradients
        for a, b in zip(rq.grads(X)[:2], rbf.grads(X)):
            np.testing.assert_allclose(a, b, atol=tol)


def test_rational_quadratic_heavier_tail_than_rbf():
    """At equal s2/l, small-alpha RQ decays slower than the RBF at long range
    (the many-lengthscale mixture keeps more correlation in the tail) -- the
    property that makes it useful for medium-term structure."""
    x0 = np.array([[0.0]])
    far = np.array([[6.0]])
    s2, l = 1.0, 1.0
    k_rbf = RBF(s2=s2, l=l)(x0, far).item()
    k_rq = RationalQuadratic(s2=s2, l=l, alpha=0.5)(x0, far).item()
    assert k_rq > k_rbf > 0


# -- ARD (per-dimension lengthscales) -------------------------------------


@pytest.mark.parametrize("fixed", [None, ["l1"]], ids=["free", "l1_fixed"])
def test_ard_gradients_match_finite_differences_multidim(fixed):
    """Per-dimension log-lengthscale gradients (and log-s2) against central
    differences on genuinely multi-dimensional input, including a frozen dim."""
    k = ARD(s2=1.3, lengthscales=[0.5, 1.7, 0.9], fixed=fixed)
    X = RNG.uniform(-2, 2, size=(8, 3))
    analytic = k.grads(X)
    numeric = finite_diff_grads(k, X)
    assert len(analytic) == len(numeric) == k.n_params
    for a, n in zip(analytic, numeric):
        np.testing.assert_allclose(a, n, rtol=1e-5, atol=1e-7)


def test_ard_equals_isotropic_rbf_when_lengthscales_equal():
    """With every l_d set to the same value, ARD is exactly the isotropic RBF
    (the ARD-weighted distance collapses to the ordinary squared distance)."""
    X = RNG.uniform(-2, 2, size=(9, 3))
    ard = ARD(s2=1.6, lengthscales=[0.8, 0.8, 0.8])
    rbf = RBF(s2=1.6, l=0.8)
    np.testing.assert_allclose(ard(X, X), rbf(X, X), atol=1e-12)


def test_ard_matrices_are_positive_semidefinite_multidim():
    k = ARD(s2=1.1, lengthscales=[0.7, 2.0])
    X = RNG.uniform(-3, 3, size=(40, 2))
    K = k(X, X)
    np.testing.assert_allclose(K, K.T, atol=1e-12)
    eigs = np.linalg.eigvalsh(K)
    assert eigs.min() > -1e-9 * eigs.max()


def test_ard_broadcasts_scalar_lengthscale_with_dim():
    """dim= with a scalar lengthscale builds D equal lengthscales; theta has
    1 + D entries and n_params counts them all."""
    k = ARD(s2=1.0, lengthscales=1.0, dim=4)
    assert k.dim_in == 4 and k.n_params == 5
    np.testing.assert_allclose(np.exp(k._theta[1:]), np.ones(4))


def test_theta_roundtrip_through_composites():
    k = Sum(RBF(), Product(Matern(nu=2.5), Periodic()))
    theta = RNG.standard_normal(k.n_params)
    k.theta = theta
    np.testing.assert_allclose(k.theta, theta)


# -- fixed (frozen) parameters --------------------------------------------


def test_fixed_param_hides_from_theta_and_grads():
    """A fixed parameter drops out of the free interface: theta, n_params, and
    grads all report only the remaining free parameters, in order."""
    free = Periodic(s2=1.2, l=0.8, p=2.3)
    frozen = Periodic(s2=1.2, l=0.8, p=2.3, fixed=["p"])
    assert free.n_params == 3 and frozen.n_params == 2
    # theta of the frozen kernel is the free kernel's theta without log p
    np.testing.assert_allclose(frozen.theta, free.theta[:2])
    X = RNG.uniform(-2, 2, size=(6, 1))
    assert len(frozen.grads(X)) == 2
    # the two reported grads are exactly the s2 and l grads of the free kernel
    for a, b in zip(frozen.grads(X), free.grads(X)[:2]):
        np.testing.assert_allclose(a, b)


def test_fixed_param_keeps_its_value_and_is_untouched_by_theta_set():
    """Setting theta only writes the free entries; the frozen period stays at
    its constructed value and the covariance is identical to a free kernel that
    happens to share those free values."""
    k = Periodic(s2=1.0, l=1.0, p=1.0, fixed=["p"])
    k.theta = np.log([3.0, 0.5])  # new s2, l -- period must stay at 1.0
    s2, l, p = np.exp(k._theta)
    np.testing.assert_allclose([s2, l, p], [3.0, 0.5, 1.0])
    ref = Periodic(s2=3.0, l=0.5, p=1.0)
    X = RNG.uniform(-2, 2, size=(6, 1))
    np.testing.assert_allclose(k(X, X), ref(X, X))


def test_fixed_param_gradients_match_finite_differences():
    """The reduced gradient list still matches central differences taken over
    the free parameters only."""
    k = Periodic(s2=1.2, l=0.8, p=2.3, fixed=["p"])
    X = RNG.uniform(-2, 2, size=(7, 1))
    analytic = k.grads(X)
    numeric = finite_diff_grads(k, X)  # iterates over n_params == free count
    assert len(analytic) == len(numeric) == k.n_params
    for a, n in zip(analytic, numeric):
        np.testing.assert_allclose(a, n, rtol=1e-5, atol=1e-7)


def test_fixed_param_in_composite_optimizes_only_free_dims():
    """Freezing a parameter inside a Sum/Product shortens the composite's free
    theta by exactly one and leaves the others addressable."""
    k = Sum(RBF(s2=1.0, l=1.0),
            Product(Periodic(s2=1.0, l=1.0, p=1.0, fixed=["p"]), RBF()))
    assert k.n_params == 2 + (2 + 2)  # periodic contributes s2, l only
    theta = RNG.standard_normal(k.n_params)
    k.theta = theta
    np.testing.assert_allclose(k.theta, theta)


def test_fixed_mask_accepts_boolean_array_and_rejects_bad_names():
    kb = RBF(s2=2.0, l=0.7, fixed=[True, False])
    assert kb.n_params == 1
    np.testing.assert_allclose(kb.theta, np.log([0.7]))
    with pytest.raises(ValueError):
        Periodic(fixed=["period"])  # not a valid param name
