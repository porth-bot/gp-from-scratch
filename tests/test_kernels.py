"""Kernels: gradient checks, positive-definiteness, composition rules."""

import numpy as np
import pytest

from gp.kernels import (
    ARD,
    RBF,
    Gibbs,
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
    Gibbs(s2=1.3, a=0.0, b=-0.4),
    Gibbs(s2=0.9, a=0.3, b=0.6),
    Sum(RBF(s2=1.0, l=0.5), Matern(nu=1.5, s2=0.5, l=2.0)),
    Product(RBF(s2=1.0, l=1.5), Periodic(s2=0.9, l=1.1, p=1.7)),
]
IDS = ["rbf", "matern12", "matern32", "matern52", "periodic",
       "rq_alpha0.5", "rq_alpha5", "ard_1d", "gibbs_tilt_down", "gibbs_tilt_up",
       "sum", "product"]


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


# ---------------------------------------------------------------------------
# Gibbs (nonstationary, input-dependent lengthscale)
# ---------------------------------------------------------------------------
def test_gibbs_recovers_rbf_exactly_when_the_lengthscale_is_constant():
    """b = 0 makes l(x) = e^a constant, and the kernel must collapse to RBF.

    The Gibbs prefactor sqrt(2 l l' / (l^2 + l'^2)) becomes 1 and the exponent
    -(d^2)/(l^2 + l'^2) becomes -d^2/(2 l^2) -- this repo's RBF convention on
    the nose. If this fails, the prefactor or the factor of 2 is wrong.
    """
    X = RNG.uniform(-3, 3, size=(25, 1))
    for s2, a in [(1.0, 0.0), (2.3, -0.4), (0.6, 1.1)]:
        gibbs = Gibbs(s2=s2, a=a, b=0.0)
        rbf = RBF(s2=s2, l=np.exp(a))
        np.testing.assert_allclose(gibbs(X, X), rbf(X, X), rtol=1e-12, atol=1e-14)


def test_gibbs_log_lengthscale_gradient_reduces_to_the_rbf_gradient_at_b_zero():
    # dK/da must equal RBF's dK/d(log l) when the lengthscale is constant --
    # the analytic cross-check quoted in the Gibbs docstring.
    X = RNG.uniform(-2, 2, size=(12, 1))
    gibbs = Gibbs(s2=1.4, a=0.2, b=0.0)
    rbf = RBF(s2=1.4, l=np.exp(0.2))
    np.testing.assert_allclose(gibbs.grads(X)[1], rbf.grads(X)[1], rtol=1e-10)


def test_gibbs_is_unit_variance_normalized_however_the_lengthscale_varies():
    # k(x, x) = s2 for every x, even where l(x) is changing fast: the prefactor
    # is exactly the normalizer that makes this true.
    X = RNG.uniform(-4, 4, size=(30, 1))
    k = Gibbs(s2=1.7, a=0.0, b=-0.9)
    np.testing.assert_allclose(np.diag(k(X, X)), 1.7, rtol=1e-12)


def test_gibbs_correlates_less_where_the_lengthscale_is_shorter():
    """The behavioral content: same separation, weaker correlation where rough.

    With b < 0 the lengthscale shrinks as x grows, so a pair of points a fixed
    distance apart should be *less* correlated out at large x than at small x.
    That is the whole reason to use this kernel.
    """
    k = Gibbs(s2=1.0, a=0.0, b=-0.8)  # l(x) = exp(-0.8 x): shrinks with x
    gap = 0.3
    corr = []
    for centre in (-2.0, 0.0, 2.0):
        X = np.array([[centre - gap / 2], [centre + gap / 2]])
        corr.append(k(X, X)[0, 1])
    assert corr[0] > corr[1] > corr[2]  # rougher (shorter l) as x increases
    assert corr[2] < 0.5 * corr[0]


def test_gibbs_rejects_multidimensional_input():
    k = Gibbs()
    with pytest.raises(ValueError):
        k(RNG.uniform(size=(5, 2)), RNG.uniform(size=(5, 2)))


def test_gibbs_ml_ii_recovers_the_tilt_and_beats_rbf_on_a_chirp():
    """On data whose roughness genuinely varies, ML-II should find it.

    sin(x^3) is a chirp: its local wavelength shrinks monotonically with x, so
    a monotone lengthscale is exactly the right shape and a stationary RBF must
    compromise. The evidence should prefer Gibbs, and the learned tilt b should
    be negative (lengthscale shrinking as x grows).
    """
    from gp.gp import GPRegressor
    from gp.optimize import adam_maximize

    rng = np.random.default_rng(0)
    X = rng.uniform(0.0, 2.2, size=(120, 1))
    y = np.sin(X[:, 0] ** 3) + 0.1 * rng.standard_normal(120)

    def fit(kernel):
        m = GPRegressor(kernel, noise_var=0.1)
        best, _ = adam_maximize(
            lambda p: m.lml_and_grad(X, y, p), m.params, lr=0.05, steps=400
        )
        m.params = best
        m.fit(X, y)
        return m

    gibbs = fit(Gibbs(s2=1.0, a=0.0, b=0.0))
    rbf = fit(RBF(s2=1.0, l=1.0))

    b = gibbs.kernel._theta[2]
    assert b < -0.2  # lengthscale shrinks with x, as the chirp demands
    assert gibbs.log_marginal_likelihood() > rbf.log_marginal_likelihood() + 5.0


# ---------------------------------------------------------------------------
# Cross-covariance gradients: dK(X1, X2)/dtheta with X1 != X2
# ---------------------------------------------------------------------------
#
# The symmetric case above is what exact ML-II needs. The sparse bound needs
# the cross case, because its hyperparameter gradient runs through k(Z, X)
# with Z and X different point sets (gp/sparse.py, Sec. 9.6). A gradient that
# is right on the diagonal block and wrong off it would pass every test above.


def finite_diff_grads_cross(kernel, X1, X2, eps=1e-6):
    """Central differences of K(X1, X2) w.r.t. each free theta_i."""
    out = []
    theta0 = kernel.theta
    for i in range(kernel.n_params):
        bumped = []
        for sign in (+1, -1):
            t = theta0.copy()
            t[i] += sign * eps
            kernel.theta = t
            bumped.append(kernel(X1, X2))
        out.append((bumped[0] - bumped[1]) / (2 * eps))
    kernel.theta = theta0
    return out


@pytest.mark.parametrize("kernel", ALL_KERNELS, ids=IDS)
def test_cross_covariance_gradients_match_finite_differences(kernel):
    X1 = RNG.uniform(-2, 2, size=(6, 1))
    X2 = RNG.uniform(-2, 2, size=(9, 1))          # different size AND positions
    analytic = kernel.grads(X1, X2)
    numeric = finite_diff_grads_cross(kernel, X1, X2)
    assert len(analytic) == kernel.n_params
    for a, n in zip(analytic, numeric):
        assert a.shape == (6, 9)
        np.testing.assert_allclose(a, n, rtol=1e-5, atol=1e-7)


def test_cross_gradients_reduce_to_the_symmetric_ones_when_the_sets_coincide():
    """grads(X) is grads(X, X): the one-argument spelling every existing caller
    uses must not have changed meaning."""
    X = RNG.uniform(-2, 2, size=(7, 1))
    for kernel in ALL_KERNELS:
        for a, b in zip(kernel.grads(X), kernel.grads(X, X)):
            np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_ard_cross_gradients_match_finite_differences_multidim():
    k = ARD(s2=1.3, lengthscales=[0.5, 1.7, 0.9])
    X1 = RNG.uniform(-2, 2, size=(5, 3))
    X2 = RNG.uniform(-2, 2, size=(8, 3))
    for a, n in zip(k.grads(X1, X2), finite_diff_grads_cross(k, X1, X2)):
        np.testing.assert_allclose(a, n, rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------------------
# Input-space derivatives: dk(x, x')/dx
# ---------------------------------------------------------------------------
#
# Parameter gradients move the kernel; these move the *points*. They exist so
# inducing locations can be optimized (gp/sparse.py), and they are the piece
# with no symmetric-case safety net -- nothing else in the repo differentiates
# a kernel with respect to its inputs.

DIFFERENTIABLE_KERNELS = [
    RBF(s2=2.0, l=0.7),
    Matern(nu=1.5, s2=0.8, l=1.3),
    Matern(nu=2.5, s2=1.1, l=0.5),
    Periodic(s2=1.2, l=0.8, p=2.3),
    RationalQuadratic(s2=1.3, l=0.9, alpha=0.5),
    RationalQuadratic(s2=0.7, l=1.4, alpha=5.0),
    Sum(RBF(s2=1.0, l=0.5), Matern(nu=1.5, s2=0.5, l=2.0)),
    Product(RBF(s2=1.0, l=1.5), Periodic(s2=0.9, l=1.1, p=1.7)),
]
DIFF_IDS = ["rbf", "matern32", "matern52", "periodic", "rq_alpha0.5",
            "rq_alpha5", "sum", "product"]


def finite_diff_dK_dX1(kernel, X1, X2, eps=1e-6):
    """Central differences of K(X1, X2) w.r.t. each coordinate of X1."""
    n1, d = X1.shape
    out = np.zeros((n1, X2.shape[0], d))
    for i in range(n1):
        for a in range(d):
            bumped = []
            for sign in (+1, -1):
                Xp = X1.copy()
                Xp[i, a] += sign * eps
                bumped.append(kernel(Xp, X2))
            # only row i moves, so only row i of the difference is meaningful
            out[i, :, a] = ((bumped[0] - bumped[1]) / (2 * eps))[i]
    return out


@pytest.mark.parametrize("kernel", DIFFERENTIABLE_KERNELS, ids=DIFF_IDS)
def test_input_derivatives_match_finite_differences_1d(kernel):
    X1 = RNG.uniform(-2, 2, size=(5, 1))
    X2 = RNG.uniform(-2, 2, size=(7, 1))
    analytic = kernel.dK_dX1(X1, X2)
    assert analytic.shape == (5, 7, 1)
    np.testing.assert_allclose(
        analytic, finite_diff_dK_dX1(kernel, X1, X2), rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize(
    "kernel",
    [RBF(s2=1.4, l=0.9), Matern(nu=2.5, s2=0.9, l=1.2),
     RationalQuadratic(s2=1.1, l=0.8, alpha=1.5),
     ARD(s2=1.3, lengthscales=[0.5, 1.7, 0.9])],
    ids=["rbf", "matern52", "rq", "ard"],
)
def test_input_derivatives_match_finite_differences_3d(kernel):
    """Multi-dimensional inputs, where a mis-broadcast lengthscale or a swapped
    axis is easy to write and invisible in 1D."""
    X1 = RNG.uniform(-2, 2, size=(4, 3))
    X2 = RNG.uniform(-2, 2, size=(6, 3))
    analytic = kernel.dK_dX1(X1, X2)
    assert analytic.shape == (4, 6, 3)
    np.testing.assert_allclose(
        analytic, finite_diff_dK_dX1(kernel, X1, X2), rtol=1e-5, atol=1e-7
    )


def test_input_derivative_vanishes_at_coincident_points_for_stationary_kernels():
    """k(x, x) = k(0) is constant, so the derivative on the diagonal is zero --
    the fact that lets the Kuu diagonal drop out of the Z-gradient rather than
    having to be special-cased (Sec. 9.6)."""
    X = RNG.uniform(-2, 2, size=(6, 2))
    for kernel in (RBF(s2=1.2, l=0.8), Matern(nu=1.5, s2=1.0, l=1.1),
                   ARD(s2=1.0, lengthscales=[0.7, 1.3])):
        g = kernel.dK_dX1(X, X)
        np.testing.assert_allclose(np.einsum("iia->ia", g), 0.0, atol=1e-14)


def test_stationary_input_derivative_is_antisymmetric_in_its_arguments():
    """For k(x, x') = f(x - x'), dk/dx = -dk/dx'. Checked by swapping the two
    argument sets: dK_dX1(X2, X1) transposed must be the negative."""
    X1 = RNG.uniform(-2, 2, size=(5, 2))
    X2 = RNG.uniform(-2, 2, size=(4, 2))
    for kernel in (RBF(s2=1.3, l=0.9), RationalQuadratic(s2=1.0, l=1.1, alpha=2.0)):
        a = kernel.dK_dX1(X1, X2)                   # (5, 4, 2)
        b = kernel.dK_dX1(X2, X1).transpose(1, 0, 2)  # (5, 4, 2)
        np.testing.assert_allclose(a, -b, rtol=1e-12, atol=1e-14)


def test_non_differentiable_kernels_refuse_rather_than_approximate():
    """Matern nu=0.5 has a genuine cusp at r=0 and Gibbs simply has not been
    derived. Both must raise: returning a large finite number for the OU cusp
    would hand an optimizer a silently wrong direction."""
    X1 = RNG.uniform(-2, 2, size=(4, 1))
    X2 = RNG.uniform(-2, 2, size=(3, 1))
    for kernel in (Matern(nu=0.5, s2=1.0, l=1.0), Gibbs(s2=1.0, a=0.0, b=0.3)):
        with pytest.raises(NotImplementedError):
            kernel.dK_dX1(X1, X2)
    # and the refusal propagates through composition rather than being swallowed
    with pytest.raises(NotImplementedError):
        Sum(RBF(), Matern(nu=0.5)).dK_dX1(X1, X2)
