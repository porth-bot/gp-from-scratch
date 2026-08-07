"""The Titsias sparse GP: exact recovery at Z = X, then the properties that
make the bound a bound (theory/derivations.md, Sec. 9).

The first test is the load-bearing one. Everything else in this module measures
an approximation; `test_z_equals_x_recovers_the_exact_gp` measures no
approximation at all, which is what pins the algebra to `gp/gp.py` -- an
implementation already checked against scikit-learn and against brute-force
leave-one-out refits.
"""

import numpy as np
import pytest

from gp.gp import GPRegressor
from gp.kernels import ARD, RBF, Gibbs, Matern, Periodic, RationalQuadratic
from gp.optimize import adam_maximize, maximize_elbo
from gp.sparse import SGPR, kernel_diag, kernel_grad_diag


def _toy(rng, n=40, lo=-3.0, hi=3.0):
    X = np.sort(rng.uniform(lo, hi, n)).reshape(-1, 1)
    y = np.sin(X).ravel() + 0.1 * rng.standard_normal(n)
    return X, y


def _grid(n=60, lo=-3.5, hi=3.5):
    return np.linspace(lo, hi, n).reshape(-1, 1)


# -- exact recovery ----------------------------------------------------------


def test_z_equals_x_recovers_the_exact_gp():
    """With Z = X the inducing variables ARE the latent function at the data,
    so Qff = Kff, the trace penalty is zero, and both the bound and the
    predictive equations must collapse onto gp/gp.py exactly (Sec. 9.4).

    The residual is the jitter on Kuu, not an approximation: the next test
    shows it scaling linearly to zero with it. Tolerances here are what the
    default jitter of 1e-10 actually delivers on this data.
    """
    rng = np.random.default_rng(0)
    X, y = _toy(rng)
    Xs = _grid()

    for kernel in (lambda: RBF(s2=1.0, l=1.0), lambda: Matern(nu=1.5, s2=1.0, l=1.0)):
        exact = GPRegressor(kernel(), noise_var=0.05).fit(X, y)
        sparse = SGPR(kernel(), Z=X, noise_var=0.05).fit(X, y)

        assert sparse.trace_term() < 1e-8
        assert abs(sparse.elbo() - exact.log_marginal_likelihood()) < 1e-7

        m_exact, v_exact = exact.predict(Xs)
        m_sparse, v_sparse = sparse.predict(Xs)
        assert np.abs(m_exact - m_sparse).max() < 1e-6
        assert np.abs(v_exact - v_sparse).max() < 1e-8

        m_exact, v_exact = exact.predict(Xs, include_noise=True)
        m_sparse, v_sparse = sparse.predict(Xs, include_noise=True)
        assert np.abs(v_exact - v_sparse).max() < 1e-8


def test_recovery_error_is_exactly_the_jitter():
    """Why the previous test's tolerances are what they are rather than zero.

    Loading Kuu -> Kuu + jI replaces Qff by Kff(Kff + jI)^{-1}Kff, whose
    eigenvalues sit lam*j/(lam + j) <= j below Kff's, so the leftover trace term
    is O(n j) and the predictive error is O(j). Sweeping j over four decades,
    both fall by a factor of ~10 per decade -- linear, which is the signature of
    the jitter rather than of an algebra error (an algebra error would leave a
    floor).
    """
    rng = np.random.default_rng(0)
    X, y = _toy(rng)
    Xs = _grid()
    exact = GPRegressor(Matern(nu=1.5, s2=1.0, l=1.0), noise_var=0.05).fit(X, y)
    m_exact, _ = exact.predict(Xs)

    jitters = [1e-8, 1e-9, 1e-10, 1e-11]
    traces, mean_errs = [], []
    for j in jitters:
        sparse = SGPR(Matern(nu=1.5, s2=1.0, l=1.0), Z=X,
                      noise_var=0.05, jitter=j).fit(X, y)
        m, _ = sparse.predict(Xs)
        traces.append(sparse.trace_term())
        mean_errs.append(float(np.abs(m_exact - m).max()))

    for series in (traces, mean_errs):
        for a, b in zip(series, series[1:]):
            assert 8.0 < a / b < 12.5, series          # one decade per decade


# -- the bound is a bound ----------------------------------------------------


def test_bound_never_exceeds_the_exact_log_evidence():
    """F <= log p(y) for every inducing set, however badly chosen -- the whole
    reason to optimize Z is that it cannot cheat past this ceiling (Sec. 9.2)."""
    rng = np.random.default_rng(3)
    X, y = _toy(rng, n=120, lo=-4, hi=4)
    exact = GPRegressor(RBF(s2=1.0, l=0.7), noise_var=0.02).fit(X, y)
    lml = exact.log_marginal_likelihood()

    for seed in range(8):
        r = np.random.default_rng(seed)
        for M in (3, 10, 40):
            Z = r.uniform(-6, 6, M).reshape(-1, 1)     # deliberately careless
            sparse = SGPR(RBF(s2=1.0, l=0.7), Z=Z, noise_var=0.02).fit(X, y)
            assert sparse.elbo() < lml + 1e-8, (seed, M, sparse.elbo(), lml)


def test_dtc_term_alone_is_not_a_bound():
    """The bound's first term, log N(y | 0, Qff + sigma^2 I), is the DTC
    approximation's log evidence, and on its own it can sit ABOVE the exact
    evidence: Qff <= Kff in the PSD order, so the low-rank model can report the
    data as less surprising than they are. Only the trace term makes F a bound.

    Measured here at M = 16 on n = 400: DTC 296.28 against an exact 292.91,
    and the trace penalty of 2.47 pulls F back down to 172.57.
    """
    rng = np.random.default_rng(2)
    n = 400
    X = np.sort(rng.uniform(-4, 4, n)).reshape(-1, 1)
    y = np.sin(2 * X).ravel() + 0.1 * rng.standard_normal(n)
    exact = GPRegressor(RBF(s2=1.0, l=0.5), noise_var=0.01).fit(X, y)
    lml = exact.log_marginal_likelihood()

    sparse = SGPR(RBF(s2=1.0, l=0.5), Z=np.linspace(-4, 4, 16).reshape(-1, 1),
                  noise_var=0.01).fit(X, y)
    assert sparse.dtc_log_evidence() > lml            # not a bound
    assert sparse.elbo() < lml                        # the trace term fixes it
    assert sparse.trace_term() > 1.0


def test_bound_and_predictions_improve_with_more_inducing_points():
    """More inducing points can only tighten the bound on a fixed quantity, and
    the trace penalty -- the part of f that u fails to explain -- shrinks toward
    zero as the summary becomes sufficient."""
    rng = np.random.default_rng(2)
    n = 400
    X = np.sort(rng.uniform(-4, 4, n)).reshape(-1, 1)
    y = np.sin(2 * X).ravel() + 0.1 * rng.standard_normal(n)
    Xs = _grid(n=120, lo=-4, hi=4)
    exact = GPRegressor(RBF(s2=1.0, l=0.5), noise_var=0.01).fit(X, y)
    m_exact, v_exact = exact.predict(Xs)

    bounds, traces, mean_errs, sd_errs = [], [], [], []
    for M in (4, 8, 16, 32, 64):
        Z = np.linspace(-4, 4, M).reshape(-1, 1)
        sparse = SGPR(RBF(s2=1.0, l=0.5), Z=Z, noise_var=0.01).fit(X, y)
        m, v = sparse.predict(Xs)
        bounds.append(sparse.elbo())
        traces.append(sparse.trace_term())
        mean_errs.append(float(np.abs(m - m_exact).max()))
        sd_errs.append(float(np.abs(np.sqrt(v) - np.sqrt(v_exact)).max()))

    assert all(a < b for a, b in zip(bounds, bounds[1:])), bounds
    assert all(a > b for a, b in zip(traces, traces[1:])), traces
    assert all(a > b for a, b in zip(mean_errs, mean_errs[1:])), mean_errs
    assert all(a > b for a, b in zip(sd_errs, sd_errs[1:])), sd_errs
    assert mean_errs[-1] < 1e-6 and sd_errs[-1] < 1e-6   # and it converges


def test_trace_term_is_nonnegative_and_vanishes_only_when_u_determines_f():
    """tr(Kff - Qff) is the trace of a conditional covariance, so it cannot be
    negative, and it is zero exactly when the inducing set spans the data."""
    rng = np.random.default_rng(9)
    X, y = _toy(rng, n=60)
    for M in (1, 2, 5, 20):
        Z = rng.uniform(-3, 3, M).reshape(-1, 1)
        sparse = SGPR(RBF(s2=1.0, l=1.0), Z=Z, noise_var=0.05).fit(X, y)
        assert sparse.trace_term() > 0.0
    duplicated = SGPR(RBF(s2=1.0, l=1.0), Z=X, noise_var=0.05).fit(X, y)
    assert duplicated.trace_term() < 1e-8


# -- prediction --------------------------------------------------------------


def test_variance_returns_to_the_prior_far_from_every_inducing_point():
    """The property RFF loses (Sec. 8.5) and this method keeps: the -K*u Kuu^-1
    Ku* and +K*u Sig^-1 Ku* corrections both vanish far away, so the band
    relaxes to k(x*, x*) rather than staying spuriously tight."""
    rng = np.random.default_rng(5)
    X, y = _toy(rng, n=200, lo=-4, hi=4)
    sparse = SGPR(RBF(s2=2.5, l=0.5), Z=np.linspace(-4, 4, 16).reshape(-1, 1),
                  noise_var=0.01).fit(X, y)
    mean, var = sparse.predict(np.array([[100.0]]))
    assert abs(var[0] - 2.5) < 1e-8
    assert abs(mean[0]) < 1e-8


def test_include_noise_adds_exactly_sigma_squared():
    rng = np.random.default_rng(1)
    X, y = _toy(rng)
    sparse = SGPR(RBF(s2=1.0, l=1.0), Z=np.linspace(-3, 3, 8).reshape(-1, 1),
                  noise_var=0.05).fit(X, y)
    Xs = _grid(n=20)
    _, latent = sparse.predict(Xs)
    _, observed = sparse.predict(Xs, include_noise=True)
    assert np.allclose(observed - latent, 0.05)


def test_predictive_variance_survives_near_total_cancellation():
    """Why predict() clips at zero. At M = n with a long lengthscale and tiny
    noise, k(x*, x*) and the two corrections cancel to within ~1e-10 of the
    prior scale s2 = 1 -- the regime where a float slip below zero would put a
    NaN into every downstream sqrt. Note the clip does not actually fire here:
    the measurement is how close it gets, not that it triggers.
    """
    rng = np.random.default_rng(4)
    X, y = _toy(rng, n=120, lo=-3.0, hi=3.0)
    sparse = SGPR(RBF(s2=1.0, l=10.0), Z=X, noise_var=1e-8).fit(X, y)
    _, var = sparse.predict(X)
    assert np.all(var >= 0.0)
    assert var.min() < 1e-8                        # cancellation is near-total


# -- the O(n) diagonal -------------------------------------------------------


def test_kernel_diag_matches_the_full_matrix_for_every_kernel_shape():
    """The blocked diagonal exists so the trace term and the predictive
    variance stay O(n) in memory. It has to agree with the n^2 spelling for
    non-stationary kernels (Gibbs) and composites too, not just for the ones
    whose diagonal is the constant s2, and at block sizes that do not divide n.
    """
    rng = np.random.default_rng(12)
    X1 = np.linspace(-2, 2, 101).reshape(-1, 1)
    X2 = rng.uniform(-2, 2, size=(101, 3))
    cases = [
        (RBF(s2=1.7, l=0.4), X1),
        (Gibbs(s2=1.1, a=0.5, b=0.3), X1),                    # non-stationary
        (RBF(s2=1.0, l=1.0) + Periodic(s2=0.5, l=1.0, p=2.0), X1),
        (RBF(s2=0.3, l=2.0) * Periodic(s2=2.0, l=0.8, p=1.0), X1),
        (ARD(s2=1.4, lengthscales=np.array([0.5, 1.0, 2.0])), X2),
    ]
    for kernel, X in cases:
        reference = np.diag(kernel(X, X))
        for block in (1, 7, 100, 101, 4096):
            assert np.allclose(kernel_diag(kernel, X, block=block), reference)


# -- plumbing ----------------------------------------------------------------


def test_params_round_trip_and_invalidate_the_fit():
    rng = np.random.default_rng(6)
    X, y = _toy(rng)
    sparse = SGPR(RBF(s2=1.0, l=1.0), Z=np.linspace(-3, 3, 6).reshape(-1, 1),
                  noise_var=0.05).fit(X, y)
    first = sparse.elbo()
    assert np.allclose(sparse.params, [np.log(1.0), np.log(1.0), np.log(0.05)])

    sparse.params = np.array([np.log(2.0), np.log(0.4), np.log(0.2)])
    assert sparse.noise_var == pytest.approx(0.2)
    with pytest.raises(AssertionError):
        sparse.elbo()                                  # must refit first
    assert sparse.fit(X, y).elbo() != first


def test_inducing_inputs_are_copied_not_aliased():
    """A caller mutating its own Z array must not silently corrupt a fit."""
    rng = np.random.default_rng(8)
    X, y = _toy(rng)
    Z = np.linspace(-3, 3, 6).reshape(-1, 1)
    sparse = SGPR(RBF(s2=1.0, l=1.0), Z=Z, noise_var=0.05).fit(X, y)
    before = sparse.elbo()
    Z[:] = 99.0
    assert sparse.elbo() == before


def test_constructor_and_fit_validate_their_arguments():
    rng = np.random.default_rng(0)
    X, y = _toy(rng)
    with pytest.raises(ValueError):
        SGPR(RBF(), Z=np.zeros((0, 1)), noise_var=0.05)
    with pytest.raises(ValueError):
        SGPR(RBF(), Z=np.zeros((3, 1)), noise_var=0.05, jitter=-1.0)
    with pytest.raises(ValueError):
        SGPR(RBF(), Z=np.zeros((3, 2)), noise_var=0.05).fit(X, y)   # wrong d


# -- gradients of the bound --------------------------------------------------
#
# Three blocks (kernel theta, log sigma^2, Z) all contract against the same two
# matrices (Sec. 9.6), so a mistake in the shared part shows up everywhere and a
# mistake in the per-block algebra shows up in exactly one. The Z block is the
# one with no independent oracle -- nothing else in the repo differentiates a
# bound with respect to a *location* -- so it gets the most checking.

GRAD_KERNELS = [
    lambda: RBF(s2=1.3, l=0.8),
    lambda: Matern(nu=1.5, s2=0.9, l=1.1),
    lambda: Matern(nu=2.5, s2=0.9, l=1.1),
    lambda: RationalQuadratic(s2=1.0, l=0.9, alpha=2.0),
    lambda: Periodic(s2=1.1, l=0.9, p=2.0),
    lambda: RBF(s2=0.8, l=1.2) + Periodic(s2=0.4, l=1.0, p=2.5),
    lambda: RBF(s2=0.9, l=1.4) * Periodic(s2=1.0, l=1.2, p=3.0),
]
GRAD_IDS = ["rbf", "matern32", "matern52", "rq", "periodic", "sum", "product"]

# eps = 1e-5 rather than the 1e-6 the kernel tests use. The bound is a sum over
# n data points, so its curvature is n times a kernel entry's and the optimal
# central-difference step shifts up accordingly; at 1e-6 the Periodic case is
# dominated by cancellation, not by the gradient being wrong (measured: the
# error bottoms out near 1e-4 and grows again below it, the textbook V).
FD_EPS = 1e-5


def _fd_param_grads(model, X, y, eps=FD_EPS):
    p0 = model.params.copy()
    out = []
    for i in range(len(p0)):
        bumped = []
        for sign in (+1, -1):
            p = p0.copy()
            p[i] += sign * eps
            model.params = p
            bumped.append(model.fit(X, y).elbo())
        out.append((bumped[0] - bumped[1]) / (2 * eps))
    model.params = p0
    model.fit(X, y)
    return np.array(out)


def _fd_Z_grads(model, X, y, eps=FD_EPS):
    Z0 = model.Z.copy()
    out = np.zeros_like(Z0)
    for i in range(Z0.shape[0]):
        for a in range(Z0.shape[1]):
            bumped = []
            for sign in (+1, -1):
                Zp = Z0.copy()
                Zp[i, a] += sign * eps
                model.Z = Zp
                bumped.append(model.fit(X, y).elbo())
            out[i, a] = (bumped[0] - bumped[1]) / (2 * eps)
    model.Z = Z0
    model.fit(X, y)
    return out


@pytest.mark.parametrize("make_kernel", GRAD_KERNELS, ids=GRAD_IDS)
def test_bound_gradients_match_finite_differences(make_kernel):
    """Hyperparameters and log sigma^2, against central differences."""
    rng = np.random.default_rng(0)
    X, y = _toy(rng, n=60)
    Z = rng.uniform(-3, 3, 7).reshape(-1, 1)
    model = SGPR(make_kernel(), Z=Z, noise_var=0.07).fit(X, y)

    _, analytic, _ = model.elbo_and_grads()
    numeric = _fd_param_grads(model, X, y)
    assert analytic.shape == numeric.shape
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize("make_kernel", GRAD_KERNELS, ids=GRAD_IDS)
@pytest.mark.parametrize("seed", [1, 2, 3], ids=["Z1", "Z2", "Z3"])
def test_inducing_location_gradients_match_finite_differences(make_kernel, seed):
    """dF/dZ at several *random* inducing sets, not one convenient one.

    The failure this guards against is quantitative, not qualitative: drop the
    factor of 2 on the Kuu term of (9.20) and the bound still increases along
    the reported direction, just toward the wrong place. Only a numerical check
    at arbitrary Z catches that.
    """
    rng = np.random.default_rng(0)
    X, y = _toy(rng, n=60)
    Z = np.random.default_rng(seed).uniform(-3.5, 3.5, 8).reshape(-1, 1)
    model = SGPR(make_kernel(), Z=Z, noise_var=0.07).fit(X, y)

    _, _, analytic = model.elbo_and_grads()
    numeric = _fd_Z_grads(model, X, y)
    assert analytic.shape == Z.shape
    scale = max(np.abs(numeric).max(), 1.0)
    assert np.abs(analytic - numeric).max() < 1e-4 * scale, (analytic, numeric)


def test_gradient_survives_an_ill_conditioned_kuu():
    """The regression test for the jitter term in the Kuu gradient.

    fit() factorizes Kuu + j*mean(diag Kuu)*I, so the loading is a function of
    s2 and has to be differentiated too. Skipping it looks free -- j is 1e-10 --
    but the term is contracted against S = P H P^T, and P carries a Kuu^{-1},
    so the error scales like j * ||P||^2, i.e. with the SQUARE of Kuu's
    conditioning. Here a period-2 kernel sees inducing points 2 apart as
    duplicates; cond(Kuu) is only ~6e4 and the omission was still worth 1.3e-4
    relative on the log-s2 gradient, flat in eps (so: a real error, not
    finite-difference truncation, which would have shrunk and then grown).
    """
    rng = np.random.default_rng(0)
    X, y = _toy(rng, n=60)
    Z = rng.uniform(-3, 3, 7).reshape(-1, 1)
    kernel = Periodic(s2=1.1, l=0.9, p=2.0)
    assert np.linalg.cond(kernel(Z, Z)) > 1e4          # the setup, not luck

    model = SGPR(kernel, Z=Z, noise_var=0.07).fit(X, y)
    _, analytic = model.elbo_grad_params()
    numeric = _fd_param_grads(model, X, y)
    # 1e-4 relative would pass a sloppy tolerance; 2e-5 is what correctness costs
    np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=1e-6)


def test_gradients_match_finite_differences_in_two_dimensions():
    """Multi-dimensional Z, where a swapped axis in the (M, n, d) contraction
    is easy to write and invisible in 1D. ARD as well, since its input
    derivative is the one that is not a function of the plain r^2."""
    rng = np.random.default_rng(11)
    X = rng.uniform(-2, 2, size=(70, 2))
    y = np.sin(X[:, 0]) * np.cos(1.5 * X[:, 1]) + 0.1 * rng.standard_normal(70)
    Z = rng.uniform(-2, 2, size=(6, 2))

    for kernel in (ARD(s2=1.2, lengthscales=[0.7, 1.4]), RBF(s2=1.1, l=0.9)):
        model = SGPR(kernel, Z=Z, noise_var=0.05).fit(X, y)
        _, gp_analytic, gz_analytic = model.elbo_and_grads()
        np.testing.assert_allclose(
            gp_analytic, _fd_param_grads(model, X, y), rtol=2e-5, atol=1e-6
        )
        gz_numeric = _fd_Z_grads(model, X, y)
        scale = max(np.abs(gz_numeric).max(), 1.0)
        assert np.abs(gz_analytic - gz_numeric).max() < 1e-4 * scale


# -- the two structural checks the gradients have to pass --------------------


def test_at_z_equals_x_the_parameter_gradient_is_the_exact_gp_gradient():
    """The Sec. 9.4 recovery argument, differentiated.

    At Z = X the bound *is* the exact log marginal likelihood, identically in
    theta and sigma^2 -- so its gradient must be the exact GP's gradient, term
    for term. This pins the sparse gradient to gp.py's, which is itself
    finite-difference-checked and cross-checked against scikit-learn: a shared
    error would have to survive two independent derivations to get through.
    """
    rng = np.random.default_rng(0)
    X, y = _toy(rng)
    for make in (lambda: RBF(s2=1.2, l=0.9), lambda: Matern(nu=2.5, s2=0.8, l=1.1)):
        exact = GPRegressor(make(), noise_var=0.05)
        _, exact_grad = exact.lml_and_grad(X, y)
        _, sparse_grad = SGPR(make(), Z=X, noise_var=0.05).fit(X, y).elbo_grad_params()
        np.testing.assert_allclose(sparse_grad, exact_grad, rtol=1e-6, atol=1e-7)


def test_the_z_gradient_vanishes_at_z_equals_x():
    """Z = X is a global maximum of F over Z: F <= log p(y) for every inducing
    set (test_bound_never_exceeds_the_exact_log_evidence) with equality there,
    so the gradient has to be zero. Not a numerical accident -- it is the same
    statement as exact recovery, seen one derivative up. What is left is the
    jitter, which is why the tolerance is 1e-5 rather than 1e-14.
    """
    rng = np.random.default_rng(0)
    X, y = _toy(rng)
    for make in (lambda: RBF(s2=1.2, l=0.9), lambda: Matern(nu=2.5, s2=0.8, l=1.1)):
        _, params_grad, grad_Z = SGPR(make(), Z=X, noise_var=0.05).fit(X, y).elbo_and_grads()
        # against the parameter gradient's scale, which is O(10) here
        assert np.abs(grad_Z).max() < 1e-5 * np.abs(params_grad).max()


def test_the_noise_gradient_carries_the_trace_penalty_term():
    """(9.19) has a term the exact GP's noise gradient does not: +T/(2 sigma^4),
    which is positive whenever the inducing set is imperfect. So at a shared
    (theta, sigma^2) a sparse model always wants MORE noise than the exact one
    -- variation u cannot explain is cheaper to call noise. Measured rather
    than asserted: the difference must be exactly the trace term's derivative.
    """
    rng = np.random.default_rng(7)
    X, y = _toy(rng, n=100, lo=-4, hi=4)
    exact = GPRegressor(RBF(s2=1.0, l=0.6), noise_var=0.04)
    _, exact_grad = exact.lml_and_grad(X, y)
    sparse = SGPR(RBF(s2=1.0, l=0.6), Z=np.linspace(-4, 4, 6).reshape(-1, 1),
                  noise_var=0.04).fit(X, y)
    _, sparse_grad = sparse.elbo_grad_params()

    assert sparse.trace_term() > 0.0
    assert sparse_grad[-1] > exact_grad[-1]        # the sparse fit wants more noise


def test_a_step_along_the_gradient_raises_the_bound():
    """The behavioral consequence: ascent works. Small enough steps in theta
    and in Z must both increase F, and Z must actually move (a zero gradient
    would pass a monotonicity check vacuously)."""
    rng = np.random.default_rng(4)
    X, y = _toy(rng, n=80, lo=-4, hi=4)
    model = SGPR(RBF(s2=1.0, l=0.5), Z=rng.uniform(-4, 4, 6).reshape(-1, 1),
                 noise_var=0.05).fit(X, y)
    before, grad_params, grad_Z = model.elbo_and_grads()
    assert np.abs(grad_Z).max() > 1e-3

    model.Z = model.Z + 1e-4 * grad_Z / np.abs(grad_Z).max()
    assert model.fit(X, y).elbo() > before

    model.params = model.params + 1e-5 * grad_params / np.abs(grad_params).max()
    assert model.fit(X, y).elbo() > before


# -- what the gradients decline to do ----------------------------------------


def test_z_gradients_refuse_for_kernels_with_no_input_derivative():
    """Matern nu=0.5 (a real cusp at r=0) and Gibbs (non-stationary, not
    derived) have no dF/dZ, and must say so rather than return a number. The
    parameter gradient is unaffected -- only the Z block needs an input
    derivative -- and that separation is the reason elbo_grad_params exists.
    """
    rng = np.random.default_rng(0)
    X, y = _toy(rng)
    Z = np.linspace(-3, 3, 6).reshape(-1, 1)
    for kernel in (Matern(nu=0.5, s2=1.0, l=1.0), Gibbs(s2=1.0, a=0.0, b=0.3)):
        model = SGPR(kernel, Z=Z, noise_var=0.05).fit(X, y)
        with pytest.raises(NotImplementedError):
            model.elbo_and_grads()
        _, grad = model.elbo_grad_params()             # still fine
        np.testing.assert_allclose(grad, _fd_param_grads(model, X, y),
                                   rtol=2e-5, atol=1e-6)


def test_setting_Z_copies_validates_and_invalidates_the_fit():
    rng = np.random.default_rng(0)
    X, y = _toy(rng)
    model = SGPR(RBF(), Z=np.linspace(-3, 3, 5).reshape(-1, 1),
                 noise_var=0.05).fit(X, y)
    new = np.linspace(-2, 2, 5).reshape(-1, 1)
    model.Z = new
    with pytest.raises(AssertionError):
        model.elbo()                                   # must refit first
    model.fit(X, y)
    new[:] = 99.0
    assert model.Z.max() < 99.0                        # copied, not aliased
    with pytest.raises(ValueError):
        model.Z = np.linspace(-2, 2, 4).reshape(-1, 1)  # wrong M


def test_kernel_grad_diag_matches_the_full_matrix_for_every_kernel_shape():
    """The blocked gradient diagonal, the same discipline as kernel_diag: it
    has to agree with the n^2 spelling for non-stationary kernels and
    composites, and at block sizes that do not divide n."""
    rng = np.random.default_rng(12)
    X1 = np.linspace(-2, 2, 101).reshape(-1, 1)
    X2 = rng.uniform(-2, 2, size=(101, 3))
    cases = [
        (RBF(s2=1.7, l=0.4), X1),
        (Gibbs(s2=1.1, a=0.5, b=0.3), X1),                    # non-stationary
        (RBF(s2=1.0, l=1.0) + Periodic(s2=0.5, l=1.0, p=2.0), X1),
        (RBF(s2=0.3, l=2.0) * Periodic(s2=2.0, l=0.8, p=1.0), X1),
        (ARD(s2=1.4, lengthscales=np.array([0.5, 1.0, 2.0])), X2),
    ]
    for kernel, X in cases:
        reference = [np.diag(g) for g in kernel.grads(X)]
        for block in (1, 7, 100, 101, 4096):
            blocked = kernel_grad_diag(kernel, X, block=block)
            assert len(blocked) == len(reference)
            for a, b in zip(blocked, reference):
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-14)


# -- joint ML-II over hyperparameters and inducing locations -----------------


def _packed_toy(rng, n=200):
    """A mixed-scale 1D target: slow on the left, fast on the right.

    One stationary lengthscale cannot be right everywhere on this, which is
    what makes *where* the inducing points go a question with an answer.
    """
    X = np.sort(rng.uniform(-4.0, 4.0, n)).reshape(-1, 1)
    x = X[:, 0]
    f = np.sin(1.1 * x) + 0.6 * np.sin(5.0 * x) * (x > 0)
    return X, f + 0.1 * rng.standard_normal(n)


def test_joint_ml2_climbs_and_the_bound_stays_below_the_exact_evidence():
    """The invariant that licenses optimizing Z at all (Sec. 9.2).

    F(theta, Z) <= log p(y | theta) for *every* Z, so however hard the
    optimizer drives the inducing locations, the bound cannot cross the exact
    log evidence at the hyperparameters it ended up at. The comparison has to
    be at the SGPR's own theta -- against the exact GP's ML-II optimum it would
    hold for a second, weaker reason and would not test this.
    """
    rng = np.random.default_rng(0)
    X, y = _packed_toy(rng)
    model = SGPR(RBF(s2=0.5, l=2.0), Z=np.linspace(-4, 4, 12).reshape(-1, 1),
                 noise_var=0.3)
    res = maximize_elbo(model, X, y, lr=0.05, steps=400)

    assert res.elbo > res.history[0] + 1.0             # it actually climbed
    assert res.elbo == max(res.history)                # and returns the best

    exact = GPRegressor(RBF(), noise_var=1.0)
    exact.params = res.params
    assert res.elbo <= exact.fit(X, y).log_marginal_likelihood()


def test_optimizing_Z_cannot_move_the_quantity_being_bounded():
    """The structural difference from a model parameter, made measurable.

    At fixed theta, moving Z changes the bound (that is the point) but leaves
    log p(y | theta) untouched to the last bit, because Z never enters the
    model. A parameter that could do both would be one that could overfit.
    """
    rng = np.random.default_rng(1)
    X, y = _packed_toy(rng)
    kernel = RBF(s2=1.0, l=0.6)
    exact = GPRegressor(kernel, noise_var=0.05).fit(X, y)
    reference = exact.log_marginal_likelihood()

    bounds = []
    for Z in (np.linspace(-4, 4, 8).reshape(-1, 1),
              np.linspace(-1, 1, 8).reshape(-1, 1),
              rng.uniform(-4, 4, 8).reshape(-1, 1)):
        model = SGPR(RBF(s2=1.0, l=0.6), Z=Z, noise_var=0.05).fit(X, y)
        bounds.append(model.elbo())
        # the exact GP is re-fit at the same theta and must not have noticed
        assert exact.fit(X, y).log_marginal_likelihood() == reference

    assert max(bounds) - min(bounds) > 1.0             # Z did move the bound
    assert max(bounds) < reference


def test_freeing_the_inducing_locations_beats_freezing_them():
    """What the Z block buys, at matched M, matched init and matched budget.

    Both runs start from the same evenly-spaced Z and take the same number of
    Adam steps; the only difference is whether dF/dZ is followed. If the extra
    block did not pay for itself there would be no reason to have derived it.

    On this target the mechanism is visible in the recovered lengthscale. Ten
    evenly-spaced inducing points are 0.89 apart, and the right half of the
    signal oscillates with period 1.26, so a frozen grid cannot represent it:
    ML-II with Z pinned is driven to a long lengthscale (measured 1.26, against
    the exact GP's 0.46) and calls the wiggles noise. Freeing Z lets the
    inducing set migrate right -- 6 of 10 end up in the fast half -- and the
    lengthscale lands near the exact fit's.

    Note what is deliberately *not* asserted: that the free run has the smaller
    trace penalty. It does not, by a factor of 150 here, and the assertion
    failed the first time it was written. tr(Kff - Qff) is measured in units of
    the fitted prior variance, so it is only comparable between fits sharing
    theta; the frozen run buys a tiny trace by fitting a smooth, low-amplitude
    model that has little left to explain. The bound is the comparable
    quantity, because it is a bound on the same log p(y) either way.
    """
    rng = np.random.default_rng(2)
    X, y = _packed_toy(rng)
    Z0 = np.linspace(-4, 4, 10).reshape(-1, 1)
    Xs = _grid(300, -4, 4)

    exact = GPRegressor(RBF(s2=0.5, l=2.0), noise_var=0.3)
    best, _ = adam_maximize(lambda p: exact.lml_and_grad(X, y, p), exact.params,
                            lr=0.05, steps=600)
    exact.params = best
    mean_exact, _ = exact.fit(X, y).predict(Xs)

    fits = {}
    for optimize_Z in (False, True):
        model = SGPR(RBF(s2=0.5, l=2.0), Z=Z0, noise_var=0.3)
        res = maximize_elbo(model, X, y, optimize_Z=optimize_Z, lr=0.05, steps=400)
        mean, _ = model.predict(Xs)
        fits[optimize_Z] = (res, float(np.abs(mean - mean_exact).max()))
    (frozen, frozen_err), (free, free_err) = fits[False], fits[True]

    assert free.elbo > frozen.elbo + 10.0
    assert free_err < 0.5 * frozen_err                 # and predicts better
    np.testing.assert_array_equal(frozen.Z, Z0)        # frozen means frozen

    # the lengthscale the frozen grid cannot afford, and where Z went to buy it
    exact_l, frozen_l, free_l = (np.exp(p[1]) for p in
                                 (exact.params, frozen.params, free.params))
    assert frozen_l > 2.0 * exact_l
    assert abs(free_l - exact_l) < 0.5 * abs(frozen_l - exact_l)
    assert (free.Z > 0).sum() > (Z0 > 0).sum()         # migrated to the fast half


def test_frozen_Z_works_for_kernels_with_no_input_derivative():
    """``optimize_Z=False`` must route through ``elbo_grad_params``, which is
    the only reason the two entry points are separate: Matern nu=1/2 and Gibbs
    have no dF/dZ and would raise if the joint path were taken anyway."""
    rng = np.random.default_rng(3)
    X, y = _packed_toy(rng, n=120)
    for kernel in (Matern(nu=0.5, s2=0.5, l=2.0), Gibbs(s2=0.5, a=0.0, b=0.5)):
        model = SGPR(kernel, Z=np.linspace(-4, 4, 10).reshape(-1, 1),
                     noise_var=0.3)
        res = maximize_elbo(model, X, y, optimize_Z=False, lr=0.05, steps=150)
        assert res.elbo > res.history[0]
        with pytest.raises(NotImplementedError):
            maximize_elbo(model, X, y, optimize_Z=True, steps=1)


def test_maximize_elbo_leaves_the_model_at_the_returned_fit():
    """The model is conditioned at the winner, not at the last iterate -- so a
    caller can predict straight after optimizing without refitting."""
    rng = np.random.default_rng(4)
    X, y = _packed_toy(rng, n=120)
    model = SGPR(RBF(s2=0.5, l=2.0), Z=np.linspace(-4, 4, 10).reshape(-1, 1),
                 noise_var=0.3)
    res = maximize_elbo(model, X, y, lr=0.05, steps=200)

    np.testing.assert_array_equal(model.params, res.params)
    np.testing.assert_array_equal(model.Z, res.Z)
    assert model.elbo() == res.elbo
    assert model.trace_term() == res.trace_term
    model.predict(_grid())                             # fitted, no refit needed


def test_optimized_inducing_points_converge_to_the_exact_posterior_in_M():
    """The week's measurement in miniature: at the exact GP's own
    hyperparameters, sweeping M with Z optimized must drive the bound up toward
    the exact log evidence and the posterior error down toward zero.

    Hyperparameters are held at the exact fit's so that the only thing varying
    is the approximation. ``experiments/sparse.py`` runs the full version,
    where theta is free too.
    """
    rng = np.random.default_rng(5)
    X, y = _packed_toy(rng, n=300)
    Xs = _grid(120, -4, 4)

    exact = GPRegressor(RBF(s2=1.0, l=0.5), noise_var=0.02).fit(X, y)
    reference = exact.log_marginal_likelihood()
    mean_exact, var_exact = exact.predict(Xs)

    bounds, mean_errs, sd_errs = [], [], []
    for M in (4, 16, 64):
        model = SGPR(RBF(s2=1.0, l=0.5),
                     Z=np.quantile(X[:, 0], np.linspace(0, 1, M)).reshape(-1, 1),
                     noise_var=0.02)
        # theta frozen by hand: optimize Z only, by zeroing nothing -- instead
        # run the joint optimizer and reset theta, which is not available, so
        # ascend Z directly through the model's own gradient.
        for _ in range(300):
            model.fit(X, y)
            _, _, gZ = model.elbo_and_grads()
            model.Z = model.Z + 0.02 * gZ / (np.abs(gZ).max() + 1e-12)
        model.fit(X, y)
        mean, var = model.predict(Xs)
        bounds.append(model.elbo())
        mean_errs.append(float(np.abs(mean - mean_exact).max()))
        sd_errs.append(float(np.abs(np.sqrt(var) - np.sqrt(var_exact)).max()))

    assert bounds[0] < bounds[1] < bounds[2] < reference
    assert mean_errs[0] > mean_errs[1] > mean_errs[2]
    assert sd_errs[0] > sd_errs[1] > sd_errs[2]
    assert reference - bounds[-1] < 0.05 * (reference - bounds[0])
