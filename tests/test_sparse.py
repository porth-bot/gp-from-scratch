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
from gp.kernels import ARD, RBF, Gibbs, Matern, Periodic
from gp.sparse import SGPR, kernel_diag


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
