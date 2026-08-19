"""The Laplace approximation: its algebra, its limits, and the oracle judging it.

``gp/laplace.py`` is the first thing in this repo whose answer is *not* exact,
and unlike the sparse GPs of Sec. 11-13 its error has no bound attached -- the
variational free energy is always below the true evidence, the Laplace
approximation is simply near it. So the testing strategy is different: pin the
algebra against cases where the exact answer is known, then measure the error
where it is not.

Four groups.

1. **The likelihoods**, elementwise: every derivative against central
   differences, the stable log-sigmoid against the naive one where the naive one
   still works, and the target validation.
2. **The Gaussian control.** With a Gaussian likelihood the approximation is
   exact, so ``LaplaceGP`` must reproduce ``GPRegressor`` -- mode, predictive
   mean, predictive variance and log marginal likelihood -- through completely
   separate code. This is the test that would catch an error in the
   ``B = I + sqrt(W) K sqrt(W)`` rearrangement, the determinant, or Alg. 3.2.
3. **Exact quadrature at n = 1**, where the posterior is a one-dimensional
   integral and can be done to machine precision. That gives Laplace's *actual*
   error -- and its sign, which is the part worth knowing.
4. **The experiment's oracle, validated against a known answer.** The prior
   importance sampler in ``experiments/laplace.py`` is what judges the
   approximation, so it is itself checked on a Gaussian likelihood, where
   ``p(y)`` is the exact log marginal likelihood.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from gp.gp import GPRegressor  # noqa: E402
from gp.kernels import RBF, Matern  # noqa: E402
from gp.laplace import (Bernoulli, GaussianLikelihood, LaplaceGP,  # noqa: E402
                        Poisson, grid_search)

LIKS = [
    ("bernoulli", Bernoulli(), np.array([1.0, -1.0, 1.0, -1.0, 1.0])),
    ("poisson", Poisson(), np.array([0.0, 1.0, 3.0, 7.0, 2.0])),
    ("gaussian", GaussianLikelihood(0.25), np.array([0.3, -1.2, 0.0, 2.1, -0.4])),
]
F = np.array([-2.5, -0.4, 0.0, 0.9, 1.7])


# ---------------------------------------------------------------------------
# 1. the likelihoods
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,lik,y", LIKS)
def test_first_derivative_by_finite_differences(name, lik, y):
    h = 1e-6
    fd = (lik.log_pdf(y, F + h) - lik.log_pdf(y, F - h)) / (2 * h)
    np.testing.assert_allclose(fd, lik.d1(y, F), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("name,lik,y", LIKS)
def test_second_derivative_by_finite_differences(name, lik, y):
    h = 1e-5
    fd = (lik.d1(y, F + h) - lik.d1(y, F - h)) / (2 * h)
    np.testing.assert_allclose(fd, lik.d2(y, F), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("name,lik,y", LIKS)
def test_all_three_likelihoods_are_log_concave(name, lik, y):
    """W = -d2 >= 0 everywhere, which is what makes sqrt(W) and the Newton
    guarantees legitimate."""
    f = np.linspace(-8, 8, 201)
    yy = np.resize(y, f.shape)
    assert np.all(lik.d2(yy, f) <= 0)


def test_bernoulli_log_pdf_is_stable_in_the_tails():
    """A confident correct point must not go through log(1 - 1e-18), and a
    confident wrong one must return a finite number rather than -inf."""
    lik = Bernoulli()
    y = np.array([1.0, 1.0])
    f = np.array([50.0, -50.0])
    lp = lik.log_pdf(y, f)
    assert np.all(np.isfinite(lp))
    assert lp[0] == pytest.approx(0.0, abs=1e-20)
    assert lp[1] == pytest.approx(-50.0, rel=1e-12)
    # and it agrees with the naive form where the naive form is safe
    mid = np.linspace(-20, 20, 41)
    naive = np.log(1.0 / (1.0 + np.exp(-mid)))
    np.testing.assert_allclose(lik.log_pdf(np.ones_like(mid), mid), naive, atol=1e-12)


def test_bernoulli_curvature_is_bounded_by_a_quarter():
    """W <= 1/4 for the logit link, whatever the data -- so B's conditioning is
    controlled by the kernel alone. Poisson has no such bound, which is why the
    line search exists."""
    f = np.linspace(-30, 30, 1001)
    y = np.ones_like(f)
    assert (-Bernoulli().d2(y, f)).max() <= 0.25 + 1e-15
    assert (-Poisson().d2(np.zeros_like(f), f)).max() > 1e10


def test_target_validation():
    assert np.all(Bernoulli().check_targets([0, 1, 1, 0]) == [-1, 1, 1, -1])
    assert np.all(Bernoulli().check_targets([-1, 1]) == [-1, 1])
    with pytest.raises(ValueError):
        Bernoulli().check_targets([0, 2])
    with pytest.raises(ValueError):
        Poisson().check_targets([1.0, -2.0])
    with pytest.raises(ValueError):
        Poisson().check_targets([1.5])
    with pytest.raises(ValueError):
        GaussianLikelihood(0.0)


# ---------------------------------------------------------------------------
# 2. the Gaussian control: Laplace must BE the exact GP
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(0)
    X = np.linspace(-3, 3, 40).reshape(-1, 1)
    return X, rng.normal(np.sin(X).ravel(), 0.3)


@pytest.mark.parametrize("kernel_kw", [dict(s2=1.3, l=0.8), dict(s2=0.4, l=2.5)])
@pytest.mark.parametrize("noise", [0.09, 1.0])
def test_gaussian_likelihood_reproduces_the_exact_gp(regression_data, kernel_kw,
                                                     noise):
    X, y = regression_data
    Xs = np.linspace(-4, 4, 61).reshape(-1, 1)
    exact = GPRegressor(RBF(**kernel_kw), noise_var=noise).fit(X, y)
    lap = LaplaceGP(RBF(**kernel_kw), GaussianLikelihood(noise)).fit(X, y)

    em, ev = exact.predict(Xs)
    lm, lv = lap.predict(Xs)
    np.testing.assert_allclose(lm, em, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(lv, ev, rtol=1e-10, atol=1e-12)
    assert lap.log_marginal_likelihood() == pytest.approx(
        exact.log_marginal_likelihood(), rel=1e-11)
    # the mode is the posterior mean at the training inputs; the residual is the
    # two models' different jitter placement (both add JITTER * mean(diag K),
    # but only one of them then removes it in predict), not the iteration
    np.testing.assert_allclose(lap.fit_.f, exact.predict(X)[0], atol=1e-8)


def test_gaussian_likelihood_converges_in_one_newton_step(regression_data):
    """Psi is exactly quadratic there, so Newton is exact in one step and the
    second iteration is the one that detects it."""
    X, y = regression_data
    lap = LaplaceGP(RBF(s2=1.0, l=1.0), GaussianLikelihood(0.1)).fit(X, y)
    assert lap.fit_.n_iter <= 2


# ---------------------------------------------------------------------------
# 3. the Newton solve on a real non-Gaussian problem
# ---------------------------------------------------------------------------
def _classification_data(n=20, seed=3):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-3, 3, n)).reshape(-1, 1)
    y = np.sign(np.sin(1.2 * X.ravel()))
    y[y == 0] = 1.0
    return X, y


@pytest.mark.parametrize("lik,data", [
    (Bernoulli(), _classification_data()),
    (Poisson(), (np.linspace(-2, 2, 15).reshape(-1, 1),
                 np.array([0, 0, 1, 1, 2, 3, 5, 8, 6, 4, 3, 1, 1, 0, 0.0]))),
])
def test_mode_is_stationary(lik, data):
    """``K^-1 fhat == grad log p(y | fhat)``, computed by two different routes."""
    X, y = data
    m = LaplaceGP(RBF(s2=2.0, l=1.0), lik).fit(X, y)
    assert m.fit_.stationarity(lik, m.y) < 1e-9


@pytest.mark.parametrize("start", [-4.0, -1.0, 0.5, 3.0])
def test_the_mode_is_unique(start):
    """Psi is concave, so every start lands on the same maximum. If it did not,
    every number in Sec. 16 would depend on the initialization."""
    X, y = _classification_data()
    base = LaplaceGP(RBF(s2=4.0, l=1.0), Bernoulli()).fit(X, y)
    other = LaplaceGP(RBF(s2=4.0, l=1.0), Bernoulli()).fit(
        X, y, f_init=np.full(len(y), start))
    np.testing.assert_allclose(other.fit_.f, base.fit_.f, atol=1e-7)
    assert other.log_marginal_likelihood() == pytest.approx(
        base.log_marginal_likelihood(), abs=1e-9)


def test_the_objective_never_decreases():
    """What the line search is for. A bare Newton step on a Poisson likelihood
    (curvature e^f) can overshoot, and an objective that goes down is how that
    shows up."""
    X = np.linspace(-2, 2, 25).reshape(-1, 1)
    y = np.round(np.exp(1.5 * np.sin(2 * X.ravel())) * 2).astype(float)
    m = LaplaceGP(RBF(s2=9.0, l=0.4), Poisson()).fit(X, y)
    psi = np.array(m.fit_.objective)
    assert np.all(np.diff(psi) >= -1e-12)
    assert m.fit_.stationarity(m.likelihood, m.y) < 1e-8


def test_it_survives_a_near_singular_kernel_matrix():
    """The whole point of factorizing ``B`` instead of ``K``: K here is
    numerically singular (near-duplicate inputs, long lengthscale) and the
    solve still lands on a stationary point."""
    X = np.array([[0.0], [1e-6], [2e-6], [1.0], [1.0 + 1e-6], [2.0]])
    y = np.array([1.0, 1.0, 1.0, -1.0, -1.0, -1.0])
    K = RBF(s2=1.0, l=3.0)(X, X)
    assert np.linalg.cond(K) > 1e12
    m = LaplaceGP(RBF(s2=1.0, l=3.0), Bernoulli()).fit(X, y)
    assert m.fit_.stationarity(m.likelihood, m.y) < 1e-8
    assert np.isfinite(m.log_marginal_likelihood())


def test_predictive_variance_relaxes_to_the_prior_far_away():
    X, y = _classification_data()
    m = LaplaceGP(RBF(s2=2.0, l=0.7), Bernoulli()).fit(X, y)
    _, var = m.predict(np.array([[0.0], [200.0]]))
    assert var[1] == pytest.approx(2.0, rel=1e-6)
    assert var[0] < 2.0
    p, _ = m.predict_prob(np.array([[200.0]]))
    assert p[0] == pytest.approx(0.5, abs=1e-6)     # symmetric prior, no data


def test_dimension_mismatch_refuses():
    with pytest.raises(ValueError):
        LaplaceGP(RBF(), Bernoulli()).fit(np.zeros((4, 1)), np.ones(3))


# ---------------------------------------------------------------------------
# 4. quadrature: the exact answer at n = 1
# ---------------------------------------------------------------------------
def _exact_n1(lik, y, s2, lo=-40.0, hi=40.0, n=400_001):
    """Exact posterior at n = 1 by dense quadrature: (log Z, mean, sd)."""
    f = np.linspace(lo, hi, n)
    logp = lik.log_pdf(np.full(n, y), f) - 0.5 * f ** 2 / s2 \
        - 0.5 * np.log(2 * np.pi * s2)
    m = logp.max()
    w = np.exp(logp - m)
    dx = f[1] - f[0]
    Z = w.sum() * dx
    mean = (w * f).sum() * dx / Z
    var = (w * (f - mean) ** 2).sum() * dx / Z
    return float(np.log(Z) + m), float(mean), float(np.sqrt(var))


@pytest.mark.parametrize("s2", [0.25, 1.0, 4.0, 16.0])
def test_laplace_against_exact_quadrature_at_n_one(s2):
    """At n = 1 the exact posterior is a 1-D integral, so the approximation's
    error is available exactly -- with its sign, which is the finding.

    Two things are asserted rather than merely printed: Laplace *underestimates*
    the evidence, and it *underestimates* the posterior spread. Both directions
    hold at every amplitude tried, and both are the systematic behaviour the
    experiment then measures at larger n against an importance-sampling oracle.
    """
    lik = Bernoulli()
    X = np.zeros((1, 1))
    y = np.array([1.0])
    m = LaplaceGP(RBF(s2=s2, l=1.0), lik).fit(X, y)
    log_Z, mean, sd = _exact_n1(lik, 1.0, s2)

    assert m.log_marginal_likelihood() < log_Z              # one-signed
    assert m.log_marginal_likelihood() == pytest.approx(log_Z, abs=0.25)
    assert m.fit_.f[0] < mean                               # mode below the mean
    _, var = m.predict(X)
    assert np.sqrt(var[0]) < sd                             # over-confident
    assert np.sqrt(var[0]) == pytest.approx(sd, rel=0.3)


def test_the_error_grows_with_the_prior_amplitude():
    """The headline of Sec. 16's sweep, at n = 1 where it needs no oracle."""
    lik = Bernoulli()
    errs = []
    for s2 in (0.25, 1.0, 4.0, 16.0, 64.0):
        m = LaplaceGP(RBF(s2=s2, l=1.0), lik).fit(np.zeros((1, 1)), np.array([1.0]))
        log_Z, _, _ = _exact_n1(lik, 1.0, s2)
        errs.append(abs(m.log_marginal_likelihood() - log_Z))
    assert np.all(np.diff(errs) > 0)


# ---------------------------------------------------------------------------
# 5. predictive averaging
# ---------------------------------------------------------------------------
def test_gauss_hermite_averaging_matches_monte_carlo():
    X, y = _classification_data()
    m = LaplaceGP(RBF(s2=4.0, l=1.0), Bernoulli()).fit(X, y)
    Xs = np.linspace(-3, 3, 9).reshape(-1, 1)
    p, var = m.predict_prob(Xs)
    p_fine, _ = m.predict_prob(Xs, nodes=200)
    np.testing.assert_allclose(p, p_fine, atol=1e-12)

    mean, _ = m.predict(Xs)
    rng = np.random.default_rng(0)
    draws = mean[:, None] + np.sqrt(var)[:, None] * rng.standard_normal((len(Xs), 400_000))
    np.testing.assert_allclose(p, Bernoulli().mean(draws).mean(axis=1), atol=3e-3)


def test_averaging_pulls_probabilities_toward_one_half():
    """``E[sigma(f)]`` is closer to 0.5 than ``sigma(E[f])`` by Jensen on each
    side of the inflection -- so the plug-in is over-confident, always."""
    X, y = _classification_data()
    m = LaplaceGP(RBF(s2=9.0, l=1.0), Bernoulli()).fit(X, y)
    Xs = np.linspace(-3, 3, 41).reshape(-1, 1)
    p_avg, _ = m.predict_prob(Xs)
    p_plug = Bernoulli().mean(m.predict(Xs)[0])
    assert np.all(np.abs(p_avg - 0.5) <= np.abs(p_plug - 0.5) + 1e-12)
    assert np.max(np.abs(p_avg - p_plug)) > 0.02


# ---------------------------------------------------------------------------
# 6. grid search
# ---------------------------------------------------------------------------
def test_grid_search_returns_the_argmax_of_its_own_table():
    X, y = _classification_data(n=12)
    s2g = np.geomspace(0.5, 8.0, 5)
    lg = np.geomspace(0.3, 3.0, 4)
    best, best_ml, table = grid_search(lambda s2, l: RBF(s2=s2, l=l), Bernoulli(),
                                       X, y, (s2g, lg))
    assert table.shape == (5, 4)
    assert best_ml == pytest.approx(table.max())
    i, j = np.unravel_index(int(np.argmax(table)), table.shape)
    assert best == (pytest.approx(s2g[i]), pytest.approx(lg[j]))


def test_on_edge_flags_a_grid_that_is_too_small():
    """The check that caught a boundary argmax being reported as an optimum."""
    from laplace import on_edge

    grids = (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
    interior = np.zeros((3, 3))
    interior[1, 1] = 1.0
    assert on_edge(interior, grids) == []
    corner = np.zeros((3, 3))
    corner[2, 0] = 1.0
    assert on_edge(corner, grids) == [0, 1]


# ---------------------------------------------------------------------------
# 7. the experiment's oracle, checked against a known answer
# ---------------------------------------------------------------------------
def test_prior_importance_oracle_recovers_the_exact_log_evidence():
    """The oracle judges the approximation, so it is judged first.

    With a Gaussian likelihood ``p(y)`` is the exact log marginal likelihood, and
    the sampler must find it. Its own standard error is reported by the spread
    across seeds; the tolerance below is that spread, not a guess.
    """
    from laplace import prior_importance_oracle

    rng = np.random.default_rng(1)
    X = np.linspace(-2, 2, 6).reshape(-1, 1)
    y = rng.normal(np.sin(X).ravel(), 0.5)
    Xs = np.array([[0.0], [3.0]])
    kernel = RBF(s2=1.0, l=1.0)
    exact = GPRegressor(RBF(s2=1.0, l=1.0), noise_var=0.25).fit(X, y)

    runs = [prior_importance_oracle(kernel, GaussianLikelihood(0.25), X, y, Xs,
                                    n_samples=400_000, seed=s) for s in range(4)]
    log_Zs = np.array([r["log_Z"] for r in runs])
    assert log_Zs.mean() == pytest.approx(exact.log_marginal_likelihood(),
                                          abs=4 * log_Zs.std(ddof=1) + 0.02)
    # and the posterior mean it reports is the exact GP's
    means = np.array([r["mean"] for r in runs]).mean(axis=0)
    np.testing.assert_allclose(means, exact.predict(Xs)[0], atol=0.03)


def test_oracle_weights_are_bounded_for_a_bernoulli_likelihood():
    """The property that makes prior importance sampling a legitimate oracle
    here rather than a second approximation: the weights are the likelihood,
    which is in (0, 1], so the estimator of p(y) has finite variance by
    construction."""
    lik = Bernoulli()
    f = np.linspace(-50, 50, 501)
    assert np.all(np.exp(lik.log_pdf(np.ones_like(f), f)) <= 1.0)


def test_a_second_kernel_family_works_too():
    """Nothing in the module is RBF-specific; Matern has a different diagonal
    and a different gradient structure."""
    X, y = _classification_data(n=15)
    m = LaplaceGP(Matern(nu=2.5, s2=2.0, l=1.0), Bernoulli()).fit(X, y)
    assert m.fit_.stationarity(m.likelihood, m.y) < 1e-9
    p, _ = m.predict_prob(np.linspace(-3, 3, 11).reshape(-1, 1))
    assert np.all((p > 0) & (p < 1))
