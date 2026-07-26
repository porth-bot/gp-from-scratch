"""Random Fourier features: unbiasedness, the D^{-1/2} rate, and the
weight-space/function-space identity against the exact GP."""

import numpy as np
import pytest

from gp.gp import GPRegressor
from gp.kernels import ARD, RBF
from gp.rff import RFFMap, RFFRegressor


def _grid(n=40, lo=-3.0, hi=3.0):
    return np.linspace(lo, hi, n).reshape(-1, 1)


def _kernel_error(D, seed, l=1.0, s2=1.5, offset=False, X=None):
    """Max |k_D - k| over a grid, for one draw of the frequencies."""
    X = _grid() if X is None else X
    rng = np.random.default_rng(seed)
    phi = RFFMap(D, lengthscale=l, s2=s2, n_dims=1, rng=rng, offset=offset)
    return float(np.abs(phi(X, X) - RBF(s2=s2, l=l)(X, X)).max())


# -- the feature map ---------------------------------------------------------


def test_cos_sin_map_reproduces_the_prior_variance_exactly():
    """z(x).z(x) = s2 for every x -- cos^2 + sin^2 = 1 per frequency, not just
    in expectation. The offset variant only gets this right on average."""
    rng = np.random.default_rng(0)
    X = _grid()
    paired = RFFMap(64, lengthscale=0.7, s2=2.5, n_dims=1, rng=rng)
    assert np.allclose(np.sum(paired.transform(X) ** 2, axis=1), 2.5, atol=1e-12)

    offset = RFFMap(64, lengthscale=0.7, s2=2.5, n_dims=1, rng=rng, offset=True)
    diag = np.sum(offset.transform(X) ** 2, axis=1)
    assert np.abs(diag - 2.5).max() > 1e-3        # fluctuates around s2
    assert np.abs(diag.mean() - 2.5) < 0.5        # but is centered on it


def test_rff_kernel_is_unbiased_for_the_rbf():
    """Averaging k_D over independent draws of the frequencies converges to the
    exact RBF: the estimator has no bias to trade against its variance."""
    X = _grid(n=12)
    exact = RBF(s2=1.0, l=1.2)(X, X)
    acc = np.zeros_like(exact)
    n_draws = 400
    for seed in range(n_draws):
        rng = np.random.default_rng(seed)
        phi = RFFMap(32, lengthscale=1.2, s2=1.0, n_dims=1, rng=rng)
        acc += phi(X, X)
    mean_error = np.abs(acc / n_draws - exact).max()
    single_error = np.median([_kernel_error(32, s, l=1.2, s2=1.0) for s in range(20)])
    # The average of 400 draws is far better than any single draw, which is
    # only possible if the draws are centered on the truth.
    assert mean_error < single_error / 5


def test_kernel_error_decays_as_one_over_sqrt_D():
    """A Monte Carlo average of m = D/2 terms: error ~ D^{-1/2}, so a 16x
    feature budget buys a ~4x smaller error."""
    seeds = range(24)
    err_small = np.median([_kernel_error(64, s) for s in seeds])
    err_large = np.median([_kernel_error(64 * 16, s) for s in seeds])
    ratio = err_small / err_large
    assert 3.0 < ratio < 5.5, ratio


def test_paired_features_beat_the_offset_variant():
    """Same D, same expectation, lower variance (Sutherland & Schneider 2015)."""
    seeds = range(24)
    paired = np.median([_kernel_error(128, s) for s in seeds])
    offset = np.median([_kernel_error(128, s, offset=True) for s in seeds])
    assert paired < offset


def test_vector_lengthscale_approximates_the_ard_kernel():
    """A per-dimension spectral density N(0, diag(l_d^{-2})) gives the ARD RBF."""
    rng = np.random.default_rng(3)
    X = rng.uniform(-2, 2, size=(30, 2))
    ls = np.array([0.4, 2.0])
    phi = RFFMap(8192, lengthscale=ls, s2=1.3, n_dims=2, rng=rng)
    exact = ARD(s2=1.3, lengthscales=ls)(X, X)
    assert np.abs(phi(X, X) - exact).max() < 0.05


def test_feature_map_validates_its_arguments():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        RFFMap(65, lengthscale=1.0, s2=1.0, n_dims=1, rng=rng)      # odd D
    with pytest.raises(ValueError):
        RFFMap(0, lengthscale=1.0, s2=1.0, n_dims=1, rng=rng)
    with pytest.raises(ValueError):
        RFFMap(16, lengthscale=-1.0, s2=1.0, n_dims=1, rng=rng)
    with pytest.raises(ValueError):
        RFFMap(16, lengthscale=[1.0, 2.0], s2=1.0, n_dims=1, rng=rng)
    phi = RFFMap(16, lengthscale=1.0, s2=1.0, n_dims=1, rng=rng)
    with pytest.raises(ValueError):
        phi.transform(np.zeros((5, 3)))
    RFFMap(9, lengthscale=1.0, s2=1.0, n_dims=1, rng=rng, offset=True)  # odd is fine


# -- the regressor -----------------------------------------------------------


def _toy(rng, n=80):
    X = np.sort(rng.uniform(-3, 3, n)).reshape(-1, 1)
    y = np.sin(1.5 * X[:, 0]) + 0.1 * rng.standard_normal(n)
    return X, y


def test_weight_space_and_function_space_posteriors_agree_exactly():
    """The claim the module rests on: with the feature map FIXED, RFF
    regression is not an approximate GP -- it is the exact GP of the
    finite-rank kernel k_D, computed in D dimensions instead of n. So the two
    routes to the same posterior must agree to machine precision, which pins
    the only error source to k_D ~ k."""
    rng = np.random.default_rng(7)
    X, y = _toy(rng)
    Xs = _grid(n=25)
    phi = RFFMap(128, lengthscale=1.0, s2=1.0, n_dims=1, rng=rng)

    weight_space = RFFRegressor(phi, noise_var=0.05).fit(X, y)
    function_space = GPRegressor(phi, noise_var=0.05).fit(X, y)   # k_D as a kernel

    m_w, v_w = weight_space.predict(Xs)
    m_f, v_f = function_space.predict(Xs)
    assert np.abs(m_w - m_f).max() < 1e-8
    assert np.abs(v_w - v_f).max() < 1e-8

    m_w, v_w = weight_space.predict(Xs, include_noise=True)
    m_f, v_f = function_space.predict(Xs, include_noise=True)
    assert np.abs(v_w - v_f).max() < 1e-8


def test_posterior_mean_error_shrinks_with_more_features():
    """The approximation converges where it counts: predictions."""
    rng = np.random.default_rng(11)
    X, y = _toy(rng, n=120)
    Xs = _grid(n=60)
    exact = GPRegressor(RBF(s2=1.0, l=1.0), noise_var=0.01).fit(X, y)
    mean_exact, _ = exact.predict(Xs)

    errors = []
    for D in (16, 64, 256, 1024):
        per_seed = []
        for seed in range(9):
            phi = RFFMap(D, lengthscale=1.0, s2=1.0, n_dims=1,
                         rng=np.random.default_rng(100 + seed))
            mean_rff, _ = RFFRegressor(phi, noise_var=0.01).fit(X, y).predict(Xs)
            per_seed.append(np.abs(mean_rff - mean_exact).max())
        errors.append(float(np.median(per_seed)))

    assert all(a > b for a, b in zip(errors, errors[1:])), errors
    assert errors[-1] < 0.02                      # and it actually gets close
    assert errors[0] / errors[-1] > 5.0


def test_variance_starvation_when_n_far_exceeds_D():
    """The honest failure mode. A rank-D model has D degrees of freedom; once
    the data has spent them all, the posterior is confident everywhere --
    including in a gap where the exact GP correctly says it does not know."""
    rng = np.random.default_rng(5)
    x = rng.uniform(-4, 4, 4000)
    x = x[np.abs(x) > 1.0]                        # a gap in [-1, 1]
    X = np.sort(x).reshape(-1, 1)
    y = np.sin(X[:, 0]) + 0.1 * rng.standard_normal(len(X))
    gap = np.array([[0.0]])

    exact = GPRegressor(RBF(s2=1.0, l=0.5), noise_var=0.01).fit(X, y)
    _, var_exact = exact.predict(gap)

    starved = RFFRegressor(
        RFFMap(32, lengthscale=0.5, s2=1.0, n_dims=1, rng=np.random.default_rng(1)),
        noise_var=0.01,
    ).fit(X, y)
    _, var_starved = starved.predict(gap)

    rich = RFFRegressor(
        RFFMap(2048, lengthscale=0.5, s2=1.0, n_dims=1, rng=np.random.default_rng(1)),
        noise_var=0.01,
    ).fit(X, y)
    _, var_rich = rich.predict(gap)

    # The exact GP is genuinely uncertain in the gap (l = 0.5, gap half-width 1).
    assert var_exact[0] > 0.2
    # D = 32 << n: the gap uncertainty is essentially gone.
    assert var_starved[0] < var_exact[0] / 5
    # Enough features and it comes back.
    assert var_rich[0] > var_exact[0] / 2
