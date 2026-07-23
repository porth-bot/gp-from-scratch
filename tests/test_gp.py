"""GP regression: LML gradient check, sklearn oracle, interpolation, calibration."""

import numpy as np
import pytest

from gp.gp import GPRegressor, sample_prior
from gp.kernels import ARD, RBF, Matern, Periodic
from gp.optimize import adam_maximize, maximize_lml_multistart


def make_data(rng, n=25):
    X = rng.uniform(-3, 3, size=(n, 1))
    y = np.sin(2.0 * X[:, 0]) + 0.1 * rng.standard_normal(n)
    return X, y


def test_lml_gradient_matches_finite_differences():
    rng = np.random.default_rng(1)
    X, y = make_data(rng)
    model = GPRegressor(RBF(s2=1.3, l=0.8), noise_var=0.05)
    p0 = model.params
    _, analytic = model.lml_and_grad(X, y, p0)

    eps = 1e-6
    numeric = np.empty_like(p0)
    for i in range(len(p0)):
        pp, pm = p0.copy(), p0.copy()
        pp[i] += eps
        pm[i] -= eps
        lp, _ = model.lml_and_grad(X, y, pp)
        lm, _ = model.lml_and_grad(X, y, pm)
        numeric[i] = (lp - lm) / (2 * eps)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-4, atol=1e-6)


def test_agrees_with_sklearn_oracle():
    """Same kernel, same fixed hyperparameters, same data: posterior mean,
    posterior std, and log marginal likelihood must match scikit-learn's
    independent implementation."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF as SkRBF
    from sklearn.gaussian_process.kernels import ConstantKernel

    rng = np.random.default_rng(2)
    X, y = make_data(rng, n=30)
    Xs = np.linspace(-4, 4, 50)[:, None]
    s2, l, noise = 1.7, 0.9, 0.03

    ours = GPRegressor(RBF(s2=s2, l=l), noise_var=noise).fit(X, y)
    mean, var = ours.predict(Xs)

    sk = GaussianProcessRegressor(
        kernel=ConstantKernel(s2, "fixed") * SkRBF(l, "fixed"),
        alpha=noise,          # sklearn's alpha is added to K's diagonal = noise var
        optimizer=None,
    ).fit(X, y)
    sk_mean, sk_std = sk.predict(Xs, return_std=True)

    np.testing.assert_allclose(mean, sk_mean, atol=1e-8)
    np.testing.assert_allclose(np.sqrt(var), sk_std, atol=1e-6)
    np.testing.assert_allclose(
        ours.log_marginal_likelihood(), sk.log_marginal_likelihood_value_, atol=1e-8
    )


def _brute_force_loo(kernel_factory, noise_var, X, y, noise=None):
    """Ground truth: literally refit the GP on all points except i and predict
    the held-out observation (include_noise=True), for every i."""
    n = len(y)
    means, vars_ = np.empty(n), np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        m = GPRegressor(kernel_factory(), noise_var=noise_var)
        if noise is None:
            m.fit(X[keep], y[keep])
            m_var_i = m.noise_var
        else:
            m.fit(X[keep], y[keep], noise=noise[keep])
            m_var_i = noise[i]
        mean_i, var_i = m.predict(X[i : i + 1])       # latent band
        means[i] = mean_i[0]
        vars_[i] = var_i[0] + m_var_i                 # add back point i's noise
    return means, vars_


def test_closed_form_loo_matches_brute_force_refits():
    """The R&W 5.4.2 closed form must equal n explicit leave-one-out refits,
    for both the LOO predictive mean/variance and the log-CV score."""
    rng = np.random.default_rng(7)
    X, y = make_data(rng, n=18)
    kf = lambda: RBF(s2=1.4, l=0.9)
    model = GPRegressor(kf(), noise_var=0.08).fit(X, y)

    mean, var, log_pred = model.loo()
    bf_mean, bf_var = _brute_force_loo(kf, 0.08, X, y)
    np.testing.assert_allclose(mean, bf_mean, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(var, bf_var, rtol=1e-8, atol=1e-8)

    # per-point log density recomputed from the brute-force predictions
    bf_log = -0.5 * (np.log(2 * np.pi * bf_var) + (y - bf_mean) ** 2 / bf_var)
    np.testing.assert_allclose(log_pred, bf_log, rtol=1e-8, atol=1e-8)
    assert np.isclose(model.loo_log_predictive(), bf_log.sum())


def test_closed_form_loo_matches_brute_force_heteroscedastic():
    """The identity is purely linear-algebraic in K_y, so it holds unchanged
    when the diagonal carries per-point (heteroscedastic) noise."""
    rng = np.random.default_rng(8)
    X, y = make_data(rng, n=15)
    noise = rng.uniform(0.02, 0.2, size=len(y))
    kf = lambda: Matern(nu=2.5, s2=1.1, l=0.8)
    model = GPRegressor(kf(), noise_var=0.05).fit(X, y, noise=noise)

    mean, var, _ = model.loo()
    bf_mean, bf_var = _brute_force_loo(kf, 0.05, X, y, noise=noise)
    np.testing.assert_allclose(mean, bf_mean, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(var, bf_var, rtol=1e-7, atol=1e-8)


def test_noiseless_gp_interpolates():
    rng = np.random.default_rng(3)
    X, _ = make_data(rng, n=12)
    y = np.cos(X[:, 0])
    model = GPRegressor(RBF(s2=1.0, l=1.0), noise_var=1e-10).fit(X, y)
    mean, var = model.predict(X)
    np.testing.assert_allclose(mean, y, atol=1e-5)
    assert var.max() < 1e-6


def test_posterior_is_calibrated_on_gp_samples():
    """House rule: test the claim the model actually makes. Draw a function
    from the prior, observe part of it with noise, predict the rest -- the
    95% credible intervals must cover ~95% of held-out latent values."""
    rng = np.random.default_rng(4)
    kernel = Matern(nu=2.5, s2=2.0, l=1.0)
    grid = np.linspace(-5, 5, 400)[:, None]
    K = kernel(grid, grid) + 1e-10 * np.eye(400)
    f = np.linalg.cholesky(K) @ rng.standard_normal(400)  # exact prior draw

    covered, total = 0, 0
    for _ in range(20):
        idx = rng.permutation(400)
        train, test = idx[:60], idx[60:160]
        noise = 0.05
        y = f[train] + np.sqrt(noise) * rng.standard_normal(60)
        model = GPRegressor(Matern(nu=2.5, s2=2.0, l=1.0), noise_var=noise)
        model.fit(grid[train], y)
        mean, var = model.predict(grid[test])
        z = np.abs(f[test] - mean) / np.sqrt(var)
        covered += int((z < 1.96).sum())
        total += len(test)
        f = np.linalg.cholesky(K) @ rng.standard_normal(400)  # fresh function
    coverage = covered / total
    assert 0.90 < coverage < 0.985, coverage


def _bump_field(P):
    x, y = P[:, 0], P[:, 1]
    return (
        np.exp(-((x - 1.0) ** 2 + (y - 1.0) ** 2))
        - 0.6 * np.exp(-((x + 1.2) ** 2 + (y + 0.8) ** 2) / 1.5)
    )


def test_2d_spatial_gp_recovers_a_scattered_field():
    """A GP on 2D inputs interpolates a smooth field from scattered noisy
    samples: held-out RMSE well below the field amplitude, and the 95% band
    covers the latent surface. (The experiments/spatial2d.py figure.)"""
    rng = np.random.default_rng(0)
    noise = 0.05
    Xtr = rng.uniform(-3, 3, size=(140, 2))
    ytr = _bump_field(Xtr) + noise * rng.standard_normal(140)

    model = GPRegressor(RBF(s2=0.5, l=1.0), noise_var=noise ** 2)
    best, _ = adam_maximize(
        lambda p: model.lml_and_grad(Xtr, ytr, p), model.params, lr=0.05, steps=300
    )
    model.params = best
    model.fit(Xtr, ytr)

    Xte = rng.uniform(-3, 3, size=(500, 2))
    truth = _bump_field(Xte)
    mean, var = model.predict(Xte)
    rmse = np.sqrt(np.mean((mean - truth) ** 2))
    cover = np.mean(np.abs(truth - mean) / np.sqrt(var) <= 1.96)
    assert rmse < 0.1                 # amplitude ~1, so this is a tight fit
    assert 0.90 < cover <= 1.0


def test_2d_predictive_variance_grows_away_from_data():
    """The core spatial-GP claim: uncertainty is small near observations and
    relaxes to the prior variance far from all of them. Train only in the
    lower-left quadrant, then compare the variance near the data vs in the
    empty opposite corner."""
    rng = np.random.default_rng(1)
    Xtr = rng.uniform(-3.0, -1.0, size=(80, 2))  # data confined to one corner
    ytr = np.sin(Xtr[:, 0]) + np.cos(Xtr[:, 1])
    s2 = 1.0
    model = GPRegressor(RBF(s2=s2, l=1.0), noise_var=1e-3).fit(Xtr, ytr)

    _, v_near = model.predict(np.array([[-2.0, -2.0]]))  # inside the cluster
    _, v_far = model.predict(np.array([[2.5, 2.5]]))     # far away
    assert v_far[0] > 5 * v_near[0]        # far is much less certain
    assert abs(v_far[0] - s2) < 0.05 * s2  # and has relaxed to the prior var


def test_ml2_recovers_known_hyperparameters():
    """Sample data FROM a GP with known (s2, l, noise); maximizing the
    marginal likelihood from a generic init must land near the truth."""
    rng = np.random.default_rng(5)
    true = dict(s2=2.0, l=0.8, noise=0.01)
    X = np.sort(rng.uniform(-4, 4, size=(150, 1)), axis=0)
    kernel = RBF(s2=true["s2"], l=true["l"])
    K = kernel(X, X) + true["noise"] * np.eye(150)
    y = np.linalg.cholesky(K + 1e-12 * np.eye(150)) @ rng.standard_normal(150)

    model = GPRegressor(RBF(s2=1.0, l=2.0), noise_var=0.5)  # generic init
    best, _ = adam_maximize(
        lambda p: model.lml_and_grad(X, y, p), model.params, lr=0.05, steps=250
    )
    s2_hat, l_hat = np.exp(best[0]), np.exp(best[1])
    noise_hat = np.exp(best[2])
    assert 0.55 < l_hat / true["l"] < 1.6      # lengthscale is well-identified
    assert 0.3 < s2_hat / true["s2"] < 3.0     # variance: weakly identified at n=150
    assert noise_hat < 0.05                    # far below the signal variance


def test_ml2_ard_suppresses_an_irrelevant_input():
    """The behavioral ARD claim: with a response that depends only on x0, ML-II
    on an ARD-RBF drives the irrelevant axis's lengthscale far above the
    relevant one (1/l1 -> ~0), so predictions become invariant to x1."""
    rng = np.random.default_rng(3)
    X = rng.uniform(-3.0, 3.0, size=(120, 2))
    y = np.sin(2.0 * X[:, 0]) + 0.3 * X[:, 0] + 0.1 * rng.standard_normal(120)

    model = GPRegressor(ARD(s2=1.0, lengthscales=[1.0, 1.0]), noise_var=0.2)
    best, _ = adam_maximize(
        lambda p: model.lml_and_grad(X, y, p), model.params, lr=0.05, steps=500
    )
    l0, l1 = np.exp(best[1]), np.exp(best[2])
    assert l1 / l0 > 10.0          # noise axis suppressed by >10x
    assert l0 < 3.0                # relevant axis keeps a finite, sane scale

    # suppression made concrete: moving x1 barely moves the prediction
    model.params = best
    model.fit(X, y)
    grid0 = np.linspace(-3, 3, 50)
    m_lo, _ = model.predict(np.column_stack([grid0, np.full(50, -2.0)]))
    m_hi, _ = model.predict(np.column_stack([grid0, np.full(50, 2.0)]))
    assert np.max(np.abs(m_lo - m_hi)) < 0.05


def test_sample_prior_shape_and_reproducibility():
    X = np.linspace(-3, 3, 40)[:, None]
    s1 = sample_prior(RBF(l=1.0), X, n_samples=5, rng=np.random.default_rng(0))
    s2 = sample_prior(RBF(l=1.0), X, n_samples=5, rng=np.random.default_rng(0))
    assert s1.shape == (5, 40)
    np.testing.assert_allclose(s1, s2)  # same seed -> same draws


def test_sample_prior_empirical_covariance_matches_kernel():
    """Many prior draws must have empirical covariance ~ K(X, X): this is the
    defining property of the GP prior, and the check that L z is wired right."""
    X = np.linspace(-2, 2, 12)[:, None]
    kernel = Matern(nu=1.5, s2=1.3, l=0.7)
    draws = sample_prior(kernel, X, n_samples=200_000, rng=np.random.default_rng(1))
    emp = np.cov(draws.T)
    np.testing.assert_allclose(emp, kernel(X, X), atol=0.03)


def test_periodic_prior_draws_are_periodic():
    """A draw from a Periodic-kernel prior repeats every period: evaluated at
    x and x + p, the two values coincide within a single sample path (their
    prior correlation is exactly 1)."""
    period = 1.3
    x = np.linspace(0, period, 25, endpoint=False)
    X = np.concatenate([x, x + period])[:, None]
    draws = sample_prior(Periodic(l=1.0, p=period), X, 8, np.random.default_rng(2))
    lo, hi = draws[:, :25], draws[:, 25:]
    np.testing.assert_allclose(lo, hi, atol=1e-3)  # holds to the ~sqrt(jitter) floor


def test_sklearn_parity_benchmark_stays_exact():
    """The sklearn-parity benchmark's core comparison must stay at numerical
    parity (identical exact-GP math) across sizes, not just at one n."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sklearn_parity import compare

    rng = np.random.default_rng(0)
    for n in (50, 150):
        r = compare(n, rng, n_timing=1)
        assert r["mean_maxdiff"] < 1e-6, r
        assert r["std_maxdiff"] < 1e-6, r


def test_heteroscedastic_fit_matches_manual_diag_noise_gp():
    """A per-point noise vector must reproduce the exact-GP posterior with
    that noise on the diagonal -- checked against a direct linear-algebra
    computation, and against the scalar path when the vector is constant."""
    rng = np.random.default_rng(7)
    X = np.sort(rng.uniform(-3, 3, size=(24, 1)), axis=0)
    y = np.sin(2 * X[:, 0]) + 0.1 * rng.standard_normal(24)
    Xs = np.linspace(-3, 3, 40)[:, None]
    kernel = RBF(s2=1.2, l=0.7)

    noise = 0.01 + 0.2 * (X[:, 0] > 0)  # heteroscedastic: noisier on the right
    model = GPRegressor(kernel, noise_var=0.05).fit(X, y, noise=noise)
    mean, var = model.predict(Xs)

    # manual exact GP with diag(noise) on the training block
    K = kernel(X, X) + np.diag(noise) + 1e-10 * np.mean(np.diag(kernel(X, X))) * np.eye(24)
    Ks = kernel(X, Xs)
    L = np.linalg.cholesky(K)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    mean_ref = Ks.T @ alpha
    v = np.linalg.solve(L, Ks)
    var_ref = np.diag(kernel(Xs, Xs)) - np.sum(v**2, axis=0)
    np.testing.assert_allclose(mean, mean_ref, atol=1e-10)
    np.testing.assert_allclose(var, var_ref, atol=1e-10)

    # constant vector == scalar path
    m_vec = GPRegressor(kernel, noise_var=0.05).fit(X, y, noise=np.full(24, 0.05))
    m_scal = GPRegressor(kernel, noise_var=0.05).fit(X, y)
    np.testing.assert_allclose(m_vec.predict(Xs)[0], m_scal.predict(Xs)[0], atol=1e-12)


def test_heteroscedastic_noise_wrong_shape_raises():
    rng = np.random.default_rng(0)
    X, y = make_data(rng, n=20)
    import pytest

    with pytest.raises(ValueError):
        GPRegressor(RBF(), noise_var=0.05).fit(X, y, noise=np.ones(5))


def test_two_stage_heteroscedastic_improves_calibration():
    """The two-stage heteroscedastic fit must beat the homoscedastic one where
    it matters: lower test NLL, and 95% intervals that actually cover ~95% in
    the noisy region instead of under-covering it."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from heteroscedastic import evaluate

    m = evaluate(seed=0)
    assert m["hetero_nll"] < m["homo_nll"]                     # better likelihood
    homo_right, hetero_right = m["homo_cover"][1], m["hetero_cover"][1]
    assert homo_right < 0.90                                   # homo under-covers noisy region
    assert abs(hetero_right - 0.95) < abs(homo_right - 0.95)   # hetero closer to nominal


# -- ML-II multi-restart (defeating LML multimodality) -----------------------

def _multimodal_1d():
    """A 12-point set whose RBF+noise evidence has two competing optima: a
    short-lengthscale "it's signal" mode and a long-lengthscale "it's all
    noise" mode (Rasmussen & Williams Fig. 5.5). Sparse samples of a wiggly
    function make both explanations locally credible."""
    rng = np.random.default_rng(0)
    X = np.linspace(-4, 4, 12).reshape(-1, 1)
    y = np.sin(1.7 * X).ravel() + 0.15 * rng.standard_normal(12)
    return X, y


# log-space box for [log s2, log l, log noise_var] used by the restarts.
_BOX = np.log([[1e-2, 1e2], [1e-1, 1e2], [1e-4, 1e0]])


def _single_fit(l0, noise0, X, y, steps=400):
    m = GPRegressor(RBF(s2=1.0, l=l0), noise_var=noise0)
    best, _ = adam_maximize(lambda p: m.lml_and_grad(X, y, p), m.params,
                            lr=0.05, steps=steps)
    return m.lml_and_grad(X, y, best)[0], np.exp(m.kernel.theta[1])  # lml, l


def test_lml_surface_is_actually_multimodal():
    """Guard the premise: two initializations must land in genuinely different
    optima (different evidence AND different lengthscale), or the multi-restart
    test below would be vacuous."""
    X, y = _multimodal_1d()
    lml_signal, l_signal = _single_fit(0.3, 0.1, X, y)   # short-l basin
    lml_noise, l_noise = _single_fit(30.0, 0.6, X, y)    # long-l "all noise" basin
    assert lml_signal > lml_noise + 1.0                  # distinct optima
    assert l_noise > 5.0 * l_signal                      # and distinct lengthscales


def test_multistart_escapes_a_bad_starting_basin():
    """Started in the bad (long-lengthscale) basin, a single fit stays stuck;
    multi-start over the log-space box finds the far better signal explanation
    and leaves the model conditioned there."""
    X, y = _multimodal_1d()
    stuck_lml, _ = _single_fit(30.0, 0.6, X, y)

    model = GPRegressor(RBF(s2=1.0, l=30.0), noise_var=0.6)
    res = maximize_lml_multistart(model, X, y, n_restarts=8, bounds=_BOX,
                                  rng=np.random.default_rng(0), steps=400)
    assert res.lml > stuck_lml + 1.0                     # escaped the bad basin
    assert res.lml == np.max(res.lmls)                   # reported the best restart
    # model is left fit at the winner: its own LML matches the reported value
    assert np.isclose(model.log_marginal_likelihood(), res.lml)
    assert model._fitted


def test_multistart_keeps_a_good_init_when_it_is_already_best():
    """With keep_init=True restart 0 is the model's own params, so multi-start
    can never return worse evidence than the single fit it wraps."""
    X, y = _multimodal_1d()
    # start already in the good basin
    good = GPRegressor(RBF(s2=1.0, l=0.5), noise_var=0.05)
    single_best, _ = adam_maximize(lambda p: good.lml_and_grad(X, y, p),
                                   good.params, lr=0.05, steps=400)
    single_lml = good.lml_and_grad(X, y, single_best)[0]

    model = GPRegressor(RBF(s2=1.0, l=0.5), noise_var=0.05)
    model.params = single_best                            # seed with the single fit
    res = maximize_lml_multistart(model, X, y, n_restarts=6, bounds=_BOX,
                                  rng=np.random.default_rng(1), steps=400)
    assert res.lml >= single_lml - 1e-6                   # never worse than the kept init


def test_multistart_is_reproducible_and_validates_bounds():
    X, y = _multimodal_1d()
    kw = dict(n_restarts=5, bounds=_BOX, steps=200)
    a = maximize_lml_multistart(GPRegressor(RBF(1.0, 1.0), 0.1), X, y,
                                rng=np.random.default_rng(3), **kw)
    b = maximize_lml_multistart(GPRegressor(RBF(1.0, 1.0), 0.1), X, y,
                                rng=np.random.default_rng(3), **kw)
    assert np.allclose(a.lmls, b.lmls) and np.allclose(a.params, b.params)
    # wrong-shaped bounds are rejected (params here are length 3)
    with pytest.raises(ValueError):
        maximize_lml_multistart(GPRegressor(RBF(1.0, 1.0), 0.1), X, y,
                                bounds=np.log([[1e-2, 1e2]]))
