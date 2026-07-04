"""GP regression: LML gradient check, sklearn oracle, interpolation, calibration."""

import numpy as np

from gp.gp import GPRegressor
from gp.kernels import RBF, Matern
from gp.optimize import adam_maximize


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
