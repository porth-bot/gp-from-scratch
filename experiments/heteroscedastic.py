"""Heteroscedastic regression via a two-stage GP fit.

A standard GP assumes one noise level everywhere. When the noise varies with
the input, a homoscedastic fit is forced into a single compromise band: it
over-covers where the data are clean and under-covers where they are noisy.

The classic remedy (Goldberg, Williams & Bishop 1998) is a *second* GP that
models the log-noise as a function of x:

1. Fit a homoscedastic signal GP; take its **leave-one-out** residuals
   r_i (closed form, Rasmussen & Williams 5.4.2: r_i = alpha_i / [K^{-1}]_ii).
   LOO residuals are used instead of in-sample residuals because an
   interpolating GP drives in-sample residuals to zero and would badly
   under-estimate the noise.
2. Fit a smooth GP to z_i = log r_i^2. This estimates E[log r^2], which for
   r ~ N(0, s^2) is log s^2 - 1.2704 (since E[log chi^2_1] = psi(1/2)+log 2 =
   -1.2704). Undo that bias: s^2(x) = exp(z_hat(x) + 1.2704).
3. Refit the signal GP with the per-point variances s^2(x_i) on the diagonal
   (GPRegressor.fit(..., noise=...)).

The payoff is calibration, not a lower average error: the heteroscedastic
band tracks the true noise, so nominal 95% intervals cover ~95% *in every
region*, which the homoscedastic band cannot.

Run:  python experiments/heteroscedastic.py
"""

import numpy as np
from common import plt, savefig

from gp.gp import GPRegressor, _chol_solve
from gp.kernels import RBF

# -E[log chi^2_1] = -(digamma(1/2) + log 2); undoes the log-square-residual bias
LOG_CHI2_BIAS = 1.2704


def make_data(rng, n):
    """y = sin(1.3 x) with noise growing linearly left->right across [-3, 3]."""
    X = np.sort(rng.uniform(-3.0, 3.0, size=(n, 1)), axis=0)
    sigma = 0.05 + 0.45 * (X[:, 0] + 3.0) / 6.0
    y = np.sin(1.3 * X[:, 0]) + sigma * rng.standard_normal(n)
    return X, y, sigma


def two_stage_heteroscedastic(X, y, signal_l=0.7, noise_l=1.5, noise_var0=0.08):
    """Return (homo_gp, hetero_gp, noise_gp). ``noise_gp.predict`` gives
    log-noise; recover the noise variance with exp(pred + LOG_CHI2_BIAS)."""
    homo = GPRegressor(RBF(s2=1.0, l=signal_l), noise_var=noise_var0).fit(X, y)

    n = len(y)
    Kinv = _chol_solve(homo.L, np.eye(n))
    loo_resid = homo.alpha / np.diag(Kinv)         # R&W 5.4.2
    z = np.log(loo_resid**2 + 1e-8)

    noise_gp = GPRegressor(RBF(s2=1.0, l=noise_l), noise_var=1.0).fit(X, z)
    noise_tr = np.exp(noise_gp.predict(X)[0] + LOG_CHI2_BIAS)

    hetero = GPRegressor(RBF(s2=1.0, l=signal_l), noise_var=noise_var0).fit(
        X, y, noise=noise_tr
    )
    return homo, hetero, noise_gp


def predictive(gp, noise_gp, Xs):
    """Predictive mean and *observation* variance (latent + modeled noise)."""
    mean, var = gp.predict(Xs)
    if noise_gp is None:
        return mean, var + gp.noise_var
    noise = np.exp(noise_gp.predict(Xs)[0] + LOG_CHI2_BIAS)
    return mean, var + noise


def _coverage(y, mean, var, z=1.96):
    return float(np.mean(np.abs(y - mean) < z * np.sqrt(var)))


def _nll(y, mean, var):
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mean) ** 2 / var))


def evaluate(seed=0, n_train=150, n_test=400):
    rng = np.random.default_rng(seed)
    X, y, _ = make_data(rng, n_train)
    Xte, yte, _ = make_data(rng, n_test)
    homo, hetero, noise_gp = two_stage_heteroscedastic(X, y)

    mh, vh = predictive(homo, None, Xte)
    mm, vm = predictive(hetero, noise_gp, Xte)
    left = Xte[:, 0] < 0
    return {
        "homo_nll": _nll(yte, mh, vh),
        "hetero_nll": _nll(yte, mm, vm),
        "homo_cover": (_coverage(yte[left], mh[left], vh[left]),
                       _coverage(yte[~left], mh[~left], vh[~left])),
        "hetero_cover": (_coverage(yte[left], mm[left], vm[left]),
                         _coverage(yte[~left], mm[~left], vm[~left])),
    }


def main():
    rng = np.random.default_rng(0)
    X, y, sigma = make_data(rng, 150)
    homo, hetero, noise_gp = two_stage_heteroscedastic(X, y)
    grid = np.linspace(-3, 3, 300)[:, None]
    mh, vh = predictive(homo, None, grid)
    mm, vm = predictive(hetero, noise_gp, grid)
    xs = grid[:, 0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True, sharey=True)
    for ax, title, m, v in [
        (axes[0], "Homoscedastic: one noise level", mh, vh),
        (axes[1], "Heteroscedastic: two-stage noise(x)", mm, vm),
    ]:
        ax.scatter(X[:, 0], y, s=8, c="0.5", alpha=0.6, zorder=1)
        ax.plot(xs, m, "C0", lw=1.5, zorder=3)
        sd = np.sqrt(v)
        ax.fill_between(xs, m - 1.96 * sd, m + 1.96 * sd, color="C0", alpha=0.2, zorder=2)
        ax.plot(xs, np.sin(1.3 * xs), "k--", lw=1.0, alpha=0.7, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.suptitle(
        "Noise grows left→right. The homoscedastic band is too wide on the "
        "left and too narrow on the right; the two-stage band tracks it "
        "(dashed = truth, shaded = 95%).",
        fontsize=9,
    )
    savefig(fig, "heteroscedastic.png")

    m = evaluate()
    print(f"\n{'':14s}{'left 95% cover':>16}{'right 95% cover':>16}{'test NLL':>12}")
    print(f"{'homoscedastic':14s}{m['homo_cover'][0]:>16.3f}"
          f"{m['homo_cover'][1]:>16.3f}{m['homo_nll']:>12.3f}")
    print(f"{'heteroscedastic':14s}{m['hetero_cover'][0]:>16.3f}"
          f"{m['hetero_cover'][1]:>16.3f}{m['hetero_nll']:>12.3f}")


if __name__ == "__main__":
    main()
