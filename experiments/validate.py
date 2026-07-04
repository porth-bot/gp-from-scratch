"""Experiment 1: is the GP's uncertainty real, and does ML-II find the truth?

Three checks, all with known answers:

1. **Calibration across levels.** On functions drawn from a known GP prior,
   the posterior's z-scores must be standard normal: nominal q% intervals
   should cover q% of held-out latent values, for every q, not just 95%.
2. **Hyperparameter recovery.** Data generated from known (s2, l, noise);
   ML-II from a generic init should land near the truth, replicated over
   seeds to show the spread.
3. **The LML surface.** Contours over (log l, log noise) with the Adam path
   overlaid -- the fit/complexity tradeoff made visible.

Run:  python experiments/validate.py
"""

import numpy as np

from common import plt, savefig
from gp.gp import GPRegressor
from gp.kernels import RBF, Matern
from gp.optimize import adam_maximize

SEED = 20260703


def calibration_curve():
    rng = np.random.default_rng(SEED)
    kernel_truth = dict(nu=2.5, s2=2.0, l=1.0)
    grid = np.linspace(-5, 5, 300)[:, None]
    K = Matern(**kernel_truth)(grid, grid) + 1e-10 * np.eye(300)
    Lk = np.linalg.cholesky(K)

    zs = []
    for _ in range(40):
        f = Lk @ rng.standard_normal(300)
        idx = rng.permutation(300)
        train, test = idx[:50], idx[50:150]
        noise = 0.05
        y = f[train] + np.sqrt(noise) * rng.standard_normal(len(train))
        model = GPRegressor(Matern(**kernel_truth), noise_var=noise).fit(grid[train], y)
        mean, var = model.predict(grid[test])
        zs.append((f[test] - mean) / np.sqrt(var))
    z = np.concatenate(zs)

    from math import erf
    nominal = np.linspace(0.05, 0.99, 40)
    # central interval at level q corresponds to |z| < Phi^{-1}((1+q)/2);
    # invert empirically: coverage(q) = P(|z| < c_q)
    c = np.abs(z)
    empirical = []
    for q in nominal:
        # c_q via bisection on the standard normal CDF
        lo, hi = 0.0, 5.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if erf(mid / np.sqrt(2.0)) < q:
                lo = mid
            else:
                hi = mid
        empirical.append(float((c < 0.5 * (lo + hi)).mean()))

    fig, ax = plt.subplots(figsize=(4.2, 4.0), constrained_layout=True)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(nominal, empirical, lw=1.8, label=f"GP posterior (n={len(z)} z-scores)")
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage")
    ax.set_title("Credible intervals mean what they say", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    savefig(fig, "calibration.png")
    print(f"95% nominal -> {float((np.abs(z) < 1.96).mean()):.3f} empirical")


def hyperparameter_recovery():
    true = dict(s2=2.0, l=0.8)
    true_noise = 0.05
    estimates = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        X = np.sort(rng.uniform(-4, 4, size=(150, 1)), axis=0)
        K = RBF(**true)(X, X) + true_noise * np.eye(150)
        y = np.linalg.cholesky(K + 1e-12 * np.eye(150)) @ rng.standard_normal(150)
        model = GPRegressor(RBF(s2=1.0, l=2.0), noise_var=0.5)
        best, _ = adam_maximize(
            lambda p: model.lml_and_grad(X, y, p), model.params, lr=0.05, steps=250
        )
        estimates.append(np.exp(best))
    est = np.array(estimates)  # columns: s2, l, noise

    fig, axes = plt.subplots(1, 3, figsize=(9, 2.9), constrained_layout=True)
    for ax, col, name, truth in zip(
        axes, est.T, [r"$\sigma_f^2$", r"$\ell$", r"$\sigma_n^2$"],
        [true["s2"], true["l"], true_noise],
    ):
        ax.scatter(np.zeros_like(col), col, s=18, alpha=0.7)
        ax.axhline(truth, color="k", ls="--", lw=1, label="truth")
        ax.set_xticks([])
        ax.set_title(name)
        ax.set_yscale("log")
    axes[0].legend(fontsize=8)
    fig.suptitle("ML-II estimates across 8 replicate datasets (log scale)", y=1.05)
    savefig(fig, "hyperparam_recovery.png")
    print("median estimates (s2, l, noise):", np.round(np.median(est, axis=0), 3),
          "| truth:", (true["s2"], true["l"], true_noise))


def lml_surface():
    rng = np.random.default_rng(SEED + 1)
    X = np.sort(rng.uniform(-4, 4, size=(120, 1)), axis=0)
    true_kernel = RBF(s2=1.5, l=0.7)
    K = true_kernel(X, X) + 0.05 * np.eye(120)
    y = np.linalg.cholesky(K + 1e-12 * np.eye(120)) @ rng.standard_normal(120)

    model = GPRegressor(RBF(s2=1.5, l=3.0), noise_var=1.0)  # s2 held at truth-ish
    path = []
    best, _ = adam_maximize(
        lambda p: model.lml_and_grad(X, y, p),
        model.params, lr=0.06, steps=200,
        callback=lambda t, th, v: path.append(th.copy()),
    )
    path = np.array(path)

    log_l = np.linspace(-2.2, 1.8, 60)
    log_n = np.linspace(-5.5, 1.0, 60)
    Z = np.empty((len(log_n), len(log_l)))
    probe = GPRegressor(RBF(s2=1.5, l=1.0), noise_var=0.1)
    for i, ln_ in enumerate(log_n):
        for j, ll_ in enumerate(log_l):
            probe.params = np.array([np.log(1.5), ll_, ln_])
            probe.fit(X, y)
            Z[i, j] = probe.log_marginal_likelihood()

    fig, ax = plt.subplots(figsize=(5.4, 4.0), constrained_layout=True)
    levels = np.quantile(Z, np.linspace(0.55, 1.0, 24))
    cs = ax.contourf(log_l, log_n, Z, levels=levels, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="log marginal likelihood")
    ax.plot(path[:, 1], path[:, 2], "w.-", ms=3, lw=1, label="Adam path")
    ax.plot(np.log(0.7), np.log(0.05), "r*", ms=12, label="truth")
    ax.set_xlabel(r"$\log \ell$")
    ax.set_ylabel(r"$\log \sigma_n^2$")
    ax.set_title("Evidence surface: fit vs complexity", loc="left")
    ax.legend(loc="lower right", fontsize=8)
    savefig(fig, "lml_surface.png")


if __name__ == "__main__":
    calibration_curve()
    hyperparameter_recovery()
    lml_surface()
