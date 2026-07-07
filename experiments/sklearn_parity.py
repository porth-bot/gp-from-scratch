"""Honest parity + speed benchmark against scikit-learn.

The test suite already asserts that our posterior mean, std, and log marginal
likelihood match scikit-learn's to 1e-8 at one dataset size. This script makes
that quantitative across sizes and adds the fair-and-square question a
from-scratch NumPy implementation should answer out loud: how much slower is
it than a mature library doing the identical exact-GP math?

Both fit the *same* kernel at the *same* fixed hyperparameters (no optimizer),
so any output difference is pure numerics and any time difference is pure
implementation overhead -- this is a parity check, not a modeling contest.

Run:  python experiments/sklearn_parity.py
"""

import time

import numpy as np
from common import plt, savefig

from gp.gp import GPRegressor
from gp.kernels import RBF

S2, LENGTHSCALE, NOISE = 1.4, 0.8, 0.05
SIZES = (50, 100, 200, 400, 800)


def _make_data(n, rng):
    X = np.sort(rng.uniform(-4, 4, size=(n, 1)), axis=0)
    K = RBF(s2=S2, l=LENGTHSCALE)(X, X) + NOISE * np.eye(n)
    y = np.linalg.cholesky(K + 1e-10 * np.eye(n)) @ rng.standard_normal(n)
    return X, y


def compare(n, rng, n_timing=3):
    """Fit both implementations at n training points; return output
    differences and best-of-``n_timing`` fit+predict wall times (seconds)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF as SkRBF
    from sklearn.gaussian_process.kernels import ConstantKernel

    X, y = _make_data(n, rng)
    Xs = np.linspace(-4, 4, 200)[:, None]

    def time_ours():
        t = time.perf_counter()
        m = GPRegressor(RBF(s2=S2, l=LENGTHSCALE), noise_var=NOISE).fit(X, y)
        mean, var = m.predict(Xs)
        return time.perf_counter() - t, mean, np.sqrt(var)

    def time_sklearn():
        t = time.perf_counter()
        sk = GaussianProcessRegressor(
            kernel=ConstantKernel(S2, "fixed") * SkRBF(LENGTHSCALE, "fixed"),
            alpha=NOISE, optimizer=None,
        ).fit(X, y)
        mean, std = sk.predict(Xs, return_std=True)
        return time.perf_counter() - t, mean, std

    t_ours, mean_o, std_o = min((time_ours() for _ in range(n_timing)), key=lambda r: r[0])
    t_sk, mean_s, std_s = min((time_sklearn() for _ in range(n_timing)), key=lambda r: r[0])
    return {
        "n": n,
        "mean_maxdiff": float(np.max(np.abs(mean_o - mean_s))),
        "std_maxdiff": float(np.max(np.abs(std_o - std_s))),
        "t_ours": t_ours,
        "t_sklearn": t_sk,
    }


def main():
    rng = np.random.default_rng(0)
    rows = [compare(n, rng) for n in SIZES]

    print(f"\n{'n':>5} {'max|Δmean|':>12} {'max|Δstd|':>12} "
          f"{'ours (ms)':>10} {'sklearn (ms)':>13} {'ratio':>7}")
    for r in rows:
        print(f"{r['n']:>5} {r['mean_maxdiff']:>12.2e} {r['std_maxdiff']:>12.2e} "
              f"{1e3*r['t_ours']:>10.1f} {1e3*r['t_sklearn']:>13.1f} "
              f"{r['t_ours']/r['t_sklearn']:>6.2f}x")

    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(4.6, 3.2), constrained_layout=True)
    ax.plot(ns, [1e3 * r["t_ours"] for r in rows], "o-", label="ours (NumPy)")
    ax.plot(ns, [1e3 * r["t_sklearn"] for r in rows], "s-", label="scikit-learn")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("training points n")
    ax.set_ylabel("fit + predict (ms)")
    ax.set_title("Exact-GP parity: identical math, timing overhead only")
    ax.legend()
    savefig(fig, "sklearn_parity.png")


if __name__ == "__main__":
    main()
