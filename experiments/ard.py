"""ARD: per-dimension lengthscales discover which inputs matter.

Two inputs, but only x0 drives the response:

    y = sin(2 x0) + 0.3 x0 + noise,     x1 ~ Uniform, entirely irrelevant.

An isotropic RBF must pick *one* lengthscale for both axes; on this data that
is a compromise between the wiggle along x0 and the flatness along x1. The ARD
kernel gives each axis its own l_d, and ML-II has no reason to keep any
dependence on x1 -- covariance structure along a noise axis can only add the
log|K| complexity penalty without improving the fit -- so the optimizer drives
l1 up until the kernel is flat along x1 (inverse lengthscale 1/l1 -> 0). The
model *discovers* that x0 is the only relevant input, and the learned 1/l_d
read out as a relevance ranking (MacKay 1994; Rasmussen & Williams 2006, 5.1).

Run:  python experiments/ard.py
"""

import numpy as np

from common import plt, savefig
from gp.gp import GPRegressor
from gp.kernels import ARD, RBF
from gp.optimize import adam_maximize

SEED = 3


def make_data(rng, n=120):
    X = rng.uniform(-3.0, 3.0, size=(n, 2))
    y = np.sin(2.0 * X[:, 0]) + 0.3 * X[:, 0] + 0.1 * rng.standard_normal(n)
    return X, y


def fit_ard(X, y):
    model = GPRegressor(ARD(s2=1.0, lengthscales=[1.0, 1.0]), noise_var=0.2)
    best, hist = adam_maximize(
        lambda p: model.lml_and_grad(X, y, p), model.params, lr=0.05, steps=500
    )
    model.params = best
    model.fit(X, y)
    return model, hist


def fit_isotropic(X, y):
    model = GPRegressor(RBF(s2=1.0, l=1.0), noise_var=0.2)
    best, _ = adam_maximize(
        lambda p: model.lml_and_grad(X, y, p), model.params, lr=0.05, steps=500
    )
    model.params = best
    model.fit(X, y)
    return model


def main():
    rng = np.random.default_rng(SEED)
    X, y = make_data(rng)

    ard, _ = fit_ard(X, y)
    s2 = np.exp(ard.kernel._theta[0])
    l0, l1 = np.exp(ard.kernel._theta[1:])
    print("=" * 60)
    print("ARD-RBF on y = sin(2 x0) + 0.3 x0 + noise  (x1 irrelevant)")
    print("=" * 60)
    print(f"learned  s2={s2:.3f}  l0={l0:.3f}  l1={l1:.3f}  "
          f"noise={ard.noise_var:.3f}")
    print(f"inverse lengthscales (relevance):  1/l0={1/l0:.3f}  1/l1={1/l1:.3f}")
    print(f"relevance ratio l1/l0 = {l1 / l0:.1f}x  "
          f"(x1 is suppressed: 1/l1 near 0)")
    print(f"ARD  log-evidence: {ard.log_marginal_likelihood():.2f}")

    iso = fit_isotropic(X, y)
    print(f"isotropic-RBF log-evidence: {iso.log_marginal_likelihood():.2f}  "
          f"(ARD should be >= : it can decouple the two axes)")

    # Panel A: inverse-lengthscale (relevance) bars.
    # Panel B: posterior mean vs x0 at three fixed x1 slices -- they coincide,
    #          i.e. predictions ignore x1 (the suppression, made visible).
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)

    axes[0].bar(["x0\n(relevant)", "x1\n(noise)"], [1.0 / l0, 1.0 / l1],
                color=["C3", "C0"])
    axes[0].set_ylabel(r"inverse lengthscale $1/\ell_d$")
    axes[0].set_title("ARD relevance: x1 driven to ~0", loc="left")

    grid0 = np.linspace(-3, 3, 200)
    for x1v, ls in [(-2.0, "-"), (0.0, "--"), (2.0, ":")]:
        Xs = np.column_stack([grid0, np.full_like(grid0, x1v)])
        mean, var = ard.predict(Xs)
        axes[1].plot(grid0, mean, ls, lw=1.6, label=fr"$x_1={x1v:+.0f}$")
    axes[1].plot(grid0, np.sin(2 * grid0) + 0.3 * grid0, color="k", lw=0.8,
                 alpha=0.5, label="truth")
    axes[1].scatter(X[:, 0], y, s=8, alpha=0.3, color="gray", label="data")
    axes[1].set_xlabel(r"$x_0$")
    axes[1].set_ylabel(r"posterior mean")
    axes[1].set_title("Prediction is invariant to $x_1$", loc="left")
    axes[1].legend(fontsize=7, ncol=2)

    fig.suptitle("ARD automatically suppresses an irrelevant input", x=0.02,
                 ha="left")
    savefig(fig, "ard_relevance.png")


if __name__ == "__main__":
    main()
