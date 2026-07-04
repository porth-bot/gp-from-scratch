"""Experiment 3: watch a neural network become a Gaussian process.

Three views of the infinite-width correspondence, all against analytic
formulas (gp/ntk.py, derived in theory/derivations.md Sec. 6-7):

1. **At initialization**: for ONE hidden layer the output covariance equals
   the NNGP kernel 2*kappa1 *exactly at every width* (it is a mean of i.i.d.
   per-neuron terms) -- so the honest convergence statement is about
   GAUSSIANITY, not covariance. We measure excess kurtosis of f(x) across
   an ensemble of inits: for a sum of m i.i.d. terms it must decay exactly
   like 1/m (CLT made quantitative). Deeper networks would also have
   covariance converging; conflating the two is a common sloppiness this
   experiment is careful about.
2. **During training**: a real network trained by full-batch GD stays close
   to the closed-form linearized (NTK) dynamics; the deviation shrinks
   like ~ 1/sqrt(width).
3. **The picture**: one wide trained network overlaid on the NTK kernel
   regression it is secretly performing.

Run:  python experiments/ntk_experiments.py   (~1-2 min)
"""

import numpy as np

from common import plt, savefig
from gp.nn import TwoLayerReLU, _augment
from gp.ntk import gd_prediction, nngp_kernel, ntk_kernel

SEED = 20260703


def ensemble_outputs(width, n_nets, x, rng, chunk=20_000):
    """f(x) at a single input across an ensemble of random inits (chunked)."""
    xa = _augment(np.array([[x]]))[0]
    out = np.empty(n_nets)
    done = 0
    while done < n_nets:
        k = min(chunk, n_nets - done)
        W = rng.standard_normal((k, width, 2))
        a = rng.standard_normal((k, width))
        z = W @ xa                                   # (k, width)
        out[done : done + k] = np.sqrt(2.0 / width) * np.sum(np.maximum(z, 0) * a, axis=1)
        done += k
    return out


def excess_kurtosis(samples):
    c = samples - samples.mean()
    return float(np.mean(c**4) / np.mean(c**2) ** 2 - 3.0)


def nngp_convergence():
    """Covariance is exact at any width here (verified: the m=4 line below is
    already at MC-noise level vs 2*kappa1). Gaussianity is what converges:
    excess kurtosis of the sum of m i.i.d. per-neuron terms is exactly
    kappa_neuron / m."""
    rng = np.random.default_rng(SEED)
    x = 0.8
    n_nets = 400_000
    var_target = nngp_kernel(np.array([[x]]), np.array([[x]]))[0, 0]

    widths = [4, 16, 64, 256]
    kurts, var_errs = [], []
    for m in widths:
        f = ensemble_outputs(m, n_nets, x, rng)
        kurts.append(abs(excess_kurtosis(f)))
        var_errs.append(abs(float(np.var(f)) - var_target) / var_target)
        print(f"width {m:4d}: |excess kurtosis| = {kurts[-1]:.4f}   "
              f"|var - NNGP|/NNGP = {var_errs[-1]:.4f}")

    fig, ax = plt.subplots(figsize=(4.8, 3.4), constrained_layout=True)
    ax.loglog(widths, kurts, "o-", label=f"|excess kurtosis| ({n_nets:,} nets)")
    ref = kurts[0] * (np.array(widths) / widths[0]) ** -1.0
    ax.loglog(widths, ref, "k--", lw=1, label=r"$\propto 1/m$ (CLT rate)")
    ax.loglog(widths, var_errs, "s:", color="C2", alpha=0.8,
              label="|variance error| (exact: MC noise only)")
    ax.set_xlabel("width $m$")
    ax.set_ylabel("deviation from the NNGP limit")
    ax.set_title("What converges is Gaussianity, not covariance", loc="left")
    ax.legend(fontsize=7)
    savefig(fig, "nngp_convergence.png")


def linearization_error():
    X_tr = np.linspace(-2, 2, 12)[:, None]
    y = np.sin(2.0 * X_tr[:, 0])
    X_te = np.linspace(-2.5, 2.5, 60)[:, None]
    lr, steps = 0.05, 600
    widths = [64, 256, 1024, 4096]
    errs = []
    for m in widths:
        per_seed = []
        for seed in range(5):
            rng = np.random.default_rng(1000 + seed)
            net = TwoLayerReLU(m, rng)
            f0_tr, f0_te = net.forward(X_tr), net.forward(X_te)
            for _ in range(steps):
                net.gd_step(X_tr, y, lr)
            lin = gd_prediction(X_tr, y, X_te, f0_tr, f0_te, lr, steps)
            per_seed.append(np.max(np.abs(net.forward(X_te) - lin)))
        errs.append(per_seed)
        print(f"width {m:5d}: max |net - linearized| = {np.mean(per_seed):.4f} "
              f"(+/- {np.std(per_seed):.4f}, 5 seeds)")

    means = [np.mean(e) for e in errs]
    stds = [np.std(e) for e in errs]
    fig, ax = plt.subplots(figsize=(4.6, 3.4), constrained_layout=True)
    ax.errorbar(widths, means, yerr=stds, fmt="o-", capsize=3,
                label="trained net vs NTK dynamics")
    ref = means[0] * (np.array(widths) / widths[0]) ** -0.5
    ax.loglog(widths, ref, "k--", lw=1, label=r"$\propto 1/\sqrt{m}$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("width $m$")
    ax.set_ylabel(r"$\max_x |f_{\mathrm{net}}(x) - f_{\mathrm{lin}}(x)|$")
    ax.set_title("Training stays on the tangent plane", loc="left")
    ax.legend(fontsize=8)
    savefig(fig, "ntk_linearization.png")


def overlay():
    rng = np.random.default_rng(SEED + 7)
    X_tr = np.linspace(-2, 2, 12)[:, None]
    y = np.sin(2.0 * X_tr[:, 0])
    grid = np.linspace(-3, 3, 300)[:, None]
    lr, steps = 0.05, 3000

    net = TwoLayerReLU(4096, rng)
    f0_tr, f0_grid = net.forward(X_tr), net.forward(grid)
    for _ in range(steps):
        net.gd_step(X_tr, y, lr)
    lin = gd_prediction(X_tr, y, grid, f0_tr, f0_grid, lr, steps)

    # infinite-training limit: ridgeless NTK kernel regression (+ f0 transient)
    T_tr = ntk_kernel(X_tr, X_tr)
    T_g = ntk_kernel(grid, X_tr)
    limit = f0_grid + T_g @ np.linalg.solve(T_tr, y - f0_tr)

    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    ax.plot(grid, net.forward(grid), lw=1.8, label="trained net (m=4096, GD)")
    ax.plot(grid, lin, "--", lw=1.4, label="closed-form NTK dynamics (same step)")
    ax.plot(grid, limit, ":", lw=1.4, label=r"NTK kernel regression ($k\to\infty$)")
    ax.plot(X_tr, y, "ko", ms=5, label="train points")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.set_title("A wide network is doing kernel regression", loc="left")
    ax.legend(fontsize=8, loc="lower left")
    savefig(fig, "ntk_overlay.png")


if __name__ == "__main__":
    nngp_convergence()
    linearization_error()
    overlay()
