"""FITC against VFE: the same code, one flag, and a failure that is not shared.

Section 9.7 derives FITC as the Titsias factorization with the shortfall moved:
where VFE subtracts `tr(Kff - Qff)` as a penalty, FITC puts its diagonal
`Lambda = diag(Kff - Qff)` back inside the covariance and maximizes an ordinary
log evidence. `gp/sparse.py` implements them as one class, so everything below
compares two objectives rather than two codebases.

The claim under test is the standard one (Bauer, van der Wilk & Rasmussen 2016):
Lambda is free heteroscedastic noise, so FITC can explain misfit point by point
instead of raising `sigma^2` -- and therefore fits a noise variance below the
truth and reports error bars that are too narrow. VFE cannot: its penalty is
paid in units of `sigma^2`, so unexplained variation always pushes the fitted
noise up.

Ground truth is free at this size. The design is 150 points in six tight
clusters, `f(x) = sin(2x) + 0.4 cos(5x)`, observation noise `sigma^2 = 0.09`
known by construction, and an exact GP at its own ML-II fit as the reference
posterior. Everything is five data replicates, because a single replicate cannot
tell a bias from a seed.

Five measurements, all as replicate means over the five seeds:

1. **Fitted noise variance against M.** The headline, and the claim holds. On
   the clumped design FITC lands at 0.028-0.052, i.e. 0.31x to 0.58x the true
   0.09, at every M and never trending toward it. VFE lands at 0.109 (M=6)
   falling to 0.0918, which is the exact GP's own ML-II value to four decimals
   -- so by M=20 the approximation has stopped costing anything at all, and the
   whole of its small-M error is on the conservative side.
2. **The shortfall each method tolerates.** VFE drives `tr(Kff - Qff)` from 0.45
   to 0.000 over the sweep. FITC's *rises*, 8.7 -> 10.4 at M=20, and is still
   5.5 at M=40 where VFE's is zero -- four orders of magnitude apart on the same
   data with the same kernel. FITC is not failing to reduce the shortfall. It
   has no reason to, and its Z go somewhere else -- panel 4 shows one of them
   leaving the data entirely, at x = 4.7 with the data ending at 3.2, which is
   a thing a variational parameter cannot usefully do and a model parameter can.
3. **Calibration on held-out data** from the same design: predictive sd relative
   to the exact GP, 95% interval coverage, and NLPD. FITC is the overconfident
   one where the data are -- 0.956x the exact sd at M=20, coverage 0.880 against
   a nominal 0.95, NLPD 0.611 against the exact GP's 0.266. VFE tracks the exact
   GP: coverage 0.959-0.960 against its 0.959, and sd ratio 1.000 at M >= 20.
4. **The same statistic on a uniform grid, where the sign flips.** At M=20
   FITC's sd is 0.956x the exact GP's on held-out data and 1.215x averaged over
   the axis, because between the clusters its variance blows back up past the
   truth. Over-wide in the gaps and over-narrow on the data is one failure, not
   two: Lambda replacing a noise level it has no reason to get right. Reporting
   only the grid average would have hidden the half that hurts.
5. **A uniform design as the control**, to separate the method from the data.
   The bias survives -- FITC at 0.042-0.067 -- but the damage does: held-out
   coverage recovers to 0.911-0.935 and the sd ratio never falls below 0.99.
   The pathology needs somewhere cheap to hide misfit, and clustered inputs are
   that.

Two honest qualifications, both about the noise result.

*The collapse to zero is a tail, not the typical fit.* "FITC drives sigma^2 to
zero" is how this is usually described, and the cell means above show a 1.6x to
3.2x bias instead. The collapse is nonetheless real and it is in these runs: 2
of the 20 clumped fits land below a fifth of the true noise, the worst at
0.00129, which is 1.4% of it. Same mechanism, heavy tail. Both numbers are
printed rather than one.

*FITC's noise is not monotone in M and the seed spread is large* (+- 0.018 at
M=20, against VFE's +- 0.009). Reading a trend into the four M values would be
reading noise; what is stable across every seed and every M is the sign.

Run:  python experiments/fitc.py     (~3 min)

References
----------
Snelson & Ghahramani (2006), Sparse GPs using pseudo-inputs.
Bauer, van der Wilk & Rasmussen (2016), Understanding probabilistic sparse
Gaussian process approximations.
"""

import time

import numpy as np

from common import savefig
import matplotlib.pyplot as plt

from gp.gp import GPRegressor
from gp.kernels import RBF
from gp.optimize import maximize_elbo, maximize_lml_multistart
from gp.sparse import SGPR

TRUE_NOISE = 0.09                      # sigma^2 = 0.3^2, known by construction
LO, HI = -3.0, 3.0
M_GRID = (6, 10, 20, 40)
SEEDS = (1, 2, 3, 4, 5)
INIT = dict(s2=1.0, l=1.0)
INIT_NOISE = 0.2
LR, STEPS = 0.05, 1500
METHODS = ("vfe", "fitc")


def truth(X):
    return np.sin(2.0 * X).ravel() + 0.4 * np.cos(5.0 * X).ravel()


def clumped(seed, n_per=25):
    """Six tight clusters of 25. The design the pathology needs.

    Lambda is large wherever the inducing set is thin, so clustered inputs give
    FITC somewhere cheap to charge its misfit: serve a few clusters exactly and
    let the diagonal absorb the rest. Measurement 5 runs the uniform control.
    """
    rng = np.random.default_rng(seed)
    centres = np.linspace(LO, HI, 6)
    X = np.sort(np.concatenate(
        [c + 0.12 * rng.standard_normal(n_per) for c in centres]
    )).reshape(-1, 1)
    return X, truth(X) + np.sqrt(TRUE_NOISE) * rng.standard_normal(len(X))


def uniform(seed, n=150):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(LO, HI, n)).reshape(-1, 1)
    return X, truth(X) + np.sqrt(TRUE_NOISE) * rng.standard_normal(n)


def quantile_Z(X, M):
    return np.quantile(X[:, 0], np.linspace(0.02, 0.98, M)).reshape(-1, 1)


def fit_exact(X, y, seed):
    model = GPRegressor(RBF(**INIT), noise_var=INIT_NOISE)
    maximize_lml_multistart(model, X, y, n_restarts=4,
                            rng=np.random.default_rng(seed), steps=400)
    return model


def calibration(model, Xt, yt, var_ref):
    """Predictive sd relative to the reference, 95% coverage, and NLPD.

    Observation-level bands (`include_noise=True`) throughout: the quantity the
    two methods disagree about is sigma^2, and a latent band hides it. Coverage
    is of the interval a user would actually draw, and NLPD scores the whole
    predictive density rather than its width alone -- a model can be narrow and
    lucky on one and not the other.
    """
    mean, var = model.predict(Xt, include_noise=True)
    z = (yt - mean) / np.sqrt(var)
    return (
        float(np.mean(np.sqrt(var)) / np.mean(np.sqrt(var_ref))),
        float(np.mean(np.abs(z) < 1.959964)),
        float(np.mean(0.5 * np.log(2.0 * np.pi * var) + 0.5 * z**2)),
    )


def sweep(design, label):
    """One design, every (seed, M, method). Returns a dict of stacked arrays."""
    grid = np.linspace(LO - 0.5, HI + 0.5, 300).reshape(-1, 1)
    shape = (len(SEEDS), len(M_GRID))
    out = {f"{m}_{k}": np.zeros(shape)
           for m in METHODS
           for k in ("noise", "trace", "sd", "cov", "nlpd", "grid_sd")}
    out["exact_noise"] = np.zeros(len(SEEDS))
    out["exact_cov"] = np.zeros(len(SEEDS))
    out["exact_nlpd"] = np.zeros(len(SEEDS))

    for i, seed in enumerate(SEEDS):
        X, y = design(seed)
        Xt, yt = design(seed + 100)             # held-out, same design
        exact = fit_exact(X, y, seed)
        _, var_ref = exact.predict(Xt, include_noise=True)
        _, var_ref_grid = exact.predict(grid, include_noise=True)
        out["exact_noise"][i] = np.exp(exact.params[-1])
        _, out["exact_cov"][i], out["exact_nlpd"][i] = calibration(
            exact, Xt, yt, var_ref
        )

        for j, M in enumerate(M_GRID):
            Z0 = quantile_Z(X, M)
            for method in METHODS:
                model = SGPR(RBF(**INIT), Z=Z0.copy(), noise_var=INIT_NOISE,
                             method=method)
                res = maximize_elbo(model, X, y, lr=LR, steps=STEPS)
                sd, cov, nlpd = calibration(model, Xt, yt, var_ref)
                _, var_grid = model.predict(grid, include_noise=True)
                out[f"{method}_noise"][i, j] = np.exp(res.params[-1])
                out[f"{method}_trace"][i, j] = res.trace_term
                out[f"{method}_sd"][i, j] = sd
                out[f"{method}_cov"][i, j] = cov
                out[f"{method}_nlpd"][i, j] = nlpd
                out[f"{method}_grid_sd"][i, j] = float(
                    np.mean(np.sqrt(var_grid)) / np.mean(np.sqrt(var_ref_grid))
                )
        print(f"  [{label}] seed {seed}: exact sigma^2 "
              f"{out['exact_noise'][i]:.4f}, "
              f"vfe {out['vfe_noise'][i]}, fitc {out['fitc_noise'][i]}")
    return out


def report(res, label):
    print(f"\n--- {label} ---")
    print(f"true sigma^2 = {TRUE_NOISE:.3f};  exact-GP ML-II "
          f"{res['exact_noise'].mean():.4f} +- {res['exact_noise'].std():.4f}"
          f"  (held-out coverage {res['exact_cov'].mean():.3f}, "
          f"NLPD {res['exact_nlpd'].mean():.3f})")
    head = ("   M | fitted sigma^2      | tr(Kff-Qff) | sd/exact | cov  | NLPD "
            "| grid sd/exact")
    print(head)
    print("-" * len(head))
    for j, M in enumerate(M_GRID):
        for method in METHODS:
            n = res[f"{method}_noise"][:, j]
            row = (f"{M:4d} | {method:4s} {n.mean():.4f} +- {n.std():.4f} "
                   f"| {res[f'{method}_trace'][:, j].mean():11.3f} "
                   f"| {res[f'{method}_sd'][:, j].mean():8.3f} "
                   f"| {res[f'{method}_cov'][:, j].mean():.3f}"
                   f"| {res[f'{method}_nlpd'][:, j].mean():.3f}"
                   f"| {res[f'{method}_grid_sd'][:, j].mean():.3f}")
            print(row)

    # The tail, counted rather than eyeballed. "FITC drives sigma^2 to zero" is
    # the usual description and the cell means above do not show it; the
    # per-replicate minima do. Both are the same mechanism -- Lambda absorbing
    # the misfit -- and only the tail is dramatic, so both get reported.
    for method in METHODS:
        noise = res[f"{method}_noise"]
        collapsed = noise < 0.2 * TRUE_NOISE
        print(f"{method:4s}: min over all {noise.size} fits "
              f"{noise.min():.5f} ({noise.min() / TRUE_NOISE:.1%} of the truth); "
              f"{int(collapsed.sum())} below 20% of it")


def panel_fits(ax, seed=1, M=20):
    """One replicate's bands, so the summary statistics have a picture."""
    X, y = clumped(seed)
    grid = np.linspace(LO - 0.6, HI + 0.6, 400).reshape(-1, 1)
    xs = grid.ravel()
    exact = fit_exact(X, y, seed)
    mean_e, var_e = exact.predict(grid, include_noise=True)
    sd_e = np.sqrt(var_e)

    ax.plot(X[:, 0], y, ".", color="0.6", ms=2.6, zorder=0)
    ax.fill_between(xs, mean_e - 1.96 * sd_e, mean_e + 1.96 * sd_e,
                    color="k", alpha=0.10, lw=0)
    ax.plot(xs, mean_e, "k--", lw=1.2, zorder=4,
            label=f"exact GP, $\\sigma^2$={np.exp(exact.params[-1]):.3f}")

    for method, colour in zip(METHODS, ("C0", "C3")):
        model = SGPR(RBF(**INIT), Z=quantile_Z(X, M), noise_var=INIT_NOISE,
                     method=method)
        res = maximize_elbo(model, X, y, lr=LR, steps=STEPS)
        mean, var = model.predict(grid, include_noise=True)
        sd = np.sqrt(var)
        ax.plot(xs, mean + 1.96 * sd, color=colour, lw=1.1)
        ax.plot(xs, mean - 1.96 * sd, color=colour, lw=1.1,
                label=f"{method.upper()}, $\\sigma^2$={np.exp(res.params[-1]):.3f}")
        ax.plot(res.Z.ravel(), np.full(M, -2.05 if method == "vfe" else -2.35),
                "^", color=colour, ms=3.5)

    ax.set_ylim(-2.6, 3.2)
    ax.set_xlabel("x        ($\\triangle$ = learned $Z$)")
    ax.set_ylabel("y")
    ax.set_title(f"4. One replicate, $M={M}$: 95% observation band edges",
                 loc="left")
    ax.legend(fontsize=6.5, loc="upper center", ncol=3)


def figure(clump, unif):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), constrained_layout=True)
    Ms = np.array(M_GRID, dtype=float)

    def band(ax, res, key, method, colour, label, ls="-"):
        mean = res[f"{method}_{key}"].mean(axis=0)
        sd = res[f"{method}_{key}"].std(axis=0)
        ax.plot(Ms, mean, ls, color=colour, marker="o", ms=3.5, lw=1.4,
                label=label)
        ax.fill_between(Ms, mean - sd, mean + sd, color=colour, alpha=0.15, lw=0)

    ax = axes[0, 0]
    ax.axhline(TRUE_NOISE, color="k", lw=1.0, ls=":", label="true $\\sigma^2$")
    ax.axhline(clump["exact_noise"].mean(), color="0.45", lw=1.0,
               label="exact GP ML-II")
    band(ax, clump, "noise", "vfe", "C0", "VFE")
    band(ax, clump, "noise", "fitc", "C3", "FITC")
    ax.set_xscale("log")
    ax.set_xticks(M_GRID)
    ax.set_xticklabels(M_GRID)
    ax.minorticks_off()
    ax.set_xlabel("inducing points $M$")
    ax.set_ylabel("fitted $\\sigma^2$")
    ax.set_title("1. The noise the two methods think they see", loc="left")
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    band(ax, clump, "trace", "vfe", "C0", "VFE")
    band(ax, clump, "trace", "fitc", "C3", "FITC")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(M_GRID)
    ax.set_xticklabels(M_GRID)
    ax.minorticks_off()
    ax.set_xlabel("inducing points $M$")
    ax.set_ylabel("$\\mathrm{tr}(K_{ff} - Q_{ff})$ at the fit")
    ax.set_title("2. The shortfall only one of them pays for", loc="left")
    ax.legend(fontsize=7)

    ax = axes[1, 0]
    ax.axhline(1.0, color="k", lw=1.0, ls=":", label="exact GP")
    band(ax, clump, "sd", "vfe", "C0", "VFE, on held-out data")
    band(ax, clump, "sd", "fitc", "C3", "FITC, on held-out data")
    band(ax, clump, "grid_sd", "fitc", "C3", "FITC, on a uniform grid", ls="--")
    band(ax, unif, "sd", "fitc", "C1", "FITC, uniform design", ls="-.")
    ax.set_xscale("log")
    ax.set_xticks(M_GRID)
    ax.set_xticklabels(M_GRID)
    ax.minorticks_off()
    ax.set_xlabel("inducing points $M$")
    ax.set_ylabel("mean predictive sd / exact GP's")
    ax.set_title("3. Too narrow on the data, too wide between", loc="left")
    ax.legend(fontsize=6.5)

    panel_fits(axes[1, 1])
    savefig(fig, "fitc.png")


def main():
    t0 = time.time()
    print("FITC vs VFE, five replicates per cell "
          f"(true sigma^2 = {TRUE_NOISE})\n")
    clump = sweep(clumped, "clumped")
    unif = sweep(uniform, "uniform")
    report(clump, "clumped design (six clusters of 25)")
    report(unif, "uniform design (n = 150), the control")
    figure(clump, unif)
    print(f"\ntotal {time.time() - t0:.0f}s")
    return clump, unif


if __name__ == "__main__":
    main()
