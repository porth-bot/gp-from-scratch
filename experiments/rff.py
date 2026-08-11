"""Random Fourier features: what the O(n^3) -> O(n D^2) trade actually costs.

Four measurements, in the order the trade should be interrogated:

1. **Does the kernel converge, and how fast?** Max |k_D - k| over a grid,
   against a D^{-1/2} reference line. The rate is the Monte Carlo rate, so
   every factor of 4 in features buys a factor of 2 in accuracy -- an
   unforgiving exchange rate, and the reason RFF is a *scaling* tool rather
   than an *accuracy* tool.

2. **Does the posterior converge?** Max |mean_RFF - mean_exact| over a test
   grid at the same hyperparameters, so the only difference between the two
   models is the feature approximation.

3. **Is it actually faster?** Wall-clock for fit + predict as n grows, exact
   GP versus a fixed feature budget. The exact GP's Cholesky is n^3/3; RFF's
   is D^3/3 plus an n D^2 Gram accumulation, so the curves have different
   slopes on a log-log plot and cross.

4. **What breaks?** Predictive uncertainty across a gap in the data. This is
   where the approximation stops being a rounding error: with n >> D the
   posterior variance collapses everywhere, and the model is confidently
   wrong exactly where an exact GP would say it does not know.

Run:  python experiments/rff.py     (~1 min)
"""

import time

import numpy as np

from common import savefig
import matplotlib.pyplot as plt

from gp.gp import GPRegressor
from gp.kernels import RBF
from gp.rff import RFFMap, RFFRegressor

S2 = 1.0
LENGTHSCALE = 0.5
NOISE_VAR = 0.01
N_SEEDS = 9


def dataset(n, rng, gap=None):
    """n noisy samples of a wiggly function on [-4, 4], optionally with a gap."""
    x = rng.uniform(-4.0, 4.0, size=int(n * 1.6))
    if gap is not None:
        x = x[np.abs(x) > gap]
    x = np.sort(x[:n]).reshape(-1, 1)
    y = np.sin(2.0 * x[:, 0]) + 0.5 * np.cos(0.7 * x[:, 0])
    y = y + np.sqrt(NOISE_VAR) * rng.standard_normal(len(x))
    return x, y


# -- 1. kernel approximation error -------------------------------------------


def kernel_error_vs_D(Ds):
    X = np.linspace(-4, 4, 60).reshape(-1, 1)
    exact = RBF(s2=S2, l=LENGTHSCALE)(X, X)
    out = []
    for D in Ds:
        errs = []
        for seed in range(N_SEEDS):
            phi = RFFMap(D, LENGTHSCALE, S2, 1, np.random.default_rng(seed))
            errs.append(float(np.abs(phi(X, X) - exact).max()))
        out.append((float(np.median(errs)), float(np.min(errs)), float(np.max(errs))))
    return np.array(out)


# -- 2. posterior mean error --------------------------------------------------


def posterior_error_vs_D(Ds, n=800):
    rng = np.random.default_rng(0)
    X, y = dataset(n, rng)
    Xs = np.linspace(-4, 4, 300).reshape(-1, 1)
    exact = GPRegressor(RBF(s2=S2, l=LENGTHSCALE), noise_var=NOISE_VAR).fit(X, y)
    mean_exact, var_exact = exact.predict(Xs)
    sd_exact = np.sqrt(var_exact)

    rows = []
    for D in Ds:
        mean_errs, sd_errs = [], []
        for seed in range(N_SEEDS):
            phi = RFFMap(D, LENGTHSCALE, S2, 1, np.random.default_rng(seed))
            m, v = RFFRegressor(phi, noise_var=NOISE_VAR).fit(X, y).predict(Xs)
            mean_errs.append(float(np.abs(m - mean_exact).max()))
            sd_errs.append(float(np.abs(np.sqrt(v) - sd_exact).max()))
        rows.append((float(np.median(mean_errs)), float(np.min(mean_errs)),
                     float(np.max(mean_errs)), float(np.median(sd_errs))))
    return np.array(rows), float(np.abs(mean_exact).max())


# -- 3. wall clock ------------------------------------------------------------


def timing_vs_n(ns, D=512, repeats=3):
    """Fit + predict wall-clock, exact GP vs RFF, at a fixed feature budget."""
    Xs = np.linspace(-4, 4, 200).reshape(-1, 1)
    rows = []
    for n in ns:
        rng = np.random.default_rng(1)
        X, y = dataset(n, rng)

        t_exact = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            m = GPRegressor(RBF(s2=S2, l=LENGTHSCALE), noise_var=NOISE_VAR).fit(X, y)
            mean_exact, _ = m.predict(Xs)
            t_exact.append(time.perf_counter() - t0)

        t_rff = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            phi = RFFMap(D, LENGTHSCALE, S2, 1, np.random.default_rng(0))
            r = RFFRegressor(phi, noise_var=NOISE_VAR).fit(X, y)
            mean_rff, _ = r.predict(Xs)
            t_rff.append(time.perf_counter() - t0)

        err = float(np.abs(mean_rff - mean_exact).max())
        rows.append((min(t_exact), min(t_rff), err))
        print(f"    n={n:5d}  exact {min(t_exact):7.3f}s   "
              f"RFF(D={D}) {min(t_rff):7.3f}s   "
              f"speedup {min(t_exact) / min(t_rff):5.1f}x   "
              f"mean err {err:.3e}")
    return np.array(rows)


# -- 4. variance starvation ---------------------------------------------------


def starvation(n=3000, gap=1.2):
    """Posterior across a gap, over ``N_SEEDS`` draws of the feature map.

    One draw is not a measurement here. The frequencies are random, and at
    D = 64 how much error bar survives depends heavily on which eight-ish
    frequencies happened to come up: across nine draws the posterior sd at the
    gap centre spans 0.088 to 0.635. This function therefore fits every draw
    and returns the whole spread, plus the *median* draw for plotting -- the
    figure used to show ``default_rng(0)``, which turned out to be the worst of
    the nine, so the number the README quoted was a tail and not a typical fit.
    The qualitative claim survives (even the best draw is 34% below exact); the
    headline digits did not.
    """
    rng = np.random.default_rng(2)
    X, y = dataset(n, rng, gap=gap)
    Xs = np.linspace(-4, 4, 400).reshape(-1, 1)
    exact = GPRegressor(RBF(s2=S2, l=LENGTHSCALE), noise_var=NOISE_VAR).fit(X, y)
    out = {"X": X, "y": y, "Xs": Xs, "gap": gap, "exact": exact.predict(Xs)}
    centre = int(np.argmin(np.abs(Xs.ravel())))
    for D in (64, 2048):
        draws = []
        for seed in range(N_SEEDS):
            phi = RFFMap(D, LENGTHSCALE, S2, 1, np.random.default_rng(seed))
            draws.append(RFFRegressor(phi, noise_var=NOISE_VAR).fit(X, y).predict(Xs))
        sds = np.array([np.sqrt(v[centre]) for _, v in draws])
        out[D] = draws[int(np.argsort(sds)[len(sds) // 2])]      # median draw
        out[(D, "sd_spread")] = (float(np.median(sds)), float(sds.min()),
                                 float(sds.max()))
    return out


def main():
    Ds = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096])

    print("1. kernel approximation  (max |k_D - k| over a 60-point grid, "
          f"{N_SEEDS} seeds)")
    kerr = kernel_error_vs_D(Ds)
    for D, (med, lo, hi) in zip(Ds, kerr):
        print(f"    D={D:5d}   median {med:.4f}   [{lo:.4f}, {hi:.4f}]")
    print(f"    error ratio D=16 -> D=4096 (256x features): "
          f"{kerr[0, 0] / kerr[-1, 0]:.1f}x  (D^-1/2 predicts 16x)")

    print("\n2. posterior agreement with the exact GP  (n=800, same "
          "hyperparameters)")
    perr, signal = posterior_error_vs_D(Ds)
    print(f"    posterior mean spans +-{signal:.2f}")
    for D, (med, lo, hi, sderr) in zip(Ds, perr):
        print(f"    D={D:5d}   mean err {med:.4f} [{lo:.4f}, {hi:.4f}]   "
              f"sd err {sderr:.4f}")

    print("\n3. wall-clock, fit + predict")
    ns = np.array([500, 1000, 2000, 4000, 8000])
    tim = timing_vs_n(ns)

    print("\n4. variance starvation across a gap in the data")
    star = starvation()
    Xs = star["Xs"]
    xs = Xs.ravel()
    in_gap = np.abs(xs) < star["gap"]
    centre = int(np.argmin(np.abs(xs)))
    mean_exact, var_exact = star["exact"]
    sd_exact = np.sqrt(var_exact[centre])
    print(f"    exact GP posterior sd at the gap centre: {sd_exact:.3f}")
    for D in (64, 2048):
        med, lo, hi = star[(D, "sd_spread")]
        print(f"    RFF D={D:5d}: sd at the centre, median of {N_SEEDS} draws "
              f"{med:.3f} [{lo:.3f}, {hi:.3f}]   "
              f"({med / sd_exact:.2f}x the exact sd)")
    # Is the mean "still fine" in the gap? Split the error by region rather
    # than reporting one max over the whole domain, which is dominated by the
    # 70% of it that has data in it and hides the answer.
    print("    max |mean_D - mean_exact|, split by region (median draw):")
    for D in (64, 2048):
        err = np.abs(star[D][0] - mean_exact)
        print(f"    RFF D={D:5d}: in the gap {err[in_gap].max():.3f}   "
              f"where the data are {err[~in_gap].max():.3f}   "
              f"(ratio {err[in_gap].max() / err[~in_gap].max():.0f}x)")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(Ds, kerr[:, 0], "o-", color="C0", lw=1.6, ms=4, label="max |k_D - k|")
    ax.fill_between(Ds, kerr[:, 1], kerr[:, 2], color="C0", alpha=0.15,
                    label=f"min-max over {N_SEEDS} seeds")
    ref = kerr[0, 0] * np.sqrt(Ds[0] / Ds)
    ax.plot(Ds, ref, "--", color="0.4", lw=1.2, label="$D^{-1/2}$ reference")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("features $D$")
    ax.set_ylabel("kernel error")
    ax.set_title("1. The kernel converges at the Monte Carlo rate", loc="left")
    ax.legend(fontsize=7.5)

    ax = axes[0, 1]
    ax.plot(Ds, perr[:, 0], "o-", color="C2", lw=1.6, ms=4,
            label="posterior mean")
    ax.fill_between(Ds, perr[:, 1], perr[:, 2], color="C2", alpha=0.15)
    ax.plot(Ds, perr[:, 3], "s--", color="C3", lw=1.4, ms=3.5,
            label="posterior sd")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("features $D$")
    ax.set_ylabel("max deviation from exact GP")
    ax.set_title("2. So does the posterior (n = 800)", loc="left")
    ax.legend(fontsize=7.5)

    ax = axes[1, 0]
    ax.plot(ns, tim[:, 0], "o-", color="C0", lw=1.6, ms=4, label="exact GP  $O(n^3)$")
    ax.plot(ns, tim[:, 1], "o-", color="C1", lw=1.6, ms=4,
            label="RFF, $D=512$  $O(nD^2)$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.set_xlabel("training points $n$")
    ax.set_ylabel("fit + predict, seconds")
    ax.set_title("3. Different slopes, so the curves cross", loc="left")
    ax.legend(fontsize=7.5)

    ax = axes[1, 1]
    xs = star["Xs"].ravel()
    ax.plot(star["X"].ravel(), star["y"], ".", color="0.75", ms=1.2, zorder=0)
    for key, color, label, ls in ((2048, "C2", "RFF $D=2048$", "-"),
                                  (64, "C3", "RFF $D=64$", "-"),
                                  ("exact", "k", "exact GP", "--")):
        mean, var = star[key]
        sd = np.sqrt(var)
        ax.fill_between(xs, mean - 1.96 * sd, mean + 1.96 * sd,
                        color=color, alpha=0.16, lw=0)
        ax.plot(xs, mean, color=color, lw=1.4, ls=ls, label=label, zorder=3)
    m64, lo64, hi64 = star[(64, "sd_spread")]
    m2k, lo2k, hi2k = star[(2048, "sd_spread")]
    ax.annotate(f"posterior sd at $x=0$, median of {N_SEEDS} feature draws:\n"
                f"exact {sd_exact:.2f}   "
                f"$D$=2048 {m2k:.2f} [{lo2k:.2f}, {hi2k:.2f}]   "
                f"$D$=64 {m64:.2f} [{lo64:.2f}, {hi64:.2f}]",
                xy=(0.5, 0.03), xycoords="axes fraction", ha="center",
                fontsize=7.5, color="0.25")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("4. Variance starvation: $n=3000$, gap in $|x|<1.2$", loc="left")
    ax.legend(fontsize=7.5, loc="upper center", ncol=3)

    savefig(fig, "rff.png")
    return kerr, perr, tim


if __name__ == "__main__":
    main()
