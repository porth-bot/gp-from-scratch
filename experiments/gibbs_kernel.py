"""The Gibbs kernel: when one lengthscale is not enough.

Every other kernel in this repo is stationary -- ``k(x, x')`` depends only on
``x - x'``, so a single lengthscale governs the entire input space. That is a
modeling *assumption*, and this experiment is about what it costs when the
assumption is false.

The target is a chirp, ``y = sin(x^3) + noise`` on ``x in [0, 2.2]``. Its local
wavelength shrinks monotonically: the instantaneous frequency is
``d(x^3)/dx = 3x^2``, which is 0.75 rad/unit at x = 0.5 (wavelength ~8) and
14.5 rad/unit at x = 2.2 (wavelength ~0.43). So the function is nearly flat on
the left and rapidly oscillating on the right -- a ~20x range of lengthscale
inside one dataset.

A stationary RBF has to pick one number for that, and whatever it picks is
wrong somewhere. The interesting question is *how* it goes wrong, and the
answer is visible in the posterior: a lengthscale short enough to track the
right-hand oscillations makes the model needlessly uncertain on the smooth
left, and one long enough to be confident on the left smooths the right-hand
structure away entirely.

The Gibbs (1997) kernel lets the lengthscale vary with the input,

    k(x, x') = s2 sqrt( 2 l(x) l(x') / (l(x)^2 + l(x')^2) )
                  exp( -(x - x')^2 / (l(x)^2 + l(x')^2) ),

which is PSD for any positive ``l(.)``. Here ``l(x) = exp(a + b x)``, so the
tilt ``b`` is learned by ML-II along with everything else -- nothing about the
chirp is told to the model. A monotone lengthscale is exactly the right shape
for a chirp; it would be the wrong shape for, say, a single localized bump, and
the README says so.

What is reported: the learned tilt and the lengthscale range it implies, the
log evidence of both models, and held-out accuracy split into the smooth and
rough halves of the domain -- because the *aggregate* numbers hide the failure,
which is regional.

An honest wrinkle worth stating up front, because measuring it is the point of
the density sweep at the end. The evidence prefers Gibbs at every sample size
tried (+5 to +11 nats), which is the model-selection question and it is not
close. But the *predictive* advantage nearly vanishes when the data is dense:
with 140 points the two models' held-out errors are identical to three decimals.
The reason is that a stationary RBF is not helpless -- it copes by choosing the
short lengthscale that the rough region demands (l = 0.19, versus the 1.65 the
smooth region wants) and then leaning on sheer data density to interpolate the
smooth half anyway. That crutch is only available while the data is dense. Thin
it out and the stationary model has to fall back on its (wrong, far too short)
correlation range, and the gap opens: at n = 40 the Gibbs kernel cuts
smooth-region RMSE by ~18%. So the correct claim is not "nonstationary kernels
predict better", it is "a stationary kernel pays for its wrong lengthscale in
the currency of data, and you notice when data is what you are short of".

Run:  python experiments/gibbs_kernel.py     # ~15 s
"""

from __future__ import annotations

import numpy as np

from common import plt, savefig
from gp.gp import GPRegressor
from gp.kernels import RBF, Gibbs
from gp.optimize import adam_maximize

SEED = 0
X_LO, X_HI = 0.0, 2.2
SPLIT = 1.4  # x below this is the "smooth" region, above it the "rough" one
NOISE_SD = 0.1


def truth(x):
    return np.sin(x ** 3)


def make_data(rng, n=140):
    X = rng.uniform(X_LO, X_HI, size=(n, 1))
    y = truth(X[:, 0]) + NOISE_SD * rng.standard_normal(n)
    return X, y


def fit(kernel, X, y, steps=600):
    model = GPRegressor(kernel, noise_var=0.1)
    best, _ = adam_maximize(
        lambda p: model.lml_and_grad(X, y, p), model.params, lr=0.05, steps=steps
    )
    model.params = best
    model.fit(X, y)
    return model


def region_report(model, Xs, ys, name):
    """Held-out RMSE / NLL / 95% coverage, split at the smooth-rough boundary."""
    mean, var = model.predict(Xs, include_noise=True)
    sd = np.sqrt(var)
    rows = []
    for label, mask in (
        ("smooth (x < 1.4)", Xs[:, 0] < SPLIT),
        ("rough  (x > 1.4)", Xs[:, 0] >= SPLIT),
    ):
        m, s, t = mean[mask], sd[mask], ys[mask]
        rmse = float(np.sqrt(np.mean((m - t) ** 2)))
        nll = float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + (t - m) ** 2 / (2 * s ** 2)))
        cov = float(np.mean(np.abs(t - m) <= 1.96 * s))
        rows.append((label, rmse, nll, cov, float(s.mean())))
        print(f"  {name:6s} {label}  RMSE {rmse:.3f}  NLL {nll:6.3f}  "
              f"95% coverage {cov:.2f}  mean sd {s.mean():.3f}")
    return rows


def density_sweep(counts=(40, 60, 90, 140)):
    """Does the nonstationary kernel actually predict better? It depends on n.

    The stationary model's coping strategy -- pick the short lengthscale, lean
    on data density -- only works while the data IS dense. This sweep is the
    measurement that separates "Gibbs is a better model" (always, by evidence)
    from "Gibbs predicts better" (only when data is scarce).
    """
    rows = []
    print("\nDoes it predict better? Only where the density crutch runs out:")
    print("   n   evidence gap   smooth-region RMSE      held-out NLL")
    print("                       Gibbs     RBF       Gibbs     RBF")
    for n in counts:
        rng = np.random.default_rng(SEED)
        X, y = make_data(rng, n=n)
        Xs = rng.uniform(X_LO, X_HI, size=(400, 1))
        ys = truth(Xs[:, 0]) + NOISE_SD * rng.standard_normal(400)

        out = {}
        for name, kern in (("gibbs", Gibbs(s2=1.0, a=0.0, b=0.0)),
                           ("rbf", RBF(s2=1.0, l=1.0))):
            m = fit(kern, X, y)
            mean, var = m.predict(Xs, include_noise=True)
            sd = np.sqrt(var)
            sm = Xs[:, 0] < SPLIT
            out[name] = {
                "lml": m.log_marginal_likelihood(),
                "rmse_smooth": float(np.sqrt(np.mean((mean[sm] - ys[sm]) ** 2))),
                "nll": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2)
                                     + (ys - mean) ** 2 / (2 * sd ** 2))),
            }
        rows.append({"n": n, **{f"{k}_{s}": out[s][k] for s in out for k in out[s]}})
        print(f"  {n:3d}   {out['gibbs']['lml'] - out['rbf']['lml']:+7.2f} nats   "
              f"{out['gibbs']['rmse_smooth']:.3f}   {out['rbf']['rmse_smooth']:.3f}     "
              f"{out['gibbs']['nll']:+.3f}  {out['rbf']['nll']:+.3f}")
    print("  -> evidence always prefers Gibbs; the PREDICTIVE gap opens only as")
    print("     n falls and the stationary model loses its density crutch.")
    return rows


def main():
    rng = np.random.default_rng(SEED)
    X, y = make_data(rng, n=60)
    Xs = rng.uniform(X_LO, X_HI, size=(400, 1))  # held-out inputs
    ys = truth(Xs[:, 0]) + NOISE_SD * rng.standard_normal(400)

    print("=" * 72)
    print("Nonstationary target: y = sin(x^3) + N(0, 0.1^2) on [0, 2.2], n = 60")
    print("  local wavelength ~8 at x=0.5, ~0.43 at x=2.2  (a ~20x spread)")
    print("=" * 72)

    gibbs = fit(Gibbs(s2=1.0, a=0.0, b=0.0), X, y)
    rbf = fit(RBF(s2=1.0, l=1.0), X, y)

    s2_g, a, b = gibbs.kernel._theta
    l_lo = float(np.exp(a + b * X_LO))
    l_hi = float(np.exp(a + b * X_HI))
    l_rbf = float(np.exp(rbf.kernel._theta[1]))

    print(f"\nGibbs learned:  a={a:+.3f}  b={b:+.3f}  "
          f"-> l(0)={l_lo:.3f}, l(2.2)={l_hi:.3f}  ({l_lo / l_hi:.1f}x range)")
    print(f"RBF learned:    l={l_rbf:.3f}  (one number for the whole domain)")
    print(f"\nlog evidence:   Gibbs {gibbs.log_marginal_likelihood():8.2f}")
    print(f"                RBF   {rbf.log_marginal_likelihood():8.2f}")
    print(f"                (difference: "
          f"{gibbs.log_marginal_likelihood() - rbf.log_marginal_likelihood():+.2f} nats)")

    print("\nHeld-out, split by region (the aggregate number hides the failure):")
    region_report(gibbs, Xs, ys, "Gibbs")
    region_report(rbf, Xs, ys, "RBF")

    sweep = density_sweep()

    # ---------------------------------------------------------------- figure
    grid = np.linspace(X_LO, X_HI, 500).reshape(-1, 1)
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 3.6), constrained_layout=True)

    for ax, model, name in ((axes[0], rbf, "RBF (stationary)"),
                            (axes[1], gibbs, "Gibbs (nonstationary)")):
        mean, var = model.predict(grid, include_noise=True)
        sd = np.sqrt(var)
        ax.fill_between(grid[:, 0], mean - 1.96 * sd, mean + 1.96 * sd,
                        alpha=0.25, color="C0", label="95%")
        ax.plot(grid[:, 0], truth(grid[:, 0]), "k-", lw=1.0, alpha=0.6, label="truth")
        ax.plot(grid[:, 0], mean, "C0-", lw=1.5, label="posterior mean")
        ax.scatter(X[:, 0], y, s=7, color="0.35", alpha=0.6, zorder=3, label="data")
        ax.axvline(SPLIT, color="0.7", ls=":", lw=0.8)
        ax.set_xlabel("x")
        ax.set_ylim(-2.2, 2.2)
        ax.set_title(name, loc="left")
    axes[0].set_ylabel("y")
    axes[0].legend(fontsize=7, ncol=2, loc="lower left")

    axes[2].plot(grid[:, 0], gibbs.kernel.lengthscale(grid), "C1-", lw=1.8,
                 label=r"Gibbs  $\ell(x)=e^{a+bx}$")
    axes[2].axhline(l_rbf, color="C0", ls="--", lw=1.5, label=r"RBF  $\ell$ (constant)")
    # the target's own local wavelength / 2pi, as a reference scale
    xs = grid[:, 0]
    local = 1.0 / np.maximum(3.0 * xs ** 2, 1e-6)
    axes[2].plot(xs, local, "k:", lw=1.0, alpha=0.7,
                 label=r"target's local scale $1/3x^2$")
    axes[2].set_yscale("log")
    axes[2].set_ylim(1e-2, 5)
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("lengthscale")
    axes[2].set_title("the learned lengthscale tracks the chirp", loc="left")
    axes[2].legend(fontsize=7)

    ns = [r["n"] for r in sweep]
    axes[3].plot(ns, [r["rmse_smooth_gibbs"] for r in sweep], "C1o-", lw=1.6,
                 label="Gibbs")
    axes[3].plot(ns, [r["rmse_smooth_rbf"] for r in sweep], "C0s--", lw=1.6,
                 label="RBF")
    axes[3].set_xlabel("training points n")
    axes[3].set_ylabel("held-out RMSE, smooth region")
    axes[3].set_title("the gap opens as data thins", loc="left")
    axes[3].legend(fontsize=7)
    axes[3].grid(alpha=0.25)

    fig.suptitle(
        "A stationary kernel must pick one lengthscale for a function that has many "
        "— and pays for it in data",
        x=0.01, ha="left", fontsize=11,
    )
    savefig(fig, "gibbs_kernel.png")


if __name__ == "__main__":
    main()
