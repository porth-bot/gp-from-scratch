"""A 2D spatial GP: interpolate a scattered field and *see* the uncertainty.

The GP story is clearest in 2D, where the posterior mean and -- crucially --
the posterior *standard deviation* are both surfaces you can look at. We sample
a smooth two-bump field at scattered locations with a little noise, fit an
isotropic RBF GP by maximizing the evidence, and plot three panels:

    (a) the truth with the sampled points,
    (b) the posterior mean (the interpolant), and
    (c) the posterior std -- small near data, swelling toward the prior in the
        gaps and past the edges.

Panel (c) is the whole point: a GP does not just interpolate, it reports where
it is guessing. We check that against a held-out grid: RMSE where the field is
well sampled, and the 95% coverage of the latent surface.

Run:  python experiments/spatial2d.py
"""

import numpy as np

from common import plt, savefig
from gp.gp import GPRegressor
from gp.kernels import RBF
from gp.optimize import adam_maximize

SEED = 0
NOISE = 0.05


def field(P):
    """A smooth two-bump target on [-3, 3]^2 (amplitude ~1)."""
    x, y = P[:, 0], P[:, 1]
    return (
        np.exp(-((x - 1.0) ** 2 + (y - 1.0) ** 2))
        - 0.6 * np.exp(-((x + 1.2) ** 2 + (y + 0.8) ** 2) / 1.5)
    )


def main():
    rng = np.random.default_rng(SEED)

    # scattered training samples (a real spatial dataset is rarely gridded)
    n = 140
    Xtr = rng.uniform(-3, 3, size=(n, 2))
    ytr = field(Xtr) + NOISE * rng.standard_normal(n)

    model = GPRegressor(RBF(s2=0.5, l=1.0), noise_var=NOISE ** 2)
    best, _ = adam_maximize(
        lambda p: model.lml_and_grad(Xtr, ytr, p), model.params, lr=0.05, steps=400
    )
    model.params = best
    model.fit(Xtr, ytr)

    s2 = np.exp(model.kernel._theta[0])
    l = np.exp(model.kernel._theta[1])
    print("=" * 60)
    print("2D spatial GP: two-bump field, isotropic RBF, ML-II")
    print("=" * 60)
    print(f"learned  s2={s2:.3f}  l={l:.3f}  noise_sd={np.sqrt(model.noise_var):.3f}"
          f"  (true noise_sd={NOISE})")
    print(f"log-evidence: {model.log_marginal_likelihood():.2f}")

    # held-out accuracy + calibration on a dense grid
    g = np.linspace(-3, 3, 60)
    XX, YY = np.meshgrid(g, g)
    grid = np.column_stack([XX.ravel(), YY.ravel()])
    truth = field(grid)
    mean, var = model.predict(grid)
    sd = np.sqrt(var)
    rmse = np.sqrt(np.mean((mean - truth) ** 2))
    z = np.abs(truth - mean) / sd
    cover = float(np.mean(z <= 1.96))
    print(f"held-out RMSE: {rmse:.4f}  (field amplitude ~1)")
    print(f"latent 95% coverage on the grid: {cover:.3f}")
    print(f"posterior sd range: {sd.min():.3f} (near data) .. {sd.max():.3f} "
          f"(gaps/edges);  prior sd sqrt(s2)={np.sqrt(s2):.3f}")

    # ---- figure -----------------------------------------------------------
    Z_truth = truth.reshape(XX.shape)
    Z_mean = mean.reshape(XX.shape)
    Z_sd = sd.reshape(XX.shape)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
    vlim = np.max(np.abs(Z_truth))
    for ax, Z, title in [
        (axes[0], Z_truth, "(a) truth + samples"),
        (axes[1], Z_mean, "(b) posterior mean"),
    ]:
        cf = ax.contourf(XX, YY, Z, levels=20, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        ax.set_title(title, loc="left")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        fig.colorbar(cf, ax=ax, shrink=0.85)
    axes[0].scatter(Xtr[:, 0], Xtr[:, 1], s=8, c="k", alpha=0.5)

    cf = axes[2].contourf(XX, YY, Z_sd, levels=20, cmap="viridis")
    axes[2].scatter(Xtr[:, 0], Xtr[:, 1], s=8, c="w", alpha=0.6, edgecolors="none")
    axes[2].set_title("(c) posterior std (uncertainty)", loc="left")
    axes[2].set_xlabel("$x$")
    axes[2].set_ylabel("$y$")
    fig.colorbar(cf, ax=axes[2], shrink=0.85)

    fig.suptitle(
        "A 2D GP interpolates the field (b) and reports where it is guessing (c)",
        x=0.02, ha="left",
    )
    savefig(fig, "spatial2d.png")


if __name__ == "__main__":
    main()
