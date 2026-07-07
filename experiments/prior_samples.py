"""Pedagogical figure: prior sample paths under different kernels.

Before any data is seen, a GP *is* its kernel: the kernel alone fixes how
rough or smooth candidate functions are and whether they carry structure like
periodicity. This draws several functions from the zero-mean prior for five
standard kernels (all with the same lengthscale and variance so only the
kernel *family* differs), which is the clearest way to build intuition for
what each kernel assumes.

The smoothness ladder is the Matern family: nu = 1/2 gives continuous but
nowhere-differentiable paths (Ornstein-Uhlenbeck), nu = 3/2 once-
differentiable, nu = 5/2 twice-differentiable, and the RBF (the nu -> infinity
limit) infinitely differentiable. The Periodic kernel produces exactly
repeating draws. See theory/derivations.md Sec. 3 for the derivations.

Run:  python experiments/prior_samples.py
"""

import numpy as np
from common import plt, savefig

from gp.gp import sample_prior
from gp.kernels import RBF, Matern, Periodic

N_PATHS = 5
LENGTHSCALE = 1.0


def main():
    X = np.linspace(-3.0, 3.0, 400)[:, None]
    rng = np.random.default_rng(0)

    panels = [
        ("Matérn ν=1/2\n(continuous, non-differentiable)", Matern(nu=0.5, l=LENGTHSCALE)),
        ("Matérn ν=3/2\n(once differentiable)", Matern(nu=1.5, l=LENGTHSCALE)),
        ("Matérn ν=5/2\n(twice differentiable)", Matern(nu=2.5, l=LENGTHSCALE)),
        ("RBF  (ν→∞, C^∞ smooth)", RBF(l=LENGTHSCALE)),
        ("Periodic  (period 1, exactly repeating)", Periodic(l=LENGTHSCALE, p=1.0)),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(15, 2.9), constrained_layout=True)
    xs = X[:, 0]
    for ax, (title, kernel) in zip(axes, panels):
        draws = sample_prior(kernel, X, N_PATHS, rng)
        for f in draws:
            ax.plot(xs, f, lw=1.0, alpha=0.9)
        ax.fill_between(xs, -2.0, 2.0, color="0.85", alpha=0.4, zorder=0)  # +-2 sd prior band
        ax.set_title(title)
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(-3.2, 3.2)
        ax.set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    fig.suptitle(
        "GP prior sample paths: the kernel alone sets smoothness and structure "
        f"(same lengthscale l={LENGTHSCALE:g}, variance 1; shaded = ±2 sd prior band)",
        fontsize=10,
    )
    savefig(fig, "prior_samples.png")


if __name__ == "__main__":
    main()
