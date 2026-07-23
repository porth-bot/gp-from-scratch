"""ML-II is a non-convex problem: the evidence is multimodal, and multi-start fixes it.

Type-II maximum likelihood (maximizing the log marginal likelihood over the
hyperparameters) is the standard way to set a GP's kernel, but the objective is
non-convex and genuinely multimodal (Rasmussen & Williams 2006, Sec. 5.4.1,
Fig. 5.5). The textbook two-mode picture: for a sparse, wiggly dataset an RBF
can explain it as

  - **signal**: a short lengthscale with little observation noise (the function
    really does wiggle), or
  - **noise**: a long, flat lengthscale with large observation noise (the
    wiggles are measurement error around a nearly constant mean).

Both are local optima of the evidence. A single gradient ascent commits to
whichever basin its initialization sits in; ``gp.optimize.maximize_lml_multistart``
restarts from several log-space initializations and keeps the highest-evidence
fit.

This script builds the two-mode dataset, shows the two competing fits, profiles
the log evidence along the lengthscale (freezing the lengthscale on a grid and
optimizing the other hyperparameters at each point, using the repo's fixed-param
mask), and marks where a naive long-lengthscale start lands versus where
multi-start lands.

Run:  python experiments/multistart.py
"""

import numpy as np

from common import savefig
import matplotlib.pyplot as plt

from gp.gp import GPRegressor
from gp.kernels import RBF
from gp.optimize import adam_maximize, maximize_lml_multistart


def multimodal_dataset():
    """12 sparse samples of a wiggly signal -- both explanations are credible."""
    rng = np.random.default_rng(0)
    X = np.linspace(-4, 4, 12).reshape(-1, 1)
    y = np.sin(1.7 * X).ravel() + 0.15 * rng.standard_normal(12)
    return X, y


def fit_from(X, y, l0, noise0, steps=500):
    """Single Adam ML-II fit from a given initialization; return the fit model."""
    m = GPRegressor(RBF(s2=1.0, l=l0), noise_var=noise0)
    best, _ = adam_maximize(lambda p: m.lml_and_grad(X, y, p), m.params,
                            lr=0.05, steps=steps)
    m.params = best
    m.fit(X, y)
    return m


def profile_lml_over_lengthscale(X, y, ls, steps=300):
    """For each fixed lengthscale, optimize (s2, noise) and record the max LML.

    Uses the kernel's fixed-parameter mask to freeze ``l`` so the profile is the
    best evidence *achievable at that lengthscale* -- the honest 1D slice of a
    3D surface, not a single arbitrary cross-section.
    """
    out = np.empty_like(ls)
    for i, l in enumerate(ls):
        m = GPRegressor(RBF(s2=1.0, l=float(l), fixed=["l"]), noise_var=0.1)
        best, _ = adam_maximize(lambda p: m.lml_and_grad(X, y, p), m.params,
                                lr=0.05, steps=steps)
        out[i] = m.lml_and_grad(X, y, best)[0]
    return out


def main():
    X, y = multimodal_dataset()
    xs = np.linspace(-5, 5, 300).reshape(-1, 1)

    # The two competing single-start fits.
    signal = fit_from(X, y, l0=0.3, noise0=0.1)    # short-l basin
    noise = fit_from(X, y, l0=30.0, noise0=0.6)     # long-l "all noise" basin
    lml_signal = signal.log_marginal_likelihood()
    lml_noise = noise.log_marginal_likelihood()

    # Multi-start from the SAME bad long-lengthscale model, over a broad box.
    box = np.log([[1e-2, 1e2], [1e-1, 1e2], [1e-4, 1e0]])   # log s2, l, noise
    model = GPRegressor(RBF(s2=1.0, l=30.0), noise_var=0.6)
    res = maximize_lml_multistart(model, X, y, n_restarts=12, bounds=box,
                                  rng=np.random.default_rng(0), steps=500)
    s2_star, l_star = np.exp(model.kernel.theta)
    noise_star = model.noise_var

    print(f"signal-mode single fit : LML {lml_signal:7.3f}  "
          f"l {np.exp(signal.kernel.theta[1]):6.3f}  noise {signal.noise_var:.4f}")
    print(f"noise-mode  single fit : LML {lml_noise:7.3f}  "
          f"l {np.exp(noise.kernel.theta[1]):6.3f}  noise {noise.noise_var:.4f}")
    print(f"multi-start (from noise init): LML {res.lml:7.3f}  "
          f"l {l_star:6.3f}  noise {noise_star:.4f}  best_restart {res.best_restart}")
    finite = res.lmls[np.isfinite(res.lmls)]
    print(f"  per-restart LML range: [{finite.min():.2f}, {finite.max():.2f}] "
          f"over {len(finite)} finite of {len(res.lmls)} restarts")

    # -- figure: the two fits (left) + the evidence profile in l (right) -------
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.8),
                                   constrained_layout=True)

    for m, color, label, lml in (
        (signal, "C0", "signal mode (short l)", lml_signal),
        (noise, "C3", "noise mode (long l)", lml_noise),
    ):
        mean, var = m.predict(xs, include_noise=True)
        sd = np.sqrt(var)
        axL.plot(xs.ravel(), mean, color=color, lw=1.6,
                 label=f"{label}, LML {lml:.1f}")
        axL.fill_between(xs.ravel(), mean - 1.96 * sd, mean + 1.96 * sd,
                         color=color, alpha=0.15)
    axL.plot(X.ravel(), y, "k.", ms=8, label="data")
    axL.set_xlabel("x")
    axL.set_ylabel("y")
    axL.set_title("Two ML-II optima explain the same data", loc="left")
    axL.legend(loc="upper center", fontsize=7.5)

    ls = np.geomspace(0.1, 60, 80)
    prof = profile_lml_over_lengthscale(X, y, ls)
    axR.plot(ls, prof, "0.3", lw=1.6)
    axR.axvline(np.exp(signal.kernel.theta[1]), color="C0", ls="--", lw=1.2,
                label="signal-mode l")
    axR.axvline(np.exp(noise.kernel.theta[1]), color="C3", ls="--", lw=1.2,
                label="noise-mode l")
    axR.plot(l_star, res.lml, "*", color="C2", ms=13,
             label="multi-start winner")
    axR.set_xscale("log")
    axR.set_xlabel("lengthscale  l  (log scale)")
    axR.set_ylabel("profile log evidence  max_{s², σ²} LML")
    axR.set_title("The evidence is multimodal in the lengthscale", loc="left")
    axR.legend(loc="lower center", fontsize=7.5)

    savefig(fig, "multistart.png")
    return res


if __name__ == "__main__":
    main()
