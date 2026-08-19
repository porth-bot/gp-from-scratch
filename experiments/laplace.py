"""Non-Gaussian likelihoods: what the Laplace approximation gets, and misses.

Every other experiment here has a Gaussian likelihood, which is what makes the
posterior closed form. This one replaces it with a Bernoulli logit -- binary
labels -- and the posterior stops being tractable. ``gp/laplace.py`` supplies
the standard fix (fit the Gaussian that matches the posterior mode and its
curvature). The question this file answers is how wrong that is, and where.

Three measurements, in increasing order of how much they can go wrong.

**1. The Gaussian control.** With a Gaussian likelihood the Laplace
approximation is *exact*, so ``LaplaceGP`` must reproduce ``GPRegressor`` -- the
mode against the posterior mean, the predictive mean and variance, and the
approximate evidence against the exact log marginal likelihood. This is the
check that the algebra and the ``B = I + sqrt(W) K sqrt(W)`` rearrangement are
right, and it is run before anything approximate is measured, exactly as Sec. 11
runs ``Z = X`` before measuring a sparse GP.

**2. The approximation, against an oracle that does not share its assumptions.**
The exact evidence ``p(y) = int p(y|f) N(f|0,K) df`` is available by importance
sampling *from the prior*: draw ``f ~ N(0,K)``, weight by ``prod_i p(y_i|f_i)``,
average. For a Bernoulli likelihood the weights are bounded by 1, which is the
property that makes this a usable oracle rather than a second approximation --
the estimator of ``p(y)`` is unbiased and its variance is finite by
construction, so its own error can be estimated from repeated draws instead of
assumed. Sec. 10's annealed-importance-sampling study is the cautionary tale
here (an ESS reading a perfect 1.000 while the answer was short by the missing
mode), so the oracle is run at several independent seeds and its spread is
reported next to the quantity it is judging. It degrades as ``p(y)`` shrinks --
that is measured too, and it is why the sweeps below stop where they do.

**3. Where the approximation is worst.** Two knobs, because the failure is a
property of the posterior's shape rather than of the data size alone: the
kernel amplitude ``s2`` (a bigger prior variance lets the latent run further
into the flat tails of the logit, where the log-likelihood is very
non-quadratic) and the number of observations.

Also measured, because it is the mistake that is easiest to make with a fitted
model in hand: the difference between ``sigmoid(mu_*)`` and
``E[sigmoid(f_*)]``. The first ignores the latent uncertainty entirely.

Outputs: figures/laplace.png, and the numbers in the printed tables.
"""

import argparse

import numpy as np

from common import savefig
from gp.gp import GPRegressor
from gp.kernels import RBF
from gp.laplace import Bernoulli, GaussianLikelihood, LaplaceGP, grid_search


# ---------------------------------------------------------------------------
# 1. The Gaussian control
# ---------------------------------------------------------------------------
def gaussian_control(n=40, s2=1.3, l=0.8, noise=0.09, seed=0, verbose=True):
    """Laplace with a Gaussian likelihood must be the exact GP, to floating point."""
    rng = np.random.default_rng(seed)
    X = np.linspace(-3, 3, n).reshape(-1, 1)
    y = rng.normal(np.sin(X).ravel(), np.sqrt(noise))
    Xs = np.linspace(-4, 4, 61).reshape(-1, 1)

    exact = GPRegressor(RBF(s2=s2, l=l), noise_var=noise).fit(X, y)
    lap = LaplaceGP(RBF(s2=s2, l=l), GaussianLikelihood(noise)).fit(X, y)

    em, ev = exact.predict(Xs)
    lm, lv = lap.predict(Xs)
    out = {
        "mean": float(np.max(np.abs(em - lm))),
        "var": float(np.max(np.abs(ev - lv))),
        "log_ml": float(abs(exact.log_marginal_likelihood()
                            - lap.log_marginal_likelihood())),
        # the mode is the posterior mean at the training inputs; the residual
        # here is the two models' different jitter placement, not the Newton
        # iteration, which converges in one step for a quadratic Psi
        "mode": float(np.max(np.abs(lap.fit_.f - exact.predict(X)[0]))),
        "n_iter": lap.fit_.n_iter,
        "stationarity": lap.fit_.stationarity(lap.likelihood, lap.y),
    }
    if verbose:
        print("1. Gaussian likelihood: Laplace must BE the exact GP")
        print(f"   predictive mean   max |diff| {out['mean']:.3e}")
        print(f"   predictive var    max |diff| {out['var']:.3e}")
        print(f"   log marginal likelihood diff {out['log_ml']:.3e}")
        print(f"   mode vs posterior mean       {out['mode']:.3e}"
              f"   (Newton iterations: {out['n_iter']})")
    return out


# ---------------------------------------------------------------------------
# 2. The oracle
# ---------------------------------------------------------------------------
def prior_importance_oracle(kernel, lik, X, y, Xs, n_samples=2_000_000, seed=0,
                            chunk=50_000, jitter=1e-10):
    """Exact posterior quantities by importance sampling from the GP prior.

    Draw ``[f, f_*] ~ N(0, K)`` jointly over the training and test inputs,
    weight each draw by ``prod_i p(y_i | f_i)`` -- the training likelihood only,
    since the test points carry no data -- and average. Returns the log
    evidence, the posterior mean and sd of the latent at ``Xs``, the posterior
    predictive probability there, and the effective sample size.

    Why this is an oracle and not another approximation: the weights are the
    likelihood, which for a Bernoulli is in (0, 1], so ``mean(w)`` is an
    unbiased estimator of ``p(y)`` with variance bounded by ``E[w^2] <= E[w]``.
    Nothing about the posterior's shape is assumed. What it *does* cost is
    sample size, because ``p(y)`` falls roughly geometrically in n and the
    weights concentrate; :func:`oracle_reliability` measures that directly
    rather than trusting the ESS, following Sec. 10's finding that an ESS can
    read 1.000 on an answer that is wrong.
    """
    X = np.atleast_2d(X)
    Xs = np.atleast_2d(Xs)
    n, m = X.shape[0], Xs.shape[0]
    Z = np.vstack([X, Xs])
    K = kernel(Z, Z)
    K = K + np.eye(n + m) * (jitter * float(np.mean(np.diag(K))))
    L = np.linalg.cholesky(K)

    rng = np.random.default_rng(seed)
    y = lik.check_targets(y)

    # One pass with an online log-sum-exp rescaling: keep the running maximum
    # log weight and rescale the accumulators whenever a chunk beats it. The
    # alternative, storing every weight, is n_samples x (n+m) floats; the
    # naive alternative, exponentiating against a fixed offset, underflows to
    # zero as soon as p(y) is small, which is the regime the sweeps end in.
    M = -np.inf
    acc_w = acc_w2 = 0.0
    acc_f = np.zeros(m)
    acc_f2 = np.zeros(m)
    acc_p = np.zeros(m)
    count = 0
    for start in range(0, n_samples, chunk):
        size = min(chunk, n_samples - start)
        f = L @ rng.standard_normal((n + m, size))        # (n+m, size)
        logw = lik.log_pdf(y[:, None], f[:n]).sum(axis=0)
        chunk_max = float(logw.max())
        if chunk_max > M:
            rescale = np.exp(M - chunk_max) if np.isfinite(M) else 0.0
            acc_w *= rescale
            acc_w2 *= rescale ** 2
            acc_f *= rescale
            acc_f2 *= rescale
            acc_p *= rescale
            M = chunk_max
        w = np.exp(logw - M)
        fs = f[n:]
        acc_w += float(w.sum())
        acc_w2 += float((w ** 2).sum())
        acc_f += fs @ w
        acc_f2 += (fs ** 2) @ w
        acc_p += lik.mean(fs) @ w
        count += size

    max_logw = M
    log_Z = float(np.log(acc_w / count) + max_logw)
    mean = acc_f / acc_w
    var = np.maximum(acc_f2 / acc_w - mean ** 2, 0.0)
    prob = acc_p / acc_w
    ess = float(acc_w ** 2 / acc_w2)
    return {"log_Z": log_Z, "mean": mean, "sd": np.sqrt(var), "prob": prob,
            "ess": ess, "ess_frac": ess / count}


def oracle_reliability(kernel, lik, X, y, Xs, seeds=(0, 1, 2, 3, 4), **kw):
    """Run the oracle at several seeds; return the mean and the observed spread.

    This is the oracle's own error bar, measured rather than derived, and it is
    what decides whether a Laplace error of a given size is resolvable at all.
    """
    runs = [prior_importance_oracle(kernel, lik, X, y, Xs, seed=s, **kw)
            for s in seeds]
    log_Zs = np.array([r["log_Z"] for r in runs])
    probs = np.array([r["prob"] for r in runs])
    return {
        "log_Z": float(log_Zs.mean()),
        "log_Z_sd": float(log_Zs.std(ddof=1)),
        "prob": probs.mean(axis=0),
        "prob_sd": probs.std(axis=0, ddof=1).max(),
        "ess_frac": float(np.mean([r["ess_frac"] for r in runs])),
        "runs": runs,
    }


# ---------------------------------------------------------------------------
# 3. Where the approximation is worst
# ---------------------------------------------------------------------------
def make_data(n, seed=0, freq=1.2, flip=0.0):
    """n points on [-3,3] with labels from ``sign(sin(freq x))``, optionally noisy."""
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(-3.0, 3.0, n)).reshape(-1, 1)
    y = np.sign(np.sin(freq * X.ravel()))
    y[y == 0] = 1.0
    if flip > 0:
        mask = rng.random(n) < flip
        y[mask] *= -1.0
    return X, y


def _compare(kernel_factory, X, y, Xs, seeds, n_samples):
    """One cell: Laplace against the oracle, on the same kernel and data."""
    lik = Bernoulli()
    model = LaplaceGP(kernel_factory(), lik).fit(X, y)
    mu, var = model.predict(Xs)
    p_avg, _ = model.predict_prob(Xs)
    p_plug = lik.mean(mu)
    ref = oracle_reliability(kernel_factory(), lik, X, y, Xs, seeds=seeds,
                             n_samples=n_samples)
    return {
        "log_ml": model.log_marginal_likelihood(),
        "log_Z": ref["log_Z"],
        "log_Z_sd": ref["log_Z_sd"],
        "log_Z_err": model.log_marginal_likelihood() - ref["log_Z"],
        "prob_err": float(np.max(np.abs(p_avg - ref["prob"]))),
        "prob_sd": ref["prob_sd"],
        "plugin_gap": float(np.max(np.abs(p_avg - p_plug))),
        # mean posterior sd of the latent, Laplace over oracle: < 1 means the
        # approximation is over-confident about f
        "latent_sd_ratio": float(np.mean(np.sqrt(var)) / float(np.mean(
            np.array([r["sd"] for r in ref["runs"]]).mean(axis=0)))),
        "ess_frac": ref["ess_frac"],
        "n_iter": model.fit_.n_iter,
    }


def amplitude_sweep(amplitudes=(0.25, 1.0, 4.0, 16.0, 64.0), n=12, l=1.0,
                    seeds=(0, 1, 2, 3, 4), n_samples=1_000_000, verbose=True):
    """Laplace's error against the prior amplitude, at fixed data.

    ``s2`` is the knob that controls how non-Gaussian the posterior is. With a
    tiny prior variance the latent cannot leave the region where ``log
    sigmoid`` is close to quadratic and Laplace has almost nothing to get
    wrong; with a large one the posterior mass sits out in the tails where the
    log-likelihood is linear, and a Gaussian fitted at the mode is a poor
    description of it.
    """
    X, y = make_data(n, seed=3)
    Xs = np.linspace(-3, 3, 13).reshape(-1, 1)
    rows = []
    for s2 in amplitudes:
        row = _compare(lambda s2=s2: RBF(s2=s2, l=l), X, y, Xs, seeds, n_samples)
        row["s2"] = s2
        rows.append(row)
        if verbose:
            print(f"   s2={s2:6.2f}  log Z: Laplace {row['log_ml']:9.4f} vs exact "
                  f"{row['log_Z']:9.4f} +- {row['log_Z_sd']:.4f}   "
                  f"err {row['log_Z_err']:+8.4f}   max|dp| {row['prob_err']:.4f} "
                  f"(oracle +-{row['prob_sd']:.4f})   sd ratio "
                  f"{row['latent_sd_ratio']:.4f}   ESS {row['ess_frac']:.4f}")
    return rows


def size_sweep(sizes=(4, 8, 16, 24, 32), s2=4.0, l=1.0, seeds=(0, 1, 2, 3, 4),
               n_samples=1_000_000, verbose=True):
    """Laplace's error against n -- and the oracle's own limit, in the same table.

    Two things move together here and only one of them is about Laplace. The
    posterior concentrates as n grows, which by the Bernstein-von Mises
    argument should make a Gaussian approximation *better*. But ``p(y)`` falls
    roughly geometrically in n, so the prior-importance oracle's effective
    sample size falls with it, and past some n the oracle can no longer resolve
    the error it is being asked to measure. Both columns are printed so the
    reader can see which one ran out first.
    """
    Xs = np.linspace(-3, 3, 13).reshape(-1, 1)
    rows = []
    for n in sizes:
        X, y = make_data(n, seed=3)
        row = _compare(lambda: RBF(s2=s2, l=l), X, y, Xs, seeds, n_samples)
        row["n"] = n
        rows.append(row)
        if verbose:
            print(f"   n={n:3d}  log Z err {row['log_Z_err']:+8.4f} "
                  f"(oracle +-{row['log_Z_sd']:.4f})   max|dp| {row['prob_err']:.4f} "
                  f"(oracle +-{row['prob_sd']:.4f})   sd ratio "
                  f"{row['latent_sd_ratio']:.4f}   ESS frac {row['ess_frac']:.5f}")
    return rows


def on_edge(table, grids):
    """Is the argmax on the boundary of the grid? Returns the offending axes.

    A grid search reports an argmax whether or not the surface has an interior
    maximum, and reporting a boundary point as "the ML-II optimum" is how a
    too-small grid becomes a published number. The first version of the sweep
    below did exactly that -- it stopped at ``s2 = 64`` and reported 64 -- and
    the real optimum for the separable arm is at 271. So the check is run and
    printed rather than left to whoever reads the table.
    """
    idx = np.unravel_index(int(np.argmax(table)), table.shape)
    return [ax for ax, (i, g) in enumerate(zip(idx, grids))
            if i == 0 or i == len(g) - 1]


def evidence_surface(n=24, flip=0.0, s2_grid=None, l_grid=None, verbose=True):
    """The approximate log evidence over (s2, l), by grid search.

    There are no hyperparameter gradients in ``gp/laplace.py``, so this is how
    ML-II is done there. Returning the whole surface rather than the argmax is
    deliberate: Sec. 9 found ML-II multimodal on the Gaussian likelihood, and a
    grid is the one method that shows the surface for free -- and shows when the
    argmax has run into the edge of it (:func:`on_edge`).

    ``flip`` corrupts a fraction of the labels, which is the second arm: with
    perfectly separable labels the evidence keeps rewarding a larger amplitude
    for a long way (the latent can grow without ever contradicting the data),
    and flipping labels is what puts an interior optimum back.
    """
    s2_grid = np.geomspace(0.25, 1e4, 21) if s2_grid is None else s2_grid
    l_grid = np.geomspace(0.1, 6.0, 17) if l_grid is None else l_grid
    X, y = make_data(n, seed=3, flip=flip)
    best, best_ml, table = grid_search(lambda s2, l: RBF(s2=s2, l=l), Bernoulli(),
                                       X, y, (s2_grid, l_grid))
    edges = on_edge(table, (s2_grid, l_grid))
    if verbose:
        flag = f"   ON THE GRID EDGE in axes {edges}" if edges else ""
        print(f"   flip={flip:.2f}: best (s2, l) = ({best[0]:.3f}, {best[1]:.3f}) "
              f"at log Z_hat = {best_ml:.4f}{flag}")
    return {"s2_grid": s2_grid, "l_grid": l_grid, "table": table, "flip": flip,
            "best": best, "best_log_ml": best_ml, "edges": edges, "X": X, "y": y}


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------
def figure(demo, amp_rows, size_rows, surfaces, name="laplace.png"):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.0))

    # (a) the latent posterior, Laplace vs oracle
    ax = axes[0, 0]
    Xs, mu, sd = demo["Xs"].ravel(), demo["mu"], demo["sd"]
    ax.fill_between(Xs, mu - 2 * sd, mu + 2 * sd, alpha=0.25, color="C0", lw=0,
                    label="Laplace $\\pm 2$ sd")
    ax.plot(Xs, mu, color="C0", lw=1.6, label="Laplace mean")
    ax.plot(Xs, demo["omu"], color="C3", ls="--", lw=1.4, label="exact mean")
    ax.plot(Xs, demo["omu"] + 2 * demo["osd"], color="C3", ls=":", lw=1.1,
            label="exact $\\pm 2$ sd")
    ax.plot(Xs, demo["omu"] - 2 * demo["osd"], color="C3", ls=":", lw=1.1)
    ax.set_xlabel("$x$")
    ax.set_ylabel("latent $f$")
    ax.set_title("(a) latent posterior, $n=%d$, $s^2=%.0f$"
                 % (len(demo["y"]), demo["s2"]), loc="left")
    ax.legend(fontsize=7, loc="upper right")

    # (b) predictive probability: averaged, plug-in, exact
    ax = axes[0, 1]
    ax.plot(Xs, demo["p_avg"], color="C0", lw=1.6, label=r"Laplace $E[\sigma(f_*)]$")
    ax.plot(Xs, demo["p_plug"], color="C2", ls="-.", lw=1.4,
            label=r"plug-in $\sigma(\mu_*)$")
    ax.plot(Xs, demo["oprob"], color="C3", ls="--", lw=1.4, label="exact")
    ax.plot(demo["X"].ravel(), (demo["y"] > 0).astype(float), "k|", ms=9, mew=1.2,
            label="labels")
    ax.axhline(0.5, color="0.6", lw=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("$x$")
    ax.set_ylabel(r"$p(y_*=+1 \mid y)$")
    ax.set_title("(b) averaging over the latent is not optional", loc="left")
    ax.legend(fontsize=7, loc="center right")

    # (c) error vs prior amplitude
    ax = axes[0, 2]
    s2 = [r["s2"] for r in amp_rows]
    ax.plot(s2, np.abs([r["log_Z_err"] for r in amp_rows]), "o-", color="C0",
            label=r"$|\Delta \log Z|$ (nats)")
    ax.fill_between(s2, 1e-6, [r["log_Z_sd"] for r in amp_rows], color="C3",
                    alpha=0.25, lw=0, label="oracle's own sd")
    ax.plot(s2, [r["prob_err"] for r in amp_rows], "s-", color="C2",
            label=r"$\max |\Delta p|$")
    ax.plot(s2, [1 - r["latent_sd_ratio"] for r in amp_rows], "^-", color="C4",
            label="latent sd shortfall")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=5e-5)
    ax.set_xlabel("prior amplitude $s^2$")
    ax.set_title("(c) the error is set by the prior", loc="left")
    ax.legend(fontsize=7, loc="lower right")

    # (d) error vs n, with the oracle's ESS on a twin axis
    ax = axes[1, 0]
    ns = [r["n"] for r in size_rows]
    ax.plot(ns, np.abs([r["log_Z_err"] for r in size_rows]), "o-", color="C0",
            label=r"$|\Delta \log Z|$")
    ax.plot(ns, [r["log_Z_sd"] for r in size_rows], "--", color="C3",
            label="oracle's own sd")
    ax.set_yscale("log")
    ax.set_xlabel("$n$")
    ax.set_ylabel("nats")
    ax.set_title("(d) and not by $n$ -- until the oracle dies", loc="left")
    ax.legend(fontsize=7, loc="center left")
    tw = ax.twinx()
    tw.plot(ns, [r["ess_frac"] for r in size_rows], ":", color="0.4")
    tw.set_yscale("log")
    tw.set_ylabel("oracle ESS fraction", color="0.4")
    tw.spines["right"].set_visible(True)

    # (e) evidence profile over the amplitude, both label arms
    ax = axes[1, 1]
    for surf, color, style in zip(surfaces, ("C0", "C3"), ("-", "--")):
        prof = surf["table"].max(axis=1)
        ax.plot(surf["s2_grid"], prof, style, color=color,
                label=f"{int(round(100 * surf['flip']))}% labels flipped")
        ax.plot([surf["best"][0]], [surf["best_log_ml"]], "o", color=color, ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("prior amplitude $s^2$")
    ax.set_ylabel(r"$\log \hat{Z}$ (profiled over $\ell$)")
    ax.set_title("(e) ML-II by grid search: where it peaks", loc="left")
    ax.legend(fontsize=7, loc="lower left")

    # (f) the surface itself, noisy arm
    ax = axes[1, 2]
    surf = surfaces[-1]
    T = surf["table"]
    im = ax.pcolormesh(surf["s2_grid"], surf["l_grid"], T.T, shading="nearest",
                       cmap="viridis", vmin=np.percentile(T, 25), vmax=T.max())
    ax.plot([surf["best"][0]], [surf["best"][1]], "w*", ms=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("prior amplitude $s^2$")
    ax.set_ylabel(r"lengthscale $\ell$")
    ax.set_title(f"(f) the surface, {int(round(100 * surf['flip']))}% flipped",
                 loc="left")
    fig.colorbar(im, ax=ax, label=r"$\log \hat{Z}$")

    fig.tight_layout()
    savefig(fig, name)


def demo(n=12, s2=4.0, l=1.0, seeds=(0, 1, 2, 3, 4), n_samples=2_000_000):
    """The 1-D picture: one dataset, Laplace and the oracle side by side."""
    X, y = make_data(n, seed=3)
    Xs = np.linspace(-3.2, 3.2, 81).reshape(-1, 1)
    lik = Bernoulli()
    model = LaplaceGP(RBF(s2=s2, l=l), lik).fit(X, y)
    mu, var = model.predict(Xs)
    p_avg, _ = model.predict_prob(Xs)
    ref = oracle_reliability(RBF(s2=s2, l=l), lik, X, y, Xs, seeds=seeds,
                             n_samples=n_samples)
    osd = np.array([r["sd"] for r in ref["runs"]]).mean(axis=0)
    omu = np.array([r["mean"] for r in ref["runs"]]).mean(axis=0)
    return {"X": X, "y": y, "Xs": Xs, "s2": s2, "mu": mu, "sd": np.sqrt(var),
            "p_avg": p_avg, "p_plug": lik.mean(mu), "omu": omu, "osd": osd,
            "oprob": ref["prob"], "ess_frac": ref["ess_frac"]}


def main(quick=False):
    n_samples = 100_000 if quick else 1_000_000
    seeds = (0, 1) if quick else (0, 1, 2, 3, 4)

    control = gaussian_control()

    print("\n2. Bernoulli logit vs an importance-sampling oracle")
    print("   prior amplitude sweep (n=12, l=1):")
    amp = amplitude_sweep(seeds=seeds, n_samples=n_samples)
    print("   data-size sweep (s2=4, l=1):")
    size = size_sweep(seeds=seeds, n_samples=n_samples)

    d = demo(seeds=seeds, n_samples=2 * n_samples)
    gap = float(np.max(np.abs(d["p_avg"] - d["p_plug"])))
    print(f"\n3. plug-in vs averaged predictive probability: max gap {gap:.4f} "
          f"at latent sd up to {d['sd'].max():.3f}")
    print(f"   Laplace's own error against exact on the same grid: "
          f"{np.max(np.abs(d['p_avg'] - d['oprob'])):.4f}")

    print("\n4. ML-II by grid search over the approximate evidence")
    surfaces = [evidence_surface(n=24, flip=f) for f in (0.0, 0.15)]

    figure(d, amp, size, surfaces)
    return control, amp, size, d, surfaces


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)
