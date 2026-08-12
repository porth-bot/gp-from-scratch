"""Experiment 2: Mauna Loa CO2 -- structured kernels on real data.

The classic GP demonstration (Rasmussen & Williams 2006, Sec. 5.4.3): monthly
CO2 concentrations are a smooth rising trend + an annual cycle whose shape
drifts slowly + short-term weather noise. Kernels compose by addition
(independent additive processes), so the model is read off the physics:

    k = RBF_trend + Periodic * RBF_decay + RQ_medium + Matern32_short  (+ noise)

The Periodic * RBF product is the key move: exactly-periodic correlation
modulated by a long-lengthscale RBF, i.e. "seasonal, but this year's cycle
resembles next year's more than one 30 years out".

The seasonal PERIOD is frozen at exactly 1.0 year (``fixed=["p"]``). It is
known from the physics, and freezing it is also a numerical necessity: near a
phase mismatch the periodic log-period gradient is enormous (order 1e3 at this
init), which destabilizes Adam and made an earlier free-period run diverge and
fall back to its initialization. With the period pinned, the remaining
hyperparameters optimize smoothly at lr=0.01 over 800 steps.

Honest evaluation FIRST: fit on data up to 2015.0 only, forecast the held-out
2015-2026 months (true out-of-sample -- the model never sees them), report
RMSE and 95% coverage. Then refit on everything for the 2040 extrapolation
figure.

That evaluation produced the repo's sharpest negative result -- ML-II raises
the in-sample evidence and forecasts WORSE than the unoptimized hand-set
kernel -- with a diagnosis attached: the evidence buys in-sample wiggle by
SHORTENING the trend lengthscale, and an RBF trend mean-reverts beyond its
lengthscale, so it undershoots an 11-year continued rise. R&W's own CO2 kernel
carries a fourth, medium-term term that this one lacked, which is what would
let the trend stay long. So the diagnosis is testable, and three composites are
fitted here rather than one:

    base   trend + seasonal + short                    (what the repo shipped)
    rq     trend + seasonal + RationalQuadratic + short (R&W's remedy)
    rbf    trend + seasonal + RBF + short               (the control)

The control is the point. If the fix is RationalQuadratic's heavy tail -- a
scale mixture carrying several lengthscales at once -- then an ordinary RBF in
the same slot, with the same init, should not reproduce it. Measured below.

Run:  python experiments/co2.py    (~3 min: four ML-II fits on n~700)
"""

import os

import numpy as np

from common import plt, savefig
from gp.gp import GPRegressor
from gp.kernels import RBF, Matern, Periodic, RationalQuadratic
from gp.optimize import adam_maximize

# Resolved against this file, not the working directory: the README documents
# running the experiments from experiments/, where a CWD-relative "data/..."
# does not exist.
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..", "data", "co2_mm_mlo.txt")


def load_co2():
    raw = np.loadtxt(DATA, comments="#")
    t, ppm = raw[:, 2], raw[:, 3]
    keep = ppm > 0  # missing months are flagged negative
    return t[keep], ppm[keep]


def make_kernel(medium=None):
    """The CO2 composite, optionally with a medium-term term in the third slot.

    ``medium`` is None (the original three-term kernel, 8 free params),
    "rq" (RationalQuadratic, 11) or "rbf" (the control, 10). The trend,
    seasonal and short-term components are identical in all three, and the two
    medium terms share an init (s2=0.66, l=1.2, R&W Sec. 5.4.3's values), so
    the rq-vs-rbf comparison isolates the scale mixture and nothing else.
    """
    trend = RBF(s2=50.0**2, l=40.0)
    # Period frozen at 1.0 yr (known physics + gradient stability); see module docstring.
    seasonal = Periodic(s2=4.0, l=1.3, p=1.0, fixed=["p"]) * RBF(s2=1.0, l=90.0)
    short = Matern(nu=1.5, s2=0.5, l=1.0)
    if medium is None:
        return trend + seasonal + short
    if medium == "rq":
        return trend + seasonal + RationalQuadratic(s2=0.66, l=1.2, alpha=0.78) + short
    if medium == "rbf":
        return trend + seasonal + RBF(s2=0.66, l=1.2) + short
    raise ValueError(f"medium must be None, 'rq' or 'rbf'; got {medium!r}")


def fit(t, y, steps=800, lr=0.01, medium=None):
    X = t[:, None]
    y_mean = y.mean()
    model = GPRegressor(make_kernel(medium), noise_var=0.05)
    best, hist = adam_maximize(
        lambda p: model.lml_and_grad(X, y - y_mean, p),
        model.params, lr=lr, steps=steps,
    )
    model.params = best
    model.fit(X, y - y_mean)
    return model, y_mean, hist


def main(steps=800):
    t, y = load_co2()
    print(f"{len(t)} monthly observations, {t.min():.1f} - {t.max():.1f}")
    if steps != 800:
        print(f"(--steps {steps}: the convergence check, not the shipped run)")

    # ---- honest out-of-sample test: train < 2015, predict >= 2015 ----
    train = t < 2015.0
    Xtr, ytr = t[train][:, None], y[train]
    horizon = t[~train].max() - 2015.0

    def heldout(model, y_mean):
        mu, var = model.predict(t[~train][:, None], include_noise=True)
        resid = y[~train] - (mu + y_mean)
        rmse = float(np.sqrt(np.mean(resid**2)))
        cover = float(np.mean(np.abs(resid) < 1.96 * np.sqrt(var)))
        return rmse, cover

    # Reference: the hand-set kernel WITHOUT optimization (physically motivated
    # inits). This is the "prior knowledge" baseline the ML-II optima are judged
    # against on the 11-year extrapolation.
    y_mean = ytr.mean()
    ref = GPRegressor(make_kernel(), noise_var=0.05).fit(Xtr, ytr - y_mean)
    ref_rmse, ref_cover = heldout(ref, y_mean)
    ref_trend_l = float(np.exp(ref.kernel.theta[1]))

    print(f"held-out 2015-{t.max():.1f} ({horizon:.1f} yr), n_train = {train.sum()}:")
    print(f"  {'model':<26} {'LML':>18}  {'RMSE':>8}  {'95% cov':>7}  {'trend l':>7}"
          f"  {'last 1/4':>9}")
    print(f"  {'hand-set init (no opt)':<26} {'--':>18}  {ref_rmse:8.2f}  "
          f"{ref_cover:7.2f}  {ref_trend_l:6.0f} yr")

    rows = {}
    for medium, label in ((None, "ML-II, base"),
                          ("rq", "ML-II, + RationalQuad"),
                          ("rbf", "ML-II, + RBF (control)")):
        model, ym, hist = fit(Xtr[:, 0], ytr, steps=steps, medium=medium)
        lml0, lml_best = hist[0], max(hist)
        # ML-II must improve the evidence it is optimizing, in every arm.
        assert lml_best > lml0, f"{label}: ML-II must improve the marginal likelihood"
        rmse, cover = heldout(model, ym)
        trend_l = float(np.exp(model.kernel.theta[1]))
        rows[medium] = dict(lml=lml_best, rmse=rmse, cover=cover, trend_l=trend_l)
        # How much of the climb arrived in the last quarter of the run: the
        # cheap, always-printed version of "is 800 steps enough?". The
        # expensive version is --steps 2000, which moves no RMSE here by more
        # than 0.001 ppm.
        tail = lml_best - max(hist[:-len(hist) // 4])
        print(f"  {label:<26} {lml0:7.1f} ->{lml_best:8.1f}  {rmse:8.2f}  "
              f"{cover:7.2f}  {trend_l:6.0f} yr  {tail:+9.2f}")

    base, rq, rbf = rows[None], rows["rq"], rows["rbf"]
    print(
        "\n  The negative result stands, weakened. ML-II on the shipped kernel still\n"
        "  extrapolates WORSE than the unoptimized init ({:.2f} vs {:.2f} ppm): it buys\n"
        "  in-sample evidence by SHORTENING the trend to {:.0f} yr, and an RBF trend\n"
        "  mean-reverts beyond its lengthscale, so it undershoots an {:.0f}-yr rise.\n"
        "  A medium-term term confirms that diagnosis mechanically -- with one\n"
        "  present the evidence now prefers a trend of {:.0f} yr, LONGER than the\n"
        "  hand-set {:.0f}, and the RMSE falls {:.1f}x to {:.2f} ppm.\n"
        "\n  But it is not the scale mixture that does it. A plain RBF in the same\n"
        "  slot, same init, lands at {:.2f} ppm against RationalQuadratic's {:.2f} --\n"
        "  a {:.0f}% difference. What was missing was a medium-term component of ANY\n"
        "  shape, not a heavy tail. The evidence can tell them apart ({:+.1f} nats for\n"
        "  RQ) and the 11-year forecast cannot.\n"
        "\n  And none of it fixes the calibration: 95% coverage is {:.2f} for the best\n"
        "  ML-II fit against {:.2f} for the hand-set init. ML-II maximizes evidence,\n"
        "  not forecast skill, and that is still the result."
        .format(base["rmse"], ref_rmse, base["trend_l"], horizon,
                rq["trend_l"], ref_trend_l, base["rmse"] / rq["rmse"], rq["rmse"],
                rbf["rmse"], rq["rmse"], 100 * abs(rbf["rmse"] - rq["rmse"]) / rq["rmse"],
                rq["lml"] - rbf["lml"], rq["cover"], ref_cover)
    )

    # ---- refit on all data, extrapolate to 2040 ----
    # The RQ composite: better evidence AND better hold-out than the base one,
    # so it is what the headline figure should be showing.
    model_all, mean_all, _ = fit(t, y, steps=steps, medium="rq")
    t_star = np.linspace(t.min(), 2040.0, 2000)
    mu_s, var_s = model_all.predict(t_star[:, None], include_noise=True)
    mu_s = mu_s + mean_all
    sd_s = np.sqrt(var_s)

    # theta reports FREE params only -- the period is frozen and printed apart.
    names = (["trend s2", "trend l", "per s2", "per l", "decay s2", "decay l",
              "med s2", "med l", "med alpha", "short s2", "short l"])
    learned = np.exp(model_all.kernel.theta)
    assert len(names) == learned.size, (len(names), learned.size)
    print("learned hyperparameters:")
    for n, v in zip(names, learned):
        print(f"  {n:>9}: {v:10.4f}")
    print(f"  {'per p':>9}: {1.0:10.4f}   (fixed at 1 yr)")
    print(f"  {'noise s2':>9}: {model_all.noise_var:10.4f}")

    fig, axes = plt.subplots(
        1, 2, figsize=(10, 3.6), constrained_layout=True,
        gridspec_kw={"width_ratios": [2.2, 1.0]},
    )
    ax = axes[0]
    ax.plot(t, y, ".", ms=1.2, alpha=0.5, label="NOAA monthly mean")
    ax.plot(t_star, mu_s, lw=0.9, color="C1", label="GP mean")
    ax.fill_between(t_star, mu_s - 1.96 * sd_s, mu_s + 1.96 * sd_s,
                    color="C1", alpha=0.25, lw=0, label="95% predictive")
    ax.axvline(t.max(), color="gray", ls=":", lw=1)
    ax.set_xlabel("year")
    ax.set_ylabel(r"CO$_2$ (ppm)")
    ax.set_title("Mauna Loa CO$_2$: trend + season + medium-term + short-term",
                 loc="left")
    ax.legend(loc="upper left", fontsize=7)

    ax = axes[1]
    zoom = t_star > 2022
    ax.plot(t_star[zoom], mu_s[zoom], lw=1.0, color="C1")
    ax.fill_between(t_star[zoom], (mu_s - 1.96 * sd_s)[zoom],
                    (mu_s + 1.96 * sd_s)[zoom], color="C1", alpha=0.25, lw=0)
    recent = t > 2022
    ax.plot(t[recent], y[recent], ".", ms=2.5, alpha=0.7)
    ax.axvline(t.max(), color="gray", ls=":", lw=1)
    ax.set_xlabel("year")
    ax.set_title("The forecast keeps the seasons", loc="left")
    savefig(fig, "co2_forecast.png")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Mauna Loa CO2 GP experiment")
    ap.add_argument("--steps", type=int, default=800,
                    help="Adam steps per ML-II fit (800 ships; 2000 is the "
                         "convergence check the README quotes)")
    main(**vars(ap.parse_args()))
