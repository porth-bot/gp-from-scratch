"""Experiment 2: Mauna Loa CO2 -- structured kernels on real data.

The classic GP demonstration (Rasmussen & Williams 2006, Sec. 5.4.3): monthly
CO2 concentrations are a smooth rising trend + an annual cycle whose shape
drifts slowly + short-term weather noise. Kernels compose by addition
(independent additive processes), so the model is read off the physics:

    k = RBF_trend + Periodic * RBF_decay + Matern32_short      (+ noise)

The Periodic * RBF product is the key move: exactly-periodic correlation
modulated by a long-lengthscale RBF, i.e. "seasonal, but this year's cycle
resembles next year's more than one 30 years out".

Honest evaluation FIRST: fit on data up to 2015.0 only, forecast the held-out
2015-2026 months (true out-of-sample -- the model never sees them), report
RMSE and 95% coverage. Then refit on everything for the 2040 extrapolation
figure.

Run:  python experiments/co2.py    (~2 min: ML-II on n~700 with 10 params)
"""

import numpy as np

from common import plt, savefig
from gp.gp import GPRegressor
from gp.kernels import RBF, Matern, Periodic
from gp.optimize import adam_maximize

DATA = "data/co2_mm_mlo.txt"


def load_co2():
    raw = np.loadtxt(DATA, comments="#")
    t, ppm = raw[:, 2], raw[:, 3]
    keep = ppm > 0  # missing months are flagged negative
    return t[keep], ppm[keep]


def make_kernel():
    trend = RBF(s2=50.0**2, l=40.0)
    seasonal = Periodic(s2=4.0, l=1.3, p=1.0) * RBF(s2=1.0, l=90.0)
    short = Matern(nu=1.5, s2=0.5, l=1.0)
    return trend + seasonal + short


def fit(t, y, steps=350):
    X = t[:, None]
    y_mean = y.mean()
    model = GPRegressor(make_kernel(), noise_var=0.05)
    best, hist = adam_maximize(
        lambda p: model.lml_and_grad(X, y - y_mean, p),
        model.params, lr=0.03, steps=steps,
    )
    model.params = best
    model.fit(X, y - y_mean)
    return model, y_mean, hist


def main():
    t, y = load_co2()
    print(f"{len(t)} monthly observations, {t.min():.1f} - {t.max():.1f}")

    # ---- honest out-of-sample test: train < 2015, predict >= 2015 ----
    train = t < 2015.0
    model, y_mean, _ = fit(t[train], y[train])
    mu, var = model.predict(t[~train][:, None], include_noise=True)
    resid = y[~train] - (mu + y_mean)
    rmse = float(np.sqrt(np.mean(resid**2)))
    cover = float(np.mean(np.abs(resid) < 1.96 * np.sqrt(var)))
    horizon = t[~train].max() - 2015.0
    print(f"held-out 2015-{t.max():.1f} ({horizon:.1f} yr): "
          f"RMSE = {rmse:.2f} ppm, 95% coverage = {cover:.2f}")

    # ---- refit on all data, extrapolate to 2040 ----
    model_all, mean_all, _ = fit(t, y)
    t_star = np.linspace(t.min(), 2040.0, 2000)
    mu_s, var_s = model_all.predict(t_star[:, None], include_noise=True)
    mu_s = mu_s + mean_all
    sd_s = np.sqrt(var_s)

    names = (["trend s2", "trend l", "per s2", "per l", "per p",
              "decay s2", "decay l", "short s2", "short l"])
    learned = np.exp(model_all.kernel.theta)
    print("learned hyperparameters:")
    for n, v in zip(names, learned):
        print(f"  {n:>9}: {v:10.4f}")
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
    ax.set_title("Mauna Loa CO$_2$: trend + drifting seasonality + short-term",
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
    main()
