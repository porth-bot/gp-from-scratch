"""Two ways to spend a rank budget: random features against inducing points.

Sections 8 and 9 both replace the n x n Gram matrix with something of rank R,
and both cost O(n R^2). They differ in *where the rank goes*. RFF picks R basis
functions before seeing the data -- sinusoids spanning the whole domain, drawn
from the kernel's spectral density. SGPR picks R basis functions k(., z_m) that
are local and sited on the input space. Section 10 measured what the first
choice costs: across a gap in the data the posterior collapses onto a
confidently wrong answer. This script puts the second choice on the same axes,
in the same setting, and asks whether the rank was the problem or the basis was.

The setting is imported from ``rff.py`` rather than restated, so "the same
setup" is a fact about the code and not a claim in a docstring: same target,
same n = 3000 with a gap in |x| < 1.2, same kernel hyperparameters, same noise.
Every model here is given the *same fixed* hyperparameters -- the ones the RFF
comparison used -- so the only difference between arms is the approximation.

**Nothing is optimized.** Z is placed by three cheap rules and left alone
(Day 3 already measured what ML-II over Z is worth). That is deliberate: if
inducing points need tuning to beat random features, the comparison is between
a tuned method and an untuned one, and it is worth knowing whether the win
survives without it.

Four measurements:

1. **The error bar across the gap, against rank.** The headline. Both methods
   at matched R, scored against the exact posterior sd at the gap centre.
   RFF is averaged over draws of the feature map *and* draws of the data,
   because Sec. 10's original single-draw number turned out to be a tail.

2. **Matched cost, measured.** Wall-clock for fit + predict at each rank, then
   the error-bar error plotted against the seconds it cost rather than against
   R. Matched *rank* and matched *cost* are not the same comparison and the
   answer differs between them, so both are reported.

3. **Where the mean is wrong, not just the error bar.** Split by region: in
   the gap versus where the data are.

4. **Where inducing points fail too.** The counterweight, and it is a real
   one. The win in (1) belongs to the *placement rule*, not to inducing points
   as such: with Z on a blind uniform grid at M = 8-16, SGPR throws away a
   large part of the error bar in the gap in the same way RFF does (though not
   as much of it -- RFF is still worse at every matched rank). The quantile
   rule never does. The mechanism is DTC's deterministic training conditional,
   and it is diagnosed here rather than asserted.

Run:  python experiments/rff_vs_sparse.py     (~90 s)
"""

import time

import numpy as np

from common import savefig
import matplotlib.pyplot as plt

from rff import LENGTHSCALE, NOISE_VAR, S2, dataset

from gp.gp import GPRegressor
from gp.kernels import RBF
from gp.rff import RFFMap, RFFRegressor
from gp.sparse import SGPR

N = 3000
GAP = 1.2
RANKS = (4, 8, 12, 16, 24, 32, 64, 128, 256, 512, 1024, 2048)
DATA_SEEDS = 5
RFF_DRAWS = 9
GRID = np.linspace(-4.0, 4.0, 400).reshape(-1, 1)
TOL = 0.05                      # "close enough" for the cost-to-accuracy table


def kernel():
    return RBF(s2=S2, l=LENGTHSCALE)


# -- inducing-point placement rules, all data-blind or nearly so --------------


def z_quantile(X, M):
    """Data quantiles: uniform in data mass. Puts nothing inside the gap."""
    return np.quantile(X[:, 0], np.linspace(0.0, 1.0, M)).reshape(-1, 1)


def z_grid(X, M):
    """A uniform grid over the domain. Ignores the data entirely, so it *does*
    put inducing points inside the gap -- which is measurement 4's subject."""
    return np.linspace(-4.0, 4.0, M).reshape(-1, 1)


def z_subset(X, M):
    """A random subset of the inputs -- the standard cheap default."""
    idx = np.random.default_rng(0).choice(len(X), size=M, replace=False)
    return X[np.sort(idx)]


Z_RULES = {"quantile": z_quantile, "grid": z_grid, "subset": z_subset}


# -- scoring ------------------------------------------------------------------


def score(mean, var, mean_exact, sd_exact, in_gap, centre):
    """Every arm is scored against the exact posterior, never against another
    approximation. Signed deficit, because the direction matters: negative is
    overconfident (claims to know more than the exact GP does) and positive is
    conservative, and only one of those is a safety failure."""
    sd = np.sqrt(var)
    mean_err = np.abs(mean - mean_exact)
    return {
        "sd_centre": float(sd[centre]),
        "sd_ratio": float(sd[centre] / sd_exact[centre]),
        "worst_deficit": float((sd - sd_exact).min()),
        "mean_err_gap": float(mean_err[in_gap].max()),
        "mean_err_data": float(mean_err[~in_gap].max()),
        "sd_err_data": float(np.abs(sd - sd_exact)[~in_gap].max()),
    }


def sweep():
    """Both methods at every rank, over DATA_SEEDS datasets."""
    xs = GRID.ravel()
    in_gap = np.abs(xs) < GAP
    centre = int(np.argmin(np.abs(xs)))

    rows = {"rff": []}
    for name in Z_RULES:
        rows[("sgpr", name)] = []

    exact_sd_centre = []
    for seed in range(DATA_SEEDS):
        X, y = dataset(N, np.random.default_rng(seed), gap=GAP)
        mean_exact, var_exact = (GPRegressor(kernel(), noise_var=NOISE_VAR)
                                 .fit(X, y).predict(GRID))
        sd_exact = np.sqrt(var_exact)
        exact_sd_centre.append(float(sd_exact[centre]))

        for R in RANKS:
            for draw in range(RFF_DRAWS):
                phi = RFFMap(R, LENGTHSCALE, S2, 1, np.random.default_rng(draw))
                m, v = RFFRegressor(phi, noise_var=NOISE_VAR).fit(X, y).predict(GRID)
                rows["rff"].append(
                    dict(rank=R, seed=seed, draw=draw,
                         **score(m, v, mean_exact, sd_exact, in_gap, centre)))
            for name, rule in Z_RULES.items():
                m, v = (SGPR(kernel(), rule(X, R), noise_var=NOISE_VAR)
                        .fit(X, y).predict(GRID))
                rows[("sgpr", name)].append(
                    dict(rank=R, seed=seed,
                         **score(m, v, mean_exact, sd_exact, in_gap, centre)))
    return rows, float(np.mean(exact_sd_centre))


def collect(rows, key, field):
    """median / min / max of ``field`` at each rank."""
    out = []
    for R in RANKS:
        vals = [r[field] for r in rows[key] if r["rank"] == R]
        out.append((float(np.median(vals)), float(np.min(vals)),
                    float(np.max(vals))))
    return np.array(out)


# -- 2. wall clock ------------------------------------------------------------


def timings(repeats=3):
    """fit + predict at each rank, one dataset, best of ``repeats``.

    Best-of rather than mean: the quantity wanted is the cost of the linear
    algebra, and every source of noise here (scheduling, turbo, other load)
    only ever adds time.
    """
    X, y = dataset(N, np.random.default_rng(0), gap=GAP)
    t_rff, t_sgpr = [], []
    for R in RANKS:
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            phi = RFFMap(R, LENGTHSCALE, S2, 1, np.random.default_rng(0))
            RFFRegressor(phi, noise_var=NOISE_VAR).fit(X, y).predict(GRID)
            ts.append(time.perf_counter() - t0)
        t_rff.append(min(ts))

        Z = z_quantile(X, R)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            SGPR(kernel(), Z, noise_var=NOISE_VAR).fit(X, y).predict(GRID)
            ts.append(time.perf_counter() - t0)
        t_sgpr.append(min(ts))

    t0 = time.perf_counter()
    GPRegressor(kernel(), noise_var=NOISE_VAR).fit(X, y).predict(GRID)
    t_exact = time.perf_counter() - t0
    return np.array(t_rff), np.array(t_sgpr), t_exact


def first_within(ok, times):
    """Smallest rank satisfying ``ok(i)``, and what it cost. None if no rank in
    the sweep gets there."""
    for i, R in enumerate(RANKS):
        if ok(i):
            return i, R, float(times[i])
    return None


def cost_to_accuracy(curves, defic, rows, exact_sd, span, t_rff, t_sgpr):
    """Cost to reach a *fixed accuracy*, with accuracy defined on both moments.

    The error bar alone is not a usable criterion, and the reason is worth
    stating rather than hiding behind a threshold: at R = 4 or 8 SGPR reports
    the sd at the gap centre as 1.0000, which is within 5% of the exact 0.9655
    -- but only because Q** is ~0 there, so the model has returned the *prior*.
    It is right about the error bar for the reason a model that knows nothing
    is right, and its posterior mean at that rank is off by 1.35 on a signal
    spanning 3.0. Scoring on the sd alone would hand the comparison to whichever
    method degrades toward the prior fastest, which is not a virtue.

    So a rank counts only if it gets *both*: the error bar at the gap centre
    within TOL of exact, and the posterior mean within TOL of the exact
    posterior mean everywhere, relative to the signal's own span. Both
    components are reported separately as well, so the joint number is not a
    threshold doing the arguing.
    """
    out = {}
    for label, key, times in (("RFF", "rff", t_rff),
                              ("SGPR (quantile Z)", ("sgpr", "quantile"), t_sgpr)):
        sd_c = curves[key]
        gap = collect(rows, key, "mean_err_gap")
        dat = collect(rows, key, "mean_err_data")
        mean_err = np.maximum(gap[:, 0], dat[:, 0])
        sd_ok = np.abs(sd_c[:, 0] - exact_sd) / exact_sd <= TOL
        mean_ok = mean_err <= TOL * span
        out[label] = {
            "sd_only": first_within(lambda i: sd_ok[i], times),
            "mean_only": first_within(lambda i: mean_ok[i], times),
            "both": first_within(lambda i: sd_ok[i] and mean_ok[i], times),
            "sd": sd_c[:, 0], "mean_err": mean_err, "times": times,
        }
    return out


# -- 4. the counterweight: where inducing points lose the error bar too -------


def dtc_diagnosis(M=12):
    """Decompose the SGPR predictive variance at the gap centre.

        var = k** - Q**  +  k*u Sigma ku*,      Sigma = (Kuu + s^-2 Kuf Kfu)^-1

    The first pair is the safety margin -- how much prior variance the inducing
    set fails to explain at x*, which is what makes the band relax when x* is
    far from every z. The second is the posterior variance of the inducing
    values themselves, read out at x*.

    The pathology is what happens when an inducing point sits *at* x* in a
    region with no data. Then Q** -> k**, the safety margin goes to zero, and
    the entire error bar is inherited from the posterior over u. Under DTC that
    posterior is far too tight, because the training conditional p(f | u) is
    taken to be deterministic: every one of the n observations is treated as a
    noise-only-corrupted linear readout of M numbers, so u is over-determined
    by data that is not actually near it. The exact GP makes no such claim.
    """
    X, y = dataset(N, np.random.default_rng(0), gap=GAP)
    _, var_exact = GPRegressor(kernel(), noise_var=NOISE_VAR).fit(X, y).predict(GRID)
    centre = int(np.argmin(np.abs(GRID.ravel())))
    xstar = GRID[centre:centre + 1]
    k = kernel()

    out = {"sd_exact": float(np.sqrt(var_exact[centre])), "M": M, "rows": []}
    for name in ("quantile", "grid"):
        Z = Z_RULES[name](X, M)
        model = SGPR(k, Z, noise_var=NOISE_VAR).fit(X, y)
        _, v = model.predict(xstar)

        Kuu = k(Z, Z) + model.jitter * np.eye(M)
        Kus = k(Z, xstar)
        Kuf = k(Z, X)
        kss = float(k(xstar, xstar)[0, 0])
        Qss = float((Kus.T @ np.linalg.solve(Kuu, Kus))[0, 0])
        Sigma = np.linalg.inv(Kuu + Kuf @ Kuf.T / NOISE_VAR)
        u_term = float((Kus.T @ Sigma @ Kus)[0, 0])
        near = int(np.argmax(Kus[:, 0]))
        out["rows"].append({
            "rule": name,
            "sd": float(np.sqrt(v[0])),
            "kss": kss, "Qss": Qss, "margin": kss - Qss, "u_term": u_term,
            "dist_to_nearest_Z": float(np.abs(Z[:, 0]).min()),
            "nearest_z": float(Z[near, 0]),
            # how much data the dominant inducing point is told it explains
            "data_weight": float(np.sum(Kuf[near] ** 2) / NOISE_VAR),
        })
    return out


def main():
    xs = GRID.ravel()
    in_gap = np.abs(xs) < GAP
    centre = int(np.argmin(np.abs(xs)))

    print(f"setting: n={N}, gap in |x|<{GAP}, RBF(s2={S2}, l={LENGTHSCALE}), "
          f"noise={NOISE_VAR}, {DATA_SEEDS} datasets, {RFF_DRAWS} feature "
          f"draws each\nhyperparameters are fixed and identical in every arm; "
          f"Z is never optimized\n")

    rows, exact_sd = sweep()
    print(f"exact GP posterior sd at the gap centre: {exact_sd:.4f}\n")

    print("1. error bar across the gap: sd at the centre, median over "
          "datasets (and feature draws)")
    print(f"{'rank':>6} | {'RFF':>22} | {'SGPR quantile Z':>16} | "
          f"{'SGPR grid Z':>12} | {'SGPR subset Z':>13}")
    curves = {k: collect(rows, k, "sd_centre")
              for k in (["rff"] + [("sgpr", n) for n in Z_RULES])}
    for i, R in enumerate(RANKS):
        r = curves["rff"][i]
        print(f"{R:>6} | {r[0]:>8.4f} [{r[1]:.3f}, {r[2]:.3f}] | "
              f"{curves[('sgpr', 'quantile')][i, 0]:>16.4f} | "
              f"{curves[('sgpr', 'grid')][i, 0]:>12.4f} | "
              f"{curves[('sgpr', 'subset')][i, 0]:>13.4f}")

    print("\n   worst overconfidence anywhere on the grid, min(sd_approx - "
          "sd_exact), median over datasets")
    defic = {k: collect(rows, k, "worst_deficit")
             for k in (["rff"] + [("sgpr", n) for n in Z_RULES])}
    for i, R in enumerate(RANKS):
        print(f"{R:>6} | {defic['rff'][i, 0]:>22.4f} | "
              f"{defic[('sgpr', 'quantile')][i, 0]:>16.4f} | "
              f"{defic[('sgpr', 'grid')][i, 0]:>12.4f} | "
              f"{defic[('sgpr', 'subset')][i, 0]:>13.4f}")

    print("\n2. wall-clock, fit + predict (n=3000, best of 3)")
    t_rff, t_sgpr, t_exact = timings()
    print(f"    exact GP: {t_exact:.3f}s")
    for i, R in enumerate(RANKS):
        print(f"    R={R:>5}: RFF {t_rff[i]:.4f}s   SGPR {t_sgpr[i]:.4f}s   "
              f"SGPR/RFF {t_sgpr[i] / t_rff[i]:.1f}x")

    X0, y0 = dataset(N, np.random.default_rng(0), gap=GAP)
    m0, _ = GPRegressor(kernel(), noise_var=NOISE_VAR).fit(X0, y0).predict(GRID)
    span = float(m0.max() - m0.min())
    c2a = cost_to_accuracy(curves, defic, rows, exact_sd, span, t_rff, t_sgpr)
    print(f"\n   cost to reach a fixed accuracy: error bar within {TOL:.0%} of "
          f"exact AND\n   posterior mean within {TOL:.0%} of the signal span "
          f"({span:.2f}, so {TOL * span:.3f}) everywhere")
    for label, d in c2a.items():
        parts = []
        for crit in ("sd_only", "mean_only", "both"):
            hit = d[crit]
            parts.append(f"{crit}: " + ("never" if hit is None
                                        else f"R={hit[1]} at {hit[2]:.4f}s"))
        print(f"    {label:20s} " + "   ".join(parts))
        hit = d["both"]
        if hit is not None:
            print(f"    {'':20s} -> {hit[2] / t_exact:.3f}x the exact GP's "
                  f"{t_exact:.3f}s")
    both = {k: v["both"] for k, v in c2a.items()}
    if both["RFF"] is None and both["SGPR (quantile Z)"] is not None:
        hit = both["SGPR (quantile Z)"]
        print(f"    RFF does not reach it at any rank up to {RANKS[-1]}; its "
              f"gap mean error is still\n    "
              f"{c2a['RFF']['mean_err'][-1]:.3f} there, falling by only "
              f"{c2a['RFF']['mean_err'][RANKS.index(64)] / c2a['RFF']['mean_err'][-1]:.1f}x "
              f"over a 32x rank increase, so the rank it would\n    need is far "
              f"outside this sweep -- and R={hit[1]} already costs SGPR "
              f"{hit[2]:.4f}s.")

    print("\n3. max |mean error|, in the gap vs where the data are "
          "(median over datasets)")
    for label, key in (("RFF", "rff"), ("SGPR quantile Z", ("sgpr", "quantile"))):
        g = collect(rows, key, "mean_err_gap")
        d = collect(rows, key, "mean_err_data")
        print(f"    {label}")
        for i, R in enumerate(RANKS):
            print(f"      R={R:>5}: gap {g[i, 0]:.4f}   data {d[i, 0]:.4f}   "
                  f"ratio {g[i, 0] / max(d[i, 0], 1e-12):>8.1f}x")

    print("\n4. the counterweight: SGPR loses the error bar too, and here is "
          "why")
    diag = dtc_diagnosis()
    print(f"    at the gap centre, M={diag['M']}, exact sd "
          f"{diag['sd_exact']:.4f}")
    for r in diag["rows"]:
        print(f"    Z rule '{r['rule']:>8}': sd {r['sd']:.4f}   "
              f"nearest z at {r['nearest_z']:+.3f}   "
              f"k** {r['kss']:.4f}  Q** {r['Qss']:.4f}  "
              f"margin k**-Q** {r['margin']:.4f}  u-term {r['u_term']:.4f}")
    print("    (the grid rule puts an inducing point in the gap, which spends "
          "the\n     safety margin k**-Q** and leaves the band to be inherited "
          "from a DTC\n     posterior over u that the data outside the gap has "
          "over-determined.)")

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), constrained_layout=True)

    # (a) the picture, at matched rank 64
    X, y = dataset(N, np.random.default_rng(0), gap=GAP)
    mean_exact, var_exact = (GPRegressor(kernel(), noise_var=NOISE_VAR)
                             .fit(X, y).predict(GRID))
    sds = []
    preds = []
    for draw in range(RFF_DRAWS):
        phi = RFFMap(64, LENGTHSCALE, S2, 1, np.random.default_rng(draw))
        preds.append(RFFRegressor(phi, noise_var=NOISE_VAR).fit(X, y).predict(GRID))
        sds.append(np.sqrt(preds[-1][1][centre]))
    rff64 = preds[int(np.argsort(sds)[RFF_DRAWS // 2])]
    sgpr64 = (SGPR(kernel(), z_quantile(X, 64), noise_var=NOISE_VAR)
              .fit(X, y).predict(GRID))

    ax = axes[0, 0]
    ax.plot(X.ravel(), y, ".", color="0.78", ms=1.2, zorder=0)
    for (mean, var), color, label, ls in (
            (sgpr64, "C0", "SGPR, $M=64$", "-"),
            (rff64, "C3", "RFF, $D=64$ (median draw)", "-"),
            ((mean_exact, var_exact), "k", "exact GP", "--")):
        sd = np.sqrt(var)
        ax.fill_between(xs, mean - 1.96 * sd, mean + 1.96 * sd, color=color,
                        alpha=0.16, lw=0)
        ax.plot(xs, mean, color=color, lw=1.4, ls=ls, label=label, zorder=3)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("1. Same rank, same cost order, same data", loc="left")
    ax.legend(fontsize=7.5, loc="upper center", ncol=3)

    # (b) sd at the gap centre against rank
    ax = axes[0, 1]
    ax.axhline(exact_sd, color="k", ls="--", lw=1.2, label="exact GP")
    c = curves["rff"]
    ax.plot(RANKS, c[:, 0], "o-", color="C3", lw=1.6, ms=4, label="RFF (median)")
    ax.fill_between(RANKS, c[:, 1], c[:, 2], color="C3", alpha=0.15,
                    label=f"RFF, {DATA_SEEDS}x{RFF_DRAWS} runs")
    ax.plot(RANKS, curves[("sgpr", "quantile")][:, 0], "o-", color="C0",
            lw=1.6, ms=4, label="SGPR, quantile Z")
    ax.plot(RANKS, curves[("sgpr", "grid")][:, 0], "s--", color="C1", lw=1.4,
            ms=3.5, label="SGPR, blind grid Z")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("rank $R$ ($D$ features, or $M$ inducing points)")
    ax.set_ylabel("posterior sd at the gap centre")
    ax.set_title("2. The error bar in the gap, against rank", loc="left")
    ax.legend(fontsize=7.5, loc="lower right")

    # (c) matched cost
    ax = axes[1, 0]
    err_rff = np.abs(curves["rff"][:, 0] - exact_sd)
    err_sg = np.abs(curves[("sgpr", "quantile")][:, 0] - exact_sd)
    ax.plot(t_rff, err_rff, "o-", color="C3", lw=1.6, ms=4, label="RFF")
    ax.plot(t_sgpr, err_sg, "o-", color="C0", lw=1.6, ms=4, label="SGPR, quantile Z")
    for i, R in enumerate(RANKS):
        if R in (16, 64, 256, 2048):
            ax.annotate(f"{R}", (t_rff[i], err_rff[i]), fontsize=6.5,
                        color="C3", xytext=(2, 4), textcoords="offset points")
            ax.annotate(f"{R}", (t_sgpr[i], err_sg[i]), fontsize=6.5,
                        color="C0", xytext=(2, -8), textcoords="offset points")
    ax.axhline(TOL * exact_sd, color="0.5", ls=":", lw=1.1,
               label=f"{TOL:.0%} of the exact sd")
    ax.axvline(t_exact, color="k", ls="--", lw=1.1, label="exact GP's cost")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("fit + predict, seconds")
    ax.set_ylabel("|sd at gap centre $-$ exact|")
    ax.set_title("3. Matched cost, not matched rank", loc="left")
    ax.legend(fontsize=7.5)

    # (d) the counterweight
    ax = axes[1, 1]
    M = diag["M"]
    ax.plot(xs, np.sqrt(var_exact), "k--", lw=1.4, label="exact GP")
    for name, color in (("quantile", "C0"), ("grid", "C1")):
        Z = Z_RULES[name](X, M)
        _, v = SGPR(kernel(), Z, noise_var=NOISE_VAR).fit(X, y).predict(GRID)
        ax.plot(xs, np.sqrt(v), color=color, lw=1.5,
                label=f"SGPR, {name} Z")
        ax.plot(Z[:, 0], np.full(M, -0.04 if name == "grid" else -0.09),
                "|", color=color, ms=7, mew=1.4)
    ax.axvspan(-GAP, GAP, color="0.9", zorder=0)
    ax.annotate("gap", xy=(0.0, 1.02), fontsize=7.5, color="0.4", ha="center")
    ax.set_xlabel("x   (ticks below the axis mark the inducing points)")
    ax.set_ylabel("posterior sd")
    ax.set_title(f"4. Where inducing points fail too ($M={M}$)", loc="left")
    ax.legend(fontsize=7.5, loc="center right")

    savefig(fig, "rff_vs_sparse.png")
    return curves, defic, (t_rff, t_sgpr, t_exact), diag


if __name__ == "__main__":
    main()
