"""Sparse GPs: does the bound converge, and where do the inducing points go?

Section 9 derived the Titsias bound and Day 2 checked its gradients. This is
the first script that *drives* them: joint ML-II over kernel hyperparameters,
the noise, and the inducing locations Z (`gp.optimize.maximize_elbo`).

The setting is chosen so that ground truth is free. n = 800 in one dimension is
small enough that the exact GP still runs, so every approximate quantity is
scored against the real thing rather than against another approximation:

- the bound F(M) against the exact log marginal likelihood, which it must
  approach *from below* -- the ceiling is a theorem (Sec. 9.2), so a crossing
  would be an algebra bug, not a good result;
- the sparse posterior mean and sd against the exact posterior, on a grid.

The target is mixed-scale on purpose: slow on the left, five times faster on
the right, with the data uniform over both halves. That decouples the two
reasons an inducing point might cluster somewhere. Data density is flat, so any
asymmetry in the learned Z is the *function* being harder on the right, not
there being more of it.

Five measurements:

1. **Convergence in M**, optimized Z against Z frozen on the data quantiles.
   The gap to the exact evidence, on a log axis, is the honest version of
   "the bound gets tight".
2. **Posterior error in M**, mean and sd separately. The sd is the one that
   matters: Sec. 10's random-features result was a variance failure that the
   mean never showed.
3. **Where the inducing points end up**, per M, against the 50/50 split a
   purely geometric method would give. The expectation going in was that they
   would cluster in the fast half. They barely do: 56% of them at M=64, a
   spacing ratio of 1.33x for a 5x frequency ratio, and 1.0-1.1x at every
   smaller M. Two follow-ups try to say why, and one of them fails.
4. **A density control** -- same target, data drawn 3:1 toward the *slow*
   half. It reverses the sign of the asymmetry (41% in the fast half, Z 1.44x
   denser on the slow side), so a 3:1 density skew outweighs the entire
   difficulty effect. Alongside it, a finite-difference split of dF/dZ into
   its two terms, written to check whether the force on Z is simply blind to
   y -- the trace penalty is a sum of conditional variances and never sees the
   targets. It is not blind: the y-aware DTC term carries 4.5x the force at
   the initialization. That hypothesis is refuted, and the reading that
   survives is the unglamorous one -- a *stationary* kernel has one
   lengthscale for both halves, so the resolution Z have to sample at is the
   same on both sides no matter how the function behaves. That reading is an
   interpretation, not a measurement; the repo has no non-stationary kernel
   with an input derivative to test it against (Gibbs has no dK_dX1).
5. **What optimizing Z is worth in units of M** -- how many frozen inducing
   points it takes to match an optimized set of a given size.

Run:  python experiments/sparse.py     (~70 s)
"""

import time

import numpy as np

from common import savefig
import matplotlib.pyplot as plt

from gp.gp import GPRegressor
from gp.kernels import RBF
from gp.optimize import adam_maximize, maximize_elbo
from gp.sparse import SGPR

N = 800
NOISE_SD = 0.1
LO, HI = -4.0, 4.0
MS = (2, 4, 8, 12, 16, 24, 32, 48, 64)
STEPS = 400
LR = 0.05
INIT = dict(s2=0.5, l=2.0)          # deliberately not the truth, for both arms
INIT_NOISE = 0.3


def target(x):
    """Slow everywhere, plus a five-times-faster component on x > 0 only.

    The discontinuity in the *derivative structure* at x = 0 is the point: a
    single stationary lengthscale has to serve both halves, so the exact GP
    pays for the fast half everywhere, and the inducing set is the only thing
    in the model that can spend its budget unevenly.
    """
    return np.sin(1.1 * x) + 0.6 * np.sin(5.0 * x) * (x > 0)


def dataset(seed=0, slow_share=0.5):
    """n samples of the target; ``slow_share`` of them from the slow half.

    ``slow_share=0.5`` is uniform over [LO, HI] and is the main setting. The
    skewed version is measurement 4's control: it changes the data density
    without touching the function, which is exactly the confound measurement 3
    cannot resolve on its own.
    """
    rng = np.random.default_rng(seed)
    n_slow = int(round(N * slow_share))
    X = np.concatenate([rng.uniform(LO, 0.0, n_slow),
                        rng.uniform(0.0, HI, N - n_slow)])
    X = np.sort(X).reshape(-1, 1)
    y = target(X[:, 0]) + NOISE_SD * rng.standard_normal(N)
    return X, y


def quantile_Z(X, M):
    """Inducing points at the data quantiles -- the standard fixed baseline.

    Uniform-in-data-mass rather than uniform-in-x, so the frozen arm is not
    handicapped by the data being unevenly spread (here it is not, but the
    choice should not depend on that).
    """
    return np.quantile(X[:, 0], np.linspace(0.0, 1.0, M)).reshape(-1, 1)


# -- 0. the exact GP: ground truth for everything below -----------------------


def exact_fit(X, y):
    model = GPRegressor(RBF(**INIT), noise_var=INIT_NOISE)
    best, _ = adam_maximize(lambda p: model.lml_and_grad(X, y, p), model.params,
                            lr=LR, steps=STEPS)
    model.params = best
    return model.fit(X, y)


# -- 1-3. the sweep over M ----------------------------------------------------


def sweep(X, y, Xs, exact, Ms=MS):
    """Joint ML-II at each M, with Z free and with Z frozen at the quantiles."""
    mean_exact, var_exact = exact.predict(Xs)
    sd_exact = np.sqrt(var_exact)
    lml = exact.log_marginal_likelihood()

    rows = {True: [], False: []}
    Zs = {}
    for M in Ms:
        for optimize_Z in (False, True):
            model = SGPR(RBF(**INIT), Z=quantile_Z(X, M), noise_var=INIT_NOISE)
            t0 = time.perf_counter()
            res = maximize_elbo(model, X, y, optimize_Z=optimize_Z,
                                lr=LR, steps=STEPS)
            secs = time.perf_counter() - t0
            mean, var = model.predict(Xs)
            rows[optimize_Z].append((
                float(lml - res.elbo),
                float(np.abs(mean - mean_exact).max()),
                float(np.abs(np.sqrt(var) - sd_exact).max()),
                float(res.trace_term),
                float(np.exp(res.params[1])),            # fitted lengthscale
                float(np.exp(res.params[-1])),           # fitted noise variance
                secs,
            ))
            if optimize_Z:
                Zs[M] = np.sort(res.Z.ravel())
    return {k: np.array(v) for k, v in rows.items()}, Zs


def matching_M(free, frozen, Ms=MS):
    """How many frozen inducing points buy an optimized set of size M.

    Reported as a bracket rather than an interpolated number: the frozen sweep
    is on the same coarse grid of M, so the honest statement is "somewhere
    between these two", not a fitted crossing.
    """
    out = []
    for i, M in enumerate(Ms):
        target_gap = free[i, 0]
        hits = [Ms[j] for j in range(len(Ms)) if frozen[j, 0] <= target_gap]
        out.append(min(hits) if hits else None)
    return out


def spacing(Z, side):
    """Median gap between consecutive inducing points on one side of x = 0.

    The density read-out that matters: counting how many Z sit in a half
    conflates a half that is twice as wide with one that is twice as busy,
    and here the halves are the same width but need not be equally busy.
    """
    part = Z[Z > 0] if side == "fast" else Z[Z <= 0]
    return float(np.median(np.diff(part))) if part.size > 1 else float("nan")


def force_split(model, X, y, eps=1e-5):
    """Split dF/dZ into its two terms, by finite differences on each.

    F = DTC - tr(Kff - Qff) / (2 sigma^2), so the force on an inducing point is
    a sum of two forces, and measurement 3 turns on which one dominates. The
    trace penalty is a sum of conditional variances k(x_i,x_i) - q(x_i,x_i):
    it depends on where the data *are* and not at all on what y does there.
    Only the DTC term ever sees y. If the y-blind term carries most of the
    force, "Z go where the function is hard" cannot be the main story, and the
    weak asymmetry in measurement 3 is explained rather than just noted.

    Central differences on the two public read-outs, so nothing here reaches
    into SGPR's internals; the two pieces are checked against the analytic
    gradient, which is what makes the split trustworthy.
    """
    _, _, grad_Z = model.elbo_and_grads()
    Z0 = model.Z.copy()
    g_dtc = np.zeros_like(Z0)
    g_pen = np.zeros_like(Z0)
    for idx in np.ndindex(Z0.shape):
        two = []
        for sign in (+1.0, -1.0):
            Z = Z0.copy()
            Z[idx] += sign * eps
            model.Z = Z
            model.fit(X, y)
            two.append((model.dtc_log_evidence(), model.trace_term()))
        g_dtc[idx] = (two[0][0] - two[1][0]) / (2 * eps)
        g_pen[idx] = -(two[0][1] - two[1][1]) / (2 * eps) / (2 * model.noise_var)
    model.Z = Z0
    model.fit(X, y)
    return grad_Z, g_dtc, g_pen


def main():
    X, y = dataset()
    Xs = np.linspace(LO, HI, 400).reshape(-1, 1)
    fast = X[:, 0] > 0

    print(f"target: sin(1.1x) + 0.6 sin(5x) on x>0, n={N}, noise sd={NOISE_SD}")
    print(f"data:   {int((~fast).sum())} points in the slow half, "
          f"{int(fast.sum())} in the fast half (uniform by construction)")

    t0 = time.perf_counter()
    exact = exact_fit(X, y)
    lml = exact.log_marginal_likelihood()
    print(f"\n0. exact GP, ML-II from s2={INIT['s2']}, l={INIT['l']}, "
          f"sigma^2={INIT_NOISE}   [{time.perf_counter() - t0:.1f}s]")
    print(f"    fitted  s2={np.exp(exact.params[0]):.4f}  "
          f"l={np.exp(exact.params[1]):.4f}  "
          f"sigma^2={np.exp(exact.params[-1]):.5f}  (true noise var "
          f"{NOISE_SD**2:.2f})")
    print(f"    log marginal likelihood {lml:.3f}   <- the ceiling below")

    rows, Zs = sweep(X, y, Xs, exact)
    free, frozen = rows[True], rows[False]
    control_Z: "dict[int, np.ndarray]" = {}

    print("\n1-2. sweep over M.  gap = exact LML - F, so it cannot go negative")
    print("     M |        Z optimized              |        Z frozen (quantiles)")
    print("       |   gap    mean err  sd err     l |   gap    mean err  sd err     l")
    for M, a, b in zip(MS, free, frozen):
        print(f"  {M:4d} | {a[0]:8.2f} {a[1]:8.4f} {a[2]:7.4f} {a[4]:6.3f} |"
              f" {b[0]:8.2f} {b[1]:8.4f} {b[2]:7.4f} {b[4]:6.3f}")

    for name, arr in (("optimized", free), ("frozen", frozen)):
        assert (arr[:, 0] > -1e-6).all(), f"{name} arm crossed the ceiling"
        rises = np.where(np.diff(arr[:, 0]) > 0)[0]
        print(f"    {name:9s}: gap monotone in M? "
              f"{'yes' if rises.size == 0 else f'NO, rises at M={[MS[i + 1] for i in rises]}'}"
              f"   gap at M={MS[-1]}: {arr[-1, 0]:.2e} nats")
    # The posterior errors are NOT guaranteed monotone the way the gap is, and
    # do not come out monotone here. Each M is a separate ML-II fit, so it is a
    # different model, not a refinement of the previous one -- only the *bound*
    # is a bound on one fixed quantity. Reported rather than smoothed.
    for name, arr, col in (("optimized", free, 1), ("optimized", free, 2),
                           ("frozen", frozen, 1), ("frozen", frozen, 2)):
        rises = np.where(np.diff(arr[:, col]) > 0)[0]
        label = "mean" if col == 1 else "sd  "
        print(f"    {name:9s} {label} error monotone? "
              f"{'yes' if rises.size == 0 else f'no, rises at M={[MS[i + 1] for i in rises]}'}")

    print("\n3. where the optimized inducing points went "
          "(data are 50/50 across x=0, the fast half is on the right)")
    for M in (4, 8, 16, 32, 64):
        Z = Zs[M]
        slow, fast_sp = spacing(Z, "slow"), spacing(Z, "fast")
        print(f"    M={M:3d}: {int((Z > 0).sum()):3d}/{M} in the fast half "
              f"({(Z > 0).mean():.0%})   median spacing  slow {slow:.3f}  "
              f"fast {fast_sp:.3f}   ({slow / fast_sp:.2f}x denser where it is hard)"
              f"   Z span [{Z.min():.2f}, {Z.max():.2f}]")
    print(f"    (data span [{X.min():.2f}, {X.max():.2f}] -- nothing pins Z "
          "inside it, and at small M\n     a surplus inducing point is pushed "
          "out past the data rather than kept)")

    print("\n   the same spacings in units of each fit's own lengthscale, which "
          "is the\n   transferable version -- what matters is how finely Z "
          "sample the KERNEL:")
    for M in MS[3:]:
        l = free[MS.index(M), 4]
        print(f"    M={M:3d}: slow {spacing(Zs[M], 'slow') / l:.3f} l   "
              f"fast {spacing(Zs[M], 'fast') / l:.3f} l"
              f"    gap {free[MS.index(M), 0]:8.2f} nats")

    print("\n   is the force on Z even looking at y? the two terms of dF/dZ, "
          "split by\n   finite differences. The trace penalty is a sum of "
          "conditional variances and\n   never sees y; only the DTC term does.")
    for tag, optimize_Z in (("at the quantile init (theta fitted, Z not yet moved)",
                             False),
                            ("at the converged joint optimum", True)):
        probe = SGPR(RBF(**INIT), Z=quantile_Z(X, 16), noise_var=INIT_NOISE)
        maximize_elbo(probe, X, y, optimize_Z=optimize_Z, lr=LR, steps=STEPS)
        total, g_dtc, g_pen = force_split(probe, X, y)
        n_pen, n_dtc = np.linalg.norm(g_pen), np.linalg.norm(g_dtc)
        print(f"    {tag}")
        print(f"      trace penalty (y-blind) {n_pen:8.2f} | DTC data fit "
              f"(sees y) {n_dtc:8.2f} | net {np.linalg.norm(total):8.2f}"
              f"   ratio {n_pen / n_dtc:5.2f}")
        print(f"      the two sum to the analytic dF/dZ to "
              f"{np.abs(g_dtc + g_pen - total).max():.1e}  (a check on the split)")
    print("    So the y-aware term carries most of the force, and the weak "
          "asymmetry in 3\n    is NOT the inducing points being blind to the "
          "data. The hypothesis this\n    measurement was written to confirm "
          "is the one it refuted.")

    print("\n4. control: the same target with the data drawn 3:1 toward the "
          "SLOW half.")
    print("   If Z chase difficulty they should stay right; if they chase data "
          "density\n   they should follow the data left. Measured:")
    Xc, yc = dataset(seed=0, slow_share=0.75)
    exact_c = exact_fit(Xc, yc)
    for M in (16, 32, 64):
        model = SGPR(RBF(**INIT), Z=quantile_Z(Xc, M), noise_var=INIT_NOISE)
        res = maximize_elbo(model, Xc, yc, lr=LR, steps=STEPS)
        Z = np.sort(res.Z.ravel())
        control_Z[M] = Z
        slow, fast_sp = spacing(Z, "slow"), spacing(Z, "fast")
        print(f"    M={M:3d}: {int((Z > 0).sum()):3d}/{M} in the fast half "
              f"({(Z > 0).mean():.0%})   median spacing  slow {slow:.3f}  "
              f"fast {fast_sp:.3f}   ({fast_sp / slow:.2f}x denser where the "
              f"DATA are)")
    print(f"    (exact GP on the skewed draw: l={np.exp(exact_c.params[1]):.4f}, "
          f"the same lengthscale -- the function did not change)")

    print("\n5. what optimizing Z is worth, in inducing points")
    for M, need in zip(MS, matching_M(free, frozen)):
        if need is None:
            print(f"    optimized M={M:3d}: no frozen M<={MS[-1]} in this sweep "
                  f"reaches its gap of {free[MS.index(M), 0]:.2f}")
        else:
            print(f"    optimized M={M:3d}: matched by frozen M={need:3d}"
                  f"  ({need / M:.2g}x)")

    print(f"\n    wall clock per fit: optimized {free[:, 6].mean():.2f}s mean, "
          f"frozen {frozen[:, 6].mean():.2f}s mean ({STEPS} Adam steps each)")

    # ---- figure -------------------------------------------------------------
    mean_exact, var_exact = exact.predict(Xs)
    sd_exact = np.sqrt(var_exact)
    Ms = np.array(MS)

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(Ms, free[:, 0], "o-", color="C0", lw=1.6, ms=4, label="$Z$ optimized")
    ax.plot(Ms, frozen[:, 0], "s--", color="C1", lw=1.6, ms=4,
            label="$Z$ frozen at data quantiles")
    ax.axhline(0.0, color="k", lw=1.0, ls=":")
    ax.annotate("exact $\\log p(y)$ -- a ceiling, not a target",
                xy=(Ms[1], 0.0), xytext=(0, 6), textcoords="offset points",
                fontsize=7.5, color="0.3")
    ax.set_xscale("log", base=2)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(-0.02, 4e3)                  # the gap is non-negative by theorem
    ax.set_xticks(Ms)
    ax.set_xticklabels([str(m) for m in Ms])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("inducing points $M$")
    ax.set_ylabel("$\\log p(y) - F$   (nats, symlog)")
    ax.set_title("1. The bound closes on the evidence from below", loc="left")
    ax.legend(fontsize=7.5)

    ax = axes[0, 1]
    for arr, color, ls, tag in ((free, "C0", "-", "optimized"),
                                (frozen, "C1", "--", "frozen")):
        ax.plot(Ms, arr[:, 1], "o" + ls, color=color, lw=1.5, ms=3.5,
                label=f"mean, {tag}")
        ax.plot(Ms, arr[:, 2], "s" + ls, color=color, lw=1.2, ms=3.0,
                alpha=0.55, label=f"sd, {tag}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(Ms)
    ax.set_xticklabels([str(m) for m in Ms])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("inducing points $M$")
    ax.set_ylabel("max deviation from the exact posterior")
    ax.set_title("2. And so does the posterior it implies", loc="left")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1, 0]
    ax.axvspan(0.0, HI, color="C3", alpha=0.07, lw=0)
    rows_ = [(M, Zs[M], "C0", "uniform data") for M in (16, 32, 64)]
    rows_ += [(M, control_Z[M], "C3", "3:1 data toward the slow half")
              for M in (16, 32, 64)]
    seen = set()
    for i, (M, Z, color, label) in enumerate(rows_):
        ax.plot(Z, np.full(len(Z), i), "|", color=color, ms=9, mew=1.3,
                label=None if label in seen else label)
        seen.add(label)
    ax.set_yticks(range(len(rows_)))
    ax.set_yticklabels([f"$M$={M}" for M, *_ in rows_], fontsize=7.5)
    ax.set_ylim(-1.1, len(rows_) - 0.3)
    ax.set_xlim(LO, HI)
    ax.axhline(2.5, color="0.8", lw=0.8)
    ax.set_xlabel("x   (shaded: the fast half of the function)")
    ax.set_title("3. A weak pull toward the hard half, reversed by density",
                 loc="left")
    ax.legend(fontsize=7, loc="lower center", ncol=2, borderaxespad=0.1)
    for i, (M, Z, _, _) in enumerate(rows_):
        ax.annotate(f"{int((Z > 0).sum())}/{M} right",
                    xy=(HI, i), xytext=(-2, 3), textcoords="offset points",
                    ha="right", fontsize=6.5, color="0.35")

    # Observation-level bands here, not latent ones: what M = 8 does wrong is
    # not that its error bar on f is too small -- it is that it buys the misfit
    # by inflating sigma^2, which only an observation band shows. That is the
    # noise-gradient behaviour gp/sparse.py derives, seen from the outside.
    ax = axes[1, 1]
    M_show = 8
    model = SGPR(RBF(**INIT), Z=quantile_Z(X, M_show), noise_var=INIT_NOISE)
    res = maximize_elbo(model, X, y, lr=LR, steps=STEPS)
    mean, var = model.predict(Xs, include_noise=True)
    sd = np.sqrt(var)
    _, var_exact_obs = exact.predict(Xs, include_noise=True)
    sd_exact_obs = np.sqrt(var_exact_obs)
    xs = Xs.ravel()
    ax.plot(X[:, 0], y, ".", color="0.8", ms=1.2, zorder=0)
    ax.fill_between(xs, mean_exact - 1.96 * sd_exact_obs,
                    mean_exact + 1.96 * sd_exact_obs, color="k", alpha=0.12, lw=0)
    ax.plot(xs, mean_exact, "k--", lw=1.3,
            label=f"exact GP, $\\sigma^2$={np.exp(exact.params[-1]):.3f}", zorder=3)
    ax.fill_between(xs, mean - 1.96 * sd, mean + 1.96 * sd,
                    color="C0", alpha=0.18, lw=0)
    ax.plot(xs, mean, color="C0", lw=1.4,
            label=f"SGPR $M={M_show}$, $\\sigma^2$={np.exp(res.params[-1]):.3f}",
            zorder=3)
    ax.set_xlim(LO - 0.9, HI + 0.9)
    ax.plot(res.Z.ravel(), np.full(M_show, ax.get_ylim()[0] + 0.12), "^",
            color="C0", ms=5, label="learned $Z$", zorder=4)
    ax.set_xlabel("x   (axis widened: one inducing point left the data)")
    ax.set_ylabel("y")
    ax.set_title(f"4. Too few $Z$: the misfit is bought as noise", loc="left")
    ax.legend(fontsize=7, loc="upper left", ncol=1)

    savefig(fig, "sparse.png")
    return free, frozen, Zs


if __name__ == "__main__":
    main()
