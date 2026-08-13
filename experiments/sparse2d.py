"""Does the 1D sparse-GP result survive a change of dimension?

The Limitations section names this as an open question, in the specific form
that makes it answerable:

    Every sparse measurement here is one-dimensional. [...] In d dimensions Z
    carries Md free parameters, quantiles stop being defined, and the
    density-over-difficulty result measured in Sec. 11 is exactly the kind of
    finding that need not survive a change of dimension. Nothing here tests
    whether it does.

This tests whether it does, at d = 2, where the exact GP still runs and so every
approximate quantity is still scored against the real posterior rather than
against another approximation.

The target is the Sec. 11 target with one extra slow axis:

    f(x1, x2) = sin(1.1 x1) + sin(1.1 x2) + 0.6 sin(5 x1) [x1 > 0],

so the difficulty asymmetry is the same object it was in 1D -- a five-times
faster component on half the domain -- and the second axis adds dimension
without adding structure. That is deliberate. If the 1D result fails here, the
change of dimension is the only candidate explanation.

Four measurements, and the first one was not planned.

0. **The ceiling has to be found before anything can be scored against it.**
   Every gap below is measured against the exact GP's log marginal likelihood,
   so ML-II has to reach the global optimum or the whole experiment is scored
   against the wrong number. Lifting Sec. 11's setup to 2D and carrying its
   initialization over unchanged does not: from ``l = 2.0`` ML-II stops at a
   local optimum **1222 nats** below the one every other start finds, fitting
   ``sigma^2 = 0.095`` against a true noise variance of 0.010 -- it gives up on
   the fast component and calls it noise. Sec. 9 documents ML-II multimodality
   on a 1D problem already; this is the same failure met by accident, in the
   ordinary act of reusing a working script one dimension up, which is how it
   will be met in practice. So the exact fit is a multistart and the table is
   reported.

1. **What replaces the quantiles.** Sec. 11's frozen arm puts Z at the data
   quantiles, which is a rule that exists only in 1D. Three candidates are run
   against each other and against a free (optimized) arm: a random subset of the
   data, k-means centroids (Lloyd from scratch, k-means++ seeding -- the
   standard practical default), and a uniform grid over the data's bounding box.
   The gap to the exact log evidence decides, and the answer determines what the
   rest of the week should initialize Z with.

2. **The density-over-difficulty result, re-run.** In 1D, optimized Z put 56% of
   themselves in the fast half at M = 64 -- a weak pull toward difficulty -- and
   a 3:1 data-density skew toward the *slow* half reversed the sign entirely
   (41%), so density outweighed the whole difficulty effect. Both arms are run
   again at d = 2, with the identical skew, and the question is whether the
   reversal reproduces.

   Only some rows can answer it. The *sparse* model makes the same choice of
   optimum that measurement 0 found for the exact one, and at small M it has no
   capacity to represent the short lengthscale, so it takes the smooth one: at
   M <= 64 every arm's gap sits at ~1222 nats, which is the basin separation
   itself and not a statement about inducing points. Those cells are marked
   ``[smooth]`` and are not read as evidence about difficulty, because their
   model does not think the target has any.

3. **Whether the transferable number transfers.** Sec. 11 offered "Z spacing
   about 0.7 fitted lengthscales" as the form of the result that ought to carry
   past its own setting. A number offered as transferable is a claim, and this
   is the first setting available to check it in: the same statistic is measured
   in 2D as the median nearest-neighbour distance among Z, in units of the
   fitted lengthscale.

What comes out: optimizing Z is not a refinement here, it is what lets the model
leave the wrong basin at all (M = 128, gap 615 against 1222 for every frozen
placement); k-means is the best of the three frozen rules; 1D's "spacing ~ 0.7
fitted lengthscales" transfers, reading 0.74 at M = 256; and the budget does
not, since 1D reached a 1.4-nat gap at M = 24 where 2D needs M = 256 to reach 54.

The density question does **not** get a yes or a no. Its raw numbers look like a
reproduction (0.500 -> 0.398 at M = 256, a -0.102 shift against 1D's -0.150) but
the skewed arm never escapes the smooth basin at any M run here, so the two arms
are fitting different functions and the shift cannot be read the way 1D's was.
What the control does show is a stronger statement than it was built for: at
d = 2, thinning the hard half 3:1 stops the sparse model from resolving that
half at all, rather than redistributing Z within a model that still sees it.

Run:  python experiments/sparse2d.py     (~8 min; the M = 256 cells dominate)
"""

import time

import numpy as np

from common import plt, savefig
from gp.gp import GPRegressor
from gp.kernels import RBF
from gp.optimize import adam_maximize, maximize_elbo
from gp.sparse import SGPR

N = 1600
NOISE_SD = 0.1
LO, HI = -4.0, 4.0
MS = (4, 8, 16, 32, 64, 128, 256)
STEPS = 400
LR = 0.05
SEED = 0

# Deliberately not the truth (which is s2 ~ 0.75, l ~ 0.53, sigma^2 ~ 0.010),
# but *not* Sec. 11's l = 2.0 either -- see MULTISTART below and measurement 0.
INIT = dict(s2=0.5, l=0.5)
INIT_NOISE = 0.3

# Starting points for measurement 0. The first is Sec. 11's own initialization,
# carried over unchanged; the rest are ordinary alternatives.
MULTISTART = (
    dict(s2=0.5, l=2.0, noise=0.3),      # the 1D script's init, verbatim
    dict(s2=0.5, l=0.5, noise=0.3),
    dict(s2=1.0, l=0.3, noise=0.05),
    dict(s2=1.0, l=1.0, noise=0.01),
)


def target(X):
    """The Sec. 11 target lifted to 2D: one hard half-plane, one plain axis.

    ``sin(1.1 x1) + sin(1.1 x2) + 0.6 sin(5 x1) [x1 > 0]``. The extra term in
    x2 keeps the second axis from being ignorable -- an ARD-free isotropic RBF
    would otherwise have nothing to spend resolution on there -- while carrying
    no difficulty asymmetry of its own, so x1 > 0 remains the only hard region.
    """
    x1, x2 = X[:, 0], X[:, 1]
    return np.sin(1.1 * x1) + np.sin(1.1 * x2) + 0.6 * np.sin(5.0 * x1) * (x1 > 0)


def dataset(seed=SEED, slow_share=0.5, n=N):
    """n samples on the square; ``slow_share`` of them from x1 < 0.

    ``slow_share=0.5`` is uniform over the square and is the main setting; 0.75
    is measurement 2's 3:1 density skew toward the slow half, the same skew
    Sec. 11 used, applied to x1 only so the control changes density and nothing
    else.
    """
    rng = np.random.default_rng(seed)
    n_slow = int(round(n * slow_share))
    x1 = np.concatenate([rng.uniform(LO, 0.0, n_slow),
                         rng.uniform(0.0, HI, n - n_slow)])
    x2 = rng.uniform(LO, HI, n)
    X = np.stack([x1, x2], axis=1)
    return X, target(X) + NOISE_SD * rng.standard_normal(n)


# ---------------------------------------------------------------------------
# Inducing-point initializers: the 2D replacements for "the data quantiles"
# ---------------------------------------------------------------------------
def subset_Z(X, M, seed=0):
    """M distinct data points, drawn without replacement. The cheapest rule."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(M, X.shape[0]), replace=False)
    return X[idx].copy()


def kmeans_Z(X, M, seed=0, iters=100, tol=1e-8, n_init=10):
    """Lloyd's algorithm with k-means++ seeding, best of ``n_init`` restarts.

    The standard practical default for inducing-point initialization, and the
    natural generalization of "spread them through the data" to d dimensions.
    Written here rather than imported because scikit-learn is a *test* oracle in
    this repo, not a dependency of the library or the experiments -- and having
    an independent implementation is exactly what lets the tests check this one
    against it.

    ``n_init`` is not a detail. k-means is a local method, and a single start
    was measurably worse than the oracle here -- 31% higher inertia at M = 4 on
    this data -- which would have handicapped the frozen-k-means arm of the
    experiment against the others for reasons that have nothing to do with
    inducing points. Restarting and keeping the lowest inertia is what
    scikit-learn does by default and is what the test compares against.
    """
    best, best_inertia = None, np.inf
    for i in range(n_init):
        centres, inertia = _kmeans_once(X, M, seed=(seed, i), iters=iters, tol=tol)
        if inertia < best_inertia:
            best, best_inertia = centres, inertia
    return best


def _kmeans_once(X, M, seed, iters, tol):
    """One k-means++ seeding followed by Lloyd iterations. Returns
    ``(centres, inertia)``.

    k-means++ seeding: the first centre is a uniform draw, and each subsequent
    centre is drawn with probability proportional to its squared distance to the
    nearest centre chosen so far. That is what keeps the initialization from
    putting two centres in the same clump, which plain uniform seeding does
    often enough to matter.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    M = min(M, n)

    centres = np.empty((M, X.shape[1]))
    centres[0] = X[rng.integers(n)]
    d2 = np.sum((X - centres[0]) ** 2, axis=1)
    for j in range(1, M):
        total = d2.sum()
        if total <= 0:                      # all points already coincide
            centres[j] = X[rng.integers(n)]
        else:
            centres[j] = X[rng.choice(n, p=d2 / total)]
        d2 = np.minimum(d2, np.sum((X - centres[j]) ** 2, axis=1))

    # ``prev = inf`` on the first pass would make the relative tolerance below
    # ``tol * inf = inf`` and the test ``inf <= inf`` true, breaking out after a
    # single Lloyd update -- which is what the first version of this did, and it
    # cost 16% of inertia at M = 16 against the oracle. ``None`` means "no
    # previous value to compare against yet".
    prev = None
    for _ in range(iters):
        # (n, M) squared distances, then the nearest centre for each point
        dist = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        label = dist.argmin(axis=1)
        inertia = float(dist[np.arange(n), label].sum())
        for j in range(M):
            take = label == j
            if take.any():
                centres[j] = X[take].mean(axis=0)
            # An empty cluster keeps its previous centre rather than being
            # re-seeded; with k-means++ starts it is rare, and moving it would
            # make the result depend on the re-seeding rule instead of the data.
        if prev is not None and prev - inertia <= tol * max(1.0, prev):
            break
        prev = inertia

    # ``inertia`` inside the loop belongs to the centres *before* that pass's
    # update, so score the centres actually being returned.
    final = ((X[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2).min(axis=1).sum()
    return centres, float(final)


def grid_Z(X, M):
    """A near-square uniform grid over the data's bounding box.

    The purely geometric baseline: it ignores where the data are entirely, which
    is the property that makes it a useful control for measurement 2. M is
    rounded down to the largest ``a * b <= M`` with a near-square aspect, and the
    actual count is reported rather than silently differing from the requested
    one.
    """
    a = int(np.floor(np.sqrt(M)))
    b = int(M // a)
    lo, hi = X.min(axis=0), X.max(axis=0)
    g1 = np.linspace(lo[0], hi[0], a)
    g2 = np.linspace(lo[1], hi[1], b)
    A, B = np.meshgrid(g1, g2, indexing="ij")
    return np.stack([A.ravel(), B.ravel()], axis=1)


INITIALIZERS = {
    "subset": lambda X, M: subset_Z(X, M, seed=SEED),
    "kmeans": lambda X, M: kmeans_Z(X, M, seed=SEED),
    "grid": grid_Z,
}


# ---------------------------------------------------------------------------
# Read-outs
# ---------------------------------------------------------------------------
def exact_fit(X, y, init=None):
    """ML-II on the exact GP from one starting point."""
    init = INIT if init is None else init
    model = GPRegressor(RBF(s2=init["s2"], l=init["l"]),
                        noise_var=init.get("noise", INIT_NOISE))
    best, _ = adam_maximize(lambda p: model.lml_and_grad(X, y, p), model.params,
                            lr=LR, steps=STEPS)
    model.params = best
    return model.fit(X, y)


def exact_multistart(X, y, inits=MULTISTART):
    """ML-II from every starting point; return ``(best_model, rows)``.

    This is not defensive coding, it is measurement 0. The exact GP's log
    marginal likelihood is the ceiling every bound below is scored against, so
    if ML-II stops at a local optimum the whole experiment is measured against
    the wrong number -- and on this target it does, from Sec. 11's own
    initialization. Sec. 9 of the README already documents ML-II
    multimodality on a 1D problem; this is the same failure met by accident,
    while lifting a 1D script to 2D, which is how it will be met in practice.
    """
    rows, best_model, best_lml = [], None, -np.inf
    for init in inits:
        model = exact_fit(X, y, init)
        lml = model.log_marginal_likelihood()
        rows.append(dict(init=init, s2=float(np.exp(model.params[0])),
                         l=float(np.exp(model.params[1])),
                         noise=float(np.exp(model.params[-1])), lml=float(lml)))
        if lml > best_lml:
            best_model, best_lml = model, lml
    return best_model, rows


def fast_share(Z):
    """Fraction of inducing points in the hard half-plane x1 > 0.

    The 2D version of Sec. 11's count. A purely geometric method gives 0.5 by
    construction when the data are uniform, so that is the null this is read
    against.
    """
    return float((Z[:, 0] > 0).mean())


def nn_spacing(Z):
    """Median nearest-neighbour distance among the inducing points.

    Sec. 11 measured the median gap between consecutive Z, which is what
    "nearest neighbour" means on a line. In 2D consecutive is undefined and
    nearest neighbour is not, so this is the statistic that reduces to Sec. 11's
    when d = 1 -- the point being to compare like with like when the "0.7
    lengthscales" number is checked.
    """
    if Z.shape[0] < 2:
        return float("nan")
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d2, np.inf)
    return float(np.median(np.sqrt(d2.min(axis=1))))


def half_spacing(Z, side):
    """Nearest-neighbour spacing among the Z on one side of x1 = 0.

    Counting Z per half conflates "more of them" with "packed tighter"; Sec. 11
    reported both for that reason and so does this.
    """
    part = Z[Z[:, 0] > 0] if side == "fast" else Z[Z[:, 0] <= 0]
    return nn_spacing(part)


def density_shift(uniform_rows, control_rows, index=-1):
    """Change in the fast-half share caused by the density skew, ``control -
    uniform``.

    Signed so it is directly comparable with Sec. 11's 1D number: there the
    share went 0.560 -> 0.410 under the same 3:1 skew, i.e. **-0.150**. A
    negative value means the skew pulled inducing points toward the slow half,
    which is the direction that says density outweighs difficulty. Written as a
    function with a test rather than inline in a print, because the first
    version of that print had the subtraction the other way round and reported
    a sign-flipped number next to the 1D one it was being compared with.
    """
    return float(control_rows[index]["fast_share"] - uniform_rows[index]["fast_share"])


def in_smooth_basin(row, exact_lengthscale, factor=2.0):
    """Did this cell's own ML-II land in the long-lengthscale optimum?

    Measurement 0 shows the exact GP has two optima 1222 nats apart, one of
    which explains the fast component and one of which calls it noise. The
    *sparse* model has the same choice to make, and at small M it does not have
    the capacity to represent the short lengthscale, so it takes the smooth one.
    A cell that did is not a coarser answer to the same question -- it is
    fitting a different function -- and its inducing points cannot be read as
    evidence about where difficulty pulls them, because its model does not think
    there is any difficulty. Flagged so those rows are not compared with the
    ones that escaped.
    """
    return row["lengthscale"] > factor * exact_lengthscale


def fit_arm(X, y, Xs, exact_pred, M, init, optimize_Z):
    """One (initializer, free/frozen) cell: ML-II, then the read-outs."""
    Z0 = INITIALIZERS[init](X, M)
    model = SGPR(RBF(**INIT), Z=Z0, noise_var=INIT_NOISE)
    res = maximize_elbo(model, X, y, optimize_Z=optimize_Z, lr=LR, steps=STEPS)
    mean, var = model.predict(Xs)
    mean_exact, var_exact = exact_pred
    lengthscale = float(np.exp(model.params[1]))
    return dict(
        M=int(res.Z.shape[0]),
        elbo=float(res.objective),
        mean_err=float(np.sqrt(np.mean((mean - mean_exact) ** 2))),
        sd_err=float(np.sqrt(np.mean((np.sqrt(var) - np.sqrt(var_exact)) ** 2))),
        lengthscale=lengthscale,
        trace=float(res.trace_term),
        fast_share=fast_share(res.Z),
        spacing=nn_spacing(res.Z),
        spacing_fast=half_spacing(res.Z, "fast"),
        spacing_slow=half_spacing(res.Z, "slow"),
        Z=res.Z,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def eval_grid(n=60):
    g = np.linspace(LO, HI, n)
    A, B = np.meshgrid(g, g, indexing="ij")
    return np.stack([A.ravel(), B.ravel()], axis=1)


def main():
    t_start = time.perf_counter()
    X, y = dataset()
    Xs = eval_grid()

    print(f"target: sin(1.1 x1) + sin(1.1 x2) + 0.6 sin(5 x1) on x1>0")
    print(f"data:   n={N} uniform on [{LO}, {HI}]^2, noise sd={NOISE_SD}, "
          f"{int((X[:, 0] > 0).sum())} points in the fast half")

    t0 = time.perf_counter()
    exact, starts = exact_multistart(X, y)
    lml = exact.log_marginal_likelihood()
    exact_pred = exact.predict(Xs)
    print(f"\n0. the ceiling, and whether ML-II finds it  "
          f"[{time.perf_counter() - t0:.1f}s]")
    print(f"    {'init (s2, l, noise)':>24} | {'s2':>7} {'l':>7} {'sigma^2':>9} "
          f"{'LML':>10}")
    for r in starts:
        init = r["init"]
        label = "({s2}, {l}, {noise})".format(**init)
        tag = "  <- Sec. 11's init" if init is MULTISTART[0] else ""
        print(f"    {label:>24} | {r['s2']:>7.4f} {r['l']:>7.4f} "
              f"{r['noise']:>9.5f} {r['lml']:>10.2f}{tag}")
    spread = max(r["lml"] for r in starts) - min(r["lml"] for r in starts)
    print(f"    spread across starts: {spread:.1f} nats "
          f"(true noise var {NOISE_SD ** 2:.3f})")
    print(f"    best log marginal likelihood {lml:.3f}   <- the ceiling below")

    # -- 1. which initializer replaces the quantiles -------------------------
    print("\n1. what replaces 'the data quantiles' in 2D.  "
          "gap = exact LML - F (>= 0 always)")
    print("        |            Z frozen at the init            |   Z optimized")
    print("      M |   subset     kmeans       grid             |   from kmeans")
    frozen_rows = {k: [] for k in INITIALIZERS}
    free_rows = []
    for M in MS:
        line = f"  {M:5d} |"
        for init in ("subset", "kmeans", "grid"):
            r = fit_arm(X, y, Xs, exact_pred, M, init, optimize_Z=False)
            frozen_rows[init].append(r)
            line += f" {lml - r['elbo']:9.2f}"
        r = fit_arm(X, y, Xs, exact_pred, M, "kmeans", optimize_Z=True)
        free_rows.append(r)
        line += f"             | {lml - r['elbo']:9.2f}"
        print(line + f"   (M={r['M']}, grid M={frozen_rows['grid'][-1]['M']})")

    for name, rows in list(frozen_rows.items()) + [("free", free_rows)]:
        gaps = np.array([lml - r["elbo"] for r in rows])
        assert (gaps > -1e-6).all(), f"{name} crossed the ceiling: {gaps.min()}"
        rises = np.where(np.diff(gaps) > 0)[0]
        print(f"    {name:7s}: gap monotone in M? "
              f"{'yes' if rises.size == 0 else 'NO at M=' + str([MS[i + 1] for i in rises])}"
              f"   gap at M={MS[-1]}: {gaps[-1]:.3g} nats")

    # -- 2. the density-over-difficulty control ------------------------------
    l_exact = float(np.exp(exact.params[1]))
    print("\n2. where the optimized Z go, and what moves them")
    print(f"   uniform data (50/50 across x1=0); a geometric method gives 0.500")
    print(f"   [smooth] marks a cell whose own ML-II is in the long-lengthscale "
          f"basin\n   (fitted l > 2 x the exact GP's {l_exact:.3f}); its model does "
          f"not resolve the\n   fast component at all, so its Z say nothing about "
          f"difficulty")
    for M, r in zip(MS, free_rows):
        flag = "  [smooth]" if in_smooth_basin(r, l_exact) else ""
        print(f"    M={M:4d}: fast share {r['fast_share']:.3f}   "
              f"nn spacing fast {r['spacing_fast']:.3f} slow {r['spacing_slow']:.3f}"
              f"   (slow/fast {r['spacing_slow'] / r['spacing_fast']:.2f}x)"
              f"   l={r['lengthscale']:.3f}{flag}")

    Xc, yc = dataset(slow_share=0.75)
    print(f"\n   density control: same target, {int((Xc[:, 0] <= 0).sum())} / "
          f"{int((Xc[:, 0] > 0).sum())} points slow / fast (3:1 toward slow)")
    exact_c, _ = exact_multistart(Xc, yc)
    pred_c = exact_c.predict(Xs)
    l_exact_c = float(np.exp(exact_c.params[1]))
    control_rows = []
    for M in MS:
        r = fit_arm(Xc, yc, Xs, pred_c, M, "kmeans", optimize_Z=True)
        control_rows.append(r)
        flag = "  [smooth]" if in_smooth_basin(r, l_exact_c) else ""
        print(f"    M={M:4d}: fast share {r['fast_share']:.3f}   "
              f"nn spacing fast {r['spacing_fast']:.3f} slow {r['spacing_slow']:.3f}"
              f"   (slow/fast {r['spacing_slow'] / r['spacing_fast']:.2f}x)"
              f"   l={r['lengthscale']:.3f}{flag}")

    uni = np.array([r["fast_share"] for r in free_rows])
    con = np.array([r["fast_share"] for r in control_rows])
    print(f"\n    uniform  fast share: {uni.min():.3f}-{uni.max():.3f} "
          f"(at M={MS[-1]}: {uni[-1]:.3f})")
    print(f"    3:1 slow fast share: {con.min():.3f}-{con.max():.3f} "
          f"(at M={MS[-1]}: {con[-1]:.3f})")
    print(f"    Sec. 11 in 1D:       0.560 uniform -> 0.410 under the same skew, "
          f"i.e. -0.150")
    print("\n    the skew's effect, at the M where both arms have escaped the "
          "smooth basin:")
    for i, M in enumerate(MS):
        if in_smooth_basin(free_rows[i], l_exact) or \
                in_smooth_basin(control_rows[i], l_exact_c):
            continue
        print(f"      M={M:4d}: {uni[i]:.3f} -> {con[i]:.3f}   "
              f"shift {density_shift(free_rows, control_rows, i):+.3f}")

    # -- 3. does "0.7 fitted lengthscales" transfer? -------------------------
    print("\n3. Z spacing in units of the fitted lengthscale "
          "(Sec. 11's transferable form: ~0.7)")
    print("      M | free Z: spacing      l    ratio | kmeans frozen: ratio")
    for M, r, f in zip(MS, free_rows, frozen_rows["kmeans"]):
        print(f"  {M:5d} | {r['spacing']:14.3f} {r['lengthscale']:6.3f} "
              f"{r['spacing'] / r['lengthscale']:8.2f} | "
              f"{f['spacing'] / f['lengthscale']:20.2f}")
    ratios = np.array([r["spacing"] / r["lengthscale"] for r in free_rows])
    print(f"    free-Z ratio spans {ratios.min():.2f}-{ratios.max():.2f}; "
          f"1D reported ~0.7 across its sweep")

    figure(X, y, free_rows, control_rows, frozen_rows, lml, l_exact, l_exact_c)
    print(f"\ntotal {time.perf_counter() - t_start:.0f}s")


def figure(X, y, free_rows, control_rows, frozen_rows, lml, l_exact, l_exact_c):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    ax = axes[0]
    for init, style in (("subset", "o--"), ("kmeans", "s--"), ("grid", "^--")):
        ax.loglog(MS, [lml - r["elbo"] for r in frozen_rows[init]], style,
                  label=f"frozen: {init}", alpha=0.8)
    ax.loglog(MS, [lml - r["elbo"] for r in free_rows], "o-", color="k",
              label="optimized $Z$")
    ax.set_xlabel("$M$ inducing points")
    ax.set_ylabel("exact LML $-$ $F$   (nats)")
    ax.set_title("1. what replaces the quantiles in 2D")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Panel 2: hollow markers for cells whose own ML-II is in the smooth basin.
    # Those points are not a coarser version of the filled ones -- their model
    # has not resolved the fast half -- so they must not read as one series.
    ax = axes[1]
    ax.axhline(0.5, color="0.6", lw=1, ls=":")
    for rows_, colour, marker, name, l_ref in (
            (free_rows, "C0", "o", "uniform data", l_exact),
            (control_rows, "C1", "s", "3:1 density toward slow", l_exact_c)):
        share = [r["fast_share"] for r in rows_]
        smooth = [in_smooth_basin(r, l_ref) for r in rows_]
        ax.semilogx(MS, share, "-", color=colour, label=name)
        ax.scatter([m for m, s in zip(MS, smooth) if not s],
                   [v for v, s in zip(share, smooth) if not s],
                   c=colour, marker=marker, s=30, zorder=3)
        ax.scatter([m for m, s in zip(MS, smooth) if s],
                   [v for v, s in zip(share, smooth) if s],
                   facecolors="none", edgecolors=colour, marker=marker, s=30,
                   zorder=3)
    ax.set_xlabel("$M$")
    ax.set_ylabel("share of $Z$ in the fast half")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("2. difficulty vs data density, at $d=2$\n"
                 "(hollow: model still in the smooth basin)", fontsize=9)
    ax.legend(fontsize=8)

    # Panel 3 shows the largest M, which is the only uniform-arm cell that has
    # both escaped the basin and reached a bound worth reading. An earlier
    # version showed M = 64, whose model has not resolved the fast half at all.
    ax = axes[2]
    M_show = MS[-1]
    i = len(MS) - 1
    ax.scatter(X[:, 0], X[:, 1], s=3, c="0.85", label="data")
    Zf = free_rows[i]["Z"]
    ax.scatter(Zf[:, 0], Zf[:, 1], s=14, c="C0", marker="o",
               label=f"optimized $Z$ ($M={M_show}$)")
    ax.axvline(0.0, color="0.4", lw=1)
    ax.set_xlabel("$x_1$   (right of the line: the fast half)")
    ax.set_ylabel("$x_2$")
    ax.set_title("3. where they land (uniform data)")
    ax.legend(loc="upper left", fontsize=7)

    fig.tight_layout()
    savefig(fig, "sparse2d.png")


if __name__ == "__main__":
    main()
