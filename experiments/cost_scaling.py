"""What the exact GP and SGPR actually cost, in seconds and in bytes.

Sections 9-11 argued about approximation *quality* -- how much posterior the
inducing points give back, and where random features throw it away. None of it
measured the thing the whole exercise is for. The exact GP is said to cost
O(n^3) time and O(n^2) memory, SGPR O(n M^2) and O(n M), and so far those are
docstrings, not measurements.

This script measures them, and reports three things the asymptotics alone do
not tell you:

1. **The empirical exponents**, fitted in log-log space (`gp.bench`), against
   the theoretical 3 and 2. A sweep that never reaches the asymptotic regime
   will report an exponent below theory, so the local exponent between each
   consecutive pair is printed alongside the global fit -- if the sequence is
   still climbing at the largest n, the fit is a lower bound, and saying so is
   the difference between measuring and confirming.

2. **The crossover**, which is not where most of the interest is but is where
   the honest reading starts: at fixed M the sparse method is cheaper at every
   n above a small threshold, and *below* it the exact GP wins because a
   sparse fit still has to touch all n points and then do M-dimensional
   algebra on top.

3. **The wall.** The exact GP does not get gradually slower and then stop; it
   stops because an n x n float64 matrix does not fit in RAM, and the peak is
   several copies of it, not one. The n where that happens on this machine is
   extrapolated from the measured law and reported as an extrapolation -- the
   sweep stops well below it on purpose, because the honest way to find that
   number is not to page a laptop to death.

Cheaper is meaningless without accuracy, so every n where the exact GP ran also
reports what the M = 64 approximation gave up there: the gap in log evidence,
and the worst error in posterior mean and sd across a grid. On this easy 1D
target the answer is "essentially nothing", which is a fact about the target --
Sec. 11 has the setting where the same M is not nearly enough.

Method note: peak RSS is a high-water mark that cannot be reset within a
process, so each cell (one method, one n, one M) runs in its own subprocess and
reports one line of JSON. That also keeps allocator state, BLAS thread pools,
and page-cache warmth from leaking between cells.

Two memory numbers are reported per cell and they measure different things:
``numpy`` is the peak of what NumPy requested (reproducible run to run,
exactly 8 bytes a word; the number of temporaries a call allocates does depend
on the NumPy build) and ``rss`` is the peak the OS actually backed (not
reproducible even run to run -- see the wall section -- includes BLAS
workspace, excludes untouched pages). The tracing pass
is a second run of the same work, so it is skipped once a single run
costs more than ``TRACE_MAX_SECONDS``; RSS comes free from the timed runs and
is always reported.

Run:  python experiments/cost_scaling.py            (~2 min)
      python experiments/cost_scaling.py --max-n 8000     (~25 s, skips the top)
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np
from common import plt, savefig

from gp.bench import best_of, local_exponents, max_rss, power_law_fit, traced_peak
from gp.gp import GPRegressor
from gp.kernels import RBF
from gp.sparse import SGPR

S2, LENGTHSCALE, NOISE = 1.0, 0.7, 0.01
SEED = 0
GRID = np.linspace(-4.0, 4.0, 200).reshape(-1, 1)

# The sweep. Exact stops at 16000 (about a minute and 4-5 GB here); sparse
# keeps going to where an exact fit would need a machine nobody has.
N_EXACT = (64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000)
N_SPARSE = N_EXACT + (32000, 64000, 128000)
M_DEFAULT = 64
M_SWEEP = (8, 16, 32, 64, 128, 256, 512, 1024)
N_FOR_M_SWEEP = 16000

TRACE_MAX_SECONDS = 10.0    # above this, skip the second (traced) run
ASYMPTOTIC_FROM = 2000      # fit the exponent on the tail, not the overhead
RAM_FRACTION = 0.6          # refuse a cell predicted to want more than this


def dataset(n):
    """n points of a smooth 1D target with light noise, deterministic in n.

    Same seed for every n on purpose: the sweep is about cost, and cost should
    not be averaged over datasets that differ in difficulty. It is a GP-friendly
    target so that the accuracy column measures the approximation and not the
    model's inability to fit the data.
    """
    rng = np.random.default_rng(SEED)
    X = np.sort(rng.uniform(-4.0, 4.0, size=(n, 1)), axis=0)
    y = np.sin(2.0 * X).ravel() + np.sqrt(NOISE) * rng.standard_normal(n)
    return X, y


def total_ram_bytes():
    """Physical RAM, or None where the OS will not say."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (ValueError, AttributeError, OSError):
        return None


# -- one cell, in its own process --------------------------------------------


def _work(method, n, M):
    """fit + objective + predict, the whole user-visible operation."""
    X, y = dataset(n)
    kernel = RBF(s2=S2, l=LENGTHSCALE)
    if method == "exact":
        model = GPRegressor(kernel, noise_var=NOISE).fit(X, y)
        obj = model.log_marginal_likelihood()
    else:
        Z = np.quantile(X, np.linspace(0.0, 1.0, M)).reshape(-1, 1)
        model = SGPR(kernel, Z=Z, noise_var=NOISE).fit(X, y)
        obj = model.objective()
    mean, var = model.predict(GRID)
    return obj, mean, np.sqrt(var)


def _components(method, n, M):
    """Time the two pieces the cost model is made of, separately.

    The point is to have an answer ready when the measured exponent misses the
    theoretical one, instead of blaming "constants". Both methods are a sum of
    two very different kinds of work:

      exact:  build K = k(X, X), which is O(n^2) transcendental evaluations at
              memory bandwidth, then factorize it, which is O(n^3) BLAS flops.
      SGPR:   build Kuf = k(Z, X), O(n M) the same way, then the Gram product
              A D^-1 A^T, which is O(n M^2) BLAS flops.

    A sweep whose largest size has not yet let the flops overtake the kernel
    evaluation will report an exponent below theory, and this is what says so.
    """
    X, y = dataset(n)
    kernel = RBF(s2=S2, l=LENGTHSCALE)
    if method == "exact":
        _, t_build = best_of(lambda: kernel(X, X))
        K = kernel(X, X)
        K.flat[:: n + 1] += NOISE
        _, t_solve = best_of(lambda: np.linalg.cholesky(K))
        return {"build": t_build.seconds, "solve": t_solve.seconds}
    Z = np.quantile(X, np.linspace(0.0, 1.0, M)).reshape(-1, 1)
    _, t_build = best_of(lambda: kernel(Z, X))
    A = np.linalg.solve(np.linalg.cholesky(kernel(Z, Z) + 1e-8 * np.eye(M)),
                        kernel(Z, X))
    _, t_solve = best_of(lambda: (A / NOISE) @ A.T)
    return {"build": t_build.seconds, "solve": t_solve.seconds}


def run_cell(method, n, M):
    """Time it, read its peak RSS, then (if cheap) trace it and break it down.

    Order matters. RSS is read straight after the timed runs and before
    anything else, because it is a high-water mark: the tracing pass allocates
    tracemalloc's own bookkeeping on top of the same arrays, and letting that
    into the number would inflate the memory curve by a machine-dependent
    amount. Tracing is a separate pass for the mirror-image reason -- its
    per-allocation Python hook would inflate the seconds.

    The traced and component passes are skipped once a single run costs more
    than ``TRACE_MAX_SECONDS``, so the top of the sweep costs one run, not
    four. The columns then read ``--``, rather than being quietly dropped.
    """
    (obj, mean, sd), timing = best_of(lambda: _work(method, n, M))
    rss = max_rss() - _BASELINE_RSS
    traced, parts = None, None
    if timing.seconds <= TRACE_MAX_SECONDS:
        _, traced = traced_peak(lambda: _work(method, n, M))
        parts = _components(method, n, M)
    return {
        "method": method, "n": n, "M": M,
        "seconds": timing.seconds, "reps": timing.reps,
        "rss_bytes": rss,
        "numpy_bytes": traced,
        "parts": parts,
        "objective": obj,
        "mean": mean.tolist(), "sd": sd.tolist(),
    }


def cell_main(argv):
    spec = json.loads(argv[0])
    print(json.dumps(run_cell(spec["method"], spec["n"], spec["M"])))


# -- the sweep ----------------------------------------------------------------


def run_sweep(cells, ram):
    """Run each cell in a fresh interpreter; skip what will not fit.

    The skip rule extrapolates from the last exact cell that ran, rather than
    from a formula: the measured RSS constant is not the 8 n^2 of the theory
    (see the write-up), so a formula-based guard would be wrong in whichever
    direction the constant happens to sit.
    """
    results, last_exact = [], None
    for spec in cells:
        if spec["method"] == "exact" and last_exact is not None and ram:
            predicted = last_exact["rss_bytes"] * (spec["n"] / last_exact["n"]) ** 2
            if predicted > RAM_FRACTION * ram:
                print(f"  skip exact n={spec['n']}: predicted peak "
                      f"{predicted / 1e9:.1f} GB > {RAM_FRACTION:.0%} of "
                      f"{ram / 1e9:.1f} GB RAM")
                continue
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--cell", json.dumps(spec)],
            capture_output=True, text=True,
        )
        if out.returncode != 0:
            print(f"  FAILED {spec}: {out.stderr.strip().splitlines()[-1:]}")
            continue
        row = json.loads(out.stdout)
        results.append(row)
        if spec["method"] == "exact":
            last_exact = row
        label = spec["method"] + (f" M={spec['M']}" if spec["method"] != "exact" else "")
        print(f"  {label:<12s} n={spec['n']:>7d}  {row['seconds']:>9.4f}s "
              f"({row['reps']} rep)  rss {row['rss_bytes'] / 1e6:>8.1f} MB")
    return results


def by(results, method, M=None, n=None):
    rows = [r for r in results if r["method"] == method
            and (M is None or r["M"] == M) and (n is None or r["n"] == n)]
    return sorted(rows, key=lambda r: (r["n"], r["M"]))


def fitted(rows, key, xkey="n", floor=0):
    """Fit an exponent on the tail (x >= floor), falling back to everything.

    The floor is there to keep the overhead-dominated small sizes out of an
    asymptotic claim. A truncated sweep (``--max-n``) can leave fewer than two
    points above it, and then a fit over the whole range is the only thing
    available -- every caller prints the x-range it actually used, so the
    weaker fit reports itself.
    """
    usable = [r for r in rows if r[key]]
    tail = [r for r in usable if r[xkey] >= floor]
    rows = tail if len(tail) >= 2 else usable
    xs = np.array([r[xkey] for r in rows], dtype=float)
    ys = np.array([r[key] for r in rows], dtype=float)
    return power_law_fit(xs, ys), xs, ys


def crossover(exact_rows, sparse_rows):
    """The n where the two timing curves meet, by linear interpolation in
    log-log space between the bracketing measured points.

    Interpolated between measurements rather than read off the two fitted laws:
    near the crossover both curves are in their overhead-dominated regime,
    where neither fit is any good.
    """
    ns = sorted(set(r["n"] for r in exact_rows) & set(r["n"] for r in sparse_rows))
    t_e = {r["n"]: r["seconds"] for r in exact_rows}
    t_s = {r["n"]: r["seconds"] for r in sparse_rows}
    diffs = [(n, np.log(t_e[n]) - np.log(t_s[n])) for n in ns]
    for (n0, d0), (n1, d1) in zip(diffs, diffs[1:]):
        if d0 <= 0.0 < d1:          # exact goes from cheaper to dearer
            w = -d0 / (d1 - d0)
            return float(np.exp(np.log(n0) + w * (np.log(n1) - np.log(n0))))
    return None


# -- reporting ----------------------------------------------------------------


def report(results, ram):
    exact = by(results, "exact")
    sparse = by(results, "sgpr", M=M_DEFAULT)
    msweep = [r for r in by(results, "sgpr", n=N_FOR_M_SWEEP) if r["M"] in M_SWEEP]

    print(f"\n{'':>7}{'exact':^34s}  {'SGPR M=64':^34s}")
    print(f"{'n':>7} {'sec':>9} {'rss MB':>10} {'numpy MB':>10} "
          f"{'sec':>9} {'rss MB':>10} {'numpy MB':>10} {'speedup':>9}")
    t_s = {r["n"]: r["seconds"] for r in sparse}
    for r in sparse:
        e = next((x for x in exact if x["n"] == r["n"]), None)
        cols = []
        for row in (e, r):
            if row is None:
                cols += [f"{'--':>9}", f"{'--':>10}", f"{'--':>10}"]
            else:
                npb = ("--" if row["numpy_bytes"] is None
                       else f"{row['numpy_bytes'] / 1e6:.1f}")
                cols += [f"{row['seconds']:>9.4f}", f"{row['rss_bytes'] / 1e6:>10.1f}",
                         f"{npb:>10}"]
        speed = f"{e['seconds'] / t_s[r['n']]:>8.1f}x" if e else f"{'--':>9}"
        print(f"{r['n']:>7} " + " ".join(cols) + " " + speed)

    print("\nempirical exponents (log-log fit; theory in brackets)")
    for name, rows, keys in (("exact", exact, (("seconds", 3), ("rss_bytes", 2),
                                               ("numpy_bytes", 2))),
                             ("SGPR M=64", sparse, (("seconds", 1), ("rss_bytes", 1),
                                                    ("numpy_bytes", 1)))):
        for key, theory in keys:
            law, xs, _ = fitted(rows, key, floor=ASYMPTOTIC_FROM)
            allrows = fitted(rows, key)
            local = local_exponents(allrows[1], allrows[2])
            print(f"  {name:<10s} {key:<12s} {law.exponent:5.2f} [{theory}]  "
                  f"r2 {law.r2:.4f}  n in [{int(xs.min())}, {int(xs.max())}]  "
                  f"local: " + " ".join(f"{v:.2f}" for v in local))

    if msweep:
        print(f"\nscaling in M at n = {N_FOR_M_SWEEP} (theory: time 2, memory 1)")
        for key, theory in (("seconds", 2), ("rss_bytes", 1), ("numpy_bytes", 1)):
            law, xs, _ = fitted(msweep, key, xkey="M")
            local = local_exponents([r["M"] for r in msweep],
                                    [r[key] for r in msweep])
            print(f"  {key:<12s} {law.exponent:5.2f} [{theory}]  r2 {law.r2:.4f}  "
                  f"local: " + " ".join(f"{v:.2f}" for v in local))

    print("\nwhere the time goes: kernel evaluation vs linear algebra")
    print("  (build = k(X,X) or k(Z,X); solve = Cholesky or the A D^-1 A^T Gram)")
    for label, rows, xkey in (("exact", exact, "n"),
                              (f"SGPR M={M_DEFAULT}", sparse, "n"),
                              (f"SGPR n={N_FOR_M_SWEEP}", msweep, "M")):
        rows = [r for r in rows if r["parts"]]
        if not rows:
            continue
        print(f"  {label}:")
        print(f"    {xkey:>7} {'build s':>9} {'solve s':>9} {'solve share':>12}")
        for r in rows:
            b, s = r["parts"]["build"], r["parts"]["solve"]
            print(f"    {r[xkey]:>7} {b:>9.4f} {s:>9.4f} {s / (b + s):>11.0%}")
        over = [r for r in rows if r["parts"]["solve"] > r["parts"]["build"]]
        verdict = (f"linear algebra overtakes kernel evaluation at "
                   f"{xkey} = {over[0][xkey]}" if over else
                   f"kernel evaluation still dominates at every {xkey} measured "
                   f"(up to {rows[-1][xkey]})")
        print(f"    -> {verdict}")

    n_cross = crossover(exact, sparse)
    if n_cross:
        print(f"\ncrossover: SGPR M={M_DEFAULT} becomes the cheaper fit at "
              f"n ~ {n_cross:.0f}")
        print(f"  below it the exact GP wins: a sparse fit still touches all n "
              f"points\n  and then does M x M algebra the exact one does not.")

    law_rss, xs, ys = fitted(exact, "rss_bytes", floor=ASYMPTOTIC_FROM)
    law_np, xs_np, _ = fitted(exact, "numpy_bytes", floor=ASYMPTOTIC_FROM)
    print("\nthe wall (exact GP) -- everything below the line was run, "
          "everything above it is extrapolated")
    print(f"  largest exact fit actually run: n = {int(xs.max())}, "
          f"peak {ys.max() / 1e9:.2f} GB resident")

    # Two laws, because they are not equally trustworthy and the difference is
    # the point. NumPy bytes are what the process ASKED for: deterministic, and
    # here exactly 3 copies of an n x n float64 Gram matrix = 24 n^2. RSS is
    # what the OS BACKED, which is a high-water mark over touched pages and is
    # NOT reproducible -- five runs of this script on one idle machine put the
    # n=16000 peak at 4.15, 4.46, 5.20, 5.46 and 5.89 GB (a 42% spread) and the
    # fitted exponent anywhere in [1.75, 1.92]. So the headline extrapolation is
    # the NumPy one, and the RSS one is reported next to it as a single run's
    # observation rather than as a law.
    print(f"  requested (NumPy, deterministic): {law_np.prefactor:.3g} * "
          f"n^{law_np.exponent:.2f} bytes  (r2 {law_np.r2:.4f}), "
          f"= 3 copies of 8n^2")
    print(f"  resident (RSS, this run only):    {law_rss.prefactor:.3g} * "
          f"n^{law_rss.exponent:.2f} bytes  (r2 {law_rss.r2:.4f})")
    print(f"  ------------------------------- extrapolated from here down")
    for n in (1e5, 1e6):
        print(f"  n = {n:>9,.0f} would request {law_np.predict(n) / 1e9:>10,.1f} GB"
              f"   (this run's RSS law: {law_rss.predict(n) / 1e9:,.1f} GB)")
    if ram:
        print(f"  {ram / 1e9:.1f} GB (this machine) is requested at "
              f"n ~ {law_np.solve_for(ram):,.0f}"
              f"   (this run's RSS law: n ~ {law_rss.solve_for(ram):,.0f})")

    # How closely the OS actually backs the request, printed rather than
    # asserted in prose -- this is the ratio whose run-to-run drift is the
    # reason the wall above is fitted on the NumPy column.
    print("\n  resident / requested, where both were measured:")
    for e in exact:
        if e.get("numpy_bytes") and e.get("rss_bytes") and e["n"] >= ASYMPTOTIC_FROM:
            print(f"    n = {e['n']:>6}: {e['rss_bytes'] / e['numpy_bytes']:.3f}")

    print("\nwhat the approximation gave up (M = 64, same data, same grid)")
    print(f"{'n':>7} {'exact lml':>12} {'ELBO gap':>10} {'max|dmean|':>11} "
          f"{'max rel sd err':>15}")
    for e in exact:
        s = next((x for x in sparse if x["n"] == e["n"]), None)
        if s is None:
            continue
        sd_e, sd_s = np.array(e["sd"]), np.array(s["sd"])
        print(f"{e['n']:>7} {e['objective']:>12.2f} "
              f"{e['objective'] - s['objective']:>10.2e} "
              f"{np.abs(np.array(e['mean']) - np.array(s['mean'])).max():>11.2e} "
              f"{np.abs(sd_s / sd_e - 1.0).max():>15.2e}")
    return exact, sparse, msweep


# -- figure -------------------------------------------------------------------


def figure(exact, sparse, msweep, ram):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4), constrained_layout=True)

    ax = axes[0]
    for rows, label, style in ((exact, "exact GP", "o-"),
                               (sparse, f"SGPR M={M_DEFAULT}", "s-")):
        ns = [r["n"] for r in rows]
        ax.plot(ns, [r["seconds"] for r in rows], style, label=label, ms=4)
        law, xs, _ = fitted(rows, "seconds", floor=ASYMPTOTIC_FROM)
        grid = np.array([xs.min(), max(ns)])
        ax.plot(grid, law.predict(grid), ":", color="0.4", lw=1)
        ax.annotate(f"$n^{{{law.exponent:.2f}}}$", (grid[-1], law.predict(grid)[-1]),
                    textcoords="offset points", xytext=(-2, -12), fontsize=8,
                    color="0.3", ha="right", va="top")
    n_cross = crossover(exact, sparse)
    if n_cross:
        ax.axvline(n_cross, color="0.7", lw=1, ls="--")
        ax.annotate(f"crossover\nn ~ {n_cross:.0f}", (n_cross, 0.55),
                    xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(6, 0), fontsize=8,
                    color="0.4")
    ax.set(xscale="log", yscale="log", xlabel="training points n",
           ylabel="fit + evidence + predict (s)", title="Time")
    ax.legend(loc="upper left")

    ax = axes[1]
    for rows, label, style in ((exact, "exact GP", "o-"),
                               (sparse, f"SGPR M={M_DEFAULT}", "s-")):
        ax.plot([r["n"] for r in rows], [r["rss_bytes"] / 1e9 for r in rows],
                style, label=label, ms=4)
    # Extrapolate on the deterministic column, not the resident one: RSS varies
    # ~30% run to run at the top size, so a wall drawn from it moves with it.
    law, xs, _ = fitted(exact, "numpy_bytes", floor=ASYMPTOTIC_FROM)
    grid = np.geomspace(xs.min(), 3e5, 40)
    ax.plot(grid, law.predict(grid) / 1e9, ":", color="0.4", lw=1,
            label=f"requested $\\propto n^{{{law.exponent:.2f}}}$, extrapolated")
    if ram:
        ax.axhline(ram / 1e9, color="0.7", lw=1, ls="--")
        n_wall = law.solve_for(ram)
        ax.annotate(f"this machine's RAM\nreached at n ~ {n_wall:,.0f}",
                    (grid[0], ram / 1e9), textcoords="offset points",
                    xytext=(2, 4), fontsize=8, color="0.4")
    # "peak memory", not "peak resident set": the points are resident bytes but
    # the extrapolation is the requested law, for the reproducibility reason in
    # the wall section above.
    ax.set(xscale="log", yscale="log", xlabel="training points n",
           ylabel="peak memory (GB)", title="Memory, and where it stops")
    ax.legend(loc="upper left")

    ax = axes[2]
    if msweep:
        Ms = [r["M"] for r in msweep]
        ax.plot(Ms, [r["seconds"] for r in msweep], "s-", ms=4, label="time (s)")
        ax.plot(Ms, [r["rss_bytes"] / 1e9 for r in msweep], "^-", ms=4,
                label="peak RSS (GB)")
        for key, dy in (("seconds", 10), ("rss_bytes", -12)):
            law, xs, _ = fitted(msweep, key, xkey="M")
            scale = 1.0 if key == "seconds" else 1e-9
            ax.plot(xs, law.predict(xs) * scale, ":", color="0.4", lw=1)
            ax.annotate(f"$M^{{{law.exponent:.2f}}}$",
                        (xs[-1], law.predict(xs)[-1] * scale),
                        textcoords="offset points", xytext=(-4, dy), fontsize=8,
                        color="0.3", ha="right")
    ax.set(xscale="log", yscale="log", xlabel="inducing points M",
           ylabel="cost", title=f"SGPR in M, at n = {N_FOR_M_SWEEP}")
    ax.legend(loc="upper left")

    savefig(fig, "cost_scaling.png")


_BASELINE_RSS = 0.0


def main():
    global _BASELINE_RSS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-n", type=int, default=max(N_SPARSE))
    parser.add_argument("--cell", nargs=1)
    args = parser.parse_args()
    if args.cell:
        # A cell subprocess: baseline the RSS after imports, so the number
        # reported is the workload's footprint and not the interpreter's.
        _BASELINE_RSS = max_rss()
        return cell_main(args.cell)

    ram = total_ram_bytes()
    ram_note = f"{ram / 1e9:.1f} GB RAM" if ram else "RAM unknown"
    print(f"machine: {os.cpu_count()} cpus, {ram_note}, {sys.platform}")
    cells = ([{"method": "exact", "n": n, "M": 0}
              for n in N_EXACT if n <= args.max_n]
             + [{"method": "sgpr", "n": n, "M": M_DEFAULT}
                for n in N_SPARSE if n <= args.max_n]
             + [{"method": "sgpr", "n": N_FOR_M_SWEEP, "M": M}
                for M in M_SWEEP if M != M_DEFAULT and N_FOR_M_SWEEP <= args.max_n])
    print(f"running {len(cells)} cells, one subprocess each")
    results = run_sweep(cells, ram)
    exact, sparse, msweep = report(results, ram)
    figure(exact, sparse, msweep, ram)


if __name__ == "__main__":
    main()
