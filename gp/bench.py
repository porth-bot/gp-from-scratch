"""Measuring cost: wall clock, peak memory, and the exponent they scale at.

Every complexity claim in this repo -- O(n^3) time and O(n^2) memory for the
exact GP (gp/gp.py), O(n M^2) and O(n M) for the sparse one (gp/sparse.py),
O(n R^2) and O(n R) for random features (gp/rff.py) -- is an assertion about
asymptotics that nothing here has yet checked. This module is the machinery
for checking them, kept separate from the experiment that uses it
(experiments/cost_scaling.py) so that the measurement itself can be tested
against cases with known answers instead of being trusted because it looks
reasonable.

Three things are measured, and they are not equally portable:

- **Wall clock** (`best_of`). Machine-dependent in its absolute value and
  fairly reproducible in its *exponent*, which is the part the theory predicts.
- **NumPy bytes** (`traced_peak`). The peak of what NumPy asked the allocator
  for, via tracemalloc, which NumPy reports into. Deterministic: the same code
  on the same inputs requests the same bytes on every machine and every run.
  This is the portable memory claim.
- **Resident set** (`max_rss`). What the OS actually gave the process, at its
  high-water mark. This is the number that decides whether a fit runs at all,
  and it is *not* the same as the traced one in either direction: it counts
  BLAS workspace and allocator slack that NumPy never requested, and it misses
  pages that were allocated but never touched (a calloc'd n x n matrix whose
  diagonal alone is written costs n pages of RSS, not n^2 words).

The split matters for a repro claim. `experiments/sklearn_parity.py` already
draws this line for timing -- the accuracy column is portable, the wall-clock
column is not -- and the same line runs through here.

Peak-RSS caveat that shapes how this module is used: `getrusage` reports a
high-water *mark*, monotone over the life of the process, so it cannot be
reset between two workloads in one process. Measuring the peak of several
workloads therefore needs one process each. That is why
`experiments/cost_scaling.py` runs its sweep as one subprocess per cell rather
than as a loop.
"""

from __future__ import annotations

import resource
import sys
import time
import tracemalloc
from typing import Callable, NamedTuple, Sequence, TypeVar

import numpy as np

T = TypeVar("T")

# getrusage's ru_maxrss is bytes on macOS/BSD and kilobytes on Linux. POSIX
# does not fix the unit and there is no runtime way to ask, so it is a platform
# table; getting it wrong is a silent factor of 1024.
_RSS_UNIT = 1.0 if sys.platform == "darwin" else 1024.0


class Timing(NamedTuple):
    """Result of `best_of`: the best time seen, and how many runs it took."""

    seconds: float
    reps: int


class PowerLaw(NamedTuple):
    """A fitted ``y = a * x^b``, with the quality of the fit alongside it.

    ``r2`` is computed in log-log space, where the fit was done. It is a
    goodness-of-fit for the *straight line*, not for the curve -- an r2 of
    0.999 on three decades of x still permits a visible systematic bend, which
    is exactly what a cost curve with a fixed overhead does at small x. Read it
    with the residuals, not instead of them.
    """

    exponent: float
    prefactor: float
    r2: float

    def predict(self, x: "float | np.ndarray") -> "float | np.ndarray":
        """a * x^b."""
        return self.prefactor * np.asarray(x, dtype=float) ** self.exponent

    def solve_for(self, y: float) -> float:
        """The x at which the fitted law reaches y: (y / a)^(1/b).

        Used to answer "at what n does this stop fitting in memory", which is a
        question about a value of x no run ever reached -- so the answer is an
        extrapolation and should be reported as one.
        """
        if self.exponent == 0.0:
            raise ValueError("a flat law never reaches a new y")
        return float((y / self.prefactor) ** (1.0 / self.exponent))


def best_of(fn: "Callable[[], T]", min_seconds: float = 0.2,
            max_reps: int = 7) -> "tuple[T, Timing]":
    """Run ``fn`` repeatedly; return its last result and the *best* time.

    Minimum, not mean. Wall clock is a lower bound plus contamination -- other
    processes, frequency scaling, page faults -- and the contamination is
    one-sided, so the minimum is the closest estimate of the cost of the work
    itself. The mean would measure the machine's mood.

    Repetitions stop once the accumulated time passes ``min_seconds`` (or
    ``max_reps`` is hit), so cheap cells get averaged over many runs and
    expensive ones run once. A single run of a 60-second fit is already far
    above timer noise; a single run of a 3-millisecond one is not.

    Examples
    --------
    >>> import time
    >>> out, timing = best_of(lambda: time.sleep(0.01) or "done", min_seconds=0.02)
    >>> out
    'done'
    >>> timing.reps >= 2 and timing.seconds >= 0.009
    True
    """
    if max_reps < 1:
        raise ValueError("max_reps must be at least 1")
    best = np.inf
    total = 0.0
    reps = 0
    result: T
    while reps < max_reps:
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        total += dt
        reps += 1
        if total >= min_seconds:
            break
    return result, Timing(float(best), reps)


def traced_peak(fn: "Callable[[], T]") -> "tuple[T, int]":
    """Run ``fn`` under tracemalloc; return its result and the peak bytes.

    Counts what NumPy *requested* -- NumPy routes its array allocations through
    tracemalloc's hook, so an (n, n) float64 array registers exactly 8 n^2
    bytes whether or not the pages are ever touched. The number is therefore
    reproducible run to run, which `max_rss` is not -- but reproducible is not
    the same as identical everywhere: how many temporaries a given call
    allocates is a property of the NumPy build, and a fit that peaks at three
    copies of the Gram matrix on NumPy 2.5.0 peaks at two on 2.0.2, where the
    Cholesky's working copy is taken out of tracemalloc's sight.

    What it does not see: memory allocated inside compiled dependencies that do
    not report to tracemalloc. LAPACK's Cholesky is out-of-place in NumPy, so
    its output *is* counted, but any scratch space the BLAS takes for itself is
    not. Peak RSS is the check on that, and the two are reported side by side
    in `experiments/cost_scaling.py` for exactly this reason.

    Tracing costs time (every allocation takes a Python-level hook), so this
    must not wrap a call that is also being timed.

    Examples
    --------
    >>> import numpy as np
    >>> n = 500
    >>> _, peak = traced_peak(lambda: np.zeros((n, n)))
    >>> peak >= 8 * n * n
    True
    >>> peak < 8 * n * n * 1.05          # one matrix, not two
    True
    """
    already = tracemalloc.is_tracing()
    if not already:
        tracemalloc.start()
    else:                       # nested use: do not clobber the outer trace
        tracemalloc.reset_peak()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not already:
            tracemalloc.stop()
    return result, int(peak)


def max_rss() -> float:
    """Peak resident set of this process, in bytes, since it started.

    A high-water mark: monotone, and not resettable. Two workloads in one
    process share one number, and the second can only be measured as an
    increment over the first if it allocates more than the first ever did --
    which is why the sweep in `experiments/cost_scaling.py` forks per cell.

    That also sets a floor on what this can measure at all. Starting the
    interpreter and importing NumPy is itself a workload, and its mark is the
    one every later measurement has to clear: about 27 MB on the machine the
    committed figures come from, and about 490 MB on the GitHub runner, where
    the BLAS reserves far more up front. Any cell whose own footprint is below
    that floor reports a delta near zero -- not because it used no memory, but
    because it never touched more than the interpreter already had. Traced
    NumPy bytes have no such floor, which is the other reason both are
    reported.
    """
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * _RSS_UNIT


def power_law_fit(x: "Sequence[float] | np.ndarray",
                  y: "Sequence[float] | np.ndarray") -> PowerLaw:
    """Least squares for ``y = a x^b``, fitted as a line in log-log space.

    Fitting the line rather than the curve is a choice with consequences worth
    stating. Minimizing squared error in log space weights every decade of x
    equally and every point's *relative* error equally, which is what a scaling
    exponent means; minimizing it in linear space would let the largest n
    dominate the fit entirely. The cost is that the fit is no longer unbiased
    for a, and that a point at t = 0 is not representable at all.

    Requires at least two points with x > 0 and y > 0.

    Examples
    --------
    An exact power law is recovered exactly:

    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 4.0, 8.0])
    >>> law = power_law_fit(x, 3.5 * x ** 2.25)
    >>> round(law.exponent, 12), round(law.prefactor, 10), round(law.r2, 12)
    (2.25, 3.5, 1.0)

    And inverted, which is how the memory wall is located:

    >>> round(law.solve_for(3.5 * 100.0 ** 2.25), 6)
    100.0
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape}, {y.shape}")
    if x.size < 2:
        raise ValueError("need at least two points to fit an exponent")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("power-law fitting needs strictly positive x and y")

    lx, ly = np.log(x), np.log(y)
    slope, intercept = np.polyfit(lx, ly, 1)
    resid = ly - (slope * lx + intercept)
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    # A flat y is a legitimate fit (exponent 0) with no variance to explain;
    # calling that r2 = 1 is the convention that keeps the degenerate case from
    # dividing by zero, and it is honest here because the residuals are zero.
    r2 = 1.0 if ss_tot == 0.0 else 1.0 - float(np.sum(resid**2)) / ss_tot
    return PowerLaw(float(slope), float(np.exp(intercept)), r2)


def local_exponents(x: "Sequence[float] | np.ndarray",
                    y: "Sequence[float] | np.ndarray") -> np.ndarray:
    """Exponent between each consecutive pair: log(y2/y1) / log(x2/x1).

    The fitted exponent of `power_law_fit` is one number for the whole sweep,
    and it hides the thing worth seeing: a cost curve typically has a *rising*
    local exponent, starting near 0 where fixed overhead dominates and
    approaching the asymptotic value only at the largest sizes. Reporting the
    global fit alone would let a sweep that never reached the asymptotic regime
    quietly report an exponent below the theory and call it agreement.

    Returns an array of length ``len(x) - 1``.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 4.0])
    >>> np.round(local_exponents(x, 2.0 * x ** 3), 10)
    array([3., 3.])
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two points for a local exponent")
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("local exponents need strictly positive x and y")
    return np.diff(np.log(y)) / np.diff(np.log(x))
