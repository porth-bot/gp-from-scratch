"""The cost-measurement machinery, checked against cases with known answers.

A benchmark is code, and code that measures is the easiest kind to get quietly
wrong: a timer that reports the mean of a contaminated sample, a memory probe
in the wrong unit, an exponent fitted in linear space. Every claim
experiments/cost_scaling.py makes about how the exact GP and SGPR scale is a
claim these four functions computed correctly, so they are pinned here against
allocations of a known size, sleeps of a known length, and power laws whose
exponents are known exactly.
"""

import pathlib
import subprocess
import sys
import time

import numpy as np
import pytest

from gp.bench import (
    best_of,
    local_exponents,
    max_rss,
    power_law_fit,
    traced_peak,
)


# -- timing -------------------------------------------------------------------


def test_best_of_returns_the_minimum_not_the_mean():
    """One slow run must not move the reported time. The whole reason for the
    minimum is that scheduler noise is one-sided: it can only add."""
    delays = iter([0.05, 0.01, 0.01])

    def work():
        time.sleep(next(delays))
        return "done"

    out, timing = best_of(work, min_seconds=1e9, max_reps=3)
    assert out == "done"
    assert timing.reps == 3
    assert timing.seconds < 0.03          # the 0.05 run is not in the answer


def test_best_of_stops_early_once_the_budget_is_met():
    """An expensive cell should be measured once, not seven times."""
    calls = []

    def work():
        calls.append(1)
        time.sleep(0.05)

    _, timing = best_of(work, min_seconds=0.01, max_reps=7)
    assert timing.reps == 1 and len(calls) == 1


def test_best_of_repeats_a_cheap_call_up_to_the_cap():
    _, timing = best_of(lambda: None, min_seconds=1e9, max_reps=4)
    assert timing.reps == 4


def test_best_of_rejects_a_nonsense_repeat_count():
    with pytest.raises(ValueError, match="at least 1"):
        best_of(lambda: None, max_reps=0)


# -- memory -------------------------------------------------------------------


def test_traced_peak_counts_a_float64_matrix_at_eight_bytes_a_word():
    """The portable memory claim rests on this: NumPy reports its allocations
    to tracemalloc, so an (n, n) float64 array registers exactly 8 n^2 bytes."""
    n = 400
    _, peak = traced_peak(lambda: np.zeros((n, n)))
    assert 8 * n * n <= peak < 8 * n * n * 1.02


def test_exact_fit_peak_is_a_small_multiple_of_the_gram_matrix():
    """The memory wall in experiments/cost_scaling.py is extrapolated from a
    law of the form ``c * 8 n^2``, so both halves of that need pinning -- and
    they turn out to deserve very different confidence.

    **The exponent is portable, the constant is not.** The peak is a whole
    number of n x n float64 copies and grows exactly as n^2 everywhere; how
    many copies depends on the NumPy build. On the pinned 2.5.0 it is 3.00 --
    K, LAPACK's working copy, and the factor, three because K stays alive
    while the Cholesky runs -- and on CI's 2.0.2 it is 2.03, because there the
    working copy is taken somewhere tracemalloc cannot see it. So the wall's
    24 n^2 is this environment's constant, which is why cost_scaling.py fits
    the prefactor rather than hardcoding it, and why the README quotes the
    figure with its NumPy version attached.

    This is still the trustworthy column: it is bit-identical across runs on
    one environment, where peak RSS on the same fit varied 4.15-5.89 GB across
    five runs of one idle machine (n = 16000).
    """
    from gp.gp import GPRegressor
    from gp.kernels import RBF

    def peak_copies(n):
        rng = np.random.default_rng(0)
        X = rng.uniform(-3, 3, size=(n, 1))
        y = np.sin(X[:, 0])
        _, peak = traced_peak(lambda: GPRegressor(RBF(s2=1.0, l=0.8),
                                                  noise_var=0.1).fit(X, y))
        return peak / (8 * n * n)

    small, large = peak_copies(400), peak_copies(800)

    # The law: doubling n quadruples the peak, so the copy count is flat in n.
    assert large == pytest.approx(small, rel=0.02), (small, large)
    # The constant: at least 2 (the matrix and its factor must both be live),
    # never more than 3, and close to a whole number of copies either way.
    for copies in (small, large):
        assert 1.98 < copies < 3.05, copies
        assert min(abs(copies - c) for c in (2, 3)) < 0.05, copies


def test_traced_peak_sees_the_peak_not_the_survivor():
    """Two matrices alive at once, one returned: the peak must show both. This
    is the property that makes the number meaningful for a Cholesky, whose
    input and output are both live at the moment of highest use."""
    n = 400
    one = 8 * n * n

    def two_then_one():
        a = np.zeros((n, n))
        b = a + 1.0
        del a
        return b

    _, peak = traced_peak(two_then_one)
    assert 2 * one <= peak < 2.02 * one


def test_traced_peak_returns_the_value_and_leaves_tracing_off():
    was_tracing_before = _is_tracing()
    out, _ = traced_peak(lambda: 17)
    assert out == 17
    assert _is_tracing() == was_tracing_before


def _is_tracing():
    import tracemalloc

    return tracemalloc.is_tracing()


def test_max_rss_is_in_bytes_on_this_platform():
    """ru_maxrss is bytes on macOS and kilobytes on Linux, and there is no
    runtime way to ask which -- so the unit table is a claim about the platform
    that deserves a test. A Python process with NumPy imported is comfortably
    above 8 MB and nowhere near 8 TB; only one of the two unit choices puts the
    measured number inside that window."""
    rss = max_rss()
    assert 8e6 < rss < 8e12, (rss, sys.platform)


def test_max_rss_covers_a_touched_array_and_never_comes_back_down():
    """Two properties in one fresh process, both about history.

    First, the mark covers memory that is really in use: after writing every
    page of an N-byte array, the peak is at least about N. That is the
    assertion that pins the *unit* against real use -- a kilobyte/byte mix-up
    misses it by 1024x.

    Second, and this is the one that shapes `experiments/cost_scaling.py`: the
    mark does not come back down when the array is freed, so a second, smaller
    workload in the same process is unmeasurable, which is why the sweep runs
    one subprocess per cell rather than a loop.

    What this deliberately does *not* assert is that the mark went up by the
    size of the array, and two CI failures are the reason. On this machine a
    fresh interpreter peaks at 27 MB, so a 200 MB write moved it by 200 MB. On
    the GitHub runner, importing NumPy leaves a 490 MB mark, and writing a
    495 MB array moved it by 32 MB -- the allocator handed back pages the
    process had already touched and released, which are resident and counted
    already. Both observations are correct behavior for a high-water mark of
    *resident* pages, and only the covering property is portable between them.
    """
    script = """
import numpy as np
from gp.bench import max_rss
mb = 1e6
before = max_rss()
want = 1.2 * before + 200e6            # larger than the mark it has to clear
n = int((want / 8) ** 0.5)
block = np.ones((n, n))                # 8 n^2 bytes, every page written
after = max_rss()
del block
small = np.ones((100, 100))            # 80 KB
small += 1.0
print(before / mb, after / mb, max_rss() / mb, 8 * n * n / mb)
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
    )
    before, after, after_free, written = (float(v) for v in out.stdout.split())
    assert after >= 0.9 * written          # the mark covers what was resident
    assert after >= before                 # monotone
    assert after_free >= after             # and never comes back down


# -- scaling exponents ---------------------------------------------------------


def test_power_law_fit_recovers_an_exact_law():
    x = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    law = power_law_fit(x, 1.7e-9 * x**3)
    assert law.exponent == pytest.approx(3.0, abs=1e-10)
    assert law.prefactor == pytest.approx(1.7e-9, rel=1e-9)
    assert law.r2 == pytest.approx(1.0, abs=1e-12)


def test_power_law_fit_is_not_dominated_by_the_largest_point():
    """The fit is in log space, so each decade counts once. Fitted in linear
    space, the top point would swamp the rest: here the two smallest points sit
    exactly on a different exponent, and a linear-space fit would barely see
    them move the answer at all."""
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    y = np.array([1.0, 100.0, 1e4, 1e6])          # exactly x^2
    law = power_law_fit(x, y)
    assert law.exponent == pytest.approx(2.0, abs=1e-12)

    # perturb the smallest point by 2x; a log-space fit must notice
    y_perturbed = y.copy()
    y_perturbed[0] *= 2.0
    assert abs(power_law_fit(x, y_perturbed).exponent - 2.0) > 0.05


def test_power_law_r2_falls_when_the_curve_bends():
    """A cost curve with fixed overhead is not a power law at small x, and r2
    is what says so. Straight in log-log gives 1; overhead plus a cubic does
    not."""
    x = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    assert power_law_fit(x, 3.0 * x**2).r2 == pytest.approx(1.0, abs=1e-12)
    assert power_law_fit(x, 50.0 + 3.0 * x**2).r2 < 0.99


def test_power_law_solve_for_inverts_the_fit():
    """The memory wall is an extrapolation through this inverse, so it has to
    invert exactly, including well outside the fitted range."""
    x = np.array([100.0, 200.0, 400.0])
    law = power_law_fit(x, 8.0 * x**2)
    assert law.solve_for(8.0 * 5e4**2) == pytest.approx(5e4, rel=1e-9)
    assert law.predict(law.solve_for(1e10)) == pytest.approx(1e10, rel=1e-9)


def test_power_law_fit_refuses_input_it_cannot_take_a_log_of():
    with pytest.raises(ValueError, match="positive"):
        power_law_fit([1.0, 2.0], [1.0, 0.0])
    with pytest.raises(ValueError, match="two points"):
        power_law_fit([1.0], [1.0])
    with pytest.raises(ValueError, match="same shape"):
        power_law_fit([1.0, 2.0], [1.0, 2.0, 3.0])


def test_local_exponents_expose_a_ramp_the_global_fit_averages_away():
    """The case this exists for: costs that are flat at small n and cubic at
    large n. One fitted exponent reports something in between and looks like a
    disagreement with theory; the local sequence shows the approach."""
    x = np.array([1.0, 10.0, 100.0, 1000.0, 1e4])
    y = 1.0 + 1e-6 * x**3                          # overhead, then cubic
    local = local_exponents(x, y)
    assert local[0] < 0.5                          # overhead regime
    assert local[-1] == pytest.approx(3.0, abs=0.01)   # asymptotic regime
    assert power_law_fit(x, y).exponent < 2.6      # the average of the two


def test_local_exponents_length_and_guards():
    x = np.array([1.0, 2.0, 4.0, 8.0])
    assert len(local_exponents(x, x**2)) == 3
    with pytest.raises(ValueError, match="two points"):
        local_exponents([1.0], [1.0])
    with pytest.raises(ValueError, match="positive"):
        local_exponents([1.0, 2.0], [1.0, -1.0])
