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


def test_max_rss_grows_when_a_large_array_is_touched_and_never_comes_back_down():
    """Two properties in one fresh process, because both are about history.

    First, the mark responds to real use: writing a 200 MB array moves it by
    about 200 MB. Touched, not merely allocated -- RSS counts pages the process
    has written.

    Second, and this is the one that shapes `experiments/cost_scaling.py`: the
    mark does not come back down when the array is freed. A second, smaller
    workload in the same process is therefore unmeasurable, which is why the
    sweep runs one subprocess per cell rather than a loop. Measured in a fresh
    interpreter because a test that asserts "the peak just went up" is
    otherwise at the mercy of whatever the rest of the suite allocated first.
    """
    script = """
import numpy as np
from gp.bench import max_rss
mb = 1e6
before = max_rss()
block = np.ones((5000, 5000))          # 200 MB, fully written
after = max_rss()
del block
small = np.ones((100, 100))            # 80 KB
small += 1.0
print(before / mb, after / mb, max_rss() / mb)
"""
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True,
        cwd=str(pathlib.Path(__file__).resolve().parents[1]),
    )
    before, after, after_free = (float(v) for v in out.stdout.split())
    assert after - before > 150            # the 200 MB write shows up
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
