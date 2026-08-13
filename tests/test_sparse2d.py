"""The 2D sparse experiment's own machinery, checked before it is believed.

``experiments/sparse2d.py`` answers a question the README's Limitations section
poses -- whether Sec. 11's density-over-difficulty result survives a change of
dimension -- and the answer depends on three pieces of new code: a from-scratch
k-means (the 2D replacement for "put Z at the data quantiles"), a
nearest-neighbour spacing statistic that must reduce to Sec. 11's
consecutive-gap statistic when d = 1, and the target itself.

k-means is checked against scikit-learn, which this repo uses as a test-only
independent oracle and never as a dependency of the library or the experiments.
The comparison is on inertia rather than on the labels: Lloyd's algorithm is a
local method with an arbitrary cluster ordering, so identical assignments are
not a fair thing to demand, while an inertia materially worse than the oracle's
would mean this implementation is not doing the job.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

import sparse2d as S  # noqa: E402


def _inertia(X, C):
    d2 = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)
    return float(d2.min(axis=1).sum())


# ---------------------------------------------------------------------------
# k-means
# ---------------------------------------------------------------------------
def test_kmeans_matches_the_sklearn_oracle_on_inertia():
    sk = pytest.importorskip("sklearn.cluster")
    X, _ = S.dataset(seed=1, n=600)
    for M in (4, 16, 40):
        mine = S.kmeans_Z(X, M, seed=0)
        oracle = sk.KMeans(n_clusters=M, n_init=10, random_state=0).fit(X)
        assert mine.shape == (M, 2)
        # Both are local optima of the same objective; ours must not be worse
        # than the oracle's by more than a few percent.
        assert _inertia(X, mine) <= 1.05 * float(oracle.inertia_)


def test_kmeans_actually_runs_to_a_fixed_point():
    """Lloyd must stop because it converged, not because it ran out of loop.

    This is the test that catches a broken termination condition, and it is
    here because the first version had one: ``prev = inf`` made the relative
    tolerance ``tol * inf = inf``, so ``inf <= inf`` fired on the first pass and
    every restart returned its seeded centres after a single update. The inertia
    was 16% above the oracle's and nothing else noticed. A converged result is a
    fixed point of the update, so raising the iteration cap must not improve it.
    """
    X, _ = S.dataset(seed=1, n=600)
    for M in (4, 16):
        tight = _inertia(X, S._kmeans_once(X, M, seed=(0, 0), iters=1000, tol=0.0)[0])
        default = _inertia(X, S._kmeans_once(X, M, seed=(0, 0), iters=100, tol=1e-8)[0])
        assert default <= tight * 1.0001


def test_kmeans_once_reports_the_inertia_of_the_centres_it_returns():
    """The restart loop picks the best by this number, so it has to describe the
    centres actually handed back -- not the ones from before the last update."""
    X, _ = S.dataset(seed=2, n=300)
    centres, inertia = S._kmeans_once(X, 10, seed=(0, 0), iters=100, tol=1e-8)
    assert abs(inertia - _inertia(X, centres)) < 1e-9


def test_kmeans_restarts_keep_the_best(monkeypatch):
    """``n_init`` must be a minimum over restarts, not the last one."""
    X, _ = S.dataset(seed=1, n=600)
    per = [S._kmeans_once(X, 16, seed=(0, i), iters=100, tol=1e-8)[1]
           for i in range(10)]
    assert abs(_inertia(X, S.kmeans_Z(X, 16, seed=0, n_init=10)) - min(per)) < 1e-9
    # and restarting is worth something here, or n_init would be dead weight
    assert min(per) < max(per) * 0.99


def test_kmeans_recovers_well_separated_clusters():
    """Three tight blobs far apart have one obvious answer, so a correct
    implementation must find it exactly -- no tolerance on the assignment."""
    rng = np.random.default_rng(0)
    centres = np.array([[-10.0, -10.0], [0.0, 10.0], [10.0, -10.0]])
    X = np.repeat(centres, 60, axis=0) + 0.05 * rng.standard_normal((180, 2))
    Z = S.kmeans_Z(X, 3, seed=0)
    # each true centre is matched by exactly one found centre
    dist = np.linalg.norm(Z[:, None, :] - centres[None, :, :], axis=2)
    assert sorted(dist.argmin(axis=1)) == [0, 1, 2]
    assert dist.min(axis=1).max() < 0.05


def test_kmeans_never_increases_inertia_with_more_centres():
    X, _ = S.dataset(seed=2, n=400)
    prev = np.inf
    for M in (2, 4, 8, 16, 32):
        inertia = _inertia(X, S.kmeans_Z(X, M, seed=0))
        assert inertia <= prev + 1e-9
        prev = inertia


def test_kmeans_is_deterministic_given_the_seed():
    X, _ = S.dataset(seed=3, n=300)
    assert np.array_equal(S.kmeans_Z(X, 12, seed=5), S.kmeans_Z(X, 12, seed=5))


def test_kmeans_handles_more_centres_than_points():
    X, _ = S.dataset(seed=4, n=7)
    assert S.kmeans_Z(X, 20, seed=0).shape == (7, 2)


# ---------------------------------------------------------------------------
# the other initializers
# ---------------------------------------------------------------------------
def test_subset_returns_distinct_data_points():
    X, _ = S.dataset(seed=0, n=500)
    Z = S.subset_Z(X, 30, seed=0)
    assert Z.shape == (30, 2)
    assert len(np.unique(Z, axis=0)) == 30
    for z in Z:
        assert np.min(np.linalg.norm(X - z, axis=1)) == 0.0


def test_grid_covers_the_bounding_box_and_reports_its_real_size():
    """M is rounded down to a near-square product, so the returned count can be
    smaller than requested -- the experiment prints the actual M for that
    reason, and this pins the behaviour rather than the intention."""
    X, _ = S.dataset(seed=0, n=500)
    lo, hi = X.min(axis=0), X.max(axis=0)
    for M, expected in ((4, 4), (16, 16), (32, 30), (64, 64), (128, 121)):
        Z = S.grid_Z(X, M)
        assert Z.shape[0] == expected
        assert np.allclose(Z.min(axis=0), lo) and np.allclose(Z.max(axis=0), hi)


# ---------------------------------------------------------------------------
# the spacing statistic must be Sec. 11's statistic
# ---------------------------------------------------------------------------
def test_nn_spacing_reduces_to_the_1d_consecutive_gap():
    """Sec. 11 measured the median gap between *consecutive* Z on a line. On a
    line the nearest neighbour of an interior point is one of its two
    neighbours, so on points spaced by a constant step the two statistics agree
    exactly -- which is what makes the 2D number comparable to the 1D one."""
    z = np.linspace(-4.0, 4.0, 17).reshape(-1, 1)
    step = float(np.median(np.diff(z[:, 0])))
    padded = np.concatenate([z, np.zeros_like(z)], axis=1)
    assert abs(S.nn_spacing(padded) - step) < 1e-12


def test_nn_spacing_on_a_known_lattice():
    g = np.linspace(0.0, 1.0, 5)
    A, B = np.meshgrid(g, g, indexing="ij")
    Z = np.stack([A.ravel(), B.ravel()], axis=1)
    assert abs(S.nn_spacing(Z) - 0.25) < 1e-12


def test_nn_spacing_is_undefined_for_a_single_point():
    assert np.isnan(S.nn_spacing(np.zeros((1, 2))))


def test_fast_share_counts_the_hard_half_plane():
    Z = np.array([[-1.0, 0.0], [1.0, 5.0], [2.0, -3.0], [-0.5, 1.0]])
    assert abs(S.fast_share(Z) - 0.5) < 1e-12
    # the two fast-half points are (1, 5) and (2, -3): sqrt(1 + 64)
    assert abs(S.half_spacing(Z, "fast") - np.sqrt(65.0)) < 1e-12


# ---------------------------------------------------------------------------
# the target and the density control
# ---------------------------------------------------------------------------
def test_target_reduces_to_the_1d_target_along_x2_equals_zero():
    """The 2D target is the Sec. 11 target plus a slow term in x2, so setting
    x2 = 0 (where sin(1.1 x2) vanishes) must return Sec. 11's function exactly.
    Imported from ``sparse.py`` rather than retyped, so the two cannot drift."""
    from sparse import target as target_1d

    x1 = np.linspace(-4.0, 4.0, 401)
    X = np.stack([x1, np.zeros_like(x1)], axis=1)
    assert np.max(np.abs(S.target(X) - target_1d(x1))) < 1e-14


def test_only_the_x1_axis_carries_the_fast_component():
    """The claim that x1 > 0 is the *only* hard region. Swapping the axes must
    change the field, and the x2 marginal structure must have no fast part:
    along x1 = -2 (slow half) the field is a pure 1.1-frequency sine in x2."""
    x2 = np.linspace(-4.0, 4.0, 801)
    X = np.stack([np.full_like(x2, -2.0), x2], axis=1)
    f = S.target(X) - np.sin(1.1 * -2.0)
    assert np.max(np.abs(f - np.sin(1.1 * x2))) < 1e-14


def test_density_control_skews_x1_and_leaves_x2_alone():
    """The control must change data density and nothing else -- if it also moved
    the x2 marginal, measurement 2 would not isolate density."""
    Xu, _ = S.dataset(slow_share=0.5, n=4000)
    Xc, _ = S.dataset(slow_share=0.75, n=4000)
    assert abs((Xu[:, 0] <= 0).mean() - 0.50) < 0.02
    assert abs((Xc[:, 0] <= 0).mean() - 0.75) < 0.02
    # x2 is uniform on the same range in both arms
    for X in (Xu, Xc):
        assert abs(X[:, 1].mean()) < 0.15
        assert X[:, 1].min() > S.LO - 1e-9 and X[:, 1].max() < S.HI + 1e-9


def test_dataset_is_reproducible():
    a, ya = S.dataset(seed=7, n=200)
    b, yb = S.dataset(seed=7, n=200)
    assert np.array_equal(a, b) and np.array_equal(ya, yb)


# ---------------------------------------------------------------------------
# the two read-outs the conclusions are stated in
# ---------------------------------------------------------------------------
def test_density_shift_has_the_same_sign_convention_as_the_1d_result():
    """Sec. 11's 1D number is 0.560 -> 0.410 under the skew, i.e. **-0.150**.

    The 2D number is printed next to it, so it has to be the same subtraction.
    The first version of that line computed ``uniform - control`` and reported
    +0.102 where the comparable 1D figure is -0.150 -- same direction in the
    data, opposite sign on the page.
    """
    uniform = [{"fast_share": 0.560}]
    control = [{"fast_share": 0.410}]
    assert abs(S.density_shift(uniform, control) - (-0.150)) < 1e-12

    # a skew toward the slow half that pulls Z that way is negative; the
    # opposite case must come out positive, or the sign carries no information
    assert S.density_shift([{"fast_share": 0.40}], [{"fast_share": 0.55}]) > 0


def test_density_shift_indexes_the_cell_it_is_asked_for():
    uniform = [{"fast_share": 0.5}, {"fast_share": 0.6}]
    control = [{"fast_share": 0.3}, {"fast_share": 0.1}]
    assert abs(S.density_shift(uniform, control, 0) - (-0.2)) < 1e-12
    assert abs(S.density_shift(uniform, control, 1) - (-0.5)) < 1e-12
    assert abs(S.density_shift(uniform, control) - (-0.5)) < 1e-12


def test_smooth_basin_flag_separates_the_two_optima():
    """The flag exists so that cells fitting a different function are not read
    as coarser answers to the same question. The two optima measured here are
    l = 0.535 and l = 1.807, so the 2x rule separates them with room to spare."""
    exact_l = 0.5349
    assert not S.in_smooth_basin({"lengthscale": 0.5349}, exact_l)
    assert not S.in_smooth_basin({"lengthscale": 0.615}, exact_l)
    assert S.in_smooth_basin({"lengthscale": 1.8073}, exact_l)
    assert S.in_smooth_basin({"lengthscale": 2.216}, exact_l)
    # and the boundary is where it says it is
    assert not S.in_smooth_basin({"lengthscale": 2 * exact_l - 1e-9}, exact_l)
    assert S.in_smooth_basin({"lengthscale": 2 * exact_l + 1e-9}, exact_l)


def test_the_two_optima_do_not_appear_at_a_quarter_of_the_data():
    """Measurement 0's 1222-nat split is *not* asserted here, because it does
    not survive being made cheap.

    A subsampled version was tried first: at n = 400 both starts converge to the
    same long lengthscale (2.10 against 2.10), so at a quarter of the data there
    is only one optimum to find and the phenomenon is absent rather than
    smaller. That is worth pinning as its own fact -- the multimodality is a
    property of this target *at this sample size*, not something the target has
    intrinsically -- and it stops the README's claim from being read as more
    general than it was measured to be. The full-size split is measured by the
    experiment itself, which ``reproduce.sh`` runs.
    """
    X, y = S.dataset(seed=0, n=400)
    import sparse2d

    saved = sparse2d.STEPS
    sparse2d.STEPS = 120
    try:
        long_start = S.exact_fit(X, y, dict(s2=0.5, l=2.0, noise=0.3))
        short_start = S.exact_fit(X, y, dict(s2=0.5, l=0.5, noise=0.3))
    finally:
        sparse2d.STEPS = saved

    l_long = float(np.exp(long_start.params[1]))
    l_short = float(np.exp(short_start.params[1]))
    assert abs(l_long - l_short) < 0.1 * l_long, (l_long, l_short)


def test_multistart_returns_the_best_start():
    X, y = S.dataset(seed=0, n=300)
    import sparse2d

    saved = sparse2d.STEPS
    sparse2d.STEPS = 80
    try:
        best, rows = S.exact_multistart(X, y)
    finally:
        sparse2d.STEPS = saved

    assert len(rows) == len(S.MULTISTART)
    assert abs(best.log_marginal_likelihood()
               - max(r["lml"] for r in rows)) < 1e-9


# ---------------------------------------------------------------------------
# one small end-to-end cell
# ---------------------------------------------------------------------------
def test_a_small_arm_runs_and_stays_under_the_exact_evidence():
    """The bound is a bound in 2D as well, which is the assertion the
    experiment makes on every cell -- checked here on a cheap one so the suite
    covers it without running the experiment."""
    from gp.gp import GPRegressor
    from gp.kernels import RBF

    X, y = S.dataset(seed=0, n=200)
    Xs = S.eval_grid(n=12)
    exact = GPRegressor(RBF(**S.INIT), noise_var=S.INIT_NOISE).fit(X, y)
    lml = exact.log_marginal_likelihood()

    import sparse2d

    saved = sparse2d.STEPS
    sparse2d.STEPS = 25
    try:
        r = S.fit_arm(X, y, Xs, exact.predict(Xs), 8, "kmeans", optimize_Z=True)
    finally:
        sparse2d.STEPS = saved

    assert r["M"] == 8
    assert r["elbo"] <= lml + 1e-6, "the bound crossed the exact log evidence"
    assert 0.0 <= r["fast_share"] <= 1.0
    assert np.isfinite(r["spacing"])
