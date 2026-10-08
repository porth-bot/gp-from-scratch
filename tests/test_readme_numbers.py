"""The README's measured numbers, checked against the logs the runs wrote.

Three of the five repos in this series already have this instrument, and in
every one of them it found drifted numbers on its first run: three in
grokking's README, seven in diffusion's, seven in pinn's. This repo did not
have it, and could not: it committed **no logs at all**. Twenty-one figures and
a `reproduce.sh` that checks all of them byte-for-byte, so no *figure* here can
go stale -- and nothing whatever looking at a number typed into a table, of
which the README has several hundred across sixteen result sections.

`experiments/common.save_results` closes the first half of that: an experiment
now writes the quantities its section quotes to `logs/<name>.json` beside the
figure it saves, and `reproduce.sh` reports a log that comes back different the
same way it reports a figure that does. This file is the second half.

**It covers ten of the sixteen sections so far**, and `NOT_YET` below names the
other seven explicitly rather than leaving the hole to be discovered:
`test_every_result_section_is_either_instrumented_or_listed` fails when a
section is added or renamed, so extending the README forces a decision about
its numbers instead of quietly widening the gap. Sections go in as their
experiments learn to write logs, cheapest script first: §§1 and 9 arrived
together, then §§4 to 7, all of whose scripts finish in seconds, then §§3 and 8.

**Three drifted cells so far, and all three are the same mistake.** Every one
is a number rounded once by the script and then rounded again by hand, in the
direction that flattered the section:

* §9 gave the multi-start fit a lengthscale of 1.29, which is the *signal-mode*
  fit's value. `multistart.py` prints three decimals, and 1.285 rounded up by
  hand becomes 1.29 while the 1.2845 behind it goes down to 1.28. The two rows
  are separate Adam runs that land in the same basin, and the table presented
  them as landing on identical parameters.
* §5's two-stage fit covered 0.965 on the noisy half, printed at three decimals
  and typed as 0.96.
* §4's n=100 max|dstd| read 1.6e-10 for a measured 1.6512e-10.

The fix in each case is to quote the precision the script prints, not to nudge
the last digit, because a table written to one digit fewer than the run reports
invites the same error again. §§1, 3, 6 and 7 came back clean.

§8 was a different kind of miss. Its cells were all right, and the sentence
under its table was not: "at n=140 the two are indistinguishable to three
decimals", beside a row reading 0.110 / 0.112 and -0.868 / -0.848. No n in the
sweep satisfies it. The sentence now quotes the smooth-region RMSE gaps, which
`test_section_8_the_gap_is_paid_in_data` checks.

Where a column is not the kind of thing a stored number can pin, this file says
so instead of asserting it: §4's residues move with the BLAS and its
milliseconds with the machine, so the timings are not logged at all and the
claim under test is "every residue below 1e-9" rather than the digits. See
`test_section_4_parity_residues_are_below_the_claimed_order`.

Pure stdlib plus numpy (which the whole suite already needs), so this runs
wherever the rest of the tests do. No matplotlib, no experiment imports.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
LOGS = ROOT / "logs"
README = (ROOT / "README.md").read_text()

# Which result section each committed log backs. Keys are log stems; values are
# the "### N." the section heading opens with.
INSTRUMENTED = {
    "validate": "1.",
    "ntk_experiments": "3.",
    "sklearn_parity": "4.",
    "heteroscedastic": "5.",
    "ard": "6.",
    "spatial2d": "7.",
    "gibbs_kernel": "8.",
    "multistart": "9.",
    "rff": "10.",
    "sparse": "11.",
}

# The sections whose experiments do not write a log yet. Listed, not silent:
# the test below pins this against the README's own headings.
NOT_YET = [
    "2.",
    "12.", "13.", "14.", "15.", "16.",
]


# -- reading the README ------------------------------------------------------

def section(number: str) -> str:
    """The text of the `### <number> ...` result section."""
    parts = re.split(r"^### ", README, flags=re.M)
    hits = [p for p in parts if p.startswith(number + " ")]
    assert len(hits) == 1, f"'### {number}' matched {len(hits)} sections"
    return hits[0]


def cells(text: str, row_starts_with: str) -> list[str]:
    """The cells of the one markdown table row beginning with the given label.

    Bold markers and the non-breaking minus the README uses for negative
    numbers (U+2212, not a hyphen) are stripped here, so a caller compares
    numbers rather than typography.
    """
    rows = [
        ln for ln in text.splitlines()
        if ln.startswith("|") and ln.lstrip("| ").startswith(row_starts_with)
    ]
    assert len(rows) == 1, (
        f"{row_starts_with!r} begins {len(rows)} table rows, expected 1"
    )
    # Split on unescaped pipes only: a label like `\|excess kurtosis\|` is
    # one cell, not three.
    out = [c.strip().replace("**", "").replace("−", "-")
           for c in re.split(r"(?<!\\)\|", rows[0].strip().strip("|"))]
    return out


def quoted(text: str, pattern: str) -> str:
    """The one capture group of `pattern` in `text`."""
    hits = re.findall(pattern, text)
    assert len(hits) == 1, f"{pattern!r} matched {len(hits)} times, expected 1"
    return hits[0]


def assert_rounds_to(measured: float, printed: str, what: str) -> None:
    """The README's rounded number is what the measurement rounds to.

    Checking a tolerance would accept 0.0165 printed as 0.016; checking the
    rounding does not. The decimals come from the printed string, so the test
    holds the README to the precision it chose to claim.
    """
    printed = printed.strip()
    decimals = len(printed.split(".")[1]) if "." in printed else 0
    assert round(float(measured), decimals) == float(printed), (
        f"{what}: README prints {printed}, the log holds {measured!r}, which "
        f"rounds to {round(float(measured), decimals)} at {decimals} dp"
    )


def assert_formats_to(measured: float, printed: str, what: str) -> None:
    """Same idea as ``assert_rounds_to``, for a number written as 1.7e-10.

    ``assert_rounds_to`` counts decimals after the point, which is meaningless
    in scientific notation ("1.7e-10" would read as five). Here the printed
    string fixes the significant digits and the measurement is reformatted the
    same way, which is how the cell was produced in the first place.
    """
    mantissa = printed.strip().split("e")[0]
    sig = len(mantissa.split(".")[1]) if "." in mantissa else 0
    formatted = f"{float(measured):.{sig}e}"
    assert formatted == printed.strip(), (
        f"{what}: README prints {printed}, the log holds {measured!r}, which "
        f"formats to {formatted}"
    )


def log(name: str) -> dict:
    path = LOGS / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} is not present; run its experiment")
    return json.loads(path.read_text())


# -- Sec. 1: is the uncertainty real? (experiments/validate.py) --------------

def test_section_1_calibration_at_the_nominal_95_level():
    cal = log("validate")["calibration"]
    body = section("1.")
    assert_rounds_to(
        cal["coverage_at_nominal_95"],
        cells(body, "empirical coverage")[1],
        "Sec. 1 empirical coverage at nominal 95%",
    )
    # "On 40 functions drawn from a known Matern prior" -- and 100 held-out
    # latents per function, which is where the z-score count comes from.
    assert int(quoted(body, r"On (\d+) functions drawn")) == cal["n_functions"]
    assert cal["n_zscores"] == cal["n_functions"] * 100


def test_section_1_calibration_curve_tracks_the_diagonal():
    """The claim the figure makes, as a number rather than a picture.

    "nominal q% intervals should cover q%, *for every* q, not just 95%" is the
    section's actual assertion; the single table cell above only pins q = 95.
    The measured worst deviation across the 40-point nominal grid is 0.030.
    """
    cal = log("validate")["calibration"]
    nominal = np.asarray(cal["curve_nominal"])
    empirical = np.asarray(cal["curve_empirical"])
    assert nominal.shape == empirical.shape
    assert abs(empirical - nominal).max() < 0.05, (
        "the reliability curve no longer tracks the diagonal, which §1's "
        f"'for every q' claim rests on (worst gap {abs(empirical - nominal).max():.3f})"
    )


def test_section_1_hyperparameter_recovery_table():
    rec = log("validate")["recovery"]
    body = section("1.")
    assert int(quoted(body, r"over (\d+) replicate datasets")) == rec["n_replicates"]
    for row, key in (
        (r"$\sigma_f^2$", "s2"),
        (r"$\ell$", "l"),
        (r"$\sigma_n^2$", "noise_var"),
    ):
        _, truth, median = cells(body, row)
        assert float(truth) == rec["truth"][key], (
            f"Sec. 1 quotes a truth of {truth} for {key}, the run generated "
            f"data from {rec['truth'][key]}"
        )
        assert_rounds_to(rec["median"][key], median,
                         f"Sec. 1 median ML-II estimate of {key}")


# -- Sec. 3: wide networks are GPs (experiments/ntk_experiments.py) --------

def width_headers(text: str) -> list[list[int]]:
    """The `| width $m$ | ...` header of each table in a section, in order."""
    return [
        [int(c) for c in re.split(r"\s*\|\s*", ln.strip().strip("|"))[1:]]
        for ln in text.splitlines() if ln.startswith("| width $m$ |")
    ]


def test_section_3_gaussianity_table():
    d = log("ntk_experiments")["nngp_convergence"]
    body = section("3.")
    assert width_headers(body)[0] == d["widths"]
    kurt = cells(body, r"\|excess kurtosis\|")[1:]
    var = cells(body, r"\|var − NNGP\|/NNGP")[1:]
    assert len(kurt) == len(var) == len(d["widths"])
    for m, k_cell, k in zip(d["widths"], kurt, d["abs_excess_kurtosis"]):
        assert_rounds_to(k, k_cell, f"§3 |excess kurtosis| at m={m}")
    for m, v_cell, v in zip(d["widths"], var, d["rel_var_error"]):
        assert_rounds_to(v, v_cell, f"§3 relative variance error at m={m}")


def test_section_3_variance_is_exact_and_gaussianity_arrives_at_rate_one_over_m():
    """The two claims the table is there to support, stated as numbers.

    "its *variance* matches at any width (to <=0.3% here)": the worst relative
    error across the four widths, against the bound the prose types. "but
    *Gaussianity* only arrives at rate 1/m": for a sum of m i.i.d. per-neuron
    terms the excess kurtosis is exactly kappa/m, so the log-log slope over
    widths 4..256 has to be -1 up to Monte Carlo noise. Measured: -1.006.
    """
    d = log("ntk_experiments")["nngp_convergence"]
    body = section("3.")
    bound = float(quoted(body, r"to \$\\le([\d.]+)\\%\$\s+here")) / 100
    assert max(d["rel_var_error"]) <= bound, (
        f"§3 says the variance matches to <={bound:.1%}; the worst width is "
        f"off by {max(d['rel_var_error']):.2%}"
    )
    slope = np.polyfit(np.log(d["widths"]), np.log(d["abs_excess_kurtosis"]), 1)[0]
    assert abs(slope + 1.0) < 0.1, (
        f"§3 says Gaussianity arrives at rate 1/m; the kurtosis decays with "
        f"log-log slope {slope:.3f}"
    )


def test_section_3_linearization_table_and_its_trend():
    d = log("ntk_experiments")["linearization"]
    body = section("3.")
    assert width_headers(body)[1] == d["widths"]
    gap = cells(body, r"max \|net − linearized\|")[1:]
    assert len(gap) == len(d["widths"])
    for m, cell, per_seed, mean in zip(d["widths"], gap, d["per_seed_max_gap"],
                                       d["mean_max_gap"]):
        # The cell is a mean over seeds, so check it is the mean of the seeds
        # the log holds and not a best or last seed.
        assert len(per_seed) == 5
        assert abs(float(np.mean(per_seed)) - mean) < 1e-12
        assert_rounds_to(mean, cell, f"§3 max|net - linearized| at m={m}")
    # "The max gap ... shrinks with width": at every step, not just end to end.
    means = d["mean_max_gap"]
    assert all(a > b for a, b in zip(means, means[1:])), (
        f"§3 says the gap shrinks with width; the means are {means}"
    )


# -- Sec. 4: parity vs scikit-learn (experiments/sklearn_parity.py) ----------

def test_section_4_parity_residues_are_below_the_claimed_order():
    """The portable half of §4, and the only half worth asserting.

    The residues are the last bits of two different orderings of the same
    O(n^3) arithmetic, so they move with the BLAS the way the section's
    millisecond columns move with the machine. "The from-scratch math is
    correct to ~1e-10" is the claim that survives a rebuild, so that is what is
    checked here, at every size the script runs rather than only the three the
    table shows. The timings are not in the log at all.
    """
    parity = log("sklearn_parity")["parity"]
    assert len(parity) >= 3
    for n, row in parity.items():
        for col in ("mean_maxdiff", "std_maxdiff"):
            assert 0.0 <= row[col] < 1e-9, (
                f"§4 claims agreement to ~1e-10; at n={n} the {col} is "
                f"{row[col]:.2e}"
            )
    claimed = quoted(section("4."), r"correct to ~1e-(\d+)")
    worst = max(r[c] for r in parity.values() for c in ("mean_maxdiff", "std_maxdiff"))
    assert worst < 10 ** -(int(claimed) - 1), (
        f"§4 says ~1e-{claimed}; the worst residue is {worst:.2e}"
    )


def test_section_4_table_cells_match_the_run_that_produced_them():
    """The typed digits, against the log, on the pinned build only.

    This is the check that caught the n=100 std cell reading 1.6e-10 for a
    measured 1.6512e-10. It is meaningful because the log and the README were
    produced by the same pinned environment; a reader on another BLAS should
    expect these to differ, which is why the log is whitelisted in
    reproduce.sh and why the test above is the one that carries the claim.
    """
    parity = log("sklearn_parity")["parity"]
    body = section("4.")
    for n in ("100", "400", "800"):
        _, mean_cell, std_cell, *_ = cells(body, f"{n} |")
        assert_formats_to(parity[n]["mean_maxdiff"], mean_cell,
                          f"§4 n={n} max|dmean|")
        assert_formats_to(parity[n]["std_maxdiff"], std_cell,
                          f"§4 n={n} max|dstd|")


# -- Sec. 5: heteroscedastic noise (experiments/heteroscedastic.py) ----------

@pytest.mark.parametrize("row,arm", [
    ("homoscedastic", "homoscedastic"),
    ("heteroscedastic", "heteroscedastic"),
])
def test_section_5_calibration_table(row, arm):
    d = log("heteroscedastic")[arm]
    body = section("5.")
    _, left, right, nll = cells(body, row + " |")
    # The cells carry a parenthetical ("1.000 (over-covers)"); take the number.
    for cell, key, label in (
        (left, "cover_left", "left coverage"),
        (right, "cover_right", "right coverage"),
        (nll, "nll", "test NLL"),
    ):
        assert_rounds_to(d[key], cell.split()[0], f"§5 {arm} {label}")


def test_section_5_the_two_stage_fit_is_the_better_calibrated_one():
    """The section's actual argument, which no single cell states.

    "The gain is calibration": the homoscedastic fit misses in opposite
    directions on the two halves while the two-stage fit does not, and it wins
    on NLL. Checking the ordering means a rerun that reshuffled the numbers
    could not leave the prose standing.
    """
    d = log("heteroscedastic")
    homo, het = d["homoscedastic"], d["heteroscedastic"]
    assert homo["cover_left"] > 0.95 > homo["cover_right"], (
        "§5 says the single-noise fit over-covers on the clean half and "
        f"under-covers on the noisy one; got {homo['cover_left']:.3f} and "
        f"{homo['cover_right']:.3f}"
    )
    for side in ("cover_left", "cover_right"):
        assert abs(het[side] - 0.95) < 0.05, (
            f"§5 calls the two-stage fit well-calibrated across the domain; "
            f"{side} is {het[side]:.3f}"
        )
    assert het["nll"] < homo["nll"]


def test_section_5_the_log_chi_squared_bias_matches_its_closed_form():
    """E[log chi^2_1] = psi(1/2) + log 2, which is -(gamma + log 2).

    The correction is a hand-typed constant in both the script and the prose,
    and it has a closed form, so nothing has to take it on trust. psi(1/2) =
    -gamma - 2 log 2, so the whole thing collapses to -(gamma + log 2) and
    needs no digamma (this package is NumPy-only).
    """
    d = log("heteroscedastic")
    exact = -(float(np.euler_gamma) + math.log(2.0))
    assert_rounds_to(exact, f"{d['log_chi2_1_bias']:.4f}",
                     "the log-chi^2_1 bias the script applies")
    printed = quoted(section("5."), r"\\log\\chi\^2_1\]=(-[\d.]+)")
    assert_rounds_to(exact, printed, "§5's quoted E[log chi^2_1]")


# -- Sec. 6: ARD (experiments/ard.py) ----------------------------------------

def test_section_6_relevance_table():
    d = log("ard")
    body = section("6.")
    _, l0, l1 = cells(body, r"learned $\ell_d$")
    assert_rounds_to(d["ard"]["l"][0], l0, "§6 relevant-axis lengthscale")
    assert_rounds_to(d["ard"]["l"][1], l1, "§6 noise-axis lengthscale")
    _, r0, r1 = cells(body, "relevance")
    assert_rounds_to(d["ard"]["relevance"][0], r0, "§6 relevant-axis relevance")
    assert_rounds_to(d["ard"]["relevance"][1], r1, "§6 noise-axis relevance")
    # The relevance column has to be the reciprocal of the one above it.
    for l, r in zip(d["ard"]["l"], d["ard"]["relevance"]):
        assert abs(1.0 / l - r) < 1e-12


def test_section_6_suppression_ratio_and_evidence_gain():
    d = log("ard")
    body = section("6.")
    assert_rounds_to(d["lengthscale_ratio"],
                     quoted(body, r"driven \*\*(\d+)× larger"),
                     "§6 lengthscale ratio")
    assert_rounds_to(d["ard"]["lml"], quoted(body, r"likelihood\n\*\*([\d.]+)\*\* vs"),
                     "§6 ARD log evidence")
    assert_rounds_to(d["isotropic"]["lml"],
                     quoted(body, r"isotropic RBF's \*\*([\d.]+)\*\*"),
                     "§6 isotropic log evidence")
    assert d["ard"]["lml"] > d["isotropic"]["lml"], (
        "ARD nests the isotropic kernel, so it cannot score lower"
    )


# -- Sec. 7: a 2D spatial GP (experiments/spatial2d.py) ----------------------

def test_section_7_accuracy_calibration_and_uncertainty_range():
    d = log("spatial2d")
    body = section("7.")
    assert_rounds_to(d["held_out_rmse"],
                     cells(body, "held-out RMSE")[1], "§7 held-out RMSE")
    assert_rounds_to(d["latent_coverage_95"],
                     cells(body, "latent 95% coverage")[1], "§7 latent coverage")
    lo, hi = cells(body, "posterior sd, near data")[1].split("→")
    assert_rounds_to(d["posterior_sd_min"], lo, "§7 posterior sd near data")
    assert_rounds_to(d["posterior_sd_max"], hi, "§7 posterior sd in the gaps")


def test_section_7_the_uncertainty_swells_but_stays_under_the_prior():
    """§7's point (panel c), as an ordering rather than a picture.

    "the standard deviation collapses at each observation and swells in the
    gaps and past the domain edges, relaxing toward the prior" -- so the gap
    value has to sit strictly between the near-data value and the prior sd. It
    does, and not by a little: 0.019 to 0.113 against a prior 0.204, so the
    field is still far from having given up where it is guessing most.
    """
    d = log("spatial2d")
    assert d["posterior_sd_min"] < d["posterior_sd_max"] < d["prior_sd"], (
        f"sd range {d['posterior_sd_min']:.3f}..{d['posterior_sd_max']:.3f} "
        f"against a prior sd of {d['prior_sd']:.3f}"
    )
    # The field's amplitude is ~1 and the RMSE is a few percent of it, which is
    # the "interpolates" half of the section's claim.
    assert d["held_out_rmse"] < 0.05


# -- Sec. 8: the Gibbs kernel (experiments/gibbs_kernel.py) -----------------

def test_section_8_learned_lengthscales_and_evidence():
    d = log("gibbs_kernel")
    body = section("8.")
    assert int(quoted(body, r"ML-II recovers exactly that \(\$n=(\d+)\$\)")) == d["n"]
    _, l_cell, lml_cell = cells(body, "RBF (stationary)")
    assert_rounds_to(d["rbf"]["l"], quoted(l_cell, r"\\ell = ([\d.]+)"),
                     "§8 RBF lengthscale")
    assert_rounds_to(d["rbf"]["lml"], lml_cell, "§8 RBF log evidence")

    _, range_cell, lml_cell = cells(body, "**Gibbs** |")
    g = d["gibbs"]
    lo, hi = re.findall(r"=([\d.]+)", range_cell)
    assert_rounds_to(g["l_at_lo"], lo, "§8 Gibbs l(0)")
    assert_rounds_to(g["l_at_hi"], hi, "§8 Gibbs l(2.2)")
    assert_rounds_to(g["l_at_lo"] / g["l_at_hi"],
                     quoted(range_cell, r"\(([\d.]+)× range\)"), "§8 lengthscale range")
    # l(x) = exp(a + b x), so the endpoints have to come from the logged tilt.
    assert abs(math.exp(g["a"]) - g["l_at_lo"]) < 1e-12
    assert abs(math.exp(g["a"] + 2.2 * g["b"]) - g["l_at_hi"]) < 1e-12
    assert g["b"] < 0, "the chirp gets rougher to the right, so l(x) must fall"

    gibbs_lml, gap = re.findall(r"([+\d.]+)", lml_cell)[:2]
    assert_rounds_to(g["lml"], gibbs_lml, "§8 Gibbs log evidence")
    assert_rounds_to(g["lml"] - d["rbf"]["lml"], gap, "§8 evidence gap")

    # The prose's "(0.18, versus the 1.55 the smooth region wants)".
    rbf_l, smooth_l = re.findall(
        r"\(\$([\d.]+)\$, versus the \$([\d.]+)\$ the smooth", body)[0]
    assert_rounds_to(d["rbf"]["l"], rbf_l, "§8 RBF lengthscale in the prose")
    assert_rounds_to(g["l_at_lo"], smooth_l, "§8 smooth-region lengthscale in the prose")


def test_section_8_density_sweep_table():
    sweep = log("gibbs_kernel")["density_sweep"]
    body = section("8.")
    assert [r["n"] for r in sweep] == [40, 60, 90, 140]
    for r in sweep:
        _, gap, rmse, nll = cells(body, f"{r['n']} |")
        assert_rounds_to(r["lml_gibbs"] - r["lml_rbf"], gap.split()[0],
                         f"§8 n={r['n']} evidence gap")
        g_rmse, b_rmse = (c.strip() for c in rmse.split("/"))
        assert_rounds_to(r["rmse_smooth_gibbs"], g_rmse, f"§8 n={r['n']} Gibbs RMSE")
        assert_rounds_to(r["rmse_smooth_rbf"], b_rmse, f"§8 n={r['n']} RBF RMSE")
        g_nll, b_nll = (c.strip() for c in nll.split("/"))
        assert_rounds_to(r["nll_gibbs"], g_nll, f"§8 n={r['n']} Gibbs NLL")
        assert_rounds_to(r["nll_rbf"], b_nll, f"§8 n={r['n']} RBF NLL")
    # The n=60 row of the sweep is the same fit as the headline table.
    row60 = next(r for r in sweep if r["n"] == 60)
    assert row60["lml_gibbs"] == log("gibbs_kernel")["gibbs"]["lml"]


def test_section_8_the_gap_is_paid_in_data():
    """§8's argument, which the table implies and nothing checked.

    The evidence prefers Gibbs at every n, and the smooth-region advantage is
    largest where the data is thinnest. This file's first run caught the
    sentence under the table claiming the two models were "indistinguishable to
    three decimals" at n=140, where its own table reads 0.110 / 0.112 and
    -0.868 / -0.848. It never was true at any n in the sweep, so the sentence
    now quotes the RMSE gaps, and those are checked here.
    """
    sweep = log("gibbs_kernel")["density_sweep"]
    body = section("8.")
    assert all(r["lml_gibbs"] > r["lml_rbf"] for r in sweep)
    adv = {r["n"]: r["rmse_smooth_rbf"] - r["rmse_smooth_gibbs"] for r in sweep}
    assert max(adv, key=adv.get) == min(adv), (
        f"§8 says the predictive gap opens as data thins; it is largest at "
        f"n={max(adv, key=adv.get)}"
    )
    g90, g140, g40 = re.findall(
        r"At \$n=90\$ and \$n=140\$\s+the two smooth-region RMSEs are ([\d.]+) "
        r"and ([\d.]+) apart; at \$n=40\$ they are ([\d.]+)", body)[0]
    for n, printed in ((90, g90), (140, g140), (40, g40)):
        assert_rounds_to(adv[n], printed, f"§8 smooth-region RMSE gap at n={n}")
    row40 = next(r for r in sweep if r["n"] == 40)
    cut = 1 - row40["rmse_smooth_gibbs"] / row40["rmse_smooth_rbf"]
    assert_rounds_to(100 * cut, quoted(body, r"cuts\s+smooth-region RMSE by ~(\d+)%"),
                     "§8 RMSE cut at n=40")


# -- Sec. 9: ML-II is non-convex (experiments/multistart.py) -----------------

@pytest.mark.parametrize("row,arm", [
    ("signal-mode single start", "signal_single_start"),
    ("noise-mode single start", "noise_single_start"),
    ("**multi-start**", "multistart"),
])
def test_section_9_table_rows(row, arm):
    res = log("multistart")[arm]
    body = section("9.")
    _, l, noise, lml = cells(body, row)
    assert_rounds_to(res["l"], l, f"Sec. 9 {arm} lengthscale")
    assert_rounds_to(res["lml"], lml, f"Sec. 9 {arm} log evidence")
    if noise == "~0":
        assert res["noise_var"] < 1e-3, (
            f"Sec. 9 writes {arm}'s noise as ~0; the fit left it at "
            f"{res['noise_var']:.2e}"
        )
    else:
        assert_rounds_to(res["noise_var"], noise, f"Sec. 9 {arm} noise")


def test_section_9_multistart_gain_over_the_bad_basin():
    d = log("multistart")
    gain = d["multistart"]["lml"] - d["noise_single_start"]["lml"]
    printed = quoted(section("9."), r"\(\+([\d.]+) nats\)")
    assert_rounds_to(gain, printed, "Sec. 9 multi-start gain")
    assert gain > 0, (
        f"§9 quotes a gain of +{printed} nats; multi-start came back worse "
        f"than its own init ({gain:+.3f})"
    )


def test_section_9_the_two_short_lengthscale_rows_are_as_close_as_claimed():
    """The gap the prose quotes between the two fits that share a basin.

    This sentence exists because this file's first run caught the multi-start
    row quoting 1.29, which is the signal-mode fit's lengthscale double-rounded
    off the script's own three-decimal print: the two rows are separate Adam
    runs and land at 1.287 and 1.285. Stating the gap invites the same drift,
    so the gap is checked too.
    """
    d = log("multistart")
    body = section("9.")
    dl = abs(d["multistart"]["l"] - d["signal_single_start"]["l"])
    dlml = abs(d["multistart"]["lml"] - d["signal_single_start"]["lml"])
    assert_rounds_to(dl, quoted(body, r"land ([\d.]+) apart in \$\\ell\$"),
                     "Sec. 9 lengthscale gap between the two short-l fits")
    assert_rounds_to(dlml, quoted(body, r"([\d.]+)\s*\napart in log evidence"),
                     "Sec. 9 log-evidence gap between the two short-l fits")


def test_section_9_dataset_size_and_restart_count():
    body = section("9.")
    d = log("multistart")
    assert int(quoted(body, r"On a (\d+)-point set")) == 12
    # keep_init=True means restart 0 is the model's own parameters, so the
    # wrapper "can never return worse evidence than the single fit it wraps".
    assert d["multistart"]["lml"] >= d["noise_single_start"]["lml"]
    assert d["multistart"]["n_finite"] == d["multistart"]["n_restarts"]


def test_section_9_profile_is_one_sharp_peak_then_a_flat_plateau():
    """The right panel's shape, which §9 describes in words and nothing checked.

    Two numbers carry it: the peak sits at the signal-mode lengthscale (the
    profile grid steps by 1.084x, so half a step is 4.2% and the tolerance has
    to clear that), and the long-lengthscale end is flat -- 0.003 nats across
    l >= 20. The plateau height is also a cross-check on the table: the peak
    stands 8.29 nats above it, which is the same +8.3 the prose quotes for
    escaping the basin.
    """
    d = log("multistart")
    l = np.asarray(d["profile"]["l"])
    prof = np.asarray(d["profile"]["lml"])
    peak_l = l[prof.argmax()]
    signal_l = d["signal_single_start"]["l"]
    assert abs(peak_l / signal_l - 1.0) < 0.10, (
        f"the evidence profile peaks at l={peak_l:.3f}, not at the signal-mode "
        f"fit's l={signal_l:.3f}"
    )
    plateau = prof[l >= 20.0]
    assert plateau.size >= 10
    assert plateau.max() - plateau.min() < 0.05, (
        f"the long-lengthscale plateau spans {plateau.max() - plateau.min():.3f} "
        "nats, so §9's 'long flat plateau' no longer describes it"
    )
    gain = d["multistart"]["lml"] - d["noise_single_start"]["lml"]
    assert abs((prof.max() - plateau.max()) - gain) < 0.1, (
        "the profile's peak-to-plateau height no longer agrees with the "
        "table's multi-start gain, so one of the two is measuring something else"
    )


# -- Sec. 10: random Fourier features (experiments/rff.py) -------------------

def test_section_10_kernel_error_table():
    """The first column of the rate table: kernel approximation error."""
    d = log("rff")
    body = section("10.")
    for D in ("16", "64", "256", "1024", "4096"):
        row = cells(body, f"{D} |")
        assert_rounds_to(d["kernel_error"][D]["median"], row[1],
                         f"§10 kernel error at D={D}")


def test_section_10_posterior_error_table():
    """The posterior columns: mean error (median), worst draw, and sd error."""
    d = log("rff")
    body = section("10.")
    for D in ("16", "64", "256", "1024", "4096"):
        row = cells(body, f"{D} |")
        pe = d["posterior_error"][D]
        # mean error median — strip &nbsp; and italic markup
        mean_raw = re.sub(r"&nbsp;|\*", "", row[2]).split("(")[0].strip()
        assert_rounds_to(pe["mean_err_median"], mean_raw,
                         f"§10 posterior mean error at D={D}")
        # worst draw (only D=16 shows it)
        if "worst draw" in row[2]:
            worst = re.search(r"worst draw ([\d.]+)", row[2]).group(1)
            assert_rounds_to(pe["mean_err_worst"], worst,
                             f"§10 posterior mean error worst draw at D={D}")
        # sd error median
        assert_rounds_to(pe["sd_err_median"], row[3],
                         f"§10 posterior sd error at D={D}")


def test_section_10_kernel_error_ratio():
    d = log("rff")
    body = section("10.")
    ratio = d["kernel_error_ratio_16_to_4096"]
    printed = quoted(body, r"256× the features bought \*\*([\d.]+)×")
    assert_rounds_to(ratio, printed, "§10 kernel error ratio 16->4096")


def test_section_10_signal_amplitude():
    d = log("rff")
    body = section("10.")
    printed = quoted(body, r"posterior mean spanning \$\\pm ([\d.]+)\$")
    assert_rounds_to(d["signal_amplitude"], printed,
                     "§10 signal amplitude in the rate-table caption")


def test_section_10_timing_mean_errors():
    """The mean-error column of the timing table is deterministic and logged.

    Wall-clock seconds and speedup are machine-dependent and not tested.
    The README writes the exponent without a leading zero (2.2e-2, not
    2.2e-02), so the comparison normalises the format.
    """
    d = log("rff")
    body = section("10.")
    for n in ("500", "1000", "2000", "4000", "8000"):
        row = cells(body, f"{n} |")
        cell = row[4].strip()
        mantissa, _, exp = cell.partition("e")
        sig = len(mantissa.split(".")[1]) if "." in mantissa else 0
        formatted = f"{d['timing_mean_error'][n]:.{sig}e}"
        fmt_m, _, fmt_e = formatted.partition("e")
        fmt_norm = f"{fmt_m}e{int(fmt_e)}"
        cell_norm = f"{mantissa}e{int(exp)}"
        assert fmt_norm == cell_norm, (
            f"§10 timing mean error at n={n}: README prints {cell}, log holds "
            f"{d['timing_mean_error'][n]!r}, which formats to {fmt_norm}"
        )


def test_section_10_starvation_table():
    """The variance-starvation table and its surrounding prose."""
    d = log("rff")
    st = d["starvation"]
    body = section("10.")

    # exact sd
    printed_sd = quoted(body, r"gap centre is \*\*([\d.]+)\*\*")
    assert_rounds_to(st["exact_sd"], printed_sd, "§10 exact sd at gap centre")
    exact_row = cells(body, "exact GP |")
    assert_rounds_to(st["exact_sd"], exact_row[1], "§10 exact sd in table")

    # D=2048 row — cells contain &nbsp; between the number and the parenthetical
    d2k = st["D2048"]
    row_2k = cells(body, "$D = 2048$")
    sd_2k_raw = row_2k[1].replace("&nbsp;", " ").split("(")[0].strip()
    assert_rounds_to(d2k["sd_median"], sd_2k_raw, "§10 D=2048 sd median")
    ratio_2k = re.search(r"\(([\d.]+)×\)", row_2k[1]).group(1)
    assert_rounds_to(d2k["sd_median"] / st["exact_sd"], ratio_2k,
                     "§10 D=2048 sd ratio to exact")
    range_2k = re.findall(r"\[([\d.]+),\s*([\d.]+)\]", row_2k[2])
    assert len(range_2k) == 1
    assert_rounds_to(d2k["sd_min"], range_2k[0][0], "§10 D=2048 sd min")
    assert_rounds_to(d2k["sd_max"], range_2k[0][1], "§10 D=2048 sd max")
    assert_rounds_to(d2k["mean_err_gap"], row_2k[3], "§10 D=2048 mean err in gap")
    assert_rounds_to(d2k["mean_err_data"], row_2k[4], "§10 D=2048 mean err on data")

    # D=64 row — same &nbsp; treatment
    d64 = st["D64"]
    row_64 = cells(body, "$D = 64$")
    sd_64_raw = row_64[1].replace("&nbsp;", " ").split("(")[0].strip()
    assert_rounds_to(d64["sd_median"], sd_64_raw, "§10 D=64 sd median")
    ratio_64 = re.search(r"\(([\d.]+)×\)", row_64[1]).group(1)
    assert_rounds_to(d64["sd_median"] / st["exact_sd"], ratio_64,
                     "§10 D=64 sd ratio to exact")
    range_64 = re.findall(r"\[([\d.]+),\s*([\d.]+)\]", row_64[2])
    assert len(range_64) == 1
    assert_rounds_to(d64["sd_min"], range_64[0][0], "§10 D=64 sd min")
    assert_rounds_to(d64["sd_max"], range_64[0][1], "§10 D=64 sd max")
    assert_rounds_to(d64["mean_err_gap"], row_64[3], "§10 D=64 mean err in gap")
    assert_rounds_to(d64["mean_err_data"], row_64[4], "§10 D=64 mean err on data")


def test_section_10_starvation_prose_numbers():
    """Prose claims in the starvation subsection."""
    d = log("rff")
    st = d["starvation"]
    body = section("10.")

    # "even the best draw is 34% below exact"
    best_ratio = st["D64"]["sd_max"] / st["exact_sd"]
    # The text wraps across lines, so match with re.DOTALL-like flexibility.
    pct_below = quoted(body, r"best draw is (\d+)%\s+below exact")
    assert round((1 - best_ratio) * 100) == int(pct_below), (
        f"§10 says best draw is {pct_below}% below exact; measured "
        f"{(1 - best_ratio) * 100:.0f}%"
    )

    # "the spread runs 7x"
    spread = st["D64"]["sd_max"] / st["D64"]["sd_min"]
    printed_spread = quoted(body, r"the spread runs\s+(\d+)×")
    assert round(spread) == int(printed_spread), (
        f"§10 says the D=64 spread is {printed_spread}x; measured {spread:.1f}x"
    )

    # "off by 0.527 in the gap and 0.009 where the data are, a 56x difference"
    gap_err = quoted(body, r"off by \*\*([\d.]+)\*\* in\s+the gap")
    assert_rounds_to(st["D64"]["mean_err_gap"], gap_err,
                     "§10 D=64 gap error in prose")
    data_err = re.findall(r"([\d.]+) where the data are, a (\d+)×", body)
    assert len(data_err) == 1, f"data-error prose matched {len(data_err)} times"
    assert_rounds_to(st["D64"]["mean_err_data"], data_err[0][0],
                     "§10 D=64 data error in prose")
    ratio = st["D64"]["mean_err_gap"] / st["D64"]["mean_err_data"]
    assert round(ratio) == int(data_err[0][1]), (
        f"§10 says gap/data ratio is {data_err[0][1]}x; measured {ratio:.0f}x"
    )

    # "Even D=2048 ... is 22x worse in the gap than on the data"
    ratio_2k = st["D2048"]["mean_err_gap"] / st["D2048"]["mean_err_data"]
    printed_2k = quoted(body, r"(\d+)× worse in the gap than on the data")
    assert round(ratio_2k) == int(printed_2k), (
        f"§10 says D=2048 gap/data ratio is {printed_2k}x; measured {ratio_2k:.0f}x"
    )

    # "posterior mean spanning +-1.5" in the starvation subsection
    sig_starv = quoted(body, r"against a posterior\s+mean spanning \$\\pm([\d.]+)\$")
    assert_rounds_to(d["signal_amplitude"], sig_starv,
                     "§10 signal amplitude in starvation prose")


# -- Sec. 11: sparse GPs (experiments/sparse.py) -----------------------------

def test_section_11_exact_gp_parameters():
    """The exact GP's fitted hyperparameters and log evidence."""
    d = log("sparse")
    body = section("11.")
    ell = quoted(body, r"ML-II fit \(\$\\ell = ([\d.]+)\$")
    assert_rounds_to(d["exact"]["lengthscale"], ell, "§11 exact GP lengthscale")
    sig2 = quoted(body, r"\$\\sigma\^2 = ([\d.]+)\$")
    assert_rounds_to(d["exact"]["noise_var"], sig2, "§11 exact GP noise variance")
    lml = quoted(body, r"log evidence \*\*([\d.]+)\*\*")
    assert_rounds_to(d["exact"]["log_marginal_likelihood"], lml,
                     "§11 exact GP log marginal likelihood")


def test_section_11_sweep_table():
    """All cells of the M-sweep table: gap, mean err, sd err for both arms."""
    d = log("sparse")
    body = section("11.")
    M_list = d["sweep"]["M"]
    opt = d["sweep"]["optimized"]
    frz = d["sweep"]["frozen"]
    table_Ms = [2, 4, 8, 12, 16, 24, 32, 64]
    for M in table_Ms:
        i = M_list.index(M)
        row = cells(body, f"{M} ")
        opt_gap_cell = row[1]
        opt_mean_cell = row[2]
        opt_sd_cell = row[3]
        frz_gap_cell = row[4]
        frz_mean_cell = row[5]
        frz_sd_cell = row[6]
        if "e" in opt_gap_cell.lower() or "−" in opt_gap_cell:
            norm = opt_gap_cell.replace("−", "-")
            assert_formats_to(opt["gap"][i], norm,
                              f"§11 opt gap at M={M}")
        else:
            assert_rounds_to(opt["gap"][i], opt_gap_cell,
                             f"§11 opt gap at M={M}")
        assert_rounds_to(opt["mean_err"][i], opt_mean_cell,
                         f"§11 opt mean err at M={M}")
        assert_rounds_to(opt["sd_err"][i], opt_sd_cell,
                         f"§11 opt sd err at M={M}")
        if "e" in frz_gap_cell.lower() or "−" in frz_gap_cell:
            norm = frz_gap_cell.replace("−", "-")
            assert_formats_to(frz["gap"][i], norm,
                              f"§11 frozen gap at M={M}")
        else:
            assert_rounds_to(frz["gap"][i], frz_gap_cell,
                             f"§11 frozen gap at M={M}")
        assert_rounds_to(frz["mean_err"][i], frz_mean_cell,
                         f"§11 frozen mean err at M={M}")
        assert_rounds_to(frz["sd_err"][i], frz_sd_cell,
                         f"§11 frozen sd err at M={M}")


def test_section_11_gap_is_monotone_and_nonnegative():
    """The bound's defining property: gap >= 0, decreasing in M."""
    d = log("sparse")
    for arm in ("optimized", "frozen"):
        gaps = d["sweep"][arm]["gap"]
        assert all(g >= -1e-6 for g in gaps), (
            f"§11 {arm} arm has a negative gap: the bound is above the ceiling"
        )
        assert all(a >= b for a, b in zip(gaps, gaps[1:])), (
            f"§11 {arm} arm's gap is not monotone in M"
        )


def test_section_11_matching_M():
    """'optimized M=8 matches frozen M=12, optimized 32 matches frozen 48'."""
    d = log("sparse")
    body = section("11.")
    M_list = d["sweep"]["M"]
    opt_gaps = d["sweep"]["optimized"]["gap"]
    frz_gaps = d["sweep"]["frozen"]["gap"]
    pairs = re.findall(r"optimized\s+(?:\$M\{=\})?(\d+)\$?\s+matches\s+frozen\s+(?:\$M\{=\})?(\d+)\$?", body)
    assert len(pairs) >= 2, f"§11 should name at least 2 matching pairs; found {pairs}"
    for opt_s, frz_s in pairs:
        opt_M, frz_M = int(opt_s), int(frz_s)
        opt_gap = opt_gaps[M_list.index(opt_M)]
        frz_gap = frz_gaps[M_list.index(frz_M)]
        assert frz_gap <= opt_gap, (
            f"§11 says optimized M={opt_M} matches frozen M={frz_M}, "
            f"but frozen gap {frz_gap:.4f} > optimized gap {opt_gap:.4f}"
        )
        frz_i = M_list.index(frz_M)
        for j in range(frz_i):
            assert frz_gaps[j] > opt_gap, (
                f"§11 says optimized M={opt_M} matches frozen M={frz_M}, "
                f"but frozen M={M_list[j]} already has gap {frz_gaps[j]:.4f} "
                f"<= optimized gap {opt_gap:.4f}"
            )


def test_section_11_spacing_in_lengthscales():
    """The spacing claim: 'gap collapses at about 0.7 fitted lengthscales'."""
    d = log("sparse")
    body = section("11.")
    M_list = d["sweep"]["M"]
    Zs = d["inducing_points"]
    opt_lengths = d["sweep"]["optimized"]["lengthscale"]
    for M_str, ell_str, gap_str in re.findall(
        r"\$M\{=\}(\d+)\$:\s+([\d.]+)\s+\$\\ell\$,\s+gap\s+([\d.]+)\s+nats",
        body
    ):
        M = int(M_str)
        i = M_list.index(M)
        Z = np.array(Zs[str(M)])
        l = opt_lengths[i]
        slow_Z = Z[Z <= 0]
        slow_spacing = float(np.median(np.diff(slow_Z))) / l
        assert_rounds_to(slow_spacing, ell_str,
                         f"§11 slow spacing in ℓ at M={M}")
        assert_rounds_to(d["sweep"]["optimized"]["gap"][i], gap_str,
                         f"§11 gap at M={M} in spacing prose")


def test_section_11_sd_error_nonmonotonicity():
    """The specific M values where sd error rises are named in the prose."""
    d = log("sparse")
    body = section("11.")
    opt_sd = d["sweep"]["optimized"]["sd_err"]
    frz_sd = d["sweep"]["frozen"]["sd_err"]
    M_list = d["sweep"]["M"]
    opt_rises = [M_list[i + 1] for i in range(len(opt_sd) - 1) if opt_sd[i + 1] > opt_sd[i]]
    frz_rises = [M_list[i + 1] for i in range(len(frz_sd) - 1) if frz_sd[i + 1] > frz_sd[i]]
    m = re.search(r"sd error rises at \$M = (\d+)\$ and again at (\d+)", body)
    assert m, "§11 should name two M values where opt sd rises"
    assert int(m.group(1)) in opt_rises, f"§11 says opt sd rises at M={m.group(1)} but data disagrees"
    assert int(m.group(2)) in opt_rises, f"§11 says opt sd rises at M={m.group(2)} but data disagrees"
    m2 = re.search(r"frozen arm.s at (\d+) and\s+(\d+)", body)
    assert m2, "§11 should name two M values where frozen sd rises"
    assert int(m2.group(1)) in frz_rises, f"§11 says frozen sd rises at M={m2.group(1)} but data disagrees"
    assert int(m2.group(2)) in frz_rises, f"§11 says frozen sd rises at M={m2.group(2)} but data disagrees"


def test_section_11_inducing_point_clustering():
    """M=64 optimum: 36/64 (56%) in the fast half, 1.15x denser."""
    d = log("sparse")
    body = section("11.")
    Z = np.array(d["inducing_points"]["64"])
    n_fast = int((Z > 0).sum())
    printed_n = int(quoted(body, r"puts (\d+) of 64"))
    assert n_fast == printed_n, (
        f"§11 says {printed_n} of 64 in fast half; measured {n_fast}"
    )
    printed_pct = quoted(body, r"puts \d+ of 64 \((\d+)%\)")
    assert round(100 * n_fast / 64) == int(printed_pct), (
        f"§11 says {printed_pct}%; measured {100 * n_fast / 64:.0f}%"
    )
    slow_sp = float(np.median(np.diff(Z[Z <= 0])))
    fast_sp = float(np.median(np.diff(Z[Z > 0])))
    ratio = slow_sp / fast_sp
    printed_ratio = quoted(body, r"spacing ([\d.]+)× denser")
    assert_rounds_to(ratio, printed_ratio,
                     "§11 spacing ratio at M=64")


def test_section_11_control_density():
    """The 3:1 control: exact GP ℓ, count and spacing at M=64."""
    d = log("sparse")
    body = section("11.")
    ell_ctrl = quoted(body, r"still fits \$\\ell = ([\d.]+)\$")
    assert_rounds_to(d["control"]["exact_lengthscale"], ell_ctrl,
                     "§11 control exact GP lengthscale")
    Z_ctrl = np.array(d["control"]["inducing_points"]["64"])
    n_fast_ctrl = int((Z_ctrl > 0).sum())
    printed_ctrl_n = int(quoted(body, r"(\d+) of 64 \(\d+%\) in the fast half,\s+[\d.]+× denser"))
    assert n_fast_ctrl == printed_ctrl_n, (
        f"§11 control says {printed_ctrl_n} of 64 fast; measured {n_fast_ctrl}"
    )
    printed_ctrl_pct = quoted(body, r"\d+ of 64 \((\d+)%\) in the fast half,\s+[\d.]+× denser")
    assert round(100 * n_fast_ctrl / 64) == int(printed_ctrl_pct)
    slow_sp = float(np.median(np.diff(Z_ctrl[Z_ctrl <= 0])))
    fast_sp = float(np.median(np.diff(Z_ctrl[Z_ctrl > 0])))
    ratio_ctrl = fast_sp / slow_sp
    printed_ctrl_ratio = quoted(body, r"([\d.]+)× denser where the \*\*data\*\*")
    assert_rounds_to(ratio_ctrl, printed_ctrl_ratio,
                     "§11 control spacing ratio at M=64")


def test_section_11_force_split():
    """The dF/dZ split: trace 92.37, DTC 419.95, agreement 1.1e-06."""
    d = log("sparse")
    body = section("11.")
    fs = d["force_split_init"]
    printed_trace = quoted(body, r"trace contributes ([\d.]+)")
    assert_rounds_to(fs["trace_norm"], printed_trace,
                     "§11 trace penalty norm")
    printed_dtc = quoted(body, r"DTC term.s ([\d.]+)")
    assert_rounds_to(fs["dtc_norm"], printed_dtc,
                     "§11 DTC data-fit norm")
    printed_agree = quoted(body, r"to ([\d.]+e[−\-]\d+)")
    norm_agree = printed_agree.replace("−", "-")
    assert_formats_to(fs["agreement"], norm_agree,
                      "§11 force split agreement")


# -- the instrument's own coverage -------------------------------------------

def test_every_committed_log_is_checked_by_this_file():
    on_disk = {p.stem for p in LOGS.glob("*.json")} if LOGS.exists() else set()
    unchecked = on_disk - set(INSTRUMENTED)
    assert not unchecked, (
        f"logs/{sorted(unchecked)} are committed but no test here reads them, "
        "so writing them bought nothing"
    )


def test_every_result_section_is_either_instrumented_or_listed():
    headings = re.findall(r"^### (\d+)\. ", README, flags=re.M)
    numbered = [n + "." for n in headings]
    accounted = set(INSTRUMENTED.values()) | set(NOT_YET)
    assert set(numbered) == accounted, (
        "the README's result sections and this file's own bookkeeping "
        f"disagree: {sorted(set(numbered) ^ accounted)}. A new section needs "
        "either a log and a check above, or a line in NOT_YET."
    )
    assert not set(INSTRUMENTED.values()) & set(NOT_YET)
    assert len(numbered) == len(set(numbered)), "a section number is repeated"


def test_the_instrumented_sections_name_the_scripts_that_write_their_logs():
    for stem, number in INSTRUMENTED.items():
        assert f"experiments/{stem}.py" in section(number), (
            f"§{number} no longer names experiments/{stem}.py, but "
            f"logs/{stem}.json is checked against it"
        )
