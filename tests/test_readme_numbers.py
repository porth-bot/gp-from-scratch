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

**It covers two of the sixteen sections so far**, and `NOT_YET` below names the
other fourteen explicitly rather than leaving the hole to be discovered:
`test_every_result_section_is_either_instrumented_or_listed` fails when a
section is added or renamed, so extending the README forces a decision about
its numbers instead of quietly widening the gap. Sections go in as their
experiments learn to write logs; the two here are the two whose scripts run in
about ten seconds each, which is why they came first.

The first run made it four repos for four. §1 came back clean: the coverage
cell, the three recovery medians and both replicate counts all round to what is
printed. §9 did not. Its table gave the multi-start fit a lengthscale of 1.29,
which is the *signal-mode* fit's value; the multi-start fit is at 1.2845, so it
rounds to 1.28. The two are separate Adam runs from different initializations
that land in the same basin, and the table presented them as landing on the
same parameters. The mechanism is a double rounding rather than a transcription
slip: `multistart.py` prints the lengthscale to three decimals, and 1.285
rounded up by hand becomes 1.29 while the number it came from does not. So the
fix is not just the one cell -- all three lengthscales now carry the precision
the script prints, which is the only version of that table where the rounding
cannot be done twice.

Pure stdlib plus numpy (which the whole suite already needs), so this runs
wherever the rest of the tests do. No matplotlib, no experiment imports.
"""

from __future__ import annotations

import json
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
    "multistart": "9.",
}

# The sections whose experiments do not write a log yet. Listed, not silent:
# the test below pins this against the README's own headings.
NOT_YET = [
    "2.", "3.", "4.", "5.", "6.", "7.", "8.",
    "10.", "11.", "12.", "13.", "14.", "15.", "16.",
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
    out = [c.strip().replace("**", "").replace("−", "-")
           for c in rows[0].strip().strip("|").split("|")]
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
