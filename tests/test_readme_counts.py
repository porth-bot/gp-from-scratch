"""The counts the README quotes about the repo itself, checked against it.

Every other number in this README is a measurement, and §14's discipline is
that measurements come from committed logs. Three numbers are not measurements
of a GP at all -- how many tests there are, how many figures ship, and which of
them are allowed to differ on a rerun -- and those are exactly the ones nothing
was recomputing. The test count had drifted to 340 against a real 366 by the
time a fresh-clone run noticed, which is a small error with an unpleasant
property: it is in the paragraph that tells a reader what a correct run looks
like, so the one number a newcomer can check first was the wrong one.

The figure checks below are structural rather than numerical: they pin the
README's "17 of the 21" against the PNGs actually committed, and pin the four
files it names as machine-dependent against the whitelist ``reproduce.sh``
applies when it reports drift. Those two lists disagreeing is how a figure
quietly acquires permission to change.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
REPRODUCE = (ROOT / "reproduce.sh").read_text()


def _whole_suite(config) -> bool:
    """True when this run collected the whole suite, so a count means something.

    ``pytest tests/test_gp.py`` or ``-k something`` collects a subset; asserting
    a total against that would fail for a reason that has nothing to do with the
    README. testpaths is ["gp", "tests"], which is what bare ``pytest`` uses.
    """
    if config.option.keyword or config.option.markexpr:
        return False
    return sorted(config.args) == ["gp", "tests"]


def test_the_readme_quotes_the_number_of_tests_there_are(request):
    m = re.search(r"^pytest\s+# (\d+) tests", README, re.M)
    assert m, "the README's Reproduce block no longer quotes a test count"
    claimed = int(m.group(1))
    if not _whole_suite(request.config):
        pytest.skip("a partial run's count says nothing about the README's")
    collected = request.session.testscollected
    assert claimed == collected, (
        f"README says {claimed} tests, this run collected {collected}. "
        "Adding tests without updating that line is how it drifted before."
    )


def test_the_readme_counts_the_figures_that_are_actually_committed():
    tracked = subprocess.run(
        ["git", "ls-files", "figures/*.png"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    m = re.search(r"regenerates\s+(\d+) of the (\d+) committed PNGs", README)
    assert m, "the README no longer states how many PNGs reproduce byte-for-byte"
    exact, total = int(m.group(1)), int(m.group(2))
    assert total == len(tracked), (
        f"README claims {total} committed figures, git tracks {len(tracked)}"
    )
    assert exact == total - 4, (
        "the README's exact/total split no longer matches its own four "
        "wall-clock exceptions"
    )


def test_the_figures_the_readme_excuses_are_the_ones_reproduce_sh_excuses():
    m = re.search(r'TIMING_FIGURES="([^"]+)"', REPRODUCE)
    assert m, "reproduce.sh no longer whitelists the wall-clock figures"
    whitelisted = {Path(p).name for p in m.group(1).split()}
    # The README names them in prose, as `sklearn_parity.png` and friends.
    excused = set(re.findall(r"`(\w+\.png)`", README)) & set(
        p.name for p in (ROOT / "figures").glob("*.png")
    )
    assert whitelisted <= excused, (
        f"reproduce.sh excuses {sorted(whitelisted - excused)}, which the "
        "README does not name as machine-dependent"
    )
    assert len(whitelisted) == 4
