"""The committed figures and logs must be in the form the code writes them.

``reproduce.sh`` ends by checking every regenerated file in ``figures/`` and
``logs/`` byte-for-byte against the committed copy. That check is only as
useful as its false-positive rate, and two things put bytes in those files
that have nothing to do with the experiments:

- matplotlib stamps a ``Software`` tEXt chunk with its own version into every
  PNG. On matplotlib 3.11.2 all 21 regenerated figures differed from their
  3.11.0 copies with identical pixels. ``common.savefig`` now drops the chunk,
  and the committed figures were stripped of it (pixels unchanged).
- ``logs/fitc.json`` was committed as single-line JSON, not the indented,
  key-sorted form ``common.save_results`` writes, so a rerun that changed no
  number still showed up as drift.

These tests read the files, so they cost milliseconds and need no rerun.
"""

import json
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

FIGURES = sorted((ROOT / "figures").glob("*.png"))
LOGS = sorted((ROOT / "logs").glob("*.json"))


def png_text_keys(data):
    """Keywords of every tEXt/zTXt/iTXt chunk in a PNG's bytes."""
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    keys, i = [], 8
    while i < len(data):
        (n,) = struct.unpack(">I", data[i : i + 4])
        kind = data[i + 4 : i + 8]
        if kind in (b"tEXt", b"zTXt", b"iTXt"):
            keys.append(data[i + 8 : i + 8 + n].split(b"\x00", 1)[0].decode())
        i += 12 + n
    return keys


def test_there_are_figures_and_logs_to_check():
    # An empty glob would make every parametrised test below vacuous.
    assert len(FIGURES) >= 21
    assert len(LOGS) >= 13


@pytest.mark.parametrize("path", FIGURES, ids=lambda p: p.name)
def test_committed_figure_carries_no_software_version(path):
    assert "Software" not in png_text_keys(path.read_bytes())


def test_savefig_writes_no_software_version(tmp_path, monkeypatch):
    common = pytest.importorskip("common")
    monkeypatch.setattr(common, "FIGDIR", str(tmp_path))
    fig, ax = common.plt.subplots()
    ax.plot([0, 1], [0, 1])
    common.savefig(fig, "probe.png")
    assert "Software" not in png_text_keys((tmp_path / "probe.png").read_bytes())


def test_png_parser_sees_the_chunk_matplotlib_writes_by_default(tmp_path):
    # Without this, a parser that found no chunks at all would pass the two
    # tests above for the wrong reason.
    common = pytest.importorskip("common")
    fig = common.plt.figure()
    fig.savefig(tmp_path / "default.png")
    common.plt.close(fig)
    assert "Software" in png_text_keys((tmp_path / "default.png").read_bytes())


@pytest.mark.parametrize("path", LOGS, ids=lambda p: p.name)
def test_committed_log_is_in_save_results_format(path):
    text = path.read_text()
    canonical = json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
    assert text == canonical, (
        f"{path.name} is not byte-for-byte what common.save_results writes, so "
        "reproduce.sh would report it as drift on a rerun that changed nothing"
    )
