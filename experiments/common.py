"""Shared plotting setup for the experiment scripts."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    }
)

FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def savefig(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.abspath(os.path.join(FIGDIR, name))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
