#!/usr/bin/env bash
#
# Regenerate every figure in figures/ and every number in the README, end to
# end, from a clean checkout.
#
#     ./reproduce.sh              # full suite, ~5 min
#     PYTHON=/path/to/python ./reproduce.sh
#
# There is nothing to download and no cached state: the library is NumPy, the
# only data file (data/co2_mm_mlo.txt, the Mauna Loa monthly record) is
# committed, and every experiment is seeded, so a rerun recomputes the figures
# rather than replaying stored ones. (The two torch repos in the series ship
# trained checkpoints instead, because retraining them takes hours. Here the
# whole suite is minutes.)
#
# Determinism: every experiment draws from np.random.default_rng(<fixed seed>)
# and NumPy guarantees its bit generators are stable across versions, so the
# datasets, fits, and tables reproduce exactly on the pinned environment in
# requirements.txt. The one machine-dependent column is the NumPy-vs-scikit-learn
# wall-clock ratio in the parity benchmark; its accuracy column (agreement to
# ~1e-10) is the portable claim.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-}"
if [ -z "${PY}" ]; then
    if [ -x .venv/bin/python ]; then PY="$PWD/.venv/bin/python"; else PY="python3"; fi
fi

echo "=================================================================="
echo "gp-from-scratch: full reproduction"
echo "python:     $("$PY" -V 2>&1)  ($PY)"
"$PY" - <<'EOF'
import importlib
for name in ("numpy", "matplotlib", "sklearn"):
    try:
        m = importlib.import_module(name)
        print(f"{name+':':11s} {m.__version__}")
    except ImportError:
        print(f"{name+':':11s} MISSING")
EOF
echo "=================================================================="

started=$SECONDS

step() {  # step <label> <script> [args...]
    local label="$1"; shift
    echo
    echo "------------------------------------------------------------------"
    echo ">>> $label"
    echo "------------------------------------------------------------------"
    local t0=$SECONDS
    "$PY" "$@"
    echo "    [${label}: $((SECONDS - t0))s]"
}

# The test suite first: the tests are what license the README's claims (the
# sklearn oracle, the finite-difference gradient checks, the LOO identity
# against brute-force refits). If they fail, the figures below are not worth
# regenerating.
step "test suite (unit tests + doctests)" -m pytest -q
step "static type check (mypy, gp/)"      -m mypy

step "1. kernel prior gallery"                          experiments/prior_samples.py
step "2. exact-GP validation (calibration, recovery)"   experiments/validate.py
step "3. Mauna Loa CO2 (ML-II, 9 free params, twice)"   experiments/co2.py
step "4. NNGP / NTK correspondence"                     experiments/ntk_experiments.py
step "5. parity + speed vs scikit-learn"                experiments/sklearn_parity.py
step "6. two-stage heteroscedastic noise"               experiments/heteroscedastic.py
step "7. ARD: per-dimension relevance"                  experiments/ard.py
step "8. 2D spatial field"                              experiments/spatial2d.py
step "9. Gibbs kernel (input-dependent lengthscale)"    experiments/gibbs_kernel.py
step "10. ML-II multimodality and multi-start"          experiments/multistart.py
step "11. random Fourier features (rate, speed, gaps)"  experiments/rff.py
step "12. sparse GPs: joint ML-II over Z, convergence"  experiments/sparse.py
step "13. FITC vs VFE: the noise it hides in Lambda"    experiments/fitc.py
step "14. features vs inducing points, same gap"        experiments/rff_vs_sparse.py
step "15. cost: seconds, bytes, and the exponents"      experiments/cost_scaling.py

echo
echo "=================================================================="
echo "done in $((SECONDS - started))s. figures/:"
ls -1 figures/
echo "=================================================================="

# Check the reproduction claim instead of asserting it. Everything above was
# just regenerated from seeded code, so any figure that now differs from the
# committed copy is either an expected machine-dependent one (the four that plot
# wall-clock) or real drift between the code and what the repo ships -- which is
# how figures/co2_forecast.png and figures/ntk_linearization.png sat stale, an
# older environment's bytes, until someone rebuilt them from a clean clone.
TIMING_FIGURES="figures/sklearn_parity.png figures/rff.png figures/rff_vs_sparse.png figures/cost_scaling.png"
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    changed=$(git status --porcelain -- figures/ | awk '{print $2}')
    unexpected=""
    for f in ${changed}; do
        case " ${TIMING_FIGURES} " in
            *" ${f} "*) ;;
            *) unexpected="${unexpected} ${f}" ;;
        esac
    done
    echo
    if [ -z "${changed}" ]; then
        echo "reproduction: every committed figure came back byte-for-byte."
        echo "(Even the two wall-clock plots landed on identical bytes here.)"
    elif [ -z "${unexpected}" ]; then
        echo "reproduction: byte-for-byte except the wall-clock plots, as documented:"
        for f in ${changed}; do echo "    ${f}   (plots timing; machine-dependent)"; done
    else
        echo "reproduction: UNEXPECTED drift -- these differ from the committed copies"
        echo "and do not plot wall-clock, so the repo is shipping figures its own"
        echo "code no longer produces. Inspect, then commit the regenerated files:"
        for f in ${unexpected}; do echo "    ${f}"; done
        echo "(the wall-clock plots may also differ; that is expected: ${TIMING_FIGURES})"
    fi
    echo "=================================================================="
fi
