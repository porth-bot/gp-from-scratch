#!/usr/bin/env bash
#
# Regenerate every figure in figures/ and every number in the README, end to
# end, from a clean checkout.
#
#     ./reproduce.sh              # full suite, ~4 min
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

echo
echo "=================================================================="
echo "done in $((SECONDS - started))s. figures/:"
ls -1 figures/
echo "=================================================================="
