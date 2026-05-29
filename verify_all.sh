#!/usr/bin/env bash
# verify_all.sh — run every reproducibility check against the frozen solver.
# Exits 0 only if every check passes.

set -e

# Expected frozen-solver MD5
EXPECTED_MD5="846ce6aae16623c6ca4a551f86df869c"

echo "============================================================"
echo "ValHaLA — full reproducibility check"
echo "============================================================"

# Cross-platform MD5
if command -v md5sum >/dev/null 2>&1; then
    ACTUAL_MD5=$(md5sum valhala_solver_standalone.py | awk '{print $1}')
elif command -v md5 >/dev/null 2>&1; then
    ACTUAL_MD5=$(md5 -q valhala_solver_standalone.py)
else
    echo "ERROR: neither md5sum nor md5 is available."
    exit 1
fi

if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
    echo "ERROR: solver MD5 mismatch."
    echo "  expected: $EXPECTED_MD5"
    echo "  actual:   $ACTUAL_MD5"
    exit 1
fi
echo "[OK] valhala_solver_standalone.py MD5 verified: $ACTUAL_MD5"
echo

echo "------------------------------------------------------------"
echo "(1/5) reproduce_paper.py — headline quantities"
echo "------------------------------------------------------------"
python3 reproduce_paper.py

echo
echo "------------------------------------------------------------"
echo "(2/5) verify_figures_tables.py — figure and table values"
echo "------------------------------------------------------------"
python3 verify_figures_tables.py

echo
echo "------------------------------------------------------------"
echo "(3/5) validation_battery.py — Table 3 verdicts"
echo "------------------------------------------------------------"
python3 validation_battery.py

echo
echo "------------------------------------------------------------"
echo "(4/5) run_demonstrations.py — manuscript figures"
echo "------------------------------------------------------------"
python3 run_demonstrations.py

echo
echo "------------------------------------------------------------"
echo "(5/5) make_threshold_figure.py — Figure 1"
echo "------------------------------------------------------------"
python3 make_threshold_figure.py

echo
echo "============================================================"
echo "All reproducibility checks passed."
echo "Seven PDF figures are now in the current directory:"
ls -1 *.pdf
echo "============================================================"
