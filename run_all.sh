#!/usr/bin/env bash
# Reproduce every Phase 1 artefact from scratch: data -> EDA -> models -> figures -> docs -> notebooks.
# Takes roughly 90 minutes on 2 CPU cores (most of it the two Random Forest runs); pass --fast to skip them.
set -euo pipefail
cd "$(dirname "$0")"

MODELS="HA Persistence Ridge-local XGB-local XGB-spatial RF-local RF-spatial"
if [[ "${1:-}" == "--fast" ]]; then MODELS="HA Persistence Ridge-local XGB-local XGB-spatial"; fi

echo "== 1/6 data";        python scripts/download_data.py
echo "== 2/6 EDA";         python -W ignore scripts/run_eda.py
echo "== 3/6 models";      python -W ignore scripts/run_experiments.py --models $MODELS
echo "== 4/6 figures";     python -W ignore scripts/make_result_figures.py
echo "== 5/6 docs"; python scripts/fill_docs.py
echo "== 6/6 notebooks";   python scripts/build_notebooks.py && jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks/*.ipynb
python -m pytest tests -q
echo "Done. Report: report/report_simple.html (print to PDF from Chrome to refresh report_simple.pdf)"
