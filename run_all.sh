#!/usr/bin/env bash
# Reproduce every Phase 1 + Phase 2 artefact: data -> EDA -> ML models -> figures -> DL models -> DL evaluation -> notebooks.
# Phase 1 takes ~90 min on 2 CPU cores (pass --fast to skip the Random Forests); Phase 2 training adds ~6 h on CPU.
set -euo pipefail
cd "$(dirname "$0")"

MODELS="HA Persistence Ridge-local XGB-local XGB-spatial RF-local RF-spatial"
if [[ "${1:-}" == "--fast" ]]; then MODELS="HA Persistence Ridge-local XGB-local XGB-spatial"; fi

echo "== 1/9 data";        python scripts/download_data.py
echo "== 2/9 EDA";         python -W ignore scripts/run_eda.py
echo "== 3/9 ML models";      python -W ignore scripts/run_experiments.py --models $MODELS
echo "== 4/9 figures";     python -W ignore scripts/make_result_figures.py
echo "== 5/9 docs"; python scripts/fill_docs.py
echo "== 6/9 hybrid inputs"; python -W ignore scripts/xgb_oof.py          # out-of-fold XGBoost forecasts (~20 min)
echo "== 7/9 DL models";    python -W ignore scripts/train_dl.py --all --skip-done   # ~6 h on 2 CPU cores, minutes on a GPU
echo "== 8/9 DL evaluation"; python -W ignore scripts/evaluate_dl.py && python -W ignore scripts/make_arch_figure.py && python -W ignore scripts/make_dl_figures.py && python scripts/make_p2_numbers.py
echo "== 9/9 notebooks";   python scripts/build_notebooks.py && jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=1200 notebooks/*.ipynb
python -m pytest tests -q
echo "Done. Phase 1 report: report/report.pdf · Phase 2 report: cd report/phase2 && pdflatex report_phase2 && bibtex report_phase2 && pdflatex report_phase2 && pdflatex report_phase2"
