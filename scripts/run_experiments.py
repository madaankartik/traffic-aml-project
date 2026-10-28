#!/usr/bin/env python3
"""Train and evaluate all Phase 1 models on METR-LA.

Usage
-----
    python scripts/run_experiments.py                       # all models
    python scripts/run_experiments.py --models HA Persistence XGB-local
    python scripts/run_experiments.py --evaluate-only       # recompute metrics from saved predictions

Outputs (results/)
------------------
    pred_<model>.npz              validation + test predictions, importances, best iterations
    metrics_horizon.csv           MAE / RMSE / MAPE per model and horizon (test split)
    metrics_peak_offpeak.csv      the same, split by peak vs off-peak target periods
    metrics_by_hour.csv           MAE by hour of day of the target
    metrics_congestion.csv        congestion-detection precision / recall / F1 (speed < 35 mph)
    metrics_validation.csv        validation-split metrics (model selection evidence)
    per_sensor_mae.csv            per-sensor MAE of every model (for the congestion/error maps)
    feature_importance.csv        mean normalised feature importance per model
    best_iterations.csv           early-stopping round chosen per (sensor, horizon) for the XGBoost models
    split_summary.csv, config.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trafficml import config as C                      # noqa: E402
from trafficml import data, evaluate as E, features as F, models as M   # noqa: E402

ALL_MODELS = ["HA", "Persistence", "Ridge-local", "XGB-local", "XGB-spatial", "RF-local", "RF-spatial"]


def fit_model(name: str, fs: F.FeatureSet, cfg: C.ExperimentConfig) -> M.ModelOutput:
    if name == "HA":
        return M.historical_average(fs)
    if name == "Persistence":
        return M.persistence(fs)
    if name == "Ridge-local":
        return M.ridge_per_segment(fs, spatial=False, params=cfg.ridge)
    if name == "Ridge-spatial":
        return M.ridge_per_segment(fs, spatial=True, params=cfg.ridge)
    if name == "XGB-local":
        return M.xgb_per_segment(fs, spatial=False, params=cfg.xgb)
    if name == "XGB-spatial":
        return M.xgb_per_segment(fs, spatial=True, params=cfg.xgb)
    if name == "RF-local":
        return M.rf_per_segment(fs, spatial=False, params=cfg.rf)
    if name == "RF-spatial":
        return M.rf_per_segment(fs, spatial=True, params=cfg.rf)
    raise ValueError(name)


def evaluate_all(fs: F.FeatureSet, names: list) -> None:
    hz = fs.names["targets"]
    Y_test = fs.targets[fs.split.test]
    Y_val = fs.targets[fs.split.val]
    tpeak = E.target_peak_flags(fs.peak, fs.split.test)
    thours = np.stack([fs.index.hour.values[fs.split.test + s] for s in C.HORIZONS.values()], axis=1)

    horizon, peak, by_hour, cong, val_rows, psm_rows, imp_rows, bi_rows = [], [], [], [], [], [], [], []
    for name in names:
        try:
            out = M.load_output(name)
        except FileNotFoundError:
            print(f"  [skip] no predictions for {name}")
            continue
        horizon.append(E.horizon_metrics(Y_test, out.test_pred, hz, name).assign(fit_seconds=out.fit_seconds))
        peak.append(E.peak_offpeak_metrics(Y_test, out.test_pred, tpeak, hz, name))
        by_hour.append(E.metrics_by_hour(Y_test, out.test_pred, thours, hz, name))
        cong.append(E.congestion_detection(Y_test, out.test_pred, hz, name))
        val_rows.append(E.horizon_metrics(Y_val, out.val_pred, hz, name))
        psm = pd.DataFrame(E.per_sensor_mae(Y_test, out.test_pred), columns=hz)
        psm.insert(0, "sensor_id", fs.sensor_ids); psm.insert(0, "model", name)
        psm_rows.append(psm)
        imp = M.importance_table(out)
        if not imp.empty:
            imp_rows.append(imp.assign(model=name)[["model", "feature", "importance"]])
        if out.best_iterations is not None:
            bi = pd.DataFrame(out.best_iterations, columns=hz)
            bi.insert(0, "sensor_id", fs.sensor_ids); bi.insert(0, "model", name)
            bi_rows.append(bi)

    # one tidy file per kind of result (long format: one row per model x item)
    pd.concat(horizon).to_csv(C.RESULTS_DIR / "metrics_horizon.csv", index=False)
    pd.concat(peak).to_csv(C.RESULTS_DIR / "metrics_peak_offpeak.csv", index=False)
    pd.concat(by_hour).to_csv(C.RESULTS_DIR / "metrics_by_hour.csv", index=False)
    pd.concat(cong).to_csv(C.RESULTS_DIR / "metrics_congestion.csv", index=False)
    pd.concat(val_rows).to_csv(C.RESULTS_DIR / "metrics_validation.csv", index=False)
    pd.concat(psm_rows).to_csv(C.RESULTS_DIR / "per_sensor_mae.csv", index=False)
    if imp_rows:
        pd.concat(imp_rows).to_csv(C.RESULTS_DIR / "feature_importance.csv", index=False)
    if bi_rows:
        pd.concat(bi_rows).to_csv(C.RESULTS_DIR / "best_iterations.csv", index=False)
    print("\n=== Test metrics by horizon ===")
    print(pd.concat(horizon).pivot(index="model", columns="horizon", values="MAE").round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=ALL_MODELS)
    ap.add_argument("--evaluate-only", action="store_true")
    args = ap.parse_args()

    np.random.seed(C.RANDOM_SEED)
    cfg = C.ExperimentConfig()
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    with open(C.RESULTS_DIR / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)

    print("Loading data ...", flush=True)
    X_df, _, graph = data.load_all()
    fs = F.build_features(X_df, graph, cfg)
    fs.split.summary(fs.index).to_csv(C.RESULTS_DIR / "split_summary.csv", index=False)
    print(fs.split.summary(fs.index).to_string(index=False), flush=True)

    if not args.evaluate_only:
        for name in args.models:
            print(f"\n>>> {name}", flush=True)
            t0 = time.time()
            out = fit_model(name, fs, cfg)
            M.save_output(out)
            print(f"<<< {name} done in {time.time() - t0:.0f}s", flush=True)

    evaluate_all(fs, ALL_MODELS if args.evaluate_only else args.models)


if __name__ == "__main__":
    main()
