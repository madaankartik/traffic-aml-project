"""Score the Phase 2 models with exactly the Phase 1 evaluation code and compare.

Outputs (results/dl/):
  metrics_horizon.csv, metrics_peak_offpeak.csv, metrics_congestion.csv, metrics_by_hour.csv,
  metrics_validation.csv, seeds_summary.csv, diebold_mariano.csv, ablations.csv,
  per_sensor_mae.csv, attention_analysis.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficml import config as C, evaluate as E
from trafficml.dl.data import build_graph_data

DL = C.RESULTS_DIR / "dl"
HZ = list(C.HORIZONS)
P1_MODELS = ["HA", "Persistence", "Ridge-local", "XGB-local", "XGB-spatial", "RF-local", "RF-spatial"]


def load_preds():
    preds = {}
    for m in P1_MODELS:
        f = C.RESULTS_DIR / f"pred_{m}.npz"
        if f.exists():
            d = np.load(f, allow_pickle=True); preds[m] = (d["val_pred"], d["test_pred"])
    for f in sorted(DL.glob("pred_*.npz")):
        d = np.load(f); preds[f.stem[5:]] = (d["val_pred"], d["test_pred"])
    return preds


def dm_test(err_a, err_b, h):
    """Diebold-Mariano test on absolute errors (network-mean per timestamp), Newey-West variance
    with h-1 lags.  Negative statistic = model A more accurate.  Returns (DM, p two-sided)."""
    d = np.nanmean(np.abs(err_a), axis=1) - np.nanmean(np.abs(err_b), axis=1)
    d = d[~np.isnan(d)]
    T = len(d); dbar = d.mean(); dc = d - dbar
    var = np.sum(dc * dc) / T
    for k in range(1, h):
        var += 2 * (1 - k / h) * np.sum(dc[k:] * dc[:-k]) / T
    dm = dbar / np.sqrt(var / T)
    return float(dm), float(2 * (1 - stats.norm.cdf(abs(dm))))


def main():
    gd = build_graph_data()
    sp = gd.split
    Yv, Yt = gd.y_all[sp.val], gd.y_all[sp.test]
    tpeak = E.target_peak_flags(gd.peak, sp.test)
    thours = np.stack([gd.index.hour.values[sp.test + s] for s in C.HORIZONS.values()], axis=1)
    preds = load_preds()
    H, P, CG, BH, V, PS = [], [], [], [], [], []
    for m, (pv, pt) in preds.items():
        H.append(E.horizon_metrics(Yt, pt, HZ, m)); P.append(E.peak_offpeak_metrics(Yt, pt, tpeak, HZ, m))
        CG.append(E.congestion_detection(Yt, pt, HZ, m)); BH.append(E.metrics_by_hour(Yt, pt, thours, HZ, m))
        V.append(E.horizon_metrics(Yv, pv, HZ, m))
        psm = E.per_sensor_mae(Yt, pt)
        PS.append(pd.DataFrame({"model": m, "sensor_id": gd.sensor_ids, **{h: psm[:, k] for k, h in enumerate(HZ)}}))
    cat = lambda L: pd.concat(L, ignore_index=True)
    hor = cat(H); hor.to_csv(DL / "metrics_horizon.csv", index=False)
    cat(P).to_csv(DL / "metrics_peak_offpeak.csv", index=False)
    cat(CG).to_csv(DL / "metrics_congestion.csv", index=False)
    cat(BH).to_csv(DL / "metrics_by_hour.csv", index=False)
    cat(V).to_csv(DL / "metrics_validation.csv", index=False)
    cat(PS).to_csv(DL / "per_sensor_mae.csv", index=False)

    # seeds of the main model
    seeds = [m for m in ("ST-GAT-short", "ST-GAT-seed1", "ST-GAT-seed2") if m in preds]
    if len(seeds) > 1:
        s = hor[hor.model.isin(seeds)].groupby("horizon")[["MAE", "RMSE", "MAPE"]].agg(["mean", "std"])
        s.columns = [f"{a}_{b}" for a, b in s.columns]; s.reset_index().to_csv(DL / "seeds_summary.csv", index=False)

    # significance vs the best Phase 1 model
    rows = []
    ref = preds["XGB-spatial"][1]
    for m in preds:
        if m in P1_MODELS:
            continue
        for k, (hn, steps) in enumerate(C.HORIZONS.items()):
            ea = preds[m][1][:, :, k] - Yt[:, :, k]; eb = ref[:, :, k] - Yt[:, :, k]
            dm, p = dm_test(ea, eb, steps)
            rows.append({"model": m, "vs": "XGB-spatial", "horizon": hn, "DM": dm, "p_value": p})
    pd.DataFrame(rows).to_csv(DL / "diebold_mariano.csv", index=False)

    # ablation table: design ablations are relative to ST-GAT-short (same 12-epoch budget)
    base = hor[(hor.model == ("ST-GAT-short" if "ST-GAT-short" in preds else "ST-GAT"))].set_index("horizon")["MAE"]
    ab = []
    for m in hor.model.unique():
        r = hor[hor.model == m].set_index("horizon")["MAE"]
        ab.append({"model": m, **{h: r.get(h, np.nan) for h in HZ + ["average"]},
                   **{f"delta_{h}": r.get(h, np.nan) - base.get(h, np.nan) for h in HZ}})
    pd.DataFrame(ab).to_csv(DL / "ablations.csv", index=False)
    print(hor[hor.horizon != "average"].pivot(index="model", columns="horizon", values="MAE").round(2))


if __name__ == "__main__":
    main()
