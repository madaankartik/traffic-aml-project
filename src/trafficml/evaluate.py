"""Masked forecasting metrics by horizon, period (peak / off-peak), sensor and time of day.

All metrics ignore timestamps where the *ground truth* is missing (sensor
outage), following the masked-metric convention of Li et al. (2018, DCRNN).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Elementary masked metrics
# --------------------------------------------------------------------------- #
def _mask(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return ~np.isnan(y_true) & ~np.isnan(y_pred)


def masked_mae(y_true, y_pred) -> float:
    m = _mask(y_true, y_pred)
    return float(np.mean(np.abs(y_true[m] - y_pred[m])))


def masked_rmse(y_true, y_pred) -> float:
    m = _mask(y_true, y_pred)
    return float(np.sqrt(np.mean(np.square(y_true[m] - y_pred[m]))))


def masked_mape(y_true, y_pred) -> float:
    """Mean absolute percentage error (%); the ground truth is never 0 after masking."""
    m = _mask(y_true, y_pred)
    return float(100.0 * np.mean(np.abs(y_true[m] - y_pred[m]) / np.abs(y_true[m])))


def all_metrics(y_true, y_pred) -> dict:
    return {"MAE": masked_mae(y_true, y_pred), "RMSE": masked_rmse(y_true, y_pred),
            "MAPE": masked_mape(y_true, y_pred), "n_obs": int(_mask(y_true, y_pred).sum())}


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #
def horizon_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, horizon_names: list, model: str) -> pd.DataFrame:
    """Y_* have shape (samples, sensors, horizons)."""
    rows = []
    for h, name in enumerate(horizon_names):
        rows.append({"model": model, "horizon": name, **all_metrics(Y_true[:, :, h], Y_pred[:, :, h])})
    rows.append({"model": model, "horizon": "average", **all_metrics(Y_true, Y_pred)})
    return pd.DataFrame(rows)


def target_peak_flags(peak: np.ndarray, sample_idx: np.ndarray, horizons: dict = C.HORIZONS) -> np.ndarray:
    """(samples, horizons) boolean: is the *target* timestamp t + h in a peak period?"""
    return np.stack([peak[sample_idx + s] for s in horizons.values()], axis=1)


def peak_offpeak_metrics(Y_true, Y_pred, tpeak: np.ndarray, horizon_names: list, model: str) -> pd.DataFrame:
    rows = []
    for h, name in enumerate(horizon_names):
        for period, sel in (("peak", tpeak[:, h]), ("off-peak", ~tpeak[:, h])):
            rows.append({"model": model, "horizon": name, "period": period,
                         **all_metrics(Y_true[sel, :, h], Y_pred[sel, :, h])})
    return pd.DataFrame(rows)


def per_sensor_mae(Y_true, Y_pred) -> np.ndarray:
    """(sensors, horizons) masked MAE."""
    N, H = Y_true.shape[1], Y_true.shape[2]
    out = np.full((N, H), np.nan)
    for n in range(N):
        for h in range(H):
            m = _mask(Y_true[:, n, h], Y_pred[:, n, h])
            if m.any():
                out[n, h] = np.mean(np.abs(Y_true[m, n, h] - Y_pred[m, n, h]))
    return out


def metrics_by_hour(Y_true, Y_pred, target_hours: np.ndarray, horizon_names: list, model: str) -> pd.DataFrame:
    """MAE per hour of day of the *target* timestamp; target_hours has shape (samples, horizons)."""
    rows = []
    for h, name in enumerate(horizon_names):
        for hour in range(24):
            sel = target_hours[:, h] == hour
            if sel.any():
                rows.append({"model": model, "horizon": name, "hour": hour,
                             "MAE": masked_mae(Y_true[sel, :, h], Y_pred[sel, :, h])})
    return pd.DataFrame(rows)


def congestion_detection(Y_true, Y_pred, horizon_names: list, model: str,
                         threshold: float = C.CONGESTION_THRESHOLD) -> pd.DataFrame:
    """Treat "speed < threshold" as a congestion event and score the forecast as a detector."""
    rows = []
    for h, name in enumerate(horizon_names):
        m = _mask(Y_true[:, :, h], Y_pred[:, :, h])
        t = Y_true[:, :, h][m] < threshold
        p = Y_pred[:, :, h][m] < threshold
        tp = np.sum(t & p); fp = np.sum(~t & p); fn = np.sum(t & ~p)
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else np.nan
        rows.append({"model": model, "horizon": name, "precision": prec, "recall": rec, "F1": f1,
                     "congested_share": float(t.mean())})
    return pd.DataFrame(rows)
