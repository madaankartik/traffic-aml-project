"""Feature engineering for segment-level traffic-speed forecasting.

Design principles (from the project brief and the forecasting literature):

* **Lagged speed features** - the last hour (12 x 5 min) of the segment's own speed.
* **Calendar features** - cyclic (sin/cos) time-of-day and day-of-week encodings, so
  23:55 and 00:00 are neighbours and Sunday is next to Monday.
* **Segment features** - a per-sensor *historical average* (HA) profile over the
  2016 time-of-week slots, estimated on training timestamps only, plus the current
  deviation from that profile.
* **Neighbour (spatial-lag) features** - adjacency-weighted mean speed of the
  upstream and downstream sensors.  Motivated by the LWR kinematic-wave model:
  congestion propagates *backwards* along the road, so upstream sensors see a jam
  later than downstream ones.  These features are the "cheap" classical stand-in
  for the message passing a Graph Attention Network will learn in Phase 2.
* **Sensor outages are masked** (NaN), never treated as 0 mph; tree models
  (XGBoost, sklearn >= 1.4 forests) handle NaN natively, Ridge receives an
  HA-imputed copy.
* **Chronological split** with a one-hour gap; no random shuffling of sequences.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config as C
from .data import SensorGraph


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
@dataclass
class Split:
    """Index arrays of *sample timestamps* ("now" instants) for each partition."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_end: int      # first timestamp *not* in train (used to fit statistics)
    val_end: int

    def summary(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        rows = []
        for name, idx in (("train", self.train), ("val", self.val), ("test", self.test)):
            rows.append({"split": name, "n_samples": len(idx), "start": index[idx[0]], "end": index[idx[-1]]})
        return pd.DataFrame(rows)


def chronological_split(n_steps: int, n_lags: int = C.N_LAGS, max_horizon: int = max(C.HORIZONS.values()),
                        ratios=C.SPLIT_RATIOS, gap: int = C.GAP_STEPS) -> Split:
    """Split timestamps 70/10/20 in time order.

    A sample at time t uses features from [t-n_lags+1, t] and targets up to
    t+max_horizon.  A ``gap`` of max_horizon steps is left between partitions so
    that no training *target* falls inside the validation/test period and no
    validation/test *feature window* reaches back into training targets.
    """
    train_end = int(n_steps * ratios[0])
    val_end = int(n_steps * (ratios[0] + ratios[1]))
    train = np.arange(n_lags - 1, train_end - max_horizon)
    val = np.arange(train_end + gap, val_end - max_horizon)
    test = np.arange(val_end + gap, n_steps - max_horizon)
    return Split(train=train, val=val, test=test, train_end=train_end, val_end=val_end)


# --------------------------------------------------------------------------- #
# Calendar features
# --------------------------------------------------------------------------- #
TIME_FEATURE_NAMES = ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend", "hour"]


def time_of_day_slot(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.hour * 60 + index.minute) // C.STEP_MINUTES


def time_of_week_slot(index: pd.DatetimeIndex) -> np.ndarray:
    return index.dayofweek * C.STEPS_PER_DAY + time_of_day_slot(index)


def make_time_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Cyclic encodings of time-of-day and day-of-week plus weekend flag and hour."""
    tod = time_of_day_slot(index).astype(float)
    dow = index.dayofweek.astype(float)
    feats = np.column_stack([
        np.sin(2 * np.pi * tod / C.STEPS_PER_DAY),
        np.cos(2 * np.pi * tod / C.STEPS_PER_DAY),
        np.sin(2 * np.pi * dow / 7.0),
        np.cos(2 * np.pi * dow / 7.0),
        (dow >= 5).astype(float),
        index.hour.astype(float),
    ]).astype(np.float32)
    return feats


def is_peak(index: pd.DatetimeIndex) -> np.ndarray:
    """Weekday morning (07-10) and evening (16-19) peak indicator."""
    weekday = index.dayofweek < 5
    hour = index.hour
    peak = np.zeros(len(index), dtype=bool)
    for start, end in C.PEAK_WINDOWS:
        peak |= (hour >= start) & (hour < end)
    return peak & weekday


# --------------------------------------------------------------------------- #
# Historical-average profile (train only)
# --------------------------------------------------------------------------- #
def historical_average_profile(X: np.ndarray, index: pd.DatetimeIndex, train_end: int) -> np.ndarray:
    """Mean speed per (time-of-week slot, sensor), using training timestamps only.

    Returns an array of shape (STEPS_PER_WEEK, N).  Slots with no observation
    fall back to the sensor's overall training mean.
    """
    tow = time_of_week_slot(index[:train_end])
    Xtr = X[:train_end]
    profile = np.full((C.STEPS_PER_WEEK, X.shape[1]), np.nan, dtype=np.float32)
    df = pd.DataFrame(Xtr)
    grouped = df.groupby(tow).mean()           # NaN-aware mean
    profile[grouped.index.values] = grouped.values.astype(np.float32)
    sensor_mean = np.nanmean(Xtr, axis=0)
    nan_rows = np.isnan(profile)
    profile[nan_rows] = np.broadcast_to(sensor_mean, profile.shape)[nan_rows]
    return profile


# --------------------------------------------------------------------------- #
# Lag, window-statistic and neighbour features
# --------------------------------------------------------------------------- #
def make_lag_tensor(X: np.ndarray, n_lags: int = C.N_LAGS) -> np.ndarray:
    """L[t, n, k] = X[t-k, n] for k = 0..n_lags-1 (NaN before the series starts)."""
    T, N = X.shape
    L = np.full((T, N, n_lags), np.nan, dtype=np.float32)
    for k in range(n_lags):
        L[k:, :, k] = X[: T - k]
    return L


WINDOW_STAT_NAMES = ["win_mean", "win_std", "win_min", "win_max", "win_frac_missing", "trend_15min"]


def make_window_stats(L: np.ndarray) -> np.ndarray:
    """Summary statistics of the one-hour lag window (NaN-aware; all-NaN windows give NaN)."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(L, axis=2)
        std = np.nanstd(L, axis=2)
        mn = np.nanmin(L, axis=2)
        mx = np.nanmax(L, axis=2)
    frac_missing = np.isnan(L).mean(axis=2)
    trend = L[:, :, 0] - L[:, :, 3]           # speed change over the last 15 minutes
    return np.stack([mean, std, mn, mx, frac_missing, trend], axis=2).astype(np.float32)


def neighbour_feature_names(lags=C.NEIGHBOUR_LAGS) -> list:
    names = []
    for k in lags:
        names += [f"up_mean_lag{k}", f"down_mean_lag{k}"]
    names += ["nbr_min_lag0", "nbr_frac_congested_lag0"]
    return names


def make_neighbour_features(X: np.ndarray, graph: SensorGraph, lags=C.NEIGHBOUR_LAGS,
                            congestion_threshold: float = C.CONGESTION_THRESHOLD) -> np.ndarray:
    """Adjacency-weighted upstream / downstream mean speeds and neighbourhood congestion.

    up_mean[t, i]   = sum_j A[j, i] X[t, j] / sum_j A[j, i] 1[X[t, j] observed]   (j != i)
    down_mean[t, i] = sum_j A[i, j] X[t, j] / sum_j A[i, j] 1[X[t, j] observed]   (j != i)

    NaN (outage) readings are excluded from both numerator and denominator.
    """
    A = graph.adjacency.copy()
    np.fill_diagonal(A, 0.0)
    W_down = A                      # W_down[i, j] = A[i, j]
    W_up = A.T                      # W_up[i, j]   = A[j, i]
    observed = ~np.isnan(X)
    Xf = np.where(observed, X, 0.0)
    M = observed.astype(np.float64)

    def weighted_mean(W):
        num = Xf @ W.T
        den = M @ W.T
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    up0, down0 = weighted_mean(W_up), weighted_mean(W_down)
    T, N = X.shape
    feats = []
    for k in lags:
        up_k = np.full((T, N), np.nan); down_k = np.full((T, N), np.nan)
        up_k[k:] = up0[: T - k]; down_k[k:] = down0[: T - k]
        feats += [up_k, down_k]

    # Neighbourhood (union of up/down) minimum speed and fraction congested at lag 0
    B = ((W_up + W_down) > 0).astype(float)
    nbr_min = np.full((T, N), np.nan); frac_cong = np.full((T, N), np.nan)
    cong = (observed & (X < congestion_threshold)).astype(float)
    cnt = M @ B.T
    with np.errstate(invalid="ignore", divide="ignore"):
        frac_cong = np.where(cnt > 0, (cong @ B.T) / cnt, np.nan)
    Xinf = np.where(observed, X, np.inf)
    for i in range(N):
        nb = np.flatnonzero(B[i])
        if len(nb):
            m = Xinf[:, nb].min(axis=1)
            nbr_min[:, i] = np.where(np.isinf(m), np.nan, m)
    feats += [nbr_min, frac_cong]
    return np.stack(feats, axis=2).astype(np.float32)


def make_targets(X: np.ndarray, horizons: dict = C.HORIZONS) -> np.ndarray:
    """Y[t, n, h] = X[t + horizon_h, n]  (NaN beyond the end or during outages)."""
    T, N = X.shape
    Y = np.full((T, N, len(horizons)), np.nan, dtype=np.float32)
    for h, steps in enumerate(horizons.values()):
        Y[: T - steps, :, h] = X[steps:]
    return Y


# --------------------------------------------------------------------------- #
# Static per-sensor descriptors (used for EDA and the pooled model)
# --------------------------------------------------------------------------- #
def sensor_static_table(X: np.ndarray, graph: SensorGraph, train_end: int,
                        congestion_threshold: float = C.CONGESTION_THRESHOLD) -> pd.DataFrame:
    Xtr = X[:train_end]
    A = graph.adjacency.copy(); np.fill_diagonal(A, 0)
    with np.errstate(all="ignore"):
        df = pd.DataFrame({
            "sensor_id": graph.sensor_ids,
            "latitude": graph.locations["latitude"].values,
            "longitude": graph.locations["longitude"].values,
            "train_mean_speed": np.nanmean(Xtr, axis=0),
            "train_std_speed": np.nanstd(Xtr, axis=0),
            "train_frac_congested": np.nanmean((Xtr < congestion_threshold).astype(float), axis=0),
            "frac_missing": np.isnan(X).mean(axis=0),
            "out_degree": (A > 0).sum(axis=1),
            "in_degree": (A > 0).sum(axis=0),
        })
    return df


# --------------------------------------------------------------------------- #
# Feature bundle
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSet:
    """All engineered arrays, aligned on the time axis.

    Shapes: T timestamps, N sensors, L lags, H horizons.
    """

    index: pd.DatetimeIndex
    sensor_ids: list
    lag: np.ndarray            # (T, N, L)
    stats: np.ndarray          # (T, N, 6)
    time: np.ndarray           # (T, 6)
    ha: np.ndarray             # (T, N)  historical average for the *current* slot
    ha_target: np.ndarray      # (T, N, H) historical average for each *target* slot
    dev: np.ndarray            # (T, N)  current speed minus HA
    nbr: np.ndarray            # (T, N, 8)
    targets: np.ndarray        # (T, N, H)
    split: Split
    profile: np.ndarray        # (2016, N)
    peak: np.ndarray           # (T,) bool, peak flag of the *sample* time
    names: dict = field(default_factory=dict)

    @property
    def n_sensors(self) -> int:
        return len(self.sensor_ids)

    def local_feature_names(self) -> list:
        return self.names["lag"] + self.names["stats"] + self.names["time"] + ["ha_now", "dev_from_ha"]

    def spatial_feature_names(self) -> list:
        return self.local_feature_names() + self.names["nbr"]

    def sensor_matrix(self, n: int, spatial: bool = False) -> np.ndarray:
        """Design matrix (T, F) for sensor n.  ``spatial=True`` appends neighbour features."""
        blocks = [self.lag[:, n, :], self.stats[:, n, :], self.time,
                  self.ha[:, n][:, None], self.dev[:, n][:, None]]
        if spatial:
            blocks.append(self.nbr[:, n, :])
        return np.concatenate(blocks, axis=1)

    def impute(self, Xn: np.ndarray, n: int) -> np.ndarray:
        """Fill NaNs for models without native missing-value support (Ridge):
        every missing speed-like value is replaced by the sensor's HA, every
        missing derived statistic by 0."""
        Xn = Xn.copy()
        ha = self.ha[:, n]
        names = self.spatial_feature_names() if Xn.shape[1] == len(self.spatial_feature_names()) else self.local_feature_names()
        for j, name in enumerate(names):
            col = Xn[:, j]
            miss = np.isnan(col)
            if not miss.any():
                continue
            if name.startswith(("lag", "win_mean", "win_min", "win_max", "up_", "down_", "nbr_min")):
                col[miss] = ha[miss]
            else:
                col[miss] = 0.0
        return Xn


def build_features(X_df: pd.DataFrame, graph: SensorGraph, cfg: C.ExperimentConfig | None = None) -> FeatureSet:
    """End-to-end feature construction from the NaN-masked speed DataFrame."""
    cfg = cfg or C.ExperimentConfig()
    X = X_df.values.astype(np.float32)
    index = X_df.index
    T, N = X.shape
    split = chronological_split(T, cfg.n_lags, max(cfg.horizons.values()), cfg.split_ratios, cfg.gap_steps)

    profile = historical_average_profile(X, index, split.train_end)
    tow = time_of_week_slot(index)
    ha = profile[tow]                                             # (T, N)
    ha_target = np.stack([profile[(tow + s) % C.STEPS_PER_WEEK] for s in cfg.horizons.values()], axis=2)

    lag = make_lag_tensor(X, cfg.n_lags)
    stats = make_window_stats(lag)
    time = make_time_features(index)
    dev = (X - ha).astype(np.float32)
    nbr = make_neighbour_features(X, graph, cfg.neighbour_lags)
    targets = make_targets(X, cfg.horizons)

    names = {
        "lag": [f"lag{k}" for k in range(cfg.n_lags)],
        "stats": list(WINDOW_STAT_NAMES),
        "time": list(TIME_FEATURE_NAMES),
        "nbr": neighbour_feature_names(cfg.neighbour_lags),
        "targets": list(cfg.horizons.keys()),
    }
    return FeatureSet(index=index, sensor_ids=list(X_df.columns), lag=lag, stats=stats, time=time, ha=ha,
                      ha_target=ha_target, dev=dev, nbr=nbr, targets=targets, split=split, profile=profile,
                      peak=is_peak(index), names=names)
