"""Unit tests for the feature pipeline and the masked metrics (no dataset required).

Run with:  python -m pytest tests -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trafficml import config as C                                     # noqa: E402
from trafficml import evaluate as E, features as F                     # noqa: E402
from trafficml.data import SensorGraph, build_adjacency, mask_outages  # noqa: E402


def toy_graph(n=4):
    ids = [f"s{i}" for i in range(n)]
    loc = pd.DataFrame({"sensor_id": ids, "latitude": np.zeros(n), "longitude": np.arange(n, dtype=float)})
    # chain s0 -> s1 -> s2 -> s3 (downstream direction), distances in metres; longer multi-hop
    # distances widen sigma so that only one-hop links survive the 0.1 kernel threshold
    rows = [(f"s{i}", f"s{i + 1}", 1000.0) for i in range(n - 1)] + [(s, s, 0.0) for s in ids]
    rows += [("s0", "s2", 3000.0), ("s1", "s3", 3000.0), ("s0", "s3", 6000.0)]
    dist = pd.DataFrame(rows, columns=["from", "to", "cost"])
    adj = build_adjacency(dist, ids, normalized_k=0.1)
    return SensorGraph(sensor_ids=ids, locations=loc, distances=dist, adjacency=adj)


def test_chronological_split_has_gaps_and_no_overlap():
    T = 1000
    s = F.chronological_split(T, n_lags=12, max_horizon=12, gap=12)
    assert s.train[-1] + 12 < s.val[0], "training targets must end before validation samples start"
    assert s.val[-1] + 12 < s.test[0]
    assert s.test[-1] + 12 < T, "last test sample must still have a 60-min target"
    assert s.train[0] == 11, "first sample needs a full one-hour lag window"
    assert len(set(s.train) & set(s.val)) == 0 and len(set(s.val) & set(s.test)) == 0


def test_lag_tensor_and_targets_are_aligned():
    X = np.arange(40, dtype=np.float32).reshape(20, 2)
    L = F.make_lag_tensor(X, n_lags=3)
    assert np.array_equal(L[5, 0], [X[5, 0], X[4, 0], X[3, 0]])
    assert np.isnan(L[0, 0, 1])                           # no history before the series starts
    Y = F.make_targets(X, {"a": 2, "b": 5})
    assert Y[3, 1, 0] == X[5, 1] and Y[3, 1, 1] == X[8, 1]
    assert np.isnan(Y[18, 0, 0])                          # beyond the end


def test_mask_outages_turns_zero_into_nan():
    df = pd.DataFrame([[0.0, 50.0], [60.0, 0.0]])
    m = mask_outages(df)
    assert np.isnan(m.iloc[0, 0]) and np.isnan(m.iloc[1, 1]) and m.iloc[0, 1] == 50.0


def test_neighbour_features_use_direction_and_ignore_missing():
    g = toy_graph()
    A = g.adjacency
    assert A[0, 1] > 0 and A[1, 0] == 0, "road distance is directional: s0 -> s1 only"
    X = np.array([[60.0, 40.0, np.nan, 20.0]] * 3, dtype=np.float32)        # 3 timestamps, 4 sensors
    nb = F.make_neighbour_features(X, g, lags=(0,))
    names = F.neighbour_feature_names((0,))
    up, down = names.index("up_mean_lag0"), names.index("down_mean_lag0")
    assert np.isclose(nb[2, 1, up], 60.0)                 # upstream of s1 is s0
    assert np.isnan(nb[2, 1, down])                       # downstream of s1 is s2, which is missing -> NaN not 0
    assert np.isnan(nb[2, 3, up])                         # s3's only upstream neighbour s2 is missing -> NaN
    assert np.isnan(nb[2, 0, up])                         # s0 has no upstream neighbour


def test_time_features_are_cyclic():
    idx = pd.to_datetime(["2012-03-04 23:55", "2012-03-05 00:00"])
    t = F.make_time_features(idx)
    assert np.allclose(t[0, :2], t[1, :2], atol=0.03), "23:55 and 00:00 must be neighbours in (sin, cos) space"
    assert t[0, 4] == 1.0 and t[1, 4] == 0.0              # Sunday is weekend, Monday is not


def test_historical_average_uses_train_only():
    T = 2 * C.STEPS_PER_WEEK
    idx = pd.date_range("2012-03-05", periods=T, freq="5min")   # Monday start
    X = np.ones((T, 1), dtype=np.float32)
    X[C.STEPS_PER_WEEK:] = 100.0                                # second week is "future"
    prof = F.historical_average_profile(X, idx, train_end=C.STEPS_PER_WEEK)
    assert np.all(prof == 1.0), "profile must ignore timestamps beyond train_end"


def test_masked_metrics_ignore_missing_truth():
    y = np.array([10.0, np.nan, 30.0]); p = np.array([12.0, 999.0, 27.0])
    assert np.isclose(E.masked_mae(y, p), 2.5)
    assert np.isclose(E.masked_rmse(y, p), np.sqrt((4 + 9) / 2))
    assert np.isclose(E.masked_mape(y, p), 100 * (0.2 + 0.1) / 2)


def test_peak_flags():
    idx = pd.to_datetime(["2012-03-05 08:00", "2012-03-05 12:00", "2012-03-10 08:00", "2012-03-05 17:30"])
    assert list(F.is_peak(idx)) == [True, False, False, True]   # Mon 8am, Mon noon, Sat 8am, Mon 5:30pm
