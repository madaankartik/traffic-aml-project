"""Phase 2 data pipeline: the Phase 1 table turned into graph-sequence tensors.

Everything that defined the Phase 1 comparison is reused unchanged so the ML vs DL
comparison is fair (same masking, same chronological split with gaps, same one-hour
history, same horizons, same sample indices, same targets):

    input  x[t] : (12 steps, 207 sensors, 7 features)   steps t-11 .. t
    target y[t] : (207 sensors, 3 horizons)              speed at t+3, t+6, t+12 (NaN = unknown)

Node features per step
    0  speed, z-scored per sensor with TRAINING statistics; unknown -> 0 after scaling
    1  mask (1 = observed, 0 = sensor outage)
    2-5 sin/cos time of day, sin/cos day of week
    6  historical-average speed of that slot (training weeks only), z-scored like speed
Unknown readings are never treated as 0 mph: the scaled value 0 means "the sensor's mean"
and the mask channel tells the network the value is missing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .. import config as C
from .. import data as D
from .. import features as F

N_FEATURES = 7


@dataclass
class GraphData:
    x_all: np.ndarray        # (T, N, 7) per-timestamp node features
    y_all: np.ndarray        # (T, N, H) raw-mph targets, NaN = unknown
    ha_target: np.ndarray    # (T, N, H) historical average of each target slot (mph)
    mean: np.ndarray         # (N,) training mean speed
    std: np.ndarray          # (N,) training std of speed
    adjacency: np.ndarray    # (N, N) DCRNN Gaussian-kernel weights, A[i,j]>0: j downstream of i
    distances_km: np.ndarray # (N, N) road distance i->j in km (inf if unknown)
    split: F.Split
    index: object
    peak: np.ndarray
    sensor_ids: list
    n_lags: int = C.N_LAGS


def build_graph_data() -> GraphData:
    X_df, _, graph = D.load_all()
    X = X_df.values.astype(np.float32)
    index = X_df.index
    T, N = X.shape
    split = F.chronological_split(T)
    tr_end = split.train_end

    mean = np.nanmean(X[:tr_end], axis=0).astype(np.float32)
    std = np.nanstd(X[:tr_end], axis=0).astype(np.float32)
    std[std < 1e-3] = 1.0

    profile = F.historical_average_profile(X, index, tr_end)
    tow = F.time_of_week_slot(index)
    ha = profile[tow]
    ha_target = np.stack([profile[(tow + s) % C.STEPS_PER_WEEK] for s in C.HORIZONS.values()], axis=2)

    z = (X - mean) / std
    mask = (~np.isnan(X)).astype(np.float32)
    z = np.where(np.isnan(z), 0.0, z).astype(np.float32)
    tfeat = F.make_time_features(index)[:, :4].astype(np.float32)          # tod sin/cos, dow sin/cos
    x_all = np.concatenate([
        z[..., None], mask[..., None],
        np.broadcast_to(tfeat[:, None, :], (T, N, 4)),
        ((ha - mean) / std)[..., None].astype(np.float32),
    ], axis=2).astype(np.float32)

    y_all = F.make_targets(X)

    ids = graph.sensor_ids
    id2 = {s: k for k, s in enumerate(ids)}
    dist = np.full((N, N), np.inf, np.float32)
    for frm, to, cost in graph.distances.itertuples(index=False):
        if frm in id2 and to in id2:
            dist[id2[frm], id2[to]] = cost / 1000.0
    return GraphData(x_all, y_all, ha_target, mean, std, np.asarray(graph.adjacency, np.float32), dist,
                     split, index, F.is_peak(index), ids)


class WindowDataset(torch.utils.data.Dataset):
    """Sample s -> (x[t-11..t], y[t], base[t]) for t = idx[s].

    ``base`` is the forecast the network corrects (zeros for a pure DL model, the
    historical average or the XGBoost forecast for residual/hybrid models)."""

    def __init__(self, gd: GraphData, idx: np.ndarray, base: np.ndarray | None = None):
        self.gd, self.idx = gd, np.asarray(idx)
        self.base = base if base is not None else np.zeros((len(idx), gd.x_all.shape[1], gd.y_all.shape[2]), np.float32)
        assert len(self.base) == len(self.idx)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, s):
        t = self.idx[s]
        L = self.gd.n_lags
        x = self.gd.x_all[t - L + 1: t + 1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.gd.y_all[t]), torch.from_numpy(self.base[s])


def batch_iter(gd: GraphData, idx: np.ndarray, base: np.ndarray | None, batch_size: int,
               shuffle: bool, rng: np.random.Generator | None = None):
    """Fast vectorised batching (no DataLoader workers on a 2-core machine)."""
    idx = np.asarray(idx)
    order = rng.permutation(len(idx)) if shuffle else np.arange(len(idx))
    L = gd.n_lags
    offs = np.arange(-L + 1, 1)
    if base is None:
        base = np.zeros((len(idx), gd.x_all.shape[1], gd.y_all.shape[2]), np.float32)
    for b in range(0, len(order), batch_size):
        sel = order[b: b + batch_size]
        t = idx[sel]
        x = gd.x_all[t[:, None] + offs[None, :]]           # (B, L, N, F)
        yield (torch.from_numpy(x), torch.from_numpy(gd.y_all[t]), torch.from_numpy(base[sel]), sel)
