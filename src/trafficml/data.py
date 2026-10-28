"""Loading METR-LA speed data and the sensor road graph.

METR-LA (Jagadish et al., 2014; released with DCRNN, Li et al., 2018) contains
average vehicle speeds (mph) from 207 loop detectors on Los Angeles County
highways, sampled every 5 minutes from 1 March 2012 to 27 June 2012
(34,272 timestamps).  A speed of exactly 0.0 denotes a sensor outage, not a
stationary jam, and is therefore treated as *missing* throughout the project.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd

from . import config as C


# --------------------------------------------------------------------------- #
# Speed series
# --------------------------------------------------------------------------- #
def load_speed(path=C.SPEED_FILE) -> pd.DataFrame:
    """Read the DCRNN ``metr-la.h5`` file into a (time x sensor) DataFrame.

    The file is a pandas "fixed" HDF store written with an old pandas version;
    reading it through h5py directly keeps us independent of pandas/pytables
    version drift.  Columns are sensor ids (strings), index is a DatetimeIndex
    at 5-minute resolution.
    """
    with h5py.File(path, "r") as f:
        g = f["df"]
        columns = [c.decode() if isinstance(c, bytes) else str(c) for c in g["axis0"][:]]
        index = pd.to_datetime(g["axis1"][:])
        values = g["block0_values"][:]
    df = pd.DataFrame(values, index=index, columns=columns)
    df.index.name = "timestamp"
    return df


def mask_outages(df: pd.DataFrame) -> pd.DataFrame:
    """Replace zero speeds (sensor outages) with NaN so they are never used as
    a 0 mph observation, neither as feature nor as target."""
    return df.mask(df <= 0.0)


# --------------------------------------------------------------------------- #
# Sensor graph
# --------------------------------------------------------------------------- #
@dataclass
class SensorGraph:
    """Road-network graph over the 207 sensors.

    Attributes
    ----------
    sensor_ids : list[str]            ordering shared with the speed columns
    locations  : DataFrame             sensor_id, latitude, longitude (same order)
    distances  : DataFrame             from, to, cost (road distance in metres), raw pairs
    adjacency  : ndarray (N, N)        Gaussian-kernel weighted adjacency, A[i, j] > 0 if
                                        j is reachable from i within the threshold
    """

    sensor_ids: list
    locations: pd.DataFrame
    distances: pd.DataFrame
    adjacency: np.ndarray

    @property
    def n(self) -> int:
        return len(self.sensor_ids)

    def neighbours_out(self, i: int) -> np.ndarray:
        """Downstream neighbours of sensor i (A[i, j] > 0, j != i)."""
        row = self.adjacency[i].copy()
        row[i] = 0.0
        return np.flatnonzero(row > 0)

    def neighbours_in(self, i: int) -> np.ndarray:
        """Upstream neighbours of sensor i (A[j, i] > 0, j != i)."""
        col = self.adjacency[:, i].copy()
        col[i] = 0.0
        return np.flatnonzero(col > 0)


def build_adjacency(distances: pd.DataFrame, sensor_ids: list, normalized_k: float = 0.1) -> np.ndarray:
    """Re-implement the DCRNN adjacency construction (Li et al., 2018, eq. 10).

        A[i, j] = exp(-(d_ij / sigma)^2)   if  >= normalized_k   else 0

    where d_ij is the *road-network* distance from i to j and sigma is the
    standard deviation of all finite pairwise distances.  Because road distance
    is directional (i -> j != j -> i), the matrix is asymmetric.
    """
    n = len(sensor_ids)
    id2ind = {sid: k for k, sid in enumerate(sensor_ids)}
    dist = np.full((n, n), np.inf, dtype=float)
    for frm, to, cost in distances.itertuples(index=False):
        if frm in id2ind and to in id2ind:
            dist[id2ind[frm], id2ind[to]] = cost
    finite = dist[~np.isinf(dist)]
    sigma = finite.std()
    adj = np.exp(-np.square(dist / sigma))
    adj[adj < normalized_k] = 0.0
    return adj


def load_graph() -> SensorGraph:
    """Load sensor ids, coordinates, pairwise road distances and the adjacency."""
    sensor_ids = open(C.SENSOR_IDS_FILE).read().strip().split(",")
    locations = pd.read_csv(C.LOCATIONS_FILE)
    locations["sensor_id"] = locations["sensor_id"].astype(str)
    locations = locations.set_index("sensor_id").loc[sensor_ids].reset_index()
    distances = pd.read_csv(C.DISTANCES_FILE, dtype={"from": str, "to": str, "cost": float})

    try:
        with open(C.ADJ_PICKLE, "rb") as f:
            pkl_ids, _, adjacency = pickle.load(f, encoding="latin1")
        assert list(pkl_ids) == sensor_ids, "adjacency pickle ordering differs from sensor ids"
    except FileNotFoundError:
        adjacency = build_adjacency(distances, sensor_ids)
    return SensorGraph(sensor_ids=sensor_ids, locations=locations, distances=distances, adjacency=np.asarray(adjacency))


def load_all():
    """Convenience loader returning (speed_df_with_NaN_outages, raw_speed_df, graph)."""
    raw = load_speed()
    graph = load_graph()
    assert list(raw.columns) == graph.sensor_ids, "speed columns and graph ordering differ"
    return mask_outages(raw), raw, graph
