"""Central configuration: paths, constants and model hyper-parameters.

Everything that an experiment depends on is declared here so that the
notebooks, scripts and report all read from a single source of truth.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
GRAPH_DIR = DATA_DIR / "sensor_graph"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

SPEED_FILE = RAW_DIR / "metr-la.h5"
LOCATIONS_FILE = GRAPH_DIR / "graph_sensor_locations.csv"
DISTANCES_FILE = GRAPH_DIR / "distances_la_2012.csv"
SENSOR_IDS_FILE = GRAPH_DIR / "graph_sensor_ids.txt"
ADJ_PICKLE = GRAPH_DIR / "adj_mx.pkl"

# Public mirrors of the DCRNN release of METR-LA (used by scripts/download_data.py)
DOWNLOAD_URLS = {
    "metr-la.h5": "https://raw.githubusercontent.com/deepkashiwa20/DL-Traff-Graph/main/METRLA/metr-la.h5",
    "graph_sensor_locations.csv": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_locations.csv",
    "distances_la_2012.csv": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/distances_la_2012.csv",
    "graph_sensor_ids.txt": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/graph_sensor_ids.txt",
    "adj_mx.pkl": "https://raw.githubusercontent.com/liyaguang/DCRNN/master/data/sensor_graph/adj_mx.pkl",
}

# --------------------------------------------------------------------------- #
# Dataset constants
# --------------------------------------------------------------------------- #
STEP_MINUTES = 5                      # sampling interval of METR-LA
STEPS_PER_DAY = 24 * 60 // STEP_MINUTES   # 288
STEPS_PER_WEEK = 7 * STEPS_PER_DAY        # 2016
FREE_FLOW_SPEED = 65.0                # mph, typical LA freeway free-flow speed (used for LWR discussion)
CONGESTION_THRESHOLD = 35.0           # mph — Caltrans PeMS convention for "congested" freeway speed

# Peak periods (weekdays only): morning 07:00–10:00, evening 16:00–19:00 local time
PEAK_WINDOWS = ((7, 10), (16, 19))

# --------------------------------------------------------------------------- #
# Experiment design
# --------------------------------------------------------------------------- #
N_LAGS = 12                           # 12 x 5 min = one hour of history (DCRNN convention)
HORIZONS = {"15min": 3, "30min": 6, "60min": 12}   # forecast horizons in steps
SPLIT_RATIOS = (0.7, 0.1, 0.2)        # train / validation / test, chronological
GAP_STEPS = 12                        # one-hour buffer between splits (no target/feature overlap)
NEIGHBOUR_LAGS = (0, 1, 2)            # neighbour speed at t, t-5, t-10 min
RANDOM_SEED = 42


@dataclass
class XGBParams:
    """Hyper-parameters for the per-segment XGBoost regressors."""

    n_estimators: int = 600
    learning_rate: float = 0.05
    max_depth: int = 6
    min_child_weight: float = 5.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    objective: str = "reg:squarederror"
    tree_method: str = "hist"
    early_stopping_rounds: int = 40
    n_jobs: int = 2

    def to_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "objective": self.objective,
            "tree_method": self.tree_method,
            "n_jobs": self.n_jobs,
            "random_state": RANDOM_SEED,
        }


@dataclass
class RFParams:
    """Hyper-parameters for the per-segment multi-output Random Forest."""

    n_estimators: int = 100
    max_depth: int | None = 14
    min_samples_leaf: int = 5
    max_features: float | str = 0.5
    n_jobs: int = 2

    def to_dict(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "n_jobs": self.n_jobs,
            "random_state": RANDOM_SEED,
        }


@dataclass
class RidgeParams:
    """Hyper-parameters for the per-segment Ridge regression baseline."""

    alpha: float = 1.0


@dataclass
class ExperimentConfig:
    """Bundle of everything a run needs; serialised to results/ for reproducibility."""

    n_lags: int = N_LAGS
    horizons: dict = field(default_factory=lambda: dict(HORIZONS))
    split_ratios: tuple = SPLIT_RATIOS
    gap_steps: int = GAP_STEPS
    neighbour_lags: tuple = NEIGHBOUR_LAGS
    seed: int = RANDOM_SEED
    xgb: XGBParams = field(default_factory=XGBParams)
    rf: RFParams = field(default_factory=RFParams)
    ridge: RidgeParams = field(default_factory=RidgeParams)
