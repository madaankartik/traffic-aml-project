"""trafficml — Phase 1 (classical ML) pipeline for METR-LA traffic-speed forecasting.

Project 4: Multi-Agent Traffic Flow Control via Graph Attention Networks and LWR Models
Team T082, Section B.

Sub-modules
-----------
config      : paths, constants and experiment hyper-parameters
data        : loading METR-LA speed series and the sensor road graph
features    : lag / calendar / segment / neighbour feature engineering and chronological splits
models      : naive baselines, per-segment Ridge, XGBoost and multi-output Random Forest
evaluate    : masked MAE / RMSE / MAPE by horizon, peak vs off-peak and per-sensor metrics
plots       : EDA and result figures with a consistent, colour-blind-safe style
"""

__version__ = "0.1.0"
