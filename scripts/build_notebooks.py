#!/usr/bin/env python3
"""Generate the two project notebooks (notebooks/01_eda.ipynb, notebooks/02_features_and_models.ipynb).

The notebooks are thin, documented front-ends over the ``trafficml`` package:
they reproduce the key tables live and display the figures produced by the
scripts, so that reviewers can read the whole Phase 1 story top to bottom.
Run ``jupyter nbconvert --execute --inplace notebooks/*.ipynb`` afterwards.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = '''import sys, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from IPython.display import Image, display, Markdown
from trafficml import config as C, data, features as F, models as M, evaluate as E
pd.set_option("display.width", 140); pd.set_option("display.precision", 3)
def fig(name, width=820):
    display(Image(filename=str(C.FIGURES_DIR / f"{name}.png"), width=width))
print("project root:", ROOT)'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def eda_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 01 — Dataset quality & exploratory data analysis (METR-LA)

**Project 4 · Team T082 · Section B** — *Multi-Agent Traffic Flow Control via Graph Attention Networks and LWR Models*, Phase 1.

This notebook audits the METR-LA dataset and explores the temporal and spatial structure that the forecasting
models must exploit. Figures are produced by `scripts/run_eda.py` (run it first); the tables below are recomputed live.

**Dataset.** 207 loop detectors on Los Angeles highways, average speed (mph) every 5 minutes, 1 Mar – 27 Jun 2012
(34 272 timestamps), plus a directed road-distance sensor graph. Released with DCRNN (Li et al., ICLR 2018).
"""),
        code(SETUP),
        md("## 1. Load the data\n`load_all()` returns the speed table with outages masked as NaN, the raw table, and the sensor graph."),
        code('''speed, raw, graph = data.load_all()
print("speed table:", speed.shape, "| from", speed.index[0], "to", speed.index[-1])
print("sampling interval:", pd.infer_freq(speed.index[:50]))
print("sensor graph: %d sensors, %d directed edges" % (graph.n, (graph.adjacency > 0).sum() - graph.n))
speed.iloc[:5, :6]'''),
        md("""## 2. Data-quality audit

A reading of exactly **0 mph is a sensor outage**, not a traffic jam (a highway never averages 0 mph for a full 5-minute
window). We mask these values and never use them as features or targets. Below: how much is missing, and how the
outages are distributed over sensors and time."""),
        code('''summary = json.load(open(C.RESULTS_DIR / "eda_summary.json"))
keys = ["frac_zero_readings", "frac_timestamps_fully_missing", "sensors_missing_gt_20pct", "worst_sensor_missing",
        "outage_runs", "outage_run_median_steps", "outage_run_p90_steps", "outage_run_max_steps", "frac_outage_steps_in_runs_ge_1day",
        "n_missing_timestamps_in_index"]
pd.Series({k: summary[k] for k in keys}, name="value").to_frame()'''),
        code('fig("fig_eda_missing")'),
        md("""**Reading the figure.** Outages are bursty: most runs are a few steps long, but a handful of day-long,
network-wide outages (spikes on the left) carry a quarter of all missing values. They occur in every split, so masking
(rather than dropping days) is the right treatment, and the models must cope with missing lags."""),
        md("## 3. Speed distribution\nThe distribution is **bimodal** — a free-flow mode near 65 mph and a congested mode below 35 mph — the two regimes of the fundamental diagram."),
        code('''obs = speed.values[~np.isnan(speed.values)]
pd.Series(obs).describe(percentiles=[.05, .25, .5, .75, .95]).to_frame("observed speed (mph)").T'''),
        code('fig("fig_eda_speed_distribution")'),
        md("## 4. Temporal structure\nTwo weekday commute peaks (evening deeper than morning), flat weekends, a strong one-hour memory and a 24-hour cycle."),
        code('fig("fig_eda_daily_profile"); fig("fig_eda_weekly_heatmap"); fig("fig_eda_autocorrelation"); fig("fig_eda_congestion_by_hour")'),
        code('''# share of congested sensors by hour (weekdays)
X = speed.values; idx = speed.index
cong = np.where(np.isnan(X), np.nan, (X < C.CONGESTION_THRESHOLD).astype(float))
share = pd.DataFrame({"hour": idx.hour, "weekday": idx.dayofweek < 5, "congested": np.nanmean(cong, axis=1)})
(share.groupby(["weekday", "hour"]).congested.mean().unstack(0) * 100).round(1).T'''),
        md("## 5. Spatial structure\nCorrelation between sensors decays with **road-network distance** — strong but local dependence, which motivates neighbour features now and a graph neural network in Phase 2."),
        code('fig("fig_eda_sensor_map"); fig("fig_eda_spatial_correlation"); fig("fig_eda_graph")'),
        code('''pd.Series({k: summary[k] for k in ["corr_median_lt1km", "corr_median_5to6km", "corr_median_10to11km",
                                     "graph_edges", "graph_mean_out_degree", "graph_density", "graph_isolated_sensors"]}, name="value").to_frame()'''),
        md("## 6. Per-sensor table and chronological split"),
        code('''sensor_tbl = pd.read_csv(C.RESULTS_DIR / "eda_sensor_table.csv")
sensor_tbl.describe().T[["mean", "min", "50%", "max"]]'''),
        code('''split = F.chronological_split(len(speed))
display(split.summary(speed.index))
fig("fig_eda_split_timeline"); fig("fig_eda_example_week")'''),
        md("""## 7. Take-aways for modelling

1. **Mask outages** (NaN), never treat them as 0 mph; tree models handle NaN natively.
2. **Lag features** over the last hour capture the strong short-term memory; **cyclic calendar features** and a
   **train-only historical-average profile** capture the 24 h / weekly cycle.
3. **Neighbour features** from the road graph capture the local spatial dependence; the decay with distance says the
   first-hop neighbourhood is the right place to start.
4. **Chronological split** with a one-hour gap; evaluate separately in **peak** periods where congestion lives.
"""),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def model_notebook():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 02 — Feature engineering, models and evaluation

**Project 4 · Team T082 · Section B** — Phase 1 (classical ML).

This notebook (1) builds the feature set, (2) trains every model class on **one sensor** live so that the method is
transparent, and (3) loads the full 207-sensor results produced by `scripts/run_experiments.py` and displays the
evaluation tables and figures.
"""),
        code(SETUP),
        md("## 1. Build features\nAll features are derived from the NaN-masked speed table and the sensor graph. Statistics that could leak the future (historical-average profile) are fitted on training timestamps only."),
        code('''speed, raw, graph = data.load_all()
fs = F.build_features(speed, graph)
print("lag tensor", fs.lag.shape, "| window stats", fs.stats.shape, "| calendar", fs.time.shape, "| neighbour", fs.nbr.shape, "| targets", fs.targets.shape)
display(fs.split.summary(fs.index))
print(len(fs.local_feature_names()), "local features;", len(fs.spatial_feature_names()), "spatial features")'''),
        code('''# the design matrix of one sensor at a few timestamps (spatial variant)
n = 0
Xn = pd.DataFrame(fs.sensor_matrix(n, spatial=True), columns=fs.spatial_feature_names(), index=fs.index)
Xn.iloc[5000:5005].T'''),
        md("""### Leakage checks
* targets are strictly in the future of every feature (`lag0` is the current speed, the first target is +15 min);
* the historical-average profile is estimated on `train_end` timestamps only;
* a 12-step gap separates train / validation / test so no training target overlaps a test feature window."""),
        code('''X = speed.values
assert np.allclose(np.nan_to_num(fs.lag[:, :, 0]), np.nan_to_num(X), atol=1e-4)
assert np.allclose(np.nan_to_num(fs.targets[:-12, :, 2]), np.nan_to_num(X[12:]), atol=1e-4)
assert fs.split.val[0] - fs.split.train[-1] > 12 and fs.split.test[0] - fs.split.val[-1] > 12
print("leakage checks passed")'''),
        md("## 2. Train every model class on one sensor (live demo)\nThe full experiment loops this over all 207 sensors; here we show it for the most congested sensor so the mechanics are visible."),
        code('''import time
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

s = int(np.nanargmin(np.nanmean(X[: fs.split.train_end], axis=0)))
print("sensor", graph.sensor_ids[s], "(most congested)")
tr, va, te = fs.split.train, fs.split.val, fs.split.test
Y = fs.targets[:, s, :]; hz = fs.names["targets"]
rows = []
for spatial in (False, True):
    Xs = fs.sensor_matrix(s, spatial)
    tag = "spatial" if spatial else "local"
    # XGBoost, one model per horizon with early stopping on the validation split
    t0 = time.time(); pred = np.full((len(te), 3), np.nan)
    for h in range(3):
        ok = tr[~np.isnan(Y[tr, h])]; okv = va[~np.isnan(Y[va, h])]
        p = C.XGBParams()
        m = XGBRegressor(**p.to_dict(), early_stopping_rounds=p.early_stopping_rounds, eval_metric="mae", verbosity=0)
        m.fit(Xs[ok], Y[ok, h], eval_set=[(Xs[okv], Y[okv, h])], verbose=False)
        pred[:, h] = m.predict(Xs[te])
    rows.append({"model": f"XGB-{tag}", **{f"MAE {hz[h]}": E.masked_mae(Y[te, h], pred[:, h]) for h in range(3)}, "seconds": time.time() - t0})
    # multi-output random forest
    t0 = time.time(); ok = tr[~np.isnan(Y[tr]).any(axis=1)]
    rf = RandomForestRegressor(**C.RFParams().to_dict()).fit(Xs[ok], Y[ok]); pr = rf.predict(Xs[te])
    rows.append({"model": f"RF-{tag}", **{f"MAE {hz[h]}": E.masked_mae(Y[te, h], pr[:, h]) for h in range(3)}, "seconds": time.time() - t0})
    # ridge
    t0 = time.time(); Xi = fs.impute(Xs, s); sc = StandardScaler().fit(Xi[tr]); Z = sc.transform(Xi)
    pr = np.column_stack([Ridge(alpha=1.0).fit(Z[tr[~np.isnan(Y[tr, h])]], Y[tr[~np.isnan(Y[tr, h])], h]).predict(Z[te]) for h in range(3)])
    rows.append({"model": f"Ridge-{tag}", **{f"MAE {hz[h]}": E.masked_mae(Y[te, h], pr[:, h]) for h in range(3)}, "seconds": time.time() - t0})
for name, out in (("HA", M.historical_average(fs)), ("Persistence", M.persistence(fs))):
    rows.append({"model": name, **{f"MAE {hz[h]}": E.masked_mae(Y[te, h], out.test_pred[:, s, h]) for h in range(3)}, "seconds": 0})
pd.DataFrame(rows).set_index("model").round(3)'''),
        md("## 3. Full results (all 207 sensors)\nProduced by `python scripts/run_experiments.py`; metrics are masked (outage targets excluded) and computed on the test split (4–27 June 2012)."),
        code('''hm = pd.read_csv(C.RESULTS_DIR / "metrics_horizon.csv")
display(Markdown("**Test MAE (mph)**")); display(hm.pivot(index="model", columns="horizon", values="MAE").round(3))
display(Markdown("**Test RMSE (mph)**")); display(hm.pivot(index="model", columns="horizon", values="RMSE").round(3))
display(Markdown("**Test MAPE (%)**")); display(hm.pivot(index="model", columns="horizon", values="MAPE").round(2))
display(Markdown("**Training wall-clock (s)**")); display(hm.groupby("model").fit_seconds.first().round(0).to_frame())'''),
        code('fig("fig_res_mae_by_horizon")'),
        md("### Validation split (used for early stopping / model selection)\nThe ranking on validation matches the ranking on test, so test numbers are not the product of test-set tuning."),
        code('''vm = pd.read_csv(C.RESULTS_DIR / "metrics_validation.csv")
vm.pivot(index="model", columns="horizon", values="MAE").round(3)'''),
        md("### Peak vs off-peak\nWeekday 07–10 h and 16–19 h target times vs everything else."),
        code('''pk = pd.read_csv(C.RESULTS_DIR / "metrics_peak_offpeak.csv")
display(pk[pk.horizon == "60min"].pivot(index="model", columns="period", values=["MAE", "RMSE", "MAPE"]).round(3))
fig("fig_res_peak_offpeak"); fig("fig_res_mae_by_hour")'''),
        md("### The spatial ablation (the 'gap')\nSame learner, with vs without the eight neighbour features."),
        code('''wide = hm[hm.horizon != "average"].pivot(index="model", columns="horizon", values="MAE")
for a, b in (("XGB-local", "XGB-spatial"), ("RF-local", "RF-spatial")):
    if a in wide.index and b in wide.index:
        print(f"{a} -> {b}: MAE reduction (%)", ((wide.loc[a] - wide.loc[b]) / wide.loc[a] * 100).round(1).to_dict())
fig("fig_res_sensor_mae_map", 900); fig("fig_res_feature_importance")'''),
        code('''imp = pd.read_csv(C.RESULTS_DIR / "feature_importance.csv")
imp[imp.model == "XGB-spatial"].head(12)'''),
        md("### Congestion detection and congestion maps\nEvent: speed < 35 mph (Caltrans PeMS congestion threshold)."),
        code('''cg = pd.read_csv(C.RESULTS_DIR / "metrics_congestion.csv")
display(cg.pivot(index="model", columns="horizon", values="F1").round(3))
fig("fig_res_congestion_f1"); fig("fig_res_congestion_map", 950); fig("fig_res_example_forecast"); fig("fig_res_pred_vs_actual")'''),
        md("""## 4. Conclusions

* Per-segment XGBoost is the strongest independent-segment model; the multi-output Random Forest is close.
* Adding LWR-motivated **neighbour features** improves every horizon, about three times more in the peaks than
  off-peak; the relative gain is largest at 15–30 min and fades at 60 min, as one-hop features should — a direct
  measurement of what independent-segment models miss, and the gap the multi-hop Phase 2 Graph Attention Network targets.
* Remaining error is concentrated in the congested regime (< 35 mph) and in the commute peaks.
"""),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


if __name__ == "__main__":
    NB_DIR.mkdir(exist_ok=True)
    nbf.write(eda_notebook(), NB_DIR / "01_eda.ipynb")
    nbf.write(model_notebook(), NB_DIR / "02_features_and_models.ipynb")
    print("notebooks written to", NB_DIR)
