"""Phase 1 forecasting models.

Every model maps the engineered features of one road segment (sensor) at time t
to the speed at t+15, t+30 and t+60 minutes.  Models are trained *per segment*
(one model per sensor) as prescribed by the project brief; the Random Forest is
a true multi-output model (all three horizons in one forest), the others fit one
regressor per horizon.

Models
------
HA          historical average of the target's time-of-week slot (train only)
Persistence last observed speed is carried forward ("naive" / random-walk)
Ridge       linear regression on standardised, HA-imputed features
XGB-local   per-segment gradient-boosted trees, own-segment features only
RF-local    per-segment multi-output Random Forest, own-segment features
XGB-spatial XGB-local + upstream/downstream neighbour features
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config as C
from .features import FeatureSet


@dataclass
class ModelOutput:
    """Predictions on the validation and test samples plus diagnostics."""

    name: str
    val_pred: np.ndarray            # (S_val, N, H)
    test_pred: np.ndarray           # (S_test, N, H)
    feature_names: list = field(default_factory=list)
    importances: np.ndarray | None = None     # (N, H, F) gain importances (XGB) or (N, F) (RF)
    best_iterations: np.ndarray | None = None # (N, H)
    fit_seconds: float = 0.0


def _empty(fs: FeatureSet):
    H = fs.targets.shape[2]
    return (np.full((len(fs.split.val), fs.n_sensors, H), np.nan, np.float32),
            np.full((len(fs.split.test), fs.n_sensors, H), np.nan, np.float32))


# --------------------------------------------------------------------------- #
# Naive baselines
# --------------------------------------------------------------------------- #
def historical_average(fs: FeatureSet) -> ModelOutput:
    return ModelOutput("HA", fs.ha_target[fs.split.val], fs.ha_target[fs.split.test])


def persistence(fs: FeatureSet) -> ModelOutput:
    """Carry the most recent *observed* speed forward; fall back to HA if the
    whole one-hour window is missing."""
    L = fs.lag                                      # (T, N, L) lag0 = now
    last = np.full(L.shape[:2], np.nan, np.float32)
    for k in range(L.shape[2] - 1, -1, -1):         # overwrite with more recent valid values
        v = L[:, :, k]
        last = np.where(np.isnan(v), last, v)
    last = np.where(np.isnan(last), fs.ha, last)
    H = fs.targets.shape[2]
    rep = np.repeat(last[:, :, None], H, axis=2)
    return ModelOutput("Persistence", rep[fs.split.val], rep[fs.split.test])


# --------------------------------------------------------------------------- #
# Per-segment learners
# --------------------------------------------------------------------------- #
def _log(i, n, t0, every=20):
    if (i + 1) % every == 0 or i + 1 == n:
        print(f"    sensor {i + 1:3d}/{n}  elapsed {time.time() - t0:6.0f}s", flush=True)


def ridge_per_segment(fs: FeatureSet, spatial: bool = False, params: C.RidgeParams | None = None) -> ModelOutput:
    params = params or C.RidgeParams()
    name = "Ridge-spatial" if spatial else "Ridge-local"
    val_pred, test_pred = _empty(fs)
    tr, va, te = fs.split.train, fs.split.val, fs.split.test
    t0 = time.time()
    for n in range(fs.n_sensors):
        X = fs.impute(fs.sensor_matrix(n, spatial), n)
        Y = fs.targets[:, n, :]
        scaler = StandardScaler().fit(X[tr])
        Xs = scaler.transform(X)
        for h in range(Y.shape[1]):
            ok = tr[~np.isnan(Y[tr, h])]
            if len(ok) < 50:
                continue
            m = Ridge(alpha=params.alpha).fit(Xs[ok], Y[ok, h])
            val_pred[:, n, h] = m.predict(Xs[va])
            test_pred[:, n, h] = m.predict(Xs[te])
        _log(n, fs.n_sensors, t0, every=50)
    names = fs.spatial_feature_names() if spatial else fs.local_feature_names()
    return ModelOutput(name, val_pred, test_pred, names, fit_seconds=time.time() - t0)


def xgb_per_segment(fs: FeatureSet, spatial: bool = False, params: C.XGBParams | None = None) -> ModelOutput:
    """One XGBoost regressor per (sensor, horizon) with early stopping on the validation split."""
    params = params or C.XGBParams()
    name = "XGB-spatial" if spatial else "XGB-local"
    val_pred, test_pred = _empty(fs)
    tr, va, te = fs.split.train, fs.split.val, fs.split.test
    names = fs.spatial_feature_names() if spatial else fs.local_feature_names()
    H = fs.targets.shape[2]
    importances = np.zeros((fs.n_sensors, H, len(names)), np.float32)
    best_iter = np.zeros((fs.n_sensors, H), int)
    t0 = time.time()
    for n in range(fs.n_sensors):
        X = fs.sensor_matrix(n, spatial)
        Y = fs.targets[:, n, :]
        for h in range(H):
            ok_tr = tr[~np.isnan(Y[tr, h])]
            ok_va = va[~np.isnan(Y[va, h])]
            if len(ok_tr) < 50 or len(ok_va) < 20:
                continue
            model = XGBRegressor(**params.to_dict(), early_stopping_rounds=params.early_stopping_rounds,
                                 eval_metric="mae", verbosity=0)
            model.fit(X[ok_tr], Y[ok_tr, h], eval_set=[(X[ok_va], Y[ok_va, h])], verbose=False)
            val_pred[:, n, h] = model.predict(X[va])
            test_pred[:, n, h] = model.predict(X[te])
            best_iter[n, h] = model.best_iteration
            booster = model.get_booster()
            gain = booster.get_score(importance_type="gain")
            for j in range(len(names)):
                importances[n, h, j] = gain.get(f"f{j}", 0.0)
        _log(n, fs.n_sensors, t0)
    return ModelOutput(name, val_pred, test_pred, names, importances, best_iter, time.time() - t0)


def rf_per_segment(fs: FeatureSet, spatial: bool = False, params: C.RFParams | None = None) -> ModelOutput:
    """One *multi-output* Random Forest per sensor predicting all horizons jointly."""
    params = params or C.RFParams()
    name = "RF-spatial" if spatial else "RF-local"
    val_pred, test_pred = _empty(fs)
    tr, va, te = fs.split.train, fs.split.val, fs.split.test
    names = fs.spatial_feature_names() if spatial else fs.local_feature_names()
    importances = np.zeros((fs.n_sensors, len(names)), np.float32)
    t0 = time.time()
    for n in range(fs.n_sensors):
        X = fs.sensor_matrix(n, spatial)
        Y = fs.targets[:, n, :]
        ok_tr = tr[~np.isnan(Y[tr]).any(axis=1)]      # multi-output: need all horizons observed
        if len(ok_tr) < 50:
            continue
        model = RandomForestRegressor(**params.to_dict())
        model.fit(X[ok_tr], Y[ok_tr])
        val_pred[:, n, :] = model.predict(X[va])
        test_pred[:, n, :] = model.predict(X[te])
        importances[n] = model.feature_importances_
        _log(n, fs.n_sensors, t0)
    return ModelOutput(name, val_pred, test_pred, names, importances, None, time.time() - t0)


# --------------------------------------------------------------------------- #
# Persistence of outputs
# --------------------------------------------------------------------------- #
def save_output(out: ModelOutput, directory=C.RESULTS_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / f"pred_{out.name}.npz", val_pred=out.val_pred, test_pred=out.test_pred,
                        feature_names=np.array(out.feature_names),
                        importances=out.importances if out.importances is not None else np.array([]),
                        best_iterations=out.best_iterations if out.best_iterations is not None else np.array([]),
                        fit_seconds=out.fit_seconds)


def load_output(name: str, directory=C.RESULTS_DIR) -> ModelOutput:
    z = np.load(directory / f"pred_{name}.npz", allow_pickle=False)
    imp = z["importances"]; bi = z["best_iterations"]
    return ModelOutput(name, z["val_pred"], z["test_pred"], list(z["feature_names"]),
                       imp if imp.size else None, bi if bi.size else None, float(z["fit_seconds"]))


def importance_table(out: ModelOutput) -> pd.DataFrame:
    """Mean normalised importance per feature across sensors (and horizons for XGB)."""
    imp = out.importances
    if imp is None:
        return pd.DataFrame()
    if imp.ndim == 3:                       # (N, H, F) -> normalise within each model
        s = imp.sum(axis=2, keepdims=True); s[s == 0] = 1
        imp = (imp / s).mean(axis=(0, 1))
    else:
        imp = imp.mean(axis=0)
    return pd.DataFrame({"feature": out.feature_names, "importance": imp}).sort_values("importance", ascending=False)
