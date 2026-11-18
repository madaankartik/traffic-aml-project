"""Out-of-fold XGBoost-spatial predictions on the TRAINING weeks (input to the hybrid model).

The hybrid (XGBoost + ST-GAT residual) needs XGBoost forecasts for every training sample.
In-sample predictions would be unrealistically accurate, so the residual network would learn
from errors that never occur on new data.  Instead the training period is cut into K
contiguous time blocks; for each block, per-sensor/per-horizon XGBoost models are fit on the
other blocks (with a one-hour gap either side) and predict the held-out block.  The number of
trees per model is the one chosen by early stopping in Phase 1 (results/pred_XGB-spatial.npz),
so the validation and test weeks are never used here.  Validation/test forecasts come from the
Phase 1 model unchanged.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from xgboost import XGBRegressor
from trafficml import config as C, data as D, features as F

K = 3

def main():
    X_df, _, graph = D.load_all()
    fs = F.build_features(X_df, graph)
    tr = fs.split.train
    p1 = np.load(C.RESULTS_DIR / "pred_XGB-spatial.npz", allow_pickle=True)
    best = p1["best_iterations"]                       # (N, H)
    H = fs.targets.shape[2]
    oof = np.full((len(tr), fs.n_sensors, H), np.nan, np.float32)
    blocks = np.array_split(np.arange(len(tr)), K)
    gap = C.GAP_STEPS
    params = C.XGBParams().to_dict()
    t0 = time.time()
    for n in range(fs.n_sensors):
        Xn = fs.sensor_matrix(n, spatial=True)
        Y = fs.targets[:, n, :]
        for k, b in enumerate(blocks):
            lo, hi = tr[b[0]], tr[b[-1]]
            fit_idx = tr[(tr < lo - gap) | (tr > hi + gap)]
            for h in range(H):
                ok = fit_idx[~np.isnan(Y[fit_idx, h])]
                if len(ok) < 50:
                    continue
                prm = dict(params); prm["n_estimators"] = int(max(best[n, h], 20))
                m = XGBRegressor(**prm, verbosity=0).fit(Xn[ok], Y[ok, h])
                oof[b, n, h] = m.predict(Xn[tr[b]])
        if (n + 1) % 10 == 0:
            print(f"sensor {n+1}/{fs.n_sensors}  {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(C.RESULTS_DIR / "xgb_spatial_oof_train.npz", train_pred=oof, train_idx=tr)
    print("done", time.time() - t0)

if __name__ == "__main__":
    main()
