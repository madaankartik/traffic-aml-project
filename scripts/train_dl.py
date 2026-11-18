"""Train the Phase 2 deep-learning models and ablations.

    python scripts/train_dl.py --runs ST-GAT GRU-nograph            # selected runs
    python scripts/train_dl.py --all                                  # everything in EXPERIMENTS
Predictions go to results/dl/pred_<name>.npz in the same layout as the Phase 1 models
(val_pred / test_pred aligned with the Phase 1 validation / test samples).
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficml import config as C
from trafficml.dl.data import build_graph_data
from trafficml.dl.train import RunConfig, train_run, save_run

torch.set_num_threads(2)

EXPERIMENTS = {
    # main models
    "GRU-nograph":   RunConfig(name="GRU-nograph", spatial="none"),
    "DCRNN-lite":    RunConfig(name="DCRNN-lite", spatial="gcn"),
    "ST-GAT":        RunConfig(name="ST-GAT", spatial="gat"),
    "Hybrid-XGB-GAT": RunConfig(name="Hybrid-XGB-GAT", spatial="gat", base="xgb", lr=5e-4),
    "ST-GAT-LWR":    RunConfig(name="ST-GAT-LWR", spatial="gat", lwr_weight=1.0),
    # ablations of the main model (change exactly one thing).  To fit the CPU budget they all use a
    # shorter, identical training budget (12 epochs) and are compared with ST-GAT-short, the main
    # model trained under the same budget; seeds also use this budget.
    "ST-GAT-short":  RunConfig(name="ST-GAT-short", spatial="gat", max_epochs=12),
    "ST-GAT-1hop":   RunConfig(name="ST-GAT-1hop", spatial="gat", n_layers=1, max_epochs=12),
    "ST-GAT-3hop":   RunConfig(name="ST-GAT-3hop", spatial="gat", n_layers=3, max_epochs=12),
    "ST-GAT-undirected": RunConfig(name="ST-GAT-undirected", spatial="gat", directed=False, max_epochs=12),
    "ST-GAT-randomgraph": RunConfig(name="ST-GAT-randomgraph", spatial="gat", graph="random", max_epochs=12),
    "ST-GAT-zerofill": RunConfig(name="ST-GAT-zerofill", spatial="gat", zero_fill=True, max_epochs=12),
    "ST-GAT-nodropout": RunConfig(name="ST-GAT-nodropout", spatial="gat", dropout=0.0, weight_decay=0.0, max_epochs=12),
    "Hybrid-XGB-nograph": RunConfig(name="Hybrid-XGB-nograph", spatial="none", base="xgb", lr=5e-4, max_epochs=12),
    # seeds of the main model
    "ST-GAT-seed1":  RunConfig(name="ST-GAT-seed1", spatial="gat", seed=1, max_epochs=12),
    "ST-GAT-seed2":  RunConfig(name="ST-GAT-seed2", spatial="gat", seed=2, max_epochs=12),
}


def make_bases(gd):
    N, H = len(gd.sensor_ids), 3
    sp = gd.split
    def mean_base(idx):
        return np.broadcast_to(gd.mean.reshape(1, N, 1), (len(idx), N, H)).astype(np.float32).copy()
    bases = {"mean": {"train": mean_base(sp.train), "val": mean_base(sp.val), "test": mean_base(sp.test)}}
    oof_f = C.RESULTS_DIR / "xgb_spatial_oof_train.npz"
    if oof_f.exists():
        oof = np.load(oof_f)
        assert np.array_equal(oof["train_idx"], sp.train)
        p1 = np.load(C.RESULTS_DIR / "pred_XGB-spatial.npz", allow_pickle=True)
        def fill(a, idx):
            return np.where(np.isnan(a), gd.ha_target[idx], a).astype(np.float32)
        bases["xgb"] = {"train": fill(oof["train_pred"], sp.train), "val": fill(p1["val_pred"], sp.val),
                        "test": fill(p1["test_pred"], sp.test)}
    return bases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()
    names = list(EXPERIMENTS) if a.all else a.runs
    gd = build_graph_data()
    bases = make_bases(gd)
    for n in names:
        cfg = EXPERIMENTS[n]
        if a.max_epochs: cfg.max_epochs = a.max_epochs
        if a.skip_done and (C.RESULTS_DIR / "dl" / f"pred_{n}.npz").exists():
            print("skip", n); continue
        print(f"== {n}", flush=True)
        res = train_run(cfg, gd, bases[cfg.base], log=lambda s: print(s, flush=True))
        save_run(res)
        print(f"== {n} done: best val MAE {res['best_val_mae']:.3f}, {res['n_params']} params, {res['fit_seconds']:.0f}s", flush=True)

if __name__ == "__main__":
    main()
