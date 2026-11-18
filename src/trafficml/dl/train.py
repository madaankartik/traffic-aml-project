"""Training, the LWR physics penalty and prediction for the Phase 2 models."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from .. import config as C
from .data import GraphData, batch_iter
from .models import STNet, random_graph_like

HORIZON_HOURS = torch.tensor([s * C.STEP_MINUTES / 60.0 for s in C.HORIZONS.values()])   # 0.25, 0.5, 1.0


@dataclass
class RunConfig:
    name: str = "ST-GAT"
    spatial: str = "gat"             # gat | gcn | none
    n_layers: int = 2                # = number of hops of message passing
    directed: bool = True
    graph: str = "road"              # road | random
    base: str = "mean"               # mean (pure DL) | xgb (hybrid residual)
    lwr_weight: float = 0.0          # physics penalty weight (0 = off)
    zero_fill: bool = False          # ablation: hide the mask and treat outages as 0 mph
    hidden: int = 64
    heads: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 30
    samples_per_epoch: int = 8000   # random training windows per epoch (consecutive windows are near-duplicates)
    patience: int = 5
    grad_clip: float = 5.0
    seed: int = 42
    extra: dict = field(default_factory=dict)


def masked_mae_t(pred, y):
    m = ~torch.isnan(y)
    return (pred[m] - y[m]).abs().mean()


class LWRPenalty:
    """Soft conservation-of-vehicles check on the forecast (Lighthill-Whitham-Richards).

    Greenshields: normalised density k = 1 - v / v_f  (k in [0,1], 1 = jam), flow q = v_f k (1-k).
    For sensor i and horizon h the discrete conservation law over dt = h hours is
        k_i(t+h) - k_i(t) + dt * (q_i - q_in,i) / dx_i  ~  0
    with q_in,i the road-graph-weighted flow of i's upstream neighbours and dx_i their mean
    road distance (miles).  The penalty is the Huber loss of that residual, averaged over
    sensors that have upstream neighbours.  It does not use the targets: it only asks the
    forecast to be physically plausible, so it acts as a regulariser.
    """

    def __init__(self, gd: GraphData):
        tr = gd.split.train_end
        X = gd.x_all[:tr, :, 0] * gd.std + gd.mean
        X = np.where(gd.x_all[:tr, :, 1] > 0, X, np.nan)
        self.vf = torch.from_numpy(np.nanpercentile(X, 95, axis=0).astype(np.float32))
        A = gd.adjacency.copy(); np.fill_diagonal(A, 0.0)
        Win = A.T                                            # Win[i, j] > 0: j upstream of i
        rs = Win.sum(1, keepdims=True); has = (rs[:, 0] > 0)
        rs[rs == 0] = 1.0
        self.P = torch.from_numpy((Win / rs).astype(np.float32))
        d = np.where(Win > 0, gd.distances_km.T, 0.0)
        dx = (d * Win).sum(1) / np.maximum(Win.sum(1), 1e-6) / 1.609          # miles
        dx[~has] = 1.0
        self.dx = torch.from_numpy(np.maximum(dx, 0.3).astype(np.float32))
        self.has = torch.from_numpy(has)
        self.mean = torch.from_numpy(gd.mean); self.std = torch.from_numpy(gd.std)

    def __call__(self, pred_mph, x):
        v0 = x[:, -1, :, 0] * self.std + self.mean                           # current speed (scaled-back)
        known = x[:, -1, :, 1] > 0
        v0 = torch.where(known, v0, pred_mph[:, :, 0])
        vf = self.vf.view(1, -1)
        k0 = (1 - v0 / vf).clamp(0, 1)
        loss = 0.0
        for h in range(pred_mph.shape[2]):
            k = (1 - pred_mph[:, :, h] / vf).clamp(0, 1)
            q = vf * k * (1 - k)
            q_in = torch.einsum("ij,bj->bi", self.P, q)
            r = (k - k0) + HORIZON_HOURS[h] * (q - q_in) / self.dx.view(1, -1)
            loss = loss + torch.nn.functional.huber_loss(r[:, self.has], torch.zeros_like(r[:, self.has]), delta=0.1)
        return loss / pred_mph.shape[2]


def build_model(cfg: RunConfig, gd: GraphData) -> STNet:
    adj = gd.adjacency if cfg.graph == "road" else random_graph_like(gd.adjacency, cfg.seed)
    return STNet(len(gd.sensor_ids), adj, cfg.spatial, cfg.n_layers, cfg.directed, hidden=cfg.hidden,
                 heads=cfg.heads, dropout=cfg.dropout)


def _prep_x(x, cfg):
    if cfg.zero_fill:                       # ablation: outage = 0 mph, no mask channel
        x = x.clone()
        z0 = (0.0 - prep_stats["mean"]) / prep_stats["std"]
        miss = x[..., 1] == 0
        x[..., 0] = torch.where(miss, z0.expand_as(x[..., 0]), x[..., 0])
        x[..., 1] = 1.0
    return x


prep_stats: dict = {}


def to_mph(z, base, std):
    return base + z * std.view(1, -1, 1)


def predict(model, gd, idx, base, cfg, batch_size=128):
    model.eval()
    std = torch.from_numpy(gd.std)
    out = np.zeros((len(idx), len(gd.sensor_ids), 3), np.float32)
    with torch.no_grad():
        for x, _, b, sel in batch_iter(gd, idx, base, batch_size, shuffle=False):
            out[sel] = to_mph(model(_prep_x(x, cfg)), b, std).numpy()
    return out


def train_run(cfg: RunConfig, gd: GraphData, bases: dict, log=print) -> dict:
    """Train with early stopping on validation masked MAE; return predictions and history."""
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    prep_stats["mean"] = torch.from_numpy(gd.mean).view(1, 1, -1)
    prep_stats["std"] = torch.from_numpy(gd.std).view(1, 1, -1)
    model = build_model(cfg, gd)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    std = torch.from_numpy(gd.std)
    lwr = LWRPenalty(gd) if cfg.lwr_weight > 0 else None
    tr, va, te = gd.split.train, gd.split.val, gd.split.test
    best, best_state, bad, hist = np.inf, None, 0, []
    t0 = time.time()
    for ep in range(cfg.max_epochs):
        model.train(); tot, nb = 0.0, 0
        sub = rng.choice(len(tr), size=min(cfg.samples_per_epoch, len(tr)), replace=False)
        for x, y, b, _ in batch_iter(gd, tr[sub], bases["train"][sub], cfg.batch_size, True, rng):
            pred = to_mph(model(_prep_x(x, cfg)), b, std)
            loss = masked_mae_t(pred, y)
            if lwr is not None:
                loss = loss + cfg.lwr_weight * lwr(pred, x)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step(); tot += float(loss); nb += 1
        vp = predict(model, gd, va, bases["val"], cfg)
        yv = gd.y_all[va]; m = ~np.isnan(yv)
        vmae = float(np.abs(vp[m] - yv[m]).mean())
        sched.step(vmae)
        hist.append({"epoch": ep + 1, "train_loss": tot / nb, "val_mae": vmae, "lr": opt.param_groups[0]["lr"],
                     "seconds": time.time() - t0})
        log(f"  [{cfg.name}] epoch {ep+1:2d} train {tot/nb:.3f}  val MAE {vmae:.3f}  {time.time()-t0:.0f}s")
        if vmae < best - 1e-4:
            best, bad = vmae, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return {"model": model, "val_pred": predict(model, gd, va, bases["val"], cfg),
            "test_pred": predict(model, gd, te, bases["test"], cfg), "history": hist,
            "best_val_mae": best, "n_params": n_params, "fit_seconds": time.time() - t0,
            "config": asdict(cfg)}


def save_run(res: dict, directory=C.RESULTS_DIR / "dl"):
    directory.mkdir(parents=True, exist_ok=True)
    name = res["config"]["name"]
    np.savez_compressed(directory / f"pred_{name}.npz", val_pred=res["val_pred"], test_pred=res["test_pred"])
    meta = {k: res[k] for k in ("history", "best_val_mae", "n_params", "fit_seconds", "config")}
    (directory / f"run_{name}.json").write_text(json.dumps(meta, indent=1))
    torch.save(res["model"].state_dict(), directory / f"model_{name}.pt")
