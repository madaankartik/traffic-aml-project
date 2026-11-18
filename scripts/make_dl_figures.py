"""Phase 2 figures (figures/fig_dl_*.png) and the attention analysis (results/dl/attention_analysis.csv)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficml import config as C
from trafficml.plots import (CATEGORICAL, CMAP_DIV, CMAP_SEQ_R, INK2, MODEL_COLOURS, MUTED, plt, save,
                             set_style, hide_spines)
from trafficml.dl.data import build_graph_data, batch_iter
from trafficml.dl.models import EdgeGATLayer
from trafficml.dl.train import RunConfig, build_model, _prep_x, prep_stats

DL = C.RESULTS_DIR / "dl"
HZ = list(C.HORIZONS)
COL = dict(MODEL_COLOURS)
COL.update({"GRU-nograph": "#9ec5f4", "DCRNN-lite": "#1baf7a", "ST-GAT": "#0d366b", "Hybrid-XGB-GAT": "#e34948",
            "ST-GAT-LWR": "#6250d6"})
MAIN = ["HA", "Persistence", "XGB-spatial", "RF-spatial", "GRU-nograph", "DCRNN-lite", "ST-GAT", "ST-GAT-LWR", "Hybrid-XGB-GAT"]
LABEL = {"HA": "Hist. average", "Persistence": "Persistence", "XGB-spatial": "XGBoost + nbrs (P1)",
         "RF-spatial": "Random Forest + nbrs (P1)", "GRU-nograph": "GRU, no graph", "DCRNN-lite": "Diffusion GCN (fixed adj.)",
         "ST-GAT": "ST-GAT", "ST-GAT-LWR": "ST-GAT + LWR loss", "Hybrid-XGB-GAT": "Hybrid XGB + ST-GAT"}


def attention_analysis(gd):
    """Where does the network look when a sensor is jammed vs free-flowing?"""
    rows = []
    sp = gd.split
    prep_stats["mean"] = torch.from_numpy(gd.mean).view(1, 1, -1); prep_stats["std"] = torch.from_numpy(gd.std).view(1, 1, -1)
    A = gd.adjacency.copy(); np.fill_diagonal(A, 0)
    for name, directed in (("ST-GAT", True), ("ST-GAT-undirected", False)):
        f = DL / f"model_{name}.pt"
        if not f.exists():
            continue
        cfg = RunConfig(name=name, spatial="gat", directed=directed)
        model = build_model(cfg, gd); model.load_state_dict(torch.load(f)); model.eval()
        layer0 = model.blocks[0].rel
        acc = {}
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(sp.test, 1500, replace=False))
        with torch.no_grad():
            for x, y, b, sel in batch_iter(gd, idx, None, 100, False):
                model(_prep_x(x, cfg))
                v_now = x[:, -1, :, 0] * torch.from_numpy(gd.std) + torch.from_numpy(gd.mean)
                known = x[:, -1, :, 1] > 0
                jam = (v_now < C.CONGESTION_THRESHOLD) & known
                free = (v_now >= 50) & known
                for r, rel in enumerate(layer0):
                    a = rel.last_alpha_full.mean(-1)                         # (B, E)
                    dst, src = rel.dst.numpy(), rel.src.numpy()
                    kind = np.where(dst == src, "self", np.where(A[dst, src] > 0, "downstream", "upstream"))
                    if directed:
                        kind = np.where(kind == "self", "self", "downstream" if r == 0 else "upstream")
                    for state, msk in (("jammed", jam), ("free", free)):
                        mm = msk[:, dst].numpy()                             # (B, E) node i state
                        for k in ("self", "downstream", "upstream"):
                            sel_e = kind == k
                            w = (a.numpy()[:, sel_e] * mm[:, sel_e]).sum()
                            key = (name, f"rel{r}", state, k)
                            acc[key] = acc.get(key, 0.0) + w
                        acc[(name, f"rel{r}", state, "_nodes")] = acc.get((name, f"rel{r}", state, "_nodes"), 0) + mm[:, np.flatnonzero(dst == src)].sum()
        for (n, rel, state, k), w in acc.items():
            if k == "_nodes":
                continue
            rows.append({"model": n, "relation": rel, "state": state, "edge_type": k,
                         "attention_share": w / max(acc[(n, rel, state, "_nodes")], 1)})
    df = pd.DataFrame(rows)
    df.to_csv(DL / "attention_analysis.csv", index=False)
    return df


def main():
    set_style()
    gd = build_graph_data()
    hor = pd.read_csv(DL / "metrics_horizon.csv"); pk = pd.read_csv(DL / "metrics_peak_offpeak.csv")
    models = [m for m in MAIN if m in set(hor.model)]

    # 1. MAE by horizon, main models
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.8 / len(models)
    for k, m in enumerate(models):
        v = hor[hor.model == m].set_index("horizon").loc[HZ, "MAE"].values
        ax.bar(np.arange(3) + (k - len(models) / 2 + 0.5) * w, v, w * 0.92, color=COL.get(m, MUTED), label=LABEL.get(m, m))
    ax.set_xticks(range(3)); ax.set_xticklabels(["15 min", "30 min", "60 min"]); ax.set_ylabel("Test MAE (mph)")
    ax.set_ylim(2.4, None); ax.set_title("Test MAE by horizon: Phase 1 ML vs Phase 2 DL")
    ax.legend(ncol=3, loc="upper left", fontsize=7); hide_spines(ax)
    save(fig, "fig_dl_mae_by_horizon")

    # 2. change vs XGB-spatial by horizon, overall / peak / off-peak
    ref = hor[hor.model == "XGB-spatial"].set_index("horizon")["MAE"]
    refp = pk[pk.model == "XGB-spatial"].set_index(["horizon", "period"])["MAE"]
    dlm = [m for m in ("GRU-nograph", "DCRNN-lite", "ST-GAT", "ST-GAT-LWR", "Hybrid-XGB-GAT") if m in models]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
    for ax, (title, getter) in zip(axes, [
            ("All test hours", lambda m, h: hor[(hor.model == m) & (hor.horizon == h)].MAE.item() / ref[h] - 1),
            ("Weekday rush hour", lambda m, h: pk[(pk.model == m) & (pk.horizon == h) & (pk.period == "peak")].MAE.item() / refp[(h, "peak")] - 1),
            ("Off-peak", lambda m, h: pk[(pk.model == m) & (pk.horizon == h) & (pk.period == "off-peak")].MAE.item() / refp[(h, "off-peak")] - 1)]):
        for k, m in enumerate(dlm):
            v = [100 * getter(m, h) for h in HZ]
            ax.plot([15, 30, 60], v, marker="o", color=COL[m], label=LABEL[m])
        ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks([15, 30, 60]); ax.set_xlabel("Horizon (min)"); ax.set_title(title)
    axes[0].set_ylabel("MAE change vs XGBoost + nbrs (%)\n(below 0 = better)")
    axes[2].legend(fontsize=7, loc="upper right")
    save(fig, "fig_dl_gain_vs_xgb")

    # 3. ablations
    ab = pd.read_csv(DL / "ablations.csv").set_index("model")
    order = [("ST-GAT-short", "ST-GAT (2 hops, directed, masked)"), ("ST-GAT-randomgraph", "random graph, same degree"),
             ("ST-GAT-undirected", "undirected edges"), ("ST-GAT-1hop", "1 hop"), ("ST-GAT-3hop", "3 hops"),
             ("ST-GAT-zerofill", "outages fed as 0 mph"), ("ST-GAT-nodropout", "no dropout / weight decay"),
             ("Hybrid-XGB-nograph", "hybrid XGB + GRU, no graph")]
    order = [(m, l) for m, l in order if m in ab.index]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.38 * len(order) + 1.2), sharey=True)
    for ax, h in zip(axes, ["15min", "60min"]):
        v = [ab.loc[m, h] for m, _ in order]
        cols = ["#0d366b" if m == "ST-GAT-short" else "#9ec5f4" for m, _ in order]
        ax.barh(range(len(order)), v, color=cols, height=0.7)
        for y, val in enumerate(v):
            ax.text(val + 0.005, y, f"{val:.2f}", va="center", fontsize=7, color=INK2)
        ax.axvline(ab.loc["ST-GAT-short", h], color=INK2, lw=0.8, ls="--")
        lo = min(v) - 0.15; ax.set_xlim(lo, max(v) + 0.15); ax.set_title(f"Test MAE at {h[:-3]} min (12-epoch budget)"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels([l for _, l in order]); axes[0].invert_yaxis()
    save(fig, "fig_dl_ablations")

    # 4. training curves
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for m in dlm:
        f = DL / f"run_{m}.json"
        if f.exists():
            h = json.loads(f.read_text())["history"]
            ax.plot([r["epoch"] for r in h], [r["val_mae"] for r in h], marker=".", color=COL[m], label=LABEL[m])
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation MAE (mph, avg of 3 horizons)"); ax.set_title("Training: validation error per epoch (early stopping)")
    ax.legend(fontsize=7); hide_spines(ax)
    save(fig, "fig_dl_training_curves")

    # 5. maps: best DL vs XGB per sensor, and a congestion snapshot
    best = hor[(hor.horizon == "average") & hor.model.isin(dlm)].sort_values("MAE").model.iloc[0]
    psm = pd.read_csv(DL / "per_sensor_mae.csv")
    loc = pd.read_csv(C.LOCATIONS_FILE); loc["sensor_id"] = loc.sensor_id.astype(str); loc = loc.set_index("sensor_id").loc[gd.sensor_ids]
    a = psm[psm.model == "XGB-spatial"]["60min"].values; b = psm[psm.model == best]["60min"].values
    gain = 100 * (a - b) / a; lim = np.nanpercentile(np.abs(gain), 95)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sc = axes[0].scatter(loc.longitude, loc.latitude, c=b, cmap="Blues", s=24, edgecolors="white", linewidths=0.4)
    fig.colorbar(sc, ax=axes[0], shrink=0.85).set_label("Test MAE at 60 min (mph)"); axes[0].set_title(f"{LABEL[best]}: 60-min MAE per sensor")
    sc2 = axes[1].scatter(loc.longitude, loc.latitude, c=gain, cmap=CMAP_DIV.reversed(), vmin=-lim, vmax=lim, s=24, edgecolors="white", linewidths=0.4)
    fig.colorbar(sc2, ax=axes[1], shrink=0.85).set_label("MAE reduction vs XGBoost + nbrs (%)")
    axes[1].set_title(f"Gain over Phase 1 best ({int((gain > 0).sum())}/{len(gain)} sensors improved)")
    for ax in axes: ax.grid(False); ax.set_xticks(np.arange(-118.5, -118.15, 0.1))
    save(fig, "fig_dl_sensor_gain_map")

    sp = gd.split
    t = int(np.flatnonzero(gd.index[sp.test] == pd.Timestamp("2012-06-13 16:30"))[0])
    xgb = np.load(C.RESULTS_DIR / "pred_XGB-spatial.npz", allow_pickle=True)["test_pred"]
    bp = np.load(DL / f"pred_{best}.npz")["test_pred"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), sharex=True, sharey=True)
    for ax, (title, v) in zip(axes, [("Observed speed, 17:30", gd.y_all[sp.test][t, :, 2]),
                                     ("Phase 1: XGBoost + nbrs", xgb[t, :, 2]),
                                     (f"Phase 2: {LABEL[best]}", bp[t, :, 2])]):
        sc = ax.scatter(loc.longitude, loc.latitude, c=v, cmap=CMAP_SEQ_R, vmin=10, vmax=70, s=24, edgecolors="white", linewidths=0.4)
        ax.set_title(title, fontsize=9); ax.grid(False); ax.set_xticks(np.arange(-118.5, -118.15, 0.1))
    fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02).set_label("Speed (mph), darker = congested")
    fig.suptitle("Congestion map, Wed 13 Jun 2012 17:30 (forecasts issued 16:30, 60 min ahead)", fontsize=10, fontweight="bold")
    save(fig, "fig_dl_congestion_map")

    # 6. attention
    att = attention_analysis(gd)
    und = att[att.model == "ST-GAT-undirected"]
    if len(und):
        piv = und.groupby(["state", "edge_type"]).attention_share.sum().unstack()
        piv = piv.div(piv.sum(1), axis=0)[["self", "downstream", "upstream"]].loc[["free", "jammed"]]
        fig, ax = plt.subplots(figsize=(6.5, 2.6))
        left = np.zeros(2)
        for k, c in zip(piv.columns, ["#c3c2b7", "#e34948", "#2a78d6"]):
            ax.barh(["free-flowing (≥50 mph)", "jammed (<35 mph)"], piv[k].values, left=left, color=c, label=k, height=0.6)
            for y, (l, v) in enumerate(zip(left, piv[k].values)):
                if v > 0.06: ax.text(l + v / 2, y, f"{100*v:.0f}%", ha="center", va="center", fontsize=8, color="white")
            left += piv[k].values
        ax.set_xlim(0, 1); ax.set_xlabel("Share of attention (first GAT layer, undirected model)")
        ax.set_title("Where the network looks: itself, downstream or upstream neighbours"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.35))
        ax.grid(False); save(fig, "fig_dl_attention")
    print("best DL model:", best)


if __name__ == "__main__":
    main()
