#!/usr/bin/env python3
"""Turn saved predictions and metric CSVs into result figures and LaTeX tables.

Reads results/*.csv and results/pred_*.npz, writes figures/fig_res_*.png|pdf
and report/tables/*.tex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trafficml import config as C                                   # noqa: E402
from trafficml import data, evaluate as E, features as F, models as M   # noqa: E402
from trafficml.plots import (CATEGORICAL, CMAP_DIV, CMAP_SEQ, CMAP_SEQ_R, GRID, INK2, MODEL_COLOURS,   # noqa: E402
                             MUTED, annotate_bars, hide_spines, save, set_style)

TABLES = C.ROOT / "report" / "tables"
ORDER = ["HA", "Persistence", "Ridge-local", "XGB-local", "RF-local", "XGB-spatial", "RF-spatial"]
HZ = list(C.HORIZONS.keys())


def available(names):
    return [n for n in names if (C.RESULTS_DIR / f"pred_{n}.npz").exists()]


def latex_table(df: pd.DataFrame, path: Path, caption: str, label: str, fmt="%.2f", col_format=None,
                star: bool = False, size: str = "\\small"):
    """Write a LaTeX table; ``star=True`` spans both columns (table*)."""
    body = df.to_latex(index=False, float_format=lambda x: fmt % x, escape=True, column_format=col_format)
    body = body.replace("\\toprule", "\\hline").replace("\\midrule", "\\hline").replace("\\bottomrule", "\\hline")
    TABLES.mkdir(parents=True, exist_ok=True)
    env = "table*" if star else "table"
    with open(path, "w") as f:
        f.write("\\begin{%s}[t]\n\\centering\n\\caption{%s}\n\\label{%s}\n%s\n%s\\end{%s}\n" % (env, caption, label, size, body, env))


def main() -> None:
    set_style()
    models = available(ORDER)
    hm = pd.read_csv(C.RESULTS_DIR / "metrics_horizon.csv")
    pk = pd.read_csv(C.RESULTS_DIR / "metrics_peak_offpeak.csv")
    bh = pd.read_csv(C.RESULTS_DIR / "metrics_by_hour.csv")
    cg = pd.read_csv(C.RESULTS_DIR / "metrics_congestion.csv")
    vm = pd.read_csv(C.RESULTS_DIR / "metrics_validation.csv")
    hm = hm[hm.model.isin(models)]; pk = pk[pk.model.isin(models)]; bh = bh[bh.model.isin(models)]; cg = cg[cg.model.isin(models)]

    # ------------------------------------------------------------------ LaTeX tables
    wide = hm[hm.horizon != "average"].pivot(index="model", columns="horizon", values=["MAE", "RMSE", "MAPE"])
    rows = []
    for m in models:
        r = {"Model": m}
        for h in HZ:
            r[f"MAE {h}"] = wide.loc[m, ("MAE", h)]; r[f"RMSE {h}"] = wide.loc[m, ("RMSE", h)]; r[f"MAPE {h} (\\%)"] = wide.loc[m, ("MAPE", h)]
        rows.append(r)
    tbl = pd.DataFrame(rows)
    # compact: one block per horizon
    compact = pd.DataFrame({"Model": models})
    for h in HZ:
        compact[f"{h} MAE"] = [wide.loc[m, ("MAE", h)] for m in models]
        compact[f"{h} RMSE"] = [wide.loc[m, ("RMSE", h)] for m in models]
        compact[f"{h} MAPE"] = [wide.loc[m, ("MAPE", h)] for m in models]
    # two-level header written by hand so the table fits the page width
    TABLES.mkdir(parents=True, exist_ok=True)
    best = {(h, k): compact[f"{h} {k}"].min() for h in HZ for k in ("MAE", "RMSE", "MAPE")}
    lines = ["\\begin{table*}[t]", "\\centering",
             "\\caption{Test-set forecasting error by horizon (MAE and RMSE in mph, MAPE in \\%; masked over 207 sensors, 4--27 June 2012). Best value per column in bold.}",
             "\\label{tab:horizon}", "\\small", "\\begin{tabular}{l|rrr|rrr|rrr}", "\\hline",
             "Model & \\multicolumn{3}{c|}{15 min ahead} & \\multicolumn{3}{c|}{30 min ahead} & \\multicolumn{3}{c}{60 min ahead} \\\\",
             " & MAE & RMSE & MAPE & MAE & RMSE & MAPE & MAE & RMSE & MAPE \\\\", "\\hline"]
    for m in models:
        cells = []
        for h in HZ:
            for k in ("MAE", "RMSE", "MAPE"):
                v = compact.loc[compact.Model == m, f"{h} {k}"].iloc[0]
                s = "%.2f" % v if k != "MAPE" else "%.1f" % v
                cells.append("\\textbf{%s}" % s if abs(v - best[(h, k)]) < 1e-9 else s)
        lines.append(m + " & " + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}", ""]
    with open(TABLES / "metrics_horizon.tex", "w") as f:
        f.write("\n".join(lines))
    pk60 = pk[pk.horizon == "60min"].pivot(index="model", columns="period", values=["MAE", "RMSE", "MAPE"])
    pk15 = pk[pk.horizon == "15min"].pivot(index="model", columns="period", values=["MAE"])
    pkt = pd.DataFrame({"Model": models,
                        "15 min peak": [pk15.loc[m, ("MAE", "peak")] for m in models],
                        "15 min off-peak": [pk15.loc[m, ("MAE", "off-peak")] for m in models],
                        "60 min peak": [pk60.loc[m, ("MAE", "peak")] for m in models],
                        "60 min off-peak": [pk60.loc[m, ("MAE", "off-peak")] for m in models],
                        "60 min peak MAPE": [pk60.loc[m, ("MAPE", "peak")] for m in models],
                        "60 min off-peak MAPE": [pk60.loc[m, ("MAPE", "off-peak")] for m in models]})
    latex_table(pkt, TABLES / "metrics_peak.tex",
                "Peak (weekdays 07--10 h and 16--19 h) versus off-peak test MAE (mph) and MAPE (\\%).", "tab:peak",
                col_format="l|rr|rr|rr", star=True)
    cgt = cg[cg.horizon == "60min"][["model", "precision", "recall", "F1"]].set_index("model").loc[models].reset_index()
    cgt.columns = ["Model", "Precision", "Recall", "F1"]
    latex_table(cgt, TABLES / "metrics_congestion.tex",
                "Congestion detection at the 60-minute horizon (event: speed $<$ 35 mph).", "tab:congestion", fmt="%.3f", col_format="lrrr")
    vmt = vm[(vm.horizon != "average") & vm.model.isin(models)].pivot(index="model", columns="horizon", values="MAE").loc[models].reset_index()
    vmt.columns = ["Model"] + [h.replace("min", " min") for h in HZ]
    fit = hm.groupby("model")["fit_seconds"].first()
    vmt["Fit (s)"] = [round(fit[m]) for m in models]
    latex_table(vmt, TABLES / "metrics_validation.tex", "Validation-split MAE (mph) by horizon, used for model selection, and training wall-clock time (2 CPU cores).",
                "tab:validation", col_format="lrrrr", size="\\footnotesize")

    # ------------------------------------------------------------------ Fig: MAE by horizon
    fig, ax = plt.subplots(figsize=(8, 3.4))
    nm = len(models); w = 0.8 / nm
    for k, m in enumerate(models):
        vals = [wide.loc[m, ("MAE", h)] for h in HZ]
        bars = ax.bar(np.arange(len(HZ)) + (k - nm / 2 + 0.5) * w, vals, width=w * 0.9, color=MODEL_COLOURS[m], label=m, linewidth=0)
        annotate_bars(ax, bars, fontsize=6)
    ax.set_xticks(range(len(HZ))); ax.set_xticklabels([f"{h} ahead" for h in HZ]); ax.set_ylabel("Test MAE (mph)")
    ax.set_title("Test MAE by horizon: trees beat naive and linear baselines; neighbour features help at every horizon"); ax.legend(ncol=4, loc="upper left"); hide_spines(ax)
    save(fig, "fig_res_mae_by_horizon")

    # ------------------------------------------------------------------ Fig: peak vs off-peak
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, h in zip(axes, ["15min", "60min"]):
        sub = pk[pk.horizon == h].pivot(index="model", columns="period", values="MAE").loc[models]
        x = np.arange(nm)
        b1 = ax.bar(x - 0.2, sub["peak"], width=0.36, color=[MODEL_COLOURS[m] for m in models], linewidth=0)
        b2 = ax.bar(x + 0.2, sub["off-peak"], width=0.36, color=[MODEL_COLOURS[m] for m in models], alpha=0.4, linewidth=0)
        annotate_bars(ax, b1, fontsize=6); annotate_bars(ax, b2, fontsize=6)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=30, ha="right"); ax.set_title(f"{h} horizon: peak (solid) vs off-peak (faded)")
        hide_spines(ax)
    axes[0].set_ylabel("Test MAE (mph)")
    save(fig, "fig_res_peak_offpeak")

    # ------------------------------------------------------------------ Fig: MAE by hour of day (60 min)
    fig, ax = plt.subplots(figsize=(7.5, 3.3))
    for m in [x for x in ["HA", "Persistence", "XGB-local", "RF-local", "XGB-spatial"] if x in models]:
        sub = bh[(bh.model == m) & (bh.horizon == "60min")].sort_values("hour")
        ax.plot(sub.hour, sub.MAE, color=MODEL_COLOURS[m], marker="o", markersize=3, label=m)
    for s, e in C.PEAK_WINDOWS: ax.axvspan(s, e, color=MUTED, alpha=0.08, linewidth=0)
    ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("Hour of day (target time)"); ax.set_ylabel("Test MAE at 60 min (mph)")
    ax.set_title("Errors concentrate in the peaks (all days; shaded = weekday peak windows)"); ax.legend(ncol=3); hide_spines(ax)
    save(fig, "fig_res_mae_by_hour")

    # ------------------------------------------------------------------ Fig: congestion-detection F1
    fig, ax = plt.subplots(figsize=(7, 3))
    for k, m in enumerate(models):
        sub = cg[cg.model == m].set_index("horizon").loc[HZ]
        bars = ax.bar(np.arange(len(HZ)) + (k - nm / 2 + 0.5) * w, sub.F1, width=w * 0.9, color=MODEL_COLOURS[m], label=m, linewidth=0)
        annotate_bars(ax, bars, fontsize=6)
    ax.set_xticks(range(len(HZ))); ax.set_xticklabels([f"{h} ahead" for h in HZ]); ax.set_ylabel("F1 (speed < 35 mph)"); ax.set_ylim(0, 1)
    ax.set_title("Congestion-event detection quality"); ax.legend(ncol=4, loc="upper right"); hide_spines(ax)
    save(fig, "fig_res_congestion_f1")

    # ------------------------------------------------------------------ Prediction-level figures
    X_df, _, graph = data.load_all()
    fs = F.build_features(X_df, graph)
    te = fs.split.test; Y = fs.targets[te]; hz60 = HZ.index("60min")
    preds = {m: M.load_output(m) for m in models}
    best = "XGB-spatial" if "XGB-spatial" in preds else models[-1]
    loc = graph.locations

    # feature importance (best XGB)
    imp_name = "XGB-spatial" if "XGB-spatial" in models else "XGB-local"
    imp_all = pd.read_csv(C.RESULTS_DIR / "feature_importance.csv")
    imp = imp_all[imp_all.model == imp_name].head(16)[::-1]
    def group(f):
        if f.startswith(("up_", "down_", "nbr_")): return "neighbour (spatial)"
        if f.startswith("lag"): return "own lags"
        if f.startswith(("win_", "trend")): return "window statistics"
        if f in ("ha_now", "dev_from_ha"): return "historical average"
        return "calendar"
    gcol = {"own lags": CATEGORICAL[0], "neighbour (spatial)": CATEGORICAL[1], "window statistics": CATEGORICAL[2],
            "historical average": CATEGORICAL[3], "calendar": CATEGORICAL[4]}
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.barh(imp.feature, imp.importance, color=[gcol[group(f)] for f in imp.feature], linewidth=0)
    for g, c in gcol.items(): ax.bar([], [], color=c, label=g)
    ax.set_xlabel("Mean normalised gain importance (over 207 sensors x 3 horizons)"); ax.set_title(f"Top features of {imp_name}")
    ax.legend(loc="lower right"); hide_spines(ax)
    save(fig, "fig_res_feature_importance")

    # example forecast: most congested sensor, a test Wednesday
    s = int(np.nanargmin(np.nanmean(X_df.values[: fs.split.train_end], axis=0)))
    tidx = fs.index[te + C.HORIZONS["60min"]]
    day = (tidx >= "2012-06-13") & (tidx < "2012-06-14")
    fig, ax = plt.subplots(figsize=(9, 3.3))
    ax.plot(tidx[day], Y[day, s, hz60], color=INK2, linewidth=1.3, label="observed")
    for m, c in (("HA", MODEL_COLOURS["HA"]), ("XGB-local", MODEL_COLOURS["XGB-local"]), (best, MODEL_COLOURS[best])):
        if m in preds:
            ax.plot(tidx[day], preds[m].test_pred[day, s, hz60], color=c, linewidth=1.1, label=f"{m} (60-min-ahead)")
    ax.set_ylabel("Speed (mph)"); ax.set_title(f"Sensor {graph.sensor_ids[s]} on Wed 13 Jun 2012: 60-minute-ahead forecasts")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.16)); hide_spines(ax)
    import matplotlib.dates as mdates; ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    save(fig, "fig_res_example_forecast")

    # per-sensor MAE map and spatial improvement map
    psm_all = pd.read_csv(C.RESULTS_DIR / "per_sensor_mae.csv")
    psm_best = psm_all[psm_all.model == best]["60min"].values
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    sc = axes[0].scatter(loc.longitude, loc.latitude, c=psm_best, cmap=CMAP_SEQ, s=24, edgecolors="white", linewidths=0.4)
    fig.colorbar(sc, ax=axes[0], shrink=0.85).set_label("Test MAE at 60 min (mph)")
    axes[0].set_title(f"{best}: 60-min MAE per sensor")
    if "XGB-local" in models and best == "XGB-spatial":
        psm_local = psm_all[psm_all.model == "XGB-local"]["60min"].values
        gain = 100 * (psm_local - psm_best) / psm_local
        lim = np.nanpercentile(np.abs(gain), 95)
        sc2 = axes[1].scatter(loc.longitude, loc.latitude, c=gain, cmap=CMAP_DIV.reversed(), vmin=-lim, vmax=lim, s=24, edgecolors="white", linewidths=0.4)
        fig.colorbar(sc2, ax=axes[1], shrink=0.85).set_label("MAE reduction from neighbour features (%)")
        axes[1].set_title(f"Gain from neighbour features ({int((gain > 0).sum())}/{len(gain)} sensors improved)")
    for a in axes:
        a.grid(False); a.set_xlabel("Longitude"); a.set_ylabel("Latitude"); a.set_xticks(np.arange(-118.5, -118.15, 0.1))
    save(fig, "fig_res_sensor_mae_map")

    # congestion map at a peak instant: observed vs forecast
    t_sel = np.flatnonzero(tidx == pd.Timestamp("2012-06-13 17:30"))[0]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4), sharex=True, sharey=True)
    panels = [("Observed speed", Y[t_sel, :, hz60])]
    for m in ("HA", best):
        if m in preds: panels.append((f"{m} forecast issued 16:30", preds[m].test_pred[t_sel, :, hz60]))
    for ax, (title, v) in zip(axes, panels):
        sc = ax.scatter(loc.longitude, loc.latitude, c=v, cmap=CMAP_SEQ_R, vmin=10, vmax=70, s=24, edgecolors="white", linewidths=0.4)
        ax.set_title(title); ax.grid(False); ax.set_xticks(np.arange(-118.5, -118.15, 0.1))
    fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.02).set_label("Speed (mph), darker = congested")
    fig.suptitle("Congestion map, Wed 13 Jun 2012 17:30 (evening peak)", fontsize=10, fontweight="bold")
    save(fig, "fig_res_congestion_map")

    # predicted vs actual (hexbin) and error by speed regime
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    yt = Y[:, :, hz60].ravel(); yp = preds[best].test_pred[:, :, hz60].ravel(); ok = ~np.isnan(yt) & ~np.isnan(yp)
    hb = axes[0].hexbin(yt[ok], yp[ok], gridsize=45, cmap=CMAP_SEQ, bins="log", linewidths=0)
    axes[0].plot([0, 70], [0, 70], color=CATEGORICAL[1], linewidth=1, linestyle="--")
    axes[0].set_xlabel("Observed speed (mph)"); axes[0].set_ylabel("Predicted speed (mph)"); axes[0].set_title(f"{best}, 60 min: predicted vs observed"); axes[0].grid(False)
    fig.colorbar(hb, ax=axes[0]).set_label("count (log)")
    bins = np.arange(0, 75, 5); centres = bins[:-1] + 2.5
    for m in [x for x in ["HA", "Persistence", "XGB-local", best] if x in preds]:
        p = preds[m].test_pred[:, :, hz60].ravel()
        mae = [np.nanmean(np.abs(yt[ok][(yt[ok] >= a) & (yt[ok] < b)] - p[ok][(yt[ok] >= a) & (yt[ok] < b)])) for a, b in zip(bins[:-1], bins[1:])]
        axes[1].plot(centres, mae, color=MODEL_COLOURS[m], marker="o", markersize=3, label=m)
    axes[1].axvline(C.CONGESTION_THRESHOLD, color=MUTED, linestyle=":", linewidth=1)
    axes[1].set_xlabel("Observed speed bin (mph)"); axes[1].set_ylabel("Test MAE at 60 min (mph)"); axes[1].set_title("Error by speed regime: congestion is the hard part"); axes[1].legend()
    hide_spines(axes[1])
    save(fig, "fig_res_pred_vs_actual")

    # best-iteration distribution for XGB (regularisation evidence)
    if (C.RESULTS_DIR / "best_iterations.csv").exists():
        bi_all = pd.read_csv(C.RESULTS_DIR / "best_iterations.csv")
        bi = bi_all[bi_all.model == imp_name]
        fig, ax = plt.subplots(figsize=(6, 2.8))
        for k, h in enumerate(HZ):
            ax.hist(bi[h], bins=30, color=CATEGORICAL[k], alpha=0.75, linewidth=0, label=h)
        ax.set_xlabel("Best boosting round (early stopping on validation MAE)"); ax.set_ylabel("Sensors"); ax.set_title(f"{imp_name}: early-stopping rounds per sensor"); ax.legend(); hide_spines(ax)
        save(fig, "fig_res_best_iterations")

    print("Done. Models:", models)


if __name__ == "__main__":
    main()
