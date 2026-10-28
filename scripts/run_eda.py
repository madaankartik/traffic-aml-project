#!/usr/bin/env python3
"""Exploratory data analysis and data-quality assessment of METR-LA.

Produces figures/fig_eda_*.png|pdf and results/eda_summary.json + results/eda_sensor_table.csv.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trafficml import config as C                                   # noqa: E402
from trafficml import data, features as F                           # noqa: E402
from trafficml.plots import (CATEGORICAL, CMAP_SEQ, CMAP_SEQ_R, INK2, MUTED, GRID, AXIS,   # noqa: E402
                             annotate_bars, hide_spines, save, set_style)


def main() -> None:
    set_style()
    C.RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    X_df, raw, graph = data.load_all()
    X = X_df.values
    idx = X_df.index
    T, N = X.shape
    split = F.chronological_split(T)
    summary: dict = {}

    # ------------------------------------------------------------------ basic facts
    summary["n_timestamps"] = int(T)
    summary["n_sensors"] = int(N)
    summary["start"] = str(idx[0]); summary["end"] = str(idx[-1])
    summary["sampling_minutes"] = C.STEP_MINUTES
    summary["n_days"] = float((idx[-1] - idx[0]).total_seconds() / 86400)
    summary["frac_zero_readings"] = float((raw.values == 0).mean())
    summary["frac_missing_after_mask"] = float(np.isnan(X).mean())
    obs = X[~np.isnan(X)]
    summary["speed_mean"] = float(obs.mean()); summary["speed_median"] = float(np.median(obs))
    summary["speed_std"] = float(obs.std()); summary["speed_min"] = float(obs.min()); summary["speed_max"] = float(obs.max())
    summary["speed_p05"] = float(np.percentile(obs, 5)); summary["speed_p95"] = float(np.percentile(obs, 95))
    summary["frac_congested_obs"] = float((obs < C.CONGESTION_THRESHOLD).mean())
    summary["frac_timestamps_fully_missing"] = float(np.isnan(X).all(axis=1).mean())
    # gaps in the timestamp index
    expected = pd.date_range(idx[0], idx[-1], freq=f"{C.STEP_MINUTES}min")
    summary["n_missing_timestamps_in_index"] = int(len(expected) - T)
    # sensors
    sensor_tbl = F.sensor_static_table(X, graph, split.train_end)
    sensor_tbl.to_csv(C.RESULTS_DIR / "eda_sensor_table.csv", index=False)
    summary["sensors_missing_gt_20pct"] = int((sensor_tbl["frac_missing"] > 0.2).sum())
    summary["sensors_missing_gt_50pct"] = int((sensor_tbl["frac_missing"] > 0.5).sum())
    summary["worst_sensor_missing"] = float(sensor_tbl["frac_missing"].max())
    A = graph.adjacency.copy(); np.fill_diagonal(A, 0)
    summary["graph_edges"] = int((A > 0).sum())
    summary["graph_mean_out_degree"] = float((A > 0).sum(axis=1).mean())
    summary["graph_density"] = float((A > 0).mean())
    summary["graph_isolated_sensors"] = int((((A > 0).sum(axis=1) + (A > 0).sum(axis=0)) == 0).sum())
    dist = graph.distances
    finite = dist[(dist["from"] != dist["to"])]["cost"]
    summary["road_distance_pairs"] = int(len(dist)); summary["road_distance_median_m"] = float(finite.median())
    # outage run lengths
    runs = []
    for n in range(N):
        m = np.isnan(X[:, n]).astype(int)
        d = np.diff(np.concatenate([[0], m, [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        runs += list(ends - starts)
    runs = np.array(runs)
    summary["outage_runs"] = int(len(runs)); summary["outage_run_median_steps"] = float(np.median(runs))
    summary["outage_run_p90_steps"] = float(np.percentile(runs, 90)); summary["outage_run_max_steps"] = int(runs.max())
    summary["frac_outage_steps_in_runs_ge_1day"] = float(runs[runs >= C.STEPS_PER_DAY].sum() / runs.sum())

    tod = F.time_of_day_slot(idx); dow = idx.dayofweek.values
    weekday = dow < 5

    # ------------------------------------------------------------------ 1. sensor map
    fig, ax = plt.subplots(figsize=(6.2, 5))
    sc = ax.scatter(sensor_tbl["longitude"], sensor_tbl["latitude"], c=sensor_tbl["train_mean_speed"],
                    cmap=CMAP_SEQ_R, s=26, linewidths=0.4, edgecolors="white")
    for i in range(N):                                   # draw road-graph edges faintly
        for j in graph.neighbours_out(i):
            ax.plot([sensor_tbl.longitude[i], sensor_tbl.longitude[j]], [sensor_tbl.latitude[i], sensor_tbl.latitude[j]],
                    color=GRID, linewidth=0.4, zorder=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.8); cb.set_label("Mean speed on training period (mph)")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); ax.grid(False)
    ax.set_xticks(np.arange(-118.55, -118.15, 0.1))
    ax.set_title("METR-LA: 207 loop detectors with road-graph edges (darker = slower)")
    save(fig, "fig_eda_sensor_map")

    # ------------------------------------------------------------------ 2. speed distribution
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))
    axes[0].hist(raw.values.ravel(), bins=71, range=(0, 71), color=CATEGORICAL[0], linewidth=0)
    axes[0].axvspan(-0.5, 1, color=CATEGORICAL[7], alpha=0.25, linewidth=0)
    axes[0].set_title("Raw readings (red band: 0 mph = outage)"); axes[0].set_xlabel("Speed (mph)"); axes[0].set_ylabel("Count")
    axes[1].hist(obs, bins=70, range=(1, 71), color=CATEGORICAL[0], linewidth=0)
    axes[1].axvline(C.CONGESTION_THRESHOLD, color=CATEGORICAL[1], linestyle="--", linewidth=1.2)
    axes[1].text(C.CONGESTION_THRESHOLD - 1, axes[1].get_ylim()[1] * 0.9, "congested\n< 35 mph", ha="right", fontsize=7.5, color=INK2)
    axes[1].set_title("After masking: free-flow vs congested modes"); axes[1].set_xlabel("Speed (mph)")
    for a in axes: hide_spines(a)
    save(fig, "fig_eda_speed_distribution")

    # ------------------------------------------------------------------ 3. daily profile
    net = np.nanmean(X, axis=1)
    prof = pd.DataFrame({"tod": tod, "net": net, "weekday": weekday})
    g = prof.groupby(["weekday", "tod"])["net"]
    mean, p10, p90 = g.mean().unstack(0), g.quantile(0.1).unstack(0), g.quantile(0.9).unstack(0)
    hours = np.arange(C.STEPS_PER_DAY) / 12
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for flag, lab, col in ((True, "Weekday", CATEGORICAL[0]), (False, "Weekend", CATEGORICAL[1])):
        ax.fill_between(hours, p10[flag], p90[flag], color=col, alpha=0.15, linewidth=0)
        ax.plot(hours, mean[flag], color=col, label=f"{lab} mean (band: 10th–90th pct.)")
    for s, e in C.PEAK_WINDOWS:
        ax.axvspan(s, e, color=MUTED, alpha=0.08, linewidth=0)
    ax.set_xlim(0, 24); ax.set_xticks(range(0, 25, 3)); ax.set_xlabel("Hour of day"); ax.set_ylabel("Network-mean speed (mph)")
    ax.set_title("Daily profile: two weekday peaks (shaded 07–10, 16–19) vs flat weekends"); ax.legend(loc="lower left"); hide_spines(ax)
    save(fig, "fig_eda_daily_profile")

    # ------------------------------------------------------------------ 4. weekly heatmap
    week = pd.DataFrame({"dow": dow, "hour": idx.hour, "net": net}).groupby(["dow", "hour"])["net"].mean().unstack()
    fig, ax = plt.subplots(figsize=(7.5, 2.9))
    im = ax.imshow(week.values, aspect="auto", cmap=CMAP_SEQ_R, interpolation="nearest")
    ax.set_yticks(range(7)); ax.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("Hour of day"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("Mean speed (mph)")
    ax.set_title("Time-of-week structure of network-mean speed (darker = slower)")
    save(fig, "fig_eda_weekly_heatmap")

    # ------------------------------------------------------------------ 5. missing data
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), gridspec_kw={"width_ratios": [1.2, 1]})
    daily_missing = pd.Series(np.isnan(X).mean(axis=1), index=idx).resample("D").mean()
    axes[0].fill_between(daily_missing.index, daily_missing.values * 100, color=CATEGORICAL[0], alpha=0.9, linewidth=0)
    axes[0].axvline(idx[split.train_end], color=INK2, linestyle=":", linewidth=1); axes[0].axvline(idx[split.val_end], color=INK2, linestyle=":", linewidth=1)
    axes[0].set_ylabel("% readings missing (per day)"); axes[0].set_title("Outages over time (dotted = split boundaries)")
    axes[0].tick_params(axis="x", labelrotation=30)
    srt = np.sort(sensor_tbl["frac_missing"].values)[::-1] * 100
    axes[1].bar(np.arange(N), srt, width=1.0, color=CATEGORICAL[0], linewidth=0)
    axes[1].axhline(20, color=CATEGORICAL[1], linestyle="--", linewidth=1); axes[1].text(N - 2, 21, "20 %", ha="right", fontsize=7.5, color=INK2)
    axes[1].set_xlabel("Sensors (sorted)"); axes[1].set_ylabel("% missing"); axes[1].set_title(f"Missingness per sensor ({summary['sensors_missing_gt_20pct']} sensor > 20 %)")
    for a in axes: hide_spines(a)
    save(fig, "fig_eda_missing")

    # ------------------------------------------------------------------ 6. spatial correlation vs distance
    rng = np.random.default_rng(C.RANDOM_SEED)
    sub = X[split.train_end - 12 * C.STEPS_PER_DAY: split.train_end]         # last 12 training days
    corr = pd.DataFrame(sub).corr().values
    id2 = {s: k for k, s in enumerate(graph.sensor_ids)}
    pairs = dist[(dist["from"] != dist["to"]) & (dist["cost"] < 15000)
                 & dist["from"].isin(id2) & dist["to"].isin(id2)]
    pairs = pairs.sample(min(len(pairs), 20000), random_state=C.RANDOM_SEED)
    pi = pairs["from"].map(id2).values; pj = pairs["to"].map(id2).values
    r = corr[pi, pj]; d_km = pairs["cost"].values / 1000
    ok = ~np.isnan(r)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.scatter(d_km[ok], r[ok], s=4, color=CATEGORICAL[0], alpha=0.18, linewidths=0, label="sensor pair")
    bins = np.arange(0, 15.5, 1.0); centres = bins[:-1] + 0.5
    med = [np.nanmedian(r[ok][(d_km[ok] >= a) & (d_km[ok] < b)]) for a, b in zip(bins[:-1], bins[1:])]
    ax.plot(centres, med, color=CATEGORICAL[1], marker="o", markersize=4, label="median per 1 km bin")
    ax.set_xlim(0, 12.5)
    ax.set_xlabel("Road-network distance between sensors (km)"); ax.set_ylabel("Pearson correlation of speeds")
    ax.set_title("Spatial dependence decays with road distance (motivates graph models)"); ax.legend(loc="upper right"); hide_spines(ax)
    save(fig, "fig_eda_spatial_correlation")
    summary["corr_median_lt1km"] = float(med[0]); summary["corr_median_5to6km"] = float(med[5]); summary["corr_median_10to11km"] = float(med[10])

    # ------------------------------------------------------------------ 7. autocorrelation
    def acf(x, maxlag):
        x = pd.Series(x).interpolate(limit_direction="both").values
        x = x - x.mean(); den = np.dot(x, x)
        return np.array([np.dot(x[:-k], x[k:]) / den if k else 1.0 for k in range(maxlag + 1)])
    lags = np.arange(0, 2 * C.STEPS_PER_DAY + 1)
    a_net = acf(net, lags[-1]); a_s = acf(X[:, 0], lags[-1])
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(lags / 12, a_net, color=CATEGORICAL[0], label="network-mean speed")
    ax.plot(lags / 12, a_s, color=CATEGORICAL[1], label=f"sensor {graph.sensor_ids[0]}")
    for h in (1, 24, 48): ax.axvline(h, color=MUTED, linewidth=0.6, linestyle=":")
    ax.set_xlabel("Lag (hours)"); ax.set_ylabel("Autocorrelation"); ax.set_title("Autocorrelation: strong 1-hour memory and a 24-hour cycle")
    ax.legend(loc="upper right"); hide_spines(ax)
    save(fig, "fig_eda_autocorrelation")
    summary["acf_net_1h"] = float(a_net[12]); summary["acf_net_24h"] = float(a_net[288]); summary["acf_sensor0_1h"] = float(a_s[12])

    # ------------------------------------------------------------------ 8. example week, three sensors
    wk = (idx >= "2012-04-02") & (idx < "2012-04-09")
    picks = [int(sensor_tbl["train_mean_speed"].idxmin()), int(np.argsort(sensor_tbl["train_mean_speed"].values)[N // 2]), int(sensor_tbl["train_mean_speed"].idxmax())]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    for k, (p, lab) in enumerate(zip(picks, ["most congested sensor", "median sensor", "freest sensor"])):
        ax.plot(idx[wk], X[wk, p], color=CATEGORICAL[k], linewidth=0.9, label=f"{graph.sensor_ids[p]} ({lab})")
    ax.set_ylabel("Speed (mph)"); ax.set_title("One week (2–8 Apr 2012): recurrent weekday congestion, sensor-specific regimes")
    ax.legend(loc="lower left", ncol=3); hide_spines(ax); ax.tick_params(axis="x", labelrotation=0)
    save(fig, "fig_eda_example_week")

    # ------------------------------------------------------------------ 9. graph structure
    outdeg = (A > 0).sum(axis=1); indeg = (A > 0).sum(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3))
    axes[0].hist(outdeg, bins=np.arange(0, outdeg.max() + 2) - 0.5, color=CATEGORICAL[0], linewidth=0, label="out-degree")
    axes[0].hist(indeg, bins=np.arange(0, indeg.max() + 2) - 0.5, color=CATEGORICAL[1], linewidth=0, alpha=0.7, label="in-degree")
    axes[0].set_xlabel("Degree"); axes[0].set_ylabel("Sensors"); axes[0].set_title(f"Degree distribution ({summary['graph_edges']} directed edges)"); axes[0].legend()
    w = A[A > 0]
    axes[1].hist(w, bins=30, color=CATEGORICAL[0], linewidth=0)
    axes[1].set_xlabel("Edge weight  exp(-(d/σ)²)"); axes[1].set_ylabel("Edges"); axes[1].set_title("Gaussian-kernel edge weights (threshold 0.1)")
    for a in axes: hide_spines(a)
    save(fig, "fig_eda_graph")

    # ------------------------------------------------------------------ 10. split timeline
    daily = pd.Series(net, index=idx).resample("D").mean()
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(daily.index, daily.values, color=INK2, linewidth=1)
    for (a, b, lab, col) in ((0, split.train_end, "train 70 %", CATEGORICAL[0]), (split.train_end, split.val_end, "val 10 %", CATEGORICAL[2]), (split.val_end, T - 1, "test 20 %", CATEGORICAL[1])):
        ax.axvspan(idx[a], idx[b], color=col, alpha=0.12, linewidth=0)
        ax.text(idx[(a + b) // 2], daily.max() + 0.5, lab, ha="center", fontsize=8, color=INK2)
    ax.set_ylabel("Daily network-mean\nspeed (mph)"); ax.set_title("Chronological split (1-hour gaps between partitions)"); hide_spines(ax)
    save(fig, "fig_eda_split_timeline")

    # ------------------------------------------------------------------ 11. congestion share by hour
    cong = (X < C.CONGESTION_THRESHOLD)
    share = pd.DataFrame({"hour": idx.hour, "wd": weekday, "c": np.nanmean(np.where(np.isnan(X), np.nan, cong), axis=1)}).groupby(["wd", "hour"])["c"].mean().unstack(0) * 100
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(share.index, share[True], color=CATEGORICAL[0], marker="o", markersize=3, label="Weekday")
    ax.plot(share.index, share[False], color=CATEGORICAL[1], marker="o", markersize=3, label="Weekend")
    ax.set_xticks(range(0, 24, 3)); ax.set_xlabel("Hour of day"); ax.set_ylabel("% of sensors < 35 mph")
    ax.set_title("Share of congested sensors by hour"); ax.legend(); hide_spines(ax)
    save(fig, "fig_eda_congestion_by_hour")
    summary["peak_congested_share_weekday_pct"] = float(share[True].max()); summary["peak_congested_hour_weekday"] = int(share[True].idxmax())

    with open(C.RESULTS_DIR / "eda_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
