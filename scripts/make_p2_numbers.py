"""Export Phase 2 results as LaTeX macros (report/phase2/numbers_p2.tex) and tables, so the
report text can never disagree with results/dl/*.csv."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from trafficml import config as C

DL = C.RESULTS_DIR / "dl"
OUT = C.ROOT / "report" / "phase2"
KEY = {"HA": "HA", "Persistence": "Pers", "Ridge-local": "Ridge", "XGB-local": "XGBl", "XGB-spatial": "XGBs",
       "RF-local": "RFl", "RF-spatial": "RFs", "GRU-nograph": "GRU", "DCRNN-lite": "GCN", "ST-GAT": "GAT",
       "ST-GAT-LWR": "LWR", "Hybrid-XGB-GAT": "Hyb", "ST-GAT-short": "Short", "ST-GAT-1hop": "OneHop",
       "ST-GAT-3hop": "ThreeHop", "ST-GAT-undirected": "Undir", "ST-GAT-randomgraph": "Rand",
       "ST-GAT-zerofill": "Zero", "ST-GAT-nodropout": "NoDrop", "Hybrid-XGB-nograph": "HybNo",
       "ST-GAT-seed1": "SeedA", "ST-GAT-seed2": "SeedB"}
HW = {"15min": "Fifteen", "30min": "Thirty", "60min": "Sixty", "average": "Avg"}


def mac(name, val, fmt="%.2f"):
    v = fmt % val if isinstance(val, (float, np.floating)) else str(val)
    return f"\\newcommand{{\\{name}}}{{{v}}}"


def main():
    L = []
    hor = pd.read_csv(DL / "metrics_horizon.csv")
    for _, r in hor.iterrows():
        if r.model in KEY:
            k, h = KEY[r.model], HW[r.horizon]
            L += [mac(f"pMae{k}{h}", r.MAE), mac(f"pRmse{k}{h}", r.RMSE), mac(f"pMape{k}{h}", r.MAPE, "%.1f")]
    val = pd.read_csv(DL / "metrics_validation.csv")
    for _, r in val.iterrows():
        if r.model in KEY:
            L.append(mac(f"pVal{KEY[r.model]}{HW[r.horizon]}", r.MAE))
    pk = pd.read_csv(DL / "metrics_peak_offpeak.csv")
    for _, r in pk.iterrows():
        if r.model in KEY:
            L.append(mac(f"p{'Peak' if r.period == 'peak' else 'Off'}{KEY[r.model]}{HW[r.horizon]}", r.MAE))
    cg = pd.read_csv(DL / "metrics_congestion.csv")
    for _, r in cg.iterrows():
        if r.model in KEY:
            k, h = KEY[r.model], HW[r.horizon]
            L += [mac(f"pFone{k}{h}", r.F1), mac(f"pPrec{k}{h}", r.precision), mac(f"pRec{k}{h}", r.recall)]
    # relative change vs XGB-spatial
    ref = hor[hor.model == "XGB-spatial"].set_index("horizon").MAE
    refp = pk[pk.model == "XGB-spatial"].set_index(["horizon", "period"]).MAE
    for m in hor.model.unique():
        if m not in KEY or m in ("XGB-spatial",):
            continue
        r = hor[hor.model == m].set_index("horizon").MAE
        for h in HW:
            L.append(mac(f"pChg{KEY[m]}{HW[h]}", 100 * (r[h] / ref[h] - 1), "%+.1f"))
        rp = pk[pk.model == m].set_index(["horizon", "period"]).MAE
        for h in ("15min", "60min"):
            L.append(mac(f"pChgPeak{KEY[m]}{HW[h]}", 100 * (rp[(h, 'peak')] / refp[(h, 'peak')] - 1), "%+.1f"))
            L.append(mac(f"pChgOff{KEY[m]}{HW[h]}", 100 * (rp[(h, 'off-peak')] / refp[(h, 'off-peak')] - 1), "%+.1f"))
    # graph vs no graph, attention vs fixed
    def chg(a, b, h):
        A = hor[(hor.model == a) & (hor.horizon == h)].MAE.item(); B = hor[(hor.model == b) & (hor.horizon == h)].MAE.item()
        return 100 * (A / B - 1)
    for a, b, nm in (("ST-GAT", "GRU-nograph", "GatVsGru"), ("ST-GAT", "DCRNN-lite", "GatVsGcn"),
                     ("Hybrid-XGB-GAT", "ST-GAT", "HybVsGat"), ("ST-GAT-LWR", "ST-GAT", "LwrVsGat"),
                     ("Hybrid-XGB-GAT", "Hybrid-XGB-nograph", "HybVsHybNo")):
        if {a, b} <= set(hor.model):
            for h in HW:
                L.append(mac(f"pRel{nm}{HW[h]}", chg(a, b, h), "%+.1f"))
    # seeds
    f = DL / "seeds_summary.csv"
    if f.exists():
        s = pd.read_csv(f).set_index("horizon")
        for h in HW:
            if h in s.index:
                L += [mac(f"pSeedMean{HW[h]}", s.loc[h, "MAE_mean"]), mac(f"pSeedStd{HW[h]}", s.loc[h, "MAE_std"], "%.3f")]
    # DM tests
    dm = pd.read_csv(DL / "diebold_mariano.csv")
    for _, r in dm.iterrows():
        if r.model in KEY:
            L += [mac(f"pDm{KEY[r.model]}{HW[r.horizon]}", r.DM), mac(f"pDmP{KEY[r.model]}{HW[r.horizon]}", r.p_value, "%.3f" if r.p_value >= 0.001 else "%.0e")]
    # runs
    for m, k in KEY.items():
        f = DL / f"run_{m}.json"
        if f.exists():
            j = json.loads(f.read_text())
            L += [mac(f"pParams{k}", f"{j['n_params']:,}".replace(",", "{,}")), mac(f"pEpochs{k}", len(j["history"])),
                  mac(f"pMinutes{k}", j["fit_seconds"] / 60, "%.0f")]
    # per-sensor gain of best DL over XGB-spatial
    psm = pd.read_csv(DL / "per_sensor_mae.csv")
    best = hor[(hor.horizon == "average") & hor.model.isin(["GRU-nograph", "DCRNN-lite", "ST-GAT", "ST-GAT-LWR", "Hybrid-XGB-GAT"])].sort_values("MAE").model.iloc[0]
    a = psm[psm.model == "XGB-spatial"]["60min"].values; b = psm[psm.model == best]["60min"].values
    L += [mac("pBestModel", best), mac("pBestSensorsImproved", int(np.sum(b < a)))]
    att = DL / "attention_analysis.csv"
    if att.exists():
        d = pd.read_csv(att); u = d[d.model == "ST-GAT-undirected"]
        if len(u):
            p = u.groupby(["state", "edge_type"]).attention_share.sum().unstack()
            p = p.div(p.sum(1), axis=0)
            for st in ("free", "jammed"):
                for k in ("self", "downstream", "upstream"):
                    L.append(mac(f"pAtt{st.capitalize()}{k.capitalize()}", 100 * p.loc[st, k], "%.0f"))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "numbers_p2.tex").write_text("% generated by scripts/make_p2_numbers.py -- do not edit\n" + "\n".join(L) + "\n")
    print(len(L), "macros; best DL model:", best)


LABELS = [("HA", "Historical average", "P1"), ("Persistence", "Persistence", "P1"), ("XGB-spatial", "XGBoost + nbrs", "P1"),
          ("RF-spatial", "Random Forest + nbrs", "P1"), ("GRU-nograph", "GRU, no graph", "P2"),
          ("DCRNN-lite", "Diffusion GCN", "P2"), ("ST-GAT", "ST-GAT", "P2"), ("ST-GAT-LWR", "ST-GAT + LWR", "P2"),
          ("Hybrid-XGB-GAT", "Hybrid XGB+ST-GAT", "P2")]


def tables():
    hor = pd.read_csv(DL / "metrics_horizon.csv").set_index(["model", "horizon"])
    pk = pd.read_csv(DL / "metrics_peak_offpeak.csv").set_index(["model", "horizon", "period"])
    rows = [m for m in LABELS if (m[0], "15min") in hor.index]
    best = {h: min(hor.loc[(m, h), "MAE"] for m, _, _ in rows) for h in ("15min", "30min", "60min")}
    L = [r"\begin{table*}[t]\centering", r"\caption{Test-set masked MAE, RMSE (mph) and MAPE (\%) by horizon, 4--27 June 2012, identical samples for all models. P1 = Phase~1 model, P2 = Phase~2. Best MAE in bold.}",
         r"\label{tab:main}\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{llrrrrrrrrr}\hline",
         r" & & \multicolumn{3}{c}{15 min} & \multicolumn{3}{c}{30 min} & \multicolumn{3}{c}{60 min}\\",
         r"Model & & MAE & RMSE & MAPE & MAE & RMSE & MAPE & MAE & RMSE & MAPE\\\hline"]
    for m, lab, ph in rows:
        cells = []
        for h in ("15min", "30min", "60min"):
            r = hor.loc[(m, h)]
            mae = f"{r.MAE:.2f}"
            if abs(r.MAE - best[h]) < 1e-9: mae = r"\textbf{" + mae + "}"
            cells += [mae, f"{r.RMSE:.2f}", f"{r.MAPE:.1f}"]
        L.append(f"{lab} & {ph} & " + " & ".join(cells) + r"\\")
        if m == "RF-spatial": L.append(r"\hline")
    L += [r"\hline\end{tabular}\end{table*}"]
    (OUT / "tab_main.tex").write_text("\n".join(L) + "\n")
    L = [r"\begin{table}[t]\centering", r"\caption{Test MAE (mph) at 60 minutes, weekday rush hours (07--10, 16--19) vs off-peak, and change vs XGBoost-spatial.}",
         r"\label{tab:peak}\footnotesize", r"\begin{tabular}{lrrrr}\hline", r"Model & Peak & $\Delta$ & Off-peak & $\Delta$\\\hline"]
    rp = pk.loc[("XGB-spatial", "60min", "peak"), "MAE"]; ro = pk.loc[("XGB-spatial", "60min", "off-peak"), "MAE"]
    for m, lab, ph in rows:
        if m in ("HA", "Persistence", "RF-spatial"): continue
        a = pk.loc[(m, "60min", "peak"), "MAE"]; b = pk.loc[(m, "60min", "off-peak"), "MAE"]
        da = "--" if m == "XGB-spatial" else f"{100*(a/rp-1):+.1f}\\%"; db = "--" if m == "XGB-spatial" else f"{100*(b/ro-1):+.1f}\\%"
        L.append(f"{lab} & {a:.2f} & {da} & {b:.2f} & {db}" + r"\\")
    L += [r"\hline\end{tabular}\end{table}"]
    (OUT / "tab_peak.tex").write_text("\n".join(L) + "\n")
    ab = pd.read_csv(DL / "ablations.csv").set_index("model")
    order = [("ST-GAT-short", "reference (2 hops)"), ("ST-GAT-1hop", "1 hop"), ("ST-GAT-3hop", "3 hops"),
             ("ST-GAT-undirected", "undirected edges"), ("ST-GAT-randomgraph", "random graph"),
             ("ST-GAT-zerofill", "outages as 0 mph"), ("ST-GAT-nodropout", "no dropout/decay"),
             ("Hybrid-XGB-nograph", "hybrid, no graph")]
    L = [r"\begin{table}[t]\centering", r"\caption{Ablations under an identical 12-epoch budget: test MAE (mph) and change vs the reference ST-GAT-short.}",
         r"\label{tab:abl}\footnotesize", r"\begin{tabular}{lrrrr}\hline", r"Variant & 15 min & 60 min & $\Delta$15 & $\Delta$60\\\hline"]
    for m, lab in order:
        if m not in ab.index: continue
        r = ab.loc[m]
        d15 = "--" if m == "ST-GAT-short" else f"{r.delta_15min:+.3f}"; d60 = "--" if m == "ST-GAT-short" else f"{r.delta_60min:+.3f}"
        L.append(f"{lab} & {r['15min']:.2f} & {r['60min']:.2f} & {d15} & {d60}" + r"\\")
    L += [r"\hline\end{tabular}\end{table}"]
    (OUT / "tab_ablation.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
    tables()
