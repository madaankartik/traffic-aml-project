#!/usr/bin/env python3
"""Fill the results placeholders in README.md from results/*.csv.

Placeholders: RESULTS_TABLE, SPATIAL_GAIN_15, SPATIAL_GAIN_60, N_IMPROVED.  Idempotent:
re-running replaces the previously filled block between the HTML comment markers.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trafficml import config as C   # noqa: E402

ORDER = ["HA", "Persistence", "Ridge-local", "XGB-local", "RF-local", "XGB-spatial", "RF-spatial"]


def md_table(df: pd.DataFrame, fmt="{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def replace_block(text: str, key: str, content: str) -> str:
    start, end = f"<!-- {key}:start -->", f"<!-- {key}:end -->"
    block = f"{start}\n{content}\n{end}"
    if start in text:
        return re.sub(re.escape(start) + r".*?" + re.escape(end), lambda _: block, text, flags=re.S)
    return text.replace(key, block)


def main() -> None:
    hm = pd.read_csv(C.RESULTS_DIR / "metrics_horizon.csv")
    pk = pd.read_csv(C.RESULTS_DIR / "metrics_peak_offpeak.csv")
    cg = pd.read_csv(C.RESULTS_DIR / "metrics_congestion.csv")
    models = [m for m in ORDER if m in set(hm.model)]
    wide = hm[hm.horizon != "average"].pivot(index="model", columns="horizon", values=["MAE", "RMSE", "MAPE"])
    rows = []
    for m in models:
        rows.append({"Model": m, "MAE 15": wide.loc[m, ("MAE", "15min")], "MAE 30": wide.loc[m, ("MAE", "30min")],
                     "MAE 60": wide.loc[m, ("MAE", "60min")], "RMSE 60": wide.loc[m, ("RMSE", "60min")],
                     "MAPE 60 (%)": wide.loc[m, ("MAPE", "60min")],
                     "F1 congestion 60": float(cg[(cg.model == m) & (cg.horizon == "60min")].F1.iloc[0])})
    table = md_table(pd.DataFrame(rows))

    def mae(m, h):
        return float(wide.loc[m, ("MAE", h)])
    gain15 = 100 * (mae("XGB-local", "15min") - mae("XGB-spatial", "15min")) / mae("XGB-local", "15min")
    gain60 = 100 * (mae("XGB-local", "60min") - mae("XGB-spatial", "60min")) / mae("XGB-local", "60min")
    psm = pd.read_csv(C.RESULTS_DIR / "per_sensor_mae.csv")
    psl = psm[psm.model == "XGB-local"]["60min"].values
    pss = psm[psm.model == "XGB-spatial"]["60min"].values
    n_imp = int((pss < psl).sum())

    readme = (ROOT / "README.md").read_text()
    readme = replace_block(readme, "RESULTS_TABLE", table)
    readme = readme.replace("SPATIAL_GAIN_15", f"{gain15:.1f}").replace("SPATIAL_GAIN_60", f"{gain60:.1f}").replace("N_IMPROVED", str(n_imp))
    (ROOT / "README.md").write_text(readme)

    print("filled README.md")


if __name__ == "__main__":
    main()
