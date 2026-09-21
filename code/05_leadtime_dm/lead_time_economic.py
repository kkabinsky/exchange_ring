"""Lead-time and economic-value analysis from existing f-AnoGAN test_scores.

Addresses reviewer point 2 (predictive timing + economic value) WITHOUT
retraining: it reuses the saved per-window anomaly scores / predictions.

For each dataset x crash scenario x GAF method:
  - lead_time_days : trading days between the FIRST crash signal (predicted_label==1)
                     in the test window and the documented crash onset.
                     positive => early warning before the crash.
  - strat_return   : cumulative return of a defensive strategy that holds the
                     index when no crash is signalled and moves to cash (0%)
                     for the window following a crash signal, over the test span.
  - buyhold_return : cumulative buy-and-hold return over the same test span.
  - excess_return  : strat_return - buyhold_return (economic value of acting on the signal).
Reports the full grid (every dataset x scenario x method).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --- replication package (added): data in <repo>/input, results in <repo>/output/05_leadtime_dm ---
import os as _rp_os, sys as _rp_sys
from pathlib import Path as _RpPath
REPO_ROOT = _RpPath(__file__).resolve().parents[2]
REPO_INPUT = REPO_ROOT / "input"
REPO_OUTPUT = REPO_ROOT / "output" / "05_leadtime_dm"
REPO_OUTPUT.mkdir(parents=True, exist_ok=True)
_rp_sys.path.insert(0, str(REPO_ROOT / "code" / "common"))
# ------------------------------------------------------------------------------------------
_rp_sys.path.insert(0, str(REPO_ROOT / "code" / "04_cross_market"))
import run_market_crash_3datasets_oneclick as runner

BASE = REPO_ROOT / "output" / "04_cross_market"  # saved f-AnoGAN cell results
METHODS = ["cosine", "exponential", "arctan", "arccosh"]
WINDOW = runner.WINDOW_SIZE


def analyse(dataset: dict, scenario: dict) -> list[dict]:
    frame = runner.load_dataset_frame(dataset)
    prices = frame["price"].to_numpy(dtype=np.float64)
    dates = frame["Date"].reset_index(drop=True)
    onset = pd.Timestamp(scenario.get("crash_onset", scenario["test_start"]))
    onset_idx = int(dates.index[dates >= onset][0])

    stock = f"{dataset['code']}_{scenario['code']}"
    out_dir = BASE / stock / "out_put_four_gaf_fanogan"
    rows = []
    for method in METHODS:
        f = out_dir / method / "test_scores.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f).sort_values("window_start").reset_index(drop=True)
        # signal date = window_end date (decision made when the window closes)
        end_idx = df["window_end"].to_numpy(dtype=int)
        sig = df["predicted_label"].to_numpy(dtype=int)

        # lead time: first signal at/after start, vs crash onset
        fired = np.where(sig == 1)[0]
        if len(fired) == 0:
            lead = np.nan
            first_sig_date = None
        else:
            first_end_idx = int(end_idx[fired[0]])
            lead = (onset_idx - first_end_idx)  # trading days before onset (can be negative)
            first_sig_date = dates.iloc[first_end_idx].date().isoformat()

        # economic: defensive strategy over test span (window_end indices)
        px = prices[end_idx]
        daily_ret = np.zeros(len(px))
        daily_ret[1:] = px[1:] / px[:-1] - 1.0
        # in cash for the step AFTER a crash signal (act next period)
        in_market = np.ones(len(px))
        in_market[1:] = (sig[:-1] == 0).astype(float)
        strat = float(np.prod(1.0 + daily_ret * in_market) - 1.0)
        bh = float(px[-1] / px[0] - 1.0)
        rows.append({
            "dataset": dataset["code"],
            "scenario": scenario["code"],
            "method": method,
            "n_test": int(len(df)),
            "crash_onset": onset.date().isoformat(),
            "first_signal_date": first_sig_date,
            "lead_time_days": lead,
            "strat_return": strat,
            "buyhold_return": bh,
            "excess_return": strat - bh,
        })
    return rows


def main() -> None:
    rows = []
    for dataset in runner.DATASETS:
        for scenario in runner.SCENARIOS:
            rows += analyse(dataset, scenario)
    df = pd.DataFrame(rows)
    out = REPO_OUTPUT / "lead_time_economic_summary_recomputed.csv"  # the file used for Table 10 is in saved/
    df.to_csv(out, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {out}")

    print("\n=== Exponential vs cosine: lead-time and excess return ===")
    for (ds, sc), g in df.groupby(["dataset", "scenario"]):
        e = g[g.method == "exponential"].iloc[0]
        c = g[g.method == "cosine"].iloc[0]
        print(f"{ds:11s} {sc:32s} | lead exp={e['lead_time_days']} cos={c['lead_time_days']} "
              f"| excess exp={e['excess_return']:+.3f} cos={c['excess_return']:+.3f}")


if __name__ == "__main__":
    main()
