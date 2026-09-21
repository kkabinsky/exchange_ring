# -*- coding: utf-8 -*-
"""Econometric benchmark: the 10-year minus 3-month Treasury spread.

The spread is the standard yield-curve predictor (Estrella and Mishkin 1998). Here it is
used under the same protocol as the GAF detectors of Tables 8-9:
  - windows of 32 trading days on the prepared Treasury series (2015-05-06 to 2020-12-14);
  - test = the same 200 windows (first window start on/after 2020-01-02), crash label = the
    window overlaps the crash sub-window 2020-03-02..2020-03-30 (52 crash windows);
  - score of a window = minus the 10Y-3M spread on the last day of the window (a lower or
    negative spread counts as a stronger crash signal; only information up to the window end);
  - threshold = 0.95 quantile of the scores of the normal training windows (window start on or
    before 2019-12-30); a window is flagged when its score exceeds the threshold.
A probit P = Phi(a + b * spread) with b < 0 is a strictly increasing function of this score, so
it gives the same ranking, the same flagged windows, the same AUC/F1 and the same rank-based
DM statistics for any a and b; no coefficients need to be estimated.

The DM test against the exponential GAF-f-AnoGAN is the one used for Tables 8-9 (rank
probability, squared loss, one-sided p; positive DM means f-AnoGAN has the lower loss).

Output: output/03_bond_anomaly/benchmarks/spread_benchmark_*.csv  (runs in seconds)
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "output" / "03_bond_anomaly" / "bond_3m_1y_5y_10y_20y_compare_e20_daily"
PREP = RUN / "prepared_inputs"
OUT = REPO / "output" / "03_bond_anomaly" / "benchmarks"
MATURITIES = ["US3M", "US1Y", "US5Y", "US10Y", "US20Y"]

W = 32
TRAIN_END = pd.Timestamp("2019-12-30")
TEST_START = pd.Timestamp("2020-01-02")
N_TEST = 200
CRASH_START_DATE, CRASH_END_DATE = pd.Timestamp("2020-03-02"), pd.Timestamp("2020-03-30")
Q = 0.95


def load(mat):
    d = pd.read_excel(PREP / f"{mat}_BOND_COVID200_daily_final2.xlsx")
    d["Date"] = pd.to_datetime(d["Date"].astype(str), format="%Y%m%d")
    return d


def rank_prob(x):
    s = pd.Series(np.asarray(x, float))
    return s.rank(method="average").to_numpy() / (len(s) + 1.0)


def dm_one_sided(loss_base, loss_model):
    d = np.asarray(loss_base, float) - np.asarray(loss_model, float)
    sd = float(np.std(d, ddof=1))
    if len(d) <= 2 or sd == 0:
        return float("nan"), float("nan")
    stat = float(np.mean(d) / (sd / math.sqrt(len(d))))
    return stat, float(0.5 * math.erfc(stat / math.sqrt(2.0)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    y10, y3m = load("US10Y"), load("US3M")
    if not y10["Date"].equals(y3m["Date"]):
        raise RuntimeError("10Y and 3M series are not on the same dates")
    dates = y10["Date"].reset_index(drop=True)
    spread = (y10["cp"] - y3m["cp"]).to_numpy(float)

    starts = np.arange(0, len(spread) - W + 1)
    ends = starts + W - 1
    score = -spread[ends]                                   # higher = stronger crash signal
    start_dates, end_dates = dates[starts].to_numpy(), dates[ends].to_numpy()
    crash_start = int(np.searchsorted(dates, CRASH_START_DATE))
    crash_end_excl = int(np.searchsorted(dates, CRASH_END_DATE, side="right"))
    label = ((starts < crash_end_excl) & (starts + W > crash_start)).astype(int)

    test_pos = np.where(start_dates >= np.datetime64(TEST_START))[0][:N_TEST]
    train_normal = (start_dates <= np.datetime64(TRAIN_END)) & (label == 0)
    thr = float(np.quantile(score[train_normal], Q))

    test = pd.DataFrame({"window_start": starts[test_pos], "window_end": ends[test_pos],
                         "window_end_date": pd.to_datetime(end_dates[test_pos]).date,
                         "spread_10y_3m_at_window_end": spread[ends[test_pos]],
                         "score": score[test_pos], "threshold": thr,
                         "predicted_label": (score[test_pos] > thr).astype(int),
                         "label_0normal_1crash": label[test_pos]})
    test.to_csv(OUT / "spread_benchmark_test_scores.csv", index=False)

    y = test["label_0normal_1crash"].to_numpy()
    perf = pd.DataFrame([{
        "benchmark": "10Y-3M spread (= probit on the spread)", "n_train_normal": int(train_normal.sum()),
        "n_test": len(test), "n_test_crash": int(y.sum()), "threshold_q": Q, "threshold": thr,
        "auc": roc_auc_score(y, test["score"]),
        "precision": precision_score(y, test["predicted_label"], zero_division=0),
        "recall": recall_score(y, test["predicted_label"], zero_division=0),
        "f1": f1_score(y, test["predicted_label"], zero_division=0),
        "n_flagged": int(test["predicted_label"].sum())}])
    perf.round(6).to_csv(OUT / "spread_benchmark_performance.csv", index=False)

    rows = []
    for mat in MATURITIES:
        fa = pd.read_csv(RUN / "standalone_fanogan" / f"{mat}_BOND_COVID200_daily" / "out_put_four_gaf_fanogan"
                         / "exponential" / "test_scores.csv")
        j = fa[["window_start", "label_0normal_1crash", "anomaly_score"]].merge(
            test[["window_start", "label_0normal_1crash", "score"]], on="window_start",
            suffixes=("_fa", "_spread"), validate="one_to_one")
        if len(j) != len(test) or not (j["label_0normal_1crash_fa"] == j["label_0normal_1crash_spread"]).all():
            raise RuntimeError(f"{mat}: test windows or crash labels do not match the f-AnoGAN run")
        yy = j["label_0normal_1crash_fa"].to_numpy(float)
        loss_fa = (rank_prob(j["anomaly_score"]) - yy) ** 2
        loss_sp = (rank_prob(j["score"]) - yy) ** 2
        stat, p = dm_one_sided(loss_sp, loss_fa)
        rows.append({"maturity": mat, "n": len(j), "fanogan_exp_auc": roc_auc_score(yy, j["anomaly_score"]),
                     "spread_auc": roc_auc_score(yy, j["score"]),
                     "fanogan_exp_rank_brier": float(loss_fa.mean()), "spread_rank_brier": float(loss_sp.mean()),
                     "loss_diff_spread_minus_fanogan": float(loss_sp.mean() - loss_fa.mean()),
                     "dm_stat_positive_fanogan_better": stat, "p_one_sided_fanogan_better": p})
    dm = pd.DataFrame(rows)
    dm.round(6).to_csv(OUT / "spread_benchmark_dm_vs_fanogan_exp.csv", index=False)

    pd.set_option("display.width", 220)
    print(perf.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    print(dm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
