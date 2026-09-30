# -*- coding: utf-8 -*-
"""Print the empirical tables (Tables 5-10 and the US1M/US10Y/US30Y check of
Section 5.2) from the saved results in output/. No model is trained; this takes
a few seconds.

To retrain the models instead, run the scripts in code/03_bond_anomaly and
code/04_cross_market (see README.md)."""
import io
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "output"
sys.path.insert(0, str(REPO / "code" / "common"))
sys.path.insert(0, str(REPO / "code" / "04_cross_market"))
import run_market_crash_3datasets_oneclick as ock  # noqa: E402  (DM test used for Table 7)

M = ["cosine", "arctan", "arccosh", "exponential"]
CELLS = [("U30", "COVID-19", "DJIA_U30_COVID19_market_crash"),
         ("U30", "Russia-Ukraine", "DJIA_U30_Russia_Ukraine_war_crash"),
         ("U30", "Chinese real-estate", "DJIA_U30_Chinese_real_asset_market_crash"),
         ("U500", "COVID-19", "SP500_U500_COVID19_market_crash"),
         ("U500", "Russia-Ukraine", "SP500_U500_Russia_Ukraine_war_crash"),
         ("U500", "Chinese real-estate", "SP500_U500_Chinese_real_asset_market_crash"),
         ("DAX", "COVID-19", "DAX_INDEX_COVID19_market_crash"),
         ("DAX", "Russia-Ukraine", "DAX_INDEX_Russia_Ukraine_war_crash"),
         ("DAX", "Chinese real-estate", "DAX_INDEX_Chinese_real_asset_market_crash")]


def title(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def cell_dir(folder):
    return OUT / "04_cross_market" / folder / "out_put_four_gaf_fanogan"


# ---------------- Tables 5 and 6: bonds, CNN autoencoder and standalone f-AnoGAN ----------------
dm = pd.read_csv(OUT / "03_bond_anomaly" / "bond_3m_1y_5y_10y_20y_compare_e20_daily"
                 / "bond_3m_1y_5y_10y_20y_summary_dm_rank_score.csv")
for tab, pipe in [("Table 5", "CNN autoencoder"), ("Table 6", "f-AnoGAN standalone")]:
    title(f"{tab}  DM tests, five Treasury maturities, {pipe} (rank-Brier loss, one-sided p)")
    sub = dm.loc[dm["pipeline"] == pipe]
    order = {"US3M": 0, "US1Y": 1, "US5Y": 2, "US10Y": 3, "US20Y": 4}
    sub = sub.assign(o=sub["dataset"].map(order)).sort_values(["o", "baseline_method"],
                                                               key=lambda c: c if c.name == "o" else c.map({"cosine": 0, "arctan": 1, "arccosh": 2}))
    for _, r in sub.iterrows():
        p = r["p_one_sided_exp_better"]
        ps = "<0.0001" if p < 1e-4 else f"{p:.4f}"
        print(f"  {r['dataset']:6s} exp vs {r['baseline_method']:8s} loss diff {r['loss_diff_baseline_minus_exp']:+.4f}  "
              f"DM {r['dm_stat_positive_exp_better']:+.4f}  p {ps}")

# ---------------- Section 5.2: separate standalone f-AnoGAN check, US1M / US10Y / US30Y ----------------
title("Section 5.2  separate standalone f-AnoGAN check (20 epochs): exponential AUC and F1 per mapping")
BOND = OUT / "03_bond_anomaly"
CHECK = [("US1M", BOND / "bond_1m_30y_compare_e20" / "standalone_fanogan" / "US1M_BOND_COVID200_daily"),
         ("US10Y", BOND / "bond_3m_1y_5y_10y_20y_compare_e20_daily" / "standalone_fanogan" / "US10Y_BOND_COVID200_daily"),
         ("US30Y", BOND / "bond_1m_30y_compare_e20" / "standalone_fanogan" / "US30Y_BOND_COVID200_daily")]
for name, folder in CHECK:
    s = pd.read_csv(folder / "out_put_four_gaf_fanogan" / "comparison_summary.csv").set_index("method")
    print(f"  {name:6s} exp AUC = {s.loc['exponential', 'auc']:.3f}   F1: "
          + "  ".join(f"{m}={s.loc[m, 'f1']:.3f}" for m in M) + f"   ({folder.parent.parent.name})")
s = pd.read_csv(BOND / "bond_1m_30y_compare_e20_daily" / "standalone_fanogan" / "US30Y_BOND_COVID200_daily"
                / "out_put_four_gaf_fanogan" / "comparison_summary.csv").set_index("method")
print("  second run of the same program (bond_1m_30y_compare_e20_daily), US30Y F1: "
      + "  ".join(f"{m}={s.loc[m, 'f1']:.3f}" for m in M))

# ---------------- Tables 7 and 8: equity indices, standalone f-AnoGAN ----------------
title("Table 7  DM test exponential vs cosine (0/1 loss, two-sided p) and F1-best mapping")
print(f"{'Index':5s} {'Crash':20s} {'DM winner':12s} {'p':>7s}  F1-best")
for idx, crash, folder in CELLS:
    frames = []
    for m in ock.METHODS:
        s = pd.read_csv(cell_dir(folder) / m / "test_scores.csv")
        s["method"] = m
        frames.append(s)
    dm = ock.dm_tests(pd.concat(frames, ignore_index=True))
    r = dm.loc[dm["comparison"] == "exponential_vs_cosine"].iloc[0]
    summ = pd.read_csv(cell_dir(folder) / "comparison_summary.csv").set_index("method")
    best = summ["f1"].idxmax()
    p = "<0.001" if r["p_value"] < 0.001 else f"{r['p_value']:.3f}"
    print(f"{idx:5s} {crash:20s} {r['winner_lower_loss']:12s} {p:>7s}  {best}")

title("Table 8  F1 of standalone GAF-f-AnoGAN, four mappings")
print(f"{'Index':5s} {'Crash':20s} " + " ".join(f"{m:>11s}" for m in M))
for idx, crash, folder in CELLS:
    summ = pd.read_csv(cell_dir(folder) / "comparison_summary.csv").set_index("method")
    print(f"{idx:5s} {crash:20s} " + " ".join(f"{summ.loc[m, 'f1']:11.3f}" for m in M))

# ---------------- Table 9: USOIL, hybrid TadGAN-GAF-f-AnoGAN ----------------
title("Table 9  USOIL hybrid TadGAN-GAF-f-AnoGAN: crash AUC, squared loss (printed as Brier loss), one-sided DM p")
base = OUT / "04_cross_market" / "hybrid" / "USOIL_daily" / "stage3_fanogan"
for m in M:
    b = pd.read_csv(base / m / "backtest_detail.csv")
    y = 1 - b["true_label_user_0crash_1not"].astype(int)
    print(f"  {m:11s} AUC(crash) = {roc_auc_score(y, b['fanogan_score']):.3f}   (T={len(b)}, crash windows={int(y.sum())})")
import compute_hybrid_dm_matrix as hyb  # noqa: E402
buf = io.StringIO()
with redirect_stdout(buf):
    hyb.main()
print("  " + buf.getvalue().replace("\n", "\n  ").rstrip())

# ---------------- Table 10: lead-time consistency ----------------
title("Table 10  Lead-time consistency over the crisis cells (saved lead_time_economic_summary.csv)")
lead = pd.read_csv(OUT / "05_leadtime_dm" / "saved" / "lead_time_economic_summary.csv")
print(f"  {'Mapping':11s} {'n(L>0)':>6s} {'Mean':>6s} {'SD':>6s} {'CV':>5s} {'Worst':>6s}")
for m in M:
    x = lead.loc[lead["method"] == m, "lead_time_days"].dropna().to_numpy(float)
    print(f"  {m:11s} {int((x > 0).sum()):6d} {x.mean():6.1f} {x.std(ddof=1):6.1f} {x.std(ddof=1) / abs(x.mean()):5.2f} {x.min():6.0f}")
print("\nDone. All values above are read from output/; nothing was retrained.")
