# -*- coding: utf-8 -*-
"""Diebold-Mariano tests corrected for overlapping windows and multiple testing.

Reads only saved test scores (no training). For every DM test in Tables 5, 7, 8, 9 and in the
two benchmark comparisons (CNN autoencoder, 10Y-3M spread) it reports:

  dm_plain, p_plain   the statistic as in the original table (same loss, same variance, same side)
  dm_hac_hln, p_hac   Newey-West (Bartlett) long-run variance with lag L = W - 1 = 31, times the
                      Harvey-Leybourne-Newbold factor sqrt((T + 1 - 2h + h(h - 1)/T) / T) with
                      h = W = 32; p from a Student t distribution with T - 1 degrees of freedom
  p_bh                Benjamini-Hochberg adjusted p_hac within the family of the table

Windows of W = 32 trading days that move one day at a time share 31 observations, so the loss
differences are serially correlated up to lag 31.

Sides (as in the original tables): Tables 7, 8, 9 and the CNN benchmark are one-sided (exponential
better); Table 5 is two-sided (0/1 loss); the spread benchmark is two-sided, because its direction
was not fixed in advance.

Output: output/05_leadtime_dm/dm_corrected/dm_hac_hln_bh.csv  (runs in seconds)
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
O3 = REPO / "output" / "03_bond_anomaly"
O4 = REPO / "output" / "04_cross_market"
OUT = REPO / "output" / "05_leadtime_dm" / "dm_corrected"
BOND = O3 / "bond_3m_1y_5y_10y_20y_compare_e20_daily"
W = 32
LAG = W - 1
MATS = ["US3M", "US1Y", "US5Y", "US10Y", "US20Y"]
BASES = ["cosine", "arctan", "arccosh"]
MAPS = ["cosine", "arctan", "arccosh", "exponential"]


# ------------------------------------------------------------------ helpers
def rank_prob(x):
    s = pd.Series(np.asarray(x, float))
    return s.rank(method="average").to_numpy() / (len(s) + 1.0)


def plain_dm(d, ddof):
    d = np.asarray(d, float)
    sd = np.std(d, ddof=ddof)
    return float(d.mean() / (sd / math.sqrt(len(d)))) if sd > 0 else float("nan")


def hac_hln_dm(d, lag=LAG, h=W):
    d = np.asarray(d, float)
    T = len(d)
    e = d - d.mean()
    lrv = e @ e / T
    for k in range(1, min(lag, T - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * (e[k:] @ e[:-k]) / T
    if lrv <= 0:
        return float("nan")
    factor = math.sqrt(max(T + 1 - 2 * h + h * (h - 1) / T, 0.0) / T)
    return float(d.mean() / math.sqrt(lrv / T) * factor)


def p_normal(stat, side):
    if not np.isfinite(stat):
        return float("nan")
    return 0.5 * math.erfc(stat / math.sqrt(2)) if side == "one" else math.erfc(abs(stat) / math.sqrt(2))


def p_t(stat, T, side):
    if not np.isfinite(stat):
        return float("nan")
    return float(stats.t.sf(stat, T - 1)) if side == "one" else float(2 * stats.t.sf(abs(stat), T - 1))


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank in range(n, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * n / rank)
        adj[i] = running
    return adj


def check_contiguous(starts, name):
    s = np.sort(np.asarray(starts))
    if not np.all(np.diff(s) == 1):
        print(f"  note: {name}: window starts are not consecutive")


rows = []


def add(family, test, d, side, ddof, T=None, note=""):
    d = np.asarray(d, float)
    T = len(d) if T is None else T
    s0 = plain_dm(d, ddof)
    s1 = hac_hln_dm(d)
    rows.append({"family": family, "test": test, "T": len(d), "side": side,
                 "mean_loss_diff": float(d.mean()),
                 "dm_plain": s0, "p_plain": p_normal(s0, side),
                 "dm_hac_hln": s1, "p_hac_hln": p_t(s1, len(d), side), "note": note})


# ------------------------------------------------------------------ Tables 8 and 9 (bonds)
def fanogan_scores(mat, m):
    s = pd.read_csv(BOND / "standalone_fanogan" / f"{mat}_BOND_COVID200_daily" / "out_put_four_gaf_fanogan" / m / "test_scores.csv")
    return s.sort_values("window_start")[["window_start", "label_0normal_1crash", "anomaly_score"]]


def cnn_scores(mat, m):
    s = pd.read_csv(BOND / "cnn_autoencoder" / f"{mat}_BOND_COVID200_daily_CNN_AE" / "cnn_test_scores.csv")
    s = s[s["method"] == m]
    return s.sort_values("window_start")[["window_start", "label_0normal_1crash", "anomaly_score"]]


for fam, loader in [("Table 8 (CNN autoencoder)", cnn_scores), ("Table 9 (f-AnoGAN)", fanogan_scores)]:
    for mat in MATS:
        tabs = {m: loader(mat, m) for m in MAPS}
        check_contiguous(tabs["exponential"]["window_start"], f"{fam} {mat}")
        y = tabs["exponential"]["label_0normal_1crash"].to_numpy(float)
        loss = {m: (rank_prob(tabs[m]["anomaly_score"]) - y) ** 2 for m in MAPS}
        for b in BASES:
            if not np.array_equal(tabs[b]["window_start"].to_numpy(), tabs["exponential"]["window_start"].to_numpy()):
                raise RuntimeError(f"{fam} {mat} {b}: windows differ")
            add(fam, f"{mat} exp vs {b}", loss[b] - loss["exponential"], "one", ddof=1)

# ------------------------------------------------------------------ Table 5 (equity indices, exp vs cosine, 0/1 loss)
CELLS = [("U30", "COVID-19", "DJIA_U30_COVID19_market_crash"),
         ("U30", "Russia-Ukraine", "DJIA_U30_Russia_Ukraine_war_crash"),
         ("U30", "Chinese", "DJIA_U30_Chinese_real_asset_market_crash"),
         ("U500", "COVID-19", "SP500_U500_COVID19_market_crash"),
         ("U500", "Russia-Ukraine", "SP500_U500_Russia_Ukraine_war_crash"),
         ("U500", "Chinese", "SP500_U500_Chinese_real_asset_market_crash"),
         ("DAX", "COVID-19", "DAX_INDEX_COVID19_market_crash"),
         ("DAX", "Russia-Ukraine", "DAX_INDEX_Russia_Ukraine_war_crash"),
         ("DAX", "Chinese", "DAX_INDEX_Chinese_real_asset_market_crash")]
for idx, crash, folder in CELLS:
    base = O4 / folder / "out_put_four_gaf_fanogan"
    e = pd.read_csv(base / "exponential" / "test_scores.csv")
    c = pd.read_csv(base / "cosine" / "test_scores.csv")
    j = e.merge(c, on=["window_start", "window_end"], suffixes=("_e", "_c"), validate="one_to_one").sort_values("window_start")
    check_contiguous(j["window_start"], f"Table 5 {idx} {crash}")
    le = (j["label_0normal_1crash_e"] - j["predicted_label_e"]) ** 2
    lc = (j["label_0normal_1crash_c"] - j["predicted_label_c"]) ** 2
    add("Table 5 (equity indices)", f"{idx} {crash} exp vs cosine", lc - le, "two", ddof=1)

# ------------------------------------------------------------------ Table 7 (USOIL hybrid)
hyb = O4 / "hybrid" / "USOIL_daily" / "stage3_fanogan"
frames = {m: pd.read_csv(hyb / m / "backtest_detail.csv")[["window_start", "fanogan_score", "true_label_user_0crash_1not"]]
          for m in MAPS}
j = frames["cosine"].rename(columns={"fanogan_score": "cosine"})
for m in ["arctan", "arccosh", "exponential"]:
    j = j.merge(frames[m].rename(columns={"fanogan_score": m}), on=["window_start", "true_label_user_0crash_1not"])
j = j.sort_values("window_start")
check_contiguous(j["window_start"], "Table 7")
y7 = 1 - j["true_label_user_0crash_1not"].astype(int).to_numpy()
l7 = {m: (rank_prob(j[m]) - y7) ** 2 for m in MAPS}
for b in BASES:
    add("Table 7 (USOIL hybrid)", f"exp vs {b}", l7[b] - l7["exponential"], "one", ddof=0)

# ------------------------------------------------------------------ benchmarks (CNN-AE, spread)
spread = pd.read_csv(O3 / "benchmarks" / "spread_benchmark_test_scores.csv").sort_values("window_start")
for mat in MATS:
    fa = fanogan_scores(mat, "exponential")
    y = fa["label_0normal_1crash"].to_numpy(float)
    lfa = (rank_prob(fa["anomaly_score"]) - y) ** 2
    for m in MAPS:
        cn = cnn_scores(mat, m)
        if not np.array_equal(cn["window_start"].to_numpy(), fa["window_start"].to_numpy()):
            raise RuntimeError(f"benchmark {mat} {m}: windows differ")
        lcn = (rank_prob(cn["anomaly_score"]) - y) ** 2
        add("Benchmarks", f"{mat} f-AnoGAN exp vs CNN-AE {m}", lcn - lfa, "one", ddof=1)
    if not np.array_equal(spread["window_start"].to_numpy(), fa["window_start"].to_numpy()):
        raise RuntimeError(f"benchmark {mat} spread: windows differ")
    lsp = (rank_prob(spread["score"]) - y) ** 2
    add("Benchmarks", f"{mat} f-AnoGAN exp vs spread", lsp - lfa, "two", ddof=1,
        note="two-sided; negative DM = spread has the lower loss")

# ------------------------------------------------------------------ BH within family, save, print
df = pd.DataFrame(rows)
df["p_bh"] = np.nan
for fam, g in df.groupby("family", sort=False):
    df.loc[g.index, "p_bh"] = bh(g["p_hac_hln"].to_numpy())
df["sig_plain_5pct"] = df["p_plain"] < 0.05
df["sig_hac_hln_5pct"] = df["p_hac_hln"] < 0.05
df["sig_bh_5pct"] = df["p_bh"] < 0.05
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "dm_hac_hln_bh.csv"
df.round(6).to_csv(out, index=False)

pd.set_option("display.width", 230)
for fam, g in df.groupby("family", sort=False):
    print(f"\n=== {fam}  (T={g['T'].iloc[0]}, {len(g)} tests, side={g['side'].iloc[0]}) ===")
    print(g[["test", "dm_plain", "p_plain", "dm_hac_hln", "p_hac_hln", "p_bh"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSignificant at 5% (plain -> HAC/HLN -> HAC/HLN + BH):")
print(df.groupby("family", sort=False)[["sig_plain_5pct", "sig_hac_hln_5pct", "sig_bh_5pct"]].sum().to_string())
print(f"\nSaved: {out}")
