# -*- coding: utf-8 -*-
"""Diebold-Mariano test: exponential GAF-f-AnoGAN against the CNN autoencoder benchmark.

Reads only the saved test scores of run_bond_5maturities_compare.py (no training):
  f-AnoGAN : output/03_bond_anomaly/bond_3m_1y_5y_10y_20y_compare_e20_daily/standalone_fanogan/<job>/out_put_four_gaf_fanogan/exponential/test_scores.csv
  CNN-AE   : output/03_bond_anomaly/bond_3m_1y_5y_10y_20y_compare_e20_daily/cnn_autoencoder/<job>_CNN_AE/cnn_test_scores.csv
Both pipelines score the same 200 test windows with the same crash labels; the rows are
matched on window_start. The DM test is the one used for Tables 8-9: each score is turned
into a rank probability r/(n+1), the loss is (r/(n+1) - y)^2, d = loss_CNN - loss_fAnoGAN,
DM = mean(d) / (sd(d)/sqrt(n)), one-sided p = P(Z > DM) (positive DM: f-AnoGAN better).

Output: output/03_bond_anomaly/benchmarks/dm_fanogan_exp_vs_cnn_ae.csv  (runs in seconds)
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
RUN = REPO / "output" / "03_bond_anomaly" / "bond_3m_1y_5y_10y_20y_compare_e20_daily"
OUT = REPO / "output" / "03_bond_anomaly" / "benchmarks"
MATURITIES = ["US3M", "US1Y", "US5Y", "US10Y", "US20Y"]
MAPPINGS = ["cosine", "arctan", "arccosh", "exponential"]


def rank_prob(x):
    s = pd.Series(np.asarray(x, float))
    return s.rank(method="average").to_numpy() / (len(s) + 1.0)


def dm_one_sided(loss_base, loss_model):
    """same formula as dm_exp_vs_baselines() in run_bond_5maturities_compare.py"""
    d = np.asarray(loss_base, float) - np.asarray(loss_model, float)
    d = d[np.isfinite(d)]
    sd = float(np.std(d, ddof=1))
    if len(d) <= 2 or sd == 0:
        return float("nan"), float("nan")
    stat = float(np.mean(d) / (sd / math.sqrt(len(d))))
    return stat, float(0.5 * math.erfc(stat / math.sqrt(2.0)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for mat in MATURITIES:
        job = f"{mat}_BOND_COVID200_daily"
        fa = pd.read_csv(RUN / "standalone_fanogan" / job / "out_put_four_gaf_fanogan" / "exponential" / "test_scores.csv")
        cn_all = pd.read_csv(RUN / "cnn_autoencoder" / f"{job}_CNN_AE" / "cnn_test_scores.csv")
        for m in MAPPINGS:
            cn = cn_all.loc[cn_all["method"] == m, ["window_start", "label_0normal_1crash", "anomaly_score", "predicted_label"]]
            j = fa[["window_start", "label_0normal_1crash", "anomaly_score", "predicted_label"]].merge(
                cn, on="window_start", suffixes=("_fa", "_cnn"), validate="one_to_one")
            if not (j["label_0normal_1crash_fa"] == j["label_0normal_1crash_cnn"]).all():
                raise RuntimeError(f"{mat}/{m}: crash labels differ between the two pipelines")
            y = j["label_0normal_1crash_fa"].to_numpy(float)
            loss_fa = (rank_prob(j["anomaly_score_fa"]) - y) ** 2
            loss_cn = (rank_prob(j["anomaly_score_cnn"]) - y) ** 2
            stat, p = dm_one_sided(loss_cn, loss_fa)
            rows.append({
                "maturity": mat, "cnn_mapping": m, "n": len(j), "n_crash": int(y.sum()),
                "fanogan_exp_auc": roc_auc_score(y, j["anomaly_score_fa"]),
                "cnn_auc": roc_auc_score(y, j["anomaly_score_cnn"]),
                "fanogan_exp_f1": f1_score(y, j["predicted_label_fa"], zero_division=0),
                "cnn_f1": f1_score(y, j["predicted_label_cnn"], zero_division=0),
                "fanogan_exp_rank_brier": float(loss_fa.mean()),
                "cnn_rank_brier": float(loss_cn.mean()),
                "loss_diff_cnn_minus_fanogan": float(loss_cn.mean() - loss_fa.mean()),
                "dm_stat_positive_fanogan_better": stat,
                "p_one_sided_fanogan_better": p,
            })
    df = pd.DataFrame(rows)
    out = OUT / "dm_fanogan_exp_vs_cnn_ae.csv"
    df.round(6).to_csv(out, index=False)
    pd.set_option("display.width", 220)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
