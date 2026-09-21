# -*- coding: utf-8 -*-
"""
brier_decomposition.py
======================

Answers Reviewer 3 point 5 on converting anomaly scores into rank-based
pseudo-probabilities and evaluating them with Brier loss.

Murphy decomposition:  BS = reliability - resolution + uncertainty

reliability   miscalibration; zero is perfect
resolution    how far the forecasts move away from the base rate; larger is
              better
uncertainty   fixed by the class balance; no method can change it

The rank transform spreads forecasts uniformly over the unit interval whatever
the event frequency, so it is miscalibrated by construction. This script
measures the size of that miscalibration and compares every detector with two
baselines: a random score put through the same rank transform, and a constant
forecast at the base rate.

Run
    python brier_decomposition.py
"""
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # replication package root
RUN = os.path.join(_REPO, "input", "iran_new_run_scores")
OUT = os.path.join(_REPO, "output", "05_leadtime_dm", "sensitivity_iran_run", "brier_decomposition")

# Iran 2026 is excluded for the same reason as Tables S1-S3
# its test set ends inside the event, so there is no post-event negative
# region against which scores can be ranked
EPISODES = {"COVID-19": "results_covid", "Russia-Ukraine": "results_russia",
            "Chinese real estate": "results_chinese", "Iran 2025": "results"}
ASSETS = ["USOIL", "GOLD", "EURUSD"]
EXCLUDE = {"standalone_tadgan"}      # scored on windows of 100, not 32
# The detectors the manuscript uses, pinned here. Without this the script
# would pick up any directory added later under the results tree and the
# output would stop matching the manuscript.
PAPER_METHODS = {
    "anomaly_transformer", "autoencoder", "cnn_autoencoder", "dagmm",
    "deep_svdd", "hybrid2", "isolation_forest", "omnianomaly",
    "one_class_svm", "standalone_fanogan", "tranad", "usad", "vae",
}


def rank_pseudo_prob(s):
    """Rank transform onto (0,1), as the ensemble layer does."""
    r = pd.Series(s).rank(method="average").to_numpy(float)
    return (r - 0.5) / len(r)


def murphy(y, f, bins):
    """Split the Brier score into reliability, resolution, uncertainty."""
    y = np.asarray(y, float)
    f = np.asarray(f, float)
    n = len(y)
    pbar = float(y.mean())
    bs = float(np.mean((f - y) ** 2))
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(f, edges[1:-1], right=False), 0, bins - 1)
    rel = res = 0.0
    for k in range(bins):
        m = idx == k
        nk = int(m.sum())
        if nk == 0:
            continue
        fk = float(f[m].mean())
        ok = float(y[m].mean())
        rel += nk * (fk - ok) ** 2
        res += nk * (ok - pbar) ** 2
    rel /= n
    res /= n
    unc = pbar * (1.0 - pbar)
    return dict(brier=bs, reliability=rel, resolution=res, uncertainty=unc,
                base_rate=pbar, n=n)


def read(root, asset, method, mapping):
    p = (os.path.join(root, asset, method, mapping, "test_scores.csv")
         if mapping != "-" else os.path.join(root, asset, method, "test_scores.csv"))
    if not os.path.exists(p):
        return None
    return pd.read_csv(p).sort_values("window_start")


def discover(root):
    out = set()
    for asset in ASSETS:
        base = os.path.join(root, asset)
        if not os.path.isdir(base):
            continue
        for m in os.listdir(base):
            mp = os.path.join(base, m)
            if not os.path.isdir(mp) or m in EXCLUDE or m not in PAPER_METHODS:
                continue
            if os.path.exists(os.path.join(mp, "test_scores.csv")):
                out.add((m, "-"))
            for g in os.listdir(mp):
                if os.path.exists(os.path.join(mp, g, "test_scores.csv")):
                    out.add((m, g))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    rows = []
    for ep, sub in EPISODES.items():
        root = os.path.join(RUN, sub)
        if not os.path.isdir(root):
            continue
        for method, mapping in discover(root):
            for asset in ASSETS:
                d = read(root, asset, method, mapping)
                if d is None:
                    continue
                y = d["label_0normal_1crash"].to_numpy(int)
                s = pd.to_numeric(d["anomaly_score"],
                                  errors="coerce").to_numpy(float)
                if not np.isfinite(s).all() or y.sum() in (0, len(y)):
                    continue
                m = murphy(y, rank_pseudo_prob(s), a.bins)
                m.update(episode=ep, asset=asset, method=method,
                         mapping=mapping, kind="model")
                rows.append(m)

                # baseline: a random score through the same rank transform
                mr = murphy(y, rank_pseudo_prob(rng.normal(size=len(y))), a.bins)
                mr.update(episode=ep, asset=asset, method=method,
                          mapping=mapping, kind="random")
                rows.append(mr)

                # baseline: a constant forecast at the base rate
                mc = murphy(y, np.full(len(y), y.mean()), a.bins)
                mc.update(episode=ep, asset=asset, method=method,
                          mapping=mapping, kind="constant")
                rows.append(mc)

    df = pd.DataFrame(rows)
    if df.empty:
        print("no results")
        return
    df.to_csv(os.path.join(OUT, "brier_decomposition.csv"), index=False)
    print("Wrote brier_decomposition.csv, %d rows" % len(df))

    print()
    print("=" * 92)
    print("Brier decomposition, mean over 12 cells (Iran 2026 excluded)")
    print("=" * 92)
    g = (df[df["kind"] == "model"].groupby(["method", "mapping"])
         .agg(brier=("brier", "mean"), reliability=("reliability", "mean"),
              resolution=("resolution", "mean"),
              uncertainty=("uncertainty", "mean"), n=("brier", "count"))
         .reset_index())
    g["res_share"] = (g["resolution"] / g["uncertainty"]).round(4)
    print(g.sort_values("brier").round(4).to_string(index=False))

    print()
    print("=" * 92)
    print("Against the baselines")
    print("=" * 92)
    b = (df.groupby("kind")
         .agg(brier=("brier", "mean"), reliability=("reliability", "mean"),
              resolution=("resolution", "mean"),
              uncertainty=("uncertainty", "mean")).reset_index())
    print(b.round(4).to_string(index=False))

    mm = df[df["kind"] == "model"]
    rr = df[df["kind"] == "random"]
    print()
    print("Mean resolution: model %.4f, random score %.4f"
          % (mm["resolution"].mean(), rr["resolution"].mean()))
    print("Resolution as a share of uncertainty: model %.1f%%, random %.1f%%"
          % (100 * mm["resolution"].mean() / mm["uncertainty"].mean(),
             100 * rr["resolution"].mean() / rr["uncertainty"].mean()))
    print()
    core = mm[(mm["method"] == "hybrid2") & (mm["mapping"] == "exponential")]
    if not core.empty:
        print("ExpoGAF (hybrid2/exponential)")
        print("  Brier %.4f = reliability %.4f - resolution %.4f + uncertainty %.4f"
              % (core["brier"].mean(), core["reliability"].mean(),
                 core["resolution"].mean(), core["uncertainty"].mean()))
        print("  resolution is %.1f%% of uncertainty"
              % (100 * core["resolution"].mean() / core["uncertainty"].mean()))
    print()
    print("Output written to", OUT)


if __name__ == "__main__":
    main()
