# -*- coding: utf-8 -*-
"""
overlap_corrected_tests.py
==========================

Confidence intervals and comparisons that respect the overlap between windows.
Reviewer 4, point 4.

Windows advance by one observation and are 32 long, so a resampling scheme that
draws single windows treats dependent observations as independent and returns
an interval that is far too narrow. This program resamples contiguous blocks
instead, and resamples the cells as well, so that both sources of dependence
are carried into the interval.

    inside a cell   a moving block bootstrap with block length 32, which is the
                    shortest block whose interior carries no observation shared
                    with the block before it
    across cells    the fifteen episode-and-asset cells are themselves
                    resampled, because a method that does well on one episode
                    tends to do well on the same episode under every detector

Nothing is retrained. The per-window scores are already stored, so the result
is exact up to the resampling seed, which is fixed.

What it answers
    Whether the ranking of the detectors survives an interval that respects the
    dependence, and whether the difference between any two of them is larger
    than that interval. The point estimates are unchanged from the manuscript;
    only the uncertainty around them is corrected.

Run
    python overlap_corrected_tests.py
    python overlap_corrected_tests.py --replicates 4000 --block 32
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # replication package root
RUN = os.path.join(_REPO, "input", "iran_new_run_scores")
OUT = os.path.join(_REPO, "output", "05_leadtime_dm", "sensitivity_iran_run", "overlap_corrected")

MAPPING = "exponential"
PAPER_METHODS = [
    "anomaly_transformer", "autoencoder", "cnn_autoencoder", "dagmm",
    "deep_svdd", "hybrid2", "isolation_forest", "omnianomaly",
    "one_class_svm", "standalone_fanogan", "tranad", "usad", "vae",
]
EPISODES = {"COVID-19": "results_covid", "Russia-Ukraine": "results_russia",
            "Chinese": "results_chinese", "Iran 2025": "results",
            "Iran 2026": "results_2026"}
ASSETS = ["USOIL", "GOLD", "EURUSD"]


def auc(y, s):
    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def load(methods):
    """Scores and labels for every cell that carries both classes."""
    cells = {}
    for ev, sub in EPISODES.items():
        for asset in ASSETS:
            per = {}
            y_ref = None
            for m in methods:
                p = os.path.join(RUN, sub, asset, m, MAPPING,
                                 "test_scores.csv")
                if not os.path.isfile(p):
                    continue
                d = pd.read_csv(p).sort_values("window_start")
                y = d["label_0normal_1crash"].to_numpy(int)
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                if y_ref is None:
                    y_ref = y
                elif len(y) != len(y_ref):
                    continue
                per[m] = d["anomaly_score"].to_numpy(float)
            if per and y_ref is not None:
                cells[(ev, asset)] = (y_ref, per)
    return cells


def block_indices(n, block, rng):
    """A resample of the row positions, drawn as contiguous blocks."""
    out = []
    while len(out) < n:
        start = int(rng.integers(0, n))
        out.extend(((start + np.arange(block)) % n).tolist())
    return np.asarray(out[:n], int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default=",".join(PAPER_METHODS))
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--replicates", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    methods = [m.strip() for m in a.methods.split(",") if m.strip()]
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    cells = load(methods)
    keys = sorted(cells)
    print("cells %d, block length %d, replicates %d"
          % (len(keys), a.block, a.replicates))
    print("point estimates are unchanged; only the interval is corrected")
    print()

    point = {}
    for m in methods:
        v = [auc(cells[k][0], cells[k][1][m])
             for k in keys if m in cells[k][1]]
        v = [x for x in v if np.isfinite(x)]
        point[m] = float(np.mean(v)) if v else np.nan

    # bootstrap: resample the cells, then resample blocks inside each cell
    draws = {m: np.empty(a.replicates) for m in methods}
    for b in range(a.replicates):
        pick = rng.integers(0, len(keys), len(keys))
        acc = {m: [] for m in methods}
        for j in pick:
            y, per = cells[keys[j]]
            idx = block_indices(len(y), a.block, rng)
            yy = y[idx]
            if yy.sum() == 0 or yy.sum() == len(yy):
                continue
            for m, s in per.items():
                acc[m].append(auc(yy, s[idx]))
        for m in methods:
            v = [x for x in acc[m] if np.isfinite(x)]
            draws[m][b] = np.mean(v) if v else np.nan

    rows = []
    for m in methods:
        d = draws[m][np.isfinite(draws[m])]
        if len(d) < 10:
            continue
        rows.append({"method": m, "auc": point[m],
                     "ci_low": float(np.percentile(d, 2.5)),
                     "ci_high": float(np.percentile(d, 97.5)),
                     "width": float(np.percentile(d, 97.5)
                                    - np.percentile(d, 2.5)),
                     "replicates": int(len(d))})
    t = pd.DataFrame(rows).sort_values("auc", ascending=False)
    f = os.path.join(OUT, "overlap_corrected_tests.csv")
    t.round(6).to_csv(f, index=False)

    print("=" * 80)
    print("Mean AUC with a 95 per cent interval that respects the overlap")
    print("=" * 80)
    print("%-22s %8s %18s %8s" % ("method", "AUC", "95% interval", "width"))
    for _, r in t.iterrows():
        print("%-22s %8.4f   [%6.3f, %6.3f] %8.3f"
              % (r["method"], r["auc"], r["ci_low"], r["ci_high"], r["width"]))

    best = t.iloc[0]
    overlap = t[(t["ci_high"] >= best["ci_low"])]
    print()
    print("  the leading method is %s at %.4f" % (best["method"], best["auc"]))
    print("  %d of %d methods have an interval that overlaps its own"
          % (len(overlap), len(t)))
    print("  mean interval width %.3f, against a spread of %.3f between the"
          % (t["width"].mean(), t["auc"].max() - t["auc"].min()))
    print("  best and the worst method")

    # paired differences, resampled the same way
    print()
    print("=" * 80)
    print("Paired differences against the layered pipeline, same resampling")
    print("=" * 80)
    ref = "hybrid2"
    if ref in draws:
        print("%-22s %10s %18s %10s"
              % ("against", "mean diff", "95% interval", "excludes 0"))
        out = []
        for m in methods:
            if m == ref:
                continue
            d = draws[m] - draws[ref]
            d = d[np.isfinite(d)]
            if len(d) < 10:
                continue
            lo, hi = np.percentile(d, [2.5, 97.5])
            out.append((m, float(d.mean()), float(lo), float(hi),
                        "yes" if (lo > 0 or hi < 0) else "no"))
        out.sort(key=lambda r: -r[1])
        for m, mu, lo, hi, ex in out:
            print("%-22s %+10.4f   [%+6.3f, %+6.3f] %10s"
                  % (m, mu, lo, hi, ex))
        n_sig = sum(1 for r in out if r[4] == "yes")
        print()
        print("  %d of %d differences have an interval that excludes zero"
              % (n_sig, len(out)))

    print()
    print("Output file", f)


if __name__ == "__main__":
    main()
