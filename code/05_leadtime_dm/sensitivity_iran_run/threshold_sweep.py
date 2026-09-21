# -*- coding: utf-8 -*-
"""
threshold_sweep.py
==================

How much of the reported performance depends on where the alarm threshold was
put? Reviewer 4, point 9.

The manuscript fixes the alarm at the 0.95 quantile of the training scores and
reports the metrics that follow. Stating one threshold does not show that the
conclusion survives a different one, which is what the Reviewer asked for.

Nothing is retrained here. Every detector's per-window score is already stored,
so the sweep is a recomputation and is exact: run it twice on any machine and
the output files are identical to the byte.

Two ways of setting the threshold are swept, because they answer different
questions.

    control     the quantile is taken on the crash-free 2019 control, so the
                threshold is set by a target false-alarm rate away from any
                crash. This is the operating point a user would actually pick,
                and it makes the detectors comparable to each other.
    test        the quantile is taken on the scores of the episode being
                judged. This is closer to what the manuscript did, and it is
                reported so the two can be compared, but it uses the test
                period to set its own threshold and so flatters every method.

AUC and average precision are reported beside the threshold-dependent metrics.
They do not move with the threshold at all, which is the point: if a conclusion
holds under one and not the other, it was a statement about the threshold.

Run
    python threshold_sweep.py
    python threshold_sweep.py --quantiles 0.80,0.85,0.90,0.95,0.99
"""
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # replication package root
RUN = os.path.join(_REPO, "input", "iran_new_run_scores")
OUT = os.path.join(_REPO, "output", "05_leadtime_dm", "sensitivity_iran_run", "threshold_sweep")

# The detectors the manuscript reports, pinned. Without this the sweep would
# pick up any directory added later under the results tree and would stop
# matching the manuscript.
PAPER_METHODS = [
    "anomaly_transformer", "autoencoder", "cnn_autoencoder", "dagmm",
    "deep_svdd", "hybrid2", "isolation_forest", "omnianomaly",
    "one_class_svm", "standalone_fanogan", "tranad", "usad", "vae",
]
MAPPING = "exponential"
EPISODES = {"COVID-19": "results_covid", "Russia-Ukraine": "results_russia",
            "Chinese": "results_chinese", "Iran 2025": "results",
            "Iran 2026": "results_2026"}
CONTROL_DIR = os.path.join("covid_normal", "results")
ASSETS = ["USOIL", "GOLD", "EURUSD"]


def read_scores(sub, asset, method):
    p = os.path.join(RUN, sub, asset, method, MAPPING, "test_scores.csv")
    if not os.path.isfile(p):
        return None
    d = pd.read_csv(p).sort_values("window_start")
    return (d["anomaly_score"].to_numpy(float),
            d["label_0normal_1crash"].to_numpy(int))


def auc(y, s):
    npos, nneg = int((y == 1).sum()), int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def average_precision(y, s):
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    yy = y[o]
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def metrics(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / max(len(y), 1), "precision": prec,
            "recall": rec, "f1": f1,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else 0.0,
            "alarm_rate": float((pred == 1).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quantiles",
                    default="0.70,0.75,0.80,0.85,0.90,0.95,0.975,0.99")
    a = ap.parse_args()
    qs = [float(x) for x in a.quantiles.split(",")]
    os.makedirs(OUT, exist_ok=True)

    # thresholds from the crash-free control, one per asset, method and quantile
    control = {}
    for asset in ASSETS:
        for m in PAPER_METHODS:
            got = read_scores(CONTROL_DIR, asset, m)
            if got is not None:
                control[(asset, m)] = got[0]

    rows = []
    for ev, sub in EPISODES.items():
        for asset in ASSETS:
            for m in PAPER_METHODS:
                got = read_scores(sub, asset, m)
                if got is None:
                    continue
                s, y = got
                if y.sum() == 0 or y.sum() == len(y):
                    continue
                base = {"event": ev, "asset": asset, "method": m,
                        "n": int(len(y)), "prevalence": float(y.mean()),
                        "auc": auc(y, s),
                        "average_precision": average_precision(y, s)}
                for q in qs:
                    for how in ("control", "test"):
                        if how == "control":
                            c = control.get((asset, m))
                            if c is None:
                                continue
                            thr = float(np.quantile(c, q))
                        else:
                            thr = float(np.quantile(s, q))
                        r = dict(base)
                        r.update({"quantile": q, "threshold_from": how,
                                  "threshold": thr})
                        r.update(metrics(y, (s > thr).astype(int)))
                        rows.append(r)

    t = pd.DataFrame(rows)
    f = os.path.join(OUT, "threshold_sweep.csv")
    t.round(6).to_csv(f, index=False)
    report(t, qs)
    print()
    print("Per-cell results written to", f)


def report(t, qs):
    print("=" * 84)
    print("Threshold sensitivity, mean over the fifteen crisis cells")
    print("=" * 84)
    for how in ("control", "test"):
        s = t[t["threshold_from"] == how]
        if s.empty:
            continue
        g = s.groupby("quantile").agg(
            accuracy=("accuracy", "mean"), precision=("precision", "mean"),
            recall=("recall", "mean"), f1=("f1", "mean"),
            false_alarm=("false_alarm_rate", "mean"),
            alarm_rate=("alarm_rate", "mean"))
        print()
        print("threshold taken on the %s scores" % how)
        print(g.round(3).to_string())

    print()
    print("=" * 84)
    print("Does the ranking of the detectors change with the threshold?")
    print("=" * 84)
    s = t[t["threshold_from"] == "control"]
    piv = s.pivot_table(index="method", columns="quantile", values="f1")
    piv = piv[[q for q in qs if q in piv.columns]]
    rank = piv.rank(ascending=False)
    print("F1 by detector and quantile, with the rank in brackets")
    for m in piv.index:
        cells = "  ".join("%.3f(%d)" % (piv.loc[m, q], int(rank.loc[m, q]))
                          for q in piv.columns)
        print("  %-20s %s" % (m, cells))
    print()
    top = {q: piv[q].idxmax() for q in piv.columns}
    print("  best detector at each quantile: %s"
          % ", ".join("%.3f %s" % (q, v) for q, v in top.items()))
    n_diff = len(set(top.values()))
    print("  %d different detectors take first place across the %d thresholds"
          % (n_diff, len(top)))

    print()
    print("=" * 84)
    print("What does not move with the threshold")
    print("=" * 84)
    g = t.groupby("method").agg(auc=("auc", "mean"),
                                average_precision=("average_precision", "mean"),
                                prevalence=("prevalence", "mean"))
    print(g.round(4).to_string())
    print()
    print("  AUC and average precision are computed from the ranking of the")
    print("  scores and do not depend on where the alarm is placed. A claim")
    print("  that holds under one threshold and not another is a claim about")
    print("  the threshold, not about the detector.")


if __name__ == "__main__":
    main()
