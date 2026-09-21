# -*- coding: utf-8 -*-
"""
leadtime_far_matched.py
=======================

Compare lead time with every detector held at the same false-alarm rate.

Why this had to be rewritten
----------------------------
The earlier lead-time tables used the predicted_label column shipped with each
result file, and each method had set its own threshold. Measured on the
crash-free 2019 control the resulting false-alarm rates were:

    deep_svdd            89.3 %
    autoencoder          73.4 %
    anomaly_transformer  48.5 %
    hybrid2 (ExpoGAF)    16.9 %
    tranad                3.4 %

A method that alarms most of the time will always show a long lead time,
because it was already alarming. Comparing lead times without holding the
false-alarm rate fixed therefore means nothing, and the advantage ExpoGAF
appeared to hold over TranAD may reflect nothing more than alarming five times
as often.

What this script does instead
    1. sets each method's threshold on the crash-free control, not on the test
       set; setting it on the test set uses information from the future, which
       is the point Reviewer 3 raises about threshold calibration
    2. forces every method to the same false-alarm rate before reading lead time
    3. adds a persistence rule: k consecutive windows above the threshold before
       an alarm counts, which removes lead times created by a single spike
    4. tests the paired differences across cells with Holm correction

Run
    python leadtime_far_matched.py
    python leadtime_far_matched.py --far 0.05 --k 2
"""
import argparse
import glob
import io
import os
import sys
from itertools import product

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # replication package root
RUN = os.path.join(_REPO, "input", "iran_new_run_scores")
OUT = os.path.join(_REPO, "output", "05_leadtime_dm", "sensitivity_iran_run", "leadtime_far_matched")

# the five episodes: results directory, onset date, price data directory
EPISODES = {
    "COVID":          ("results_covid",   "2020-02-20", "datasets"),
    "Russia-Ukraine": ("results_russia",  "2022-02-11", "datasets"),
    "Chinese":        ("results_chinese", "2023-08-07", "datasets"),
    "Iran 2025":      ("results",         "2025-06-12", "datasets"),
    "Iran 2026":      ("results_2026",    "2026-06-12", "datasets_2026"),
}
# the crash-free control, used to set thresholds and measure false alarms
CONTROL = os.path.join(RUN, "covid_normal", "results")

ASSETS = {"USOIL": "USOIL_daily_final2.xlsx",
          "GOLD": "GOLD_daily_final2.xlsx",
          "EURUSD": "EURUSD_daily_final2.xlsx"}

WINDOW = 32
CORE = ("hybrid2", "exponential")          # the core ExpoGAF configuration
ABLATION = ["standalone_tadgan", "tadgan_stage", "standalone_fanogan", "hybrid2"]
# The detectors the manuscript uses, pinned here. Without this the script
# would pick up any directory added later under the results tree and the
# output would stop matching the manuscript.
PAPER_METHODS = {
    "anomaly_transformer", "autoencoder", "cnn_autoencoder", "dagmm",
    "deep_svdd", "hybrid2", "isolation_forest", "omnianomaly",
    "one_class_svm", "standalone_fanogan", "tranad", "usad", "vae",
    # this table includes standalone_tadgan, unlike the metric tables,
    # because the lead-time figures were computed before it was excluded
    "standalone_tadgan",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def load_dates(asset, data_subdir):
    """Return the date series of an asset, to map window indices to dates."""
    p = os.path.join(RUN, data_subdir, ASSETS[asset])
    df = pd.read_excel(p)
    c = {x.lower(): x for x in df.columns}
    return pd.to_datetime(df[c["date"]]).reset_index(drop=True)


def read_scores(root, asset, method, mapping):
    """Read test_scores.csv; return (scores, window-end indices) or None."""
    if mapping and mapping != "-":
        p = os.path.join(root, asset, method, mapping, "test_scores.csv")
    else:
        p = os.path.join(root, asset, method, "test_scores.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p).sort_values("window_start")
    s = pd.to_numeric(d["anomaly_score"], errors="coerce").to_numpy(float)
    if "window_end" in d.columns:
        e = d["window_end"].to_numpy(int)
    else:
        e = d["window_start"].to_numpy(int) + WINDOW - 1
    ok = np.isfinite(s)
    return s[ok], e[ok]


def alarm_state(scores, tau, k):
    """Alarm state: k consecutive windows above threshold count as an alarm."""
    hot = scores >= tau
    if k <= 1:
        return hot
    out = np.zeros(len(hot), dtype=bool)
    run = 0
    for i, h in enumerate(hot):
        run = run + 1 if h else 0
        if run >= k:
            out[i] = True
    return out


def far_of(scores, tau, k):
    """False-alarm rate on the crash-free control."""
    if len(scores) == 0:
        return np.nan
    return float(alarm_state(scores, tau, k).mean())


def tau_for_far(control_scores, target, k):
    """Find the threshold whose control false-alarm rate is closest to target.
    Take the lowest threshold that still stays at or under the target."""
    if control_scores is None or len(control_scores) == 0:
        return np.nan, np.nan
    cand = np.unique(control_scores)
    if len(cand) == 0:
        return np.nan, np.nan
    # allow a threshold above the maximum, meaning never alarm
    cand = np.concatenate([cand, [cand[-1] + abs(cand[-1]) * 1e-6 + 1e-12]])
    best_tau, best_far = np.nan, np.nan
    for t in cand[::-1]:                       # high to low = few to many alarms
        f = far_of(control_scores, t, k)
        if f <= target + 1e-12:
            best_tau, best_far = float(t), f   # still under target; try lower
        else:
            break
    if not np.isfinite(best_tau):              # target unreachable; take the least
        best_tau = float(cand[-1])
        best_far = far_of(control_scores, best_tau, k)
    return best_tau, best_far


def lead_days(scores, ends, tau, k, onset_idx, dates):
    """Lead time in days; positive means the alarm preceded the onset."""
    st = alarm_state(scores, tau, k)
    fired = np.where(st)[0]
    if len(fired) == 0:
        return np.nan, None
    e = int(min(ends[fired[0]], len(dates) - 1))
    return float(onset_idx - e), dates.iloc[e].date().isoformat()


def holm(pvals):
    """Holm correction of the p-values."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    prev = 0.0
    for rank, i in enumerate(order):
        v = (m - rank) * p[i]
        prev = max(prev, min(v, 1.0))
        adj[i] = prev
    return adj


def sign_test(diff):
    """Two-sided sign test; returns (wins, losses, ties, p)."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    w = int((d > 0).sum())
    l = int((d < 0).sum())
    t = int((d == 0).sum())
    n = w + l
    if n == 0:
        return w, l, t, np.nan
    from math import comb
    k = min(w, l)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return w, l, t, float(min(1.0, 2.0 * tail))


# ---------------------------------------------------------------------------
# main steps
# ---------------------------------------------------------------------------
def discover_methods():
    """Methods with results for the control and at least one episode."""
    found = set()
    for asset in ASSETS:
        base = os.path.join(CONTROL, asset)
        if not os.path.isdir(base):
            continue
        for m in sorted(os.listdir(base)):
            if m not in PAPER_METHODS:
                continue
            mp = os.path.join(base, m)
            if not os.path.isdir(mp):
                continue
            if os.path.exists(os.path.join(mp, "test_scores.csv")):
                found.add((m, "-"))
            for g in sorted(os.listdir(mp)):
                if os.path.exists(os.path.join(mp, g, "test_scores.csv")):
                    found.add((m, g))
    return sorted(found)


def build(fars, ks):
    methods = discover_methods()
    print("Methods with complete results: %d" % len(methods))

    date_cache = {}
    rows = []
    for ep, (sub, onset_str, dsub) in EPISODES.items():
        root = os.path.join(RUN, sub)
        if not os.path.isdir(root):
            print("  skipped %s (%s not found)" % (ep, sub))
            continue
        for asset in ASSETS:
            key = (asset, dsub)
            if key not in date_cache:
                try:
                    date_cache[key] = load_dates(asset, dsub)
                except Exception as exc:
                    print("  could not read dates %s %s: %s" % (asset, dsub, exc))
                    date_cache[key] = None
            dates = date_cache[key]
            if dates is None:
                continue
            hit = np.where(dates >= pd.Timestamp(onset_str))[0]
            if len(hit) == 0:
                continue
            onset_idx = int(hit.min())

            for method, mapping in methods:
                ev = read_scores(root, asset, method, mapping)
                ct = read_scores(CONTROL, asset, method, mapping)
                if ev is None or ct is None:
                    continue
                ev_s, ev_e = ev
                ct_s, _ = ct
                for f, k in product(fars, ks):
                    tau, achieved = tau_for_far(ct_s, f, k)
                    if not np.isfinite(tau):
                        continue
                    lead, fd = lead_days(ev_s, ev_e, tau, k, onset_idx, dates)
                    rows.append({
                        "episode": ep, "asset": asset, "method": method,
                        "mapping": mapping, "target_far": f, "k": k,
                        "tau": round(tau, 6), "control_far": round(achieved, 4),
                        "first_alarm": fd, "lead_days": lead,
                        "event_alarm_rate": round(
                            float(alarm_state(ev_s, tau, k).mean()), 4),
                        "n_event_windows": len(ev_s),
                        "n_control_windows": len(ct_s),
                    })
    return pd.DataFrame(rows)


def compare(df, fars, ks):
    """Pair ExpoGAF against every baseline at a matched false-alarm rate."""
    out = []
    for f, k in product(fars, ks):
        sub = df[(df["target_far"] == f) & (df["k"] == k)]
        core = sub[(sub["method"] == CORE[0]) & (sub["mapping"] == CORE[1])]
        core = core.set_index(["episode", "asset"])["lead_days"]
        if core.empty:
            continue
        pv, recs = [], []
        for (m, g), grp in sub.groupby(["method", "mapping"]):
            if (m, g) == CORE:
                continue
            other = grp.set_index(["episode", "asset"])["lead_days"]
            common = core.index.intersection(other.index)
            if len(common) == 0:
                continue
            d = (core.loc[common] - other.loc[common]).to_numpy(float)
            w, l, t, p = sign_test(d)
            recs.append({"target_far": f, "k": k, "baseline": "%s/%s" % (m, g),
                         "n_cells": len(common), "expogaf_earlier": w,
                         "baseline_earlier": l, "tie": t,
                         "median_lead_gap_days": round(float(np.nanmedian(d)), 1),
                         "p_sign": round(p, 4) if np.isfinite(p) else None})
            pv.append(p if np.isfinite(p) else 1.0)
        if recs:
            adj = holm(pv)
            for r, a in zip(recs, adj):
                r["p_holm"] = round(float(a), 4)
            out.extend(recs)
    return pd.DataFrame(out)


def ablation_table(df):
    """Which component adds lead time, from stored results, nothing retrained."""
    sub = df[df["method"].isin(ABLATION)]
    sub = sub[(sub["mapping"] == "exponential") | (sub["mapping"] == "-")]
    if sub.empty:
        return pd.DataFrame()
    return sub.pivot_table(index=["target_far", "k", "episode", "asset"],
                           columns="method", values="lead_days").reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--far", type=float, nargs="+", default=[0.05, 0.10, 0.20],
                    help="target false-alarm rate on the control")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3],
                    help="consecutive windows above threshold to count an alarm")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("=" * 78)
    print("Lead time at a matched false-alarm rate")
    print("=" * 78)

    df = build(a.far, a.k)
    if df.empty:
        print("no results; check that the results directories are present")
        return
    df.to_csv(os.path.join(OUT, "leadtime_far_matched.csv"), index=False)
    print("Wrote leadtime_far_matched.csv, %d rows" % len(df))

    cmp_df = compare(df, a.far, a.k)
    if not cmp_df.empty:
        cmp_df.to_csv(os.path.join(OUT, "paired_tests.csv"), index=False)
        print("Wrote paired_tests.csv, %d rows" % len(cmp_df))

    ab = ablation_table(df)
    if not ab.empty:
        ab.to_csv(os.path.join(OUT, "component_ablation.csv"), index=False)
        print("Wrote component_ablation.csv, %d rows" % len(ab))

    # summary at the headline operating point
    f0, k0 = a.far[len(a.far) // 2], a.k[0]
    print()
    print("-" * 78)
    print("Summary at false-alarm rate %.2f and k=%d" % (f0, k0))
    print("-" * 78)
    s = df[(df["target_far"] == f0) & (df["k"] == k0)]
    piv = s.pivot_table(index=["method", "mapping"], values="lead_days",
                        aggfunc=["median", "mean", "count"])
    piv.columns = ["lead_median", "lead_mean", "n"]
    print(piv.sort_values("lead_median", ascending=False).round(1).to_string())

    if not cmp_df.empty:
        print()
        print("-" * 78)
        print("ExpoGAF against the baselines at FAR %.2f and k=%d" % (f0, k0))
        print("-" * 78)
        c = cmp_df[(cmp_df["target_far"] == f0) & (cmp_df["k"] == k0)]
        print(c.sort_values("p_holm").to_string(index=False))

    print()
    print("Output written to", OUT)


if __name__ == "__main__":
    main()
