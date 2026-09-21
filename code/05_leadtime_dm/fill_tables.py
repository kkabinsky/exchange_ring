# -*- coding: utf-8 -*-
"""Extract F1 grid, bond table, and lead-time consistency from completed fanoGAN runs."""
import csv, os, math, statistics

# replication package paths
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE = os.path.join(_REPO, "output", "04_cross_market")       # equity-index cells
BOND = os.path.join(_REPO, "output", "03_bond_anomaly")       # bond runs
LEAD = os.path.join(_REPO, "output", "05_leadtime_dm", "saved")  # lead_time_economic_summary.csv used for Table 10
M = ["cosine", "arctan", "arccosh", "exponential"]

def read_summary(folder):
    """folder/out_put_four_gaf_fanogan/comparison_summary.csv -> {method: {f1, auc}}"""
    p = os.path.join(BASE, folder, "out_put_four_gaf_fanogan", "comparison_summary.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            def fnum(x):
                return float(x) if x not in ("", None) else None
            out[r["method"]] = {"f1": fnum(r["f1"]), "auc": fnum(r["auc"])}
    return out

# ---------- F1 grid (9 equity cells x 4 methods) ----------
CELLS = [
    ("U30",  "COVID-19",            "DJIA_U30_COVID19_market_crash"),
    ("U30",  "Russia--Ukraine",     "DJIA_U30_Russia_Ukraine_war_crash"),
    ("U30",  "Chinese real-estate", "DJIA_U30_Chinese_real_asset_market_crash"),
    ("U500", "COVID-19",            "SP500_U500_COVID19_market_crash"),
    ("U500", "Russia--Ukraine",     "SP500_U500_Russia_Ukraine_war_crash"),
    ("U500", "Chinese real-estate", "SP500_U500_Chinese_real_asset_market_crash"),
    ("DAX",  "COVID-19",            "DAX_INDEX_COVID19_market_crash"),
    ("DAX",  "Russia--Ukraine",     "DAX_INDEX_Russia_Ukraine_war_crash"),
    ("DAX",  "Chinese real-estate", "DAX_INDEX_Chinese_real_asset_market_crash"),
]
print("=" * 70)
print("TAB:FULL4  F1 grid (per-row best in bold)")
print("=" * 70)
for idx, crash, folder in CELLS:
    s = read_summary(folder)
    vals = {m: s.get(m, {}).get("f1") for m in M}
    present = [v for v in vals.values() if v is not None]
    best = max(present) if present else None
    cells = []
    for m in M:
        v = vals[m]
        if v is None:
            cells.append("\\TODO{}")
        elif best is not None and abs(v - best) < 1e-12:
            cells.append("\\best{%.3f}" % v)
        else:
            cells.append("%.3f" % v)
    status = "OK" if len(present) == 4 else f"MISSING {[m for m in M if vals[m] is None]}"
    print(f"{idx:5s}& {crash:20s}& " + " & ".join(cells) + f" \\\\   % {status}")

# ---------- Bond table ----------
BONDS = [
    ("US1M",  os.path.join(BOND, "bond_1m_30y_compare_e20", "standalone_fanogan", "US1M_BOND_COVID200_daily")),
    ("US30Y", os.path.join(BOND, "bond_1m_30y_compare_e20", "standalone_fanogan", "US30Y_BOND_COVID200_daily")),
    ("US10Y", os.path.join(BOND, "bond_3m_1y_5y_10y_20y_compare_e20_daily", "standalone_fanogan", "US10Y_BOND_COVID200_daily")),
]

def dm_exp_vs_cos(folder):
    """DM test: squared error of min-max normalized score vs crash label. +ve => exp better."""
    def load(method):
        p = os.path.join(BASE, folder, "out_put_four_gaf_fanogan", method, "test_scores.csv")
        sc, lab = [], []
        with open(p, newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                sc.append(float(r["anomaly_score"])); lab.append(int(r["label_0normal_1crash"]))
        lo, hi = min(sc), max(sc)
        sn = [(x - lo) / (hi - lo) if hi > lo else 0.0 for x in sc]
        return [(s - y) ** 2 for s, y in zip(sn, lab)]
    Le, Lc = load("exponential"), load("cosine")
    d = [c - e for c, e in zip(Lc, Le)]      # +ve => exp lower loss
    T = len(d); dbar = sum(d) / T
    var = statistics.pvariance(d) / T
    dm = dbar / math.sqrt(var) if var > 0 else float("nan")
    # one-sided p that exp is better (dm>0)
    p = 0.5 * math.erfc(dm / math.sqrt(2)) if dm == dm else float("nan")
    winner = "exponential" if dbar > 0 else "cosine"
    return dm, p, winner

print("\n" + "=" * 70)
print("TAB:BOND  exp AUC / F1-best / DM(exp vs cos)")
print("=" * 70)
for name, folder in BONDS:
    s = read_summary(folder)
    exp_auc = s["exponential"]["auc"]
    f1best = max(M, key=lambda m: s[m]["f1"])
    dm, p, win = dm_exp_vs_cos(folder)
    pwin = 0.5 * math.erfc(abs(dm) / math.sqrt(2))
    f1max = max(s[m]["f1"] for m in M)
    f1best_disp = "none" if f1max < 1e-9 else f1best
    print(f"{name:6s}& {exp_auc:.3f} & {f1best_disp:11s}& {win:11s}& dm={dm:+.3f} p_winner={pwin:.4f} \\\\   "
          f"% F1: " + ", ".join(f"{m}={s[m]['f1']:.3f}" for m in M))

# ---------- Lead-time consistency ----------
print("\n" + "=" * 70)
print("LEAD-TIME consistency (from lead_time_economic_summary.csv)")
print("=" * 70)
lead = {m: [] for m in M}
percell = {}
with open(os.path.join(LEAD, "lead_time_economic_summary.csv"), newline="", encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        key = (r["dataset"], r["scenario"])
        lt = r["lead_time_days"]
        val = float(lt) if lt not in ("", None) else None
        percell.setdefault(key, {})[r["method"]] = val
        if val is not None:
            lead[r["method"]].append(val)
for m in M:
    xs = lead[m]
    mean = statistics.mean(xs); sd = statistics.stdev(xs)
    print(f"  {m:11s} n={len(xs)} mean={mean:6.2f} sd={sd:5.2f} CV={sd/abs(mean):.3f} worst={min(xs):.0f}")
# mean rank (higher lead = better -> rank 4 best)
ranks = {m: [] for m in M}
for key, d in percell.items():
    avail = {m: d[m] for m in M if d.get(m) is not None}
    if len(avail) < 2: continue
    order = sorted(avail, key=lambda m: avail[m])   # ascending
    for pos, m in enumerate(order):
        ranks[m].append(pos + 1)                    # 1=worst .. 4=best
print("  mean rank (4=best):", {m: round(statistics.mean(ranks[m]), 2) for m in M if ranks[m]})
print("  cells with lead-time:", sorted(percell.keys()))
