# -*- coding: utf-8 -*-
"""Build the 4-method early-warning lead-time table.
Reads the authoritative lead_time_economic_summary.csv for the 8 cells it already
covers, and computes the missing DAX x Chinese cell directly from its (completed)
test_scores.csv, validating the day-count convention against DAX x Russia.
Read-only: touches only finished cells, not the ones currently retraining."""
import csv, sys
from pathlib import Path

# replication package paths
REPO_ROOT = Path(__file__).resolve().parents[2]
MC = REPO_ROOT / "output" / "04_cross_market"          # saved f-AnoGAN cell results
LEAD_DIR = REPO_ROOT / "output" / "05_leadtime_dm" / "saved"  # lead_time_economic_summary.csv used for Table 10
sys.path.insert(0, str(REPO_ROOT / "code" / "04_cross_market"))
sys.path.insert(0, str(REPO_ROOT / "code" / "common"))
import run_market_crash_3datasets_oneclick as ock

W = ock.WINDOW_SIZE
M = ["cosine", "arctan", "arccosh", "exponential"]
DAX = next(d for d in ock.DATASETS if d["code"] == "DAX_INDEX")

def frame_positions(onset_iso):
    frame = ock.load_dataset_frame(DAX)
    import pandas as pd
    onset = pd.Timestamp(onset_iso)
    pos = frame.index[frame["Date"] >= onset]
    onset_pos = int(pos[0]) if len(pos) else None
    dates = frame["Date"].tolist()
    return dates, onset_pos

def first_signal_start(cell, method):
    p = MC / cell / "out_put_four_gaf_fanogan" / method / "test_scores.csv"
    if not p.exists():
        return None
    with open(p, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            if int(float(r["predicted_label"])) == 1:
                return int(r["window_start"])
    return None  # no signal fired

def lead_for_cell(cell, onset_iso):
    dates, onset_pos = frame_positions(onset_iso)
    out = {}
    for m in M:
        s = first_signal_start(cell, m)
        if s is None or onset_pos is None:
            out[m] = (None, None, None); continue
        close_pos = s + W - 1
        lead_start = onset_pos - s
        lead_close = onset_pos - close_pos
        out[m] = (lead_start, lead_close, dates[close_pos].date().isoformat())
    return out

# ---- validation against DAX x Russia (in the CSV) ----
csv_lead = {}
with open(LEAD_DIR / "lead_time_economic_summary.csv", newline="", encoding="utf-8-sig") as fh:
    for r in csv.DictReader(fh):
        v = r["lead_time_days"]
        csv_lead[(r["dataset"], r["scenario"], r["method"])] = (float(v) if v not in ("", None) else None)

print("=== VALIDATION: DAX x Russia (my compute vs CSV) ===")
val = lead_for_cell("DAX_INDEX_Russia_Ukraine_war_crash", "2022-02-11")
conv_close_ok = True
for m in M:
    ls, lc, d = val[m]
    cref = csv_lead.get(("DAX_INDEX", "Russia_Ukraine_war_crash", m))
    print(f"  {m:11s} csv={cref}  lead_start={ls}  lead_close={lc}  signal_close={d}")
    if cref is not None and lc is not None and abs(cref - lc) > 0.5:
        conv_close_ok = False
print(f"  --> 'close' convention matches CSV: {conv_close_ok}")
conv = "close" if conv_close_ok else "start"

print("\n=== DAX x Chinese (computed, convention=%s) ===" % conv)
dax_cn = lead_for_cell("DAX_INDEX_Chinese_real_asset_market_crash", "2023-08-07")
for m in M:
    ls, lc, d = dax_cn[m]
    val_use = lc if conv == "close" else ls
    print(f"  {m:11s} lead={val_use}  signal_close={d}")

# ---- assemble full 4-method table (8 cells from CSV + DAX Chinese computed) ----
CELLS = [
    ("U30",  "COVID-19",            "DJIA_U30",  "COVID19_market_crash"),
    ("U30",  "Russia--Ukraine",     "DJIA_U30",  "Russia_Ukraine_war_crash"),
    ("U30",  "Chinese real-estate", "DJIA_U30",  "Chinese_real_asset_market_crash"),
    ("U500", "COVID-19",            "SP500_U500","COVID19_market_crash"),
    ("U500", "Russia--Ukraine",     "SP500_U500","Russia_Ukraine_war_crash"),
    ("U500", "Chinese real-estate", "SP500_U500","Chinese_real_asset_market_crash"),
    ("DAX",  "COVID-19",            "DAX_INDEX", "COVID19_market_crash"),
    ("DAX",  "Russia--Ukraine",     "DAX_INDEX", "Russia_Ukraine_war_crash"),
    ("DAX",  "Chinese real-estate", "DAX_INDEX", "Chinese_real_asset_market_crash"),
]
def fmt(v):
    if v is None: return "n/s"           # no signal
    return ("$%d$" % v) if v < 0 else "%d" % v

print("\n=== TAB: lead-time, ALL 4 mappings (cosine, arctan, arccosh, exponential) ===")
for idx, crash, dc, sc in CELLS:
    row = {}
    for m in M:
        if (dc == "DAX_INDEX" and sc == "Chinese_real_asset_market_crash"):
            ls, lc, _ = dax_cn[m]; row[m] = lc if conv == "close" else ls
        else:
            row[m] = csv_lead.get((dc, sc, m))
    cells = " & ".join(fmt(row[m]) for m in M)
    print(f"{idx:5s}& {crash:20s}& {cells} \\\\")
