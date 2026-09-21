"""Empirical yield-curve inversion: US 1-month vs 30-year, straight from the
combine_TTM.xlsx zero-coupon yield curve. Inverted = (30Y - 1M) < 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- replication package (added): data in <repo>/input, results in <repo>/output/01_theory_ns_var ---
import os as _rp_os, sys as _rp_sys
from pathlib import Path as _RpPath
REPO_ROOT = _RpPath(__file__).resolve().parents[2]
REPO_INPUT = REPO_ROOT / "input"
REPO_OUTPUT = REPO_ROOT / "output" / "01_theory_ns_var"
REPO_OUTPUT.mkdir(parents=True, exist_ok=True)
_rp_sys.path.insert(0, str(REPO_ROOT / "code" / "common"))
_rp_os.chdir(REPO_OUTPUT)  # original script writes into the working directory
# ------------------------------------------------------------------------------------------
SRC = REPO_INPUT / "combine_TTM.xlsx"


def main() -> None:
    df = pd.read_excel(SRC, sheet_name="Ycurve")
    df["Date"] = pd.to_datetime(df["Date"])
    d = df[["Date", "1M", "30Y"]].dropna().sort_values("Date").reset_index(drop=True)
    d["spread_30Y_1M"] = d["30Y"] - d["1M"]
    d["inverted"] = d["spread_30Y_1M"] < 0
    d.to_csv("empirical_invertible_yield.csv", index=False)

    # contiguous inversion episodes
    inv = d["inverted"].to_numpy()
    episodes = []
    i = 0
    while i < len(inv):
        if inv[i]:
            j = i
            while j + 1 < len(inv) and inv[j + 1]:
                j += 1
            episodes.append((d["Date"].iloc[i], d["Date"].iloc[j], j - i + 1,
                             float(d["spread_30Y_1M"].iloc[i:j + 1].min())))
            i = j + 1
        else:
            i += 1
    print(f"coverage: {d.Date.min().date()} .. {d.Date.max().date()}  (n={len(d)})")
    print(f"inverted days: {int(inv.sum())} ({100*inv.mean():.1f}% of sample)")
    print("\nInversion episodes (30Y < 1M):")
    ep = pd.DataFrame(episodes, columns=["start", "end", "trading_days", "max_depth_pct"])
    # pd.to_datetime keeps .dt valid when there are no episodes (30Y never below 1M in this sample)
    ep["start"] = pd.to_datetime(ep["start"]).dt.date
    ep["end"] = pd.to_datetime(ep["end"]).dt.date
    print(ep.to_string(index=False))
    ep.to_csv("empirical_inversion_episodes.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(d["Date"], d["30Y"], color="#185FA5", lw=1.3, label="30Y yield")
    ax1.plot(d["Date"], d["1M"], color="#A32D2D", lw=1.3, label="1M yield")
    ax1.fill_between(d["Date"], d["1M"], d["30Y"], where=d["inverted"],
                     color="#F09595", alpha=0.5, label="inverted (1M > 30Y)")
    ax1.set_ylabel("yield (%)"); ax1.set_title("US Treasury yields: 1-month vs 30-year (combine_TTM)")
    ax1.grid(alpha=0.3); ax1.legend(loc="upper right")

    ax2.plot(d["Date"], d["spread_30Y_1M"], color="#0F6E56", lw=1.0)
    ax2.axhline(0, color="#A32D2D", ls="--", lw=1.0)
    ax2.fill_between(d["Date"], 0, d["spread_30Y_1M"], where=d["inverted"],
                     color="#F09595", alpha=0.5)
    ax2.set_ylabel("30Y − 1M (%)"); ax2.set_xlabel("Date")
    ax2.set_title("Term spread (negative = inverted)")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("empirical_invertible_yield.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: empirical_invertible_yield.csv, empirical_inversion_episodes.csv, "
          "empirical_invertible_yield.jpg (300 dpi)")


if __name__ == "__main__":
    main()
