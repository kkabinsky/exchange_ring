"""Level-IRF of the yield curve to a 3m (nilpotent) shock, swept +10%..+50%.

X = [10y, 2y, 3m],  X_{t+1} = A X_t,
A = [[0.98,0,0],[0.80,0,1],[0,0,0]],  baseline X0 = [2.0, 3.0, 2.5] (%).
A 3m shock of +p scales the period-1 3m level to 2.5*(1+p).

Closed forms (N^2 = 0):
    10y_t   = 2.0 * 0.98^{t-1}
    2y_{t+1}= 0.8*10y_t + 3m_t      ->  2y_2 = 0.8*2.0 + 3m_shock = 1.6 + 3m_shock
    3m_t    = 0  for t >= 2
Invertible curve at period 2 (2y_2 > 10y_2) iff  3m_shock > 10y_2 - 1.6 = 0.36.
"""
from __future__ import annotations

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

A = np.array([[0.98, 0.0, 0.0],
              [0.80, 0.0, 1.0],
              [0.00, 0.0, 0.0]])
BASE = np.array([2.0, 3.0, 2.5])
SHOCKS = [0.10, 0.20, 0.30, 0.40, 0.50]
NPER = 6


def irf_levels(shock_pct: float) -> pd.DataFrame:
    x = BASE.copy()
    x[2] = BASE[2] * (1.0 + shock_pct)     # 3m shocked at period 1
    rows = []
    for t in range(1, NPER + 1):
        rows.append({"period": t, "10y": x[0], "2y": x[1], "3m": x[2],
                     "spread_2y_10y": x[1] - x[0], "inverted": x[1] > x[0]})
        x = A @ x
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 200)
    # full table: 2y level per period per shock (10y, 3m shown once since shared)
    table = pd.DataFrame({"period": range(1, NPER + 1)})
    ref = irf_levels(SHOCKS[0])
    table["10y"] = ref["10y"].round(2)
    for p in SHOCKS:
        d = irf_levels(p)
        table[f"2y_+{int(p*100)}%"] = d["2y"].round(2)
    print("=== 2y level by period and 3m-shock size (10y is shock-independent) ===")
    print(table.to_string(index=False))

    print("\n=== Inversion at period 2 (2y_2 vs 10y_2 = 1.96) ===")
    inv = []
    for p in SHOCKS:
        d = irf_levels(p)
        two_p2 = d.loc[d.period == 2, "2y"].iloc[0]
        ten_p2 = d.loc[d.period == 2, "10y"].iloc[0]
        inv.append({"shock_3m": f"+{int(p*100)}%", "3m_level": round(BASE[2]*(1+p), 2),
                    "2y_p2": round(two_p2, 2), "10y_p2": round(ten_p2, 2),
                    "spread_p2": round(two_p2 - ten_p2, 2), "inverted": two_p2 > ten_p2})
    invdf = pd.DataFrame(inv)
    print(invdf.to_string(index=False))
    table.to_csv("var_3m_shock_levels.csv", index=False)
    invdf.to_csv("var_3m_shock_inversion.csv", index=False)

    # plot: 2y at period 2 vs shock size (linear), plus 10y_2 reference line
    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = [int(p * 100) for p in SHOCKS]
    ys = [irf_levels(p).loc[1, "2y"] for p in SHOCKS]
    ax.plot(xs, ys, "o-", color="#185FA5", lw=2, label="$2y$ at period 2 (linear in shock)")
    ax.axhline(1.96, color="#BA7517", ls="--", lw=1.5, label="$10y$ at period 2 = 1.96")
    ax.fill_between(xs, 1.96, ys, where=[y > 1.96 for y in ys],
                    color="#F09595", alpha=0.4, label="inverted ($2y>10y$)")
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=9)
    ax.set_title("Nilpotent 3m shock: period-2 $2y$ response is linear in shock size")
    ax.set_xlabel("3m shock size (%)"); ax.set_ylabel("$2y$ yield at period 2 (%)")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig("var_3m_shock_levels.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: var_3m_shock_levels.csv, var_3m_shock_inversion.csv, "
          "var_3m_shock_levels.jpg (300 dpi)")


if __name__ == "__main__":
    main()
