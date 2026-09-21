"""VAR(1) impulse-response simulation of the yield curve, with Jordan-form
intuition (nilpotent vs semisimple modes), inversion detection, and the
Estrella-Mishkin (1998) probit recession probability.

State  X_t = [10y, 2y, 3m]'
Law    X_{t+1} = A X_t          (period 1 = the shock impulse X_0)

A = [[0.98, 0, 0],   # 10y: slow geometric decay  -> SEMISIMPLE eigenvalue 0.98
     [0.80, 0, 1],   # 2y : driven by 10y and 3m
     [0.00, 0, 0]]   # 3m : dies in one step       -> NILPOTENT block, N^2 = 0

The (2y,3m) sub-block [[0,1],[0,0]] is nilpotent (square = 0): a 3m shock
propagates to 2y for exactly one period, then the whole transient vanishes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from math import erf, sqrt
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
LABELS = ["10y", "2y", "3m"]
N = 20

# Estrella & Mishkin (1998) probit:  P(recession) = Phi(alpha + beta * spread)
ALPHA, BETA = -0.53, -1.03
def Phi(z): return 0.5 * (1.0 + erf(z / sqrt(2.0)))

# Normal (upward-sloping) baseline curve, in %
BASE = {"10y": 3.20, "2y": 2.80, "3m": 2.50}


def irf(x0: np.ndarray) -> pd.DataFrame:
    rows, x = [], x0.astype(float).copy()
    for t in range(1, N + 1):
        rows.append({"period": t, "10y": x[0], "2y": x[1], "3m": x[2]})
        x = A @ x
    return pd.DataFrame(rows)


def levels_and_recession(dev: pd.DataFrame) -> pd.DataFrame:
    out = dev.copy()
    for k in LABELS:
        out[f"{k}_lvl"] = BASE[k] + out[k]
    out["spread_10y_2y"] = out["10y_lvl"] - out["2y_lvl"]
    out["inverted"] = out["spread_10y_2y"] < 0
    out["prob_recession"] = out["spread_10y_2y"].apply(
        lambda s: Phi(ALPHA + BETA * s))
    return out


def closed_form_prob(eps_pct: float, ten_y0: float = 2.0, n: int = N) -> pd.DataFrame:
    """Closed-form recession probability after a 3m (nilpotent) shock.

    Since N^2 = 0, for t >= 2 the 3m channel is dead and the spread reduces to a
    pure 0.98^{t-1} decay of the 10y term:
        spread_t = -0.8 * 0.98^{t-1} * 10y0 - eps * 1[t==1]
        Prob_t   = Phi(alpha + beta*spread_t)
                 = Phi(-0.53 + 1.03*0.8*0.98^{t-1}*10y0 + 1.03*eps*1[t==1])
    """
    rows = []
    for t in range(1, n + 1):
        spread = -0.8 * (0.98 ** (t - 1)) * ten_y0 - (eps_pct if t == 1 else 0.0)
        rows.append({"period": t, "spread_10y_2y": spread,
                     "prob_recession": Phi(ALPHA + BETA * spread)})
    return pd.DataFrame(rows)


def sensitivity(ten_y0: float = 2.0) -> pd.DataFrame:
    """Prob at period 1 (impulse) and the persistent period-2+ level vs shock size."""
    rows = []
    for bps in range(0, 151, 10):
        eps = bps / 100.0  # bps -> percent
        p1 = closed_form_prob(eps, ten_y0).iloc[0]["prob_recession"]
        p2 = closed_form_prob(eps, ten_y0).iloc[1]["prob_recession"]
        rows.append({"shock_bps": bps, "shock_pct": eps,
                     "prob_t1": p1, "prob_t2plus": p2})
    return pd.DataFrame(rows)


def main() -> None:
    pd.set_option("display.width", 160)
    shock_3m = irf(np.array([0.0, 0.0, 1.0]))   # nilpotent impulse
    shock_10y = irf(np.array([1.0, 0.0, 0.0]))  # semisimple impulse

    print("=== IRF deviations: 3m shock +1% (nilpotent) ===")
    print(shock_3m.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== IRF deviations: 10y shock +1% (semisimple) ===")
    print(shock_10y.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    lvl = levels_and_recession(shock_3m)
    print("\n=== 3m shock: levels, 10y-2y spread, inversion, recession prob ===")
    show = lvl[["period", "10y_lvl", "2y_lvl", "3m_lvl",
                "spread_10y_2y", "inverted", "prob_recession"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    lvl.to_csv("irf_var_yield_curve.csv", index=False)

    # ---- figure: 3 panels ----
    fig, ax = plt.subplots(3, 1, figsize=(9, 10))
    for k, c in zip(LABELS, ["#185FA5", "#0F6E56", "#993C1D"]):
        ax[0].plot(shock_3m["period"], shock_3m[k], marker="o", label=k, color=c)
    ax[0].axhline(0, color="gray", lw=0.8)
    ax[0].set_title("IRF: +1% 3m shock (nilpotent) — 2y jumps 1 period, then dies")
    ax[0].set_ylabel("deviation (%)"); ax[0].legend(); ax[0].grid(alpha=0.3)

    for k, c in zip(LABELS, ["#185FA5", "#0F6E56", "#993C1D"]):
        ax[1].plot(shock_10y["period"], shock_10y[k], marker="o", label=k, color=c)
    ax[1].axhline(0, color="gray", lw=0.8)
    ax[1].set_title("IRF: +1% 10y shock (semisimple) — slow 0.98^t decay")
    ax[1].set_ylabel("deviation (%)"); ax[1].legend(); ax[1].grid(alpha=0.3)

    ax2 = ax[2]
    ax2.plot(lvl["period"], lvl["prob_recession"] * 100, marker="s", color="#A32D2D",
             label="P(recession) 12m ahead")
    ax2.axhline(30, color="#A32D2D", ls="--", lw=0.9, label="30% warning")
    ax2.fill_between(lvl["period"], 0, lvl["prob_recession"] * 100,
                     where=lvl["inverted"], color="#F09595", alpha=0.4,
                     label="inverted (10y<2y)")
    ax2.set_title("Estrella–Mishkin recession probability after 3m shock")
    ax2.set_xlabel("period"); ax2.set_ylabel("P(recession) %")
    ax2.set_ylim(0, 100); ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("irf_recession.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    sens = sensitivity(ten_y0=2.0)
    print("\n=== Sensitivity: 3m shock size (bps) vs recession probability ===")
    print(sens.assign(prob_t1=lambda d: (d.prob_t1*100).round(1),
                      prob_t2plus=lambda d: (d.prob_t2plus*100).round(1))
              .to_string(index=False))
    sens.to_csv("irf_sensitivity.csv", index=False)

    figs, axs = plt.subplots(figsize=(8, 4.5))
    axs.plot(sens["shock_bps"], sens["prob_t1"]*100, marker="o", color="#A32D2D",
             label="Period 1 (impulse)")
    axs.plot(sens["shock_bps"], sens["prob_t2plus"]*100, marker="s", color="#185FA5",
             label="Period 2+ (persistent, shock-independent)")
    axs.axhline(30, color="gray", ls="--", lw=0.9, label="30% warning")
    axs.set_title("Recession probability vs 3m shock size (nilpotent, closed form)")
    axs.set_xlabel("3m shock (bps)"); axs.set_ylabel("P(recession) %")
    axs.set_ylim(0, 100); axs.legend(); axs.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig("irf_sensitivity.jpg", dpi=300, bbox_inches="tight"); plt.close(figs)

    print("\nSaved: irf_var_yield_curve.csv, irf_sensitivity.csv, "
          "irf_recession.jpg, irf_sensitivity.jpg (300 dpi)")


if __name__ == "__main__":
    main()
