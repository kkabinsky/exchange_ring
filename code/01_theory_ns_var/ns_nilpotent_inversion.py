"""Yield-curve inversion from a nilpotent VAR shock, Nelson-Siegel maturity action.

Implements eqs (114)-(118):
    u(s) = e^{-lambda s} (I + s N^eps_t) u(0)          (114)
    u_S(s) = e^{-lambda s} ( u_S(0) + s*Gamma_t*u_L(0) ) (115, short)
    u_L(s) = e^{-lambda s} u_L(0)                        (115, long)
    spread(s) = u_S(s) - u_L(s)
              = e^{-lambda s} ( u_S(0) - u_L(0) + s*Gamma_t*u_L(0) )  (116)
Inversion (u_S > u_L) whenever  s*Gamma_t*u_L(0) > u_L(0) - u_S(0)   (117).
The s*e^{-lambda s} term is the Nelson-Siegel curvature footprint of the
long-short supply-demand shock.
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

LAMBDA = 0.5     # maturity decay
US0 = 1.0        # initial short component (normal curve: uS0 < uL0)
UL0 = 2.0        # initial long component
GAMMA = 0.8      # long-short nilpotent shock coupling Gamma_t


def curve(s: np.ndarray):
    decay = np.exp(-LAMBDA * s)
    uS = decay * (US0 + s * GAMMA * UL0)
    uL = decay * UL0
    return uS, uL


def main() -> None:
    s = np.linspace(0, 6, 241)
    uS, uL = curve(s)
    spread = uS - uL
    s_star = (UL0 - US0) / (GAMMA * UL0)   # inversion onset maturity (eq 117 equality)

    grid = np.array([0, 0.5, s_star, 1, 1.5, 2, 2.5, 3, 4, 5, 6])
    uSg, uLg = curve(grid)
    tbl = pd.DataFrame({
        "maturity_s": grid,
        "u_S(s)_short": uSg,
        "u_L(s)_long": uLg,
        "spread_S_minus_L": uSg - uLg,
        "inverted": (uSg - uLg) > 0,
    })
    pd.set_option("display.width", 160)
    print(f"params: lambda={LAMBDA}, uS0={US0}, uL0={UL0}, Gamma={GAMMA}")
    print(f"inversion onset maturity s* = (uL0-uS0)/(Gamma*uL0) = {s_star:.3f}\n")
    print(tbl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    tbl.to_csv("ns_nilpotent_inversion.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(s, uL, color="#BA7517", lw=2, label="$u_L(s)$ long (semisimple decay)")
    ax.plot(s, uS, color="#185FA5", lw=2, label="$u_S(s)$ short (nilpotent curvature)")
    ax.fill_between(s, uL, uS, where=(uS > uL), color="#F09595", alpha=0.45,
                    label="inversion zone ($u_S>u_L$)")
    ax.axvline(s_star, color="gray", ls="--", lw=0.9)
    ax.annotate(f"inversion onset s*={s_star:.2f}", xy=(s_star, 1.6),
                xytext=(s_star + 0.4, 1.9),
                arrowprops=dict(arrowstyle="->", color="gray"), fontsize=9)
    ax.set_title("Yield-curve inversion from nilpotent VAR shock (Nelson-Siegel action)")
    ax.set_xlabel("maturity $s$"); ax.set_ylabel("yield component")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig("ns_nilpotent_inversion.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- four-panel parameter sweep ----
    cases = [(0.4, 0.5, "A"), (1.6, 0.5, "B"), (0.8, 0.3, "C"), (0.8, 1.2, "D")]
    fig2, axes = plt.subplots(2, 2, figsize=(11, 8))
    for (g, lam, tag), ax in zip(cases, axes.ravel()):
        d = np.exp(-lam * s)
        a = d * (US0 + s * g * UL0)
        b = d * UL0
        sstar = (UL0 - US0) / (g * UL0)
        ax.plot(s, b, color="#BA7517", lw=2, label="$u_L(s)$")
        ax.plot(s, a, color="#185FA5", lw=2, label="$u_S(s)$")
        ax.fill_between(s, b, a, where=(a > b), color="#F09595", alpha=0.45)
        ax.axvline(sstar, color="gray", ls="--", lw=0.9)
        # panel label only (parameter details go in the LaTeX caption)
        ax.text(0.04, 0.93, f"({tag})", transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top")
        ax.set_xlabel("maturity $s$"); ax.set_ylabel("yield component")
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    plt.savefig("ns_inversion_four_params.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    print("\nSaved: ns_nilpotent_inversion.csv, ns_nilpotent_inversion.jpg, "
          "ns_inversion_four_params.jpg (300 dpi)")


if __name__ == "__main__":
    main()
