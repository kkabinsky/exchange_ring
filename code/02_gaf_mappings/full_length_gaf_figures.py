# -*- coding: utf-8 -*-
"""Full-length GAF examples (image size = full series length, NO rolling window)
with an exponential polar-encoding column, for three dataset groups:
equity indices, original benchmark assets, and US Treasury yields.
Style matches the supplied yield figure (viridis, 6 columns)."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.simplefilter("ignore")

# --- replication package (added): data in <repo>/input, results in <repo>/output/02_gaf_mappings ---
import os as _rp_os, sys as _rp_sys
from pathlib import Path as _RpPath
REPO_ROOT = _RpPath(__file__).resolve().parents[2]
REPO_INPUT = REPO_ROOT / "input"
REPO_OUTPUT = REPO_ROOT / "output" / "02_gaf_mappings"
REPO_OUTPUT.mkdir(parents=True, exist_ok=True)
_rp_sys.path.insert(0, str(REPO_ROOT / "code" / "common"))
_rp_os.chdir(REPO_OUTPUT)  # original script writes into the working directory
# ------------------------------------------------------------------------------------------
EXC = str(REPO_INPUT) + "/"
DISP_CAP = 1400  # display raster cap (GAF still COMPUTED at full length)


def minmax01(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    return np.zeros_like(x) if hi <= lo else (x - lo) / (hi - lo)


def phi_cosine(x):      return np.arccos(np.clip(2 * minmax01(x) - 1, -1, 1))
def phi_arctan(x):      return np.arctan(minmax01(x))
def phi_arccosh(x):     return np.arccosh(1 + minmax01(x))
def phi_exponential(x): return np.pi * (np.exp(minmax01(x)) - 1) / (np.e - 1)

PHIS = [("Cosine GAF", phi_cosine), ("Arctan GAF", phi_arctan),
        ("Arccosh GAF", phi_arccosh), ("Exponential GAF", phi_exponential)]


def gaf(phi):
    return np.cos(phi[:, None] + phi[None, :])


def disp(mat):
    n = mat.shape[0]
    if n <= DISP_CAP:
        return mat
    step = int(np.ceil(n / DISP_CAP))
    return mat[::step, ::step]


def make_figure(title, value_label, series, outfile, note):
    names = list(series.keys())
    nrows, ncols = len(names), 6
    fig = plt.figure(figsize=(ncols * 2.25, nrows * 2.25))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.18, wspace=0.12)
    col_titles = [value_label, "Exponential polar"] + [t for t, _ in PHIS]

    for r, name in enumerate(names):
        x = np.asarray(series[name], dtype=float)
        n = len(x)
        # col 0: raw series line
        ax = fig.add_subplot(gs[r, 0])
        ax.plot(x, color="black", lw=0.6)
        ax.set_xticks([])
        ax.set_ylabel(name, fontsize=12, fontweight="bold")
        if r == 0:
            ax.set_title(col_titles[0], fontsize=12, fontweight="bold")
        # col 1: exponential polar encoding (theta = exp angle, r = value, colour = time)
        axp = fig.add_subplot(gs[r, 1], projection="polar")
        xt = minmax01(x)
        theta = np.pi * (np.exp(xt) - 1) / (np.e - 1)
        axp.scatter(theta, xt, c=np.arange(n), cmap="viridis", s=2, alpha=0.8)
        axp.set_xticklabels([]); axp.set_yticklabels([])
        axp.set_ylim(0, 1)
        if r == 0:
            axp.set_title(col_titles[1], fontsize=12, fontweight="bold")
        # cols 2-5: GAF images (image size = full series length)
        for c, (tname, pfn) in enumerate(PHIS, start=2):
            axg = fig.add_subplot(gs[r, c])
            axg.imshow(disp(gaf(pfn(x))), cmap="viridis", vmin=-1, vmax=1,
                       interpolation="nearest", aspect="equal")
            axg.set_xticks([]); axg.set_yticks([])
            if r == 0:
                axg.set_title(tname, fontsize=12, fontweight="bold")

    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, 0.005, note, ha="center", fontsize=9)
    plt.savefig(outfile, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outfile}  ({nrows} rows; n=" +
          ", ".join(f"{k}:{len(v)}" for k, v in series.items()) + ")")


def load_simple(path, date_col, val_col):
    df = pd.read_excel(path, sheet_name=0)
    df = df[[date_col, val_col]].dropna()
    return df[val_col].to_numpy(dtype=float)


def main():
    # ---- equities ----
    eq = {
        "U30":  load_simple(EXC + "u30_1day.xlsx", "date", "closed"),
        "U500": load_simple(EXC + "u500_1day.xlsx", "date", "closed"),
        "DAX":  load_simple(EXC + "dax_1day.xlsx", "date", "closed"),
    }
    make_figure("Full-length equity-index GAF examples with polar encoding",
                "Price series", eq, "full_gaf_equities.jpg",
                "Each row uses the complete daily closing series (no rolling window); "
                "the GAF image size equals the full series length.")

    # ---- original benchmark assets ----
    asset = {
        "USOIL":  load_simple(EXC + "USOIL_daily.xlsx", "Date", "Close"),
        "GOLD":   load_simple(EXC + "GOLD_daily.xlsx", "Date", "Close"),
        "EURUSD": load_simple(EXC + "EURUSD_daily.xlsx", "Date", "Close"),
    }
    make_figure("Full-length benchmark-asset GAF examples with polar encoding",
                "Price series", asset, "full_gaf_assets.jpg",
                "Each row uses the complete daily closing series (no rolling window); "
                "the GAF image size equals the full series length.")

    # ---- US Treasury yields (match supplied figure: 2015-05-06..2020-12-14) ----
    yc = pd.read_excel(EXC + "combine_TTM.xlsx", sheet_name="Ycurve")
    yc["Date"] = pd.to_datetime(yc["Date"])
    yc = yc[(yc.Date >= "2015-05-06") & (yc.Date <= "2020-12-14")]
    mats = [("US1M", "1M"), ("US3M", "3M"), ("US1Y", "1Y"), ("US2Y", "2Y"),
            ("US5Y", "5Y"), ("US10Y", "10Y"), ("US20Y", "20Y"), ("US30Y", "30Y")]
    yld = {lab: yc[col].dropna().to_numpy(dtype=float) for lab, col in mats}
    make_figure("Full-length US Treasury yield GAF examples with polar encoding",
                "Yield series", yld, "full_gaf_yields.jpg",
                "Each row uses the complete daily series from 2015-05-06 to 2020-12-14 "
                "(no rolling window); the GAF image size equals the full series length.")


if __name__ == "__main__":
    main()
