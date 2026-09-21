import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

# =========================
#  GLOBAL STYLE  (IEEE / Elsevier style)
# =========================
rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
        "lines.linewidth": 1.6,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)

# =========================
#  USER SETTINGS
# =========================
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
# input file: first command-line argument, else the file used for the saved figure
FILENAME = _rp_sys.argv[1] if len(_rp_sys.argv) > 1 else str(REPO_INPUT / "polar" / "cpall_final2.xlsx")
FIELD_NAME = "cp"

USE_PARTIAL_DATA = False
START_IDX = 0
END_IDX = 100

USE_MOVING_AVERAGE = False
MA_WINDOW = 3

OUTPUT_FILENAME = "gaf_four_mappings.jpg"


# =========================
#  HELPER
# =========================
def min_max_scaling(ts: np.ndarray) -> np.ndarray:
    min_val, max_val = ts.min(), ts.max()
    if max_val == min_val:
        return np.zeros_like(ts, dtype=float)
    return (ts - min_val) / (max_val - min_val)


# =========================
#  READ FILE
# =========================
ext = FILENAME.rsplit(".", 1)[-1].lower()

if ext in ("xlsx", "xls"):
    df = pd.read_excel(FILENAME)
elif ext == "csv":
    df = pd.read_csv(FILENAME)
else:
    sys.exit(f"Unsupported file type '.{ext}'. Use .xlsx, .xls, or .csv")

if FIELD_NAME not in df.columns:
    sys.exit(
        f'Column "{FIELD_NAME}" not found.\n'
        f"Available columns: {', '.join(df.columns)}"
    )

ts = pd.to_numeric(df[FIELD_NAME], errors="coerce").values
ts = ts[np.isfinite(ts)]

if ts.size == 0:
    sys.exit(f'No valid numeric data found in column "{FIELD_NAME}".')


# =========================
#  OPTIONAL: PARTIAL DATA
# =========================
if USE_PARTIAL_DATA:
    start = max(0, START_IDX)
    end = min(len(ts), END_IDX)
    if start >= end:
        sys.exit("Invalid START_IDX / END_IDX.")
    ts = ts[start:end]


# =========================
#  OPTIONAL: SMOOTHING
# =========================
if USE_MOVING_AVERAGE:
    kernel = np.ones(MA_WINDOW) / MA_WINDOW
    ts = np.convolve(ts, kernel, mode="same")


# =========================
#  STANDARD GAF PREPARATION
# =========================
x_scaled_01 = min_max_scaling(ts)
n = len(x_scaled_01)

if n < 3:
    sys.exit("Time series is too short for plotting.")

# Standard GAF uses values scaled to [-1, 1].
x_scaled = np.clip(2 * x_scaled_01 - 1, -1, 1)
radius = (np.arange(n) + 1) / n


def build_fields(phi_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gasf = np.cos(phi_values[:, None] + phi_values[None, :])
    gadf = np.sin(phi_values[:, None] - phi_values[None, :])
    return gasf, gadf


# (a) Standard GAF mapping
phi_cos = np.arccos(x_scaled)
gasf_cos, gadf_cos = build_fields(phi_cos)

# (b) Alternative angular mappings for comparison
phi_arctan = np.arctan(x_scaled_01)
gasf_arctan, gadf_arctan = build_fields(phi_arctan)

phi_arccosh = np.arccosh(1.0 + x_scaled_01)
gasf_arccosh, gadf_arccosh = build_fields(phi_arccosh)

phi_exp = np.pi * (np.exp(x_scaled_01) - 1.0) / (np.e - 1.0)
gasf_exp, gadf_exp = build_fields(phi_exp)

mapping_specs = [
    ("(a) Cosine", phi_cos, gasf_cos, gadf_cos),
    ("(b) Arctan", phi_arctan, gasf_arctan, gadf_arctan),
    ("(c) Arccosh Hyperbolic", phi_arccosh, gasf_arccosh, gadf_arccosh),
    ("(d) Exponential", phi_exp, gasf_exp, gadf_exp),
]


# =========================
#  PLOT – STANDARD GAF FIGURE
# =========================
fig = plt.figure(figsize=(10.8, 13.9))
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(
    5, 3,
    height_ratios=[0.12, 1.0, 1.0, 1.0, 1.0],
    width_ratios=[1.0, 1.08, 1.08],
    hspace=0.48,
    wspace=0.34,
)

# Shared column headers reduce clutter and keep the matrices unobstructed.
column_headers = ["Polar plot", "GASF", "GADF"]
for col_idx, header in enumerate(column_headers):
    ax_header = fig.add_subplot(gs[0, col_idx])
    ax_header.axis("off")
    ax_header.text(
        0.5, 0.35, header,
        ha="center", va="center",
        fontsize=10.5, fontweight="bold", color="black",
    )

for row_idx, (mapping_name, phi_values, gasf, gadf) in enumerate(mapping_specs):
    grid_row = row_idx + 1
    ax_polar = fig.add_subplot(gs[grid_row, 0], projection="polar")
    ax_gasf = fig.add_subplot(gs[grid_row, 1])
    ax_gadf = fig.add_subplot(gs[grid_row, 2])

    ax_polar.plot(phi_values, radius, color="black", linewidth=1.05, zorder=2)
    ax_polar.scatter(phi_values, radius, s=6, c="black", alpha=0.55, zorder=3)
    ax_polar.set_theta_zero_location("E")
    ax_polar.set_theta_direction(1)
    ax_polar.set_thetamin(0)
    ax_polar.set_thetamax(180)
    ax_polar.set_rlabel_position(90)
    ax_polar.set_rticks([0.25, 0.50, 0.75, 1.00])
    ax_polar.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax_polar.grid(True, linestyle=":", linewidth=0.45, color="#7a7a7a", alpha=0.45)
    ax_polar.spines["polar"].set_linewidth(1.0)
    ax_polar.spines["polar"].set_color("black")
    ax_polar.tick_params(axis="both", labelsize=7.0, pad=1, colors="black")
    ax_polar.set_title(mapping_name, fontsize=10.0, fontweight="bold", pad=10)

    im_gasf = ax_gasf.imshow(
        gasf, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", aspect="equal"
    )
    ax_gasf.set_xlabel("Time index", fontsize=8.5)
    ax_gasf.set_ylabel("Time index", fontsize=8.5)
    ax_gasf.tick_params(labelsize=7.5)
    cbar_gasf = fig.colorbar(im_gasf, ax=ax_gasf, fraction=0.046, pad=0.03)
    cbar_gasf.ax.tick_params(labelsize=7.0)

    im_gadf = ax_gadf.imshow(
        gadf, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower", aspect="equal"
    )
    ax_gadf.set_xlabel("Time index", fontsize=8.5)
    ax_gadf.set_ylabel("Time index", fontsize=8.5)
    ax_gadf.tick_params(labelsize=7.5)
    cbar_gadf = fig.colorbar(im_gadf, ax=ax_gadf, fraction=0.046, pad=0.03)
    cbar_gadf.ax.tick_params(labelsize=7.0)

fig.suptitle(
    "Gramian Angular Field Comparison Across Four Angular Mappings",
    fontsize=12,
    fontweight="bold",
    y=0.995,
)

plt.savefig(
    OUTPUT_FILENAME,
    dpi=300,
    bbox_inches="tight",
    format="jpeg",
    pil_kwargs={"quality": 95},
)
print(f"Saved -> {OUTPUT_FILENAME}")
plt.show()
