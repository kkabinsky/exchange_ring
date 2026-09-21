"""
Hybrid GAF-TadGAN-f-AnoGAN anomaly / crash forecaster
=====================================================

Pipeline
--------
1. Read price series + dates from an Excel file.
2. TadGAN (LSTM encoder-decoder + dual critics) -> per-window `anomaly_score`
   and `is_anomaly` on the observed data.  Results are cached on disk.
3. LSTM-based forecaster is trained on `anomaly_score` and auto-regressively
   rolled FORECAST_DAYS steps into the future (default 200 days).
4. Four Gramian Angular Field representations (cosine, exponential, arctan,
   arccosh) transform each WINDOW_SIZE slice of anomaly scores into a 2D
   image.  For each GAF method a separate f-AnoGAN (WGAN-GP + encoder) is
   trained on the NORMAL-only windows of the training split.
5. Two evaluation regimes are produced per method:
     - BACKTEST  : last FORECAST_DAYS observed windows (ground truth
                   available from TadGAN labels).
     - FUTURE    : FORECAST_DAYS windows built from the LSTM-forecasted
                   anomaly scores (no ground truth, predictions only).
6. Classification convention (as requested):
        0 = crash
        1 = not_crash
   i.e. pred=0 when the f-AnoGAN anomaly score exceeds the threshold
   (which is set from the 95th percentile of train-normal scores), else
   pred=1.  Ground-truth labels derived from TadGAN are inverted the same
   way (y = 1 - tadgan.is_anomaly) for metric computation.
7. All four methods are compared side-by-side on accuracy / precision /
   recall / F1 / AUC (pos_label=0 for crash) plus a soft-voting ensemble.
   Per-method GAF JPG folders are also exported.

Usage
-----
    python hybrid_gaf_tadgan_fanogan.py

Default behavior (no env vars needed):
  - Scans the `excel/` folder for every file ending with `_final2.xlsx`.
  - Reads column `cp` as the price series and `Date` as the timestamp.
  - For each file creates `hybrid_results/<stock>/` containing the TadGAN
    cache, LSTM forecast, per-method f-AnoGAN models, GAF JPGs, plots and
    a detailed Excel report.  A top-level `all_stocks_summary.csv` aggregates
    metrics across stocks.

Production defaults (override with env vars if desired):
    EXCEL_DIR=excel  BATCH_SUFFIX=_final2.xlsx  EXCEL_FIELD=cp  DATE_COLUMN=Date
    FORECAST_DAYS=200       BACKTEST_DAYS=200   MIN_BACKTEST_ANOMALIES=10
    TADGAN_EPOCHS=500       TADGAN_USE_CACHE=1
    N_EPOCHS_GAN=50         N_EPOCHS_ENC=50     FC_EPOCHS=100
Single-file mode: set INPUT_EXCEL=/path/to/file.xlsx .
Fast smoke test : QUICK=1   (forces tiny epoch counts).
"""

from __future__ import annotations

import glob
import json
import os
import random
import warnings
from dataclasses import dataclass, asdict, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# --- replication package (added): data in <repo>/input, results in <repo>/output/04_cross_market ---
import os as _rp_os, sys as _rp_sys
from pathlib import Path as _RpPath
REPO_ROOT = _RpPath(__file__).resolve().parents[2]
REPO_INPUT = REPO_ROOT / "input"
REPO_OUTPUT = REPO_ROOT / "output" / "04_cross_market"
REPO_OUTPUT.mkdir(parents=True, exist_ok=True)
_rp_sys.path.insert(0, str(REPO_ROOT / "code" / "common"))
# ------------------------------------------------------------------------------------------
if __name__ == "__main__":
    (REPO_OUTPUT / "hybrid").mkdir(parents=True, exist_ok=True)
    _rp_os.chdir(REPO_OUTPUT / "hybrid")  # TadGAN module creates models/ and results/ in the working directory
import improved_tadgans_anomaly2 as tadgan

# ============================================================
# CONFIG — all defaults are production values (no env vars needed).
# Override any of these via environment variables if desired.
# ============================================================

# --- Input location -----------------------------------------------------
# Batch mode (default): scan EXCEL_DIR for files whose names end in
# BATCH_SUFFIX and run the whole pipeline on each.
# Single-file mode: set INPUT_EXCEL to a specific path.
EXCEL_DIR = os.environ.get("EXCEL_DIR", str(REPO_INPUT / "benchmark_assets")).strip()
BATCH_SUFFIX = os.environ.get("BATCH_SUFFIX", "_final2.xlsx").strip()
INPUT_EXCEL = os.environ.get("INPUT_EXCEL", "").strip() or None

EXCEL_FIELD = os.environ.get("EXCEL_FIELD", "cp").strip()
DATE_COLUMN = os.environ.get("DATE_COLUMN", "Date").strip()
EXCEL_SHEET = os.environ.get("EXCEL_SHEET", "").strip() or None

# --- Output location ---------------------------------------------------
# OUT_ROOT is the top-level folder; each stock gets its own sub-folder.
OUT_ROOT = os.environ.get("OUT_ROOT", str(REPO_OUTPUT / "hybrid")).strip()

# --- Forecast / backtest horizon ---------------------------------------
FORECAST_DAYS = int(os.environ.get("FORECAST_DAYS", "200"))
BACKTEST_DAYS = int(os.environ.get("BACKTEST_DAYS", str(FORECAST_DAYS)))
MIN_BACKTEST_ANOMALIES = int(os.environ.get("MIN_BACKTEST_ANOMALIES", "10"))
MAX_BACKTEST_FRACTION = float(os.environ.get("MAX_BACKTEST_FRACTION", "0.5"))

# --- GAF / f-AnoGAN ----------------------------------------------------
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "32"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", str(WINDOW_SIZE)))
LATENT_DIM = int(os.environ.get("LATENT_DIM", "32"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))

N_CRITIC = int(os.environ.get("N_CRITIC", "5"))
LAMBDA_GP = float(os.environ.get("LAMBDA_GP", "10"))
LR_GAN = float(os.environ.get("LR_GAN", "0.0001"))
LR_ENC = float(os.environ.get("LR_ENC", "0.0001"))
N_EPOCHS_GAN = int(os.environ.get("N_EPOCHS_GAN", "50"))
N_EPOCHS_ENC = int(os.environ.get("N_EPOCHS_ENC", "50"))
KAPPA = float(os.environ.get("KAPPA", "1.0"))
THRESHOLD_Q = float(os.environ.get("THRESHOLD_Q", "0.95"))

# --- LSTM forecaster --------------------------------------------------
FC_INPUT_LEN = int(os.environ.get("FC_INPUT_LEN", "64"))
FC_HIDDEN = int(os.environ.get("FC_HIDDEN", "64"))
FC_LAYERS = int(os.environ.get("FC_LAYERS", "2"))
FC_EPOCHS = int(os.environ.get("FC_EPOCHS", "100"))
FC_BATCH = int(os.environ.get("FC_BATCH", "64"))
FC_LR = float(os.environ.get("FC_LR", "1e-3"))
FC_DROPOUT = float(os.environ.get("FC_DROPOUT", "0.1"))

# --- TadGAN ------------------------------------------------------------
TADGAN_SIGNAL_SHAPE = int(os.environ.get("TADGAN_SIGNAL_SHAPE", "100"))
TADGAN_LATENT_DIM = int(os.environ.get("TADGAN_LATENT_DIM", "100"))
TADGAN_HIDDEN_DIM = int(os.environ.get("TADGAN_HIDDEN_DIM", "40"))
TADGAN_BATCH_SIZE = int(os.environ.get("TADGAN_BATCH_SIZE", "64"))
TADGAN_LR = float(os.environ.get("TADGAN_LR", "1e-5"))
TADGAN_EPOCHS = int(os.environ.get("TADGAN_EPOCHS", "500"))
TADGAN_N_CRITIC = int(os.environ.get("TADGAN_N_CRITIC", "5"))
TADGAN_LAMBDA_GP = float(os.environ.get("TADGAN_LAMBDA_GP", "10"))
TADGAN_CHECKPOINT_INTERVAL = int(os.environ.get("TADGAN_CHECKPOINT_INTERVAL", "10"))
TADGAN_STRIDE = int(os.environ.get("TADGAN_STRIDE", "1"))
TADGAN_NORMALIZE = os.environ.get("TADGAN_NORMALIZE", "1").strip() != "0"
TADGAN_THRESHOLD_METHOD = os.environ.get("TADGAN_THRESHOLD_METHOD", "statistical").strip()
TADGAN_THRESHOLD_VALUE = float(os.environ.get("TADGAN_THRESHOLD_VALUE", "2.0"))
TADGAN_PRUNE_FALSE_POSITIVES = os.environ.get("TADGAN_PRUNE_FALSE_POSITIVES", "1").strip() != "0"
TADGAN_PRUNE_THRESHOLD = float(os.environ.get("TADGAN_PRUNE_THRESHOLD", "0.2"))
TADGAN_FORCE_TRAIN = os.environ.get("TADGAN_FORCE_TRAIN", "0").strip() != "0"
TADGAN_USE_CACHE = os.environ.get("TADGAN_USE_CACHE", "1").strip() != "0"

SAVE_GAFS = os.environ.get("SAVE_GAFS", "1").strip() != "0"
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "92"))

RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
QUICK = os.environ.get("QUICK", "0").strip() != "0"
if QUICK:
    N_EPOCHS_GAN = min(N_EPOCHS_GAN, 3)
    N_EPOCHS_ENC = min(N_EPOCHS_ENC, 3)
    FC_EPOCHS = min(FC_EPOCHS, 10)
    N_CRITIC = 1
    TADGAN_EPOCHS = min(TADGAN_EPOCHS, 3)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device={device}", flush=True)
print(
    f"[INFO] QUICK={int(QUICK)} | FORECAST_DAYS={FORECAST_DAYS} | WINDOW_SIZE={WINDOW_SIZE} "
    f"| N_EPOCHS_GAN={N_EPOCHS_GAN} | N_EPOCHS_ENC={N_EPOCHS_ENC} | FC_EPOCHS={FC_EPOCHS} "
    f"| TADGAN_EPOCHS={TADGAN_EPOCHS}",
    flush=True,
)

# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def reset_dir(path: str) -> None:
    """Recreate the directory empty (non-recursive file wipe)."""
    ensure_dir(path)
    for name in os.listdir(path):
        fp = os.path.join(path, name)
        try:
            if os.path.isfile(fp):
                os.remove(fp)
        except OSError:
            pass


def parse_date_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    text = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    iso_mask = text.str.fullmatch(r"\d{4}-\d{2}-\d{2}([ T].*)?")
    if iso_mask.any():
        parsed.loc[iso_mask] = pd.to_datetime(text[iso_mask], errors="coerce")

    for formatted_text, fmt in [
        (text.str.zfill(6), "%d%m%y"),
        (text.str.zfill(8), "%d%m%Y"),
        (text.str.zfill(8), "%Y%m%d"),
    ]:
        parsed = parsed.fillna(pd.to_datetime(formatted_text, format=fmt, errors="coerce"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = parsed.fillna(pd.to_datetime(text, errors="coerce", dayfirst=True))
    return parsed


def load_prices_and_dates(path: str) -> Tuple[np.ndarray, np.ndarray]:
    read_kwargs = {"sheet_name": EXCEL_SHEET} if EXCEL_SHEET else {}
    df = pd.read_excel(path, **read_kwargs)
    lower_map = {c: c.lower() for c in df.columns}
    inv = {v: k for k, v in lower_map.items()}

    price_col = inv.get(EXCEL_FIELD.lower())
    if price_col is None:
        for cand in ["signal", "close", "cp", "adjclose", "adj_close"]:
            if cand in inv:
                price_col = inv[cand]
                break
    if price_col is None:
        raise RuntimeError(f"Could not find price column '{EXCEL_FIELD}' in {path}. "
                           f"Available: {list(df.columns)}")

    date_col = inv.get(DATE_COLUMN.lower())
    if date_col is None:
        for cand in ["date", "d", "trade_date"]:
            if cand in inv:
                date_col = inv[cand]
                break

    df = df.copy()
    if date_col is not None:
        df[date_col] = parse_date_series(df[date_col])
        df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
        x_axis = df[date_col].to_numpy()
    else:
        x_axis = np.arange(len(df))
    prices = df[price_col].astype(float).to_numpy()
    return prices, x_axis


# ============================================================
# GAF
# ============================================================

def minmax_01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def resize_1d(x: np.ndarray, out_size: int) -> np.ndarray:
    if len(x) == out_size:
        return x.astype(np.float32)
    xp = np.linspace(0.0, 1.0, num=len(x))
    xq = np.linspace(0.0, 1.0, num=out_size)
    return np.interp(xq, xp, x).astype(np.float32)


def gaf_cosine(series: np.ndarray, out_size: int = IMG_SIZE) -> np.ndarray:
    s = resize_1d(minmax_01(series) * 2.0 - 1.0, out_size)
    phi = np.arccos(np.clip(s, -1.0, 1.0))
    gaf = np.cos(phi[:, None] + phi[None, :])
    return gaf.astype(np.float32)


def gaf_arctan(series: np.ndarray, out_size: int = IMG_SIZE) -> np.ndarray:
    s = resize_1d(minmax_01(series) * 2.0 - 1.0, out_size)
    phi = np.arctan(s)
    gaf = np.cos(phi[:, None] + phi[None, :])
    return gaf.astype(np.float32)


def gaf_arccosh(series: np.ndarray, out_size: int = IMG_SIZE) -> np.ndarray:
    s = resize_1d(minmax_01(series) + 1.0, out_size)
    phi = np.arccosh(np.clip(s, 1.0, None))
    gaf = np.cos(phi[:, None] + phi[None, :])
    return gaf.astype(np.float32)


def gaf_exponential(series: np.ndarray, out_size: int = IMG_SIZE) -> np.ndarray:
    s = resize_1d(minmax_01(series), out_size)
    phi = np.exp(s) - 1.0
    gaf = np.cos(phi[:, None] + phi[None, :])
    return gaf.astype(np.float32)


GAF_METHODS: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "cosine": gaf_cosine,
    "exponential": gaf_exponential,
    "arctan": gaf_arctan,
    "arccosh": gaf_arccosh,
}


def gaf_to_uint8(gaf: np.ndarray) -> np.ndarray:
    return np.clip((gaf + 1.0) * 127.5, 0, 255).astype(np.uint8)


def gaf_to_color_uint8(gaf: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    norm = np.clip((gaf + 1.0) / 2.0, 0.0, 1.0)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def save_gaf_jpg(gaf: np.ndarray, path: str) -> None:
    Image.fromarray(gaf_to_uint8(gaf)).save(path, format="JPEG", quality=JPEG_QUALITY)


def save_gaf_color_jpg(gaf: np.ndarray, path: str) -> None:
    Image.fromarray(gaf_to_color_uint8(gaf)).save(path, format="JPEG", quality=JPEG_QUALITY)


# ============================================================
# STAGE 1: TadGAN source
# ============================================================

@dataclass
class TadGANSource:
    score_series: np.ndarray
    label_series: np.ndarray
    x_axis: np.ndarray
    price_index_end: np.ndarray
    results_df: pd.DataFrame
    signal_shape: int


def build_tadgan_config(stage_dir: str) -> Dict[str, object]:
    model_dir = os.path.join(stage_dir, "models")
    output_dir = os.path.join(stage_dir, "results")
    ensure_dir(stage_dir)
    ensure_dir(model_dir)
    ensure_dir(output_dir)
    return {
        "signal_shape": TADGAN_SIGNAL_SHAPE,
        "latent_dim": TADGAN_LATENT_DIM,
        "hidden_dim": TADGAN_HIDDEN_DIM,
        "batch_size": TADGAN_BATCH_SIZE,
        "lr": TADGAN_LR,
        "n_epochs": TADGAN_EPOCHS,
        "n_critic": TADGAN_N_CRITIC,
        "lambda_gp": TADGAN_LAMBDA_GP,
        "checkpoint_interval": TADGAN_CHECKPOINT_INTERVAL,
        "stride": TADGAN_STRIDE,
        "normalize": TADGAN_NORMALIZE,
        "threshold_method": TADGAN_THRESHOLD_METHOD,
        "threshold_value": TADGAN_THRESHOLD_VALUE,
        "prune_false_positives": TADGAN_PRUNE_FALSE_POSITIVES,
        "prune_threshold": TADGAN_PRUNE_THRESHOLD,
        "generate_plots": False,
        "show_plots": False,
        "export_excel": False,
        "model_dir": model_dir,
        "output_dir": output_dir,
        "device": device,
    }


def run_tadgan_source(prices: np.ndarray, x_axis: np.ndarray, out_root: str) -> TadGANSource:
    stage_dir = os.path.join(out_root, "stage1_tadgan")
    ensure_dir(stage_dir)
    input_csv = os.path.join(stage_dir, "signal_input.csv")
    output_csv = os.path.join(stage_dir, "tadgan_results.csv")
    pd.DataFrame({"signal": prices}).to_csv(input_csv, index=False)

    print(f"  [TADGAN] preparing scores ({len(prices)} raw prices)", flush=True)
    config = build_tadgan_config(stage_dir)
    expected_len = max(0, len(prices) - int(config["signal_shape"]) + 1)

    results_df: Optional[pd.DataFrame] = None
    if TADGAN_USE_CACHE and not TADGAN_FORCE_TRAIN and os.path.isfile(output_csv):
        try:
            cached = pd.read_csv(output_csv)
            if (
                "anomaly_score" in cached.columns
                and "is_anomaly" in cached.columns
                and len(cached) == expected_len
            ):
                print(f"  [TADGAN] reusing cache -> {output_csv}", flush=True)
                results_df = cached.reset_index(drop=True)
        except Exception as exc:
            print(f"  [TADGAN] cache load failed ({exc}); recomputing.", flush=True)

    if results_df is None:
        results_df = tadgan.process_file(
            input_csv, output_csv, config, force_train=TADGAN_FORCE_TRAIN
        ).reset_index(drop=True)

    score_series = results_df["anomaly_score"].to_numpy(dtype=np.float64)
    label_series = results_df["is_anomaly"].to_numpy(dtype=np.int64)
    if len(score_series) < WINDOW_SIZE + FORECAST_DAYS:
        raise RuntimeError(
            f"Too few TadGAN scores ({len(score_series)}). Need >= "
            f"WINDOW_SIZE+FORECAST_DAYS = {WINDOW_SIZE + FORECAST_DAYS}."
        )

    score_x_axis = np.asarray(x_axis)[: len(score_series)]
    price_index_end = np.minimum(
        np.arange(len(score_series), dtype=np.int64) + int(config["signal_shape"]) - 1,
        len(prices) - 1,
    )
    results_df["score_index"] = np.arange(len(results_df), dtype=np.int64)
    results_df["price_index_end"] = price_index_end
    if np.issubdtype(np.asarray(score_x_axis).dtype, np.datetime64):
        results_df["score_date"] = pd.to_datetime(score_x_axis)
        results_df["price_date_end"] = pd.to_datetime(np.asarray(x_axis)[price_index_end])

    print(
        f"  [TADGAN] scores={len(score_series)} | anomaly_windows={(label_series == 1).sum()} "
        f"| normal_windows={(label_series == 0).sum()}",
        flush=True,
    )
    return TadGANSource(
        score_series=score_series,
        label_series=label_series,
        x_axis=score_x_axis,
        price_index_end=price_index_end,
        results_df=results_df,
        signal_shape=int(config["signal_shape"]),
    )


# ============================================================
# STAGE 2A: LSTM forecaster on anomaly scores
# ============================================================

class LSTMForecaster(nn.Module):
    def __init__(self, input_len: int = FC_INPUT_LEN, hidden: int = FC_HIDDEN,
                 layers: int = FC_LAYERS, dropout: float = FC_DROPOUT):
        super().__init__()
        self.input_len = input_len
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)  # (B,)


@dataclass
class ForecastResult:
    future_scores: np.ndarray          # length FORECAST_DAYS
    train_loss: List[float]
    val_loss: List[float]
    score_mean: float
    score_std: float


def train_forecaster(scores_train: np.ndarray, out_dir: str) -> ForecastResult:
    ensure_dir(out_dir)
    mu, sd = float(np.mean(scores_train)), float(np.std(scores_train) + 1e-8)
    z = (scores_train - mu) / sd

    xs, ys = [], []
    for i in range(len(z) - FC_INPUT_LEN):
        xs.append(z[i : i + FC_INPUT_LEN])
        ys.append(z[i + FC_INPUT_LEN])
    if len(xs) < 32:
        raise RuntimeError(f"Forecaster has too few samples ({len(xs)}).")
    xs = np.array(xs, dtype=np.float32)[:, :, None]
    ys = np.array(ys, dtype=np.float32)

    split = max(1, int(len(xs) * 0.9))
    tr_ds = TensorDataset(torch.from_numpy(xs[:split]), torch.from_numpy(ys[:split]))
    va_ds = TensorDataset(torch.from_numpy(xs[split:]), torch.from_numpy(ys[split:]))
    tr_dl = DataLoader(tr_ds, batch_size=FC_BATCH, shuffle=True, drop_last=False)
    va_dl = DataLoader(va_ds, batch_size=FC_BATCH, shuffle=False, drop_last=False)

    model = LSTMForecaster().to(device)
    opt = optim.Adam(model.parameters(), lr=FC_LR)
    train_losses, val_losses = [], []
    best_val = float("inf")
    best_state = None
    for ep in range(1, FC_EPOCHS + 1):
        model.train()
        tr_sum, nb = 0.0, 0
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = F.mse_loss(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_sum += float(loss.item()); nb += 1
        train_losses.append(tr_sum / max(nb, 1))
        model.eval()
        va_sum, nv = 0.0, 0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(device), yb.to(device)
                va_sum += float(F.mse_loss(model(xb), yb).item()); nv += 1
        val_losses.append(va_sum / max(nv, 1))
        if val_losses[-1] < best_val:
            best_val = val_losses[-1]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"  [FC] epoch {ep:03d}/{FC_EPOCHS} train={train_losses[-1]:.5f} "
            f"val={val_losses[-1]:.5f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    last_window = z[-FC_INPUT_LEN:].astype(np.float32).copy()
    future = np.zeros(FORECAST_DAYS, dtype=np.float32)
    with torch.no_grad():
        for t in range(FORECAST_DAYS):
            x = torch.from_numpy(last_window[None, :, None]).to(device)
            yhat = float(model(x).item())
            future[t] = yhat
            last_window = np.concatenate([last_window[1:], np.array([yhat], dtype=np.float32)])
    future_scores = future * sd + mu

    # Loss plot
    fig = plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch"); plt.ylabel("MSE (z-score)")
    plt.title("LSTM forecaster loss")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "forecaster_loss.png"), dpi=140)
    plt.close(fig)

    # Forecast plot
    fig = plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(scores_train)), scores_train, lw=0.8, label="observed anomaly_score")
    plt.plot(np.arange(len(scores_train), len(scores_train) + FORECAST_DAYS),
             future_scores, color="red", lw=1.2, label=f"forecast (+{FORECAST_DAYS})")
    plt.xlabel("window index"); plt.ylabel("anomaly_score")
    plt.title("Observed + LSTM forecast of TadGAN anomaly score")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "forecaster_projection.png"), dpi=140)
    plt.close(fig)

    return ForecastResult(
        future_scores=future_scores.astype(np.float64),
        train_loss=train_losses,
        val_loss=val_losses,
        score_mean=mu,
        score_std=sd,
    )


# ============================================================
# STAGE 3: f-AnoGAN  (WGAN-GP + encoder) on GAF images
# ============================================================

_H = max(IMG_SIZE // 16, 2)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 512 * _H * _H)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, z):
        x = self.net(self.fc(z).view(-1, 512, _H, _H))
        if x.shape[-2:] != (IMG_SIZE, IMG_SIZE):
            x = F.interpolate(x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.LeakyReLU(0.2, True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(512 * _H * _H, 1),
        )

    def forward(self, x):
        return self.head(self.feat(x))

    def features(self, x):
        return self.feat(x)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.Linear(512 * _H * _H, LATENT_DIM),
        )

    def forward(self, x):
        return self.net(x)


def gradient_penalty(D, real, fake):
    b = real.size(0)
    alpha = torch.rand(b, 1, 1, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    out = D(interp)
    grad = torch.autograd.grad(
        out, interp, grad_outputs=torch.ones_like(out),
        create_graph=True, retain_graph=True,
    )[0]
    return ((grad.norm(2, dim=[1, 2, 3]) - 1) ** 2).mean()


def train_wgangp(G, D, loader):
    opt_G = optim.Adam(G.parameters(), lr=LR_GAN, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=LR_GAN, betas=(0.0, 0.9))
    G.train(); D.train()
    g_hist, d_hist = [], []
    for epoch in range(1, N_EPOCHS_GAN + 1):
        gsum = dsum = 0.0
        nb = 0
        for (real,) in loader:
            real = real.to(device)
            b = real.size(0)
            for _ in range(N_CRITIC):
                z = torch.randn(b, LATENT_DIM, device=device)
                fake = G(z).detach()
                gp = gradient_penalty(D, real, fake)
                d_loss = -D(real).mean() + D(fake).mean() + LAMBDA_GP * gp
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()
            z = torch.randn(b, LATENT_DIM, device=device)
            g_loss = -D(G(z)).mean()
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            gsum += float(g_loss.item()); dsum += float(d_loss.item()); nb += 1
        g_hist.append(gsum / max(nb, 1))
        d_hist.append(dsum / max(nb, 1))
        print(f"    [GAN] epoch {epoch:03d}/{N_EPOCHS_GAN} D={d_hist[-1]:.4f} G={g_hist[-1]:.4f}", flush=True)
    return g_hist, d_hist


def train_encoder(E, G, D, loader):
    opt_E = optim.Adam(E.parameters(), lr=LR_ENC, betas=(0.5, 0.999))
    E.train(); G.eval(); D.eval()
    e_hist = []
    for epoch in range(1, N_EPOCHS_ENC + 1):
        esum = 0.0
        nb = 0
        for (real,) in loader:
            real = real.to(device)
            z_hat = E(real)
            recon = G(z_hat)
            loss_rec = F.mse_loss(recon, real)
            with torch.no_grad():
                f_real = D.features(real)
            f_recon = D.features(recon)
            loss_feat = F.mse_loss(f_recon, f_real.detach())
            loss = loss_rec + KAPPA * loss_feat
            opt_E.zero_grad(); loss.backward(); opt_E.step()
            esum += float(loss.item()); nb += 1
        e_hist.append(esum / max(nb, 1))
        print(f"    [ENC] epoch {epoch:03d}/{N_EPOCHS_ENC} loss={e_hist[-1]:.6f}", flush=True)
    return e_hist


def anomaly_scores(E, G, D, x: np.ndarray, batch: int = 64) -> np.ndarray:
    E.eval(); G.eval(); D.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = torch.tensor(x[i : i + batch], dtype=torch.float32).unsqueeze(1).to(device)
            z = E(xb)
            recon = G(z)
            rec = ((xb - recon) ** 2).mean(dim=[1, 2, 3])
            f_real = D.features(xb)
            f_recon = D.features(recon)
            feat = ((f_real - f_recon) ** 2).mean(dim=[1, 2, 3])
            out.extend((rec + KAPPA * feat).detach().cpu().numpy().tolist())
    return np.array(out, dtype=np.float64)


# ============================================================
# Window building + splits
# ============================================================

def label_window(score_labels: np.ndarray) -> int:
    """Majority vote; 1 if more than half of the windowed TadGAN is_anomaly=1."""
    return int(np.sum(score_labels) > (len(score_labels) / 2.0))


def choose_backtest_length(window_labels: np.ndarray) -> int:
    """Pick a backtest length so that the held-out tail contains at least
    MIN_BACKTEST_ANOMALIES positive windows (in TadGAN convention).
    Falls back to BACKTEST_DAYS if already satisfied, bounded by
    MAX_BACKTEST_FRACTION of the total."""
    n = len(window_labels)
    cap = max(int(n * MAX_BACKTEST_FRACTION), BACKTEST_DAYS)
    length = min(BACKTEST_DAYS, n - 1)
    def n_pos_in_tail(k):
        return int(np.sum(window_labels[-k:]))
    if n_pos_in_tail(length) >= MIN_BACKTEST_ANOMALIES:
        return length
    for k in range(length, min(cap, n - 1) + 1):
        if n_pos_in_tail(k) >= MIN_BACKTEST_ANOMALIES:
            return k
    return min(cap, n - 1)


def build_windows(
    series: np.ndarray,
    score_labels: np.ndarray,
    gaf_fn: Callable[[np.ndarray, int], np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gafs, labels, starts = [], [], []
    n = len(series)
    for start in range(0, n - WINDOW_SIZE + 1):
        gafs.append(gaf_fn(series[start : start + WINDOW_SIZE], IMG_SIZE))
        labels.append(label_window(score_labels[start : start + WINDOW_SIZE]))
        starts.append(start)
    return (
        np.array(gafs, dtype=np.float32),
        np.array(labels, dtype=np.int64),
        np.array(starts, dtype=np.int64),
    )


def build_future_windows(
    history_tail: np.ndarray,
    forecast_scores: np.ndarray,
    gaf_fn: Callable[[np.ndarray, int], np.ndarray],
) -> np.ndarray:
    """Build FORECAST_DAYS windows that slide over the boundary between
    the last WINDOW_SIZE-1 observed scores and the FORECAST_DAYS
    LSTM-forecasted scores, so every window of size WINDOW_SIZE ends on a
    unique forecast timestep."""
    padded = np.concatenate([history_tail[-(WINDOW_SIZE - 1):], forecast_scores])
    gafs = []
    for t in range(FORECAST_DAYS):
        segment = padded[t : t + WINDOW_SIZE]
        gafs.append(gaf_fn(segment, IMG_SIZE))
    return np.array(gafs, dtype=np.float32)


# ============================================================
# Evaluation
# ============================================================

CRASH_LABEL = 0      # user convention: 0 = crash
NORMAL_LABEL = 1     # user convention: 1 = not crash


def tadgan_label_to_user(t: np.ndarray) -> np.ndarray:
    """TadGAN is_anomaly (1=crash) -> user convention (0=crash, 1=not_crash)."""
    return (1 - np.asarray(t, dtype=np.int64)).astype(np.int64)


def score_to_user_pred(score: np.ndarray, threshold: float) -> np.ndarray:
    """High f-AnoGAN score -> crash (=0). Below threshold -> not_crash (=1)."""
    return np.where(np.asarray(score) > threshold, CRASH_LABEL, NORMAL_LABEL).astype(np.int64)


def robust_auc(y_true_user: np.ndarray, score: np.ndarray) -> float:
    """AUC for the crash class (label=0)."""
    if len(np.unique(y_true_user)) < 2:
        return float("nan")
    crash_prob = score  # higher = more crash-like
    crash_target = (np.asarray(y_true_user) == CRASH_LABEL).astype(np.int64)
    try:
        return float(roc_auc_score(crash_target, crash_prob))
    except Exception:
        return float("nan")


def eval_metrics(y_true_user: np.ndarray, y_pred_user: np.ndarray,
                 score: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = (0, 0, 0, 0)
    # Confusion matrix in user convention: positive = crash (0)
    cm = confusion_matrix(y_true_user, y_pred_user, labels=[CRASH_LABEL, NORMAL_LABEL])
    tp, fn = int(cm[0, 0]), int(cm[0, 1])
    fp, tn = int(cm[1, 0]), int(cm[1, 1])
    acc = accuracy_score(y_true_user, y_pred_user)
    prec_crash = precision_score(y_true_user, y_pred_user, pos_label=CRASH_LABEL, zero_division=0)
    rec_crash = recall_score(y_true_user, y_pred_user, pos_label=CRASH_LABEL, zero_division=0)
    f1_crash = f1_score(y_true_user, y_pred_user, pos_label=CRASH_LABEL, zero_division=0)
    auc = robust_auc(y_true_user, score)
    return {
        "accuracy": float(acc),
        "precision_crash": float(prec_crash),
        "recall_crash": float(rec_crash),
        "f1_crash": float(f1_crash),
        "auc_crash": auc,
        "tp_crash": tp,
        "fp_crash": fp,
        "tn_normal": tn,
        "fn_crash": fn,
        "n_samples": int(len(y_true_user)),
        "n_crash": int((np.asarray(y_true_user) == CRASH_LABEL).sum()),
        "n_normal": int((np.asarray(y_true_user) == NORMAL_LABEL).sum()),
    }


# ============================================================
# GAF JPG export
# ============================================================

def export_gaf_set(
    out_root: str,
    method: str,
    split_name: str,
    gafs: np.ndarray,
    labels_user: Optional[np.ndarray],
    starts: Optional[np.ndarray],
    limit: Optional[int] = None,
) -> None:
    """Save JPGs into  out_root/anomaly_score_gaf/<method>/<split>/{crash|not_crash|unlabeled}/
    and the color version into out_root/anomaly_score_gaf_color/<method>/<split>/..."""
    base_gray = os.path.join(out_root, "anomaly_score_gaf", method, split_name)
    base_col = os.path.join(out_root, "anomaly_score_gaf_color", method, split_name)
    subfolders = ["crash", "not_crash", "unlabeled"]
    for base in (base_gray, base_col):
        for sub in subfolders:
            reset_dir(os.path.join(base, sub))

    n = len(gafs)
    take = n if limit is None else min(n, limit)
    for i in range(take):
        if labels_user is None:
            sub = "unlabeled"
        else:
            sub = "crash" if int(labels_user[i]) == CRASH_LABEL else "not_crash"
        start_idx = int(starts[i]) if starts is not None else i
        stem = f"{method}_{split_name}_{i:05d}_start{start_idx:05d}_{sub}"
        save_gaf_jpg(gafs[i], os.path.join(base_gray, sub, stem + ".jpg"))
        save_gaf_color_jpg(gafs[i], os.path.join(base_col, sub, stem + ".jpg"))


# ============================================================
# Main pipeline
# ============================================================

@dataclass
class MethodResult:
    method: str
    threshold: float
    train_normal_score_q: float
    backtest: Dict[str, float]
    future_score_mean: float
    future_score_std: float
    future_pred_crash_rate: float
    backtest_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))
    backtest_preds: np.ndarray = field(default_factory=lambda: np.zeros(0))
    backtest_true: np.ndarray = field(default_factory=lambda: np.zeros(0))
    future_scores_fanogan: np.ndarray = field(default_factory=lambda: np.zeros(0))
    future_preds: np.ndarray = field(default_factory=lambda: np.zeros(0))


def run_one_method(
    method_name: str,
    gaf_fn: Callable[[np.ndarray, int], np.ndarray],
    tad: TadGANSource,
    forecast: ForecastResult,
    out_root: str,
) -> MethodResult:
    print(f"  [METHOD] {method_name}", flush=True)
    method_dir = os.path.join(out_root, "stage3_fanogan", method_name)
    ensure_dir(method_dir)

    scores = tad.score_series
    labels = tad.label_series
    n = len(scores)

    # Build GAF windows over all observed scores
    gafs_all, labels_all, starts_all = build_windows(scores, labels, gaf_fn)
    total = len(gafs_all)
    backtest_n = choose_backtest_length(labels_all)
    train_mask = np.zeros(total, dtype=bool)
    back_mask = np.zeros(total, dtype=bool)
    train_mask[: total - backtest_n] = True
    back_mask[total - backtest_n :] = True
    print(f"    [SPLIT] train={int(train_mask.sum())} | backtest={backtest_n} "
          f"| backtest_anomalies={int(labels_all[back_mask].sum())}", flush=True)

    train_gafs = gafs_all[train_mask]
    train_labels_maj = labels_all[train_mask]
    normal_train = train_gafs[train_labels_maj == 0]  # TadGAN normal
    if len(normal_train) < max(BATCH_SIZE, 8):
        raise RuntimeError(
            f"Method {method_name}: only {len(normal_train)} normal training GAFs, not enough."
        )
    print(f"    [DATA] train_total={len(train_gafs)} | normal_used={len(normal_train)} "
          f"| backtest_windows={int(back_mask.sum())}", flush=True)

    # Train f-AnoGAN
    G = Generator().to(device)
    D = Discriminator().to(device)
    E = Encoder().to(device)

    tensor = torch.from_numpy(normal_train[:, None, :, :]).float()
    loader = DataLoader(TensorDataset(tensor), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    g_hist, d_hist = train_wgangp(G, D, loader)
    e_hist = train_encoder(E, G, D, loader)

    # Determine threshold from train normal score distribution
    train_normal_scores = anomaly_scores(E, G, D, normal_train)
    threshold = float(np.quantile(train_normal_scores, THRESHOLD_Q))

    # Backtest
    back_gafs = gafs_all[back_mask]
    back_tadgan_labels = labels_all[back_mask]
    back_user_true = tadgan_label_to_user(back_tadgan_labels)
    back_scores = anomaly_scores(E, G, D, back_gafs)
    back_preds = score_to_user_pred(back_scores, threshold)
    backtest_metrics = eval_metrics(back_user_true, back_preds, back_scores)

    # Future (LSTM forecast-based) windows
    future_gafs = build_future_windows(
        scores[-(WINDOW_SIZE - 1):], forecast.future_scores, gaf_fn
    )
    future_scores_fanogan = anomaly_scores(E, G, D, future_gafs)
    future_preds = score_to_user_pred(future_scores_fanogan, threshold)

    # Export GAF JPGs (train + backtest + future) for every anomaly-score window
    if SAVE_GAFS:
        train_starts = starts_all[train_mask]
        train_user_true = tadgan_label_to_user(labels_all[train_mask])
        back_starts = starts_all[back_mask]
        export_gaf_set(out_root, method_name, "train", train_gafs, train_user_true, train_starts)
        export_gaf_set(out_root, method_name, "backtest", back_gafs, back_user_true, back_starts)
        export_gaf_set(out_root, method_name, "future", future_gafs, None, None)

    # Loss plots
    fig = plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(g_hist, label="G"); plt.plot(d_hist, label="D")
    plt.title(f"{method_name} GAN loss"); plt.legend(); plt.grid(alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot(e_hist)
    plt.title(f"{method_name} Encoder loss"); plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(method_dir, "training_loss.png"), dpi=140)
    plt.close(fig)

    # Backtest time-series plot
    fig = plt.figure(figsize=(12, 4))
    plt.plot(back_scores, label="f-AnoGAN score", color="steelblue", lw=1.0)
    plt.axhline(threshold, color="red", ls="--", label=f"thr q={THRESHOLD_Q}")
    bad = np.where(back_user_true == CRASH_LABEL)[0]
    if len(bad) > 0:
        plt.scatter(bad, back_scores[bad], color="red", s=12, label="true crash (0)")
    pc = np.where(back_preds == CRASH_LABEL)[0]
    if len(pc) > 0:
        plt.scatter(pc, back_scores[pc], color="orange", s=8, marker="x",
                    label="pred crash (0)")
    plt.xlabel("backtest window"); plt.ylabel("score")
    plt.title(f"{method_name} backtest  acc={backtest_metrics['accuracy']:.3f} "
              f"F1(crash)={backtest_metrics['f1_crash']:.3f}")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(os.path.join(method_dir, "backtest_scores.png"), dpi=140)
    plt.close(fig)

    # Future plot
    fig = plt.figure(figsize=(12, 4))
    plt.plot(future_scores_fanogan, color="darkred", lw=1.0, label="f-AnoGAN score")
    plt.axhline(threshold, color="red", ls="--", label=f"thr q={THRESHOLD_Q}")
    crash_idx = np.where(future_preds == CRASH_LABEL)[0]
    if len(crash_idx) > 0:
        plt.scatter(crash_idx, future_scores_fanogan[crash_idx], color="red", s=12,
                    label="pred crash (0)")
    plt.xlabel(f"future window (t+1 .. t+{FORECAST_DAYS})"); plt.ylabel("score")
    plt.title(f"{method_name} future {FORECAST_DAYS}-day forecast   "
              f"pred_crash_rate={ (future_preds==CRASH_LABEL).mean():.3f}")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    fig.savefig(os.path.join(method_dir, "future_scores.png"), dpi=140)
    plt.close(fig)

    # Persist per-method detail CSVs
    pd.DataFrame({
        "backtest_idx": np.arange(len(back_scores)),
        "window_start": starts_all[back_mask],
        "fanogan_score": back_scores,
        "threshold": threshold,
        "true_tadgan_is_anomaly": back_tadgan_labels,
        "true_label_user_0crash_1not": back_user_true,
        "pred_label_user_0crash_1not": back_preds,
    }).to_csv(os.path.join(method_dir, "backtest_detail.csv"), index=False)

    pd.DataFrame({
        "forecast_step": np.arange(FORECAST_DAYS),
        "lstm_forecast_anomaly_score": forecast.future_scores,
        "fanogan_score": future_scores_fanogan,
        "threshold": threshold,
        "pred_label_user_0crash_1not": future_preds,
    }).to_csv(os.path.join(method_dir, "future_detail.csv"), index=False)

    result = MethodResult(
        method=method_name,
        threshold=threshold,
        train_normal_score_q=float(np.quantile(train_normal_scores, 0.5)),
        backtest=backtest_metrics,
        future_score_mean=float(np.mean(future_scores_fanogan)),
        future_score_std=float(np.std(future_scores_fanogan)),
        future_pred_crash_rate=float((future_preds == CRASH_LABEL).mean()),
        backtest_scores=back_scores,
        backtest_preds=back_preds,
        backtest_true=back_user_true,
        future_scores_fanogan=future_scores_fanogan,
        future_preds=future_preds,
    )
    return result


def ensemble_results(results: List[MethodResult]) -> Dict[str, object]:
    """Soft voting: z-score each method's scores, average, then threshold."""
    if not results:
        return {}
    # Align lengths
    back_len = min(len(r.backtest_scores) for r in results)
    fut_len = min(len(r.future_scores_fanogan) for r in results)

    def zscore(x: np.ndarray) -> np.ndarray:
        mu, sd = float(np.mean(x)), float(np.std(x) + 1e-8)
        return (x - mu) / sd

    back_stack = np.stack([zscore(r.backtest_scores[:back_len]) for r in results], axis=0)
    fut_stack = np.stack([zscore(r.future_scores_fanogan[:fut_len]) for r in results], axis=0)
    back_mean = back_stack.mean(axis=0)
    fut_mean = fut_stack.mean(axis=0)

    # Threshold from backtest: pick 95th percentile of the NORMAL part
    y_true = results[0].backtest_true[:back_len]
    normal_mask = y_true == NORMAL_LABEL
    if normal_mask.any():
        thr = float(np.quantile(back_mean[normal_mask], THRESHOLD_Q))
    else:
        thr = float(np.quantile(back_mean, THRESHOLD_Q))
    back_preds = score_to_user_pred(back_mean, thr)
    fut_preds = score_to_user_pred(fut_mean, thr)
    metrics = eval_metrics(y_true, back_preds, back_mean)
    return {
        "method": "ensemble",
        "threshold": thr,
        "backtest": metrics,
        "future_score_mean": float(np.mean(fut_mean)),
        "future_pred_crash_rate": float((fut_preds == CRASH_LABEL).mean()),
        "backtest_mean_zscore": back_mean,
        "future_mean_zscore": fut_mean,
        "future_preds": fut_preds,
    }


def write_excel_report(
    out_root: str,
    tad: TadGANSource,
    forecast: ForecastResult,
    results: List[MethodResult],
    ensemble: Dict[str, object],
    x_axis: np.ndarray,
) -> str:
    path = os.path.join(out_root, "hybrid_four_gaf_report.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        # summary
        rows = []
        for r in results:
            row = {"method": r.method, "threshold": r.threshold,
                   **{f"backtest_{k}": v for k, v in r.backtest.items()},
                   "future_pred_crash_rate": r.future_pred_crash_rate,
                   "future_score_mean": r.future_score_mean,
                   "future_score_std": r.future_score_std}
            rows.append(row)
        if ensemble:
            row = {"method": "ensemble", "threshold": ensemble["threshold"],
                   **{f"backtest_{k}": v for k, v in ensemble["backtest"].items()},
                   "future_pred_crash_rate": ensemble["future_pred_crash_rate"],
                   "future_score_mean": ensemble["future_score_mean"],
                   "future_score_std": float("nan")}
            rows.append(row)
        pd.DataFrame(rows).to_excel(writer, sheet_name="summary", index=False)

        # tadgan source
        tad.results_df.to_excel(writer, sheet_name="tadgan_source", index=False)

        # lstm forecast
        n_obs = len(tad.score_series)
        future_x = np.arange(n_obs, n_obs + FORECAST_DAYS)
        fc_df = pd.DataFrame({
            "forecast_step": np.arange(FORECAST_DAYS),
            "score_index_absolute": future_x,
            "lstm_forecast_anomaly_score": forecast.future_scores,
        })
        fc_df.to_excel(writer, sheet_name="lstm_forecast", index=False)

        # per-method backtest
        for r in results:
            df = pd.DataFrame({
                "backtest_idx": np.arange(len(r.backtest_scores)),
                "fanogan_score": r.backtest_scores,
                "threshold": r.threshold,
                "true_label_user_0crash_1not": r.backtest_true,
                "pred_label_user_0crash_1not": r.backtest_preds,
            })
            df.to_excel(writer, sheet_name=f"{r.method[:18]}_back", index=False)

        for r in results:
            df = pd.DataFrame({
                "forecast_step": np.arange(len(r.future_scores_fanogan)),
                "lstm_forecast_anomaly_score": forecast.future_scores[: len(r.future_scores_fanogan)],
                "fanogan_score": r.future_scores_fanogan,
                "threshold": r.threshold,
                "pred_label_user_0crash_1not": r.future_preds,
            })
            df.to_excel(writer, sheet_name=f"{r.method[:18]}_fut", index=False)

        if ensemble:
            df = pd.DataFrame({
                "backtest_idx": np.arange(len(ensemble["backtest_mean_zscore"])),
                "mean_zscore": ensemble["backtest_mean_zscore"],
                "threshold": ensemble["threshold"],
                "pred_label_user_0crash_1not": score_to_user_pred(
                    ensemble["backtest_mean_zscore"], ensemble["threshold"]),
            })
            df.to_excel(writer, sheet_name="ensemble_back", index=False)
            df2 = pd.DataFrame({
                "forecast_step": np.arange(len(ensemble["future_mean_zscore"])),
                "mean_zscore": ensemble["future_mean_zscore"],
                "threshold": ensemble["threshold"],
                "pred_label_user_0crash_1not": ensemble["future_preds"],
            })
            df2.to_excel(writer, sheet_name="ensemble_fut", index=False)
    return path


def save_comparison_plots(out_root: str, results: List[MethodResult],
                          ensemble: Dict[str, object]) -> None:
    # Bar chart of metrics
    metrics = ["accuracy", "precision_crash", "recall_crash", "f1_crash", "auc_crash"]
    names = [r.method for r in results]
    if ensemble:
        names = names + ["ensemble"]
    data = {m: [] for m in metrics}
    for r in results:
        for m in metrics:
            data[m].append(r.backtest.get(m, float("nan")))
    if ensemble:
        for m in metrics:
            data[m].append(ensemble["backtest"].get(m, float("nan")))

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    w = 0.15
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 2) * w, data[m], width=w, label=m)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Four-GAF backtest metrics (crash=positive, label=0)")
    ax.grid(axis="y", alpha=0.3); ax.legend(ncol=3, fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_root, "four_method_backtest_metrics.png"), dpi=150)
    plt.close(fig)

    # Future pred-crash rate
    rates = [r.future_pred_crash_rate for r in results]
    if ensemble:
        rates.append(ensemble["future_pred_crash_rate"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, rates, color="indianred")
    ax.set_ylabel("forecast-window crash rate"); ax.set_ylim(0, 1)
    ax.set_title(f"Future {FORECAST_DAYS}-day predicted crash rate per method")
    for i, v in enumerate(rates):
        ax.text(i, v + 0.01, f"{v:.2%}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_root, "future_crash_rate.png"), dpi=150)
    plt.close(fig)


def run_one_stock(excel_path: str, stock_out: str) -> Optional[Dict[str, object]]:
    """Run the full hybrid pipeline for a single Excel file.
    Returns a compact per-method summary dict, or None on failure."""
    ensure_dir(stock_out)
    stock_name = os.path.basename(os.path.normpath(stock_out))
    print(f"\n==================== STOCK: {stock_name} ====================", flush=True)
    print(f"[STOCK] excel={excel_path} -> {stock_out}", flush=True)

    prices, x_axis = load_prices_and_dates(excel_path)
    print(f"[DATA] prices={len(prices)} | first={x_axis[0]} | last={x_axis[-1]}", flush=True)

    tad = run_tadgan_source(prices, x_axis, stock_out)

    fc_dir = os.path.join(stock_out, "stage2_forecaster")
    ensure_dir(fc_dir)
    forecast = train_forecaster(tad.score_series, fc_dir)

    results: List[MethodResult] = []
    for name, fn in GAF_METHODS.items():
        try:
            res = run_one_method(name, fn, tad, forecast, stock_out)
            results.append(res)
        except Exception as exc:
            print(f"  [METHOD] {name} failed: {exc}", flush=True)

    if not results:
        print(f"[STOCK] {stock_name}: all methods failed, skipping.", flush=True)
        return None

    ensemble = ensemble_results(results)
    excel_report = write_excel_report(stock_out, tad, forecast, results, ensemble, x_axis)
    save_comparison_plots(stock_out, results, ensemble)

    # Human-readable summary
    print(f"\n---- SUMMARY [{stock_name}] (backtest, label 0=crash, 1=not_crash) ----", flush=True)
    header = ("method", "acc", "prec(0)", "rec(0)", "f1(0)", "auc(0)",
              "TP", "FP", "TN", "FN", "fut_crash%")
    print(" | ".join(f"{h:>11}" for h in header), flush=True)

    def fmt_row(name: str, m: Dict[str, float], fut: float) -> str:
        return " | ".join([f"{name:>11}",
                            f"{m['accuracy']:>11.3f}", f"{m['precision_crash']:>11.3f}",
                            f"{m['recall_crash']:>11.3f}", f"{m['f1_crash']:>11.3f}",
                            f"{m['auc_crash']:>11.3f}",
                            f"{m['tp_crash']:>11d}", f"{m['fp_crash']:>11d}",
                            f"{m['tn_normal']:>11d}", f"{m['fn_crash']:>11d}",
                            f"{fut:>11.3f}"])

    summary: Dict[str, object] = {"stock": stock_name, "excel": excel_path,
                                    "n_scores": int(len(tad.score_series)),
                                    "n_tadgan_anomaly_windows": int((tad.label_series == 1).sum()),
                                    "report": excel_report}
    for r in results:
        print(fmt_row(r.method, r.backtest, r.future_pred_crash_rate), flush=True)
        for k, v in r.backtest.items():
            summary[f"{r.method}_{k}"] = v
        summary[f"{r.method}_threshold"] = r.threshold
        summary[f"{r.method}_future_pred_crash_rate"] = r.future_pred_crash_rate
    if ensemble:
        print(fmt_row("ensemble", ensemble["backtest"], ensemble["future_pred_crash_rate"]), flush=True)
        for k, v in ensemble["backtest"].items():
            summary[f"ensemble_{k}"] = v
        summary["ensemble_future_pred_crash_rate"] = ensemble["future_pred_crash_rate"]
        summary["ensemble_threshold"] = ensemble["threshold"]

    print(f"[STOCK] {stock_name} done -> {excel_report}", flush=True)
    return summary


def discover_excel_files() -> List[str]:
    """Return the list of Excel files to process (batch or single)."""
    if INPUT_EXCEL:
        if not os.path.isfile(INPUT_EXCEL):
            raise FileNotFoundError(f"INPUT_EXCEL not found: {INPUT_EXCEL}")
        return [INPUT_EXCEL]
    if not os.path.isdir(EXCEL_DIR):
        raise FileNotFoundError(
            f"Excel folder '{EXCEL_DIR}' does not exist. "
            "Create it and put *_final2.xlsx files inside, or set INPUT_EXCEL."
        )
    pattern = os.path.join(EXCEL_DIR, f"*{BATCH_SUFFIX}")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '*{BATCH_SUFFIX}' under '{EXCEL_DIR}'."
        )
    return files


def derive_stock_name(path: str) -> str:
    base = os.path.basename(path)
    if base.endswith(BATCH_SUFFIX):
        return base[: -len(BATCH_SUFFIX)]
    stem, _ = os.path.splitext(base)
    return stem


def main() -> None:
    ensure_dir(OUT_ROOT)
    files = discover_excel_files()
    print(f"[MAIN] OUT_ROOT={OUT_ROOT} | n_stocks={len(files)} | "
          f"EXCEL_DIR={EXCEL_DIR} | BATCH_SUFFIX={BATCH_SUFFIX}", flush=True)
    for i, f in enumerate(files, 1):
        print(f"  [{i}/{len(files)}] {f}", flush=True)

    all_summaries: List[Dict[str, object]] = []
    skipped: List[Tuple[str, str]] = []
    for f in files:
        stock_name = derive_stock_name(f)
        stock_out = os.path.join(OUT_ROOT, stock_name)
        try:
            s = run_one_stock(f, stock_out)
            if s is not None:
                all_summaries.append(s)
            else:
                skipped.append((f, "all_methods_failed"))
        except Exception as exc:
            print(f"[STOCK] {stock_name} ERROR: {exc}", flush=True)
            skipped.append((f, str(exc)))

    # Aggregate summary across all stocks
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        agg_path = os.path.join(OUT_ROOT, "all_stocks_summary.csv")
        summary_df.to_csv(agg_path, index=False)
        print(f"\n[ALL] summary saved -> {agg_path}", flush=True)
    if skipped:
        pd.DataFrame(skipped, columns=["excel", "reason"]).to_csv(
            os.path.join(OUT_ROOT, "all_stocks_skipped.csv"), index=False
        )

    print(f"[ALL] done. success={len(all_summaries)} skipped={len(skipped)}", flush=True)


if __name__ == "__main__":
    main()
