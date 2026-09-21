import os
import time
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
    roc_curve,
)

# ============================================================
# CONFIG
# ============================================================
EXCEL_DIR = "excel"
BATCH_SUFFIX = "_final2.xlsx"
EXCEL_SHEET = os.environ.get("EXCEL_SHEET", "").strip() or None
EXCEL_FIELD = os.environ.get("EXCEL_FIELD", "cp").strip()
DATE_COLUMN = os.environ.get("DATE_COLUMN", "date").strip()
TRAIN_END_DATE = os.environ.get("TRAIN_END_DATE", "2025-06-23").strip()
TEST_START_DATE = os.environ.get("TEST_START_DATE", "2025-06-24").strip()

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "32"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", str(WINDOW_SIZE)))
LATENT_DIM = int(os.environ.get("LATENT_DIM", "32"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))

N_CRITIC = int(os.environ.get("N_CRITIC", "5"))
LAMBDA_GP = float(os.environ.get("LAMBDA_GP", "10"))
LR_GAN = float(os.environ.get("LR_GAN", "0.0001"))
LR_ENC = float(os.environ.get("LR_ENC", "0.0001"))
N_EPOCHS_GAN = int(os.environ.get("N_EPOCHS_GAN", "100"))
N_EPOCHS_ENC = int(os.environ.get("N_EPOCHS_ENC", "100"))
KAPPA = float(os.environ.get("KAPPA", "1.0"))

# Backtest region: only last FORECAST_DAYS windows are test candidates.
FORECAST_DAYS = int(os.environ.get("FORECAST_DAYS", "200"))
# Crash region in GLOBAL PRICE INDEX. Windows overlapping this interval get label=1.
CRASH_START = int(os.environ.get("CRASH_START", "940"))
CRASH_END = int(os.environ.get("CRASH_END", "985"))
# Preferred: date-based crash region. When both are set, crash labels are derived
# from the window's own dates (x_axis), which always matches the windows actually
# built here -- avoiding index-space mismatches between the runner's price frame
# and this module's price loader (which produced single-class test sets).
CRASH_ONSET_DATE = os.environ.get("CRASH_ONSET_DATE", "").strip() or None
CRASH_END_DATE = os.environ.get("CRASH_END_DATE", "").strip() or None

RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
QUICK = os.environ.get("QUICK", "0").strip() != "0"
if QUICK:
    N_EPOCHS_GAN = min(N_EPOCHS_GAN, 3)
    N_EPOCHS_ENC = min(N_EPOCHS_ENC, 3)
    BATCH_SIZE = max(BATCH_SIZE, 32)
    N_CRITIC = 1

# Anomaly threshold from train-normal score quantile.
THRESHOLD_Q = float(os.environ.get("THRESHOLD_Q", "0.95"))
EXPORT_SAMPLE_GAF = os.environ.get("EXPORT_SAMPLE_GAF", "1").strip() != "0"
SAVE_ALL_SAMPLE_IMAGES = os.environ.get("SAVE_ALL_SAMPLE_IMAGES", "1").strip() != "0"
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "92"))

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

_FORCE_CPU = os.environ.get("FORCE_CPU", "0").strip() != "0"
if torch.cuda.is_available() and not _FORCE_CPU:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"[INFO] Device: {device}", flush=True)

# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clear_export_dir(path: str) -> None:
    ensure_dir(path)
    for name in os.listdir(path):
        file_path = os.path.join(path, name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def count_jpg_files(path: str) -> int:
    if not os.path.isdir(path):
        return 0
    return sum(1 for name in os.listdir(path) if name.lower().endswith(".jpg"))


def inspect_excel_structure(filepath: str) -> Dict[str, object]:
    info: Dict[str, object] = {
        "ok": True,
        "columns": [],
        "n_rows": 0,
        "issues": [],
        "date_preview": [],
    }
    lower = filepath.lower()
    if not lower.endswith((".xlsx", ".xls")):
        return info

    try:
        sheet = 0 if EXCEL_SHEET is None else int(EXCEL_SHEET) if str(EXCEL_SHEET).isdigit() else EXCEL_SHEET
        df = pd.read_excel(filepath, sheet_name=sheet)
    except Exception as e:
        info["ok"] = False
        info["issues"] = [f"unable to read excel: {e}"]
        return info

    info["n_rows"] = int(len(df))
    info["columns"] = [str(c) for c in df.columns]
    cols_lower = {str(c).lower(): c for c in df.columns}

    issues: List[str] = []
    if EXCEL_FIELD.lower() not in cols_lower:
        issues.append(f"missing price column '{EXCEL_FIELD}'")
    else:
        price_series = pd.to_numeric(df[cols_lower[EXCEL_FIELD.lower()]], errors="coerce")
        if int(price_series.notna().sum()) == 0:
            issues.append(f"price column '{EXCEL_FIELD}' has no numeric values")

    if DATE_COLUMN.lower() not in cols_lower:
        issues.append(f"missing date column '{DATE_COLUMN}'")
    else:
        parsed_dates = parse_date_series(df[cols_lower[DATE_COLUMN.lower()]])
        valid_dates = parsed_dates.dropna()
        info["date_preview"] = [str(pd.Timestamp(v).date()) for v in valid_dates.head(5)]
        if len(valid_dates) == 0:
            issues.append(f"date column '{DATE_COLUMN}' could not be parsed")
        elif not (valid_dates >= pd.Timestamp(TEST_START_DATE)).any():
            issues.append(f"no parsed dates on/after TEST_START_DATE={TEST_START_DATE}")

    info["issues"] = issues
    info["ok"] = len(issues) == 0
    return info


def parse_date_series(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    be_mask = text.str.fullmatch(r"(24|25|26)\d{6}")
    if be_mask.any():
        be_text = text[be_mask]
        be_year = pd.to_numeric(be_text.str.slice(0, 4), errors="coerce") - 543
        be_month = be_text.str.slice(4, 6)
        be_day = be_text.str.slice(6, 8)
        gregorian_text = be_year.astype("Int64").astype(str) + "-" + be_month + "-" + be_day
        parsed.loc[be_mask] = pd.to_datetime(gregorian_text, format="%Y-%m-%d", errors="coerce")

    for formatted_text, fmt in [
        (text.str.zfill(6), "%d%m%y"),
        (text.str.zfill(8), "%d%m%Y"),
        (text.str.zfill(8), "%Y%m%d"),
    ]:
        parsed = parsed.fillna(pd.to_datetime(formatted_text, format=fmt, errors="coerce"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed = parsed.fillna(pd.to_datetime(text, errors="coerce", dayfirst=True))

    numeric = pd.to_numeric(series, errors="coerce")
    excel_mask = numeric.between(20000, 60000)
    if excel_mask.any():
        parsed.loc[excel_mask & parsed.isna()] = pd.to_datetime(
            numeric[excel_mask & parsed.isna()],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )
    return parsed


def load_prices_and_dates(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    lower = filepath.lower()
    if lower.endswith((".xlsx", ".xls")):
        sheet = 0 if EXCEL_SHEET is None else int(EXCEL_SHEET) if str(EXCEL_SHEET).isdigit() else EXCEL_SHEET
        df = pd.read_excel(filepath, sheet_name=sheet)
        cols_lower = {c.lower(): c for c in df.columns}
        key = EXCEL_FIELD.lower()
        if key not in cols_lower:
            raise KeyError(f"Column '{EXCEL_FIELD}' not found in {filepath}. Available: {list(df.columns)}")

        price_col = cols_lower[key]
        prices = pd.to_numeric(df[price_col], errors="coerce")
        valid_mask = prices.notna()

        date_key = DATE_COLUMN.lower()
        if date_key in cols_lower:
            parsed_dates = parse_date_series(df[cols_lower[date_key]])
            date_valid_mask = valid_mask & parsed_dates.notna()
            if date_valid_mask.any():
                prices = prices[date_valid_mask].astype(np.float64)
                x_axis = parsed_dates[date_valid_mask].to_numpy()
            else:
                prices = prices[valid_mask].astype(np.float64)
                x_axis = np.arange(len(prices))
        else:
            prices = prices[valid_mask].astype(np.float64)
            x_axis = np.arange(len(prices))
        return prices.to_numpy(dtype=np.float64), np.asarray(x_axis)

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()
    vals = [float(v) for v in (text.split(",") if "," in text else text.splitlines()) if v.strip()]
    prices = np.array(vals, dtype=np.float64)
    return prices, np.arange(len(prices))


def load_prices(filepath: str) -> np.ndarray:
    prices, _ = load_prices_and_dates(filepath)
    return prices


def minmax_01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def resize_1d(x: np.ndarray, out_size: int) -> np.ndarray:
    if len(x) == out_size:
        return x
    idx = np.linspace(0, len(x) - 1, out_size)
    return np.interp(idx, np.arange(len(x)), x)


def gaf_cosine(series: np.ndarray, out_size: int = IMG_SIZE) -> np.ndarray:
    s01 = minmax_01(series)
    s = np.clip(2.0 * s01 - 1.0, -1.0, 1.0)
    phi = np.arccos(s)
    phi = resize_1d(phi, out_size)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def gaf_arctan(series: np.ndarray, out_size: int = IMG_SIZE) -> np.ndarray:
    s01 = minmax_01(series)
    phi = np.arctan(s01)
    phi = resize_1d(phi, out_size)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def gaf_arccosh(series: np.ndarray, out_size: int = IMG_SIZE) -> np.ndarray:
    s01 = minmax_01(series)
    phi = np.arccosh(1.0 + s01)
    phi = resize_1d(phi, out_size)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


def gaf_exponential(series: np.ndarray, out_size: int = IMG_SIZE) -> np.ndarray:
    s01 = minmax_01(series)
    phi = np.pi * (np.exp(s01) - 1.0) / (np.e - 1.0)
    phi = resize_1d(phi, out_size)
    return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)


GAF_METHODS: Dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "cosine": gaf_cosine,
    "exponential": gaf_exponential,
    "arctan": gaf_arctan,
    "arccosh": gaf_arccosh,
}


def gaf_to_uint8(gaf: np.ndarray) -> np.ndarray:
    return ((gaf + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)


def gaf_to_color_uint8(gaf: np.ndarray, cmap_name: str = "turbo") -> np.ndarray:
    arr = ((gaf + 1.0) * 0.5).clip(0, 1)
    cmap = plt.get_cmap(cmap_name)
    rgba = cmap(arr)
    rgb = (rgba[..., :3] * 255.0).clip(0, 255).astype(np.uint8)
    return rgb


def save_gaf_jpg(gaf: np.ndarray, path: str) -> None:
    arr = gaf_to_uint8(gaf)
    im = Image.fromarray(arr).convert("RGB")
    if im.size != (IMG_SIZE, IMG_SIZE):
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    im.save(path, quality=JPEG_QUALITY)


def save_gaf_color_jpg(gaf: np.ndarray, path: str) -> None:
    arr = gaf_to_color_uint8(gaf)
    im = Image.fromarray(arr, mode="RGB")
    if im.size != (IMG_SIZE, IMG_SIZE):
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    im.save(path, quality=JPEG_QUALITY)


def label_window(start: int, end: int) -> int:
    # label: 0 = normal, 1 = crash
    return 1 if ((start < CRASH_END) and (end > CRASH_START)) else 0


@dataclass
class EvalResult:
    stock_name: str
    method: str
    n_train_normal: int
    n_test_total: int
    n_test_normal: int
    n_test_crash: int
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    mean_train_score: float
    mean_test_score: float


# ============================================================
# MODELS
# ============================================================
_H = IMG_SIZE // 16
if _H < 1:
    raise ValueError("IMG_SIZE must be at least 16.")


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
        # Transposed-conv stacks only reproduce exact sizes when IMG_SIZE is a multiple of 16.
        # Resize here so fake/reconstructed images always match real GAF tensors.
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
        out, interp,
        grad_outputs=torch.ones_like(out),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((grad.norm(2, dim=[1, 2, 3]) - 1) ** 2).mean()


def train_wgangp(G, D, loader):
    opt_G = optim.Adam(G.parameters(), lr=LR_GAN, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=LR_GAN, betas=(0.0, 0.9))
    G.train(); D.train()
    loss_g_hist, loss_d_hist = [], []
    for epoch in range(1, N_EPOCHS_GAN + 1):
        gsum = dsum = 0.0
        n_batches = 0
        for real, in loader:
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
            gsum += float(g_loss.item())
            dsum += float(d_loss.item())
            n_batches += 1
        loss_g_hist.append(gsum / max(n_batches, 1))
        loss_d_hist.append(dsum / max(n_batches, 1))
        print(f"    [GAN] epoch {epoch:03d}/{N_EPOCHS_GAN} D={loss_d_hist[-1]:.4f} G={loss_g_hist[-1]:.4f}", flush=True)
    return loss_g_hist, loss_d_hist


def train_encoder(E, G, D, loader):
    opt_E = optim.Adam(E.parameters(), lr=LR_ENC, betas=(0.5, 0.999))
    E.train(); G.eval(); D.eval()
    loss_hist = []
    for epoch in range(1, N_EPOCHS_ENC + 1):
        esum = 0.0
        n_batches = 0
        for real, in loader:
            real = real.to(device)
            z_hat = E(real)
            recon = G(z_hat)
            loss_rec = nn.functional.mse_loss(recon, real)
            with torch.no_grad():
                f_real = D.features(real)
            f_recon = D.features(recon)
            loss_feat = nn.functional.mse_loss(f_recon, f_real.detach())
            loss = loss_rec + KAPPA * loss_feat
            opt_E.zero_grad(); loss.backward(); opt_E.step()
            esum += float(loss.item())
            n_batches += 1
        loss_hist.append(esum / max(n_batches, 1))
        print(f"    [ENC] epoch {epoch:03d}/{N_EPOCHS_ENC} loss={loss_hist[-1]:.6f}", flush=True)
    return loss_hist


def anomaly_scores(E, G, D, x: np.ndarray, batch_size: int = 64) -> np.ndarray:
    E.eval(); G.eval(); D.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.tensor(x[i:i+batch_size], dtype=torch.float32).unsqueeze(1).to(device)
            z = E(xb)
            recon = G(z)
            rec = ((xb - recon) ** 2).mean(dim=[1, 2, 3])
            f_real = D.features(xb)
            f_recon = D.features(recon)
            feat = ((f_real - f_recon) ** 2).mean(dim=[1, 2, 3])
            score = rec + KAPPA * feat
            scores.extend(score.detach().cpu().numpy().tolist())
    return np.array(scores, dtype=np.float64)


# ============================================================
# PIPELINE
# ============================================================

def build_windows(prices: np.ndarray, gaf_fn: Callable[[np.ndarray, int], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gafs, labels, starts = [], [], []
    n = len(prices)
    for start in range(0, n - WINDOW_SIZE + 1):
        end = start + WINDOW_SIZE
        gafs.append(gaf_fn(prices[start:end], IMG_SIZE))
        labels.append(label_window(start, end))
        starts.append(start)
    return np.array(gafs, dtype=np.float32), np.array(labels, dtype=np.int64), np.array(starts, dtype=np.int64)


def build_date_split_masks(starts: np.ndarray, x_axis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, pd.Timestamp, pd.Timestamp]:
    x_dates = pd.to_datetime(np.asarray(x_axis), errors="coerce")
    if x_dates.isna().any():
        raise RuntimeError(
            f"Date-based split requires a valid '{DATE_COLUMN}' column in the input Excel file."
        )

    train_end_ts = pd.Timestamp(TRAIN_END_DATE)
    test_start_ts = pd.Timestamp(TEST_START_DATE)
    if test_start_ts <= train_end_ts:
        raise ValueError("TEST_START_DATE must be later than TRAIN_END_DATE.")

    window_start_dates = x_dates[starts]
    candidate_positions = np.where(window_start_dates >= test_start_ts)[0]
    if len(candidate_positions) == 0:
        raise RuntimeError(f"No rows found on or after TEST_START_DATE={TEST_START_DATE}.")

    test_indices = candidate_positions[:FORECAST_DAYS]
    test_mask = np.zeros(len(starts), dtype=bool)
    test_mask[test_indices] = True
    test_last_ts = pd.Timestamp(window_start_dates[test_indices[-1]])
    train_mask = window_start_dates <= train_end_ts
    return np.asarray(train_mask, dtype=bool), np.asarray(test_mask, dtype=bool), train_end_ts, test_last_ts


def safe_savefig(fig, out_path, **kwargs):
    """Save a matplotlib figure without ever aborting the run.

    Plots are cosmetic; the scientific outputs (metrics_summary.csv,
    test_scores.csv) must not be lost because a figure failed to save
    (e.g. intermittent FileNotFoundError from the Agg JPEG path on Windows).
    Ensures the parent directory exists, retries once as PNG, then gives up.
    """
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, **kwargs)
    except Exception as exc:
        try:
            png_path = os.path.splitext(out_path)[0] + ".png"
            os.makedirs(os.path.dirname(png_path), exist_ok=True)
            fig.savefig(png_path, **{k: v for k, v in kwargs.items() if k != "quality"})
            print(f"    [WARN] saved {png_path} (jpg failed: {exc})", flush=True)
        except Exception as exc2:
            print(f"    [WARN] could not save figure {out_path}: {exc2}", flush=True)
    finally:
        plt.close(fig)


def save_loss_plot(g_losses, d_losses, e_losses, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(d_losses, label='D loss')
    axes[0].plot(g_losses, label='G loss')
    axes[0].set_title('WGAN-GP')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(e_losses, label='Encoder loss')
    axes[1].set_title('Encoder')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    safe_savefig(fig, out_path, dpi=140, bbox_inches='tight')


def save_eval_plots(y_true, scores, y_pred, out_dir):
    ensure_dir(out_dir)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred normal(0)', 'Pred crash(1)'])
    ax.set_yticklabels(['True normal(0)', 'True crash(1)'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12)
    fig.colorbar(im, ax=ax)
    ax.set_title('Confusion Matrix')
    plt.tight_layout(); safe_savefig(fig, os.path.join(out_dir, 'confusion_matrix.jpg'), dpi=150, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(5, 4))
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc = roc_auc_score(y_true, scores)
        ax.plot(fpr, tpr, label=f'AUC={auc:.3f}')
    except Exception:
        ax.text(0.5, 0.5, 'ROC unavailable', ha='center', va='center')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); safe_savefig(fig, os.path.join(out_dir, 'roc_curve.jpg'), dpi=150, bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[y_true == 0], bins=25, alpha=0.7, label='Normal label=0')
    ax.hist(scores[y_true == 1], bins=25, alpha=0.7, label='Crash label=1')
    ax.set_title('Anomaly Score Distribution')
    ax.set_xlabel('Anomaly score')
    ax.set_ylabel('Count')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); safe_savefig(fig, os.path.join(out_dir, 'score_histogram.jpg'), dpi=150, bbox_inches='tight')


def save_labeled_gaf_set(
    gafs: np.ndarray,
    labels: np.ndarray,
    starts: np.ndarray,
    out_dir: str,
    split_name: str,
    write_txt: bool = True,
    color_out_dir: str = "",
) -> int:
    """Save all GAF images as RGB JPG and optionally a .txt label file.

    label convention:
      0 = normal
      1 = crash
    """
    if not EXPORT_SAMPLE_GAF:
        return 0
    ensure_dir(out_dir)
    if color_out_dir:
        ensure_dir(color_out_dir)
    saved = 0
    for gaf, label, start in zip(gafs, labels, starts):
        base = f"{split_name}_t{int(start):05d}_y{int(label)}"
        jpg_path = os.path.join(out_dir, base + '.jpg')
        save_gaf_jpg(gaf, jpg_path)
        if color_out_dir:
            color_jpg_path = os.path.join(color_out_dir, base + '.jpg')
            save_gaf_color_jpg(gaf, color_jpg_path)
        if write_txt:
            txt_path = os.path.join(out_dir, base + '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(str(int(label)))
        saved += 1
    return saved


def save_price_and_anomaly_figure(
    stock_name: str,
    x_axis: np.ndarray,
    prices: np.ndarray,
    aligned_scores: pd.DataFrame,
    summary: pd.DataFrame,
    out_path: str,
) -> None:
    methods = list(GAF_METHODS.keys())
    fig, axes = plt.subplots(
        1 + len(methods),
        1,
        figsize=(14, 16),
        sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1, 1, 1, 1]},
    )

    price_ax = axes[0]
    price_ax.plot(x_axis, prices, color="black", linewidth=1.3, label="Close price")
    price_ax.set_title(f"{stock_name}: close price with last {FORECAST_DAYS} days anomaly scores")
    price_ax.set_ylabel("Close")
    price_ax.grid(True, alpha=0.3)

    test_start = int(aligned_scores["window_start"].min())
    test_end = int(aligned_scores["window_end"].max())
    price_ax.axvspan(x_axis[test_start], x_axis[test_end], color="#FFE082", alpha=0.35, label="Test region")
    crash_lo = max(0, CRASH_START)
    crash_hi = min(len(prices) - 1, CRASH_END - 1)
    if crash_lo <= crash_hi:
        price_ax.axvspan(x_axis[crash_lo], x_axis[crash_hi], color="#FFCDD2", alpha=0.35, label="Crash region")
    price_ax.legend(loc="upper left")

    plot_x = x_axis[aligned_scores["window_end"].to_numpy(dtype=np.int64)]
    true_label = aligned_scores["label_0normal_1crash"].to_numpy(dtype=np.int64)
    colors = {
        "cosine": "#1565C0",
        "exponential": "#2E7D32",
        "arctan": "#6A1B9A",
        "arccosh": "#EF6C00",
    }
    for ax, method in zip(axes[1:], methods):
        score_col = f"{method}_score"
        pred_col = f"{method}_pred"
        threshold = float(summary.loc[summary["method"] == method, "threshold"].iloc[0])
        scores = aligned_scores[score_col].to_numpy(dtype=np.float64)
        preds = aligned_scores[pred_col].to_numpy(dtype=np.int64)
        ax.plot(plot_x, scores, color=colors.get(method, "#1976D2"), linewidth=1.2, label=f"{method} score")
        ax.axhline(threshold, color="#D32F2F", linestyle="--", linewidth=1.0, label=f"threshold={threshold:.4f}")
        ax.scatter(plot_x[true_label == 1], scores[true_label == 1], color="#C62828", s=12, label="true crash")
        ax.scatter(plot_x[preds == 1], scores[preds == 1], facecolors="none", edgecolors="#000000", s=24, linewidths=0.8, label="pred crash")
        ax.set_ylabel("Score")
        ax.set_title(method)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", ncols=2, fontsize=8)

    axes[-1].set_xlabel("Date" if np.issubdtype(np.asarray(x_axis).dtype, np.datetime64) else "Index")
    plt.tight_layout()
    safe_savefig(fig, out_path, dpi=150, bbox_inches="tight")


def save_stock_excel_report(stock_out: str, summary: pd.DataFrame, aligned_scores: pd.DataFrame, detail_tables: Dict[str, pd.DataFrame]) -> None:
    out_path = os.path.join(stock_out, "four_gaf_comparison.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        aligned_scores.to_excel(writer, sheet_name="aligned_test_scores", index=False)
        for method, df in detail_tables.items():
            sheet_name = f"{method[:20]}_detail"
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def run_one_method(
    stock_name: str,
    prices: np.ndarray,
    x_axis: np.ndarray,
    method_name: str,
    gaf_fn: Callable[[np.ndarray, int], np.ndarray],
    stock_out: str,
) -> Tuple[EvalResult, pd.DataFrame]:
    print(f"\n[RUN] {stock_name} | method={method_name}", flush=True)
    method_dir = os.path.join(stock_out, method_name)
    ensure_dir(method_dir)
    ensure_dir(os.path.join(method_dir, 'plots'))
    input_root = os.path.join(method_dir, 'input_gaf')
    input_color_root = os.path.join(method_dir, 'input_gaf_color')
    input_train_dir = os.path.join(input_root, 'train_normal')
    input_test_normal_dir = os.path.join(input_root, 'test_normal')
    input_test_crash_dir = os.path.join(input_root, 'test_crash')
    input_color_train_dir = os.path.join(input_color_root, 'train_normal')
    input_color_test_normal_dir = os.path.join(input_color_root, 'test_normal')
    input_color_test_crash_dir = os.path.join(input_color_root, 'test_crash')
    samples_root = os.path.join(method_dir, 'samples')
    samples_color_root = os.path.join(method_dir, 'samples_color')
    train_normal_dir = os.path.join(samples_root, 'train_normal')
    test_normal_dir = os.path.join(samples_root, 'test_normal')
    test_crash_dir = os.path.join(samples_root, 'test_crash')
    train_normal_color_dir = os.path.join(samples_color_root, 'train_normal')
    test_normal_color_dir = os.path.join(samples_color_root, 'test_normal')
    test_crash_color_dir = os.path.join(samples_color_root, 'test_crash')
    ensure_dir(input_root)
    ensure_dir(input_color_root)
    ensure_dir(input_train_dir)
    ensure_dir(input_test_normal_dir)
    ensure_dir(input_test_crash_dir)
    ensure_dir(input_color_train_dir)
    ensure_dir(input_color_test_normal_dir)
    ensure_dir(input_color_test_crash_dir)
    ensure_dir(samples_root)
    ensure_dir(samples_color_root)
    ensure_dir(train_normal_dir)
    ensure_dir(test_normal_dir)
    ensure_dir(test_crash_dir)
    ensure_dir(train_normal_color_dir)
    ensure_dir(test_normal_color_dir)
    ensure_dir(test_crash_color_dir)

    all_gafs, all_labels, all_starts = build_windows(prices, gaf_fn)
    n_windows = len(all_gafs)

    # Date-based crash labels (preferred): a window is crash=1 if its own date
    # span overlaps [CRASH_ONSET_DATE, CRASH_END_DATE]. Uses x_axis directly so
    # labels always line up with the windows built here (no index mismatch).
    if CRASH_ONSET_DATE and CRASH_END_DATE:
        x_dates = pd.to_datetime(np.asarray(x_axis), errors="coerce")
        onset = pd.Timestamp(CRASH_ONSET_DATE)
        cend = pd.Timestamp(CRASH_END_DATE)
        win_start_dates = x_dates[all_starts]
        win_end_dates = x_dates[all_starts + WINDOW_SIZE - 1]
        all_labels = ((win_start_dates <= cend) & (win_end_dates >= onset)).astype(np.int64)
        all_labels = np.asarray(all_labels, dtype=np.int64)

    train_mask, test_mask, train_end_ts, test_last_ts = build_date_split_masks(all_starts, x_axis)

    # Train uses only normal windows up to TRAIN_END_DATE.
    train_normal_mask = train_mask & (all_labels == 0)
    x_train = all_gafs[train_normal_mask]
    y_test = all_labels[test_mask]
    x_test = all_gafs[test_mask]
    starts_test = all_starts[test_mask]
    starts_train_normal = all_starts[train_normal_mask]

    if len(x_train) < max(BATCH_SIZE, 8):
        raise RuntimeError(f"Not enough normal training windows for {stock_name}/{method_name}.")

    print(
        f"    [DATA] windows total={len(all_gafs)} | train_normal={len(x_train)} "
        f"| test={len(x_test)} | crash_in_test={(y_test == 1).sum()} | normal_in_test={(y_test == 0).sum()}",
        flush=True,
    )
    print(
        f"    [SPLIT] train_end={train_end_ts.date()} | test_start={pd.Timestamp(TEST_START_DATE).date()} | "
        f"test_end={test_last_ts.date()}",
        flush=True,
    )
    if len(x_test) == 0:
        raise RuntimeError(f"No test windows for {stock_name}/{method_name}.")

    if EXPORT_SAMPLE_GAF:
        print("    [PREP] Exporting input GAF JPG files before running f-AnoGAN ...", flush=True)
        clear_export_dir(input_train_dir)
        clear_export_dir(input_test_normal_dir)
        clear_export_dir(input_test_crash_dir)
        clear_export_dir(input_color_train_dir)
        clear_export_dir(input_color_test_normal_dir)
        clear_export_dir(input_color_test_crash_dir)
        n_input_train = save_labeled_gaf_set(
            x_train,
            np.zeros(len(x_train), dtype=np.int64),
            starts_train_normal,
            input_train_dir,
            'train_normal',
            write_txt=False,
            color_out_dir=input_color_train_dir,
        )
        gt_test_normal_mask = (y_test == 0)
        gt_test_crash_mask = (y_test == 1)
        n_input_test_normal = save_labeled_gaf_set(
            x_test[gt_test_normal_mask],
            y_test[gt_test_normal_mask],
            starts_test[gt_test_normal_mask],
            input_test_normal_dir,
            'test_normal',
            write_txt=False,
            color_out_dir=input_color_test_normal_dir,
        )
        n_input_test_crash = save_labeled_gaf_set(
            x_test[gt_test_crash_mask],
            y_test[gt_test_crash_mask],
            starts_test[gt_test_crash_mask],
            input_test_crash_dir,
            'test_crash',
            write_txt=False,
            color_out_dir=input_color_test_crash_dir,
        )
        print(
            f"    [INPUT JPG] train_normal={n_input_train} -> {input_train_dir}",
            flush=True,
        )
        print(
            f"    [INPUT JPG] test_normal={n_input_test_normal} -> {input_test_normal_dir}",
            flush=True,
        )
        print(
            f"    [INPUT JPG] test_crash={n_input_test_crash} -> {input_test_crash_dir}",
            flush=True,
        )
        print(
            f"    [INPUT JPG] verified counts: "
            f"train_normal={count_jpg_files(input_train_dir)}, "
            f"test_normal={count_jpg_files(input_test_normal_dir)}, "
            f"test_crash={count_jpg_files(input_test_crash_dir)}",
            flush=True,
        )
        print(
            f"    [INPUT COLOR JPG] train_normal={count_jpg_files(input_color_train_dir)} -> {input_color_train_dir}",
            flush=True,
        )
        print(
            f"    [INPUT COLOR JPG] test_normal={count_jpg_files(input_color_test_normal_dir)} -> {input_color_test_normal_dir}",
            flush=True,
        )
        print(
            f"    [INPUT COLOR JPG] test_crash={count_jpg_files(input_color_test_crash_dir)} -> {input_color_test_crash_dir}",
            flush=True,
        )
    else:
        print("    [PREP] EXPORT_SAMPLE_GAF=0 so input GAF JPG export is skipped.", flush=True)

    print("    [RUN] Starting f-AnoGAN training ...", flush=True)

    loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)),
                        batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    G = Generator().to(device)
    D = Discriminator().to(device)
    E = Encoder().to(device)

    g_losses, d_losses = train_wgangp(G, D, loader)
    e_losses = train_encoder(E, G, D, loader)
    save_loss_plot(g_losses, d_losses, e_losses, os.path.join(method_dir, 'plots', 'training_losses.jpg'))

    train_scores = anomaly_scores(E, G, D, x_train)
    test_scores = anomaly_scores(E, G, D, x_test)
    threshold = float(np.quantile(train_scores, THRESHOLD_Q))
    y_pred = (test_scores > threshold).astype(np.int64)

    if SAVE_ALL_SAMPLE_IMAGES:
        clear_export_dir(train_normal_dir)
        clear_export_dir(test_normal_dir)
        clear_export_dir(test_crash_dir)
        clear_export_dir(train_normal_color_dir)
        clear_export_dir(test_normal_color_dir)
        clear_export_dir(test_crash_color_dir)
        n_train_saved = save_labeled_gaf_set(
            x_train,
            np.zeros(len(x_train), dtype=np.int64),
            starts_train_normal,
            train_normal_dir,
            'train_normal',
            color_out_dir=train_normal_color_dir,
        )
        test_normal_mask_pred = (y_pred == 0)
        test_crash_mask_pred = (y_pred == 1)
        n_test_normal_saved = save_labeled_gaf_set(
            x_test[test_normal_mask_pred],
            np.zeros(int(test_normal_mask_pred.sum()), dtype=np.int64),
            starts_test[test_normal_mask_pred],
            test_normal_dir,
            'test_normal',
            color_out_dir=test_normal_color_dir,
        )
        n_test_crash_saved = save_labeled_gaf_set(
            x_test[test_crash_mask_pred],
            np.ones(int(test_crash_mask_pred.sum()), dtype=np.int64),
            starts_test[test_crash_mask_pred],
            test_crash_dir,
            'test_crash',
            color_out_dir=test_crash_color_dir,
        )
        print(f"    [EXPORT] train_normal={n_train_saved} | test_normal={n_test_normal_saved} | test_crash={n_test_crash_saved}", flush=True)
        print(
            f"    [EXPORT COLOR] train_normal={count_jpg_files(train_normal_color_dir)} -> {train_normal_color_dir}",
            flush=True,
        )
        print(
            f"    [EXPORT COLOR] test_normal={count_jpg_files(test_normal_color_dir)} -> {test_normal_color_dir}",
            flush=True,
        )
        print(
            f"    [EXPORT COLOR] test_crash={count_jpg_files(test_crash_color_dir)} -> {test_crash_color_dir}",
            flush=True,
        )

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))
    try:
        auc = float(roc_auc_score(y_test, test_scores))
    except Exception:
        auc = float('nan')

    save_eval_plots(y_test, test_scores, y_pred, os.path.join(method_dir, 'plots'))

    detail_df = pd.DataFrame({
        'method': method_name,
        'window_start': starts_test,
        'window_end': starts_test + WINDOW_SIZE - 1,
        'label_0normal_1crash': y_test,
        'anomaly_score': test_scores,
        'threshold': threshold,
        'predicted_label': y_pred,
        'predicted_split_folder': np.where(y_pred == 1, 'samples/test_crash', 'samples/test_normal'),
    })
    detail_df.to_csv(os.path.join(method_dir, 'test_scores.csv'), index=False)

    result = EvalResult(
        stock_name=stock_name,
        method=method_name,
        n_train_normal=int(len(x_train)),
        n_test_total=int(len(x_test)),
        n_test_normal=int((y_test == 0).sum()),
        n_test_crash=int((y_test == 1).sum()),
        threshold=threshold,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        auc=auc,
        mean_train_score=float(np.mean(train_scores)),
        mean_test_score=float(np.mean(test_scores)),
    )

    pd.DataFrame([asdict(result)]).to_csv(os.path.join(method_dir, 'metrics_summary.csv'), index=False)
    print(f"    [RESULT] acc={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f} auc={auc:.4f}", flush=True)
    return result, detail_df


def run_one_stock(data_file: str, stock_name: str) -> List[EvalResult]:
    prices, x_axis = load_prices_and_dates(data_file)
    stock_out = os.path.join(stock_name, 'out_put_four_gaf_fanogan')
    ensure_dir(stock_out)
    print(f"\n{'='*90}\n[STOCK] {stock_name} | file={data_file} | n_prices={len(prices)}\n{'='*90}", flush=True)
    results = []
    detail_tables: Dict[str, pd.DataFrame] = {}
    for method_name, gaf_fn in GAF_METHODS.items():
        t0 = time.time()
        result, detail_df = run_one_method(stock_name, prices, x_axis, method_name, gaf_fn, stock_out)
        print(f"    [TIME] {method_name} took {(time.time()-t0)/60:.2f} min", flush=True)
        results.append(result)
        detail_tables[method_name] = detail_df
        torch.cuda.empty_cache()

    summary = pd.DataFrame([asdict(r) for r in results]).sort_values(by=['f1', 'auc', 'accuracy'], ascending=False)
    summary.to_csv(os.path.join(stock_out, 'comparison_summary.csv'), index=False)
    summary.to_excel(os.path.join(stock_out, 'comparison_summary.xlsx'), index=False)

    aligned_scores = None
    for method_name in GAF_METHODS.keys():
        detail_df = detail_tables[method_name].copy()
        detail_df = detail_df.rename(
            columns={
                'anomaly_score': f'{method_name}_score',
                'predicted_label': f'{method_name}_pred',
                'threshold': f'{method_name}_threshold',
            }
        )
        keep_cols = ['window_start', 'window_end', 'label_0normal_1crash', f'{method_name}_score', f'{method_name}_pred', f'{method_name}_threshold']
        if aligned_scores is None:
            aligned_scores = detail_df[keep_cols]
        else:
            aligned_scores = aligned_scores.merge(
                detail_df[keep_cols],
                on=['window_start', 'window_end', 'label_0normal_1crash'],
                how='inner',
            )

    aligned_scores = aligned_scores.sort_values(by='window_start').reset_index(drop=True)
    aligned_scores['close_price_at_window_end'] = prices[aligned_scores['window_end'].to_numpy(dtype=np.int64)]
    if np.issubdtype(np.asarray(x_axis).dtype, np.datetime64):
        aligned_scores['window_end_date'] = pd.to_datetime(x_axis[aligned_scores['window_end'].to_numpy(dtype=np.int64)])

    save_stock_excel_report(stock_out, summary, aligned_scores, detail_tables)
    save_price_and_anomaly_figure(
        stock_name,
        x_axis,
        prices,
        aligned_scores,
        summary,
        os.path.join(stock_out, 'price_and_anomaly_scores.jpg'),
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    methods = summary['method'].tolist()
    ax.bar(methods, summary['f1'].to_numpy(), label='F1')
    ax.plot(methods, summary['auc'].to_numpy(), marker='o', label='AUC')
    ax.plot(methods, summary['accuracy'].to_numpy(), marker='s', label='Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{stock_name}: four GAF mappings comparison')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout(); safe_savefig(fig, os.path.join(stock_out, 'comparison_chart.jpg'), dpi=150, bbox_inches='tight')
    return results


def run_all_stocks_in_excel_dir() -> None:
    if not os.path.isdir(EXCEL_DIR):
        raise FileNotFoundError(f"Excel directory not found: ./{EXCEL_DIR}")
    files = sorted([fn for fn in os.listdir(EXCEL_DIR) if fn.lower().endswith(BATCH_SUFFIX.lower()) and not fn.startswith('~$')])
    if not files:
        raise FileNotFoundError(f"No '*{BATCH_SUFFIX}' files found under ./{EXCEL_DIR}")
    all_results = []
    skipped_files = []
    for i, fn in enumerate(files, 1):
        data_file = os.path.join(EXCEL_DIR, fn)
        stock_name = fn[:-len(BATCH_SUFFIX)].strip().strip('_')
        print(f"\n{'#'*100}\n[BATCH] ({i}/{len(files)}) stock={stock_name}\n{'#'*100}", flush=True)
        excel_info = inspect_excel_structure(data_file)
        print(
            f"[CHECK] rows={excel_info.get('n_rows', 0)} | columns={len(excel_info.get('columns', []))} "
            f"| date_preview={excel_info.get('date_preview', [])}",
            flush=True,
        )
        if not bool(excel_info.get("ok", False)):
            reason = "; ".join([str(x) for x in excel_info.get("issues", [])]) or "invalid excel structure"
            print(f"[SKIP] {stock_name} skipped before run: {reason}", flush=True)
            skipped_files.append({"stock_name": stock_name, "file": data_file, "reason": reason})
            continue
        try:
            all_results.extend(run_one_stock(data_file, stock_name))
        except Exception as e:
            reason = str(e)
            print(f"[SKIP] {stock_name} failed during run and will be skipped: {reason}", flush=True)
            skipped_files.append({"stock_name": stock_name, "file": data_file, "reason": reason})
            torch.cuda.empty_cache()
            continue

    pd.DataFrame([asdict(r) for r in all_results]).to_csv('all_stocks_four_gaf_fanogan_summary.csv', index=False)
    if skipped_files:
        pd.DataFrame(skipped_files).to_csv('all_stocks_four_gaf_fanogan_skipped.csv', index=False)
        print(f"[DONE] Saved all_stocks_four_gaf_fanogan_skipped.csv ({len(skipped_files)} skipped files)", flush=True)
    print("\n[DONE] Saved all_stocks_four_gaf_fanogan_summary.csv", flush=True)


def main():
    run_all_stocks_in_excel_dir()


if __name__ == '__main__':
    main()
