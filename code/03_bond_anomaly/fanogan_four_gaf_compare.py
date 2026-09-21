import os
import time
import random
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
# --- replication package (added): data in <repo>/input, results in <repo>/output/03_bond_anomaly ---
import os as _rp_os, sys as _rp_sys
from pathlib import Path as _RpPath
REPO_ROOT = _RpPath(__file__).resolve().parents[2]
REPO_INPUT = REPO_ROOT / "input"
REPO_OUTPUT = REPO_ROOT / "output" / "03_bond_anomaly"
REPO_OUTPUT.mkdir(parents=True, exist_ok=True)
_rp_sys.path.insert(0, str(REPO_ROOT / "code" / "common"))
if __name__ == "__main__":
    _rp_os.chdir(REPO_OUTPUT)  # original script writes into the working directory
# ------------------------------------------------------------------------------------------
EXCEL_DIR = _rp_os.environ.get("EXCEL_DIR", str(REPO_INPUT / "gaf_series"))
BATCH_SUFFIX = "_final2.xlsx"
EXCEL_SHEET = os.environ.get("EXCEL_SHEET", "").strip() or None
EXCEL_FIELD = os.environ.get("EXCEL_FIELD", "cp").strip()
DATE_COLUMN = os.environ.get("DATE_COLUMN", "date").strip()

WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", "32"))
IMG_SIZE = int(os.environ.get("IMG_SIZE", str(WINDOW_SIZE)))
LATENT_DIM = int(os.environ.get("LATENT_DIM", "32"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))

N_CRITIC = int(os.environ.get("N_CRITIC", "5"))
LAMBDA_GP = float(os.environ.get("LAMBDA_GP", "10"))
LR_GAN = float(os.environ.get("LR_GAN", "0.0001"))
LR_ENC = float(os.environ.get("LR_ENC", "0.0001"))
N_EPOCHS_GAN = int(os.environ.get("N_EPOCHS_GAN", "10"))
N_EPOCHS_ENC = int(os.environ.get("N_EPOCHS_ENC", "10"))
KAPPA = float(os.environ.get("KAPPA", "1.0"))

# Backtest region: only last FORECAST_DAYS windows are test candidates.
FORECAST_DAYS = int(os.environ.get("FORECAST_DAYS", "200"))
# Crash region in GLOBAL PRICE INDEX. Windows overlapping this interval get label=1.
CRASH_START = int(os.environ.get("CRASH_START", "940"))
CRASH_END = int(os.environ.get("CRASH_END", "985"))

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
SAMPLE_EXPORT_COUNT = int(os.environ.get("SAMPLE_EXPORT_COUNT", "3"))

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

ALLOW_CPU = os.environ.get("ALLOW_CPU", "0").strip() != "0"
if not torch.cuda.is_available():
    if not ALLOW_CPU:
        raise RuntimeError("CUDA is required but not available. Set ALLOW_CPU=1 to run on CPU.")
    device = torch.device("cpu")
else:
    device = torch.device("cuda")
print(f"[INFO] Device: {device}", flush=True)

# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


DATA_END_DATE = os.environ.get("DATA_END_DATE", "").strip() or None


def load_prices(filepath: str) -> np.ndarray:
    lower = filepath.lower()
    if lower.endswith((".xlsx", ".xls")):
        sheet = 0 if EXCEL_SHEET is None else int(EXCEL_SHEET) if str(EXCEL_SHEET).isdigit() else EXCEL_SHEET
        df = pd.read_excel(filepath, sheet_name=sheet)
        cols_lower = {c.lower(): c for c in df.columns}
        if DATA_END_DATE and DATE_COLUMN.lower() in cols_lower:
            dcol = cols_lower[DATE_COLUMN.lower()]
            df = df[pd.to_datetime(df[dcol]) <= pd.Timestamp(DATA_END_DATE)].reset_index(drop=True)
        key = EXCEL_FIELD.lower()
        if key not in cols_lower:
            raise KeyError(f"Column '{EXCEL_FIELD}' not found in {filepath}. Available: {list(df.columns)}")
        s = pd.to_numeric(df[cols_lower[key]], errors="coerce").dropna()
        return s.to_numpy(dtype=np.float64)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read().strip()
    vals = [float(v) for v in (text.split(",") if "," in text else text.splitlines()) if v.strip()]
    return np.array(vals, dtype=np.float64)


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


def save_gaf_jpg(gaf: np.ndarray, path: str) -> None:
    arr = gaf_to_uint8(gaf)
    im = Image.fromarray(arr).convert("RGB")
    if im.size != (IMG_SIZE, IMG_SIZE):
        im = im.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    im.save(path, quality=92)


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
        return self.net(self.fc(z).view(-1, 512, _H, _H))


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
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close(fig)


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
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'confusion_matrix.jpg'), dpi=150, bbox_inches='tight'); plt.close(fig)

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
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'roc_curve.jpg'), dpi=150, bbox_inches='tight'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[y_true == 0], bins=25, alpha=0.7, label='Normal label=0')
    ax.hist(scores[y_true == 1], bins=25, alpha=0.7, label='Crash label=1')
    ax.set_title('Anomaly Score Distribution')
    ax.set_xlabel('Anomaly score')
    ax.set_ylabel('Count')
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'score_histogram.jpg'), dpi=150, bbox_inches='tight'); plt.close(fig)


def export_sample_gafs(gafs: np.ndarray, labels: np.ndarray, starts: np.ndarray, out_dir: str):
    if not EXPORT_SAMPLE_GAF:
        return
    ensure_dir(out_dir)
    saved = 0
    for target_label in [0, 1]:
        idx = np.where(labels == target_label)[0][:SAMPLE_EXPORT_COUNT]
        for k in idx:
            save_gaf_jpg(gafs[k], os.path.join(out_dir, f"sample_t{int(starts[k]):05d}_y{int(labels[k])}.jpg"))
            saved += 1
    print(f"    [EXPORT] sample GAF images saved: {saved}", flush=True)


def run_one_method(stock_name: str, prices: np.ndarray, method_name: str, gaf_fn: Callable[[np.ndarray, int], np.ndarray], stock_out: str) -> EvalResult:
    print(f"\n[RUN] {stock_name} | method={method_name}", flush=True)
    method_dir = os.path.join(stock_out, method_name)
    ensure_dir(method_dir)
    ensure_dir(os.path.join(method_dir, 'plots'))
    ensure_dir(os.path.join(method_dir, 'samples'))

    all_gafs, all_labels, all_starts = build_windows(prices, gaf_fn)
    n_windows = len(all_gafs)
    test_start_idx = max(0, len(prices) - FORECAST_DAYS - WINDOW_SIZE + 1)
    train_mask = all_starts < test_start_idx
    test_mask = all_starts >= test_start_idx

    # train uses only normal windows from the training period
    train_normal_mask = train_mask & (all_labels == 0)
    x_train = all_gafs[train_normal_mask]
    y_test = all_labels[test_mask]
    x_test = all_gafs[test_mask]
    starts_test = all_starts[test_mask]

    if len(x_train) < max(BATCH_SIZE, 8):
        raise RuntimeError(f"Not enough normal training windows for {stock_name}/{method_name}.")
    if len(x_test) == 0:
        raise RuntimeError(f"No test windows for {stock_name}/{method_name}.")

    export_sample_gafs(all_gafs[test_mask], y_test, starts_test, os.path.join(method_dir, 'samples'))

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
        'window_start': starts_test,
        'label_0normal_1crash': y_test,
        'anomaly_score': test_scores,
        'predicted_label': y_pred,
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
    return result


def run_one_stock(data_file: str, stock_name: str) -> List[EvalResult]:
    prices = load_prices(data_file)
    stock_out = os.path.join(stock_name, 'out_put_four_gaf_fanogan')
    ensure_dir(stock_out)
    print(f"\n{'='*90}\n[STOCK] {stock_name} | file={data_file} | n_prices={len(prices)}\n{'='*90}", flush=True)
    results = []
    for method_name, gaf_fn in GAF_METHODS.items():
        t0 = time.time()
        result = run_one_method(stock_name, prices, method_name, gaf_fn, stock_out)
        print(f"    [TIME] {method_name} took {(time.time()-t0)/60:.2f} min", flush=True)
        results.append(result)
        torch.cuda.empty_cache()

    summary = pd.DataFrame([asdict(r) for r in results]).sort_values(by=['f1', 'auc', 'accuracy'], ascending=False)
    summary.to_csv(os.path.join(stock_out, 'comparison_summary.csv'), index=False)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    methods = summary['method'].tolist()
    ax.bar(methods, summary['f1'].to_numpy(), label='F1')
    ax.plot(methods, summary['auc'].to_numpy(), marker='o', label='AUC')
    ax.plot(methods, summary['accuracy'].to_numpy(), marker='s', label='Accuracy')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'{stock_name}: four GAF mappings comparison')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(stock_out, 'comparison_chart.jpg'), dpi=150, bbox_inches='tight'); plt.close(fig)
    return results


def run_all_stocks_in_excel_dir() -> None:
    if not os.path.isdir(EXCEL_DIR):
        raise FileNotFoundError(f"Excel directory not found: ./{EXCEL_DIR}")
    files = sorted([fn for fn in os.listdir(EXCEL_DIR) if fn.lower().endswith(BATCH_SUFFIX.lower()) and not fn.startswith('~$')])
    if not files:
        raise FileNotFoundError(f"No '*{BATCH_SUFFIX}' files found under ./{EXCEL_DIR}")
    all_results = []
    for i, fn in enumerate(files, 1):
        data_file = os.path.join(EXCEL_DIR, fn)
        stock_name = fn[:-len(BATCH_SUFFIX)].strip().strip('_')
        print(f"\n{'#'*100}\n[BATCH] ({i}/{len(files)}) stock={stock_name}\n{'#'*100}", flush=True)
        all_results.extend(run_one_stock(data_file, stock_name))
    pd.DataFrame([asdict(r) for r in all_results]).to_csv('all_stocks_four_gaf_fanogan_summary.csv', index=False)
    print("\n[DONE] Saved all_stocks_four_gaf_fanogan_summary.csv", flush=True)


def main():
    run_all_stocks_in_excel_dir()


if __name__ == '__main__':
    main()
