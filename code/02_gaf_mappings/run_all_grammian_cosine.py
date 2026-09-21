from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
# ------------------------------------------------------------------------------------------
INPUT_DIR = REPO_INPUT / "gaf_series"
OUTPUT_DIR = REPO_OUTPUT / "run_all_grammian_cosine"
FIELD_NAME = "cp"
WINDOW_SIZE = 800
STEP_SIZE = 1
FILE_PATTERN = "*_final2.xlsx"
SUMMARY_CSV = OUTPUT_DIR / "summary_windows.csv"


def min_max_scaling(ts: np.ndarray) -> np.ndarray:
    min_val = np.min(ts)
    max_val = np.max(ts)
    if max_val == min_val:
        return np.zeros_like(ts, dtype=float)
    return (ts - min_val) / (max_val - min_val)


def sliding_windows(ts: np.ndarray, window_size: int, step_size: int) -> list[np.ndarray]:
    if len(ts) < window_size:
        return []

    num_segments = ((len(ts) - window_size) // step_size) + 1
    segments: list[np.ndarray] = []

    for i in range(num_segments):
        start_index = i * step_size
        segments.append(ts[start_index:start_index + window_size])

    return segments


def generate_gaf_cosine(ts: np.ndarray) -> np.ndarray:
    scaled_01 = min_max_scaling(ts)
    scaled = np.clip(2.0 * scaled_01 - 1.0, -1.0, 1.0)
    phi = np.arccos(scaled)
    return np.cos(phi[:, None] + phi[None, :])


def read_series(file_path: Path, field_name: str) -> np.ndarray:
    df = pd.read_excel(file_path)

    if field_name not in df.columns:
        raise ValueError(
            f'Column "{field_name}" not found in "{file_path.name}". '
            f'Available columns: {", ".join(df.columns)}'
        )

    ts = pd.to_numeric(df[field_name], errors="coerce").to_numpy(dtype=float)
    ts = ts[np.isfinite(ts)]

    if ts.size == 0:
        raise ValueError(f'No valid numeric data found in column "{field_name}".')

    return ts


def save_gaf_image(gaf: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, gaf, cmap="jet", vmin=-1.0, vmax=1.0, format="jpg")


def show_preview(stock_name: str, preview_image: Path | None) -> None:
    if preview_image is None:
        return

    fig, ax = plt.subplots(1, 1, figsize=(4.4, 4.0))
    image = plt.imread(preview_image)
    ax.imshow(image)
    ax.axis("off")
    ax.set_title(preview_image.stem, fontsize=9)
    fig.suptitle(f"{stock_name}: generated image preview", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()


def process_stock(input_file: Path) -> dict[str, object] | None:
    stock_name = input_file.stem.removesuffix("_final2")
    ts = read_series(input_file, FIELD_NAME)
    segments = sliding_windows(ts, WINDOW_SIZE, STEP_SIZE)

    if not segments:
        print(f"Skipped {stock_name}: time series shorter than window size {WINDOW_SIZE}")
        return None

    stock_output_dir = OUTPUT_DIR / stock_name
    saved_images: list[Path] = []

    for i, segment in enumerate(segments, start=1):
        gaf = generate_gaf_cosine(segment)
        output_file = stock_output_dir / f"image_size{WINDOW_SIZE}_{i}.jpg"
        save_gaf_image(gaf, output_file)
        saved_images.append(output_file)

    preview_image = saved_images[-1]

    print(
        f"Completed {stock_name}: field={FIELD_NAME}, "
        f"windows={len(saved_images)}, output={stock_output_dir}"
    )
    show_preview(stock_name, preview_image)
    return {
        "stock_name": stock_name,
        "input_file": input_file.name,
        "field": FIELD_NAME,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "num_images": len(saved_images),
        "output_dir": str(stock_output_dir),
        "last_image": str(preview_image),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = sorted(INPUT_DIR.glob(FILE_PATTERN))

    if not input_files:
        print(f'No files matching "{FILE_PATTERN}" found in "{INPUT_DIR}".')
        return

    processed_count = 0
    summary_rows: list[dict[str, object]] = []

    for input_file in input_files:
        try:
            result = process_stock(input_file)
            if result is not None:
                processed_count += 1
                summary_rows.append(result)
        except Exception as exc:
            print(f"Skipped {input_file.name}: {exc}")

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig")
        print(f"Saved summary CSV -> {SUMMARY_CSV}")

    print(f"Finished processing {processed_count} stock files using field '{FIELD_NAME}'.")


if __name__ == "__main__":
    main()
