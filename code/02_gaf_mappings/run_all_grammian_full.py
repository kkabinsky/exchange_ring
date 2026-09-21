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
OUTPUT_DIR = REPO_OUTPUT / "run_all_grammian_full"
FIELD_NAME = "cp"
DATE_COLUMN = "date"
START_DATE = "2016-04-04"
END_DATE = "2020-03-01"
MODEL = 3
WINDOW_SIZE = 200
STEP_SIZE = 1
FILE_PATTERN = "*_final2.xlsx"
SUMMARY_CSV = OUTPUT_DIR / "summary_windows.csv"
SHOW_PREVIEW = False


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


def generate_gaf_arctan(ts: np.ndarray) -> np.ndarray:
    scaled_01 = min_max_scaling(ts)
    phi = np.arctan(scaled_01)
    return np.cos(phi[:, None] + phi[None, :])


def generate_gaf_arccosh(ts: np.ndarray) -> np.ndarray:
    scaled_01 = min_max_scaling(ts)
    phi = np.arccosh(1.0 + scaled_01)
    return np.cos(phi[:, None] + phi[None, :])


def generate_gaf_exponential(ts: np.ndarray) -> np.ndarray:
    scaled_01 = min_max_scaling(ts)
    phi = np.pi * (np.exp(scaled_01) - 1.0) / (np.e - 1.0)
    return np.cos(phi[:, None] + phi[None, :])


def get_model_config(model: int) -> tuple[str, callable]:
    model_configs = {
        1: ("cosine", generate_gaf_cosine),
        2: ("arctan", generate_gaf_arctan),
        3: ("arccosh", generate_gaf_arccosh),
        4: ("exponential", generate_gaf_exponential),
    }

    if model not in model_configs:
        raise ValueError("MODEL must be 1 (cosine), 2 (arctan), 3 (arccosh), or 4 (exponential).")

    return model_configs[model]


def prompt_model_selection(default_model: int) -> int:
    print("Select model before running:")
    print("  1 = cosine")
    print("  2 = arctan")
    print("  3 = arccosh")
    print("  4 = exponential")
    prompt = f"Enter model number [default {default_model}]: "

    while True:
        selected = input(prompt).strip()
        if selected == "":
            return default_model

        try:
            model = int(selected)
            get_model_config(model)
            return model
        except (ValueError, TypeError):
            print("Invalid selection. Please enter 1, 2, 3, or 4.")


def filter_by_date(df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
    if DATE_COLUMN is None or (START_DATE is None and END_DATE is None):
        return df

    if DATE_COLUMN not in df.columns:
        raise ValueError(
            f'Date column "{DATE_COLUMN}" not found in "{file_path.name}". '
            f'Available columns: {", ".join(df.columns)}'
        )

    date_series = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    valid_mask = date_series.notna()

    if START_DATE is not None:
        start_date = pd.Timestamp(START_DATE)
        valid_mask &= date_series >= start_date

    if END_DATE is not None:
        end_date = pd.Timestamp(END_DATE)
        valid_mask &= date_series <= end_date

    filtered_df = df.loc[valid_mask].copy()
    if filtered_df.empty:
        raise ValueError(
            f'No rows found in "{file_path.name}" for date range '
            f"{START_DATE or '-inf'} to {END_DATE or '+inf'}."
        )

    return filtered_df


def read_filtered_dataframe(file_path: Path) -> pd.DataFrame:
    df = pd.read_excel(file_path)
    return filter_by_date(df, file_path)


def read_series(file_path: Path, field_name: str) -> np.ndarray:
    df = read_filtered_dataframe(file_path)

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


def describe_dataset(input_file: Path) -> dict[str, object]:
    df = read_filtered_dataframe(input_file)

    if FIELD_NAME not in df.columns:
        raise ValueError(
            f'Column "{FIELD_NAME}" not found in "{input_file.name}". '
            f'Available columns: {", ".join(df.columns)}'
        )

    ts = pd.to_numeric(df[FIELD_NAME], errors="coerce").to_numpy(dtype=float)
    ts = ts[np.isfinite(ts)]
    if ts.size == 0:
        raise ValueError(f'No valid numeric data found in column "{FIELD_NAME}".')

    info: dict[str, object] = {
        "file_name": input_file.name,
        "stock_name": input_file.stem.removesuffix("_final2"),
        "num_rows": int(len(df)),
        "num_values": int(ts.size),
        "start_date": None,
        "end_date": None,
    }

    if DATE_COLUMN is not None and DATE_COLUMN in df.columns:
        date_series = pd.to_datetime(df[DATE_COLUMN], errors="coerce").dropna()
        if not date_series.empty:
            info["start_date"] = date_series.min().strftime("%Y-%m-%d")
            info["end_date"] = date_series.max().strftime("%Y-%m-%d")

    return info


def print_run_configuration(input_files: list[Path], model: int) -> None:
    mapping_name, _ = get_model_config(model)
    print("\nRun configuration")
    print(f"  Model      : {model} ({mapping_name})")
    print(f"  Field      : {FIELD_NAME}")
    print(f"  Window size: {WINDOW_SIZE}")
    print(f"  Step size  : {STEP_SIZE}")
    print(f"  Date range : {START_DATE or 'start'} to {END_DATE or 'end'}")
    print(f"  Input files: {len(input_files)}")
    print("")

    for input_file in input_files:
        try:
            info = describe_dataset(input_file)
            print(
                f"- {info['file_name']}: "
                f"rows={info['num_rows']}, values={info['num_values']}, "
                f"date={info['start_date'] or 'N/A'} to {info['end_date'] or 'N/A'}"
            )
        except Exception as exc:
            print(f"- {input_file.name}: unable to inspect dataset ({exc})")

    print("")


def save_gaf_image(gaf: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, gaf, cmap="jet", vmin=-1.0, vmax=1.0, format="jpg")


def show_preview(stock_name: str, preview_image: Path | None) -> None:
    if not SHOW_PREVIEW:
        return

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
    mapping_name, gaf_generator = get_model_config(MODEL)

    if not segments:
        print(f"Skipped {stock_name}: time series shorter than window size {WINDOW_SIZE}")
        return None

    stock_output_dir = OUTPUT_DIR / stock_name
    saved_images: list[Path] = []

    for i, segment in enumerate(segments, start=1):
        gaf = gaf_generator(segment)
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
        "mapping": mapping_name,
        "date_column": DATE_COLUMN,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "window_size": WINDOW_SIZE,
        "step_size": STEP_SIZE,
        "num_images": len(saved_images),
        "output_dir": str(stock_output_dir),
        "last_image": str(preview_image),
    }


def main() -> None:
    global MODEL

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_files = sorted(INPUT_DIR.glob(FILE_PATTERN))

    if not input_files:
        print(f'No files matching "{FILE_PATTERN}" found in "{INPUT_DIR}".')
        return

    MODEL = prompt_model_selection(MODEL)
    print_run_configuration(input_files, MODEL)

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

    print(
        f"Finished processing {processed_count} stock files "
        f"using field '{FIELD_NAME}' with {get_model_config(MODEL)[0]} mapping."
    )


if __name__ == "__main__":
    main()
