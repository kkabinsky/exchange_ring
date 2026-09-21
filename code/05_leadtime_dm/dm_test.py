import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DMResult:
    T: int
    h: int
    loss: str
    dm_stat: float
    p_value_two_sided: float
    mean_loss_diff: float
    var_hat: float


def _newey_west_variance(d: np.ndarray, lag: int) -> float:
    """
    Newey–West HAC variance estimate for the sample mean of d_t.
    Returns var_hat(d_bar) = gamma0/T + 2*sum_{k=1..lag} w_k*gamma_k/T
    where gamma_k is sample autocovariance at lag k (using mean-centered d).
    """
    T = d.size
    if T < 2:
        return float("nan")
    dc = d - d.mean()
    gamma0 = float(np.dot(dc, dc) / T)
    var = gamma0
    for k in range(1, lag + 1):
        w = 1.0 - (k / (lag + 1.0))
        gamma_k = float(np.dot(dc[k:], dc[:-k]) / T)
        var += 2.0 * w * gamma_k
    return var / T


def dm_test(
    actual: np.ndarray,
    f1: np.ndarray,
    f2: np.ndarray,
    *,
    h: int = 1,
    loss: str = "mse",
) -> DMResult:
    """
    Diebold–Mariano test comparing two forecasts f1 vs f2 for actual.

    - actual, f1, f2: aligned arrays of length T
    - h: forecast horizon (use h=1 for one-step-ahead). NW lag defaults to h-1.
    - loss: 'mse' or 'mae'
    """
    if loss not in {"mse", "mae"}:
        raise ValueError("loss must be 'mse' or 'mae'")
    a = np.asarray(actual, dtype=float)
    y1 = np.asarray(f1, dtype=float)
    y2 = np.asarray(f2, dtype=float)
    if not (a.shape == y1.shape == y2.shape):
        raise ValueError("actual, f1, f2 must have same shape")

    e1 = a - y1
    e2 = a - y2
    if loss == "mse":
        L1 = e1**2
        L2 = e2**2
    else:
        L1 = np.abs(e1)
        L2 = np.abs(e2)

    d = L1 - L2
    T = d.size
    if T < 5:
        raise ValueError("Need at least 5 observations for DM test")

    lag = max(0, int(h) - 1)
    var_hat = _newey_west_variance(d, lag=lag)
    if not np.isfinite(var_hat) or var_hat <= 0:
        raise ValueError("Non-positive HAC variance; cannot compute DM statistic")

    dm = float(d.mean() / math.sqrt(var_hat))

    # Large-sample: DM ~ N(0,1). Use normal approximation (no scipy dependency).
    # two-sided p = 2*(1-Phi(|dm|))
    p = float(2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(dm) / math.sqrt(2.0)))))

    return DMResult(
        T=T,
        h=int(h),
        loss=loss,
        dm_stat=dm,
        p_value_two_sided=p,
        mean_loss_diff=float(d.mean()),
        var_hat=float(var_hat),
    )


def main() -> None:
    """
    Reads `input_dm_test.xlsx` (sheet dm_test_template) and writes DM results back.

    Expected columns (per row):
      - asset, mapping, horizon_day
      - actual
      - forecast_standalone
      - forecast_hybrid

    Rows are grouped by (asset, mapping) and DM test is computed within each group.
    """
    # replication package: read from <repo>/input/dm_test, write to <repo>/output/05_leadtime_dm
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    xlsx_path = str(repo / "input" / "dm_test" / "input_dm_test.xlsx")
    out_path = repo / "output" / "05_leadtime_dm" / "dm_test_results.xlsx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(xlsx_path, sheet_name="dm_test_template")
    required = {"asset", "mapping", "horizon_day", "actual", "forecast_standalone", "forecast_hybrid"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in dm_test_template: {sorted(missing)}")

    df = df.dropna(subset=["actual", "forecast_standalone", "forecast_hybrid"])
    if df.empty:
        raise SystemExit(
            "dm_test_template has no rows with actual/forecast values. "
            "Fill it first (or add a new sheet with actual + two forecasts)."
        )

    out_rows = []
    for (asset, mapping), g in df.groupby(["asset", "mapping"], dropna=False):
        g = g.sort_values("horizon_day")
        res_mse = dm_test(
            g["actual"].to_numpy(),
            g["forecast_standalone"].to_numpy(),
            g["forecast_hybrid"].to_numpy(),
            h=1,
            loss="mse",
        )
        res_mae = dm_test(
            g["actual"].to_numpy(),
            g["forecast_standalone"].to_numpy(),
            g["forecast_hybrid"].to_numpy(),
            h=1,
            loss="mae",
        )
        out_rows.append(
            {
                "asset": asset,
                "mapping": mapping,
                "T": res_mse.T,
                "dm_stat_mse": res_mse.dm_stat,
                "p_two_sided_mse": res_mse.p_value_two_sided,
                "mean_loss_diff_mse": res_mse.mean_loss_diff,
                "dm_stat_mae": res_mae.dm_stat,
                "p_two_sided_mae": res_mae.p_value_two_sided,
                "mean_loss_diff_mae": res_mae.mean_loss_diff,
            }
        )

    out = pd.DataFrame(out_rows).sort_values(["asset", "mapping"])
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        out.to_excel(w, sheet_name="dm_test_results", index=False)

    print(f"Wrote dm_test_results to {out_path} with {len(out)} groups.")


if __name__ == "__main__":
    main()

