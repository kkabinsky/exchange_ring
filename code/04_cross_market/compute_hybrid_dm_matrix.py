import math
from pathlib import Path

import numpy as np
import pandas as pd


def rank_prob(x):
    s = pd.Series(np.asarray(x, float))
    r = s.rank(method="average").to_numpy()
    n = len(r)
    return r / (n + 1.0)


def dm_stats(d):
    d = np.asarray(d, float)
    t = len(d)
    dc = d - d.mean()
    g0 = float((dc @ dc) / t)
    var_mean = g0 / t
    if var_mean <= 0:
        return float("nan"), float("nan"), float("nan")
    stat = float(d.mean() / math.sqrt(var_mean))
    phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    p_two = 2 * (1 - phi(abs(stat)))
    p_one_less = 1 - phi(stat)
    return stat, p_two, p_one_less


def main():
    root = Path(__file__).resolve().parents[2] / "output" / "04_cross_market" / "hybrid"  # replication package
    folder = "USOIL_daily"
    mappings = ["cosine", "arctan", "arccosh", "exponential"]

    frames = []
    for m in mappings:
        p = root / folder / "stage3_fanogan" / m / "backtest_detail.csv"
        df = pd.read_csv(p)
        df = df[["window_start", "fanogan_score", "true_label_user_0crash_1not"]].copy()
        df = df.rename(columns={"fanogan_score": f"score_{m}"})
        frames.append(df)

    base = frames[0]
    for df in frames[1:]:
        base = base.merge(df, on=["window_start", "true_label_user_0crash_1not"], how="inner")

    t = len(base)
    y = 1 - base["true_label_user_0crash_1not"].astype(int).to_numpy()

    losses = {}
    mean_loss = {}
    for m in mappings:
        p = rank_prob(base[f"score_{m}"].to_numpy())
        l = (p - y) ** 2
        losses[m] = l
        mean_loss[m] = float(l.mean())

    print(f"T={t}")
    print("y_counts:", {0: int((y == 0).sum()), 1: int((y == 1).sum())})
    print("mean Brier loss per mapping:")
    for m in mappings:
        print(f"  {m}: {mean_loss[m]:.4f}")

    print("\nExp vs others (d = L_other - L_exp, positive => exp better):")
    for other in mappings:
        if other == "exponential":
            continue
        d = losses[other] - losses["exponential"]
        stat, p_two, p_one = dm_stats(d)
        print(f"  {other} vs exponential: mean_d={d.mean():.4f}, DM={stat:.4f}, p_two={p_two:.4f}, p_one_exp_better={p_one:.4f}")


if __name__ == "__main__":
    main()
