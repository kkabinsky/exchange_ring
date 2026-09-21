from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one f-AnoGAN GAF comparison job.")
    parser.add_argument("--input", required=True, help="Input Excel/CSV path.")
    parser.add_argument("--stock-name", required=True, help="Output stock/run folder name.")
    args = parser.parse_args()

    # replication package: results go to <repo>/output/04_cross_market, shared module in <repo>/code/common
    import sys
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo / "code" / "common"))
    base = repo / "output" / "04_cross_market"
    base.mkdir(parents=True, exist_ok=True)
    os.chdir(base)

    import fanogan_four_gaf_compare_2 as fanogan

    fanogan.run_one_stock(str(Path(args.input).resolve()), args.stock_name)


if __name__ == "__main__":
    main()
