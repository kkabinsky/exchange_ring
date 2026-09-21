# -*- coding: utf-8 -*-
"""Run the TadGAN anomaly-score step (improved_tadgans_anomaly2.py) for each asset.

improved_tadgans_anomaly2.py reads every *.xlsx file in its working directory and
writes csv_data/, models/ and results/ there. This wrapper gives each asset its own
working directory under output/04_cross_market/tadgan/<asset>/ and runs the script
there, one asset at a time.

    python code/04_cross_market/run_tadgan_assets.py            # all six assets
    python code/04_cross_market/run_tadgan_assets.py oil_1day   # one asset
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().parent / "improved_tadgans_anomaly2.py"
INPUT = REPO / "input" / "tadgan_assets"
OUTPUT = REPO / "output" / "04_cross_market" / "tadgan"


def main() -> None:
    wanted = set(sys.argv[1:])
    files = sorted(INPUT.glob("*.xlsx"))
    if wanted:
        files = [f for f in files if f.stem in wanted]
    if not files:
        sys.exit(f"no input files found in {INPUT}")
    for f in files:
        work = OUTPUT / f.stem
        work.mkdir(parents=True, exist_ok=True)
        local = work / f.name
        shutil.copy2(f, local)          # the TadGAN script reads *.xlsx from its working directory
        print(f"[TADGAN] {f.stem}: working directory {work}", flush=True)
        try:
            subprocess.run([sys.executable, str(SCRIPT)], cwd=str(work), check=True)
        finally:
            local.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
