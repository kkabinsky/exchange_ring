# -*- coding: utf-8 -*-
"""Re-run the US30Y Treasury-yield bond cell at 100 epochs (it was trained at 20)
to see whether any GAF mapping attains F1 > 0. Waits for the current f-AnoGAN run
to finish first (no CPU contention), writes to an ISOLATED folder so nothing
existing is overwritten, and prints the new Table-4 values."""
import os, sys, time, csv, subprocess, datetime
from pathlib import Path

# --- replication package (added): data in <repo>/input, results in <repo>/output/03_bond_anomaly ---
import os as _rp_os, sys as _rp_sys
from pathlib import Path as _RpPath
REPO_ROOT = _RpPath(__file__).resolve().parents[2]
REPO_INPUT = REPO_ROOT / "input"
REPO_OUTPUT = REPO_ROOT / "output" / "03_bond_anomaly"
REPO_OUTPUT.mkdir(parents=True, exist_ok=True)
_rp_sys.path.insert(0, str(REPO_ROOT / "code" / "common"))
_rp_os.chdir(REPO_OUTPUT)  # original script writes into the working directory
# ------------------------------------------------------------------------------------------
MC = Path(__file__).resolve().parent   # code folder (imports run_bond_1m_30y_compare)
sys.path.insert(0, str(MC))
LOG = REPO_OUTPUT / "rerun_us30y.log"


def log(m):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def busy():
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { ($_.CommandLine -match 'run_single_fanogan|run_red_cells') "
             "-and $_.ProcessId -ne %d } | Measure-Object).Count" % os.getpid()],
            text=True, stderr=subprocess.DEVNULL).strip()
        return out not in ("", "0")
    except Exception:
        return False


def main():
    open(LOG, "w", encoding="utf-8").close()
    log("START rerun_us30y -> wait for current run, then US30Y bond @ 100 epochs (was 20)")
    while busy():
        log("current f-AnoGAN run still active -> waiting 60s")
        time.sleep(60)
    log("current run finished. Re-running US30Y at 100 epochs ...")

    import run_bond_1m_30y_compare as bond
    bond.DATASETS = [d for d in bond.DATASETS if d["dataset"] == "US30Y"]  # US30Y only
    bond.EPOCHS_SMALL = 100                                               # 20 -> 100
    bond.OUT_ROOT = REPO_OUTPUT / "bond_30y_rerun_e100"                            # isolated output
    bond.FANOGAN_ROOT = bond.OUT_ROOT / "standalone_fanogan"
    bond.run_fanogan(force=True)

    p = bond.FANOGAN_ROOT / "US30Y_BOND_COVID200_daily" / "out_put_four_gaf_fanogan" / "comparison_summary.csv"
    log(f"reading {p}")
    rows = {}
    with open(p, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            auc = r["auc"]
            rows[r["method"]] = {"f1": float(r["f1"]),
                                 "auc": float(auc) if auc not in ("", None) else None}
    order = ["cosine", "arctan", "arccosh", "exponential"]
    log("US30Y @100ep: " + " | ".join(f"{m}: F1={rows[m]['f1']:.3f} AUC={rows[m]['auc']}" for m in order))
    f1max = max(rows[m]["f1"] for m in order)
    f1best = "none" if f1max < 1e-9 else max(order, key=lambda m: rows[m]["f1"])
    log(f"=> Table 4 US30Y: exp_AUC={rows['exponential']['auc']:.3f}  F1-best={f1best}  (max F1={f1max:.3f})")
    log("DONE rerun_us30y.")


if __name__ == "__main__":
    main()
