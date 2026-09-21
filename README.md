# Exchanged Flying Rings in Gramian Angular Fields with Exponential Mapping during Market Crashes

Replication package: code, input data and saved results.
Citation details will be added when the paper is published.

## Quick check (no GPU, a few seconds)

```bash
pip install -r requirements.txt
python code/reproduce_tables.py
```

This prints Tables 5–10 and Supplementary Table S3 from the saved results in
`output/`. Nothing is retrained. The printed values are also in
`output/reproduce_tables_output.txt`.

## Folder layout

```text
code/                  programs, one folder per part of the paper
  01_theory_ns_var/    Sections 3.5, 3.5.1  Nelson-Siegel nilpotent shock, VAR simulation
  02_gaf_mappings/     Sections 3.6, 5.1    GAF images with four angular mappings
  03_bond_anomaly/     Section 5.2          Treasury yields: CNN autoencoder and f-AnoGAN
  04_cross_market/     Section 5.3          equity indices, hybrid TadGAN-GAF-f-AnoGAN
  05_leadtime_dm/      Section 5.4          lead time, Diebold-Mariano test
  common/              f-AnoGAN module shared by 03 and 04
  reproduce_tables.py  prints the tables from output/
input/                 all input data (read only)
output/                saved results, one folder per part; the programs write here
```

Every program finds `input/` and `output/` by itself, so it can be run from any folder.

## Where each table and figure comes from

Table and figure numbers refer to the revised manuscript.

| Paper | Program | Output |
|---|---|---|
| Table 1, Fig. 5 | `code/01_theory_ns_var/ns_nilpotent_inversion.py` | `output/01_theory_ns_var/ns_nilpotent_inversion.csv`, `.jpg` |
| Table 2, Fig. 6 | `code/01_theory_ns_var/ns_nilpotent_inversion.py` | `output/01_theory_ns_var/ns_inversion_four_params.jpg` |
| Tables 3–4, eq. (64) | `code/01_theory_ns_var/var_3m_shock_levels.py` | `output/01_theory_ns_var/var_3m_shock_levels.csv`, `var_3m_shock_inversion.csv` |
| supporting (VAR impulse responses) | `code/01_theory_ns_var/irf_var_simulation.py` | `output/01_theory_ns_var/irf_*.csv`, `.jpg` |
| supporting (observed 30Y − 1M spread) | `code/01_theory_ns_var/empirical_invertible_yield.py` | `output/01_theory_ns_var/empirical_*.csv`, `.jpg` |
| eq. (80), four mappings | `code/02_gaf_mappings/run_all_grammian*.py` | `output/02_gaf_mappings/<program>/` (window images) |
| Fig. 11, Supplementary Figs. S8–S9 | `code/02_gaf_mappings/full_length_gaf_figures.py` | `output/02_gaf_mappings/full_gaf_yields.jpg`, `full_gaf_equities.jpg`, `full_gaf_assets.jpg` |
| polar plot of the four mappings | `code/02_gaf_mappings/plot_polar_gaf_final.py` | `output/02_gaf_mappings/gaf_four_mappings.jpg` |
| Tables 8–9 | `code/03_bond_anomaly/run_bond_5maturities_compare.py` | `output/03_bond_anomaly/bond_3m_1y_5y_10y_20y_compare_e20_daily/` |
| Supplementary Table S3 (US30Y) | `code/03_bond_anomaly/run_bond_1m_30y_compare.py` | `output/03_bond_anomaly/bond_1m_30y_compare_e20_daily/` |
| US30Y at 100 epochs | `code/03_bond_anomaly/rerun_us30y.py` | `output/03_bond_anomaly/bond_30y_rerun_e100/` |
| benchmark: f-AnoGAN (exp) vs CNN autoencoder, DM test | `code/03_bond_anomaly/benchmark_cnn_vs_fanogan_dm.py` | `output/03_bond_anomaly/benchmarks/dm_fanogan_exp_vs_cnn_ae.csv` |
| benchmark: 10Y−3M spread (= probit on the spread) | `code/03_bond_anomaly/benchmark_spread_probit.py` | `output/03_bond_anomaly/benchmarks/spread_benchmark_*.csv` |
| Tables 5–6, eq. (81) | `code/04_cross_market/run_market_crash_3datasets_oneclick.py` | `output/04_cross_market/<index>_<crash>/` |
| Table 7 | `code/04_cross_market/hybrid_gaf_tadgan_fanogan.py`, `compute_hybrid_dm_matrix.py` | `output/04_cross_market/hybrid/USOIL_daily/` |
| TadGAN anomaly scores per asset | `code/04_cross_market/run_tadgan_assets.py` | `output/04_cross_market/tadgan/<asset>/` |
| DM tests corrected for overlapping windows (HAC/HLN) and multiple testing (Benjamini–Hochberg), Tables 5, 7, 8, 9 and benchmarks | `code/05_leadtime_dm/dm_hac_hln_bh.py` | `output/05_leadtime_dm/dm_corrected/dm_hac_hln_bh.csv` |
| Table 10, eq. (82) | `code/05_leadtime_dm/lead_time_economic.py`, `fill_tables.py`, `lead_all_methods.py` | `output/05_leadtime_dm/` |

## Running each part

Parts 01, 02 and 05 run on a normal CPU in seconds to minutes. Parts 03 and 04
train neural networks (f-AnoGAN, CNN autoencoder, TadGAN); a GPU is recommended,
and a full run can take many hours on a CPU.

```bash
# 01  theory and VAR simulation
python code/01_theory_ns_var/ns_nilpotent_inversion.py
python code/01_theory_ns_var/var_3m_shock_levels.py
python code/01_theory_ns_var/irf_var_simulation.py
python code/01_theory_ns_var/empirical_invertible_yield.py

# 02  GAF images
python code/02_gaf_mappings/full_length_gaf_figures.py
python code/02_gaf_mappings/plot_polar_gaf_final.py            # optional: path to another series
python code/02_gaf_mappings/run_all_grammian_exponential.py     # also _cosine, _arctan, _arccosh, run_all_grammian.py, _full

# 03  Treasury yields (stages: prepare, cnn, fanogan, hybrid, summarize, all)
python code/03_bond_anomaly/run_bond_5maturities_compare.py --stage summarize   # tables from saved scores
python code/03_bond_anomaly/run_bond_5maturities_compare.py --stage all --force # retrain
python code/03_bond_anomaly/run_bond_1m_30y_compare.py --stage all --force
python code/03_bond_anomaly/rerun_us30y.py

# 04  equity indices, hybrid pipeline, TadGAN
python code/04_cross_market/run_market_crash_3datasets_oneclick.py --summary-only  # reports from saved scores
python code/04_cross_market/run_market_crash_3datasets_oneclick.py --force         # retrain the nine cells
python code/04_cross_market/hybrid_gaf_tadgan_fanogan.py         # reads input/benchmark_assets
python code/04_cross_market/compute_hybrid_dm_matrix.py
python code/04_cross_market/run_tadgan_assets.py

# 05  lead time and DM tests (from the saved scores)
python code/05_leadtime_dm/fill_tables.py
python code/05_leadtime_dm/lead_all_methods.py
python code/05_leadtime_dm/lead_time_economic.py
```

The runners in 03 and 04 skip a job when its result files already exist; add
`--force` to retrain it. The TadGAN step reuses the saved TadGAN scores of the
hybrid pipeline (`stage1_tadgan/tadgan_results.csv`) unless `TADGAN_FORCE_TRAIN=1`.

## Input data

| File | Content |
|---|---|
| `input/combine_TTM.xlsx` | U.S. Treasury zero-coupon yields, 1M–30Y, sheet `Ycurve` |
| `input/u30_1day.xlsx`, `u500_1day.xlsx`, `dax_1day.xlsx` | daily closes of the DJIA, S&P 500 and DAX |
| `input/USOIL_daily.xlsx`, `GOLD_daily.xlsx`, `EURUSD_daily.xlsx` | daily WTI crude oil, gold and EUR/USD |
| `input/benchmark_assets/*_final2.xlsx` | the same three assets in the format read by the hybrid pipeline |
| `input/gaf_series/*_final2.xlsx` | series converted to GAF window images in part 02 |
| `input/polar/cpall_final2.xlsx` | series used for the saved polar plot |
| `input/bond_prepared/US1M_COVID200_daily_final2.xlsx` | input of the `us1m` preset of `cnn_gaf_autoencoder_benchmark.py` |
| `input/tadgan_assets/*.xlsx` | inputs of the TadGAN step, one file per asset |
| `input/dm_test/input_dm_test.xlsx` | template read by `dm_test.py` |
| `input/iran_new_run_scores/` | saved scores for the programs in `code/05_leadtime_dm/sensitivity_iran_run/` |

The prepared Treasury inputs used for Tables 8–9 and S3 are in
`output/03_bond_anomaly/<run>/prepared_inputs/`; `--stage prepare` writes them again
from `input/combine_TTM.xlsx`.

## Notes

- **Not included:** trained model weights (`*.pt`) and the per-window GAF images.
  The programs create them again when they run.
- **Table 10** is computed from `output/05_leadtime_dm/saved/lead_time_economic_summary.csv`.
  Running `lead_time_economic.py` on the saved scores writes
  `lead_time_economic_summary_recomputed.csv`; it has one more cell (DAX, Chinese
  real-estate), and six U30/DAX COVID-19 values differ from the saved file.
- **Mean rank:** `fill_tables.py` prints the mean rank with 4 = best.
- **`sensitivity_iran_run/`:** moving-block bootstrap for overlapping windows,
  threshold sweep (0.90, 0.95, 0.99), Brier decomposition and lead time at matched
  false-alarm rates. These four programs use a separate run with 13 detectors on
  USOIL, GOLD and EUR/USD. The saved bootstrap file was made with
  `python code/05_leadtime_dm/sensitivity_iran_run/overlap_corrected_tests.py --replicates 400`.
- **`dm_test.py`** contains the Diebold–Mariano test with the Harvey–Leybourne–Newbold
  correction. Its `main()` needs a filled `dm_test_template` sheet; the sheet in
  `input/dm_test/` has no rows yet.
- **`code/03_bond_anomaly/legacy_original_fanogan/`:** `train_wgangp.py` and
  `train_f_anogan_encoder1.py` need the `fanogan` and `mvtec_ad` modules of the
  original PyTorch f-AnoGAN implementation, which are not included. The results in
  the paper use `code/common/fanogan_four_gaf_compare_2.py`, which contains its own
  WGAN-GP and encoder.
- **Changes to the original programs:** file paths only (inputs from `input/`,
  results to `output/`). In addition, `empirical_invertible_yield.py` got a guard
  for the case with no inversion episode, `dm_test.py` writes its results to
  `output/05_leadtime_dm/dm_test_results.xlsx` instead of into the input file, and
  `lead_time_economic.py` writes to `lead_time_economic_summary_recomputed.csv`.
- Random seeds are fixed in the programs; results of GPU training can still differ
  slightly between machines.
