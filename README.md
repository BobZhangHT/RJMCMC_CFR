# RJMCMC-CFR: Bayesian Changepoint Model for Time‑Varying CFR

A research codebase for detecting changepoints in a time‑varying case fatality rate (CFR) using a Reversible‑Jump MCMC (RJMCMC) sampler with benchmarks (rtaCFR+PELT, rtaCFR+BinSeg). The repository includes simulation and real‑data notebooks as well as analysis scripts.

## Repository structure

```
.
├── analysis.py                # Loads results, computes metrics, and generates publication figures/tables
├── config.py                  # Centralized configuration (paths, priors, grids, scenario definitions)
├── data_generation.py         # Simulation DGP: cases/deaths with delay convolution
├── methods.py                 # RJMCMC sampler + rtaCFR fused‑lasso + PELT/BinSeg wrappers
├── evaluation_realdata.py     # Real‑data evaluation helpers
├── Realdata_Analysis_JP.ipynb # Real‑data pipeline (Japan)
├── Simulation_Analysis.ipynb  # Simulation pipeline
├── JP_Data.csv                # Real‑data CSV (Japan) used by the notebook(s)
└── (auto‑created at runtime)
    ├── results_cache/         # Cached rtaCFR signals
    ├── results_sensitivity/   # Sensitivity study outputs
    ├── results_main/          # Main simulation outputs
    └── plots/                 # Figures exported by analysis.py
```

> The notebooks call into the Python modules above. Path and hyperparameter knobs live in `config.py`.

## Installation (lightweight)

Tested with Python ≥ 3.9. Create and activate a fresh environment, then install requirements:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
# .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Note on solvers:** `cvxpy` defaults to ECOS for this project. The wheel for `ecos` is included in `requirements.txt`.

**Optional speed‑ups:** If you plan to run long simulations, consider enabling the Numba JIT decoration for the inner sampler in `methods.py` (comment explains how).

## Quickstart

### 1) Simulations (end‑to‑end)

The notebook `Simulation_Analysis.ipynb` runs:

1. **Sensitivity analysis** over prior hyperparameters for RJMCMC (writes to `results_sensitivity/`).

2. **Main experiments** with the optimal hyperparameters (writes to `results_main/`).

3. **Figures & tables** via `analysis.py` (written to `plots/`).

Minimal path:

```bash
# from repo root
python - <<'PY'
from analysis import main_analysis_workflow
main_analysis_workflow()
PY
```

Or open the notebook and run all cells.

Key knobs live in `config.py`:

* `SENSITIVITY_GRID` for prior sweeps

* `SCENARIOS` for DGPs

* `MCMC_ITER`, `MCMC_BURN_IN`, `K_MAX` for the RJMCMC loop

* Output directories (`results_*`, `plots/`)

### 2) Real‑data (Japan)

Open `Realdata_Analysis_JP.ipynb` and run all cells. The notebook expects:

* `JP_Data.csv` (cases/deaths; paths handled in the notebook)

* The model components from `methods.py`

Outputs:

* Posterior summaries/figures to `plots/`

* Optional cached signals to `results_cache/`

## Reproducing figures & tables

`analysis.py` provides the batch workflow used for the paper‑style figures:

```bash
python analysis.py
```

This script will:

* Load summaries from `results_sensitivity/` and derive the baseline priors

* Load `results_main/`, aggregate metrics, and export:

  * `plots/publication_figure.pdf`

  * `plots/sensitivity_analysis_heatmap.pdf`

  * `plots/publication_summary_table.csv`

  * `plots/sensitivity_analysis_table.csv`

If a directory is missing, the script will warn and exit cleanly.

## Notes & tips

* **Determinism:** A global `SEED` is defined in `config.py` and is offset by replication indices where appropriate.

* **Caching:** rtaCFR signals are cached per scenario/rep in `results_cache/` to avoid recomputation.

* **Numba:** The JIT‑accelerated helpers are in `methods.py`. The main RJMCMC loop can be toggled to JIT if you are not using multiprocessing.

* **Memory use:** `analysis.py` loads only summaries (not full posterior arrays) for aggregation to keep memory usage modest.

## Citation

If you use this repository or the ideas herein, please cite the accompanying manuscript:

> *A Bayesian Changepoint Model for Time‑varying Case Fatality Rate via RJMCMC*, Zhang, Lee, and Qu (2025).

A BibTeX entry will be provided in the project README once the preprint is public.

## License

Specify your license (e.g., MIT, Apache‑2.0) at the repository root.
