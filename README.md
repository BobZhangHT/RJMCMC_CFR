# RJMCMC-CFR: Bayesian Changepoint Detection for Time-Varying Case Fatality Rate

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

This repository implements a Bayesian changepoint model for detecting temporal changes in disease case fatality rate (CFR) using Reversible-Jump Markov Chain Monte Carlo (RJMCMC). The method is benchmarked against PELT and Binary Segmentation applied to real-time adjusted CFR (rtaCFR) signals.

**Key Features:**
- Flexible Bayesian framework for unknown number of changepoints
- Accounts for reporting delays between cases and deaths
- Comprehensive simulation study and real-data validation
- Reproducible analysis pipeline

## Installation

**Requirements:** Python ≥ 3.9

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Repository Structure

```
├── methods.py                 # RJMCMC sampler and comparison methods
├── data_generation.py         # Simulation data generation
├── analysis.py                # Analysis and visualization
├── config.py                  # Hyperparameters and settings
├── Simulation_Analysis.ipynb  # Main simulation experiments
├── Realdata_Analysis_JP.ipynb # Real-data analysis (Japan COVID-19)
├── JP_Data.csv                # Japan epidemic data
├── results/                   # Simulation outputs
└── plots/                     # Generated figures and tables
```

## Reproducing Results

### Simulation Study

Run the complete simulation pipeline:

```bash
# Option 1: Via Python script
python -c "from analysis import full_analysis_workflow; full_analysis_workflow()"

# Option 2: Via Jupyter notebook
jupyter notebook Simulation_Analysis.ipynb
```

This generates:
- Sensitivity analysis over prior specifications
- Performance comparison across scenarios
- Publication-ready figures: `plots/publication_figure.pdf`
- Results table: `plots/main_results_table.tex`

### Real-Data Analysis

Apply the method to Japan COVID-19 data:

```bash
jupyter notebook Realdata_Analysis_JP.ipynb
```

Outputs:
- `plots/japan_cfr_comparison.pdf` - Method comparison
- `plots/real_data_evaluation_summary.tex` - Performance metrics

## Configuration

Key parameters in `config.py`:
- `MCMC_ITER`, `MCMC_BURN_IN`: MCMC sampling settings
- `K_MAX`: Maximum number of changepoints
- `SENSITIVITY_GRID`: Prior hyperparameter ranges
- `SCENARIOS`: Simulation scenario definitions

## Citation

If you use this code, please cite:

```bibtex
@article{zhang2025rjmcmc,
  title={A Bayesian Changepoint Model for Time-varying Case Fatality Rate via RJMCMC},
  author={Zhang, Hengtao and Lee, Chun Yin and Qu, Yuanke},
  journal={[Journal Name]},
  year={2025},
  note={Manuscript in preparation}
}
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
