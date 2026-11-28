# RJMCMC-CFR

RJMCMC-CFR uses reversible jump Markov chain Monte Carlo to estimate when the
case fatality rate (CFR) among reported cases changes. The model links daily
case and death counts through a case-to-death delay distribution and estimates
a piecewise-constant CFR. This repository implements the method and reproduces
the paper's simulation study and Japan COVID-19 analysis.

## Install and fit your data

Use Python 3.12 (3.12.10 for the paper's reference runs).

```bash
git clone https://github.com/BobZhangHT/RJMCMC_CFR.git
cd RJMCMC_CFR
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on macOS/Linux or
`.\.venv\Scripts\Activate.ps1` in PowerShell. Then install the pinned dependencies.

```bash
python -m pip install -r requirements.txt
```

On managed Windows systems, use a Python environment approved to load the
dependencies' native extensions.

Prepare a CSV with one row per day and columns `date`, `ct` (new reported cases)
and `dt` (new reported deaths). Use consecutive dates and nonnegative counts.
The example uses the included [Japan data](JP_Data.csv) and the study's
delay distribution. Replace the CSV path to use your own series. The
`numba-fast` backend runs without a C compiler.

```python
import pandas as pd
from methods import run_rjmcmc

daily = pd.read_csv("JP_Data.csv", parse_dates=["date"]).sort_values("date")
fit = run_rjmcmc(
    {"cases": daily["ct"].to_numpy(), "deaths": daily["dt"].to_numpy()},
    backend="numba-fast",
    seed=2025,
)
daily["cfr"] = fit["p_t_hat"]
print(daily["date"].iloc[fit["taus_est"]].dt.strftime("%Y-%m-%d").tolist())
```

For a new dataset, pass a fitted case-to-death distribution in days via
`delay_dist` (a SciPy distribution with a `.cdf()` method). Check how plausible
delay choices affect changepoint locations, and assess mixing with multiple
chains, rank-normalized R-hat and effective sample size. Interpret shifts in
light of changes in testing, case definitions and reporting.

## Reproduce the paper

Run these commands from the repository root. Jobs save checkpoints under
`results/` and write figures and tables under `plots/` by default. Use
`--no-resume` to rerun completed jobs.

### Simulation study

```bash
python Simulation_Analysis.py --stage all --workers 8 --backend auto
python Simulation_Analysis.py --stage benchmark-delay --benchmark-delay-reps 100 --workers 8
```

For the R2 convergence results and Figure 3, build the C extension with a
working C compiler. On managed systems, the built extension may require
approval to load. C and Numba use different random streams.

```bash
python setup.py build_ext --inplace
python Simulation_Analysis.py --stage diagnostics --backend c --workers 4 --results-root results/r2 --plots-dir plots/r2
```

### Japan COVID-19 application

```bash
python Realdata_Analysis_JP.py --workers 8 --backend auto
python Realdata_Analysis_JP.py --reviewer-only --workers 4 --backend numba-fast --pg-reference reference_r2.json --results-root results/r2 --plots-dir plots/r2
```

The second command runs the long R2 Japan chains and redraws Figures 5 and 6.
It uses `reference_r2.json` for the Figure 6 sensitivity results. Omit
`--pg-reference` to refit the sensitivity grid. For the main workflows,
`auto` selects C when available and uses Numba otherwise.

## Contact

Contact Hengtao Zhang at [bobzht341@gmail.com](mailto:bobzht341@gmail.com) or
open a [GitHub issue](https://github.com/BobZhangHT/RJMCMC_CFR/issues) for code questions.

[MIT License](LICENSE)
