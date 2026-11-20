# RJMCMC-CFR

Bayesian changepoint inference for a time-varying reported-case fatality rate
(CFR). Daily deaths are modeled through a convolution of observed case counts,
a case-to-death delay distribution, and a piecewise-constant latent CFR.

## Contents

```text
methods.py                 Canonical RJMCMC entry point and benchmark methods
rjmcmc_c.c                 Optional compiled C sampler backend
Simulation_Analysis.py     Parallel and resumable simulation workflow
Realdata_Analysis_JP.py    Parallel and resumable Japan analysis
workflow_runtime.py        Checkpoints, deterministic seeds, and progress bars
analysis.py                Simulation summaries, tables, and figures
evaluation_realdata.py     Real-data evaluation metrics
data_generation.py         Simulation data generation
config.py                  Default model and workflow settings
JP_Data.csv                Japan case and death data used by the workflow
events.csv                 Contextual Japan event dates
tests/                     Backend and workflow consistency tests
reference_r2.json          R2 diagnostic references and frozen Figure 6 inputs
```

Generated results, caches, figures, compiled extensions, manuscripts, and local
archives are intentionally excluded from Git.

## Installation

The R2 reference environment uses CPython 3.12.10. Use Python 3.12 and the
package versions pinned in `requirements.txt` for numerical reproduction.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The NumPy/Numba backend runs without a compiler. To build the optional C
backend in place:

```bash
python setup.py build_ext --inplace
```

On Windows, a configured MSVC Build Tools installation or MinGW compiler is
required for the extension build.

```bash
python setup.py build_ext --inplace --compiler=mingw32
```

## Sampler API

Both analysis workflows call the same canonical function:

```python
from methods import run_rjmcmc

result = run_rjmcmc(data, backend="numba", seed=2025)
```

Supported backend values are:

- `numba`: portable NumPy/Numba implementation.
- `numba-fast`: the equivalent local-move implementation used for the R2 Japan
  long chains; supports storage thinning without allocating unsaved draws.
- `c`: compiled extension; raises a build hint when unavailable.
- `auto`: uses the compiled extension when available and otherwise uses Numba.

The C and NumPy random streams differ; their posterior summaries are tested
for statistical agreement. The fast local kernel also has seeded trajectory
and storage-equivalence tests against the original NumPy kernel.

## Simulation Workflow

Run the complete simulation, analysis, and plotting pipeline:

```bash
python Simulation_Analysis.py --stage all --workers 8 --backend auto
```

Useful focused stages include:

```bash
python Simulation_Analysis.py --stage benchmark-delay --benchmark-delay-reps 100 --workers 8
python Simulation_Analysis.py --stage sensitivity --workers 8 --backend auto
```

To reproduce the R2 simulation diagnostics and Figure 3, build the C extension
above and run:

```bash
python Simulation_Analysis.py --stage diagnostics --backend c --workers 4 --results-root results/r2 --plots-dir plots/r2
```

The diagnostic default is 200,000 draws after 5,000 burn-in iterations, without
thinning, for four dispersed chains in each of the five scenarios. This does
not change the 20,000-draw fits in the 500-replication performance study.

## Japan Workflow

Run the real-data analysis:

```bash
python Realdata_Analysis_JP.py --workers 8 --backend auto
```

Run the convergence diagnostics and prior-sensitivity jobs without repeating
the benchmark fits:

```bash
python Realdata_Analysis_JP.py --reviewer-only --workers 4 --backend numba-fast --pg-reference reference_r2.json --results-root results/r2 --plots-dir plots/r2
```

This reproduces the R2 Japan diagnostics and Figures 5 and 6. Each chain uses
2,000,000 burn-in and 20,000,000 post-burn-in transitions, storing every 200th
state (100,000 per chain). The diagnostic seed is 20261121 with seed family
`reviewer-original-long-confirmation`, separate from the benchmark seed.
The original model, four proposals, selection schedule, acceptance ratios and
proposal settings (M=5, theta SD=0.2, split SD=0.5) are retained. All 700 daily
CFR values, changepoint count and mean CFR are checked with rank-normalized
split/folded Rhat and bulk/tail ESS. Expect roughly half an hour on the
reference workstation; runtime depends on the machine.

`--pg-reference` redraws the existing R1 sensitivity results from a compact
numeric snapshot because R2 changes only Figure 6(a)'s grayscale encoding.
The snapshot is not used in the new Japan sampling or convergence diagnostics.
Omit this option to fit a new sensitivity grid. No ignored archive or manuscript
file is needed by these commands.

| R2 reference quantity | Maximum Rhat | Minimum bulk ESS |
|---|---:|---:|
| Simulations, all five diagnostic datasets | 1.00297 | 1285.4 |
| Japan, changepoint count | 1.00003 | 82283.3 |
| Japan, mean CFR | 1.00065 | 4810.9 |
| Japan, all 700 daily CFR values | 1.00621 | 626.4 |

The largest Japan daily Rhat occurs on day 513 (27 June 2021); the minimum
daily tail ESS is 616.8. Diagnostic CSVs are under
`results/r2/realdata/reviewer/` and
`results/r2/reviewer_simulation_diagnostics/`. Plots are
`plots/r2/simulation_traceplots.pdf`, `reviewer_rjmcmc_trace.pdf` and
`p_g_sensitivity.pdf`. Plot display downsampling does not filter diagnostics.
Checkpoint provenance prevents an earlier short run from being reused as R2.

## Checkpoints and Outputs

Each independent task writes an atomic checkpoint under `results/`. Repeating
the same command resumes completed tasks by default. Use `--no-resume` to
recompute them. Parallel stages use process workers and display progress bars.

Use `--results-root` and `--plots-dir` to place generated artifacts elsewhere:

```bash
python Simulation_Analysis.py --stage diagnostics \
  --results-root results/diagnostics --plots-dir plots/diagnostics
```

Neither checkpoints nor generated plots are tracked by Git.

## Verification

Run the portable test suite:

```bash
python -m pytest -q
```

After building the C extension, run the same command again to include the
cross-backend statistical consistency tests.

For quick workflow checks:

```bash
python Simulation_Analysis.py --smoke --workers 1 --backend numba
python Realdata_Analysis_JP.py --smoke --workers 1 --backend numba
```

## License

This project is released under the [MIT License](LICENSE).

## Citation

Please cite the accompanying manuscript when using this code. Publication
metadata will be added here when the article record is available.
