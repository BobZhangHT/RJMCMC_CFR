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
```

Generated results, caches, figures, compiled extensions, manuscripts, and local
archives are intentionally excluded from Git.

## Installation

Python 3.9 or newer is required.

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
- `c`: compiled extension; raises a build hint when unavailable.
- `auto`: uses the compiled extension when available and otherwise uses Numba.

The random streams of the backends are independent. Their agreement is tested
statistically through posterior changepoint-count, inclusion-probability, and
CFR summaries rather than draw-for-draw identity.

## Simulation Workflow

Run the complete simulation, analysis, and plotting pipeline:

```bash
python Simulation_Analysis.py --stage all --workers 8 --backend auto
```

Useful focused stages include:

```bash
python Simulation_Analysis.py --stage diagnostics --workers 4 --backend auto
python Simulation_Analysis.py --stage benchmark-delay --benchmark-delay-reps 100 --workers 8
python Simulation_Analysis.py --stage sensitivity --workers 8 --backend auto
```

## Japan Workflow

Run the real-data analysis:

```bash
python Realdata_Analysis_JP.py --workers 8 --backend auto
```

Run the convergence diagnostics and prior-sensitivity jobs without repeating
the benchmark fits:

```bash
python Realdata_Analysis_JP.py --reviewer-only --workers 4 --backend auto
```

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
