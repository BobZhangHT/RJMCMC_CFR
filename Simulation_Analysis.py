"""Runnable version of ``Simulation_Analysis.ipynb``.

The three notebook stages are retained: sensitivity runs, main runs using the
selected priors, and the existing analysis/plot workflow. Every simulation is
an independent atomic checkpoint, making interrupted runs safe to resume.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import re
import time
from pathlib import Path
from typing import Any

from workflow_runtime import (
    atomic_pickle_dump,
    atomic_text_write,
    ensure_ignored_output,
    load_pickle,
    load_rjmcmc,
    run_parallel,
    stable_seed,
    call_rjmcmc,
)


SCRIPT_DIR = Path(__file__).resolve().parent


def _safe_component(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(value)).strip("-")


def _result_path(output_dir: str | os.PathLike[str], scenario: str, rep_idx: int, params: dict[str, Any]) -> Path:
    parts = [_safe_component(scenario), f"rep{rep_idx}"]
    if "PRIOR_K_GEOMETRIC_P" in params:
        parts.append(f"p{_safe_component(params['PRIOR_K_GEOMETRIC_P'])}")
    if "PRIOR_THETA_SIGMA" in params:
        parts.append(f"s{_safe_component(params['PRIOR_THETA_SIGMA'])}")
    if "DELAY_DIST_NAME" in params:
        parts.append(f"delay_{_safe_component(str(params['DELAY_DIST_NAME']).split(' ')[0])}")
    return Path(output_dir) / ("_".join(parts) + ".pkl")


def _configure_project(runtime: dict[str, Any], smoke: bool = False) -> None:
    """Set config before importing modules that copy config constants."""
    import config

    config.SIGNAL_CACHE_DIR = str(runtime["signal_cache_dir"])
    config.MAIN_RESULTS_DIR = str(runtime["main_results_dir"])
    config.SENSITIVITY_RESULTS_DIR = str(runtime["sensitivity_results_dir"])
    config.PLOTS_DIR = str(runtime["plots_dir"])
    config.SEED = int(runtime["seed"])
    if smoke:
        config.T = 24
        config.MCMC_ITER = 24
        config.MCMC_BURN_IN = 8
        config.K_MAX = 3


def simulation_worker(task: tuple[Any, ...]) -> str:
    """Run one simulation and atomically write its checkpoint."""
    scenario_name, rep_idx, output_dir, params, run_benchmarks, runtime, resume, smoke = task
    _configure_project(runtime, smoke=smoke)
    output_path = _result_path(output_dir, scenario_name, rep_idx, params)
    if resume and output_path.exists():
        return f"cached: {output_path}"

    import data_generation
    from config import SCENARIOS
    from methods import run_binseg, run_pelt, run_rtacfr
    run_rjmcmc = load_rjmcmc()

    # data_generation imports SEED as a module value, so update it for each
    # task when a worker process is reused by the executor.
    data_generation.SEED = int(runtime["seed"])
    import numpy as np

    np.random.seed(stable_seed(runtime["seed"], "simulation", scenario_name, rep_idx))
    data = data_generation.generate_dataset(SCENARIOS[scenario_name], rep_idx)
    task_seed = stable_seed(runtime["seed"], "simulation", scenario_name, rep_idx)
    results: dict[str, Any] = {
        "scenario": scenario_name,
        "rep_idx": rep_idx,
        "params": params,
        "data": data,
        "rjmcmc": call_rjmcmc(
            run_rjmcmc,
            data,
            p_geom=params.get("PRIOR_K_GEOMETRIC_P"),
            theta_sigma=params.get("PRIOR_THETA_SIGMA"),
            seed=task_seed,
            backend=runtime.get("backend"),
            delay_dist=params.get("DELAY_DIST"),
        ),
    }
    if run_benchmarks:
        results["rtacfr"] = run_rtacfr(data, scenario_name, rep_idx)
        results["pelt"] = run_pelt(data, scenario_name, rep_idx)
        results["binseg"] = run_binseg(data, scenario_name, rep_idx)

    atomic_pickle_dump(results, output_path)
    return f"completed: {output_path}"


def _benchmark_delay_path(root: str | os.PathLike[str], scenario: str, rep_idx: int, delay_name: str) -> Path:
    return Path(root) / f"{_safe_component(scenario)}_rep{rep_idx}_delay_{_safe_component(delay_name)}.pkl"


def benchmark_delay_worker(task: tuple[Any, ...]) -> str:
    """Fit both rtaCFR-based changepoint methods under one assumed delay."""
    scenario_name, rep_idx, delay_name, delay_dist, runtime, checkpoint, resume = task
    checkpoint = Path(checkpoint)
    if resume and checkpoint.exists():
        return f"cached: {checkpoint}"

    _configure_project(runtime, smoke=False)
    import data_generation
    from config import DELAY_DIST, SCENARIOS
    from methods import run_binseg, run_pelt

    data_generation.SEED = int(runtime["seed"])
    # The DGP is fixed at the reference delay; only the fitted delay changes.
    data = data_generation.generate_dataset(SCENARIOS[scenario_name], rep_idx, delay_dist_override=DELAY_DIST)
    cache_tag = f"benchmark-delay_{scenario_name}_{rep_idx}_{delay_name}"
    started = time.perf_counter()
    result = {
        "scenario": scenario_name,
        "rep_idx": rep_idx,
        "delay_setting": delay_name,
        "data": data,
        "pelt": run_pelt(data, scenario_name, rep_idx, delay_dist=delay_dist, cache_tag=cache_tag),
        "binseg": run_binseg(data, scenario_name, rep_idx, delay_dist=delay_dist, cache_tag=cache_tag),
    }
    result["runtime_seconds"] = time.perf_counter() - started
    atomic_pickle_dump(result, checkpoint)
    return f"completed: {checkpoint}"


def _runtime_paths(args: argparse.Namespace) -> dict[str, Any]:
    results_root = ensure_ignored_output(args.results_root, "results")
    plots_dir = ensure_ignored_output(args.plots_dir, "plots")
    return {
        "results_root": results_root,
        "sensitivity_results_dir": results_root / "sensitivity",
        "main_results_dir": results_root / "main",
        "signal_cache_dir": results_root / "rtacfr_cache",
        "plots_dir": plots_dir,
        "seed": args.seed,
        "backend": args.backend,
    }


def _build_tasks(args: argparse.Namespace, runtime: dict[str, Any], stage: str) -> list[tuple[Any, ...]]:
    import config

    smoke = args.smoke
    scenarios = list(config.SCENARIOS)
    if smoke:
        scenarios = scenarios[:1]

    if stage == "sensitivity":
        delays = list(config.DELAY_DIST_SENSITIVITY.items())
        priors = list(
            __import__("itertools").product(
                config.SENSITIVITY_GRID_PRIORS["PRIOR_K_GEOMETRIC_P"],
                config.SENSITIVITY_GRID_PRIORS["PRIOR_THETA_SIGMA"],
            )
        )
        if smoke:
            delays = [item for item in delays if item[0].startswith("Perfectly Matched")]
            priors = priors[:1]
        replications = 1 if smoke else args.sensitivity_reps or config.SENSITIVITY_REPLICATIONS
        tasks = []
        for delay_name, delay_dist in delays:
            for scenario_name in scenarios:
                for rep_idx in range(replications):
                    for p_geom, theta_sigma in priors:
                        params = {
                            "PRIOR_K_GEOMETRIC_P": p_geom,
                            "PRIOR_THETA_SIGMA": theta_sigma,
                            "DELAY_DIST_NAME": delay_name,
                            "DELAY_DIST": delay_dist,
                        }
                        tasks.append(
                            (
                                scenario_name,
                                rep_idx,
                                runtime["sensitivity_results_dir"],
                                params,
                                False,
                                runtime,
                                args.resume,
                                smoke,
                            )
                        )
        return tasks

    if (args.main_p_geom is None) != (args.main_theta_sigma is None):
        raise SystemExit("--main-p-geom and --main-theta-sigma must be supplied together")
    if args.main_p_geom is not None:
        optimal_params = {
            "PRIOR_K_GEOMETRIC_P": args.main_p_geom,
            "PRIOR_THETA_SIGMA": args.main_theta_sigma,
        }
    else:
        from analysis import find_optimal_hyperparameters, load_and_process_sensitivity_results

        sensitivity_df = load_and_process_sensitivity_results(runtime["sensitivity_results_dir"])
        optimal_params = find_optimal_hyperparameters(sensitivity_df)
    replications = 1 if smoke else args.main_reps or config.N_REPLICATIONS
    return [
        (
            scenario_name,
            rep_idx,
            runtime["main_results_dir"],
            optimal_params,
            True,
            runtime,
            args.resume,
            smoke,
        )
        for scenario_name in scenarios
        for rep_idx in range(replications)
    ]


def run_stage(args: argparse.Namespace, runtime: dict[str, Any], stage: str) -> None:
    tasks = _build_tasks(args, runtime, stage)
    output_dir = runtime[f"{stage}_results_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    pending = tasks
    if args.resume:
        pending = [task for task in tasks if not _result_path(task[2], task[0], task[1], task[3]).exists()]
    print(f"{stage.title()}: {len(pending)} pending tasks ({len(tasks)} total)")
    run_parallel(pending, simulation_worker, args.workers, f"{stage.title()} simulations")


def run_benchmark_delay(args: argparse.Namespace, runtime: dict[str, Any]) -> None:
    """Run and summarize matched DGP / misspecified benchmark-delay fits."""
    import config

    root = Path(runtime["results_root"]) / "benchmark_delay_sensitivity"
    root.mkdir(parents=True, exist_ok=True)
    reps = args.benchmark_delay_reps or config.SENSITIVITY_REPLICATIONS
    tasks = []
    for delay_name, delay_dist in config.DELAY_DIST_SENSITIVITY.items():
        for scenario_name in config.SCENARIOS:
            for rep_idx in range(reps):
                checkpoint = _benchmark_delay_path(root, scenario_name, rep_idx, delay_name)
                tasks.append((scenario_name, rep_idx, delay_name, delay_dist, runtime, checkpoint, args.resume))
    pending = [task for task in tasks if not (args.resume and Path(task[-2]).exists())]
    print(f"Benchmark delay sensitivity: {len(pending)} pending tasks ({len(tasks)} total)")
    run_parallel(pending, benchmark_delay_worker, args.workers, "Benchmark delay sensitivity")

    from analysis import calculate_accuracy, calculate_hausdorff, calculate_mae

    rows: list[dict[str, Any]] = []
    for task in tasks:
        checkpoint = Path(task[-2])
        if not checkpoint.exists():
            continue
        record = load_pickle(checkpoint)
        data = record["data"]
        for method in ("pelt", "binseg"):
            estimate = record[method]
            rows.append({
                "scenario": record["scenario"],
                "delay_setting": record["delay_setting"],
                "method": method,
                "rep_idx": record["rep_idx"],
                "accuracy": calculate_accuracy(len(data["true_cps"]), estimate["k_est"]),
                "hausdorff": calculate_hausdorff(data["true_cps"], estimate["taus_est"]),
                "mae": calculate_mae(data["true_p_t"], estimate["p_t_hat"]),
                "runtime_seconds": record["runtime_seconds"],
            })
    atomic_text_write(_csv_text(rows), root / "benchmark_delay_runs.csv")

    import pandas as pd

    frame = pd.DataFrame(rows)
    if not frame.empty:
        summary = frame.groupby(["delay_setting", "method"], as_index=False).agg(
            mean_accuracy=("accuracy", "mean"),
            mean_hausdorff=("hausdorff", "mean"),
            mean_mae=("mae", "mean"),
            mean_runtime_seconds=("runtime_seconds", "mean"),
            completed_runs=("rep_idx", "count"),
        )
        atomic_text_write(summary.to_csv(index=False), root / "benchmark_delay_summary.csv")
        scenario_summary = frame.groupby(
            ["scenario", "delay_setting", "method"], as_index=False
        ).agg(
            mean_accuracy=("accuracy", "mean"),
            mean_hausdorff=("hausdorff", "mean"),
            mean_mae=("mae", "mean"),
            completed_runs=("rep_idx", "count"),
        )
        atomic_text_write(
            scenario_summary.to_csv(index=False),
            root / "benchmark_delay_by_scenario.csv",
        )


def _diagnostic_worker(task: tuple[Any, ...]) -> str:
    scenario_name, chain_idx, initial_state, runtime, iterations, checkpoint, resume = task
    checkpoint = Path(checkpoint)
    if resume and checkpoint.exists():
        return f"cached: {checkpoint}"
    _configure_project(runtime, smoke=False)
    from config import PRIOR_K_GEOMETRIC_P, PRIOR_THETA_SIGMA, SCENARIOS
    from data_generation import generate_dataset
    run_rjmcmc = load_rjmcmc()

    data = generate_dataset(SCENARIOS[scenario_name], rep_idx=0)
    started = time.perf_counter()
    result = call_rjmcmc(
        run_rjmcmc,
        data,
        p_geom=PRIOR_K_GEOMETRIC_P,
        theta_sigma=PRIOR_THETA_SIGMA,
        seed=stable_seed(runtime["seed"], "diagnostic", scenario_name, chain_idx),
        backend=runtime.get("backend"),
        iterations=iterations,
        initial_state=initial_state,
        return_samples=True,
        diagnostics=True,
    )
    atomic_pickle_dump(
        {
            "scenario": scenario_name,
            "chain": chain_idx,
            "initial_state": initial_state,
            "runtime_seconds": time.perf_counter() - started,
            "result": result,
        },
        checkpoint,
    )
    return f"completed: {checkpoint}"


def _csv_text(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    if not rows:
        return "\n"
    stream = StringIO()
    fields = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def run_diagnostics(args: argparse.Namespace, runtime: dict[str, Any]) -> None:
    import numpy as np
    import config

    root = Path(runtime["results_root"]) / "reviewer_simulation_diagnostics"
    chain_root = root / "chains"
    chain_root.mkdir(parents=True, exist_ok=True)
    Path(runtime["plots_dir"]).mkdir(parents=True, exist_ok=True)
    diagnostic_k = min(5, config.K_MAX)
    spread_taus = np.linspace(2, config.T - 1, diagnostic_k, dtype=int).tolist()
    states = [
        {"k": 0, "taus": [], "theta_values": [-5.5]},
        {"k": 0, "taus": [], "theta_values": [-2.0]},
        {"k": diagnostic_k, "taus": spread_taus, "theta_values": [-5.5] * (diagnostic_k + 1)},
        {"k": diagnostic_k, "taus": spread_taus, "theta_values": [-2.0] * (diagnostic_k + 1)},
    ]
    tasks = []
    for scenario_name in config.SCENARIOS:
        for chain_idx, state in enumerate(states):
            checkpoint = chain_root / f"{_safe_component(scenario_name)}_chain{chain_idx}.pkl"
            tasks.append((scenario_name, chain_idx, state, runtime, args.diagnostic_iterations, checkpoint, args.resume))
    pending = [task for task in tasks if not (args.resume and Path(task[-2]).exists())]
    run_parallel(pending, _diagnostic_worker, args.workers, "Simulation diagnostic chains")

    records = [load_pickle(task[-2]) for task in tasks if Path(task[-2]).exists()]
    rows: list[dict[str, Any]] = []
    trace_data: dict[str, dict[str, list[tuple[int, Any]]]] = {}
    import arviz as az

    for scenario_name in config.SCENARIOS:
        scenario_records = [record for record in records if record["scenario"] == scenario_name]
        if len(scenario_records) != len(states):
            continue
        k_draws = np.stack([record["result"]["samples"]["k"][:, None] for record in scenario_records])
        p_draws = np.stack([record["result"]["samples"]["p_t"] for record in scenario_records])
        mean_p_draws = p_draws.mean(axis=2, keepdims=True)
        posterior = {"k": k_draws, "p_t": p_draws, "mean_p_t": mean_p_draws}
        try:
            inference_data = az.from_dict(posterior=posterior)
        except TypeError:
            inference_data = az.from_dict({"posterior": posterior})
        rhat_result = az.rhat(inference_data, method="rank")
        ess_result = az.ess(inference_data, method="bulk")
        selected_p_t = sorted({0, config.T // 4, config.T // 2, (3 * config.T) // 4, config.T - 1})
        for variable in posterior:
            variable_rhat = np.asarray(rhat_result[variable]).reshape(-1)
            variable_ess = np.asarray(ess_result[variable]).reshape(-1)
            rows.append(
                {
                    "scenario": scenario_name,
                    "variable": variable,
                    "coordinate": "all (worst)",
                    "max_rank_normalized_rhat": float(np.nanmax(variable_rhat)),
                    "min_bulk_ess": float(np.nanmin(variable_ess)),
                    "mean_runtime_seconds": float(np.mean([record["runtime_seconds"] for record in scenario_records])),
                }
            )
            if variable == "p_t":
                for time_idx in selected_p_t:
                    rows.append(
                        {
                            "scenario": scenario_name,
                            "variable": "p_t",
                            "coordinate": f"t={time_idx}",
                            "max_rank_normalized_rhat": float(variable_rhat[time_idx]),
                            "min_bulk_ess": float(variable_ess[time_idx]),
                            "mean_runtime_seconds": float(np.mean([record["runtime_seconds"] for record in scenario_records])),
                        }
                    )
        scenario_traces: dict[str, list[tuple[int, Any]]] = {
            "k": [(record["chain"], record["result"]["samples"]["k"]) for record in scenario_records],
            "mean(p_t)": [(record["chain"], record["result"]["samples"]["p_t"].mean(axis=1)) for record in scenario_records],
        }
        for time_idx in selected_p_t:
            scenario_traces[f"p_t[{time_idx}]"] = [
                (record["chain"], record["result"]["samples"]["p_t"][:, time_idx])
                for record in scenario_records
            ]
        trace_data[scenario_name] = scenario_traces
    atomic_text_write(_csv_text(rows), root / "diagnostics_summary.csv")

    import sys
    sys.modules.pop("jax", None)
    sys.modules.pop("jaxlib", None)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style_trace_axis(ax):
        ax.tick_params(axis="both", which="major", labelsize=13, width=1.5)
        for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
            label.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    trace_colors = ("#0F4D92", "#B64342", "#3A7D44", "#7A5195")

    fig, axes = plt.subplots(len(trace_data), 1, figsize=(12, 3 * len(trace_data)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (scenario_name, variables) in zip(axes, trace_data.items()):
        chains = variables["k"]
        for chain_idx, draws in chains:
            ax.plot(draws, color=trace_colors[chain_idx % len(trace_colors)], linewidth=0.7, alpha=0.75, label=f"Chain {chain_idx + 1}")
        ax.set_ylabel(r"$k$", fontsize=14, fontweight="bold")
        ax.set_title(scenario_name, fontsize=15, fontweight="bold")
        style_trace_axis(ax)
    axes[-1].set_xlabel("Post-burn-in draw", fontsize=14, fontweight="bold")
    legend = axes[0].legend(ncol=4, fontsize=10, frameon=False)
    for text_item in legend.get_texts():
        text_item.set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(Path(runtime["plots_dir"]) / "simulation_k_traceplots.pdf")
    plt.close(fig)

    summary_variables = ("k", f"p_t[{config.T // 4}]", f"p_t[{(3 * config.T) // 4}]")
    trace_items = list(trace_data.items())
    representative_items = [trace_items[0], trace_items[-1]] if len(trace_items) > 1 else trace_items
    fig, axes = plt.subplots(
        len(representative_items), len(summary_variables),
        figsize=(15, 3.6 * len(representative_items)), sharex=True, squeeze=False,
    )
    for row_idx, (scenario_name, variables) in enumerate(representative_items):
        for col_idx, variable in enumerate(summary_variables):
            ax = axes[row_idx, col_idx]
            for chain_idx, draws in variables[variable]:
                stride = max(1, len(draws) // 2500)
                ax.plot(
                    np.arange(0, len(draws), stride), draws[::stride],
                    color=trace_colors[chain_idx % len(trace_colors)],
                    linewidth=0.7, alpha=0.7, label=f"Chain {chain_idx + 1}",
                )
            if row_idx == 0:
                column_title = {
                    "k": r"Changepoint count $k$",
                    f"p_t[{config.T // 4}]": rf"Daily CFR $p_t$ at $t={config.T // 4 + 1}$",
                    f"p_t[{(3 * config.T) // 4}]": rf"Daily CFR $p_t$ at $t={(3 * config.T) // 4 + 1}$",
                }[variable]
                ax.set_title(column_title, fontsize=15, fontweight="bold")
            if col_idx == 0:
                y_label = r"$k$" if variable == "k" else r"CFR $p_t$"
                ax.set_ylabel(f"{scenario_name}\n{y_label}", fontsize=14, fontweight="bold")
            else:
                ax.set_ylabel(r"CFR $p_t$", fontsize=14, fontweight="bold")
            if row_idx == len(representative_items) - 1:
                ax.set_xlabel("Post-burn-in draw", fontsize=14, fontweight="bold")
            panel_letter = chr(ord("a") + row_idx * len(summary_variables) + col_idx)
            ax.text(0.015, 0.96, f"({panel_letter})", transform=ax.transAxes,
                    va="top", ha="left", fontsize=14, fontweight="bold")
            style_trace_axis(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
        ncol=4, fontsize=10, frameon=False,
    )
    for text_item in legend.get_texts():
        text_item.set_fontweight("bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(Path(runtime["plots_dir"]) / "simulation_traceplots.pdf")
    plt.close(fig)

    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(Path(runtime["plots_dir"]) / "simulation_multivariable_traceplots.pdf") as pdf:
        for scenario_name, variables in trace_data.items():
            ncols = 2
            nrows = int(np.ceil(len(variables) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), squeeze=False)
            for ax, (variable, chains) in zip(axes.ravel(), variables.items()):
                for chain_idx, draws in chains:
                    stride = max(1, len(draws) // 2500)
                    ax.plot(np.arange(0, len(draws), stride), draws[::stride], linewidth=0.55, alpha=0.6, label=f"Chain {chain_idx + 1}")
                ax.set_title(variable)
                ax.set_xlabel("Post-burn-in draw")
                ax.set_ylabel(variable)
            for ax in axes.ravel()[len(variables):]:
                ax.set_visible(False)
            axes.ravel()[0].legend(ncol=4, fontsize=8)
            fig.suptitle(scenario_name)
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "sensitivity", "main", "analysis", "diagnostics", "benchmark-delay"), default="all")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--backend", default=None, help="Optional RJMCMC backend forwarded when supported")
    parser.add_argument("--sensitivity-reps", type=int, default=None)
    parser.add_argument("--main-reps", type=int, default=None)
    parser.add_argument("--main-p-geom", type=float, default=None)
    parser.add_argument("--main-theta-sigma", type=float, default=None)
    parser.add_argument("--diagnostic-iterations", type=int, default=20000)
    parser.add_argument("--benchmark-delay-reps", type=int, default=None)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one tiny sensitivity task with reduced MCMC settings")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if args.smoke and args.stage == "all":
        args.stage = "sensitivity"
    runtime = _runtime_paths(args)
    _configure_project(runtime, smoke=args.smoke)

    if args.stage in ("all", "sensitivity"):
        run_stage(args, runtime, "sensitivity")
    if args.stage in ("all", "main"):
        run_stage(args, runtime, "main")
    if args.stage in ("all", "analysis") and not args.skip_analysis and not args.smoke:
        from analysis import full_analysis_workflow

        analysis_override = None
        if args.main_p_geom is not None and args.main_theta_sigma is not None:
            analysis_override = {
                "PRIOR_K_GEOMETRIC_P": args.main_p_geom,
                "PRIOR_THETA_SIGMA": args.main_theta_sigma,
            }
        full_analysis_workflow(optimal_params_override=analysis_override)
    if args.stage == "diagnostics":
        run_diagnostics(args, runtime)
    if args.stage == "benchmark-delay":
        run_benchmark_delay(args, runtime)


if __name__ == "__main__":
    mp.freeze_support()
    main()
