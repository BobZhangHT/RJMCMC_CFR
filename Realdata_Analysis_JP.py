"""Runnable version of ``Realdata_Analysis_JP.ipynb`` with resumable outputs."""

from __future__ import annotations

import argparse
import csv
import inspect
import multiprocessing as mp
import os
import time
from datetime import datetime
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
EVENTS = {
    "event_name": [
        "School closures requested",
        "SoE declared (Wave 1)",
        "SoE expanded nationwide",
        "SoE lifted",
        "Okinawa SoE",
        "Second SoE (Tokyo area)",
        "SoE expanded to 11 cities",
        "SoE fully lifted",
        "Vaccination begins (HCW)",
        "Moderna & AZ approved",
    ],
    "date": [
        "2020-02-27",
        "2020-04-07",
        "2020-04-16",
        "2020-05-25",
        "2020-08-01",
        "2021-01-07",
        "2021-01-13",
        "2021-09-28",
        "2021-02-17",
        "2021-05-21",
    ],
}


def _configure_project(runtime: dict[str, Any], smoke: bool) -> None:
    import config

    config.PLOTS_DIR = str(runtime["plots_dir"])
    config.SIGNAL_CACHE_DIR = str(runtime["signal_cache_dir"])
    config.SEED = int(runtime["seed"])
    config.RJMCMC_BACKEND = runtime.get("backend")
    if smoke:
        config.K_MAX = min(3, config.K_MAX)
    if smoke:
        config.MCMC_ITER = 24
        config.MCMC_BURN_IN = 8


def _runtime_paths(args: argparse.Namespace) -> dict[str, Any]:
    results_root = ensure_ignored_output(args.results_root, "results")
    plots_dir = ensure_ignored_output(args.plots_dir, "plots")
    realdata_root = results_root / ("realdata_smoke" if args.smoke else "realdata")
    return {
        "results_root": results_root,
        "realdata_root": realdata_root,
        "plots_dir": plots_dir,
        "signal_cache_dir": realdata_root / "signal_cache",
        "seed": args.seed,
        "backend": args.backend,
        "move_window": args.move_window,
        "theta_prop_sigma": args.theta_proposal_sigma,
        "global_move_prob": args.global_move_probability,
        "max_k": args.max_changepoints,
        "u_prop_sigma": args.split_proposal_sigma,
        "chain_burn_in": args.chain_burn_in,
    }


def _load_data(data_path: Path, smoke: bool):
    import pandas as pd

    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if smoke:
        df = df.head(60).copy()
    data = {"cases": df["ct"].values, "deaths": df["dt"].values}
    print(f"Loaded data for Japan from {df['date'].min().date()} to {df['date'].max().date()} (T = {len(df)} days).")
    return df, data


def _method_checkpoint(runtime: dict[str, Any], method: str) -> Path:
    return Path(runtime["realdata_root"]) / f"{method.lower()}_results.pkl"


def _fit_method(method: str, data: dict[str, Any], runtime: dict[str, Any], args: argparse.Namespace):
    checkpoint = _method_checkpoint(runtime, method)
    if args.resume and checkpoint.exists():
        return load_pickle(checkpoint)

    _configure_project(runtime, args.smoke)
    import numpy as np
    from config import REAL_DATA_PRIOR_K_GEOMETRIC_P, REAL_DATA_PRIOR_THETA_SIGMA
    from methods import get_rtacfr_signal, run_binseg, run_pelt
    run_rjmcmc = load_rjmcmc()

    np.random.seed(stable_seed(args.seed, "realdata", method))
    if method == "RJMCMC":
        result = call_rjmcmc(
            run_rjmcmc,
            data,
            p_geom=REAL_DATA_PRIOR_K_GEOMETRIC_P,
            theta_sigma=REAL_DATA_PRIOR_THETA_SIGMA,
            seed=stable_seed(args.seed, "realdata", method),
            backend=runtime.get("backend"),
            move_window=runtime.get("move_window"),
            theta_prop_sigma=runtime.get("theta_prop_sigma"),
            global_move_prob=runtime.get("global_move_prob"),
            max_k=runtime.get("max_k"),
            u_prop_sigma=runtime.get("u_prop_sigma"),
            summary_method="mode",
        )
    elif method == "RTACFR":
        result = get_rtacfr_signal(data, "JapanRealData", 0)
    elif method == "PELT":
        result = run_pelt(data, "JapanRealData", 0)
    elif method == "BinSeg":
        result = run_binseg(data, "JapanRealData", 0)
    else:
        raise ValueError(f"Unknown method: {method}")
    atomic_pickle_dump(result, checkpoint)
    return result


def _rmse_worker(task: tuple[Any, ...]):
    method, data, runtime, n_splits, seed, smoke = task
    _configure_project(runtime, smoke)
    import numpy as np
    from evaluation_realdata import calculate_out_of_sample_rmse
    from methods import run_binseg, run_pelt
    run_rjmcmc = load_rjmcmc()
    from config import DELAY_DIST

    np.random.seed(stable_seed(seed, "realdata-rmse", method))
    def flexible_rjmcmc(data_for_fit, p_geom=None, theta_sigma=None):
        return call_rjmcmc(
            run_rjmcmc,
            data_for_fit,
            p_geom=p_geom,
            theta_sigma=theta_sigma,
            seed=stable_seed(seed, "realdata-rmse", method, len(data_for_fit["cases"])),
            backend=runtime.get("backend"),
            move_window=runtime.get("move_window"),
            theta_prop_sigma=runtime.get("theta_prop_sigma"),
            global_move_prob=runtime.get("global_move_prob"),
            max_k=runtime.get("max_k"),
            u_prop_sigma=runtime.get("u_prop_sigma"),
        )

    flexible_rjmcmc.__name__ = "run_rjmcmc"
    functions = {"RJMCMC": flexible_rjmcmc, "PELT": run_pelt, "BinSeg": run_binseg}
    value = calculate_out_of_sample_rmse(data, functions[method], DELAY_DIST, n_splits=n_splits)
    return method, float(value)


def _extract_trace(result: Any) -> dict[str, Any]:
    """Extract trace-like arrays from common extended RJMCMC return shapes."""
    if not isinstance(result, dict):
        return {}
    trace = result.get("trace", result.get("samples"))
    if isinstance(trace, dict):
        return trace
    return {
        key: result[key]
        for key in ("k_samples", "theta_samples", "taus_samples")
        if key in result
    }


def _reviewer_chain_worker(task: tuple[Any, ...]):
    chain_idx, initial_state, data, runtime, chain_iterations, seed, smoke, checkpoint, resume = task
    checkpoint = Path(checkpoint)
    if resume and checkpoint.exists():
        return load_pickle(checkpoint)
    _configure_project(runtime, smoke)
    import numpy as np
    from config import REAL_DATA_PRIOR_K_GEOMETRIC_P, REAL_DATA_PRIOR_THETA_SIGMA
    run_rjmcmc = load_rjmcmc()

    started = time.perf_counter()
    call_seed = stable_seed(seed, "reviewer-chain", chain_idx)
    signature = inspect.signature(run_rjmcmc)
    supports_initial_state = "initial_state" in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    requested_state = initial_state if supports_initial_state else None
    state_status = "requested" if supports_initial_state else "unsupported_by_api"
    try:
        result = call_rjmcmc(
            run_rjmcmc,
            data,
            p_geom=REAL_DATA_PRIOR_K_GEOMETRIC_P,
            theta_sigma=REAL_DATA_PRIOR_THETA_SIGMA,
            seed=call_seed,
            backend=runtime.get("backend"),
            iterations=chain_iterations,
            burn_in=runtime.get("chain_burn_in"),
            initial_state=requested_state,
            return_samples=True,
            diagnostics=True,
            move_window=runtime.get("move_window"),
            theta_prop_sigma=runtime.get("theta_prop_sigma"),
            global_move_prob=runtime.get("global_move_prob"),
            max_k=runtime.get("max_k"),
            u_prop_sigma=runtime.get("u_prop_sigma"),
            summary_method="mode",
        )
    except TypeError:
        # Some experimental APIs advertise initial_state but use a narrower
        # schema. Keep the chain result, while making the limitation visible.
        state_status = "initial_state_rejected"
        result = call_rjmcmc(
            run_rjmcmc,
            data,
            p_geom=REAL_DATA_PRIOR_K_GEOMETRIC_P,
            theta_sigma=REAL_DATA_PRIOR_THETA_SIGMA,
            seed=call_seed,
            backend=runtime.get("backend"),
            iterations=chain_iterations,
            burn_in=runtime.get("chain_burn_in"),
            return_samples=True,
            diagnostics=True,
            move_window=runtime.get("move_window"),
            theta_prop_sigma=runtime.get("theta_prop_sigma"),
            global_move_prob=runtime.get("global_move_prob"),
            max_k=runtime.get("max_k"),
            u_prop_sigma=runtime.get("u_prop_sigma"),
            summary_method="mode",
        )
    record = {
        "chain": chain_idx,
        "requested_initial_state": initial_state,
        "initial_state_status": state_status,
        "runtime_seconds": time.perf_counter() - started,
        "result": result,
        "trace": _extract_trace(result),
    }
    atomic_pickle_dump(record, checkpoint)
    return record


def _normalise_trace_array(value: Any):
    import numpy as np

    array = np.asarray(value)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array.reshape(array.shape[0], -1)


def run_reviewer_diagnostics(data: dict[str, Any], runtime: dict[str, Any], args: argparse.Namespace) -> None:
    """Run optional chains and write honest diagnostics/artifacts.

    If the installed RJMCMC API does not return samples, the CSV records that
    limitation and no fabricated R-hat/ESS values are emitted.
    """
    import numpy as np

    reviewer_root = Path(runtime["realdata_root"]) / "reviewer"
    chain_root = reviewer_root / "chains"
    chain_root.mkdir(parents=True, exist_ok=True)
    from config import K_MAX

    configured_k_max = runtime.get("max_k")
    k_max = K_MAX if configured_k_max is None else int(configured_k_max)
    k_max = min(3, k_max) if args.smoke else k_max
    t_length = len(data["cases"])
    spread_taus = list(np.linspace(2, t_length - 1, k_max, dtype=int)) if k_max else []
    states = [
        {"k": 0, "taus": [], "theta_values": [-6.0]},
        {"k": 0, "taus": [], "theta_values": [-1.0]},
        {"k": k_max, "taus": spread_taus, "theta_values": [-6.0] * (k_max + 1)},
        {"k": k_max, "taus": spread_taus, "theta_values": [-1.0] * (k_max + 1)},
    ][: args.chains]
    tasks = [
        (
            idx,
            state,
            data,
            runtime,
            args.chain_iterations,
            args.seed,
            args.smoke,
            chain_root / f"chain_{idx:02d}.pkl",
            args.resume,
        )
        for idx, state in enumerate(states)
    ]
    pending = [task for task in tasks if not (args.resume and Path(task[-2]).exists())]
    run_parallel(pending, _reviewer_chain_worker, args.workers, "Reviewer RJMCMC chains")
    records = [load_pickle(task[-2]) for task in tasks if Path(task[-2]).exists()]

    rows = [
        {
            "chain": record["chain"],
            "runtime_seconds": record["runtime_seconds"],
            "initial_state_status": record["initial_state_status"],
            "trace_available": bool(record.get("trace")),
        }
        for record in records
    ]
    runtime_csv = reviewer_root / "chain_runtime.csv"
    atomic_text_write(_csv_text(rows), runtime_csv)

    acceptance_rows = []
    for record in records:
        diagnostics = record.get("result", {}).get("sampler_diagnostics", {})
        for name, proposed, accepted in zip(
            diagnostics.get("move_names", ()),
            diagnostics.get("proposed", ()),
            diagnostics.get("accepted", ()),
        ):
            acceptance_rows.append(
                {
                    "chain": record["chain"],
                    "move": name,
                    "proposed": int(proposed),
                    "accepted": int(accepted),
                    "acceptance_rate": float(accepted / proposed) if proposed else float("nan"),
                }
            )
    atomic_text_write(
        _csv_text(acceptance_rows or [{"status": "diagnostics_unavailable"}]),
        reviewer_root / "acceptance_rates.csv",
    )

    trace_by_name: dict[str, list[Any]] = {}
    for record in records:
        for name, value in record.get("trace", {}).items():
            if name not in {"k", "p_t"}:
                continue
            array = _normalise_trace_array(value)
            trace_by_name.setdefault(name, []).append(array)

    metric_rows: list[dict[str, Any]] = []
    coordinate_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    posterior: dict[str, Any] = {}
    selected_p_t: list[int] = []
    if not trace_by_name:
        metric_rows.append({"status": "trace_unavailable_from_run_rjmcmc"})
    else:
        try:
            import arviz as az

            for name, arrays in trace_by_name.items():
                if len({array.shape[1] for array in arrays}) != 1:
                    continue
                posterior[name] = np.stack(arrays, axis=0)
            if "p_t" in posterior:
                posterior["mean_p_t"] = posterior["p_t"].mean(axis=2, keepdims=True)
            if not posterior:
                metric_rows.append({"status": "trace_shapes_incompatible"})
            else:
                try:
                    inference_data = az.from_dict(posterior=posterior)
                except TypeError:
                    inference_data = az.from_dict({"posterior": posterior})
                rhat_result = az.rhat(inference_data, method="rank")
                ess_result = az.ess(inference_data, method="bulk")
                if "p_t" in posterior:
                    t_length = posterior["p_t"].shape[2]
                    selected_p_t = sorted(
                        {int(round(value)) for value in np.linspace(0, t_length - 1, 6)}
                    )
                for variable in posterior:
                    variable_rhat = np.asarray(rhat_result[variable]).reshape(-1)
                    variable_ess = np.asarray(ess_result[variable]).reshape(-1)
                    coordinate_prefix = "t=" if variable == "p_t" else "index="
                    finite_rhat = np.flatnonzero(np.isfinite(variable_rhat))
                    finite_ess = np.flatnonzero(np.isfinite(variable_ess))
                    worst_rhat_idx = (
                        int(finite_rhat[np.argmax(variable_rhat[finite_rhat])])
                        if len(finite_rhat) else None
                    )
                    minimum_ess_idx = (
                        int(finite_ess[np.argmin(variable_ess[finite_ess])])
                        if len(finite_ess) else None
                    )
                    metric_rows.append(
                        {
                            "variable": variable,
                            "coordinate": "all (worst)",
                            "rank_normalized_rhat": (
                                float(variable_rhat[worst_rhat_idx])
                                if worst_rhat_idx is not None else float("nan")
                            ),
                            "bulk_ess": (
                                float(variable_ess[minimum_ess_idx])
                                if minimum_ess_idx is not None else float("nan")
                            ),
                            "worst_rhat_coordinate": (
                                f"{coordinate_prefix}{worst_rhat_idx}"
                                if worst_rhat_idx is not None else "constant_or_undefined"
                            ),
                            "minimum_ess_coordinate": (
                                f"{coordinate_prefix}{minimum_ess_idx}"
                                if minimum_ess_idx is not None else "constant_or_undefined"
                            ),
                            "status": "ok" if worst_rhat_idx is not None else "constant_or_undefined",
                        }
                    )
                    if variable == "p_t":
                        coordinate_rows.extend(
                            {
                                "coordinate": f"t={time_idx}",
                                "rank_normalized_rhat": float(variable_rhat[time_idx]),
                                "bulk_ess": float(variable_ess[time_idx]),
                            }
                            for time_idx in range(len(variable_rhat))
                        )
                        distribution_rows.append(
                            {
                                "variable": "p_t",
                                "n_coordinates": len(variable_rhat),
                                "median_rhat": float(np.nanmedian(variable_rhat)),
                                "rhat_q90": float(np.nanquantile(variable_rhat, 0.90)),
                                "rhat_q95": float(np.nanquantile(variable_rhat, 0.95)),
                                "rhat_q99": float(np.nanquantile(variable_rhat, 0.99)),
                                "proportion_rhat_le_1_01": float(np.nanmean(variable_rhat <= 1.01)),
                                "proportion_rhat_le_1_05": float(np.nanmean(variable_rhat <= 1.05)),
                                "proportion_rhat_le_1_10": float(np.nanmean(variable_rhat <= 1.10)),
                            }
                        )
                        for time_idx in selected_p_t:
                            metric_rows.append(
                                {
                                    "variable": "p_t",
                                    "coordinate": f"t={time_idx}",
                                    "rank_normalized_rhat": float(variable_rhat[time_idx]),
                                    "bulk_ess": float(variable_ess[time_idx]),
                                    "status": "ok",
                                }
                            )
        except ImportError:
            metric_rows.append({"status": "arviz_unavailable"})
    atomic_text_write(_csv_text(metric_rows), reviewer_root / "rjmcmc_convergence.csv")
    atomic_text_write(_csv_text(coordinate_rows), reviewer_root / "rjmcmc_coordinate_diagnostics.csv")
    atomic_text_write(_csv_text(distribution_rows), reviewer_root / "rjmcmc_rhat_distribution.csv")

    trace_series: dict[str, list[Any]] = {}
    if "k" in posterior:
        trace_series["k"] = [chain[:, 0] for chain in posterior["k"]]
    if "p_t" in posterior:
        for time_idx in selected_p_t:
            trace_series[f"p_t[{time_idx}]"] = [chain[:, time_idx] for chain in posterior["p_t"]]
    if "mean_p_t" in posterior:
        trace_series["mean(p_t)"] = [chain[:, 0] for chain in posterior["mean_p_t"]]

    trace_rows: list[dict[str, Any]] = []
    for variable, chains in trace_series.items():
        for chain_idx, values in enumerate(chains):
            stride = max(1, len(values) // 5000)
            trace_rows.extend(
                {
                    "chain": records[chain_idx]["chain"],
                    "variable": variable,
                    "draw": draw_idx,
                    "value": float(values[draw_idx]),
                }
                for draw_idx in range(0, len(values), stride)
            )
    atomic_text_write(_csv_text(trace_rows or [{"status": "trace_unavailable"}]), reviewer_root / "rjmcmc_trace.csv")

    if trace_series:
        # Some ArviZ installations register a lazy optional JAX module. A
        # broken optional JAX binary should not prevent matplotlib output.
        import sys

        sys.modules.pop("jax", None)
        sys.modules.pop("jaxlib", None)
        plt = _configure_plotting()
        trace_colors = ("#0F4D92", "#B64342", "#3A7D44", "#7A5195")
        ncols = 2
        nrows = int(np.ceil(len(trace_series) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows), squeeze=False)
        for panel_idx, (ax, (variable, chains)) in enumerate(zip(axes.ravel(), trace_series.items())):
            for chain_idx, values in enumerate(chains):
                stride = max(1, len(values) // 5000)
                ax.plot(
                    np.arange(0, len(values), stride), values[::stride],
                    color=trace_colors[chain_idx % len(trace_colors)],
                    alpha=0.75, linewidth=0.8,
                    label=f"Chain {records[chain_idx]['chain']}",
                )
            if variable == "k":
                display_label = r"Changepoint count $k$"
                y_label = r"$k$"
            elif variable == "mean(p_t)":
                display_label = r"All-time mean CFR"
                y_label = r"Mean $p_t$"
            else:
                time_idx = int(variable.removeprefix("p_t[").removesuffix("]")) + 1
                display_label = rf"Daily CFR $p_t$ at $t={time_idx}$"
                y_label = r"CFR $p_t$"
            panel_letter = chr(ord("a") + panel_idx)
            ax.set_title(f"({panel_letter}) {display_label}", fontsize=15, fontweight="bold")
            ax.set_xlabel("Post-burn-in draw", fontsize=14, fontweight="bold")
            ax.set_ylabel(y_label, fontsize=14, fontweight="bold")
            _style_publication_axis(ax)
        for ax in axes.ravel()[len(trace_series):]:
            ax.set_visible(False)
        legend = axes.ravel()[0].legend(ncol=min(4, len(records)), fontsize=10, frameon=False)
        for text_item in legend.get_texts():
            text_item.set_fontweight("bold")
        fig.tight_layout()
        fig.savefig(Path(runtime["plots_dir"]) / "reviewer_rjmcmc_trace.png", dpi=300, bbox_inches="tight")
        fig.savefig(Path(runtime["plots_dir"]) / "reviewer_rjmcmc_trace.pdf", bbox_inches="tight")
        plt.close(fig)


def _pg_worker(task: tuple[Any, ...]):
    p_geom, data, runtime, iterations, seed, smoke, checkpoint, resume = task
    checkpoint = Path(checkpoint)
    if resume and checkpoint.exists():
        return load_pickle(checkpoint)
    _configure_project(runtime, smoke)
    from config import REAL_DATA_PRIOR_THETA_SIGMA
    run_rjmcmc = load_rjmcmc()

    started = time.perf_counter()
    result = call_rjmcmc(
        run_rjmcmc,
        data,
        p_geom=p_geom,
        theta_sigma=REAL_DATA_PRIOR_THETA_SIGMA,
        seed=stable_seed(seed, "pg-sensitivity", p_geom),
        backend=runtime.get("backend"),
        iterations=iterations,
        return_samples=True,
        move_window=runtime.get("move_window"),
        theta_prop_sigma=runtime.get("theta_prop_sigma"),
        global_move_prob=runtime.get("global_move_prob"),
        max_k=runtime.get("max_k"),
        u_prop_sigma=runtime.get("u_prop_sigma"),
        summary_method="mode",
    )
    k_draws = result.get("samples", {}).get("k", [])
    import numpy as np
    unique_k, k_counts = np.unique(k_draws, return_counts=True)
    k_posterior = {
        int(k): float(count / len(k_draws))
        for k, count in zip(unique_k, k_counts)
    } if len(k_draws) else {}
    record = {
        "p_geom": p_geom,
        "runtime_seconds": time.perf_counter() - started,
        "k_est": result.get("k_est") if isinstance(result, dict) else None,
        "taus_est": result.get("taus_est") if isinstance(result, dict) else None,
        "k_posterior": k_posterior,
        "max_pip": float(np.max(result.get("pip_array", [np.nan]))),
        "result": result,
    }
    atomic_pickle_dump(record, checkpoint)
    return record


def _csv_text(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "\n"
    fields = sorted({key for row in rows for key in row})
    output = []
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def run_pg_sensitivity(data: dict[str, Any], runtime: dict[str, Any], args: argparse.Namespace) -> None:
    plt = _configure_plotting()

    values = [float(item.strip()) for item in args.pg_values.split(",") if item.strip()]
    root = Path(runtime["realdata_root"]) / "pg_sensitivity"
    root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (p_geom, data, runtime, args.pg_iterations, args.seed, args.smoke, root / f"p_{p_geom:g}.pkl", args.resume)
        for p_geom in values
    ]
    pending = [task for task in tasks if not (args.resume and Path(task[-2]).exists())]
    run_parallel(pending, _pg_worker, args.workers, "p_g sensitivity")
    records = [load_pickle(task[-2]) for task in tasks if Path(task[-2]).exists()]
    rows = [
        {
            "p_geom": record["p_geom"],
            "runtime_seconds": record["runtime_seconds"],
            "k_est": record["k_est"],
            "taus_est": record["taus_est"],
            "k_posterior": record["k_posterior"],
            "max_pip": record["max_pip"],
        }
        for record in records
    ]
    atomic_text_write(_csv_text(rows), root / "p_g_sensitivity.csv")
    if rows:
        import numpy as np

        p_values = np.asarray([record["p_geom"] for record in records], dtype=float)
        locations = [np.sort(np.asarray(record["taus_est"], dtype=float)) for record in records]
        pip_matrix = np.stack(
            [np.asarray(record["result"]["pip_array"], dtype=float) for record in records]
        )

        max_pip = np.asarray([record["max_pip"] for record in records], dtype=float)
        k_cap = max(
            (max(record["k_posterior"], default=0) for record in records),
            default=0,
        )
        cap_mass = np.asarray(
            [record["k_posterior"].get(k_cap, 0.0) for record in records],
            dtype=float,
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.9, 1]})
        heatmap = axes[0].imshow(
            pip_matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            cmap="viridis",
            vmin=0.0,
            vmax=max(0.25, float(np.nanmax(pip_matrix))),
            extent=(1, pip_matrix.shape[1], -0.5, len(p_values) - 0.5),
        )
        for row_idx, item in enumerate(locations):
            axes[0].scatter(
                item + 1,
                np.full(item.shape, row_idx),
                s=26,
                facecolors="none",
                edgecolors="white",
                linewidths=0.8,
            )
        axes[0].set_xlabel(r"Time index $t$", fontsize=14, fontweight="bold")
        axes[0].set_ylabel(r"Geometric-prior parameter $p_g$", fontsize=14, fontweight="bold")
        axes[0].set_title("(a) Location-wise posterior inclusion probability", fontsize=15, fontweight="bold")
        axes[0].set_yticks(np.arange(len(p_values)), [f"{value:.1f}" for value in p_values])
        colorbar = fig.colorbar(heatmap, ax=axes[0], fraction=0.035, pad=0.02)
        colorbar.set_label("Posterior inclusion probability", fontsize=13, fontweight="bold")
        colorbar.ax.tick_params(labelsize=12, width=1.5)
        for label in colorbar.ax.get_yticklabels():
            label.set_fontweight("bold")
        _style_publication_axis(axes[0])

        axes[1].plot(p_values, max_pip, color="#C44E52", marker="o", label="Maximum PIP")
        axes[1].plot(
            p_values,
            cap_mass,
            color="#4C72B0",
            marker="s",
            linestyle="--",
            label=rf"$\Pr(K={k_cap}\mid\mathcal{{D}})$",
        )
        axes[1].set_xlabel(r"Geometric-prior parameter $p_g$", fontsize=14, fontweight="bold")
        axes[1].set_ylabel("Posterior probability", fontsize=14, fontweight="bold")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_xticks(p_values)
        axes[1].set_title("(b) Inclusion and model-size summaries", fontsize=15, fontweight="bold")
        legend = axes[1].legend(frameon=False, fontsize=11)
        for text_item in legend.get_texts():
            text_item.set_fontweight("bold")
        _style_publication_axis(axes[1])
        fig.tight_layout()
        fig.savefig(Path(runtime["plots_dir"]) / "p_g_sensitivity.png", dpi=300, bbox_inches="tight")
        fig.savefig(Path(runtime["plots_dir"]) / "p_g_sensitivity.pdf", bbox_inches="tight")
        plt.close(fig)


def _save_events(runtime: dict[str, Any]):
    import pandas as pd

    event_path = Path(runtime["realdata_root"]) / "events.csv"
    events_df = pd.DataFrame(EVENTS)
    atomic_text_write(events_df.to_csv(index=False), event_path)
    return event_path


def _configure_plotting():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(context="talk", style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "legend.fontsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 2.5,
            "axes.edgecolor": "black",
            "lines.linewidth": 2.5,
            "figure.figsize": (16, 10),
            "figure.dpi": 300,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        }
    )
    return plt


def _style_publication_axis(ax) -> None:
    """Match the manuscript's high-contrast tick and border styling."""
    ax.tick_params(axis="both", which="major", labelsize=13, width=1.5)
    for label in (*ax.get_xticklabels(), *ax.get_yticklabels()):
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)


def _plot_waves(df, plots_dir: Path, show: bool):
    import matplotlib.dates as mdates

    plt = _configure_plotting()
    from datetime import timedelta

    fig, ax1 = plt.subplots(figsize=(18, 10))
    ax1.set_xlabel("Date", fontsize=20)
    ax1.set_ylabel("Confirmed Cases (7-day avg)", color="tab:blue", fontsize=20)
    ax1.plot(df["date"], df["ct"].rolling(7).mean(), color="tab:blue", label="Confirmed Cases (7-day avg)", linewidth=3)
    ax1.tick_params(axis="y", labelcolor="tab:blue", labelsize=16)
    ax1.tick_params(axis="x", labelsize=16)
    ax2 = ax1.twinx()
    ax2.set_ylabel("Deaths (7-day avg)", color="tab:red", fontsize=20)
    ax2.plot(df["date"], df["dt"].rolling(7).mean(), color="tab:red", label="Deaths (7-day avg)", linewidth=3)
    ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=16)
    waves = [
        ("Wave 1", datetime(2020, 2, 1), datetime(2020, 6, 15)),
        ("Wave 2", datetime(2020, 6, 16), datetime(2020, 10, 15)),
        ("Wave 3", datetime(2020, 10, 16), datetime(2021, 2, 28)),
        ("Wave 4 - Alpha", datetime(2021, 3, 1), datetime(2021, 6, 15)),
        ("Wave 5 - Delta", datetime(2021, 6, 16), datetime(2021, 12, 15)),
    ]
    for label, start, end in waves:
        ax1.axvline(start, color="black", linestyle="--", linewidth=2.5, alpha=0.8)
        ax1.annotate(label, xy=(start + (end - start) / 2, 0.99), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=12, fontweight="bold")
    ax1.axvline(datetime(2021, 12, 16), color="black", linestyle="--", linewidth=2.5, alpha=0.8)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    fig.suptitle("Japan: Daily Confirmed Cases and Deaths", fontsize=24, y=0.96, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(plots_dir / "japan_cases_and_deaths.pdf", dpi=400, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def _plot_comparison(df, results_dict, rtacfr_signal, plots_dir: Path, show: bool):
    import matplotlib.dates as mdates

    plt = _configure_plotting()
    rjmcmc = results_dict["RJMCMC"]
    pelt = results_dict["PELT"]
    binseg = results_dict["BinSeg"]
    fig, axes = plt.subplots(3, 1, figsize=(20, 24), sharex=True)
    fig.suptitle("Comparison of CFR Estimation Methods for the Japan Dataset", fontsize=26, y=0.925, fontweight="bold")
    time_axis = df["date"]

    def clean_legend(ax):
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=14, frameon=True, fancybox=True, shadow=True)

    ax = axes[0]
    ax.plot(time_axis, rjmcmc["p_t_hat"], color="dodgerblue", label="RJMCMC posterior mean", linewidth=3)
    ax.fill_between(time_axis, rjmcmc["p_t_lower_ci"], rjmcmc["p_t_upper_ci"], color="skyblue", alpha=0.4, label="RJMCMC 95% credible interval")
    ax.plot(time_axis, rtacfr_signal, color="crimson", linestyle="--", lw=3, label="rtaCFR")
    for i, cp_idx in enumerate(rjmcmc["taus_est"]):
        ax.axvline(x=df["date"].iloc[cp_idx], color="dodgerblue", linestyle=":", lw=3, label="RJMCMC changepoints" if i == 0 else "")
    ax.set_title("(a) RJMCMC and rtaCFR", fontsize=22, fontweight="bold")
    ax.set_ylabel("Case Fatality Rate", fontsize=22)
    clean_legend(ax)

    for ax, result, color, title, label in [
        (axes[1], pelt, "darkorange", "(b) rtaCFR + PELT", "PELT changepoints"),
        (axes[2], binseg, "purple", "(c) rtaCFR + BinSeg", "BinSeg changepoints"),
    ]:
        ax.plot(time_axis, result["p_t_hat"], color=color, label=title[4:], linewidth=3)
        for i, cp_idx in enumerate(result["taus_est"]):
            ax.axvline(x=df["date"].iloc[cp_idx], color=color, linestyle=":", lw=3, label=label if i == 0 else "")
        ax.set_title(title, fontsize=22, fontweight="bold")
        ax.set_ylabel("Case Fatality Rate", fontsize=22)
        clean_legend(ax)
    axes[2].set_xlabel("Date", fontsize=22)
    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", which="major", pad=15, labelsize=18)
        ax.tick_params(axis="x", which="minor", labelsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(plots_dir / "japan_cfr_comparison.pdf", dpi=400, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    from config import REAL_DATA_PROPOSAL_THETA_SIGMA

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(SCRIPT_DIR / "JP_Data.csv"))
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--backend", default=None, help="Optional RJMCMC backend forwarded when supported")
    parser.add_argument("--move-window", type=int, default=None, help="Optional changepoint relocation half-window")
    parser.add_argument("--theta-proposal-sigma", type=float, default=REAL_DATA_PROPOSAL_THETA_SIGMA, help="Random-walk standard deviation for latent CFR updates")
    parser.add_argument("--global-move-probability", type=float, default=0.0, help="Probability that a relocation proposal uses the full interval between neighboring changepoints")
    parser.add_argument("--max-changepoints", type=int, default=None, help="Optional real-data RJMCMC changepoint cap")
    parser.add_argument("--split-proposal-sigma", type=float, default=None, help="Optional standard deviation for birth/death auxiliary splits")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--plots-dir", default="plots")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-rmse", action="store_true")
    parser.add_argument("--reviewer-jobs", action="store_true", help="Run optional multi-chain diagnostics and p_g sensitivity")
    parser.add_argument("--reviewer-only", action="store_true", help="Run reviewer diagnostics/p_g jobs without benchmark fits")
    parser.add_argument("--diagnostics-only", action="store_true", help="Run multi-chain diagnostics without benchmark or p_g sensitivity jobs")
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--chain-iterations", type=int, default=None)
    parser.add_argument("--chain-burn-in", type=int, default=None, help="Optional burn-in length for reviewer diagnostic chains")
    parser.add_argument("--pg-sensitivity", action="store_true")
    parser.add_argument("--pg-values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--pg-iterations", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use 60 rows and tiny MCMC settings; skips RMSE")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.workers < 1 or args.n_splits < 2:
        raise SystemExit("--workers must be >= 1 and --n-splits must be >= 2")
    if args.chains < 2 or args.chains > 4:
        raise SystemExit("--chains must be between 2 and 4")
    if args.reviewer_jobs:
        args.pg_sensitivity = True
    if args.reviewer_only:
        args.reviewer_jobs = True
        args.pg_sensitivity = True
    if args.smoke:
        args.skip_rmse = True
    runtime = _runtime_paths(args)
    _configure_project(runtime, args.smoke)
    Path(runtime["realdata_root"]).mkdir(parents=True, exist_ok=True)
    Path(runtime["plots_dir"]).mkdir(parents=True, exist_ok=True)

    df, data = _load_data(Path(args.data), args.smoke)
    if args.diagnostics_only:
        run_reviewer_diagnostics(data, runtime, args)
        return
    if args.reviewer_only:
        run_reviewer_diagnostics(data, runtime, args)
        run_pg_sensitivity(data, runtime, args)
        return
    _plot_waves(df, Path(runtime["plots_dir"]), args.show)
    print("Running methods on Japan data...")
    rtacfr_signal = _fit_method("RTACFR", data, runtime, args)
    results_dict = {}
    for method in ("RJMCMC", "PELT", "BinSeg"):
        results_dict[method] = _fit_method(method, data, runtime, args)

    import numpy as np
    import pandas as pd
    from config import DELAY_DIST
    from evaluation_realdata import calculate_bic, calculate_hausdorff_alignment, load_event_list

    bic_scores = {
        name: calculate_bic(data["deaths"], data["cases"], result["p_t_hat"], result["k_est"], DELAY_DIST)
        for name, result in results_dict.items()
    }
    rmse_scores: dict[str, float] = {}
    if not args.skip_rmse:
        rmse_tasks = [
            (method, data, {**runtime, "signal_cache_dir": runtime["realdata_root"] / f"rmse_cache_{method.lower()}"}, args.n_splits, args.seed, args.smoke)
            for method in ("RJMCMC", "PELT", "BinSeg")
        ]
        rmse_paths = {method: Path(runtime["realdata_root"]) / f"rmse_{method.lower()}.pkl" for method in ("RJMCMC", "PELT", "BinSeg")}
        pending = [task for task in rmse_tasks if not (args.resume and rmse_paths[task[0]].exists())]
        computed = run_parallel(pending, _rmse_worker, args.workers, "Out-of-sample RMSE")
        for method, value in computed:
            atomic_pickle_dump(value, rmse_paths[method])
        for method, path in rmse_paths.items():
            if path.exists():
                rmse_scores[method] = float(load_pickle(path))

    event_path = _save_events(runtime)
    events = load_event_list(event_path, df["date"].iloc[0])
    event_indices = events["time_idx"].tolist()
    hausdorff_scores = {
        name: calculate_hausdorff_alignment(np.array(result["taus_est"]).tolist(), event_indices)
        for name, result in results_dict.items()
    }
    summary_data = {"BIC": pd.Series(bic_scores), "RMSE": pd.Series(rmse_scores), "HD": pd.Series(hausdorff_scores)}
    summary_df = pd.DataFrame(summary_data)
    print("\n--- Quantitative Evaluation Summary ---")
    print(summary_df.round(3))
    csv_path = Path(runtime["plots_dir"]) / "real_data_evaluation_summary.csv"
    latex_path = Path(runtime["plots_dir"]) / "real_data_evaluation_summary.tex"
    summary_df.round(3).to_csv(csv_path)
    atomic_text_write(
        summary_df.round(3).to_latex(caption="Quantitative Evaluation Summary for Japan Data.", label="tab:jp_results", float_format="%.3f"),
        latex_path,
    )
    _plot_comparison(df, results_dict, rtacfr_signal, Path(runtime["plots_dir"]), args.show)
    if args.reviewer_jobs:
        run_reviewer_diagnostics(data, runtime, args)
    if args.pg_sensitivity:
        run_pg_sensitivity(data, runtime, args)


if __name__ == "__main__":
    mp.freeze_support()
    main()
