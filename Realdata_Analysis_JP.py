"""Runnable version of ``Realdata_Analysis_JP.ipynb`` with resumable outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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
    atomic_json_dump,
    rank_diagnostics,
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
        "chain_storage_stride": args.chain_storage_stride,
        "chain_seed": args.chain_seed,
        "chain_seed_family": "reviewer-original-long-confirmation",
        "data_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
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
    import numpy as np
    import importlib.metadata
    checkpoint = Path(checkpoint)
    _configure_project(runtime, smoke)
    from config import REAL_DATA_PRIOR_K_GEOMETRIC_P, REAL_DATA_PRIOR_THETA_SIGMA
    call_seed = stable_seed(runtime['chain_seed'], runtime['chain_seed_family'], chain_idx)
    signature = dict(chain=chain_idx, iterations=chain_iterations,
        burn_in=runtime['chain_burn_in'], storage_stride=runtime['chain_storage_stride'],
        seed=call_seed, initial_state=initial_state, data_sha256=runtime['data_sha256'],
        backend=runtime.get('backend') or 'numba-fast', theta_proposal_sigma=runtime['theta_prop_sigma'],
        move_window=runtime.get('move_window'), global_move_probability=runtime['global_move_prob'],
        max_k=runtime.get('max_k'), split_sigma=runtime.get('u_prop_sigma'),
        p_geom=REAL_DATA_PRIOR_K_GEOMETRIC_P, theta_prior_sigma=REAL_DATA_PRIOR_THETA_SIGMA,
        sources={name: hashlib.sha256((SCRIPT_DIR/name).read_bytes()).hexdigest()
                 for name in ['methods.py','config.py','workflow_runtime.py','Realdata_Analysis_JP.py']},
        software={name: importlib.metadata.version(name) for name in ['numpy','scipy','numba','arviz','arviz-stats']})
    signature = json.loads(json.dumps(signature, default=lambda x: x.item() if hasattr(x,'item') else str(x)))
    meta_path = checkpoint.with_suffix('.json')
    if resume and checkpoint.exists():
        if not meta_path.exists():
            raise ValueError(f'Legacy checkpoint has no R2 provenance: {checkpoint}; use a new --results-root')
        meta=json.loads(meta_path.read_text())
        if meta['signature'] != signature or meta['checkpoint_sha256'] != hashlib.sha256(checkpoint.read_bytes()).hexdigest():
            raise ValueError(f'Checkpoint settings or contents differ: {checkpoint}; use a new --results-root')
        return f'cached: {checkpoint}'
    started = time.perf_counter()
    result = call_rjmcmc(load_rjmcmc(), data,
        p_geom=REAL_DATA_PRIOR_K_GEOMETRIC_P, theta_sigma=REAL_DATA_PRIOR_THETA_SIGMA,
        seed=call_seed, backend=signature['backend'], iterations=chain_iterations,
        burn_in=runtime['chain_burn_in'], storage_stride=runtime['chain_storage_stride'],
        initial_state=initial_state, return_samples=True, diagnostics=True,
        move_window=runtime.get('move_window'), theta_prop_sigma=runtime['theta_prop_sigma'],
        global_move_prob=runtime['global_move_prob'], max_k=runtime.get('max_k'),
        u_prop_sigma=runtime.get('u_prop_sigma'), summary_method='mode')
    record=dict(chain=chain_idx, requested_initial_state=initial_state, initial_state_status='requested',
                runtime_seconds=time.perf_counter()-started, result=result, trace=_extract_trace(result), signature=signature)
    atomic_pickle_dump(record, checkpoint)
    atomic_json_dump(dict(signature=signature, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()),meta_path)
    return f'completed: {checkpoint}'


def _normalise_trace_array(value: Any):
    import numpy as np

    array = np.asarray(value)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array.reshape(array.shape[0], -1)


def run_reviewer_diagnostics(data: dict[str, Any], runtime: dict[str, Any], args: argparse.Namespace) -> None:
    """Run the original dispersed chains with the R2 lengths and storage schedule."""
    import numpy as np
    import pandas as pd
    from config import K_MAX
    reviewer_root = Path(runtime['realdata_root'])/'reviewer'
    chain_root=reviewer_root/'chains'
    chain_root.mkdir(parents=True,exist_ok=True)
    k_max=K_MAX if runtime.get('max_k') is None else int(runtime['max_k'])
    k_max=min(3,k_max) if args.smoke else k_max
    T=len(data['cases'])
    spread=np.linspace(2,T-1,k_max,dtype=int).tolist() if k_max else []
    states=[dict(k=0,taus=[],theta_values=[-6.]),dict(k=0,taus=[],theta_values=[-1.]),
            dict(k=k_max,taus=spread,theta_values=[-6.]*(k_max+1)),
            dict(k=k_max,taus=spread,theta_values=[-1.]*(k_max+1))][:args.chains]
    seeds=[stable_seed(runtime['chain_seed'],runtime['chain_seed_family'],c) for c in range(len(states))]
    if len(set(s&0xffffffff for s in seeds)) != len(states):
        raise ValueError('Diagnostic seeds collide')
    tasks=[(c,state,data,runtime,args.chain_iterations,args.seed,args.smoke,
            chain_root/f'chain_{c:02d}.pkl',args.resume) for c,state in enumerate(states)]
    # Workers validate provenance even when a checkpoint already exists.
    run_parallel(tasks,_reviewer_chain_worker,args.workers,'Reviewer RJMCMC chains')
    records=[load_pickle(t[-2]) for t in tasks]
    expected=(args.chain_iterations+args.chain_storage_stride-1)//args.chain_storage_stride
    if any(len(r['trace']['k'])!=expected for r in records):
        raise ValueError('Unexpected stored diagnostic count')
    k=np.stack([r['trace']['k'] for r in records])[:,:,None]
    p=np.stack([r['trace']['p_t'] for r in records])
    if p.shape != (len(states),expected,T):
        raise ValueError('Unexpected daily trajectory shape')
    mean=p.mean(axis=2,keepdims=True)
    dates=pd.read_csv(args.data).sort_values('date').date.iloc[:T].tolist()
    rows,coordinates=[],[]
    all_finite=True
    for name,values in [('k',k),('p_t',p),('mean_p_t',mean)]:
        rh,bulk,tail=rank_diagnostics(values)
        finite=bool(np.all(np.isfinite(rh)) and np.all(np.isfinite(bulk)) and np.all(np.isfinite(tail)))
        all_finite=all_finite and finite
        worst=int(np.argmax(rh))
        rows.append(dict(variable=name,coordinate='all (worst)',rank_normalized_rhat=float(np.max(rh)),
                         bulk_ess=float(np.min(bulk)),tail_ess=float(np.min(tail)),
                         worst_day=worst+1 if name=='p_t' else '',
                         worst_date=dates[worst] if name=='p_t' else '',n_coordinates=len(rh),
                         n_rhat_ge_1_01=int(np.sum(rh>=1.01)),all_metrics_finite=finite))
        if name=='p_t':
            coordinates=[dict(day=j+1,date=dates[j],rank_normalized_rhat=float(rh[j]),
                              bulk_ess=float(bulk[j]),tail_ess=float(tail[j])) for j in range(T)]
    atomic_text_write(_csv_text(rows),reviewer_root/'rjmcmc_convergence.csv')
    atomic_text_write(_csv_text(coordinates),reviewer_root/'rjmcmc_coordinate_diagnostics.csv')
    passed=all_finite and all(r['rank_normalized_rhat']<1.01 and r['bulk_ess']>=400 and r['tail_ess']>=400 for r in rows)
    atomic_json_dump(dict(all_monitored_coordinates_pass=passed,monitored_coordinates=T+2,
                         smoke=args.smoke,summary=rows,all_coordinates_and_chains_included=True),
                     reviewer_root/'convergence_gate.json')
    atomic_text_write(_csv_text([dict(chain=r['chain'],runtime_seconds=r['runtime_seconds'],
                                     stored_draws=expected,seed=seeds[r['chain']]) for r in records]),
                      reviewer_root/'chain_runtime.csv')
    # Keep the original six representative days and the R2 Figure 5 layout.
    import sys
    sys.modules.pop('jax', None)
    sys.modules.pop('jaxlib', None)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=sorted({int(round(v)) for v in np.linspace(0,T-1,6)})
    series=[(r'Changepoint count $k$',k[:,:,0],r'$k$')]
    series += [(rf'Daily CFR at $t={j+1}$',p[:,:,j],'CFR') for j in selected]
    series += [('All-time mean CFR',mean[:,:,0],'Mean CFR')]
    colors=['#0F4D92','#B64342','#3A7D44','#7A5195']
    with plt.rc_context(matplotlib.rcParamsDefault):
        plt.rcParams.update({'font.size':12,'pdf.fonttype':42})
        fig,axes=plt.subplots(4,2,figsize=(10.5,9.8),layout='constrained')
        for idx,(ax,(title,values,ylabel)) in enumerate(zip(axes.flat,series)):
            display_stride=max(1,expected//5000)
            x=np.arange(expected)[::display_stride]*args.chain_storage_stride+1
            for c in range(len(states)):
                ax.plot(x,values[c,::display_stride],color=colors[c],linewidth=.6,alpha=.65,label=f'Chain {c+1}')
            ax.set(title=f'({chr(97+idx)}) {title}',ylabel=ylabel,xlabel='Post-burn-in iteration')
            ax.ticklabel_format(style='sci',axis='x',scilimits=(0,0))
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=4,fontsize=11,frameon=False,loc='upper center',bbox_to_anchor=(.5,1.035))
        fig.savefig(Path(runtime['plots_dir'])/'reviewer_rjmcmc_trace.pdf',bbox_inches='tight')
        fig.savefig(Path(runtime['plots_dir'])/'reviewer_rjmcmc_trace.png',dpi=180,bbox_inches='tight')
        plt.close(fig)
    print(json.dumps(dict(convergence_gate=passed,summary=rows),indent=2))


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


def _pg_reference_records(path: Path) -> list[dict[str, Any]]:
    import json
    import numpy as np

    reference = json.loads(path.read_text(encoding="utf-8-sig"))["figure6"]
    fits = reference["fits"]
    cap, days = reference["cap"], reference["days"]
    if (cap != 10 or days != 700
            or [fit["p_g"] for fit in fits] != [0.1, 0.3, 0.5, 0.7, 0.9]):
        raise ValueError("Figure 6 reference requires the five original priors, 700 days, and cap 10.")
    records = []
    for fit in fits:
        pip = np.asarray(fit["pip"], dtype=float)
        locations = np.asarray(fit["modal_locations"])
        if (pip.shape != (days,) or not np.all((0 <= pip) & (pip <= 1))
                or fit["max_pip"] != float(pip.max())
                or not 0 <= fit["cap_mass"] <= 1
                or locations.shape != (cap,) or locations.dtype.kind not in "iu"
                or locations[0] < 0 or locations[-1] >= days or np.any(np.diff(locations) <= 0)):
            raise ValueError(f"Invalid Figure 6 plotting values for p_g={fit['p_g']}.")
        records.append({
            "p_geom": fit["p_g"], "runtime_seconds": None,
            "k_est": len(locations), "taus_est": locations.tolist(),
            "k_posterior": {cap: fit["cap_mass"]}, "max_pip": fit["max_pip"],
            "result": {"pip_array": pip},
        })
    return records


def run_pg_sensitivity(data: dict[str, Any], runtime: dict[str, Any], args: argparse.Namespace) -> None:
    plt = _configure_plotting()
    root = Path(runtime["realdata_root"]) / "pg_sensitivity"
    root.mkdir(parents=True, exist_ok=True)
    Path(runtime["plots_dir"]).mkdir(parents=True, exist_ok=True)
    if args.pg_reference is not None:
        records = _pg_reference_records(Path(args.pg_reference))
        print("Drawing Figure 6 from the historical R1 sensitivity snapshot.")
    else:
        values = [float(item.strip()) for item in args.pg_values.split(",") if item.strip()]
        tasks = [
            (p_geom, data, runtime, args.pg_iterations, args.seed, args.smoke,
             root / f"p_{p_geom:g}.pkl", args.resume)
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
    if args.pg_reference is not None:
        for row in rows:
            row["cap_mass"] = row.pop("k_posterior")[10]
            row["source"] = "Historical R1 sensitivity snapshot for the R2 redraw"
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
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
            extent=(1, pip_matrix.shape[1], -0.5, len(p_values) - 0.5),
        )
        for row_idx, item in enumerate(locations):
            axes[0].scatter(
                item + 1,
                np.full(item.shape, row_idx),
                s=26,
                facecolors="white",
                edgecolors="black",
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
    parser.add_argument("--chain-storage-stride", type=int, default=None, help="Save every Nth diagnostic state; default 200 for R2")
    parser.add_argument("--chain-seed", type=int, default=20261121, help="Independent R2 diagnostic seed; benchmark --seed is unchanged")
    parser.add_argument("--pg-sensitivity", action="store_true")
    parser.add_argument("--pg-reference", type=Path, default=None,
                        help="Redraw Figure 6 from the historical R1 snapshot in reference_r2.json")
    parser.add_argument("--pg-values", default="0.1,0.3,0.5,0.7,0.9")
    parser.add_argument("--pg-iterations", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use 60 rows and tiny MCMC settings; skips RMSE")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.pg_reference is not None and not (args.reviewer_jobs or args.reviewer_only or args.diagnostics_only):
        run_pg_sensitivity({}, {
            "realdata_root": ensure_ignored_output(args.results_root, "results") / "realdata",
            "plots_dir": ensure_ignored_output(args.plots_dir, "plots"),
        }, args)
        return
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
    if args.chain_iterations is None:
        args.chain_iterations = 24 if args.smoke else 20000000
    if args.chain_burn_in is None:
        args.chain_burn_in = 8 if args.smoke else 2000000
    if args.chain_storage_stride is None:
        args.chain_storage_stride = 1 if args.smoke else 200
    if args.chain_iterations < 4 or args.chain_burn_in < 0 or args.chain_storage_stride < 1:
        raise SystemExit("Invalid diagnostic iteration, burn-in or storage settings")
    if (args.chain_iterations + args.chain_storage_stride - 1) // args.chain_storage_stride < 4:
        raise SystemExit("At least four stored states per chain are required")
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
