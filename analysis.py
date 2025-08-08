# analysis.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re

from config import (MAIN_RESULTS_DIR, SENSITIVITY_RESULTS_DIR, PLOTS_DIR, SCENARIOS, T, N_REPLICATIONS, 
                    SENSITIVITY_GRID)

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def hausdorff_distance(A, B):
    """Calculates the Hausdorff distance between two sets of points."""
    A_list, B_list = list(A), list(B)
    if not A_list and not B_list: return 0.0
    if not A_list or not B_list: return np.inf
    
    A, B = np.array(A), np.array(B)
    term1 = np.max([np.min(np.abs(a - B)) for a in A])
    term2 = np.max([np.min(np.abs(b - A)) for b in B])
    return max(term1, term2)

def mean_absolute_error(p_hat, p_true):
    """Calculates the Mean Absolute Error (MAE) for the CFR curve."""
    return np.mean(np.abs(p_hat - p_true))

# ==============================================================================
# ANALYSIS FUNCTIONS
# ==============================================================================

def load_and_process_results(results_dir):
    """
    Helper function to load summary results from a specific directory.
    This version is memory-efficient and avoids loading large sample arrays.
    """
    all_results_list = []
    if not os.path.exists(results_dir):
        print(f"Warning: Results directory '{results_dir}' not found.")
        return pd.DataFrame()
        
    result_files = [f for f in os.listdir(results_dir) if f.endswith(".npz")]
    
    print(f"Loading and processing result files from '{results_dir}'...")
    for fname in tqdm(result_files):
        fpath = os.path.join(results_dir, fname)
        
        with np.load(fpath, allow_pickle=True) as data:
            # Load only the necessary summary keys, skip large sample arrays
            res = {
                'k_est': data['k_est'].item(), # .item() to convert 0-d array to scalar
                'taus_est': data['taus_est'],
                'p_t_hat': data['p_t_hat'],
                'true_cps': data['true_cps'],
                'true_p_t': data['true_p_t']
            }

        # Parse filename using simple string splitting
        metadata = {}
        clean_fname = fname.replace('.npz', '')
        parts = clean_fname.split('_')
        for part in parts:
            try:
                key, value = part.split('=', 1)
                metadata[key] = value
            except ValueError:
                pass
        
        res['scenario'] = metadata.get('scen', 'unknown').replace('-', ' ')
        res['replication'] = int(metadata.get('rep', -1))
        res['method'] = metadata.get('method', 'unknown')
        res['p_geom'] = float(metadata.get('p', np.nan))
        res['theta_sigma'] = float(metadata.get('s', np.nan))
        all_results_list.append(res)

    if not all_results_list:
        return pd.DataFrame()

    df = pd.DataFrame(all_results_list)
    df['k_error'] = df.apply(lambda row: abs(row['k_est'] - len(row['true_cps'])), axis=1)
    df['hausdorff'] = df.apply(lambda row: hausdorff_distance(row['true_cps'], row['taus_est']), axis=1)
    df['mae_cfr'] = df.apply(lambda row: mean_absolute_error(row['p_t_hat'], row['true_p_t']), axis=1)
    return df

def analyze_sensitivity_results(df):
    """
    Performs a comprehensive sensitivity analysis on the priors for k and theta
    and returns the optimal set of parameters.
    """
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

    rjmcmc_df = df[df['method'] == 'RJMCMC'].copy()
    if rjmcmc_df.empty:
        print("No RJMCMC results found for sensitivity analysis.")
        return {'p_geom': 0.5, 'theta_sigma': 1.5}

    sensitivity_agg = rjmcmc_df.groupby(['p_geom', 'theta_sigma']).agg(
        k_accuracy=('k_error', lambda x: np.mean(x == 0)),
        mae_cfr=('mae_cfr', 'mean')
    ).reset_index()
    
    print("\n--- Sensitivity Analysis Summary Table ---")
    print(sensitivity_agg)
    sensitivity_agg.to_csv(os.path.join(PLOTS_DIR, "sensitivity_analysis_table.csv"), index=False)

    # --- Determine Optimal Parameters ---
    sensitivity_agg['mae_norm'] = (sensitivity_agg['mae_cfr'] - sensitivity_agg['mae_cfr'].min()) / \
                                  (sensitivity_agg['mae_cfr'].max() - sensitivity_agg['mae_cfr'].min())
    sensitivity_agg['score'] = sensitivity_agg['k_accuracy'] - sensitivity_agg['mae_norm']
    
    best_params_row = sensitivity_agg.loc[sensitivity_agg['score'].idxmax()]
    baseline_params = {
        'p_geom': best_params_row['p_geom'],
        'theta_sigma': best_params_row['theta_sigma']
    }

    # --- Generate Sensitivity Analysis Heatmap Figure ---
    k_accuracy_pivot = sensitivity_agg.pivot(index='theta_sigma', columns='p_geom', values='k_accuracy')
    # MODIFIED: Pivot the normalized MAE for the heatmap
    mae_norm_pivot = sensitivity_agg.pivot(index='theta_sigma', columns='p_geom', values='mae_norm')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Sensitivity of RJMCMC Performance to Prior Hyperparameters", fontsize=16)
    sns.heatmap(k_accuracy_pivot, ax=axes[0], annot=True, cmap="viridis", fmt=".3f")
    axes[0].set_title("P(k_est = k_true)")
    axes[0].set_xlabel("Prior on k (Geometric p)"); axes[0].set_ylabel("Prior on theta (StDev)")
    
    # MODIFIED: Use the normalized MAE pivot table and update the title
    sns.heatmap(mae_norm_pivot, ax=axes[1], annot=True, cmap="magma_r", fmt=".3f")
    axes[1].set_title("Normalized MAE of CFR")
    axes[1].set_xlabel("Prior on k (Geometric p)"); axes[1].set_ylabel("Prior on theta (StDev)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(PLOTS_DIR, "sensitivity_analysis_heatmap.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return baseline_params

def analyze_main_results(df_main, baseline_params):
    """Analyzes the main simulation results using the optimal baseline parameters."""
    # Use np.isclose for robust floating-point comparison
    rjmcmc_mask = (df_main['method'] == 'RJMCMC') & \
                  np.isclose(df_main['p_geom'], baseline_params['p_geom']) & \
                  np.isclose(df_main['theta_sigma'], baseline_params['theta_sigma'])
    benchmark_mask = df_main['method'].isin(['PELT', 'BinSeg'])
    baseline_df = df_main[rjmcmc_mask | benchmark_mask]
    
    if baseline_df.empty:
        print("Warning: No data found for the main analysis with the specified baseline parameters.")
        return

    summary_table = baseline_df.groupby(['scenario', 'method']).agg(
        k_accuracy=('k_error', lambda x: np.mean(x == 0)),
        hausdorff_mean=('hausdorff', 'mean'),
        mae_cfr_mean=('mae_cfr', 'mean'),
    ).rename(columns={
        'k_accuracy': 'P(k_est=k_true)', 'hausdorff_mean': 'Hausdorff', 'mae_cfr_mean': 'MAE_CFR'
    }).round(3)
    
    print("\n--- Main Publication Summary Table (using optimal priors) ---")
    print(summary_table)
    summary_table.to_csv(os.path.join(PLOTS_DIR, "publication_summary_table.csv"))

    # --- Generate Main Publication Figure ---
    print("Generating main publication figure...")
    scen_names = list(SCENARIOS.keys())
    fig, axes = plt.subplots(len(scen_names), 3, figsize=(18, 5 * len(scen_names)), gridspec_kw={'width_ratios': [1, 1, 2]})
    fig.suptitle("Comprehensive Simulation Results for RJMCMC Sampler", fontsize=20, y=1.0)

    for i, scenario_name in enumerate(scen_names):
        scen_df = baseline_df[baseline_df['scenario'] == scenario_name]
        rjmcmc_scen_df = scen_df[scen_df['method'] == 'RJMCMC']
        true_cps, true_k = SCENARIOS[scenario_name]['true_cps'], len(SCENARIOS[scenario_name]['true_cps'])

        all_k_samples, all_taus_samples, all_p_t_samples_agg = [], [], []

        if not rjmcmc_scen_df.empty:
            print(f"Aggregating posterior samples for {scenario_name}...")
            for rep_idx in tqdm(rjmcmc_scen_df['replication'].unique()):
                p, s = baseline_params['p_geom'], baseline_params['theta_sigma']
                fname = f"scen={scenario_name.replace(' ', '-')}_rep={rep_idx}_method=RJMCMC_p={p}_s={s}.npz"
                fpath = os.path.join(MAIN_RESULTS_DIR, fname)
                if os.path.exists(fpath):
                    with np.load(fpath, allow_pickle=True) as data:
                        all_k_samples.extend(data['k_samples'])
                        k_s, t_s = data['k_samples'], data['taus_samples']
                        for j in range(len(k_s)):
                            if k_s[j] > 0: all_taus_samples.extend(t_s[j, :k_s[j]])
                        p_t_samples = data['p_t_samples']
                        all_p_t_samples_agg.append(np.vstack([np.mean(p_t_samples, axis=0), np.percentile(p_t_samples, 2.5, axis=0), np.percentile(p_t_samples, 97.5, axis=0)]))

        ax = axes[i, 0]
        if all_k_samples: sns.histplot(all_k_samples, ax=ax, discrete=True, stat='probability', color='skyblue')
        ax.axvline(true_k, color='red', linestyle='--', label=f'True k = {true_k}')
        ax.set_title(f"Posterior k\n{scenario_name}"); ax.set_xlabel("Number of Changepoints (k)"); ax.legend()

        ax = axes[i, 1]
        if all_taus_samples and true_k > 0:
            sns.histplot([t for t in all_taus_samples if t != -1], ax=ax, bins=T, color='salmon')
            for cp in true_cps:
                ax.axvline(cp, color='red', linestyle='--', label=f'True CP at {cp}')
            ax.legend()
        ax.set_title(f"Posterior Changepoint Locations"); ax.set_xlabel("Time (t)")

        ax = axes[i, 2]
        if all_p_t_samples_agg:
            avg_stats = np.mean(np.array(all_p_t_samples_agg), axis=0)
            ax.plot(avg_stats[0, :], color='dodgerblue', label='Posterior Mean Estimate')
            ax.fill_between(range(T), avg_stats[1, :], avg_stats[2, :], color='skyblue', alpha=0.4, label='95% Credible Interval')
        
        if not scen_df.empty: ax.plot(scen_df['true_p_t'].iloc[0], color='black', lw=2, label='True CFR')
        ax.set_title(f"Averaged CFR Estimate"); ax.set_xlabel("Time (t)"); ax.set_ylabel("Fatality Rate"); ax.legend(); ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "publication_figure.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def main_analysis_workflow():
    """The full, two-stage analysis workflow."""
    if not os.path.exists(PLOTS_DIR): os.makedirs(PLOTS_DIR)
    
    df_sens = load_and_process_results(SENSITIVITY_RESULTS_DIR)
    if df_sens.empty:
        print("No sensitivity results to analyze. Exiting.")
        return
    baseline_params = analyze_sensitivity_results(df_sens)
    
    df_main = load_and_process_results(MAIN_RESULTS_DIR)
    if df_main.empty:
        print("No main simulation results to analyze. Exiting.")
        return
    analyze_main_results(df_main, baseline_params)

if __name__ == "__main__":
    main_analysis_workflow()
