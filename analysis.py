# analysis.py

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from config import RESULTS_DIR, PLOTS_DIR, SCENARIOS, T, N_REPLICATIONS

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

def hausdorff_distance(A, B):
    """Calculates the Hausdorff distance between two sets of points."""
    if not A and not B: return 0.0
    if not A or not B: return np.inf
    A, B = np.array(A), np.array(B)
    term1 = np.max([np.min(np.abs(a - B)) for a in A])
    term2 = np.max([np.min(np.abs(b - A)) for b in B])
    return max(term1, term2)

def mean_absolute_error(p_hat, p_true):
    """Calculates the Mean Absolute Error (MAE) for the CFR curve."""
    return np.mean(np.abs(p_hat - p_true))

# ==============================================================================
# ANALYSIS AND PLOTTING
# ==============================================================================

def analyze_all_results():
    """Loads all simulation results, computes metrics, and generates outputs."""
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    all_results_list = []
    
    # --- Load all data from checkpoint files ---
    print("Loading results from checkpoint files...")
    for scenario_name in SCENARIOS.keys():
        for rep_idx in range(N_REPLICATIONS):
            for method_name in ["RJMCMC", "PELT", "BinSeg"]:
                fname = f"res_{scenario_name.replace(' ', '_')}_rep{rep_idx}_{method_name}.pkl"
                fpath = os.path.join(RESULTS_DIR, fname)
                if os.path.exists(fpath):
                    with open(fpath, 'rb') as f:
                        res = pickle.load(f)
                        res['scenario'] = scenario_name
                        res['replication'] = rep_idx
                        res['method'] = method_name
                        all_results_list.append(res)

    if not all_results_list:
        print("No result files found. Run simulation first.")
        return

    df = pd.DataFrame(all_results_list)

    # --- Compute Metrics ---
    print("Computing evaluation metrics...")
    df['k_error'] = df.apply(lambda row: abs(row['k_est'] - len(row['true_cps'])), axis=1)
    df['hausdorff'] = df.apply(lambda row: hausdorff_distance(row['true_cps'], row['taus_est']), axis=1)
    df['mae_cfr'] = df.apply(lambda row: mean_absolute_error(row['p_t_hat'], row['true_p_t']), axis=1)

    # --- Generate Publication Table ---
    summary_table = df.groupby(['scenario', 'method']).agg(
        k_accuracy=('k_error', lambda x: np.mean(x == 0)),
        hausdorff_mean=('hausdorff', 'mean'),
        mae_cfr_mean=('mae_cfr', 'mean'),
    ).rename(columns={
        'k_accuracy': 'P(k_est=k_true)',
        'hausdorff_mean': 'Hausdorff',
        'mae_cfr_mean': 'MAE_CFR'
    }).round(3)
    
    print("\n--- Publication Summary Table ---")
    print(summary_table)
    summary_table.to_csv(os.path.join(PLOTS_DIR, "publication_summary_table.csv"))

    # --- Generate Publication Figure ---
    print("Generating publication figure...")
    scen_names = list(SCENARIOS.keys())
    n_scenarios = len(scen_names)
    fig, axes = plt.subplots(n_scenarios, 3, figsize=(18, 5 * n_scenarios), 
                             gridspec_kw={'width_ratios': [1, 1, 2]})
    fig.suptitle("Comprehensive Simulation Results for RJMCMC Sampler", fontsize=20, y=1.02)

    for i, scenario_name in enumerate(scen_names):
        scen_df = df[df['scenario'] == scenario_name]
        rjmcmc_scen_df = scen_df[scen_df['method'] == 'RJMCMC']
        true_cps = SCENARIOS[scenario_name]['true_cps']
        true_k = len(true_cps)

        # Column 1: Histogram of posterior k
        ax = axes[i, 0]
        if not rjmcmc_scen_df.empty:
            all_k_samples = np.concatenate(rjmcmc_scen_df['k_samples'].values)
            sns.histplot(all_k_samples, ax=ax, discrete=True, stat='probability', color='skyblue')
        ax.axvline(true_k, color='red', linestyle='--', label=f'True k = {true_k}')
        ax.set_title(f"Posterior k\n{scenario_name}")
        ax.set_xlabel("Number of Changepoints (k)")
        ax.legend()

        # Column 2: Histogram of posterior taus
        ax = axes[i, 1]
        if not rjmcmc_scen_df.empty and true_k > 0:
            all_taus_samples = []
            for _, row in rjmcmc_scen_df.iterrows():
                k_s = row['k_samples']
                t_s = row['taus_samples']
                for j in range(len(k_s)):
                    if k_s[j] > 0:
                        all_taus_samples.extend(t_s[j, :k_s[j]])
            
            sns.histplot([t for t in all_taus_samples if t != -1], ax=ax, bins=T, color='salmon')
            for cp in true_cps:
                ax.axvline(cp, color='red', linestyle='--', label=f'True CP at {cp}')
            ax.legend()
        ax.set_title(f"Posterior Changepoint Locations")
        ax.set_xlabel("Time (t)")

        # Column 3: Averaged CFR estimate
        ax = axes[i, 2]
        if not rjmcmc_scen_df.empty:
            all_p_t_samples = np.vstack(rjmcmc_scen_df['p_t_samples'].values)
            p_t_mean = np.mean(all_p_t_samples, axis=0)
            p_t_lower = np.percentile(all_p_t_samples, 2.5, axis=0)
            p_t_upper = np.percentile(all_p_t_samples, 97.5, axis=0)
            ax.plot(p_t_mean, color='dodgerblue', label='Posterior Mean Estimate')
            ax.fill_between(range(T), p_t_lower, p_t_upper, color='skyblue', alpha=0.4, label='95% Credible Interval')
        
        true_p_t_plot = scen_df['true_p_t'].iloc[0]
        ax.plot(true_p_t_plot, color='black', lw=2, label='True CFR')
        ax.set_title(f"Averaged CFR Estimate")
        ax.set_xlabel("Time (t)")
        ax.set_ylabel("Fatality Rate")
        ax.legend()
        ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig_path = os.path.join(PLOTS_DIR, "publication_figure.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"\nAnalysis complete. Table and figure saved to '{PLOTS_DIR}'.")

if __name__ == "__main__":
    analyze_all_results()
