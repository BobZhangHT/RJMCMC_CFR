# analysis.py

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import RESULTS_DIR, PLOTS_DIR, SCENARIOS

def hausdorff_distance(A, B):
    """Calculates the Hausdorff distance between two sets of points."""
    if not A and not B: return 0.0
    if not A or not B: return np.inf
    
    A, B = np.array(A), np.array(B)
    term1 = np.max([np.min(np.abs(a - B)) for a in A])
    term2 = np.max([np.min(np.abs(b - A)) for b in B])
    return max(term1, term2)

def analyze_results():
    """Loads all simulation results, computes metrics, and generates plots."""
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
        
    all_results = []
    result_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".pkl")]

    for fname in result_files:
        try:
            parts = fname.replace('.pkl', '').split('_')
            method = parts[-1]
            rep_idx = int(parts[-2].replace('rep', ''))
            # Reconstruct scenario name which might contain spaces
            scenario_name = ' '.join(parts[1:-2])
            
            with open(os.path.join(RESULTS_DIR, fname), 'rb') as f:
                res = pickle.load(f)
            
            k_true = len(res.get('true_cps', []))
            k_est = res.get('k', 0)
            
            all_results.append({
                "scenario": scenario_name,
                "replication": rep_idx,
                "method": method,
                "k_true": k_true,
                "k_est": k_est,
                "k_error": abs(k_est - k_true),
                "hausdorff": hausdorff_distance(res.get('true_cps', []), res.get('taus', []))
            })
        except (IndexError, ValueError) as e:
            print(f"Could not parse filename: {fname}. Error: {e}")
            continue
        
    if not all_results:
        print("No result files found or parsed. Exiting analysis.")
        return

    df = pd.DataFrame(all_results)
    
    # Save detailed results table
    df.to_csv(os.path.join(PLOTS_DIR, "full_simulation_results.csv"), index=False)
    
    # --- Generate Summary Tables ---
    summary = df.groupby(['scenario', 'method']).agg(
        mean_k_error=('k_error', 'mean'),
        mean_hausdorff=('hausdorff', 'mean'),
        k_accuracy=('k_error', lambda x: np.mean(x == 0))
    ).reset_index()
    
    print("\n--- Summary Results ---")
    print(summary)
    summary.to_csv(os.path.join(PLOTS_DIR, "summary_results.csv"), index=False)
    
    # --- Generate Plots ---
    sns.set_theme(style="whitegrid")
    for scenario_name in SCENARIOS.keys():
        scen_df = df[df['scenario'] == scenario_name]
        
        if scen_df.empty:
            print(f"No data found for scenario: {scenario_name}. Skipping plot.")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Performance for: {scenario_name}", fontsize=16)
        
        # Boxplot for Hausdorff distance
        sns.boxplot(ax=axes[0], data=scen_df, x='method', y='hausdorff', palette='viridis')
        axes[0].set_title("Hausdorff Distance (Lower is Better)")
        axes[0].set_xlabel("Method")
        axes[0].set_ylabel("Distance")
        axes[0].set_yscale('log') # Log scale is often better for Hausdorff
        
        # Bar plot for k accuracy (P(k_est == k_true))
        k_acc_df = summary[summary['scenario'] == scenario_name]
        sns.barplot(ax=axes[1], data=k_acc_df, x='method', y='k_accuracy', palette='viridis')
        axes[1].set_title("Accuracy of Detected # of CPs")
        axes[1].set_xlabel("Method")
        axes[1].set_ylabel("Proportion Correct")
        axes[1].set_ylim(0, 1.05)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(os.path.join(PLOTS_DIR, f"plot_{scenario_name.replace(' ', '_')}.png"))
        plt.close()

    print(f"\nAnalysis complete. Tables and plots saved to '{PLOTS_DIR}' directory.")

if __name__ == "__main__":
    analyze_results()
