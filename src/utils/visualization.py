import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os

# Set global style
sns.set_theme(style="darkgrid")
plt.rcParams['figure.dpi'] = 150

def plot_training_curves(metrics_csv_path: str, save_path: str):
    """Plot Q-values, actor loss, and critic loss over training steps."""
    if not os.path.exists(metrics_csv_path):
        return
        
    df = pd.read_csv(metrics_csv_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Check if 'q_dataset_mean' exists, otherwise look for general 'q_values' or similar
    q_col = 'q_dataset_mean' if 'q_dataset_mean' in df.columns else (
        'value_loss' if 'value_loss' in df.columns else None
    )
    
    if q_col:
        sns.lineplot(data=df, x='step', y=q_col, ax=axes[0])
        axes[0].set_title('Q-Values (or Value Loss)')
        
    if 'actor_loss' in df.columns:
        sns.lineplot(data=df, x='step', y='actor_loss', ax=axes[1])
        axes[1].set_title('Actor Loss')
        
    if 'critic_loss' in df.columns:
        sns.lineplot(data=df, x='step', y='critic_loss', ax=axes[2])
        axes[2].set_title('Critic/Q Loss')
    elif 'q_loss' in df.columns:
        sns.lineplot(data=df, x='step', y='q_loss', ax=axes[2])
        axes[2].set_title('Q Loss')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_failure_curves(degradation_results: List[Dict[str, Any]], save_path: str):
    """
    Plot normalized score vs degradation level.
    Expects list of dicts with: 'algorithm', 'env', 'protocol', 'level', 'score'
    """
    if not degradation_results:
        return
        
    df = pd.DataFrame(degradation_results)
    
    # Ensure numeric levels where applicable for plotting
    protocols = df['protocol'].unique()
    
    fig, axes = plt.subplots(1, len(protocols), figsize=(5 * len(protocols), 5))
    if len(protocols) == 1:
        axes = [axes]
        
    for ax, protocol in zip(axes, protocols):
        prot_df = df[df['protocol'] == protocol]
        
        # Sort levels if possible (e.g., '10', '25', '50' for coverage)
        # It's better to sort based on the numerical value if it is a string representing a number
        # For quality it's Q0, Q25...
        
        sns.lineplot(
            data=prot_df, 
            x='level', 
            y='score', 
            hue='algorithm', 
            style='env',
            markers=True,
            dashes=False,
            ax=ax
        )
        ax.set_title(f'{protocol.capitalize()} Degradation')
        ax.set_ylabel('Normalized Score')
        ax.set_xlabel('Degradation Level')
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_q_overestimation(q_diag_results: List[Dict[str, Any]], save_path: str):
    """Plot Q(s,a_random) vs Q(s,a_data) over training."""
    pass  # Implemented if specific q_diag_results structure is provided

def plot_predictor_correlation(predicted_scores: List[float], actual_scores: List[float], labels: List[str], spearman_rho: float, save_path: str):
    """Scatter plot predicted vs actual scores."""
    plt.figure(figsize=(6, 6))
    plt.scatter(actual_scores, predicted_scores, alpha=0.7)
    
    # Plot y=x line
    min_val = min(min(actual_scores), min(predicted_scores))
    max_val = max(max(actual_scores), max(predicted_scores))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(f'Failure Predictor Correlation\nSpearman ρ = {spearman_rho:.3f}')
    plt.xlabel('Actual Normalized Score')
    plt.ylabel('Predicted Normalized Score')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_dataset_statistics_heatmap(stats_df: pd.DataFrame, save_path: str):
    """Heatmap of dataset statistics across dataset variants."""
    plt.figure(figsize=(10, 8))
    # Normalize stats for heatmap visibility if needed, or just plot raw
    # Assuming stats_df index is dataset variants and columns are stats
    sns.heatmap(stats_df, annot=True, cmap="YlGnBu", fmt=".2f")
    plt.title('Dataset Statistics Heatmap')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
