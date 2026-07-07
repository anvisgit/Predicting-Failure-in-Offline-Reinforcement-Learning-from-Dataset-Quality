#!/usr/bin/env python
<<<<<<< HEAD
"""
Analyze and visualize all experiment results.
Generates publication-quality figures.

Usage:
    python scripts/analyze_results.py --results_dir results --output_dir figures
"""
=======
>>>>>>> 61a721dcb2dba975feffcf589db14be640cebc1b
import argparse
import os
import sys
from pathlib import Path
import json
import pandas as pd
<<<<<<< HEAD

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.visualization import plot_failure_curves, plot_dataset_statistics_heatmap
from src.utils.helpers import ensure_dir

=======
import glob


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from src.utils.visualization import plot_failure_curves, plot_dataset_statistics_heatmap
from src.utils.helpers import ensure_dir
>>>>>>> 61a721dcb2dba975feffcf589db14be640cebc1b
def parse_args():
    parser = argparse.ArgumentParser(description='Analyze and visualize results')
    parser.add_argument('--results_dir', type=str, default='results')
    parser.add_argument('--output_dir', type=str, default='figures')
    return parser.parse_args()

def load_degradation_results(results_dir):
<<<<<<< HEAD
    """Load all JSON results from degradation study."""
    import glob
=======
>>>>>>> 61a721dcb2dba975feffcf589db14be640cebc1b
    data = []
    
    pattern = os.path.join(results_dir, 'degradation', '**', 'results.json')
    for filepath in glob.glob(pattern, recursive=True):
        try:
            with open(filepath, 'r') as f:
                res = json.load(f)
<<<<<<< HEAD
                
            # Flatten dict
            flat = {
                'algorithm': res.get('algorithm', 'unknown').upper(),
                'env': res.get('env', 'unknown'),
                'protocol': res.get('protocol', 'unknown'),
                'level': res.get('level', 'unknown'),
                'score': res.get('final_normalized_score', 0.0)
            }
=======
            flat = {'algorithm': res.get('algorithm', 'unknown').upper(),'env': res.get('env', 'unknown'),'protocol': res.get('protocol', 'unknown'),'level': res.get('level', 'unknown'),'score': res.get('final_normalized_score', 0.0)}
>>>>>>> 61a721dcb2dba975feffcf589db14be640cebc1b
            if 'dataset_stats' in res:
                for k, v in res['dataset_stats'].items():
                    flat[f'stat_{k}'] = v
            data.append(flat)
        except Exception as e:
<<<<<<< HEAD
            print(f"Error reading {filepath}: {e}")
            
=======
            print(f"Error reading {filepath}: {e}")   
>>>>>>> 61a721dcb2dba975feffcf589db14be640cebc1b
    return data

def main():
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    
    # 1. Plot Failure Curves from Degradation Study
    print("Loading degradation results...")
    deg_data = load_degradation_results(args.results_dir)
<<<<<<< HEAD
    
=======
>>>>>>> 61a721dcb2dba975feffcf589db14be640cebc1b
    if deg_data:
        print("Plotting failure curves...")
        plot_failure_curves(deg_data, os.path.join(out_dir, 'failure_curves.png'))
        
        # 2. Dataset statistics heatmap
        print("Plotting dataset statistics...")
        df = pd.DataFrame(deg_data)
        stat_cols = [c for c in df.columns if c.startswith('stat_')]
        if stat_cols:
            # Aggregate stats by protocol/level
            agg_stats = df.groupby(['protocol', 'level'])[stat_cols].mean()
            # Clean column names for plot
            agg_stats.columns = [c.replace('stat_', '') for c in agg_stats.columns]
            plot_dataset_statistics_heatmap(agg_stats, os.path.join(out_dir, 'dataset_stats_heatmap.png'))
    else:
        print("No degradation results found. Run Stage 2 first.")
        
    print(f"Analysis complete. Figures saved to {out_dir}/")

if __name__ == '__main__':
    main()
