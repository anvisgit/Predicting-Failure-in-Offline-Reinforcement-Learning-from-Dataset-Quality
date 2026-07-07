import argparse
import sys
import os
from pathlib import Path
import json
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.prediction.failure_predictor import cross_environment_validation
from src.utils.helpers import ensure_dir

def parse_args():
    parser = argparse.ArgumentParser(description='Run failure prediction')
    parser.add_argument('--train_envs', nargs='+', default=['halfcheetah', 'hopper'])
    parser.add_argument('--test_env', type=str, default='walker2d')
    parser.add_argument('--results_dir', type=str, default='results')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"\n--- Failure Prediction Stage ---")
    print(f"Train Envs: {args.train_envs}")
    print(f"Test Env:   {args.test_env}")
    print(f"Loading results from: {args.results_dir}/degradation\n")
    
    out_dir = ensure_dir(os.path.join(args.results_dir, 'prediction'))
    
    for algo in ['cql', 'iql']:
        print(f"\nTraining predictor for {algo.upper()}...")
        
        results = cross_environment_validation(
            train_envs=args.train_envs,
            test_env=args.test_env,
            algorithm=algo,
            results_dir=args.results_dir
        )
        
        if not results:
            print(f"Failed to run validation for {algo}. Check if degradation results exist.")
            continue
            
        train_metrics = results['train_metrics']
        test_metrics = results['test_metrics']
        
        print("\nTrain Metrics:")
        print(f"  Spearman ρ: {train_metrics['spearman_rho']:.3f} (p={train_metrics['spearman_pval']:.3e})")
        print(f"  R² Score:   {train_metrics['r2']:.3f}")
        
        print("\nTest Metrics (Cross-Env Generalization):")
        print(f"  Spearman ρ: {test_metrics['spearman_rho']:.3f} (p={test_metrics['spearman_pval']:.3e})")
        print(f"  Rank Acc:   {test_metrics['rank_accuracy']:.3f}")
        print(f"  RMSE:       {test_metrics['rmse']:.3f}")
        
        print("\nTop Features:")
        features = results['feature_importances']
        for i, (f, w) in enumerate(features.items()):
            print(f"  {i+1}. {f:<20} : {w:+.4f}")
            if i >= 4:
                break
                
        # Save results
        out_path = os.path.join(out_dir, f'{algo}_prediction_results.json')
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
