<<<<<<< HEAD
#!/usr/bin/env python
"""
Stage 1: Baseline Reproduction
Runs CQL and IQL on medium-replay for all 3 environments.
Compares results to published D4RL numbers.

Usage:
    python scripts/run_baseline.py --n_steps 200000 --seed 42
"""
=======

>>>>>>> 61a721dcb2dba975feffcf589db14be640cebc1b
import argparse
import sys
import os
from pathlib import Path
import json
<<<<<<< HEAD

=======
>>>>>>> 61a721dcb2dba975feffcf589db14be640cebc1b
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_d4rl_dataset, get_normalized_score
from src.algorithms.cql import CQLAgent
from src.algorithms.iql import IQLAgent
from src.utils.helpers import set_seed, load_config, ensure_dir
from src.utils.logger import ExperimentLogger

<<<<<<< HEAD
PUBLISHED_SCORES = {
    'cql': {'halfcheetah': 44.0, 'hopper': 86.6, 'walker2d': 74.5},
    'iql': {'halfcheetah': 47.4, 'hopper': 91.5, 'walker2d': 83.3},
}
=======
PUBLISHED_SCORES = {'cql': {'halfcheetah': 44.0, 'hopper': 86.6, 'walker2d': 74.5},'iql': {'halfcheetah': 47.4, 'hopper': 91.5, 'walker2d': 83.3},}
>>>>>>> 61a721dcb2dba975feffcf589db14be640cebc1b

def parse_args():
    parser = argparse.ArgumentParser(description='Run baseline reproduction for offline RL algorithms')
    parser.add_argument('--n_steps', type=int, default=200000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--results_dir', type=str, default='results/baseline')
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.results_dir)
    
    environments = ['halfcheetah', 'hopper', 'walker2d']
    algorithms = ['cql', 'iql']
    dataset = 'medium-replay'
    
    results = {}
    
    for algo_name in algorithms:
        results[algo_name] = {}
        config = load_config(str(ROOT / 'configs' / f'{algo_name}.yaml'))
        config['device'] = args.device
        config['seed'] = args.seed
        
        for env_name in environments:
            print(f"\nRunning {algo_name.upper()} on {env_name}-{dataset}")
            run_name = f"baseline_{algo_name}_{env_name}_{dataset}_seed{args.seed}"
            logger = ExperimentLogger(config, run_name, use_wandb=False)
            
            try:
                dataset_dict, env = load_d4rl_dataset(env_name, dataset)
                
                if algo_name == 'cql':
                    agent = CQLAgent(config)
                else:
                    agent = IQLAgent(config)
                    
                agent.build(dataset_dict, env)
                
                agent.train(
                    dataset_dict, env,
                    n_steps=args.n_steps,
                    eval_freq=args.n_steps // 10, # 10 evaluations
                    logger=logger
                )
<<<<<<< HEAD
                
                # Run final evaluation
=======
>>>>>>> 61a721dcb2dba975feffcf589db14be640cebc1b
                from src.metrics.performance import evaluate_and_normalize
                eval_res = evaluate_and_normalize(env, env_name, agent.predict, n_episodes=10)
                final_score = eval_res['normalized_score']
                
                results[algo_name][env_name] = final_score
                logger.save_results({'final_normalized_score': final_score}, 'final_eval.json')
                print(f"Final normalized score: {final_score:.1f}")
                
            except Exception as e:
                print(f"Error running {algo_name} on {env_name}: {e}")
                results[algo_name][env_name] = None
            
            logger.finish()
<<<<<<< HEAD
            
    # Print comparison table
=======
>>>>>>> 61a721dcb2dba975feffcf589db14be640cebc1b
    print("\n" + "="*60)
    print("Stage 1: Baseline Reproduction Results")
    print("="*60)
    print(f"{'Algorithm':<15} | {'Environment':<15} | {'Published':<10} | {'Reproduction':<15} | {'Gap'}")
    print("-" * 60)
    
    for algo in algorithms:
        for env in environments:
            pub = PUBLISHED_SCORES[algo][env]
            rep = results[algo].get(env)
            if rep is not None:
                gap = rep - pub
                print(f"{algo.upper():<15} | {env:<15} | {pub:<10.1f} | {rep:<15.1f} | {gap:+.1f}")
            else:
                print(f"{algo.upper():<15} | {env:<15} | {pub:<10.1f} | {'FAILED':<15} | N/A")
                
    with open(os.path.join(args.results_dir, 'baseline_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()
