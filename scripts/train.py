#!/usr/bin/env python
"""
Main training script for offline RL experiments.

Examples:
  python scripts/train.py --algorithm cql --env hopper --dataset medium-replay --seed 42
  python scripts/train.py --algorithm iql --env halfcheetah --dataset medium --n_steps 500000
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.algorithms.cql import CQLAgent
from src.algorithms.iql import IQLAgent
from src.data.loader import load_d4rl_dataset
from src.metrics.dataset_stats import compute_all_stats
from src.metrics.performance import evaluate_and_normalize
from src.utils.helpers import ensure_dir, load_config, set_seed
from src.utils.logger import ExperimentLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Train offline RL agent on D4RL dataset")
    parser.add_argument("--algorithm", type=str, choices=["cql", "iql"], required=True)
    parser.add_argument("--env", type=str, choices=["halfcheetah", "hopper", "walker2d"], required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["random", "medium", "medium-replay", "medium-expert", "expert"],
        required=True,
    )
    parser.add_argument("--n_steps", type=int, default=None, help="Override total training steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config override")
    parser.add_argument("--wandb_project", type=str, default="offline-rl-research")
    parser.add_argument("--no_wandb", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config) if args.config else ROOT / "configs" / f"{args.algorithm}.yaml"
    config = load_config(str(config_path))
    config["device"] = args.device
    config["seed"] = args.seed
    if args.n_steps:
        config["total_steps"] = args.n_steps

    set_seed(args.seed)

    run_name = f"{args.algorithm}_{args.env}_{args.dataset}_seed{args.seed}"
    logger = ExperimentLogger(config, run_name, use_wandb=not args.no_wandb)

    print(f"\n{'=' * 60}")
    print(f"Algorithm: {args.algorithm.upper()}")
    print(f"Environment: {args.env} | Dataset: {args.dataset}")
    print(f"Steps: {config['total_steps']} | Device: {args.device}")
    print(f"{'=' * 60}\n")

    print("Loading D4RL dataset...")
    dataset_dict, env = load_d4rl_dataset(args.env, args.dataset)
    print(f"Dataset size: {len(dataset_dict['observations'])} transitions")

    print("Computing dataset statistics...")
    dataset_stats = compute_all_stats(dataset_dict)
    logger.log_dataset_stats(dataset_stats)

    agent = CQLAgent(config) if args.algorithm == "cql" else IQLAgent(config)
    agent.build(dataset_dict, env)

    print("Starting training...")
    results = agent.train(
        dataset_dict,
        env,
        n_steps=config["total_steps"],
        eval_freq=config.get("eval_freq", 5000),
        logger=logger,
    )

    n_eval_episodes = config.get("eval_episodes", 10)
    print(f"Running final evaluation over {n_eval_episodes} episodes...")
    eval_results = evaluate_and_normalize(
        env,
        args.env,
        agent.predict,
        n_episodes=n_eval_episodes,
    )

    results.update(
        {
            "final_raw_score": eval_results["mean_return"],
            "final_normalized_score": eval_results["normalized_score"],
            "final_eval": eval_results,
            "config": config,
            "dataset_stats": dataset_stats,
            "env": args.env,
            "dataset": args.dataset,
            "algorithm": args.algorithm,
            "seed": args.seed,
        }
    )

    out_dir = ensure_dir(os.path.join(args.results_dir, run_name))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)

    agent.save(os.path.join(out_dir, "model.pt"))
    logger.finish()

    if results.get("best_environment_score") is not None:
        print(
            "\nBest training evaluation: "
            f"{results['best_environment_score']:.2f} "
            f"at epoch {results['best_epoch']} / step {results['best_step']}"
        )
        if results.get("best_d3rlpy_checkpoint"):
            print(f"Best d3rlpy checkpoint: {results['best_d3rlpy_checkpoint']}")
        if results.get("training_history"):
            final_train_score = results["training_history"][-1].get("environment")
            if final_train_score is not None and final_train_score < results["best_environment_score"]:
                print(
                    "WARNING: final checkpoint is worse than the best training checkpoint. "
                    "Evaluate the best checkpoint before using the final model."
                )

    print(f"\nFinal normalized score: {results['final_normalized_score']:.2f}")
    print(f"Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
