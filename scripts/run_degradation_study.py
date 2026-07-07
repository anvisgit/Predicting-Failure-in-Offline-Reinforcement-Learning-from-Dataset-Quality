#!/usr/bin/env python
"""
Stage 2: Systematic Degradation Study.

Usage:
    python scripts/run_degradation_study.py --env hopper --protocol coverage --n_steps 100000
    python scripts/run_degradation_study.py --quick
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.algorithms.cql import CQLAgent
from src.algorithms.iql import IQLAgent
from src.data.degradation import (
    get_coverage_variants,
    get_noise_variants,
    get_quality_variants,
)
from src.data.loader import load_d4rl_dataset
from src.metrics.dataset_stats import compute_all_stats
from src.metrics.performance import evaluate_and_normalize
from src.utils.helpers import ensure_dir, load_config, set_seed
from src.utils.logger import ExperimentLogger

DEFAULT_N_STEPS = 1000000


def parse_args():
    parser = argparse.ArgumentParser(description="Run systematic degradation study")
    parser.add_argument(
        "--env",
        type=str,
        default="all",
        choices=["halfcheetah", "hopper", "walker2d", "all"],
    )
    parser.add_argument(
        "--protocol",
        type=str,
        default="all",
        choices=["coverage", "quality", "noise", "all"],
    )
    parser.add_argument("--algorithm", type=str, default="all", choices=["cql", "iql", "all"])
    parser.add_argument("--n_steps", type=int, default=DEFAULT_N_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results_dir", type=str, default="results/degradation")
    parser.add_argument("--quick", action="store_true", help="Run short test on one env")
    return parser.parse_args()


def run_experiment(algo_name, env_name, protocol, level, degraded_dataset, env_obj, args):
    print(f"\n--- Running {algo_name.upper()} | {env_name} | {protocol} | level: {level} ---")

    config = load_config(str(ROOT / "configs" / f"{algo_name}.yaml"))
    config["device"] = args.device
    config["seed"] = args.seed

    run_name = f"{algo_name}_{env_name}_{protocol}_{level}_seed{args.seed}"
    logger = ExperimentLogger(config, run_name, use_wandb=False)
    logger.results_dir = Path(args.results_dir) / protocol / run_name
    ensure_dir(logger.results_dir)

    stats = compute_all_stats(degraded_dataset)
    logger.log_dataset_stats(stats)

    agent = CQLAgent(config) if algo_name == "cql" else IQLAgent(config)
    agent.build(degraded_dataset, env_obj)

    agent.train(
        degraded_dataset,
        env_obj,
        n_steps=args.n_steps,
        eval_freq=max(1, args.n_steps // 10),
        logger=logger,
    )

    n_eval_episodes = 5 if args.quick else config.get("eval_episodes", 10)
    eval_res = evaluate_and_normalize(
        env_obj,
        env_name,
        agent.predict,
        n_episodes=n_eval_episodes,
    )
    final_score = eval_res["normalized_score"]

    results = {
        "algorithm": algo_name,
        "env": env_name,
        "protocol": protocol,
        "level": level,
        "dataset_stats": stats,
        "final_raw_score": eval_res["mean_return"],
        "final_normalized_score": final_score,
        "final_eval": eval_res,
        "seed": args.seed,
        "n_eval_episodes": n_eval_episodes,
    }

    with open(logger.results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.finish()
    print(f"Completed {run_name}. Score: {final_score:.1f}")


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.quick:
        args.env = "hopper"
        if args.n_steps == DEFAULT_N_STEPS:
            args.n_steps = 5000
        args.protocol = "coverage"
        if args.algorithm == "all":
            args.algorithm = "cql"

    envs = ["halfcheetah", "hopper", "walker2d"] if args.env == "all" else [args.env]
    algos = ["cql", "iql"] if args.algorithm == "all" else [args.algorithm]
    protocols = ["coverage", "quality", "noise"] if args.protocol == "all" else [args.protocol]

    for env_name in envs:
        med_replay_data, env_obj = load_d4rl_dataset(env_name, "medium-replay")
        expert_data, _ = load_d4rl_dataset(env_name, "expert")
        random_data, _ = load_d4rl_dataset(env_name, "random")

        for protocol in protocols:
            if protocol == "coverage":
                variants = get_coverage_variants(med_replay_data, seed=args.seed)
                if args.quick:
                    variants = {"100": variants["100"], "10": variants["10"]}
            elif protocol == "quality":
                variants = get_quality_variants(expert_data, random_data, seed=args.seed)
            else:
                variants = get_noise_variants(med_replay_data, seed=args.seed)

            for level, variant_data in variants.items():
                for algo in algos:
                    try:
                        run_experiment(algo, env_name, protocol, level, variant_data, env_obj, args)
                    except Exception as exc:
                        print(f"Error in experiment {algo} {env_name} {protocol} {level}: {exc}")


if __name__ == "__main__":
    main()
