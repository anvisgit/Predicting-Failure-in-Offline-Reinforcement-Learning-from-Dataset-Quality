from typing import Any, Dict

import numpy as np
from d3rlpy.algos import CQLConfig
from d3rlpy.metrics import (
    AverageValueEstimationEvaluator,
    EnvironmentEvaluator,
    TDErrorEvaluator,
)

from .base import OfflineRLAgent
from ..data.loader import to_d3rlpy_dataset


class CQLAgent(OfflineRLAgent):
    """Conservative Q-Learning wrapper around d3rlpy."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get("device", "cpu")
        self._algo = None
        self._dataset = None

    def build(self, dataset_dict: Dict[str, np.ndarray], env) -> None:
        """Build CQL algorithm from d3rlpy config."""
        self._dataset = to_d3rlpy_dataset(dataset_dict)
        self._algo = CQLConfig(
            actor_learning_rate=self.config.get("actor_lr", 1e-4),
            critic_learning_rate=self.config.get("critic_lr", 3e-4),
            alpha_learning_rate=self.config.get("alpha_lr", 1e-4),
            conservative_weight=self.config.get("conservative_weight", 5.0),
            n_action_samples=self.config.get("n_action_samples", 10),
            initial_alpha=self.config.get("initial_alpha", 1.0),
            alpha_threshold=self.config.get("alpha_threshold", 10.0),
            batch_size=self.config.get("batch_size", 256),
        ).create(device=self.device)
        self._algo.build_with_dataset(self._dataset)

    def train(
        self,
        dataset_dict: Dict[str, np.ndarray],
        env,
        n_steps: int,
        eval_freq: int,
        logger,
    ) -> Dict[str, Any]:
        """Train CQL and return a results shell for downstream logging."""
        results = {
            "normalized_scores": [],
            "td_errors": [],
            "q_values": [],
            "steps": [],
            "training_history": [],
            "best_step": None,
            "best_epoch": None,
            "best_environment_score": None,
            "best_d3rlpy_checkpoint": None,
        }

        experiment_name = getattr(logger, "run_name", None)
        history = self._algo.fit(
            self._dataset,
            n_steps=n_steps,
            n_steps_per_epoch=eval_freq,
            experiment_name=experiment_name,
            with_timestamp=experiment_name is None,
            save_interval=1,
            evaluators={
                "environment": EnvironmentEvaluator(env, n_trials=10),
                "td_error": TDErrorEvaluator(episodes=self._dataset.episodes[:100]),
                "q_value": AverageValueEstimationEvaluator(
                    episodes=self._dataset.episodes[:100]
                ),
            },
        )

        best_record = None
        for epoch, metrics in history:
            step = int(epoch * eval_freq)
            record = {"epoch": int(epoch), "step": step, **metrics}
            results["training_history"].append(record)
            results["steps"].append(step)

            if "environment" in metrics:
                results["normalized_scores"].append(metrics["environment"])
                if logger is not None:
                    logger.log_evaluation(step, {"environment": metrics["environment"]})

                if best_record is None or metrics["environment"] > best_record["environment"]:
                    best_record = record

            if "td_error" in metrics:
                results["td_errors"].append(metrics["td_error"])
            if "q_value" in metrics:
                results["q_values"].append(metrics["q_value"])
            if logger is not None:
                logger.log_metrics(step, metrics)

        if best_record is not None:
            best_step = best_record["step"]
            results["best_step"] = best_step
            results["best_epoch"] = best_record["epoch"]
            results["best_environment_score"] = best_record["environment"]
            if experiment_name:
                results["best_d3rlpy_checkpoint"] = (
                    f"d3rlpy_logs/{experiment_name}/model_{best_step}.d3"
                )
        return results

    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self._algo.predict(observation.reshape(1, -1))[0]

    def save(self, path: str) -> None:
        self._algo.save_model(path)

    def load(self, path: str) -> None:
        self._algo.load_model(path)

    @property
    def name(self) -> str:
        return "CQL"
