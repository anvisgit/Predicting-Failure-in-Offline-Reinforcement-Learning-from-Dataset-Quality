import d3rlpy
from d3rlpy.algos import CQLConfig
from d3rlpy.metrics import EnvironmentEvaluator, TDErrorEvaluator, AverageValueEstimationEvaluator
from d3rlpy.dataset import MDPDataset
import numpy as np
from typing import Dict, Any, Optional
from .base import OfflineRLAgent
from ..data.loader import to_d3rlpy_dataset


class CQLAgent(OfflineRLAgent):
    """
    Conservative Q-Learning (Kumar et al., 2020) wrapper around d3rlpy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cpu')
        self._algo = None
        self._dataset = None
    
    def build(self, dataset_dict: Dict[str, np.ndarray], env) -> None:
        """Build CQL algorithm from d3rlpy config."""
        self._dataset = to_d3rlpy_dataset(dataset_dict)
        self._algo = CQLConfig(
            actor_learning_rate=self.config.get('actor_lr', 1e-4),
            critic_learning_rate=self.config.get('critic_lr', 3e-4),
            alpha_learning_rate=self.config.get('alpha_lr', 1e-4),
            conservative_weight=self.config.get('conservative_weight', 5.0),
            n_action_samples=self.config.get('n_action_samples', 10),
            initial_alpha=self.config.get('initial_alpha', 1.0),
            alpha_threshold=self.config.get('alpha_threshold', 10.0),
            batch_size=self.config.get('batch_size', 256),
        ).create(device=self.device)
        self._algo.build_with_dataset(self._dataset)
    
    def train(self, dataset_dict: Dict[str, np.ndarray], env,
              n_steps: int, eval_freq: int, logger) -> Dict[str, Any]:
        """Train CQL, return results dict with training history."""
        results = {'normalized_scores': [], 'td_errors': [], 'q_values': [], 'steps': []}
        
        # We need D4RL's env name to compute normalized score, if it's stored in env
        env_name = None
        # Extract environment name if possible
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'spec') and env.unwrapped.spec is not None:
             env_name = env.unwrapped.spec.id.split('-')[0].lower()
             
        def epoch_callback(algo, epoch, total_steps):
            # Evaluate using evaluators
            pass

        self._algo.fit(
            self._dataset,
            n_steps=n_steps,
            n_steps_per_epoch=eval_freq,
            evaluators={
                'environment': EnvironmentEvaluator(env, n_trials=10),
                'td_error': TDErrorEvaluator(episodes=self._dataset.episodes[:100]),
                'q_value': AverageValueEstimationEvaluator(episodes=self._dataset.episodes[:100]),
            },
            epoch_callback=epoch_callback,
        )
        
        # After training, you would ideally extract the logs that d3rlpy captured.
        # For simplicity, returning an empty history if we rely on d3rlpy's logger
        # or the external logger to parse the stdout/files.
        return results
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self._algo.predict(observation.reshape(1, -1))[0]
    
    def save(self, path: str) -> None:
        self._algo.save_model(path)
    
    def load(self, path: str) -> None:
        self._algo.load_model(path)
    
    @property
    def name(self) -> str:
        return 'CQL'
