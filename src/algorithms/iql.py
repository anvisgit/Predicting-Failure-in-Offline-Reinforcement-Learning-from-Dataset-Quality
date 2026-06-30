import d3rlpy
from d3rlpy.algos import IQLConfig
from d3rlpy.metrics import EnvironmentEvaluator, TDErrorEvaluator, AverageValueEstimationEvaluator
from d3rlpy.dataset import MDPDataset
import numpy as np
from typing import Dict, Any, Optional
from .base import OfflineRLAgent
from ..data.loader import to_d3rlpy_dataset


class IQLAgent(OfflineRLAgent):
    """
    Implicit Q-Learning (Kostrikov et al., 2021) wrapper around d3rlpy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = config.get('device', 'cpu')
        self._algo = None
        self._dataset = None
    
    def build(self, dataset_dict: Dict[str, np.ndarray], env) -> None:
        """Build IQL algorithm from d3rlpy config."""
        self._dataset = to_d3rlpy_dataset(dataset_dict)
        self._algo = IQLConfig(
            actor_learning_rate=self.config.get('actor_lr', 3e-4),
            critic_learning_rate=self.config.get('critic_lr', 3e-4),
            value_learning_rate=self.config.get('value_lr', 3e-4),
            expectile=self.config.get('expectile', 0.7),
            weight_temp=self.config.get('weight_temp', 3.0),
            max_weight=self.config.get('max_weight', 100.0),
            batch_size=self.config.get('batch_size', 256),
        ).create(device=self.device)
        self._algo.build_with_dataset(self._dataset)
    
    def train(self, dataset_dict: Dict[str, np.ndarray], env,
              n_steps: int, eval_freq: int, logger) -> Dict[str, Any]:
        """Train IQL, return results dict with training history."""
        results = {'normalized_scores': [], 'td_errors': [], 'q_values': [], 'steps': []}
        
        def epoch_callback(algo, epoch, total_steps):
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
        
        return results
    
    def predict(self, observation: np.ndarray) -> np.ndarray:
        return self._algo.predict(observation.reshape(1, -1))[0]
    
    def save(self, path: str) -> None:
        self._algo.save_model(path)
    
    def load(self, path: str) -> None:
        self._algo.load_model(path)
    
    @property
    def name(self) -> str:
        return 'IQL'
