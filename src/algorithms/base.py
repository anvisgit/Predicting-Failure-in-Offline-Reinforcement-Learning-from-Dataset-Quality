from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class OfflineRLAgent(ABC):
    """Abstract base for offline RL algorithm wrappers."""
    
    @abstractmethod
    def build(self, dataset_dict: Dict[str, np.ndarray], env) -> None:
        """Initialize algorithm with dataset and environment."""
        pass
    
    @abstractmethod
    def train(self, dataset_dict: Dict[str, np.ndarray], env,
              n_steps: int, eval_freq: int, logger) -> Dict[str, Any]:
        """Run training, return dict of all results."""
        pass
    
    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Select action for a single observation."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        pass
        
    @abstractmethod
    def load(self, path: str) -> None:
        pass
        
    @property
    @abstractmethod
    def name(self) -> str:
        pass
