from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class OfflineRLAgent(ABC):
    @abstractmethod
    def build(self, dataset_dict: Dict[str, np.ndarray], env) -> None:
        pass

    @abstractmethod
    def train(
        self,
        dataset_dict: Dict[str, np.ndarray],
        env,
        n_steps: int,
        eval_freq: int,
        logger,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, observation: np.ndarray) -> np.ndarray:
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
