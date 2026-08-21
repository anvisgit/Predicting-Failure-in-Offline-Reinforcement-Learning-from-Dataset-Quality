import random
import numpy as np
import torch
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(yaml_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Load YAML config and apply optional overrides."""
    # load default first
    default_path = Path(yaml_path).parent.parent / 'configs' / 'default.yaml'
    config = {}
    if default_path.exists():
        with open(default_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    with open(yaml_path, 'r') as f:
        algo_config = yaml.safe_load(f) or {}
    config.update(algo_config)
    if overrides:
        config.update(overrides)
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two config dicts."""
    result = base.copy()
    result.update(override)
    return result


def split_into_trajectories(
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: Optional[np.ndarray] = None
) -> List[Dict[str, np.ndarray]]:
    """
    Split a flat D4RL dataset into individual trajectories.
    Returns list of trajectory dicts, each with keys:
    observations, actions, rewards, terminals, returns.
    """
    trajectories = []
    start = 0
    N = len(observations)
    ends = np.where(
        terminals.astype(bool) | (timeouts.astype(bool) if timeouts is not None else np.zeros(N, bool))
    )[0]
    # ensure we include the last trajectory
    if len(ends) == 0 or ends[-1] != N - 1:
        ends = np.append(ends, N - 1)

    for end in ends:
        traj = {
            'observations': observations[start:end + 1],
            'actions': actions[start:end + 1],
            'rewards': rewards[start:end + 1],
            'terminals': terminals[start:end + 1],
        }
        traj['returns'] = float(traj['rewards'].sum())
        traj['length'] = end - start + 1
        trajectories.append(traj)
        start = end + 1

    return trajectories


def ensure_dir(path: str) -> str:
    """Create directory if it doesn't exist. Returns path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_device(device_str: str = 'cpu') -> torch.device:
    """Get torch device. Falls back to CPU if CUDA not available."""
    if device_str.startswith('cuda') and not torch.cuda.is_available():
        print(f"[WARNING] CUDA not available, falling back to CPU.")
        return torch.device('cpu')
    return torch.device(device_str)
