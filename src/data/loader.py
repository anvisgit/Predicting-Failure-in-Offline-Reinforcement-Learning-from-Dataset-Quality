import numpy as np
import gymnasium as gym
import h5py
import urllib.request
import os
from pathlib import Path
from d3rlpy.dataset import MDPDataset
from typing import Dict, Tuple, Optional
import torch

# Environment configurations
ENV_CONFIGS = {
    'halfcheetah': {'obs_dim': 17, 'act_dim': 6, 'ref_min_score': -280.178953, 'ref_max_score': 12135.0},
    'hopper':      {'obs_dim': 11, 'act_dim': 3, 'ref_min_score': -20.272305,  'ref_max_score': 3234.3},
    'walker2d':    {'obs_dim': 17, 'act_dim': 6, 'ref_min_score': 1.629008,    'ref_max_score': 4592.3},
}

DATASET_VARIANTS = ['random', 'medium', 'medium-replay', 'medium-expert', 'expert']
ENVIRONMENTS = ['halfcheetah', 'hopper', 'walker2d']

# Direct HDF5 download URLs from D4RL's official Berkeley servers
# Note: medium-replay is called "mixed" in the URL scheme
HDF5_URLS = {
    ('halfcheetah', 'random'):        'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_random.hdf5',
    ('halfcheetah', 'medium'):        'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium.hdf5',
    ('halfcheetah', 'medium-replay'): 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_mixed.hdf5',
    ('halfcheetah', 'medium-expert'): 'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_medium_expert.hdf5',
    ('halfcheetah', 'expert'):        'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/halfcheetah_expert.hdf5',
    ('hopper', 'random'):             'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_random.hdf5',
    ('hopper', 'medium'):             'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium.hdf5',
    ('hopper', 'medium-replay'):      'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_mixed.hdf5',
    ('hopper', 'medium-expert'):      'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_medium_expert.hdf5',
    ('hopper', 'expert'):             'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/hopper_expert.hdf5',
    ('walker2d', 'random'):           'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_random.hdf5',
    ('walker2d', 'medium'):           'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium.hdf5',
    ('walker2d', 'medium-replay'):    'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_mixed.hdf5',
    ('walker2d', 'medium-expert'):    'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_medium_expert.hdf5',
    ('walker2d', 'expert'):           'http://rail.eecs.berkeley.edu/datasets/offline_rl/gym_mujoco/walker2d_expert.hdf5',
}
# Gymnasium env IDs
GYM_ENV_IDS = {
    'halfcheetah': 'HalfCheetah-v4',
    'hopper':      'Hopper-v4',
    'walker2d':    'Walker2d-v4',
}

# Local cache directory for downloaded HDF5 files
CACHE_DIR = Path.home() / '.d4rl_cache'


def get_d4rl_env_name(env: str, dataset: str) -> str:
    return f"{env}-{dataset}-v2"


def _download_hdf5(url: str, dest: Path) -> None:
    """Download HDF5 file with progress indicator."""
    print(f"Downloading: {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(count * block_size / total_size * 100, 100)
            print(f"\r  Progress: {pct:.1f}%", end='', flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=reporthook)
    print()


def _load_hdf5(path: Path) -> Dict[str, np.ndarray]:
    """Load dataset dict from a D4RL HDF5 file."""
    with h5py.File(path, 'r') as f:
        observations = f['observations'][:]
        actions      = f['actions'][:]
        rewards      = f['rewards'][:]
        terminals    = f['terminals'][:]
        timeouts     = f['timeouts'][:] if 'timeouts' in f else np.zeros_like(terminals, dtype=bool)

        if 'next_observations' in f:
            next_observations = f['next_observations'][:]
        else:
            next_observations = np.zeros_like(observations)
            next_observations[:-1] = observations[1:]

    return {
        'observations':      observations.astype(np.float32),
        'actions':           actions.astype(np.float32),
        'rewards':           rewards.astype(np.float32),
        'terminals':         terminals.astype(np.float32),
        'next_observations': next_observations.astype(np.float32),
        'timeouts':          timeouts.astype(np.float32),
    }


def load_d4rl_dataset(env_name: str, dataset_variant: str) -> Tuple[Dict[str, np.ndarray], gym.Env]:
    """
    Load a D4RL dataset by downloading the HDF5 directly from Berkeley's servers.
    No d4rl package or Minari required. Files are cached in ~/.d4rl_cache/.
    Returns (dataset_dict, env)
    """
    key = (env_name, dataset_variant)
    if key not in HDF5_URLS:
        raise ValueError(
            f"Unknown combination: env='{env_name}', dataset='{dataset_variant}'. "
            f"Valid envs: {ENVIRONMENTS}, valid datasets: {DATASET_VARIANTS}"
        )

    url = HDF5_URLS[key]
    filename = url.split('/')[-1]
    local_path = CACHE_DIR / filename

    if local_path.exists():
        print(f"Using cached dataset: {local_path}")
    else:
        _download_hdf5(url, local_path)

    print(f"Loading dataset from {local_path} ...")
    dataset_dict = _load_hdf5(local_path)
    print(f"Dataset size: {len(dataset_dict['observations'])} transitions")

    env = gym.make(GYM_ENV_IDS[env_name])
    return dataset_dict, env


def to_d3rlpy_dataset(dataset_dict: Dict[str, np.ndarray]) -> MDPDataset:
    return MDPDataset(
        observations=dataset_dict['observations'],
        actions=dataset_dict['actions'],
        rewards=dataset_dict['rewards'],
        terminals=dataset_dict['terminals'],
        timeouts=dataset_dict.get('timeouts', np.zeros_like(dataset_dict['terminals']))
    )


def get_normalized_score(env_name: str, score: float) -> float:
    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown environment: {env_name}")
    ref_min = ENV_CONFIGS[env_name]['ref_min_score']
    ref_max = ENV_CONFIGS[env_name]['ref_max_score']
    return (score - ref_min) / (ref_max - ref_min) * 100.0


class ReplayBuffer:
    def __init__(self, dataset_dict: Dict[str, np.ndarray], device: str = 'cpu'):
        self.device            = torch.device(device)
        self.observations      = dataset_dict['observations']
        self.actions           = dataset_dict['actions']
        self.rewards           = dataset_dict['rewards']
        self.next_observations = dataset_dict['next_observations']
        self.terminals         = dataset_dict['terminals']
        self.size = len(self.observations)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            'observations':      torch.FloatTensor(self.observations[idxs]).to(self.device),
            'actions':           torch.FloatTensor(self.actions[idxs]).to(self.device),
            'rewards':           torch.FloatTensor(self.rewards[idxs]).unsqueeze(-1).to(self.device),
            'next_observations': torch.FloatTensor(self.next_observations[idxs]).to(self.device),
            'terminals':         torch.FloatTensor(self.terminals[idxs]).unsqueeze(-1).to(self.device),
        }

    def __len__(self) -> int:
        return self.size