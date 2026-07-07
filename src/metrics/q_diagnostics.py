import numpy as np
import torch
from typing import Dict


def q_value_on_dataset(
    algo,  # d3rlpy algorithm object
    observations: np.ndarray,
    actions: np.ndarray,
    sample_size: int = 1000
) -> float:
    """
    Compute mean Q-value on dataset (s,a) pairs.
    Samples up to sample_size transitions from the dataset.
    """
    N = len(observations)
    if N > sample_size:
        indices = np.random.choice(N, sample_size, replace=False)
        obs_sample = observations[indices]
        act_sample = actions[indices]
    else:
        obs_sample = observations
        act_sample = actions
        
    # predict_value returns numpy array of shape (N,)
    q_values = algo.predict_value(obs_sample, act_sample)
    return float(np.mean(q_values))


def q_value_on_random(
    algo,
    observations: np.ndarray,
    action_dim: int,
    sample_size: int = 1000,
    n_random: int = 10
) -> float:
    """
    Compute mean Q-value when actions are replaced with uniform random samples.
    Same states as dataset, random actions in [-1, 1]^action_dim.
    Measures OOD overestimation.
    """
    N = len(observations)
    if N > sample_size:
        indices = np.random.choice(N, sample_size, replace=False)
        obs_sample = observations[indices]
    else:
        obs_sample = observations
        
    all_q_values = []
    # Repeat for n_random samples per state
    for _ in range(n_random):
        random_actions = np.random.uniform(-1.0, 1.0, size=(len(obs_sample), action_dim))
        q_vals = algo.predict_value(obs_sample, random_actions)
        all_q_values.extend(q_vals)
        
    return float(np.mean(all_q_values))


def q_overestimation_gap(
    algo,
    observations: np.ndarray,
    actions: np.ndarray,
    action_dim: int,
    sample_size: int = 1000
) -> Dict[str, float]:
    """
    Compute the OOD overestimation gap:
    gap = Q(s, a_random) - Q(s, a_dataset)
    Returns dict with: q_dataset, q_random, gap, gap_ratio
    """
    q_data = q_value_on_dataset(algo, observations, actions, sample_size)
    q_rand = q_value_on_random(algo, observations, action_dim, sample_size)
    
    gap = q_rand - q_data
    # Use absolute value of data Q for ratio to avoid sign issues
    gap_ratio = gap / (abs(q_data) + 1e-6)
    
    return {
        'q_dataset': q_data,
        'q_random': q_rand,
        'gap': gap,
        'gap_ratio': gap_ratio
    }


def compute_td_error(
    algo,
    observations: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_observations: np.ndarray,
    terminals: np.ndarray,
    gamma: float = 0.99,
    sample_size: int = 1000
) -> float:
    """
    Compute mean TD error on a sample of transitions.
    """
    N = len(observations)
    if N > sample_size:
        indices = np.random.choice(N, sample_size, replace=False)
        obs_s = observations[indices]
        act_s = actions[indices]
        rew_s = rewards[indices]
        next_obs_s = next_observations[indices]
        term_s = terminals[indices]
    else:
        obs_s = observations
        act_s = actions
        rew_s = rewards
        next_obs_s = next_observations
        term_s = terminals
        
    # Get current Q values
    q_current = algo.predict_value(obs_s, act_s)
    
    # Get next actions from current policy
    next_actions = algo.predict(next_obs_s)
    
    # Get next Q values
    q_next = algo.predict_value(next_obs_s, next_actions)
    
    # Compute TD targets
    # If terminal, next Q is 0
    td_targets = rew_s + gamma * q_next * (1.0 - term_s)
    
    td_errors = np.abs(q_current - td_targets)
    return float(np.mean(td_errors))
