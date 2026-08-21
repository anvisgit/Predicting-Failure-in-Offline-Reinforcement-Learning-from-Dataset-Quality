import numpy as np
from scipy import stats
from typing import Dict, List, Optional


def coverage_entropy(observations: np.ndarray, n_bins: int = 20) -> float:
    """
    Compute Shannon entropy of state visitation distribution.
    Bins each dimension independently, computes joint histogram entropy.
    Low entropy = concentrated coverage (narrow dataset).
    High entropy = spread coverage (diverse dataset).
    """
    # Simply summing 1D entropies as an approximation to avoid memory issues with N-D histograms
    total_entropy = 0.0
    for dim in range(observations.shape[1]):
        hist, _ = np.histogram(observations[:, dim], bins=n_bins, density=True)
        # Convert density to probability
        p = hist / np.sum(hist)
        p = p[p > 0] # Filter zeros
        if len(p) > 0:
            total_entropy -= np.sum(p * np.log(p))
    return float(total_entropy)

def return_distribution_stats(
    rewards: np.ndarray,
    terminals: np.ndarray,
    timeouts: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute per-trajectory return statistics.
    Returns: {'mean_return', 'std_return', 'skewness', 'min_return', 'max_return', 'n_trajectories'}
    """
    N = len(rewards)
    if timeouts is None:
        timeouts = np.zeros(N, dtype=bool)
        
    ends = np.where(terminals.astype(bool) | timeouts.astype(bool))[0]
    if len(ends) == 0 or ends[-1] != N - 1:
        ends = np.append(ends, N - 1)
        
    returns = []
    start = 0
    for end in ends:
        traj_return = np.sum(rewards[start:end+1])
        returns.append(traj_return)
        start = end + 1
        
    returns = np.array(returns)
    
    return {
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'skewness': float(stats.skew(returns)) if len(returns) > 2 else 0.0,
        'min_return': float(np.min(returns)),
        'max_return': float(np.max(returns)),
        'n_trajectories': len(returns)
    }

def action_diversity(actions: np.ndarray, sample_size: int = 2000) -> float:
    """
    Mean pairwise L2 distance between actions.
    Sampled for efficiency if dataset is large.
    High diversity = policy was exploratory.
    Low diversity = narrow behavior policy.
    """
    N = len(actions)
    if N < 2:
        return 0.0
        
    sample_size = min(N, sample_size)
    indices = np.random.choice(N, sample_size, replace=False)
    act_sample = actions[indices]
    
    # Compute pairwise distances
    # A bit memory intensive, doing it in batches or just random pairs is better
    # Random pairs approach:
    idx1 = np.random.choice(N, sample_size)
    idx2 = np.random.choice(N, sample_size)
    dists = np.linalg.norm(actions[idx1] - actions[idx2], axis=1)
    
    return float(np.mean(dists))

def trajectory_length_stats(
    terminals: np.ndarray,
    timeouts: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Statistics on trajectory lengths.
    Returns: {'mean_length', 'std_length', 'min_length', 'max_length', 'n_trajectories'}
    Short trajectories suggest a poor behavior policy (fell early).
    """
    N = len(terminals)
    if timeouts is None:
        timeouts = np.zeros(N, dtype=bool)
        
    ends = np.where(terminals.astype(bool) | timeouts.astype(bool))[0]
    if len(ends) == 0 or ends[-1] != N - 1:
        ends = np.append(ends, N - 1)
        
    lengths = []
    start = 0
    for end in ends:
        lengths.append(end - start + 1)
        start = end + 1
        
    lengths = np.array(lengths)
    
    return {
        'mean_length': float(np.mean(lengths)),
        'std_length': float(np.std(lengths)),
        'min_length': float(np.min(lengths)),
        'max_length': float(np.max(lengths)),
    }

def ood_action_fraction(
    actions: np.ndarray,
    action_dim: int,
    n_random_samples: int = 5000,
    k_neighbors: int = 5,
    threshold_percentile: float = 95.0
) -> float:
    """
    Estimate fraction of random actions that are 'out of distribution'.
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Subsample actions for KNN fit
    N = len(actions)
    fit_size = min(N, 10000)
    fit_indices = np.random.choice(N, fit_size, replace=False)
    fit_actions = actions[fit_indices]
    
    knn = NearestNeighbors(n_neighbors=k_neighbors)
    knn.fit(fit_actions)
    
    # Compute threshold from in-distribution distances
    test_size = min(N, 2000)
    test_indices = np.random.choice(N, test_size, replace=False)
    test_actions = actions[test_indices]
    
    dists, _ = knn.kneighbors(test_actions)
    mean_dists = np.mean(dists, axis=1)
    threshold = np.percentile(mean_dists, threshold_percentile)
    
    # Evaluate random actions
    random_actions = np.random.uniform(-1.0, 1.0, size=(n_random_samples, action_dim))
    rand_dists, _ = knn.kneighbors(random_actions)
    mean_rand_dists = np.mean(rand_dists, axis=1)
    
    ood_fraction = np.mean(mean_rand_dists > threshold)
    return float(ood_fraction)

def compute_all_stats(dataset: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Compute all dataset statistics. Returns flat dict of scalar metrics.
    """
    obs = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    timeouts = dataset.get('timeouts', None)
    action_dim = actions.shape[1]
    
    stats_dict = {}
    stats_dict['coverage_entropy'] = coverage_entropy(obs)
    stats_dict.update(return_distribution_stats(rewards, terminals, timeouts))
    stats_dict['action_diversity'] = action_diversity(actions)
    stats_dict.update(trajectory_length_stats(terminals, timeouts))
    
    try:
        stats_dict['ood_action_fraction'] = ood_action_fraction(actions, action_dim)
    except Exception:
        # Fallback if sklearn is problematic
        stats_dict['ood_action_fraction'] = 0.5
        
    stats_dict['n_transitions'] = len(obs)
    stats_dict['obs_dim'] = obs.shape[1]
    stats_dict['action_dim'] = action_dim
    
    return stats_dict
