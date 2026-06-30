import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans

def split_into_trajectories(dataset: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """
    Split flat D4RL dataset into list of trajectory dicts.
    Each trajectory has: observations, actions, rewards, next_observations, terminals
    Uses terminals (and timeouts if present) to identify episode boundaries.
    """
    trajectories = []
    start = 0
    N = len(dataset['observations'])
    
    terminals = dataset['terminals'].astype(bool)
    timeouts = dataset.get('timeouts', np.zeros(N, dtype=bool)).astype(bool)
    
    ends = np.where(terminals | timeouts)[0]
    
    if len(ends) == 0 or ends[-1] != N - 1:
        ends = np.append(ends, N - 1)

    for end in ends:
        traj = {
            'observations': dataset['observations'][start:end + 1],
            'actions': dataset['actions'][start:end + 1],
            'rewards': dataset['rewards'][start:end + 1],
            'next_observations': dataset['next_observations'][start:end + 1],
            'terminals': dataset['terminals'][start:end + 1],
        }
        if 'timeouts' in dataset:
            traj['timeouts'] = dataset['timeouts'][start:end + 1]
            
        trajectories.append(traj)
        start = end + 1

    return trajectories


def trajectories_to_dataset(trajectories: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate list of trajectory dicts back into flat dataset dict."""
    if not trajectories:
        return {}
        
    keys = trajectories[0].keys()
    dataset = {}
    for key in keys:
        dataset[key] = np.concatenate([t[key] for t in trajectories], axis=0)
    return dataset


# ─── Protocol 1: Coverage Reduction ────────────────────────────────────────────

def coverage_reduction(
    dataset: Dict[str, np.ndarray],
    fraction: float,
    seed: int = 42,
    n_clusters: int = 50
) -> Dict[str, np.ndarray]:
    """
    Reduce dataset coverage by clustering states and sampling one trajectory per cluster.
    """
    np.random.seed(seed)
    trajectories = split_into_trajectories(dataset)
    
    if fraction >= 1.0:
        return dataset
        
    n_trajs = len(trajectories)
    n_keep = max(1, int(n_trajs * fraction))
    
    # Compute mean state per trajectory
    mean_states = np.array([np.mean(t['observations'], axis=0) for t in trajectories])
    
    # Cluster trajectory means
    kmeans = KMeans(n_clusters=min(n_clusters, n_trajs), random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(mean_states)
    
    # Group trajectories by cluster
    clusters = {i: [] for i in range(min(n_clusters, n_trajs))}
    for i, label in enumerate(cluster_labels):
        clusters[label].append(i)
        
    # Sample fraction of trajectories, stratified by cluster
    # To ensure nested subsets, we use a deterministic order based on distance to cluster center
    distances = kmeans.transform(mean_states)
    
    keep_indices = []
    
    # We want to pick trajectories such that smaller fractions are subsets of larger ones
    # Best way: order all trajectories by priority, then take top n_keep
    
    # Priority = distance to cluster center (smaller is higher priority) + cluster index offset
    # This ensures we pick from different clusters first
    priorities = []
    for i in range(n_trajs):
        label = cluster_labels[i]
        dist = distances[i, label]
        
        # Rank within cluster
        cluster_trajs = clusters[label]
        cluster_dists = [distances[idx, label] for idx in cluster_trajs]
        rank_in_cluster = sum(1 for d in cluster_dists if d < dist)
        
        # Global priority: pick 1st from each cluster, then 2nd from each, etc.
        priority = rank_in_cluster * n_clusters + label
        priorities.append((priority, i))
        
    priorities.sort()
    keep_indices = [idx for _, idx in priorities[:n_keep]]
    
    keep_indices.sort() # Keep original temporal order somewhat intact
    kept_trajectories = [trajectories[i] for i in keep_indices]
    
    return trajectories_to_dataset(kept_trajectories)


def get_coverage_variants(dataset: Dict[str, np.ndarray], seed: int = 42) -> Dict[str, Dict]:
    """Return all 5 coverage variants."""
    fractions = {'100': 1.0, '75': 0.75, '50': 0.50, '25': 0.25, '10': 0.10}
    return {k: coverage_reduction(dataset, v, seed=seed) for k, v in fractions.items()}


# ─── Protocol 2: Quality Degradation ───────────────────────────────────────────

def quality_degradation(
    expert_dataset: Dict[str, np.ndarray],
    random_dataset: Dict[str, np.ndarray],
    expert_fraction: float,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Mix expert and random trajectories at a controlled ratio."""
    np.random.seed(seed)
    
    expert_trajs = split_into_trajectories(expert_dataset)
    random_trajs = split_into_trajectories(random_dataset)
    
    total_size = min(len(expert_trajs), len(random_trajs))
    
    n_expert_keep = int(total_size * expert_fraction)
    n_random_keep = total_size - n_expert_keep
    
    expert_indices = np.random.choice(len(expert_trajs), n_expert_keep, replace=False)
    random_indices = np.random.choice(len(random_trajs), n_random_keep, replace=False)
    
    kept_expert = [expert_trajs[i] for i in expert_indices]
    kept_random = [random_trajs[i] for i in random_indices]
    
    mixed_trajectories = kept_expert + kept_random
    np.random.shuffle(mixed_trajectories) # Shuffle trajectory order
    
    return trajectories_to_dataset(mixed_trajectories)


def get_quality_variants(
    expert_dataset: Dict[str, np.ndarray],
    random_dataset: Dict[str, np.ndarray],
    seed: int = 42
) -> Dict[str, Dict]:
    """Return Q100, Q75, Q50, Q25, Q0 quality variants."""
    fractions = {'Q100': 1.0, 'Q75': 0.75, 'Q50': 0.50, 'Q25': 0.25, 'Q0': 0.0}
    return {k: quality_degradation(expert_dataset, random_dataset, v, seed) for k, v in fractions.items()}


# ─── Protocol 3: Gaussian Noise Injection ──────────────────────────────────────

def noise_injection(
    dataset: Dict[str, np.ndarray],
    epsilon: float,
    seed: int = 42
) -> Dict[str, np.ndarray]:
    """Add Gaussian noise to observations."""
    np.random.seed(seed)
    noisy_dataset = {k: np.copy(v) for k, v in dataset.items()}
    
    if epsilon > 0:
        noise = np.random.normal(0, 1, size=noisy_dataset['observations'].shape) * epsilon
        noisy_dataset['observations'] += noise
        
        # Apply same noise process to next_observations
        next_noise = np.random.normal(0, 1, size=noisy_dataset['next_observations'].shape) * epsilon
        noisy_dataset['next_observations'] += next_noise
        
    return noisy_dataset


def get_noise_variants(dataset: Dict[str, np.ndarray], seed: int = 42) -> Dict[str, Dict]:
    """Return noise variants."""
    epsilons = {'0.00': 0.0, '0.01': 0.01, '0.05': 0.05, '0.10': 0.10, '0.25': 0.25}
    return {k: noise_injection(dataset, v, seed) for k, v in epsilons.items()}
