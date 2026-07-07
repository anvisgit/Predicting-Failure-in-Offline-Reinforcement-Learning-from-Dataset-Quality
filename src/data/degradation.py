from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans


def split_into_trajectories(dataset: Dict[str, np.ndarray]) -> List[Dict[str, np.ndarray]]:
    """Split a flat D4RL dataset into trajectory dictionaries."""
    trajectories = []
    start = 0
    n_items = len(dataset["observations"])

    terminals = dataset["terminals"].astype(bool)
    timeouts = dataset.get("timeouts", np.zeros(n_items, dtype=bool)).astype(bool)
    ends = np.where(terminals | timeouts)[0]

    if len(ends) == 0 or ends[-1] != n_items - 1:
        ends = np.append(ends, n_items - 1)

    for end in ends:
        traj = {
            "observations": dataset["observations"][start : end + 1],
            "actions": dataset["actions"][start : end + 1],
            "rewards": dataset["rewards"][start : end + 1],
            "next_observations": dataset["next_observations"][start : end + 1],
            "terminals": dataset["terminals"][start : end + 1],
        }
        if "timeouts" in dataset:
            traj["timeouts"] = dataset["timeouts"][start : end + 1]

        trajectories.append(traj)
        start = end + 1

    return trajectories


def trajectories_to_dataset(trajectories: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Concatenate trajectory dictionaries back into a flat dataset."""
    if not trajectories:
        return {}

    return {
        key: np.concatenate([traj[key] for traj in trajectories], axis=0)
        for key in trajectories[0].keys()
    }


def coverage_reduction(
    dataset: Dict[str, np.ndarray],
    fraction: float,
    seed: int = 42,
    n_clusters: int = 50,
) -> Dict[str, np.ndarray]:
    """Reduce state coverage by selecting representative trajectories."""
    trajectories = split_into_trajectories(dataset)

    if fraction >= 1.0:
        return {key: np.copy(value) for key, value in dataset.items()}

    n_trajs = len(trajectories)
    n_keep = max(1, int(n_trajs * fraction))
    n_clusters = min(n_clusters, n_trajs)

    mean_states = np.array([np.mean(traj["observations"], axis=0) for traj in trajectories])
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = kmeans.fit_predict(mean_states)
    distances = kmeans.transform(mean_states)

    clusters = {idx: [] for idx in range(n_clusters)}
    for traj_idx, label in enumerate(cluster_labels):
        clusters[label].append(traj_idx)

    priorities = []
    for traj_idx, label in enumerate(cluster_labels):
        dist = distances[traj_idx, label]
        cluster_dists = [distances[idx, label] for idx in clusters[label]]
        rank_in_cluster = sum(1 for cluster_dist in cluster_dists if cluster_dist < dist)
        priorities.append((rank_in_cluster * n_clusters + label, traj_idx))

    priorities.sort()
    keep_indices = sorted(idx for _, idx in priorities[:n_keep])
    return trajectories_to_dataset([trajectories[idx] for idx in keep_indices])


def get_coverage_variants(dataset: Dict[str, np.ndarray], seed: int = 42) -> Dict[str, Dict]:
    fractions = {"100": 1.0, "75": 0.75, "50": 0.50, "25": 0.25, "10": 0.10}
    return {key: coverage_reduction(dataset, value, seed=seed) for key, value in fractions.items()}


def quality_degradation(
    expert_dataset: Dict[str, np.ndarray],
    random_dataset: Dict[str, np.ndarray],
    expert_fraction: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Mix expert and random trajectories at a controlled ratio."""
    rng = np.random.default_rng(seed)
    expert_trajs = split_into_trajectories(expert_dataset)
    random_trajs = split_into_trajectories(random_dataset)

    total_size = min(len(expert_trajs), len(random_trajs))
    n_expert_keep = int(total_size * expert_fraction)
    n_random_keep = total_size - n_expert_keep

    expert_indices = rng.choice(len(expert_trajs), n_expert_keep, replace=False)
    random_indices = rng.choice(len(random_trajs), n_random_keep, replace=False)

    mixed_trajectories = [expert_trajs[idx] for idx in expert_indices]
    mixed_trajectories.extend(random_trajs[idx] for idx in random_indices)
    rng.shuffle(mixed_trajectories)

    return trajectories_to_dataset(mixed_trajectories)


def get_quality_variants(
    expert_dataset: Dict[str, np.ndarray],
    random_dataset: Dict[str, np.ndarray],
    seed: int = 42,
) -> Dict[str, Dict]:
    fractions = {"Q100": 1.0, "Q75": 0.75, "Q50": 0.50, "Q25": 0.25, "Q0": 0.0}
    return {
        key: quality_degradation(expert_dataset, random_dataset, value, seed)
        for key, value in fractions.items()
    }


def noise_injection(
    dataset: Dict[str, np.ndarray],
    epsilon: float,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Add Gaussian observation noise while preserving dtype."""
    rng = np.random.default_rng(seed)
    noisy_dataset = {key: np.copy(value) for key, value in dataset.items()}

    if epsilon > 0:
        obs_noise = rng.normal(0, epsilon, size=noisy_dataset["observations"].shape)
        next_obs_noise = rng.normal(0, epsilon, size=noisy_dataset["next_observations"].shape)
        noisy_dataset["observations"] = (
            noisy_dataset["observations"] + obs_noise
        ).astype(np.float32)
        noisy_dataset["next_observations"] = (
            noisy_dataset["next_observations"] + next_obs_noise
        ).astype(np.float32)

    return noisy_dataset


def get_noise_variants(dataset: Dict[str, np.ndarray], seed: int = 42) -> Dict[str, Dict]:
    epsilons = {"0.00": 0.0, "0.01": 0.01, "0.05": 0.05, "0.10": 0.10, "0.25": 0.25}
    return {key: noise_injection(dataset, value, seed) for key, value in epsilons.items()}
