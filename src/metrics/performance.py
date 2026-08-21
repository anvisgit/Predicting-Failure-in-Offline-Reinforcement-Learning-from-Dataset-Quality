import numpy as np
import gymnasium as gym
from typing import Dict, Tuple

# D4RL reference scores for normalization
REF_SCORES = {
    'halfcheetah': {'min': -280.178953, 'max': 12135.0},
    'hopper':      {'min': -20.272305,  'max': 3234.3},
    'walker2d':    {'min': 1.629008,    'max': 4592.3},
}

def evaluate_policy(env: gym.Env, predict_fn, n_episodes: int = 10) -> Dict[str, float]:
    """
    Run policy in environment for n_episodes.
    Returns: {'mean_return': float, 'std_return': float, 'mean_length': float}
    predict_fn: callable(observation_array) -> action_array
    """
    returns = []
    lengths = []
    
    for _ in range(n_episodes):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        done = False
        episode_return = 0.0
        episode_length = 0
        
        while not done:
            action = predict_fn(obs)
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            episode_return += reward
            episode_length += 1
            
        returns.append(episode_return)
        lengths.append(episode_length)
        
    return {
        'mean_return': float(np.mean(returns)),
        'std_return': float(np.std(returns)),
        'mean_length': float(np.mean(lengths))
    }

def compute_normalized_score(env_name: str, raw_score: float) -> float:
    """
    D4RL normalized score: (score - ref_min) / (ref_max - ref_min) * 100
    env_name: 'halfcheetah', 'hopper', or 'walker2d'
    """
    if env_name not in REF_SCORES:
        raise ValueError(f"Unknown environment for normalization: {env_name}")
        
    ref_min = REF_SCORES[env_name]['min']
    ref_max = REF_SCORES[env_name]['max']
    
    return (raw_score - ref_min) / (ref_max - ref_min) * 100.0

def evaluate_and_normalize(env: gym.Env, env_name: str, predict_fn, n_episodes: int = 10) -> Dict[str, float]:
    """Evaluate policy and return both raw and normalized scores."""
    eval_results = evaluate_policy(env, predict_fn, n_episodes)
    raw_score = eval_results['mean_return']
    normalized_score = compute_normalized_score(env_name, raw_score)
    
    eval_results['normalized_score'] = normalized_score
    return eval_results
