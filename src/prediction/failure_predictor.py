import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class FailurePredictor:
    """
    Predicts offline RL algorithm performance from dataset statistics alone.
    """
    
    FEATURE_NAMES = [
        'coverage_entropy',
        'mean_return',
        'std_return', 
        'skewness',
        'action_diversity',
        'mean_length',
        'ood_action_fraction',
        'n_transitions',
    ]
    
    def __init__(self, algorithm: str = 'CQL'):
        """algorithm: 'CQL' or 'IQL' (separate predictor per algorithm)"""
        self.algorithm = algorithm
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=1.0))
        ])
        self.is_fitted = False
        self._feature_importances = None
    
    def _extract_features(self, dataset_stats: Dict[str, float]) -> np.ndarray:
        """Extract feature vector from dataset stats dict."""
        return np.array([dataset_stats.get(f, 0.0) for f in self.FEATURE_NAMES])
    
    def fit(
        self,
        dataset_stats_list: List[Dict[str, float]],
        performance_scores: List[float]
    ) -> Dict[str, float]:
        """
        Fit predictor on (dataset_stats, performance) pairs.
        Returns dict of training metrics: r2_score, spearman_rho, rmse.
        """
        X = np.array([self._extract_features(s) for s in dataset_stats_list])
        y = np.array(performance_scores)
        
        self.pipeline.fit(X, y)
        self.is_fitted = True
        
        # Feature importances from regression coefficients (after scaling)
        coef = self.pipeline.named_steps['regressor'].coef_
        self._feature_importances = dict(zip(self.FEATURE_NAMES, coef))
        
        # Training metrics
        y_pred = self.pipeline.predict(X)
        rho, pval = stats.spearmanr(y, y_pred)
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        r2 = self.pipeline.score(X, y)
        
        return {
            'spearman_rho': float(rho) if not np.isnan(rho) else 0.0,
            'spearman_pval': float(pval) if not np.isnan(pval) else 1.0,
            'rmse': float(rmse),
            'r2': float(r2)
        }
    
    def predict(self, dataset_stats: Dict[str, float]) -> float:
        """Predict normalized score for a single dataset."""
        if not self.is_fitted:
            raise ValueError("Predictor is not fitted yet.")
        X = self._extract_features(dataset_stats).reshape(1, -1)
        return float(self.pipeline.predict(X)[0])
    
    def predict_batch(self, dataset_stats_list: List[Dict[str, float]]) -> np.ndarray:
        """Predict normalized scores for multiple datasets."""
        if not self.is_fitted:
            raise ValueError("Predictor is not fitted yet.")
        X = np.array([self._extract_features(s) for s in dataset_stats_list])
        return self.pipeline.predict(X)
    
    def evaluate(
        self,
        test_stats_list: List[Dict[str, float]],
        test_scores: List[float]
    ) -> Dict[str, float]:
        """
        Evaluate predictor on held-out data.
        Returns: {'spearman_rho', 'spearman_pval', 'rmse', 'rank_accuracy'}
        """
        if not self.is_fitted:
            raise ValueError("Predictor is not fitted yet.")
            
        y_true = np.array(test_scores)
        y_pred = self.predict_batch(test_stats_list)
        
        rho, pval = stats.spearmanr(y_true, y_pred)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        # Rank accuracy: fraction of concordant pairs
        # A simple approximation for small sets
        concordant = 0
        total_pairs = 0
        n = len(y_true)
        for i in range(n):
            for j in range(i+1, n):
                true_diff = y_true[i] - y_true[j]
                pred_diff = y_pred[i] - y_pred[j]
                if (true_diff > 0 and pred_diff > 0) or (true_diff < 0 and pred_diff < 0):
                    concordant += 1
                total_pairs += 1
                
        rank_acc = concordant / total_pairs if total_pairs > 0 else 0.0
        
        return {
            'spearman_rho': float(rho) if not np.isnan(rho) else 0.0,
            'spearman_pval': float(pval) if not np.isnan(pval) else 1.0,
            'rmse': float(rmse),
            'rank_accuracy': float(rank_acc)
        }
    
    def feature_importance(self) -> Dict[str, float]:
        """Return feature importance (regression coefficients) sorted by magnitude."""
        if not self._feature_importances:
            return {}
        return dict(sorted(self._feature_importances.items(), key=lambda x: abs(x[1]), reverse=True))
    
    def save(self, path: str) -> None:
        """Save predictor state to JSON."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({'pipeline': self.pipeline, 'algorithm': self.algorithm,
                         'feature_names': self.FEATURE_NAMES,
                         'feature_importances': self._feature_importances}, f)
    
    @classmethod
    def load(cls, path: str) -> 'FailurePredictor':
        """Load predictor from saved file."""
        import pickle
        with open(path, 'rb') as f:
            state = pickle.load(f)
        predictor = cls(algorithm=state['algorithm'])
        predictor.pipeline = state['pipeline']
        predictor.is_fitted = True
        predictor._feature_importances = state['feature_importances']
        return predictor


def cross_environment_validation(
    train_envs: List[str],
    test_env: str,
    algorithm: str,
    results_dir: str
) -> Dict[str, float]:
    """
    Train predictor on train_envs, validate on test_env.
    Loads experiment results from results_dir.
    """
    import glob
    
    def load_data(envs):
        stats_list = []
        scores = []
        for env in envs:
            # Assuming results are saved in directories per env/algo
            pattern = os.path.join(results_dir, 'degradation', '**', f'{env}_{algorithm.lower()}*.json')
            for filepath in glob.glob(pattern, recursive=True):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                    
                    if 'dataset_stats' in data and 'final_normalized_score' in data:
                        stats_list.append(data['dataset_stats'])
                        scores.append(data['final_normalized_score'])
                except Exception as e:
                    print(f"Error loading {filepath}: {e}")
        return stats_list, scores
        
    train_stats, train_scores = load_data(train_envs)
    test_stats, test_scores = load_data([test_env])
    
    if not train_stats or not test_stats:
        print("Warning: Insufficient data for cross-validation")
        return {}
        
    predictor = FailurePredictor(algorithm=algorithm)
    train_metrics = predictor.fit(train_stats, train_scores)
    test_metrics = predictor.evaluate(test_stats, test_scores)
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'feature_importances': predictor.feature_importance()
    }
