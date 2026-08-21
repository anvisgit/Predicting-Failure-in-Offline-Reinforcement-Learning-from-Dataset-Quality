import os
import json
import csv
from typing import Dict, Any, Optional
from pathlib import Path

class ExperimentLogger:
    """Handles logging to wandb (if available) and local CSV/JSON files."""
    
    def __init__(self, config: Dict[str, Any], run_name: str, use_wandb: bool = False):
        self.config = config
        self.run_name = run_name
        self.use_wandb = use_wandb
        
        # Setup local results directory
        self.results_dir = Path(config.get('results_dir', 'results')) / run_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.results_dir / 'metrics.csv'
        self.eval_file = self.results_dir / 'eval.csv'
        
        self.metrics_initialized = False
        self.eval_initialized = False
        
        if self.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=config.get('wandb_project', 'offline-rl-research'),
                    name=run_name,
                    config=config
                )
            except ImportError:
                print("WARNING: wandb not installed. Falling back to local logging.")
                self.use_wandb = False
    
    def _write_csv(self, filepath: Path, data: Dict[str, Any], initialized: bool) -> bool:
        """Write dict to CSV. Creates header if not initialized."""
        file_exists = filepath.exists()
        with open(filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not initialized and not file_exists:
                writer.writeheader()
            writer.writerow(data)
        return True
    
    def log_metrics(self, step: int, metrics_dict: Dict[str, Any]):
        """Log training metrics."""
        log_data = {'step': step, **metrics_dict}
        self.metrics_initialized = self._write_csv(self.metrics_file, log_data, self.metrics_initialized)
        
        if self.use_wandb:
            import wandb
            wandb.log(log_data, step=step)
            
    def log_evaluation(self, step: int, eval_dict: Dict[str, Any]):
        """Log evaluation metrics."""
        log_data = {'step': step, **eval_dict}
        self.eval_initialized = self._write_csv(self.eval_file, log_data, self.eval_initialized)
        
        if self.use_wandb:
            import wandb
            wandb.log(log_data, step=step)
            
    def log_dataset_stats(self, stats_dict: Dict[str, Any]):
        """Log dataset statistics (usually once at start)."""
        stats_file = self.results_dir / 'dataset_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats_dict, f, indent=4)
            
        if self.use_wandb:
            import wandb
            wandb.config.update({"dataset_stats": stats_dict})
            
    def save_results(self, results_dict: Dict[str, Any], filename: str = 'results.json'):
        """Save final results dict to JSON."""
        out_file = self.results_dir / filename
        with open(out_file, 'w') as f:
            json.dump(results_dict, f, indent=4)
            
    def finish(self):
        """Clean up wandb run."""
        if self.use_wandb:
            import wandb
            wandb.finish()
