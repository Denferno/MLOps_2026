import csv
from pathlib import Path
from typing import Any, Dict

import yaml

# TODO: Add TensorBoard Support
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        base_dir: str = "experiments/results",
    ):
        self.run_dir = Path(base_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Save config to yaml in run_dir
        config_path = self.run_dir / 'config.yaml'

        with open(config_path, 'w') as file:
            yaml.dump(config, file)

        self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))

        # Initialize CSV
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        
        # Header (TODO: add the rest of things we want to track, loss, gradients, accuracy etc.)
        self.csv_writer.writerow(["epoch", "train_loss", "train_accuracy", "train_f1", "val_avg_loss","val_accuracy","val_f1","grad_norm",""]) 

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Writes metrics to CSV (and TensorBoard).
        """
        # TODO: Write other useful metrics to CSV
        row = [epoch]
        for column in ["train_loss", "train_accuracy", "train_f1", 
                      "val_avg_loss", "val_accuracy", "val_f1", 
                      "grad_norm", "learning_rate"]:
            row.append(metrics.get(column, ""))

        self.csv_writer.writerow(row) # Currently only logging epoch
        self.csv_file.flush()

        # TODO: Log to TensorBoard
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, epoch)
        self.writer.flush()

    def get_checkpoint_path(self, filename: str) -> str:
        return str(self.run_dir / filename)

    def close(self):
        self.writer.close()
        self.csv_file.close()
