import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import ExperimentTracker, setup_logger

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # TODO: Define Loss Function (Criterion)
        self.criterion = nn.CrossEntropyLoss()

        # TODO: Initialize ExperimentTracker
        self.tracker = ExperimentTracker(experiment_name="Trainer_test", config=self.config)
        
        # TODO: Initialize metric calculation (like accuracy/f1-score) if needed
        self.train_loss = 0.0
        self.train_correct = 0
        self.train_total = 0

        self.val_loss = 0.0
        self.val_correct = 0
        self.val_total = 0

        self.accuracy = 0.0
        self.f1score = 0.0

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()

        # reset epoch stats
        self.train_loss = 0.0
        self.train_correct = 0
        self.train_total = 0

        # for binary F1 (only used if num_classes == 2)
        tp = fp = fn = 0

        for image, label in tqdm(dataloader, desc=f"Train {epoch_idx}", leave=False):
            image = image.to(self.device, non_blocking=True)
            label = label.to(self.device, non_blocking=True).long()

            # 1) forward
            output = self.model(image)

            # 2) loss
            loss = self.criterion(output, label)

            # 3) backward + step
            loss.backward()
            self.optimizer.step()

            # 4) update loss
            self.train_loss += loss.item()

            # 5) predictions + accuracy
            preds = torch.argmax(output, dim=1)  # [B]
            self.train_correct += (preds == label).sum().item()
            self.train_total += label.size(0)

            # 6) F1 if binary
            if output.size(1) == 2:
                tp += ((preds == 1) & (label == 1)).sum().item()
                fp += ((preds == 1) & (label == 0)).sum().item()
                fn += ((preds == 0) & (label == 1)).sum().item()

        # averages
        avg_loss = self.train_loss / max(len(dataloader),1)
        accuracy = self.train_correct / self.train_total

        if output.size(1) == 2:
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = (2 * precision * recall) / max(precision + recall, 1e-12)
        else:
            f1 = 0.0  # keep simple for multi-class

        return avg_loss, accuracy, f1
    
    def validate(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.eval()
        
        # TODO: Implement Validation Loop
        # Remember: No gradients needed here
        self.val_loss = 0.0
        self.val_correct = 0
        self.val_total = 0

        with torch.no_grad:
            for image, label in tqdm(dataloader):
                image = image.to(self.device)
                label = label.to(self.device)

                output = self.model(image)
                loss = self.criterion(output, label)

                self.val_loss = loss.item()

                _, pred = torch.max(output.data, 1)
                total_pred += label.size(0)
                correct_pred += (pred == label).sum().items()
                
        avg_loss = self.val_loss / len(dataloader)
        accuracy = correct_pred / total_pred

        return avg_loss, accuracy

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        # TODO: Save model state, optimizer state, and config
        pass

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # TODO: Call train_epoch and validate
            # TODO: Log metrics to tracker
            # TODO: Save checkpoints
            pass
            
	# Remember to handle the trackers properly
