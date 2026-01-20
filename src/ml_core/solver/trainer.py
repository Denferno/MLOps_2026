import time
from typing import Any, Dict, Tuple
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim import lr_scheduler
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
        
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=self.config["training"]["factor"], 
            patience=self.config["training"]["patience"],
            min_lr=self.config["training"]["min_lr"])

        # TODO: Initialize metric calculation (like accuracy/f1-score) if needed
        self.train_loss = 0.0
        self.train_correct = 0
        self.train_total = 0

        self.val_loss = 0.0
        self.val_correct = 0
        self.val_total = 0

        self.accuracy = 0.0
        self.f1score = 0.0
        self.gradient_norm = 0.0

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()

        # reset epoch stats
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_pred = []
        all_labels = []
        
        for image, label in tqdm(dataloader, desc=f"Train {epoch_idx+1}", leave=True, position=0):

            # move data to the device
            image = image.to(self.device)
            label = label.to(self.device)

            # reset gradient
            self.optimizer.zero_grad()

            # forward
            output = self.model(image)
            
            # compute loss
            loss = self.criterion(output, label)

            # backpropagation 
            loss.backward()

            # grad norm
            total_norm = utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm = self.config["training"]["max_grad_norm"],
                norm_type = 2
            )
            grad_norm = total_norm.item()
            # optimization
            self.optimizer.step()
                    
            batch_size = label.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

            preds = torch.argmax(output, dim=1)
            total_correct += (preds == label).sum().item()

            all_pred.extend(preds.detach().cpu().numpy())
            all_labels.extend(label.cpu().numpy())

        avg_loss = running_loss / total_samples
        accuracy = total_correct / total_samples
        f1 = f1_score(all_labels, all_pred, average='binary')
        
        return avg_loss, accuracy, f1, grad_norm
 
    
    def validate(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.eval()
        
        # TODO: Implement Validation Loop
        # Remember: No gradients needed here

        #Het verschil is dat validation geen learning doet en alleen testen. Ook heeft het geen gradient berekening. En zit het in eval mode 
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_pred = []
        all_labels = []

        with torch.no_grad():
            for image, label in tqdm(dataloader, desc=f"Val {epoch_idx+1}", leave=False, position=0):
                image = image.to(self.device)
                label = label.to(self.device)

                # dit is de forward
                output = self.model(image)

                # compute loss
                loss = self.criterion(output, label)
                
                # hier zit dus geen backpropagation (in train_ep wel, dus gaat het niet leren

                batch_size = label.size(0)
                running_loss += loss.item() * batch_size
                total_samples += batch_size

                preds = torch.argmax(output, dim=1)
                total_correct += (preds == label).sum().item()

                all_pred.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        avg_loss = running_loss / total_samples
        accuracy = total_correct / total_samples
        f1 = f1_score(all_labels, all_pred, average='binary')

        return avg_loss, accuracy, f1

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        # TODO: Save model state, optimizer state, and config
        checkpoints = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": float(val_loss),
            "config": self.config,
        }
        torch.save(checkpoints, "checkpoint.pt")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]
        
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):           
            # TODO: Call train_epoch and validate
            train_avg_loss, train_acc, train_f1, grad_norm = self.train_epoch(train_loader, epoch)
            val_avg_loss, val_acc, val_f1 = self.validate(val_loader, epoch)
            # TODO: Log metrics to tracker

            lr_before = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_avg_loss)
            lr_after = self.optimizer.param_groups[0]['lr']

            if lr_before != lr_after:
                print(f'learning rate reduced from {lr_before} to {lr_after}')
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train: L={train_avg_loss:.4f} A={train_acc*100:5.2f}% F1={train_f1:.3f} Grad={grad_norm:.3f}| "
                f"Val: L={val_avg_loss:.4f} A={val_acc*100:5.2f}% F1={val_f1:.3f}")
            self.tracker.log_metrics(
                epoch,{'train_loss': train_avg_loss,
                       'train_accuracy': train_acc,
                       'train_f1': train_f1,
                       'val_avg_loss':val_avg_loss,
                       'val_accuracy': val_acc,
                       'val_f1': val_f1,
                       'grad_norm': grad_norm,
                       'learning_rate': lr_after 
                       }
            ) 
            # TODO: Save checkpoints
            self.save_checkpoint(epoch, val_avg_loss)
        self.tracker.close()
        
            
	# Remember to handle the trackers properly
