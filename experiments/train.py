import argparse
import torch
import torch.optim as optim
from ml_core.data import get_dataloaders
from ml_core.models import MLP
from ml_core.solver import Trainer
from ml_core.utils import load_config, seed_everything, setup_logger

logger = setup_logger("Experiment_Runner")

def main(args):
    # 1. Load Config & Set Seed
    config = load_config(args.config)
    seed = seed_everything(0)
    
    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f'Your using {device}')

    # 3. Data
    train_loader, val_loader = get_dataloaders(config)

    # 4. Model
    model = MLP()
    
    # 5. Optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # 6. Trainer & Fit
    # trainer = Trainer(...)
    # trainer.fit(...)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    main(args)
    print("Skeleton: Implement main logic first.")
