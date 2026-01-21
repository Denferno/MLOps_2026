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
    seed_everything(config["seed"])

    

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Your using {device}")

    # 3. Data
    train_loader, val_loader = get_dataloaders(config)

    # 4. Model
    model = MLP(
        input_shape=config["data"]["input_shape"],
        hidden_units=config["model"]["hidden_units"],
        dropout_rate=config["model"]["dropout_rate"],
        num_classes=config["model"]["num_classes"],
    )

    # 5. Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        momentum=config["training"].get("momentum", 0.9),
    )

    # 6. Trainer & Fit
    trainer = Trainer(model=model, optimizer=optimizer, config=config, device=device)
    if args.resume:
        ckpt = trainer.load_checkpoint(args.resume)
        logger.info(
            f"Loaded checkpoint: {args.resume} (epoch={ckpt.get('epoch')}, val_loss={ckpt.get('val_loss')})"
        )

    trainer.fit(train_loader, val_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    main(args)
    print("Skeleton: Implement main logic first.")
