import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()


def load_data(file_path: Path) -> pd.DataFrame:
    # TODO: Load CSV into Pandas DataFrame
    df = pd.read_csv(file_path)
    df.columns = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_f1",
        "val_avg_loss",
        "val_accuracy",
        "val_f1",
        "grad_norm",
        "learning_rate",
    ]
    return df


def setup_style():
    # TODO: Set seaborn theme
    plt.style.use("default")


def plot_metrics(df: pd.DataFrame, output_path: Optional[Path]):
    """
    Generate and save plots for Loss, Accuracy, and F1.
    """
    if df.empty:
        return

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # TODO: Plot Train/Val Loss
    axes[0, 0].plot(df["epoch"], df["train_loss"], label="Train Loss")
    axes[0, 0].plot(df["epoch"], df["val_avg_loss"], label="Val Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()

    # TODO: Plot Train/Val Accuracy
    axes[0, 1].plot(df["epoch"], df["train_accuracy"], label="Train Acc")
    axes[0, 1].plot(df["epoch"], df["val_accuracy"], label="Val Acc")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].legend()

    # TODO: Plot Learning Rate
    axes[1, 0].plot(df["epoch"], df["learning_rate"], label="LR")
    axes[1, 0].set_title("Learning Rate")
    axes[1, 0].legend()

    # Hide empty subplot
    axes[1, 1].axis("off")

    plt.tight_layout()

    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / "metrics.png")
    else:
        plt.show()


def main():
    args = parse_args()
    setup_style()
    df = load_data(args.input_csv)
    plot_metrics(df, args.output_dir)


if __name__ == "__main__":
    main()
