import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from ml_core.utils import load_config


def plot_gradient(config, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    path_1 = config["plot"]["plot_seed_1"]
    plot_1_data = f"{path_1}/metrics.csv"
    seed_config_1_path = f"{path_1}/config.yaml"

    with open(seed_config_1_path, "r") as f:
        seed_config_1 = yaml.safe_load(f)
    seed_1 = seed_config_1["seed"]

    path_2 = config["plot"]["plot_seed_2"]
    plot_2_data = f"{path_2}/metrics.csv"
    seed_config_2_path = f"{path_2}/config.yaml"

    with open(seed_config_2_path, "r") as f:
        seed_config_2 = yaml.safe_load(f)
    seed_2 = seed_config_2["seed"]

    path_3 = config["plot"]["plot_seed_3"]
    plot_3_data = f"{path_3}/metrics.csv"
    seed_config_3_path = f"{path_3}/config.yaml"

    with open(seed_config_3_path, "r") as f:
        seed_config_3 = yaml.safe_load(f)
    seed_3 = seed_config_3["seed"]

    df_1 = pd.read_csv(plot_1_data)
    df_2 = pd.read_csv(plot_2_data)
    df_3 = pd.read_csv(plot_3_data)

    data_1 = df_1["all_grads"].apply(ast.literal_eval)
    data_2 = df_2["all_grads"].apply(ast.literal_eval)
    data_3 = df_3["all_grads"].apply(ast.literal_eval)

    for i, all_grads in enumerate(data_1[:3]):
        axes[0, 0].plot(range(len(all_grads)), all_grads, label=f"Epoch {i + 1}")
    axes[0, 0].set_title(f"Gradient Norm seed {seed_1}")
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Gradient norm")
    axes[0, 0].legend()

    for i, all_grads in enumerate(data_2[:3]):
        axes[0, 1].plot(range(len(all_grads)), all_grads, label=f"Epoch {i + 1}")
    axes[0, 1].set_title(f"Gradient Norm seed {seed_2}")
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Gradient norm")
    axes[0, 1].legend()

    for i, all_grads in enumerate(data_3[:3]):
        axes[1, 0].plot(range(len(all_grads)), all_grads, label=f"Epoch {i + 1}")
    axes[1, 0].set_title(f"Gradient Norm seed {seed_3}")
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("Gradient norm")
    axes[1, 0].legend()

    path_lr = config["plot"]["plot_learning_rate"]
    plot_lr_data = f"{path_lr}/metrics.csv"
    seed_config_lr_path = f"{path_lr}/config.yaml"

    df_lr = pd.read_csv(plot_lr_data)

    with open(seed_config_lr_path, "r") as f:
        seed_config_lr = yaml.safe_load(f)
    seed_lr = seed_config_lr["seed"]

    axes[1, 1].plot(df_lr["epoch"], df_lr["learning_rate"], label=f"seed {seed_lr}")
    axes[1, 1].set_title("learning rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].legend()

    plt.tight_layout()

    output_path = Path(output_path)
    if output_path:
        output_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path / "metrics.png")
    else:
        plt.show()


def main():
    config = load_config("experiments/configs/train_config.yaml")
    plot_gradient(config, "experiments/results/Trainer_test")


if __name__ == "__main__":
    main()
