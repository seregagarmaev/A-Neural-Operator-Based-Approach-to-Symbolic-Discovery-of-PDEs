from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig


def save_training_curves(
    history: dict,
    figures_dir: Path,
) -> None:
    epochs = np.arange(1, len(history["train_mse"]) + 1)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(epochs, history["train_mse"], label="train MSE")
    ax.plot(epochs, history["val_mse"], label="val MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "training_curve_mse.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(epochs, history["val_rel_l2"], label="val relative L2")
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative L2")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "training_curve_relative_l2.png", dpi=200)
    plt.close(fig)


def save_prediction_figures(
    model: torch.nn.Module,
    test_loader: DataLoader,
    dataset_info: dict,
    config: ExperimentConfig,
    figures_dir: Path,
    device: torch.device,
) -> None:
    model.eval()

    x = dataset_info["x"]
    y_mean = dataset_info["y_mean"]
    y_std = dataset_info["y_std"]
    model_label = config.model.name.upper()

    xb, yb = next(iter(test_loader))

    xb = xb.to(device, non_blocking=True)
    yb = yb.to(device, non_blocking=True)

    with torch.no_grad():
        prediction = model(xb)

    xb = xb.detach().cpu().numpy()
    yb = yb.detach().cpu().numpy()
    prediction = prediction.detach().cpu().numpy()

    y_true = yb * y_std + y_mean
    y_pred = prediction * y_std + y_mean

    n_plots = min(config.visualization.n_prediction_plots, xb.shape[0])

    for i in range(n_plots):
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

        axes[0].plot(x, xb[i, :, 0])
        axes[0].set_ylabel(r"$u(x)$")
        axes[0].set_title("Input field")

        axes[1].plot(x, y_true[i, :, 0], label="true")
        axes[1].plot(x, y_pred[i, :, 0], linestyle="--", label=model_label)
        axes[1].set_ylabel(r"$(-\Delta)^s u(x)$")
        axes[1].set_title("Fractional Laplacian")
        axes[1].legend()

        axes[2].plot(x, y_pred[i, :, 0] - y_true[i, :, 0])
        axes[2].set_xlabel(r"$x$")
        axes[2].set_ylabel("error")
        axes[2].set_title("Prediction error")

        fig.tight_layout()

        fig.savefig(figures_dir / f"prediction_{i:03d}.png", dpi=200)
        plt.close(fig)