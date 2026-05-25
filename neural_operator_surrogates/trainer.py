from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import ExperimentConfig


def relative_l2(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    prediction = prediction.reshape(prediction.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    numerator = torch.linalg.norm(prediction - target, dim=1)
    denominator = torch.linalg.norm(target, dim=1)

    return torch.mean(numerator / denominator)


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_mse = 0.0
    total_rel_l2 = 0.0
    n_samples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            prediction = model(xb)

            batch_size = xb.shape[0]
            total_mse += loss_fn(prediction, yb).item() * batch_size
            total_rel_l2 += relative_l2(prediction, yb).item() * batch_size
            n_samples += batch_size

    return total_mse / n_samples, total_rel_l2 / n_samples


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    best_model_path: Path,
    dataset_info: dict,
) -> dict:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.training.scheduler_step,
        gamma=config.training.scheduler_gamma,
    )

    loss_fn = torch.nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = {
        "train_mse": [],
        "val_mse": [],
        "val_rel_l2": [],
    }

    for epoch in range(1, config.training.max_epochs + 1):
        model.train()

        total_train_loss = 0.0
        n_train_samples = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()

            prediction = model(xb)
            loss = loss_fn(prediction, yb)

            loss.backward()
            optimizer.step()

            batch_size = xb.shape[0]
            total_train_loss += loss.item() * batch_size
            n_train_samples += batch_size

        scheduler.step()

        train_mse = total_train_loss / n_train_samples
        val_mse, val_rel_l2 = evaluate(
            model=model,
            loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["val_rel_l2"].append(val_rel_l2)

        improved = val_mse < best_val_loss - config.training.early_stopping_min_delta

        if improved:
            best_val_loss = val_mse
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "y_mean": dataset_info["y_mean"],
                    "y_std": dataset_info["y_std"],
                    "best_epoch": best_epoch,
                    "best_val_mse": best_val_loss,
                    "history": history,
                },
                best_model_path,
            )
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:04d} "
                f"train_mse={train_mse:.6e} "
                f"val_mse={val_mse:.6e} "
                f"val_rel_l2={val_rel_l2:.6e} "
                f"best_epoch={best_epoch:04d} "
                f"patience={epochs_without_improvement}/"
                f"{config.training.early_stopping_patience}"
            )

        if epochs_without_improvement >= config.training.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}, "
                f"best val MSE: {best_val_loss:.6e}"
            )
            break

    return history