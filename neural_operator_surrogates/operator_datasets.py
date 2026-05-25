import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import ExperimentConfig


def periodic_distance(
    x: np.ndarray,
    centers: np.ndarray,
    length: float,
) -> np.ndarray:
    d = np.abs(x[:, None] - centers[None, :])
    return np.minimum(d, length - d)


def fractional_laplacian_periodic(
    u: np.ndarray,
    length: float,
    dx: float,
    frac_power_s: float,
) -> np.ndarray:
    n_grid = u.shape[-1]

    frequencies = np.fft.fftfreq(n_grid, d=dx)
    wave_numbers = 2.0 * np.pi * frequencies

    multiplier = np.abs(wave_numbers) ** (2.0 * frac_power_s)
    multiplier[0] = 0.0

    u_hat = np.fft.fft(u, axis=-1)
    y = np.fft.ifft(multiplier[None, :] * u_hat, axis=-1).real

    return y.astype(np.float32)


def generate_fractional_laplacian_dataset(
    n_samples: int,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    task = config.task

    length = task.x_max - task.x_min
    dx = length / task.n_grid

    x = np.linspace(
        task.x_min,
        task.x_max,
        task.n_grid,
        endpoint=False,
        dtype=np.float32,
    )

    u_all = np.empty((n_samples, task.n_grid), dtype=np.float32)

    for i in range(n_samples):
        n_gaussians = rng.integers(
            task.n_gaussians_min,
            task.n_gaussians_max + 1,
        )

        centers = rng.uniform(
            task.x_min,
            task.x_max,
            size=n_gaussians,
        )

        amplitudes = rng.uniform(
            task.amplitude_min,
            task.amplitude_max,
            size=n_gaussians,
        )

        sigmas = rng.uniform(
            task.sigma_min,
            task.sigma_max,
            size=n_gaussians,
        )

        d = periodic_distance(x, centers, length)

        u = np.sum(
            amplitudes[None, :]
            * np.exp(-(d**2) / (2.0 * sigmas[None, :] ** 2)),
            axis=1,
        )

        u = u - np.mean(u)
        u = u / np.max(np.abs(u))

        u_all[i] = u.astype(np.float32)

    y_all = fractional_laplacian_periodic(
        u=u_all,
        length=length,
        dx=dx,
        frac_power_s=task.frac_power_s,
    )

    return x, u_all, y_all


def build_fractional_laplacian_dataloaders(config: ExperimentConfig):
    rng = np.random.default_rng(config.training.seed)

    x, u_train, y_train = generate_fractional_laplacian_dataset(
        n_samples=config.split.n_train,
        config=config,
        rng=rng,
    )

    _, u_val, y_val = generate_fractional_laplacian_dataset(
        n_samples=config.split.n_val,
        config=config,
        rng=rng,
    )

    _, u_test, y_test = generate_fractional_laplacian_dataset(
        n_samples=config.split.n_test,
        config=config,
        rng=rng,
    )

    y_mean = y_train.mean()
    y_std = y_train.std()

    y_train_normalized = (y_train - y_mean) / y_std
    y_val_normalized = (y_val - y_mean) / y_std
    y_test_normalized = (y_test - y_mean) / y_std

    train_dataset = TensorDataset(
        torch.tensor(u_train[..., None], dtype=torch.float32),
        torch.tensor(y_train_normalized[..., None], dtype=torch.float32),
    )

    val_dataset = TensorDataset(
        torch.tensor(u_val[..., None], dtype=torch.float32),
        torch.tensor(y_val_normalized[..., None], dtype=torch.float32),
    )

    test_dataset = TensorDataset(
        torch.tensor(u_test[..., None], dtype=torch.float32),
        torch.tensor(y_test_normalized[..., None], dtype=torch.float32),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.split.batch_size,
        shuffle=True,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.split.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.split.batch_size,
        shuffle=False,
        pin_memory=True,
    )

    dataset_info = {
        "x": x,
        "y_mean": float(y_mean),
        "y_std": float(y_std),
    }

    return train_loader, val_loader, test_loader, dataset_info


def build_operator_dataset(config: ExperimentConfig):
    if config.task.name == "fractional_laplacian":
        return build_fractional_laplacian_dataloaders(config)

    raise ValueError(f"Unknown operator approximation task: {config.task.name}")