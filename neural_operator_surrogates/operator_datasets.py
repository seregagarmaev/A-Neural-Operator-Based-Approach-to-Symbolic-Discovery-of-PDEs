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


def generate_periodic_gaussian_mixture(
    x: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    task = config.task

    n_gaussians = rng.integers(
        task.n_gaussians_min,
        task.n_gaussians_max + 1,
    )

    centers = rng.uniform(
        0.0,
        task.L,
        size=n_gaussians,
    )

    sigmas = rng.uniform(
        task.sigma_min_fraction * task.L,
        task.sigma_max_fraction * task.L,
        size=n_gaussians,
    )

    amplitudes = rng.normal(
        size=n_gaussians,
    )

    d = periodic_distance(
        x=x,
        centers=centers,
        length=task.L,
    )

    u = np.sum(
        amplitudes[None, :]
        * np.exp(-(d**2) / (2.0 * sigmas[None, :] ** 2)),
        axis=1,
    )

    u = u - np.mean(u)
    u = u / np.max(np.abs(u))

    return u.astype(np.float32)


def spectral_first_derivative_periodic(
    u: np.ndarray,
    dx: float,
) -> np.ndarray:
    n_grid = u.shape[-1]

    frequencies = np.fft.fftfreq(n_grid, d=dx)
    wave_numbers = 2.0 * np.pi * frequencies

    u_hat = np.fft.fft(u, axis=-1)
    y = np.fft.ifft(1j * wave_numbers[None, :] * u_hat, axis=-1).real

    return y.astype(np.float32)


def spectral_second_derivative_periodic(
    u: np.ndarray,
    dx: float,
) -> np.ndarray:
    n_grid = u.shape[-1]

    frequencies = np.fft.fftfreq(n_grid, d=dx)
    wave_numbers = 2.0 * np.pi * frequencies

    u_hat = np.fft.fft(u, axis=-1)
    y = np.fft.ifft(-(wave_numbers[None, :] ** 2) * u_hat, axis=-1).real

    return y.astype(np.float32)


def fractional_laplacian_periodic(
    u: np.ndarray,
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


def poisson_gradient_periodic(
    q: np.ndarray,
    dx: float,
) -> np.ndarray:
    n_grid = q.shape[-1]

    frequencies = np.fft.fftfreq(n_grid, d=dx)
    wave_numbers = 2.0 * np.pi * frequencies

    source = q - np.mean(q, axis=-1, keepdims=True)
    source_hat = np.fft.fft(source, axis=-1)

    phi_hat = np.zeros_like(source_hat, dtype=np.complex128)

    nonzero = wave_numbers != 0.0
    phi_hat[:, nonzero] = source_hat[:, nonzero] / (wave_numbers[nonzero] ** 2)

    y = np.fft.ifft(-1j * wave_numbers[None, :] * phi_hat, axis=-1).real

    return y.astype(np.float32)


def generate_temporal_gaussian_strain_history(
    t: np.ndarray,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    task = config.task

    T = task.t_final

    n_gaussians = rng.integers(
        task.strain_gaussians_min,
        task.strain_gaussians_max + 1,
    )

    sigma_min = task.strain_sigma_min_fraction * T
    sigma_max = task.strain_sigma_max_fraction * T

    epsilon = np.zeros_like(t, dtype=np.float64)

    for _ in range(n_gaussians):
        amplitude = rng.normal()
        center = rng.uniform(0.0, T)
        sigma = rng.uniform(sigma_min, sigma_max)

        epsilon += amplitude * np.exp(
            -((t - center) ** 2) / (2.0 * sigma ** 2)
        )

    epsilon = epsilon - np.mean(epsilon)
    epsilon = task.strain_max_abs * epsilon / np.max(np.abs(epsilon))

    return epsilon.astype(np.float32)


def hereditary_integral_operator(
    epsilon: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    task = config.task

    dt = task.t_final / (task.nt - 1)
    tau = task.relaxation_time_tau
    alpha = np.exp(-dt / tau)

    weight_left = tau**2 / dt * (1.0 - alpha) - tau * alpha
    weight_right = tau - tau**2 / dt * (1.0 - alpha)

    memory = np.zeros_like(epsilon, dtype=np.float32)

    for i in range(1, task.nt):
        memory[:, i] = (
            alpha * memory[:, i - 1]
            + weight_left * epsilon[:, i - 1]
            + weight_right * epsilon[:, i]
        )

    return memory.astype(np.float32)


def generate_hereditary_integral_inputs(
    n_samples: int,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    task = config.task

    t = np.linspace(
        0.0,
        task.t_final,
        task.nt,
        endpoint=True,
        dtype=np.float32,
    )

    epsilon_all = np.empty((n_samples, task.nt), dtype=np.float32)

    for i in range(n_samples):
        epsilon_all[i] = generate_temporal_gaussian_strain_history(
            t=t,
            config=config,
            rng=rng,
        )

    return t, epsilon_all.astype(np.float32)


def generate_operator_inputs(
    n_samples: int,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    task = config.task

    if task.name == "hereditary_integral":
        return generate_hereditary_integral_inputs(
            n_samples=n_samples,
            config=config,
            rng=rng,
        )

    x = np.linspace(
        0.0,
        task.L,
        task.nx,
        endpoint=False,
        dtype=np.float32,
    )

    u_all = np.empty((n_samples, task.nx), dtype=np.float32)

    for i in range(n_samples):
        u_all[i] = generate_periodic_gaussian_mixture(
            x=x,
            config=config,
            rng=rng,
        )

    if task.name == "poisson_gradient":
        u_all = 1.0 + task.poisson_density_perturbation_amplitude * u_all

    return x, u_all.astype(np.float32)


def apply_operator(
    u: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    task = config.task

    if task.name == "hereditary_integral":
        return hereditary_integral_operator(
            epsilon=u,
            config=config,
        )

    dx = task.L / task.nx

    if task.name == "fractional_laplacian":
        return fractional_laplacian_periodic(
            u=u,
            dx=dx,
            frac_power_s=task.frac_power_s,
        )

    if task.name == "poisson_gradient":
        return poisson_gradient_periodic(
            q=u,
            dx=dx,
        )

    if task.name == "first_derivative":
        return spectral_first_derivative_periodic(
            u=u,
            dx=dx,
        )

    if task.name == "second_derivative":
        return spectral_second_derivative_periodic(
            u=u,
            dx=dx,
        )

    raise ValueError(f"Unknown operator approximation task: {task.name}")


def generate_operator_dataset(
    n_samples: int,
    config: ExperimentConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, u_all = generate_operator_inputs(
        n_samples=n_samples,
        config=config,
        rng=rng,
    )

    y_all = apply_operator(
        u=u_all,
        config=config,
    )

    return x, u_all, y_all


def build_operator_dataset(config: ExperimentConfig):
    rng = np.random.default_rng(config.training.seed)

    x, u_train, y_train = generate_operator_dataset(
        n_samples=config.split.n_train,
        config=config,
        rng=rng,
    )

    _, u_val, y_val = generate_operator_dataset(
        n_samples=config.split.n_val,
        config=config,
        rng=rng,
    )

    _, u_test, y_test = generate_operator_dataset(
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