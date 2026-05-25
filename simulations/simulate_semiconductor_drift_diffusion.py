import numpy as np
import matplotlib.pyplot as plt

from semiconductor_drift_diffusion_config import SemiconductorDriftDiffusionConfig


def make_grid(config: SemiconductorDriftDiffusionConfig):
    x = np.linspace(0.0, config.L, config.nx, endpoint=False)
    t = np.linspace(0.0, config.t_final, config.nt)

    dx = config.L / config.nx
    k = 2.0 * np.pi * np.fft.fftfreq(config.nx, d=dx)

    return x, t, k


def periodic_distance(x, center, L):
    distance = np.abs(x - center)
    return np.minimum(distance, L - distance)


def generate_periodic_gaussian_field(
    x,
    config: SemiconductorDriftDiffusionConfig,
    rng: np.random.Generator,
):
    n_gaussians = rng.integers(
        config.n_gaussians_min,
        config.n_gaussians_max + 1,
    )

    sigma_min = config.sigma_min_fraction * config.L
    sigma_max = config.sigma_max_fraction * config.L

    field = np.zeros_like(x)

    for _ in range(n_gaussians):
        amplitude = rng.normal()
        center = rng.uniform(0.0, config.L)
        sigma = rng.uniform(sigma_min, sigma_max)

        d = periodic_distance(x, center, config.L)

        field += amplitude * np.exp(-(d ** 2) / (2.0 * sigma ** 2))

    field = field - np.mean(field)
    field = field / np.max(np.abs(field))

    return field


def generate_initial_condition(
    x,
    config: SemiconductorDriftDiffusionConfig,
    rng: np.random.Generator,
):
    n_perturbation = generate_periodic_gaussian_field(x, config, rng)

    n0 = 1.0 + config.n_amplitude * n_perturbation

    return n0


def spectral_derivative(f, k):
    f_hat = np.fft.fft(f)
    df_hat = 1j * k * f_hat
    return np.fft.ifft(df_hat).real


def spectral_second_derivative(f, k):
    f_hat = np.fft.fft(f)
    d2f_hat = -(k ** 2) * f_hat
    return np.fft.ifft(d2f_hat).real


def poisson_potential(n, k):
    """
    Solves -phi_xx = n - 1 on a periodic domain.

    The zero Fourier mode is set to zero.
    """
    rhs = n - 1.0
    rhs_hat = np.fft.fft(rhs)

    phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)

    nonzero = k != 0.0
    phi_hat[nonzero] = rhs_hat[nonzero] / (k[nonzero] ** 2)

    phi = np.fft.ifft(phi_hat).real

    return phi


def electric_field(n, k):
    """
    Computes E = -phi_x from -phi_xx = n - 1.

    In Fourier space:
        phi_hat_k = n_hat_k / k^2,
        E_hat_k = -i k phi_hat_k.
    """
    rhs = n - 1.0
    rhs_hat = np.fft.fft(rhs)

    E_hat = np.zeros_like(rhs_hat, dtype=np.complex128)

    nonzero = k != 0.0
    E_hat[nonzero] = -1j * rhs_hat[nonzero] / k[nonzero]

    E = np.fft.ifft(E_hat).real

    return E


def rhs(n, k, config: SemiconductorDriftDiffusionConfig):
    D = config.diffusion_coefficient

    n_xx = spectral_second_derivative(n, k)

    E = electric_field(n, k)
    flux = n * E
    flux_x = spectral_derivative(flux, k)

    n_t = D * n_xx - flux_x

    return n_t


def rk4_step(n, dt, k, config: SemiconductorDriftDiffusionConfig):
    k1 = rhs(n, k, config)

    k2 = rhs(
        n + 0.5 * dt * k1,
        k,
        config,
    )

    k3 = rhs(
        n + 0.5 * dt * k2,
        k,
        config,
    )

    k4 = rhs(
        n + dt * k3,
        k,
        config,
    )

    n_new = n + dt / 6.0 * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )

    return n_new


def simulate_semiconductor_drift_diffusion(
    n0,
    t,
    k,
    config: SemiconductorDriftDiffusionConfig,
):
    n = n0.copy()

    n_history = np.zeros((len(t), len(n0)), dtype=np.float64)
    E_history = np.zeros((len(t), len(n0)), dtype=np.float64)
    phi_history = np.zeros((len(t), len(n0)), dtype=np.float64)

    n_history[0] = n
    E_history[0] = electric_field(n, k)
    phi_history[0] = poisson_potential(n, k)

    current_time = 0.0

    for output_id in range(1, len(t)):
        target_time = t[output_id]

        while current_time < target_time:
            dt_step = min(config.dt, target_time - current_time)

            n = rk4_step(n, dt_step, k, config)

            current_time += dt_step

        n_history[output_id] = n
        E_history[output_id] = electric_field(n, k)
        phi_history[output_id] = poisson_potential(n, k)

    return n_history, E_history, phi_history


def generate_dataset(config: SemiconductorDriftDiffusionConfig):
    rng = np.random.default_rng(config.seed)

    x, t, k = make_grid(config)

    n_all = np.zeros(
        (config.n_simulations, config.nt, config.nx),
        dtype=np.float64,
    )

    E_all = np.zeros(
        (config.n_simulations, config.nt, config.nx),
        dtype=np.float64,
    )

    phi_all = np.zeros(
        (config.n_simulations, config.nt, config.nx),
        dtype=np.float64,
    )

    n0_all = np.zeros(
        (config.n_simulations, config.nx),
        dtype=np.float64,
    )

    for i in range(config.n_simulations):
        n0 = generate_initial_condition(x, config, rng)

        n, E, phi = simulate_semiconductor_drift_diffusion(
            n0,
            t,
            k,
            config,
        )

        n0_all[i] = n0

        n_all[i] = n
        E_all[i] = E
        phi_all[i] = phi

    return {
        "x": x,
        "t": t,
        "n": n_all,
        "E": E_all,
        "phi": phi_all,
        "n0": n0_all,
        "n_amplitude": config.n_amplitude,
        "diffusion_coefficient": config.diffusion_coefficient,
    }


def save_dataset(dataset, config: SemiconductorDriftDiffusionConfig):
    config.data_dir.mkdir(parents=True, exist_ok=True)

    save_path = config.data_dir / config.dataset_filename

    np.savez_compressed(
        save_path,
        x=dataset["x"],
        t=dataset["t"],
        n=dataset["n"],
        E=dataset["E"],
        phi=dataset["phi"],
        n0=dataset["n0"],
        n_amplitude=dataset["n_amplitude"],
        diffusion_coefficient=dataset["diffusion_coefficient"],
    )

    return save_path


def plot_random_simulations(dataset, config: SemiconductorDriftDiffusionConfig):
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.plot_seed)

    sim_ids = rng.choice(
        config.n_simulations,
        size=config.n_simulations_to_plot,
        replace=False,
    )

    x = dataset["x"]
    t = dataset["t"]
    n_all = dataset["n"]
    E_all = dataset["E"]
    phi_all = dataset["phi"]

    selected_n = n_all[sim_ids]
    selected_E = E_all[sim_ids]
    selected_phi = phi_all[sim_ids]

    n_vmin = np.min(selected_n)
    n_vmax = np.max(selected_n)

    E_vmin = np.min(selected_E)
    E_vmax = np.max(selected_E)

    phi_vmin = np.min(selected_phi)
    phi_vmax = np.max(selected_phi)

    fig, axes = plt.subplots(
        config.n_simulations_to_plot,
        3,
        figsize=(12, 2.4 * config.n_simulations_to_plot),
        sharex=True,
        sharey=True,
    )

    if config.n_simulations_to_plot == 1:
        axes = axes[None, :]

    n_im = None
    E_im = None
    phi_im = None

    for row_id, sim_id in enumerate(sim_ids):
        n_im = axes[row_id, 0].imshow(
            n_all[sim_id],
            extent=[x[0], x[-1], t[-1], t[0]],
            aspect="auto",
            interpolation="nearest",
            vmin=n_vmin,
            vmax=n_vmax,
        )

        E_im = axes[row_id, 1].imshow(
            E_all[sim_id],
            extent=[x[0], x[-1], t[-1], t[0]],
            aspect="auto",
            interpolation="nearest",
            vmin=E_vmin,
            vmax=E_vmax,
        )

        phi_im = axes[row_id, 2].imshow(
            phi_all[sim_id],
            extent=[x[0], x[-1], t[-1], t[0]],
            aspect="auto",
            interpolation="nearest",
            vmin=phi_vmin,
            vmax=phi_vmax,
        )

        axes[row_id, 0].set_ylabel(r"$t$")
        axes[row_id, 0].set_title(fr"Simulation {sim_id}: $n$")
        axes[row_id, 1].set_title(fr"Simulation {sim_id}: $E=-\phi_x$")
        axes[row_id, 2].set_title(fr"Simulation {sim_id}: $\phi$")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x$")

    fig.suptitle(
        "Semiconductor drift-diffusion simulations",
        y=0.995,
    )

    cbar_n = fig.colorbar(n_im, ax=axes[:, 0], fraction=0.025, pad=0.02)
    cbar_n.set_label(r"$n(x,t)$")

    cbar_E = fig.colorbar(E_im, ax=axes[:, 1], fraction=0.025, pad=0.02)
    cbar_E.set_label(r"$E(x,t)$")

    cbar_phi = fig.colorbar(phi_im, ax=axes[:, 2], fraction=0.025, pad=0.02)
    cbar_phi.set_label(r"$\phi(x,t)$")

    figure_path = config.figures_dir / config.figure_filename
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    return figure_path, sim_ids


def main():
    config = SemiconductorDriftDiffusionConfig()

    dataset = generate_dataset(config)

    dataset_path = save_dataset(dataset, config)
    figure_path, sim_ids = plot_random_simulations(dataset, config)

    print(f"Saved dataset to: {dataset_path}")
    print(f"Saved figure to: {figure_path}")
    print(f"Plotted simulation IDs: {sim_ids}")
    print(f"Dataset shape n: {dataset['n'].shape}")
    print(f"Dataset shape E: {dataset['E'].shape}")
    print(f"Dataset shape phi: {dataset['phi'].shape}")
    print("Shape convention: field[simulation_id, time_id, space_id]")
    print(f"n amplitude: {config.n_amplitude}")
    print(f"Diffusion coefficient: {config.diffusion_coefficient}")
    print(f"Internal RK4 dt: {config.dt}")


if __name__ == "__main__":
    main()