import numpy as np
import matplotlib.pyplot as plt

from euler_poisson_config import EulerPoissonConfig


def make_grid(config: EulerPoissonConfig):
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
    config: EulerPoissonConfig,
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
    config: EulerPoissonConfig,
    rng: np.random.Generator,
):
    rho_perturbation = generate_periodic_gaussian_field(x, config, rng)
    u_field = generate_periodic_gaussian_field(x, config, rng)

    rho0 = 1.0 + config.rho_amplitude * rho_perturbation
    u0 = config.u_amplitude * u_field

    return rho0, u0


def spectral_derivative(f, k):
    f_hat = np.fft.fft(f)
    df_hat = 1j * k * f_hat
    return np.fft.ifft(df_hat).real


def poisson_potential(rho, k):
    """
    Solves -phi_xx = rho - 1 on a periodic domain.

    The zero Fourier mode is set to zero.
    """
    rhs = rho - 1.0
    rhs_hat = np.fft.fft(rhs)

    phi_hat = np.zeros_like(rhs_hat, dtype=np.complex128)

    nonzero = k != 0.0
    phi_hat[nonzero] = rhs_hat[nonzero] / (k[nonzero] ** 2)

    phi = np.fft.ifft(phi_hat).real

    return phi


def poisson_force(rho, k):
    """
    Computes F = -phi_x from -phi_xx = rho - 1.

    In Fourier space:
        phi_hat_k = rho_hat_k / k^2,
        F_hat_k = -i k phi_hat_k.
    """
    rhs = rho - 1.0
    rhs_hat = np.fft.fft(rhs)

    force_hat = np.zeros_like(rhs_hat, dtype=np.complex128)

    nonzero = k != 0.0
    force_hat[nonzero] = -1j * rhs_hat[nonzero] / k[nonzero]

    force = np.fft.ifft(force_hat).real

    return force


def rhs(rho, u, k):
    rho_u = rho * u

    rho_t = -spectral_derivative(rho_u, k)

    u_x = spectral_derivative(u, k)
    force = poisson_force(rho, k)

    u_t = -u * u_x + force

    return rho_t, u_t


def rk4_step(rho, u, dt, k):
    k1_rho, k1_u = rhs(rho, u, k)

    k2_rho, k2_u = rhs(
        rho + 0.5 * dt * k1_rho,
        u + 0.5 * dt * k1_u,
        k,
    )

    k3_rho, k3_u = rhs(
        rho + 0.5 * dt * k2_rho,
        u + 0.5 * dt * k2_u,
        k,
    )

    k4_rho, k4_u = rhs(
        rho + dt * k3_rho,
        u + dt * k3_u,
        k,
    )

    rho_new = rho + dt / 6.0 * (
        k1_rho + 2.0 * k2_rho + 2.0 * k3_rho + k4_rho
    )

    u_new = u + dt / 6.0 * (
        k1_u + 2.0 * k2_u + 2.0 * k3_u + k4_u
    )

    return rho_new, u_new


def simulate_euler_poisson(
    rho0,
    u0,
    t,
    k,
    config: EulerPoissonConfig,
):
    rho = rho0.copy()
    u = u0.copy()

    rho_history = np.zeros((len(t), len(rho0)), dtype=np.float64)
    u_history = np.zeros((len(t), len(u0)), dtype=np.float64)
    force_history = np.zeros((len(t), len(rho0)), dtype=np.float64)
    phi_history = np.zeros((len(t), len(rho0)), dtype=np.float64)

    rho_history[0] = rho
    u_history[0] = u
    force_history[0] = poisson_force(rho, k)
    phi_history[0] = poisson_potential(rho, k)

    current_time = 0.0

    for output_id in range(1, len(t)):
        target_time = t[output_id]

        while current_time < target_time:
            dt_step = min(config.dt, target_time - current_time)

            rho, u = rk4_step(rho, u, dt_step, k)

            current_time += dt_step

        rho_history[output_id] = rho
        u_history[output_id] = u
        force_history[output_id] = poisson_force(rho, k)
        phi_history[output_id] = poisson_potential(rho, k)

    return rho_history, u_history, force_history, phi_history


def generate_dataset(config: EulerPoissonConfig):
    rng = np.random.default_rng(config.seed)

    x, t, k = make_grid(config)

    rho_all = np.zeros(
        (config.n_simulations, config.nt, config.nx),
        dtype=np.float64,
    )

    u_all = np.zeros(
        (config.n_simulations, config.nt, config.nx),
        dtype=np.float64,
    )

    force_all = np.zeros(
        (config.n_simulations, config.nt, config.nx),
        dtype=np.float64,
    )

    phi_all = np.zeros(
        (config.n_simulations, config.nt, config.nx),
        dtype=np.float64,
    )

    rho0_all = np.zeros(
        (config.n_simulations, config.nx),
        dtype=np.float64,
    )

    u0_all = np.zeros(
        (config.n_simulations, config.nx),
        dtype=np.float64,
    )

    for i in range(config.n_simulations):
        rho0, u0 = generate_initial_condition(x, config, rng)

        rho, u, force, phi = simulate_euler_poisson(
            rho0,
            u0,
            t,
            k,
            config,
        )

        rho0_all[i] = rho0
        u0_all[i] = u0

        rho_all[i] = rho
        u_all[i] = u
        force_all[i] = force
        phi_all[i] = phi

    return {
        "x": x,
        "t": t,
        "rho": rho_all,
        "u": u_all,
        "force": force_all,
        "phi": phi_all,
        "rho0": rho0_all,
        "u0": u0_all,
        "rho_amplitude": config.rho_amplitude,
        "u_amplitude": config.u_amplitude,
    }


def save_dataset(dataset, config: EulerPoissonConfig):
    config.data_dir.mkdir(parents=True, exist_ok=True)

    save_path = config.data_dir / config.dataset_filename

    np.savez_compressed(
        save_path,
        x=dataset["x"],
        t=dataset["t"],
        rho=dataset["rho"],
        u=dataset["u"],
        force=dataset["force"],
        phi=dataset["phi"],
        rho0=dataset["rho0"],
        u0=dataset["u0"],
        rho_amplitude=dataset["rho_amplitude"],
        u_amplitude=dataset["u_amplitude"],
    )

    return save_path


def plot_random_simulations(dataset, config: EulerPoissonConfig):
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.plot_seed)

    sim_ids = rng.choice(
        config.n_simulations,
        size=config.n_simulations_to_plot,
        replace=False,
    )

    x = dataset["x"]
    t = dataset["t"]
    rho_all = dataset["rho"]
    u_all = dataset["u"]
    force_all = dataset["force"]

    selected_rho = rho_all[sim_ids]
    selected_u = u_all[sim_ids]
    selected_force = force_all[sim_ids]

    rho_vmin = np.min(selected_rho)
    rho_vmax = np.max(selected_rho)

    u_vmin = np.min(selected_u)
    u_vmax = np.max(selected_u)

    force_vmin = np.min(selected_force)
    force_vmax = np.max(selected_force)

    fig, axes = plt.subplots(
        config.n_simulations_to_plot,
        3,
        figsize=(12, 2.4 * config.n_simulations_to_plot),
        sharex=True,
        sharey=True,
    )

    if config.n_simulations_to_plot == 1:
        axes = axes[None, :]

    rho_im = None
    u_im = None
    force_im = None

    for row_id, sim_id in enumerate(sim_ids):
        rho_im = axes[row_id, 0].imshow(
            rho_all[sim_id],
            extent=[x[0], x[-1], t[-1], t[0]],
            aspect="auto",
            interpolation="nearest",
            vmin=rho_vmin,
            vmax=rho_vmax,
        )

        u_im = axes[row_id, 1].imshow(
            u_all[sim_id],
            extent=[x[0], x[-1], t[-1], t[0]],
            aspect="auto",
            interpolation="nearest",
            vmin=u_vmin,
            vmax=u_vmax,
        )

        force_im = axes[row_id, 2].imshow(
            force_all[sim_id],
            extent=[x[0], x[-1], t[-1], t[0]],
            aspect="auto",
            interpolation="nearest",
            vmin=force_vmin,
            vmax=force_vmax,
        )

        axes[row_id, 0].set_ylabel(r"$t$")
        axes[row_id, 0].set_title(fr"Simulation {sim_id}: $\rho$")
        axes[row_id, 1].set_title(fr"Simulation {sim_id}: $u$")
        axes[row_id, 2].set_title(fr"Simulation {sim_id}: $-\phi_x$")

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$x$")

    fig.suptitle(
        "Euler--Poisson simulations",
        y=0.995,
    )

    cbar_rho = fig.colorbar(rho_im, ax=axes[:, 0], fraction=0.025, pad=0.02)
    cbar_rho.set_label(r"$\rho(x,t)$")

    cbar_u = fig.colorbar(u_im, ax=axes[:, 1], fraction=0.025, pad=0.02)
    cbar_u.set_label(r"$u(x,t)$")

    cbar_force = fig.colorbar(force_im, ax=axes[:, 2], fraction=0.025, pad=0.02)
    cbar_force.set_label(r"$-\phi_x(x,t)$")

    figure_path = config.figures_dir / config.figure_filename
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    return figure_path, sim_ids


def main():
    config = EulerPoissonConfig()

    dataset = generate_dataset(config)

    dataset_path = save_dataset(dataset, config)
    figure_path, sim_ids = plot_random_simulations(dataset, config)

    print(f"Saved dataset to: {dataset_path}")
    print(f"Saved figure to: {figure_path}")
    print(f"Plotted simulation IDs: {sim_ids}")
    print(f"Dataset shape rho: {dataset['rho'].shape}")
    print(f"Dataset shape u: {dataset['u'].shape}")
    print(f"Dataset shape force: {dataset['force'].shape}")
    print("Shape convention: field[simulation_id, time_id, space_id]")
    print(f"rho amplitude: {config.rho_amplitude}")
    print(f"u amplitude: {config.u_amplitude}")
    print(f"Internal RK4 dt: {config.dt}")


if __name__ == "__main__":
    main()