import numpy as np
import matplotlib.pyplot as plt

from fractional_diffusion_config import FractionalDiffusionConfig


def make_grid(config: FractionalDiffusionConfig):
    x = np.linspace(0.0, config.L, config.nx, endpoint=False)
    t = np.linspace(0.0, config.t_final, config.nt)

    dx = config.L / config.nx
    k = 2.0 * np.pi * np.fft.fftfreq(config.nx, d=dx)

    return x, t, k


def periodic_distance(x, center, L):
    distance = np.abs(x - center)
    return np.minimum(distance, L - distance)


def generate_initial_condition(
    x,
    config: FractionalDiffusionConfig,
    rng: np.random.Generator,
):
    n_gaussians = rng.integers(
        config.n_gaussians_min,
        config.n_gaussians_max + 1,
    )

    sigma_min = config.sigma_min_fraction * config.L
    sigma_max = config.sigma_max_fraction * config.L

    u0 = np.zeros_like(x)

    for _ in range(n_gaussians):
        amplitude = rng.normal()
        center = rng.uniform(0.0, config.L)
        sigma = rng.uniform(sigma_min, sigma_max)

        d = periodic_distance(x, center, config.L)

        u0 += amplitude * np.exp(-(d ** 2) / (2.0 * sigma ** 2))

    u0 = u0 - np.mean(u0)
    u0 = u0 / np.max(np.abs(u0))

    return u0


def simulate_fractional_diffusion(
    u0,
    t,
    k,
    config: FractionalDiffusionConfig,
):
    u0_hat = np.fft.fft(u0)

    fractional_symbol = np.abs(k) ** (2.0 * config.s)

    u = np.zeros((len(t), len(u0)), dtype=np.float64)

    for j, tj in enumerate(t):
        decay = np.exp(-config.kappa * fractional_symbol * tj)
        u_hat_t = u0_hat * decay
        u[j] = np.fft.ifft(u_hat_t).real

    return u


def generate_dataset(config: FractionalDiffusionConfig):
    rng = np.random.default_rng(config.seed)

    x, t, k = make_grid(config)

    u_all = np.zeros(
        (config.n_simulations, config.nt, config.nx),
        dtype=np.float64,
    )

    u0_all = np.zeros(
        (config.n_simulations, config.nx),
        dtype=np.float64,
    )

    for i in range(config.n_simulations):
        u0 = generate_initial_condition(x, config, rng)
        u = simulate_fractional_diffusion(u0, t, k, config)

        u0_all[i] = u0
        u_all[i] = u

    return {
        "x": x,
        "t": t,
        "u": u_all,
        "u0": u0_all,
        "kappa": config.kappa,
        "s": config.s,
    }


def save_dataset(dataset, config: FractionalDiffusionConfig):
    config.data_dir.mkdir(parents=True, exist_ok=True)

    save_path = config.data_dir / config.dataset_filename

    np.savez_compressed(
        save_path,
        x=dataset["x"],
        t=dataset["t"],
        u=dataset["u"],
        u0=dataset["u0"],
        kappa=dataset["kappa"],
        s=dataset["s"],
    )

    return save_path


def plot_random_simulations(dataset, config: FractionalDiffusionConfig):
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.plot_seed)

    sim_ids = rng.choice(
        config.n_simulations,
        size=config.n_simulations_to_plot,
        replace=False,
    )

    x = dataset["x"]
    t = dataset["t"]
    u_all = dataset["u"]
    kappa = dataset["kappa"]
    s = dataset["s"]

    selected_u = u_all[sim_ids]

    vmin = np.min(selected_u)
    vmax = np.max(selected_u)

    fig, axes = plt.subplots(
        config.n_simulations_to_plot,
        1,
        figsize=(7, 2.4 * config.n_simulations_to_plot),
        sharex=True,
        sharey=True,
    )

    for ax, sim_id in zip(axes, sim_ids):
        im = ax.imshow(
            u_all[sim_id],
            extent=[x[0], x[-1], t[-1], t[0]],
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_ylabel(r"$t$")
        ax.set_title(fr"Simulation {sim_id}")

    axes[-1].set_xlabel(r"$x$")

    fig.suptitle(
        fr"Fractional diffusion, $\kappa={kappa:.3f}$, $s={s:.3f}$",
        y=0.995,
    )

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(r"$u(x,t)$")

    figure_path = config.figures_dir / config.figure_filename
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    return figure_path, sim_ids


def main():
    config = FractionalDiffusionConfig()

    dataset = generate_dataset(config)

    dataset_path = save_dataset(dataset, config)
    figure_path, sim_ids = plot_random_simulations(dataset, config)

    print(f"Saved dataset to: {dataset_path}")
    print(f"Saved figure to: {figure_path}")
    print(f"Plotted simulation IDs: {sim_ids}")
    print(f"Dataset shape u: {dataset['u'].shape}")
    print("Shape convention: u[simulation_id, time_id, space_id]")
    print(f"Equation parameters: kappa={config.kappa}, s={config.s}")


if __name__ == "__main__":
    main()