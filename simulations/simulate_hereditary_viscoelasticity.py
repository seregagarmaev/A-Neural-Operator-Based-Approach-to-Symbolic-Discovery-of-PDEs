import numpy as np
import matplotlib.pyplot as plt

from hereditary_viscoelasticity_config import HereditaryViscoelasticityConfig


def make_grid(config: HereditaryViscoelasticityConfig):
    t = np.linspace(0.0, config.t_final, config.nt)
    dt = t[1] - t[0]

    return t, dt


def generate_strain_history(
    t,
    config: HereditaryViscoelasticityConfig,
    rng: np.random.Generator,
):
    T = config.t_final

    n_gaussians = rng.integers(
        config.n_gaussians_min,
        config.n_gaussians_max + 1,
    )

    sigma_min = config.sigma_min_fraction * T
    sigma_max = config.sigma_max_fraction * T

    epsilon = np.zeros_like(t)

    for _ in range(n_gaussians):
        amplitude = rng.normal()
        center = rng.uniform(0.0, T)
        sigma = rng.uniform(sigma_min, sigma_max)

        epsilon += amplitude * np.exp(
            -((t - center) ** 2) / (2.0 * sigma ** 2)
        )

    epsilon = epsilon - np.mean(epsilon)
    epsilon = config.strain_max_abs * epsilon / np.max(np.abs(epsilon))

    return epsilon


def simulate_hereditary_viscoelasticity(
    epsilon,
    dt,
    config: HereditaryViscoelasticityConfig,
):
    memory = np.zeros_like(epsilon)

    alpha = np.exp(-dt / config.tau)

    w0 = config.tau ** 2 / dt * (1.0 - alpha) - config.tau * alpha
    w1 = config.tau - config.tau ** 2 / dt * (1.0 - alpha)

    for n in range(1, len(epsilon)):
        memory[n] = (
            alpha * memory[n - 1]
            + w0 * epsilon[n - 1]
            + w1 * epsilon[n]
        )

    sigma = config.E * epsilon + config.A * memory

    return memory, sigma


def generate_dataset(config: HereditaryViscoelasticityConfig):
    rng = np.random.default_rng(config.seed)

    t, dt = make_grid(config)

    epsilon_all = np.zeros(
        (config.n_simulations, config.nt),
        dtype=np.float64,
    )

    memory_all = np.zeros(
        (config.n_simulations, config.nt),
        dtype=np.float64,
    )

    sigma_all = np.zeros(
        (config.n_simulations, config.nt),
        dtype=np.float64,
    )

    for i in range(config.n_simulations):
        epsilon = generate_strain_history(t, config, rng)

        memory, sigma = simulate_hereditary_viscoelasticity(
            epsilon,
            dt,
            config,
        )

        epsilon_all[i] = epsilon
        memory_all[i] = memory
        sigma_all[i] = sigma

    return {
        "t": t,
        "epsilon": epsilon_all,
        "memory": memory_all,
        "sigma": sigma_all,
        "E": config.E,
        "A": config.A,
        "tau": config.tau,
    }


def save_dataset(dataset, config: HereditaryViscoelasticityConfig):
    config.data_dir.mkdir(parents=True, exist_ok=True)

    save_path = config.data_dir / config.dataset_filename

    np.savez_compressed(
        save_path,
        t=dataset["t"],
        epsilon=dataset["epsilon"],
        memory=dataset["memory"],
        sigma=dataset["sigma"],
        E=dataset["E"],
        A=dataset["A"],
        tau=dataset["tau"],
    )

    return save_path


def plot_random_simulations(
    dataset,
    config: HereditaryViscoelasticityConfig,
):
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.plot_seed)

    sim_ids = rng.choice(
        config.n_simulations,
        size=config.n_simulations_to_plot,
        replace=False,
    )

    t = dataset["t"]
    epsilon_all = dataset["epsilon"]
    memory_all = dataset["memory"]
    sigma_all = dataset["sigma"]

    E = dataset["E"]
    A = dataset["A"]
    tau = dataset["tau"]

    fig, axes = plt.subplots(
        config.n_simulations_to_plot,
        1,
        figsize=(7, 2.4 * config.n_simulations_to_plot),
        sharex=True,
    )

    for ax, sim_id in zip(axes, sim_ids):
        ax.plot(t, epsilon_all[sim_id], label=r"$\varepsilon(t)$")
        ax.plot(t, memory_all[sim_id], label=r"$q(t)$")
        ax.plot(t, sigma_all[sim_id], label=r"$\sigma(t)$")

        ax.set_ylabel("value")
        ax.set_title(fr"Simulation {sim_id}")
        ax.legend(loc="best")

    axes[-1].set_xlabel(r"$t$")

    fig.suptitle(
        fr"Hereditary viscoelasticity, "
        fr"$E={E:.3f}$, $A={A:.3f}$, $\tau={tau:.3f}$",
        y=0.995,
    )

    figure_path = config.figures_dir / config.figure_filename
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    return figure_path, sim_ids


def main():
    config = HereditaryViscoelasticityConfig()

    dataset = generate_dataset(config)

    dataset_path = save_dataset(dataset, config)
    figure_path, sim_ids = plot_random_simulations(dataset, config)

    print(f"Saved dataset to: {dataset_path}")
    print(f"Saved figure to: {figure_path}")
    print(f"Plotted simulation IDs: {sim_ids}")
    print(f"Dataset shape epsilon: {dataset['epsilon'].shape}")
    print(f"Dataset shape memory: {dataset['memory'].shape}")
    print(f"Dataset shape sigma: {dataset['sigma'].shape}")
    print("Shape convention: field[simulation_id, time_id]")
    print(
        f"Equation parameters: "
        f"E={config.E}, A={config.A}, tau={config.tau}"
    )


if __name__ == "__main__":
    main()