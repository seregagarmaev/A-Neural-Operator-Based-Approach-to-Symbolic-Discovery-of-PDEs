from dataclasses import dataclass
from pathlib import Path


@dataclass
class SemiconductorDriftDiffusionConfig:
    # Output paths
    data_dir: Path = Path("simulations/data")
    figures_dir: Path = Path("simulations/figures")

    # Dataset size
    n_simulations: int = 100

    # Spatial domain
    L: float = 2.0 * 3.141592653589793
    nx: int = 256

    # Saved output time grid
    t_final: float = 0.5
    nt: int = 51

    # Internal RK4 time step
    dt: float = 1.0e-3

    # Diffusion coefficient
    diffusion_coefficient: float = 0.05

    # Periodic Gaussian mixture initial conditions
    n_gaussians_min: int = 20
    n_gaussians_max: int = 30
    sigma_min_fraction: float = 0.04
    sigma_max_fraction: float = 0.15

    # Initial-condition amplitude
    n_amplitude: float = 0.05

    # Reproducibility
    seed: int = 42

    # Visualization
    n_simulations_to_plot: int = 5
    plot_seed: int = 123

    # Saved filenames
    dataset_filename: str = "semiconductor_drift_diffusion_dataset.npz"
    figure_filename: str = "semiconductor_drift_diffusion_examples.png"