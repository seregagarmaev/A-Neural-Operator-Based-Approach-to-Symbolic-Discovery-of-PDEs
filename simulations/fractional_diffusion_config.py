from dataclasses import dataclass
from pathlib import Path


@dataclass
class FractionalDiffusionConfig:
    # Output paths
    data_dir: Path = Path("simulations/data")
    figures_dir: Path = Path("simulations/figures")

    # Dataset size
    n_simulations: int = 100

    # Spatial domain
    L: float = 2.0 * 3.141592653589793
    nx: int = 256

    # Time grid
    t_final: float = 1.0
    nt: int = 51

    # Fixed fractional diffusion equation parameters
    kappa: float = 1.0
    s: float = 0.7

    # Periodic Gaussian mixture initial conditions
    n_gaussians_min: int = 20
    n_gaussians_max: int = 30
    sigma_min_fraction: float = 0.04
    sigma_max_fraction: float = 0.15

    # Reproducibility
    seed: int = 42

    # Visualization
    n_simulations_to_plot: int = 5
    plot_seed: int = 123

    # Saved filenames
    dataset_filename: str = "fractional_diffusion_dataset.npz"
    figure_filename: str = "fractional_diffusion_examples.png"