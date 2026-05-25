from dataclasses import dataclass
from pathlib import Path


@dataclass
class HereditaryViscoelasticityConfig:
    # Output paths
    data_dir: Path = Path("simulations/data")
    figures_dir: Path = Path("simulations/figures")

    # Dataset size
    n_simulations: int = 100

    # Time domain
    t_final: float = 1.0
    nt: int = 256

    # Fixed hereditary viscoelasticity parameters
    E: float = 1.0
    A: float = 0.5
    tau: float = 0.1

    # Gaussian-mixture strain histories
    n_gaussians_min: int = 20
    n_gaussians_max: int = 30
    sigma_min_fraction: float = 0.03
    sigma_max_fraction: float = 0.15

    # Strain normalization
    strain_max_abs: float = 1.0

    # Reproducibility
    seed: int = 42

    # Visualization
    n_simulations_to_plot: int = 5
    plot_seed: int = 123

    # Saved filenames
    dataset_filename: str = "hereditary_viscoelasticity_dataset.npz"
    figure_filename: str = "hereditary_viscoelasticity_examples.png"