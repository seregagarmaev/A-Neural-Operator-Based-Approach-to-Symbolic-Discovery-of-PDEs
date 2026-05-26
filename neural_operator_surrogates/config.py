from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    models_dir: Path = Path("models")
    figures_dir: Path = Path("figures")


@dataclass
class SplitConfig:
    n_train: int = 4096
    n_val: int = 512
    n_test: int = 512
    batch_size: int = 64


@dataclass
class OperatorTaskConfig:
    name: str = "fractional_laplacian"

    # Periodic spatial grid
    L: float = 2.0 * 3.141592653589793
    nx: int = 256

    # Fractional Laplacian parameter
    frac_power_s: float = 0.7

    # Gaussian-mixture input generation
    n_gaussians_min: int = 20
    n_gaussians_max: int = 30
    sigma_min_fraction: float = 0.04
    sigma_max_fraction: float = 0.15

    # Used only for the Poisson-gradient density input q = 1 + amplitude * perturbation
    poisson_density_perturbation_amplitude: float = 0.05


@dataclass
class FNOConfig:
    name: str = "fno"

    problem_dim: int = 1
    in_dim: int = 1
    out_dim: int = 1

    width: int = 64
    modes: int = 32
    n_layers: int = 4

    fun_act: str = "gelu"
    weights_norm: str = "Xavier"
    fno_arc: str = "Classic"
    rnn: bool = False
    fft_norm: str | None = None
    padding: int = 0


@dataclass
class CNOConfig:
    name: str = "cno"

    problem_dim: int = 1
    in_dim: int = 1
    out_dim: int = 1

    n_layers: int = 4
    n_res: int = 2
    n_res_neck: int = 2
    channel_multiplier: int = 16
    kernel_size: int = 3
    use_bn: bool = False


@dataclass
class TrainingConfig:
    device: str = "cuda:0"

    max_epochs: int = 1000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6

    scheduler_step: int = 100
    scheduler_gamma: float = 0.5

    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-5

    seed: int = 42


@dataclass
class VisualizationConfig:
    n_prediction_plots: int = 6


@dataclass
class ExperimentConfig:
    experiment_name: str

    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    task: OperatorTaskConfig = field(default_factory=OperatorTaskConfig)
    model: FNOConfig | CNOConfig = field(default_factory=FNOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


def make_experiments_for_task(task_name: str) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            experiment_name=f"fno_{task_name}",
            task=OperatorTaskConfig(name=task_name),
            model=FNOConfig(),
        ),
        ExperimentConfig(
            experiment_name=f"cno_{task_name}",
            task=OperatorTaskConfig(name=task_name),
            model=CNOConfig(),
        ),
    ]


def get_experiments() -> list[ExperimentConfig]:
    task_names = [
        "fractional_laplacian",
        "poisson_gradient",
        "first_derivative",
        "second_derivative",
    ]

    experiments = []

    for task_name in task_names:
        experiments.extend(make_experiments_for_task(task_name))

    return experiments