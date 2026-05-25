from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PathsConfig:
    models_dir: Path = Path("neural_operator_surrogates/models")
    figures_dir: Path = Path("neural_operator_surrogates/figures")


@dataclass
class SplitConfig:
    n_train: int = 4096
    n_val: int = 512
    n_test: int = 512
    batch_size: int = 64


@dataclass
class FractionalLaplacianTaskConfig:
    name: str = "fractional_laplacian"

    n_grid: int = 256
    x_min: float = -1.0
    x_max: float = 1.0

    frac_power_s: float = 0.7

    n_gaussians_min: int = 2
    n_gaussians_max: int = 5
    amplitude_min: float = -1.0
    amplitude_max: float = 1.0
    sigma_min: float = 0.03
    sigma_max: float = 0.18


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
    experiment_name: str = "cno_fractional_laplacian"

    paths: PathsConfig = field(default_factory=PathsConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    task: FractionalLaplacianTaskConfig = field(default_factory=FractionalLaplacianTaskConfig)

    model: CNOConfig = field(default_factory=CNOConfig)

    # To train FNO instead, change the two lines above to:
    # experiment_name: str = "fno_fractional_laplacian"
    # model: FNOConfig = field(default_factory=FNOConfig)

    training: TrainingConfig = field(default_factory=TrainingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)