from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NOMTOPathsConfig:
    neural_operator_dir: Path = Path("neural_operator_surrogates")
    neural_operator_models_dir: Path = Path("neural_operator_surrogates/models")


@dataclass
class NOMTOConfig:
    n_input_fields: int = 2
    n_output_fields: int = 1
    n_symbolic_layers: int = 2

    weight_scale: float = 0.5
    device: str = "cuda:0"

    paths: NOMTOPathsConfig = field(default_factory=NOMTOPathsConfig)

    operation_specs: list[list[dict]] = field(
        default_factory=lambda: [
            [
                {"type": "exact", "name": "id"},
                {"type": "exact", "name": "const"},
                {"type": "exact", "name": "square"},
                {"type": "exact", "name": "mul"},
                {
                    "type": "neural",
                    "name": "FracLap",
                    "task_name": "fractional_laplacian",
                    "model_name": "fnn",
                    "arity": 1,
                },
                {
                    "type": "neural",
                    "name": "PoissonGrad",
                    "task_name": "poisson_gradient",
                    "model_name": "fnn",
                    "arity": 1,
                },
                {
                    "type": "neural",
                    "name": "Dx",
                    "task_name": "first_derivative",
                    "model_name": "fnn",
                    "arity": 1,
                },
                {
                    "type": "neural",
                    "name": "Dxx",
                    "task_name": "second_derivative",
                    "model_name": "fnn",
                    "arity": 1,
                },
                {
                    "type": "neural",
                    "name": "MemoryIntegral",
                    "task_name": "hereditary_integral",
                    "model_name": "fnn",
                    "arity": 1,
                },
            ],
            [
                {"type": "exact", "name": "id"},
                {"type": "exact", "name": "const"},
                {"type": "exact", "name": "square"},
                {"type": "exact", "name": "mul"},
                {
                    "type": "neural",
                    "name": "FracLap",
                    "task_name": "fractional_laplacian",
                    "model_name": "fnn",
                    "arity": 1,
                },
                {
                    "type": "neural",
                    "name": "PoissonGrad",
                    "task_name": "poisson_gradient",
                    "model_name": "fnn",
                    "arity": 1,
                },
                {
                    "type": "neural",
                    "name": "Dx",
                    "task_name": "first_derivative",
                    "model_name": "fnn",
                    "arity": 1,
                },
                {
                    "type": "neural",
                    "name": "Dxx",
                    "task_name": "second_derivative",
                    "model_name": "fnn",
                    "arity": 1,
                },
                {
                    "type": "neural",
                    "name": "MemoryIntegral",
                    "task_name": "hereditary_integral",
                    "model_name": "fnn",
                    "arity": 1,
                },
            ],
        ]
    )