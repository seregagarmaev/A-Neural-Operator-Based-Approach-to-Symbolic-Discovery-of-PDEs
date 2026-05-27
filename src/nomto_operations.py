from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Sequence

import sympy as sp
import torch
import torch.nn as nn


class ExactOperation(nn.Module):
    def __init__(
        self,
        fname: str,
        arity: int,
        forward_fn: Callable,
        symbolic_fn: Callable,
    ) -> None:
        super().__init__()
        self.fname = fname
        self.arity = int(arity)
        self.forward_fn = forward_fn
        self.symbolic_fn = symbolic_fn

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        return self.forward_fn(*args)

    def symbolic(self, *args: sp.Expr) -> sp.Expr:
        return self.symbolic_fn(*args)


class NeuralOperatorOperation(nn.Module):
    def __init__(
        self,
        fname: str,
        arity: int,
        model: nn.Module,
    ) -> None:
        super().__init__()

        self.fname = fname
        self.arity = int(arity)
        self.model = model

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.model.eval()

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        x = torch.stack(args, dim=-1)

        y = self.model(x)

        if isinstance(y, tuple):
            y = y[0]

        if y.ndim >= 3 and y.shape[-1] == 1:
            y = y[..., 0]

        return y

    def symbolic(self, *args: sp.Expr) -> sp.Expr:
        return sp.Function(self.fname)(*args)


def id_operation() -> ExactOperation:
    return ExactOperation(
        fname="id",
        arity=1,
        forward_fn=lambda x: x,
        symbolic_fn=lambda x: x,
    )


def const_operation() -> ExactOperation:
    return ExactOperation(
        fname="const",
        arity=1,
        forward_fn=lambda x: torch.ones_like(x),
        symbolic_fn=lambda x: sp.Integer(1),
    )


def square_operation() -> ExactOperation:
    return ExactOperation(
        fname="square",
        arity=1,
        forward_fn=lambda x: x ** 2,
        symbolic_fn=lambda x: x ** 2,
    )


def cube_operation() -> ExactOperation:
    return ExactOperation(
        fname="cube",
        arity=1,
        forward_fn=lambda x: x ** 3,
        symbolic_fn=lambda x: x ** 3,
    )


def sin_operation() -> ExactOperation:
    return ExactOperation(
        fname="sin",
        arity=1,
        forward_fn=torch.sin,
        symbolic_fn=sp.sin,
    )


def cos_operation() -> ExactOperation:
    return ExactOperation(
        fname="cos",
        arity=1,
        forward_fn=torch.cos,
        symbolic_fn=sp.cos,
    )


def exp_operation() -> ExactOperation:
    return ExactOperation(
        fname="exp",
        arity=1,
        forward_fn=torch.exp,
        symbolic_fn=sp.exp,
    )


def add_operation() -> ExactOperation:
    return ExactOperation(
        fname="add",
        arity=2,
        forward_fn=lambda x, y: x + y,
        symbolic_fn=lambda x, y: x + y,
    )


def mul_operation() -> ExactOperation:
    return ExactOperation(
        fname="mul",
        arity=2,
        forward_fn=lambda x, y: x * y,
        symbolic_fn=lambda x, y: x * y,
    )


EXACT_OPERATIONS = {
    "id": id_operation,
    "const": const_operation,
    "square": square_operation,
    "cube": cube_operation,
    "sin": sin_operation,
    "cos": cos_operation,
    "exp": exp_operation,
    "add": add_operation,
    "mul": mul_operation,
}


def make_exact_operation(name: str) -> ExactOperation:
    return EXACT_OPERATIONS[name]()


def add_neural_operator_surrogates_to_path(cfg) -> None:
    no_dir = Path(cfg.paths.neural_operator_dir).resolve()

    if str(no_dir) not in sys.path:
        sys.path.insert(0, str(no_dir))


def make_surrogate_experiment_config(spec: dict):
    from config import (
        CNOConfig,
        ExperimentConfig,
        FNNConfig,
        FNOConfig,
        OperatorTaskConfig,
    )

    task_name = spec["task_name"]
    model_name = spec.get("model_name", "fnn")

    if model_name == "fno":
        model_cfg = FNOConfig()
    elif model_name == "cno":
        model_cfg = CNOConfig()
    elif model_name == "fnn":
        model_cfg = FNNConfig()
    else:
        raise ValueError(f"Unknown neural-operator model name: {model_name}")

    return ExperimentConfig(
        experiment_name=f"{model_name}_{task_name}",
        task=OperatorTaskConfig(name=task_name),
        model=model_cfg,
    )


def load_neural_operator(spec: dict, cfg) -> nn.Module:
    add_neural_operator_surrogates_to_path(cfg)

    from neural_operators import build_neural_operator

    device = torch.device(cfg.device)

    surrogate_config = make_surrogate_experiment_config(spec)
    surrogate_config.training.device = cfg.device

    model = build_neural_operator(surrogate_config, device)

    checkpoint_path = (
        Path(cfg.paths.neural_operator_models_dir)
        / surrogate_config.experiment_name
        / "best_model.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def make_neural_operation(spec: dict, cfg) -> NeuralOperatorOperation:
    model = load_neural_operator(spec, cfg)

    return NeuralOperatorOperation(
        fname=spec["name"],
        arity=spec["arity"],
        model=model,
    )


def make_operation(spec: dict, cfg) -> nn.Module:
    if spec["type"] == "exact":
        return make_exact_operation(spec["name"])

    if spec["type"] == "neural":
        return make_neural_operation(spec, cfg)

    raise ValueError(f"Unknown operation type: {spec['type']}")


def build_operations_from_specs(specs: Sequence[dict], cfg) -> list[nn.Module]:
    return [make_operation(spec, cfg) for spec in specs]


def build_operations_for_layer(cfg, layer_idx: int) -> list[nn.Module]:
    specs = cfg.operation_specs[layer_idx]
    return build_operations_from_specs(specs, cfg)