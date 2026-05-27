from __future__ import annotations

from typing import List, Sequence

import sympy as sp
import torch
import torch.nn as nn

from src.nomto_operations import build_operations_for_layer


def _threshold_multiply(coef: float, expr: sp.Expr, decimals: int) -> sp.Expr:
    threshold = 0.5 * (10.0 ** (-int(decimals)))
    if abs(coef) < threshold:
        return sp.Integer(0)
    return sp.N(coef * expr, decimals)


_BAD_SYMPY_ATOMS = (sp.oo, -sp.oo, sp.zoo, sp.nan)


def _sanitize_symbolic(expr: sp.Expr) -> sp.Expr:
    if expr is None:
        return sp.Integer(0)

    if isinstance(expr, bool):
        return sp.Integer(int(expr))

    if isinstance(expr, int):
        return sp.Integer(expr)

    if isinstance(expr, float):
        return sp.Float(expr)

    if not isinstance(expr, sp.Basic):
        return sp.Integer(0)

    if expr.has(*_BAD_SYMPY_ATOMS):
        return sp.Integer(0)

    if getattr(expr, "is_infinite", None) is True:
        return sp.Integer(0)

    if getattr(expr, "is_nan", None) is True:
        return sp.Integer(0)

    return expr


class SymbolicLayer(nn.Module):
    def __init__(
        self,
        n_input_fields: int,
        operations: Sequence[nn.Module],
        weight_scale: float = 0.5,
    ) -> None:
        super().__init__()

        self.n_input_fields = int(n_input_fields)
        self.operations = nn.ModuleList(operations)

        self.arities = [int(op.arity) for op in self.operations]
        self.n_ops = len(self.operations)
        self.n_lifted_inputs = sum(self.arities)

        self.weights = nn.Parameter(
            weight_scale * (torch.rand(self.n_input_fields, self.n_lifted_inputs) - 0.5)
        )

        self.mask = nn.Parameter(
            torch.ones_like(self.weights),
            requires_grad=False,
        )

    def effective_weights(self) -> torch.Tensor:
        return self.weights * self.mask

    def weights_for_regularization(self) -> torch.Tensor:
        return self.effective_weights().reshape(-1)

    def lift(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bc...,ck->bk...", x, self.effective_weights())

    def apply_operations(self, lifted: torch.Tensor) -> torch.Tensor:
        outputs: List[torch.Tensor] = []

        start = 0
        for op, arity in zip(self.operations, self.arities):
            args = [lifted[:, start + i] for i in range(arity)]
            outputs.append(op(*args))
            start += arity

        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lifted = self.lift(x)
        return self.apply_operations(lifted)

    def prune_by_threshold(self, threshold: float) -> int:
        with torch.no_grad():
            active = self.mask > 0.5
            small = self.weights.abs() < float(threshold)
            non_finite = ~torch.isfinite(self.weights)

            to_prune = active & (small | non_finite)
            n_pruned = int(to_prune.sum().item())

            self.mask[to_prune] = 0.0
            self.weights[to_prune] = 0.0

        return n_pruned

    def count_active_edges(self) -> int:
        return int((self.mask > 0.5).sum().item())

    def get_symbolic_output(
        self,
        symbolic_inputs: Sequence[sp.Expr],
        rounding_decimals: int = 2,
    ) -> List[sp.Expr]:
        outputs: List[sp.Expr] = []

        start = 0
        for op, arity in zip(self.operations, self.arities):
            args: List[sp.Expr] = []

            for local_idx in range(arity):
                col = start + local_idx
                mixed = sp.Integer(0)

                for j, sym_in in enumerate(symbolic_inputs):
                    coef = float(self.effective_weights()[j, col].detach().cpu())
                    mixed += _threshold_multiply(coef, sym_in, rounding_decimals)

                args.append(_sanitize_symbolic(mixed))

            expr = op.symbolic(*args)
            outputs.append(_sanitize_symbolic(expr))

            start += arity

        return outputs


class AssemblyLayer(nn.Module):
    def __init__(
        self,
        n_input_fields: int,
        n_output_fields: int = 1,
        weight_scale: float = 0.5,
    ) -> None:
        super().__init__()

        self.n_input_fields = int(n_input_fields)
        self.n_output_fields = int(n_output_fields)

        self.weights = nn.Parameter(
            weight_scale * (torch.rand(self.n_input_fields, self.n_output_fields) - 0.5)
        )

        self.mask = nn.Parameter(
            torch.ones_like(self.weights),
            requires_grad=False,
        )

    def effective_weights(self) -> torch.Tensor:
        return self.weights * self.mask

    def weights_for_regularization(self) -> torch.Tensor:
        return self.effective_weights().reshape(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bc...,co->bo...", x, self.effective_weights())

    def prune_by_threshold(self, threshold: float) -> int:
        with torch.no_grad():
            active = self.mask > 0.5
            small = self.weights.abs() < float(threshold)
            non_finite = ~torch.isfinite(self.weights)

            to_prune = active & (small | non_finite)
            n_pruned = int(to_prune.sum().item())

            self.mask[to_prune] = 0.0
            self.weights[to_prune] = 0.0

        return n_pruned

    def count_active_edges(self) -> int:
        return int((self.mask > 0.5).sum().item())

    def get_symbolic_output(
        self,
        symbolic_inputs: Sequence[sp.Expr],
        rounding_decimals: int = 2,
    ) -> List[sp.Expr]:
        outputs: List[sp.Expr] = []

        for out_idx in range(self.n_output_fields):
            expr = sp.Integer(0)

            for j, sym_in in enumerate(symbolic_inputs):
                coef = float(self.effective_weights()[j, out_idx].detach().cpu())
                expr += _threshold_multiply(coef, sym_in, rounding_decimals)

            outputs.append(_sanitize_symbolic(expr))

        return outputs


class NOMTO(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        self.cfg = cfg

        self.n_input_fields = int(cfg.n_input_fields)
        self.n_symbolic_layers = int(cfg.n_symbolic_layers)
        self.n_output_fields = int(cfg.n_output_fields)
        self.weight_scale = float(cfg.weight_scale)

        layers: List[SymbolicLayer] = []
        previous_n_ops = 0

        for layer_idx in range(self.n_symbolic_layers):
            if layer_idx == 0:
                layer_n_inputs = self.n_input_fields
            else:
                layer_n_inputs = self.n_input_fields + previous_n_ops

            operations = build_operations_for_layer(cfg, layer_idx)

            layer = SymbolicLayer(
                n_input_fields=layer_n_inputs,
                operations=operations,
                weight_scale=self.weight_scale,
            )

            layers.append(layer)
            previous_n_ops = layer.n_ops

        self.symbolic_layers = nn.ModuleList(layers)

        self.assembly_layer = AssemblyLayer(
            n_input_fields=self.n_input_fields + previous_n_ops,
            n_output_fields=self.n_output_fields,
            weight_scale=self.weight_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = x
        h = self.symbolic_layers[0](x0)

        for layer in self.symbolic_layers[1:]:
            h = layer(torch.cat([x0, h], dim=1))

        y = self.assembly_layer(torch.cat([x0, h], dim=1))
        return y

    def weights_for_regularization(self) -> List[torch.Tensor]:
        weights: List[torch.Tensor] = []

        for layer in self.symbolic_layers:
            weights.append(layer.weights_for_regularization())

        weights.append(self.assembly_layer.weights_for_regularization())

        return weights

    def prune_by_threshold(self, threshold: float) -> int:
        total = 0

        for layer in self.symbolic_layers:
            total += layer.prune_by_threshold(threshold)

        total += self.assembly_layer.prune_by_threshold(threshold)

        return total

    def count_active_edges(self) -> int:
        total = 0

        for layer in self.symbolic_layers:
            total += layer.count_active_edges()

        total += self.assembly_layer.count_active_edges()

        return total

    def count_active_edges_per_layer(self) -> tuple[List[int], int]:
        symbolic_counts = [
            layer.count_active_edges()
            for layer in self.symbolic_layers
        ]

        assembly_count = self.assembly_layer.count_active_edges()

        return symbolic_counts, assembly_count

    def get_symbolic_layer_outputs(
        self,
        symbolic_inputs: Sequence[sp.Expr],
        rounding_decimals: int = 2,
    ) -> List[List[sp.Expr]]:
        all_outputs: List[List[sp.Expr]] = []

        x0 = list(symbolic_inputs)

        h = self.symbolic_layers[0].get_symbolic_output(
            x0,
            rounding_decimals=rounding_decimals,
        )

        all_outputs.append(h)

        for layer in self.symbolic_layers[1:]:
            h = layer.get_symbolic_output(
                x0 + h,
                rounding_decimals=rounding_decimals,
            )

            all_outputs.append(h)

        return all_outputs

    def get_symbolic_output(
        self,
        symbolic_inputs: Sequence[sp.Expr],
        rounding_decimals: int = 2,
    ) -> List[sp.Expr]:
        x0 = list(symbolic_inputs)

        h = self.symbolic_layers[0].get_symbolic_output(
            x0,
            rounding_decimals=rounding_decimals,
        )

        for layer in self.symbolic_layers[1:]:
            h = layer.get_symbolic_output(
                x0 + h,
                rounding_decimals=rounding_decimals,
            )

        out = self.assembly_layer.get_symbolic_output(
            x0 + h,
            rounding_decimals=rounding_decimals,
        )

        return [_sanitize_symbolic(expr) for expr in out]