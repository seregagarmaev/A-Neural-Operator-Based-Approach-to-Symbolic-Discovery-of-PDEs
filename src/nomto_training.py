from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader


@dataclass
class NOMTOTrainingConfig:
    device: str = "cuda:0"
    seed: int = 42

    learning_rate: float = 1e-3
    weight_decay: float = 0.0

    phase1_epochs: int = 1000
    phase2_epochs: int = 1000
    phase3_epochs: int = 1000

    l1_reg_coeff_phase2: float = 1e-5

    prune_every_epochs_phase2: int = 100
    pruning_fraction_phase2: float = 0.1
    pruning_min_active_edges: int = 20

    print_every: int = 100


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_optimizer(model: torch.nn.Module, cfg: NOMTOTrainingConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )


def l1_penalty(model: torch.nn.Module) -> torch.Tensor:
    terms = model.weights_for_regularization()

    out = None
    for w in terms:
        term = w.abs().sum()
        out = term if out is None else out + term

    return out


@torch.no_grad()
def prune_fraction_by_magnitude(
    model: torch.nn.Module,
    fraction: float,
    min_active_edges: int,
) -> int:
    fraction = float(fraction)
    min_active_edges = int(min_active_edges)

    if fraction <= 0.0:
        return 0

    active_edges = model.count_active_edges()

    if active_edges <= min_active_edges:
        return 0

    entries: list[tuple[torch.nn.Module, torch.Tensor]] = []
    magnitudes: list[torch.Tensor] = []

    for layer in model.symbolic_layers:
        active = layer.mask > 0.5

        if active.any():
            flat_indices = active.view(-1).nonzero(as_tuple=False).reshape(-1)
            entries.extend((layer, idx) for idx in flat_indices)
            magnitudes.append(layer.weights.view(-1)[flat_indices].abs())

    active = model.assembly_layer.mask > 0.5

    if active.any():
        flat_indices = active.view(-1).nonzero(as_tuple=False).reshape(-1)
        entries.extend((model.assembly_layer, idx) for idx in flat_indices)
        magnitudes.append(model.assembly_layer.weights.view(-1)[flat_indices].abs())

    all_magnitudes = torch.cat(magnitudes)

    n_active = int(all_magnitudes.numel())
    n_target = int(np.floor(fraction * n_active))
    n_allowed = n_active - min_active_edges
    n_prune = max(0, min(n_target, n_allowed))

    if n_prune <= 0:
        return 0

    prune_indices = torch.topk(
        all_magnitudes,
        k=n_prune,
        largest=False,
    ).indices.tolist()

    for idx in prune_indices:
        layer, flat_idx = entries[idx]

        layer.mask.view(-1)[flat_idx] = 0.0
        layer.weights.view(-1)[flat_idx] = 0.0

    return n_prune


def run_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    l1_coeff: float = 0.0,
) -> tuple[float, float, float]:
    model.train()

    total_loss = 0.0
    total_data_loss = 0.0
    total_l1_loss = 0.0
    n_samples = 0

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)

        pred = model(X)

        data_loss = loss_fn(pred, y)

        if l1_coeff > 0.0:
            l1_loss = float(l1_coeff) * l1_penalty(model)
        else:
            l1_loss = data_loss.new_tensor(0.0)

        loss = data_loss + l1_loss

        loss.backward()
        optimizer.step()

        batch_size = X.shape[0]

        total_loss += float(loss.item()) * batch_size
        total_data_loss += float(data_loss.item()) * batch_size
        total_l1_loss += float(l1_loss.item()) * batch_size
        n_samples += batch_size

    denom = max(1, n_samples)

    return (
        total_loss / denom,
        total_data_loss / denom,
        total_l1_loss / denom,
    )


def train_nomto(
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable,
    cfg: NOMTOTrainingConfig,
) -> tuple[torch.nn.Module, dict[str, list[float]]]:
    set_seed(cfg.seed)

    device = torch.device(cfg.device)
    model = model.to(device)

    model.freeze_neural_operators()

    optimizer = make_optimizer(model, cfg)

    history = {
        "total_loss": [],
        "data_loss": [],
        "l1_loss": [],
        "active_edges": [],
        "phase": [],
    }

    global_epoch = 0

    def record(
        phase: str,
        total_loss: float,
        data_loss: float,
        l1_loss: float,
    ) -> None:
        history["total_loss"].append(total_loss)
        history["data_loss"].append(data_loss)
        history["l1_loss"].append(l1_loss)
        history["active_edges"].append(model.count_active_edges())
        history["phase"].append(phase)

    def print_status(
        phase: str,
        epoch_in_phase: int,
        total_loss: float,
        data_loss: float,
        l1_loss: float,
    ) -> None:
        if cfg.print_every <= 0:
            return

        if global_epoch == 0 or (global_epoch + 1) % cfg.print_every == 0:
            print(
                f"[{phase} | global_epoch={global_epoch + 1} | phase_epoch={epoch_in_phase + 1}] "
                f"total={total_loss:.6e}, "
                f"data={data_loss:.6e}, "
                f"l1={l1_loss:.6e}, "
                f"active_edges={model.count_active_edges()}"
            )

    for epoch in range(cfg.phase1_epochs):
        total_loss, data_loss, l1_loss = run_one_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            l1_coeff=0.0,
        )

        record("phase1", total_loss, data_loss, l1_loss)
        print_status("PHASE1", epoch, total_loss, data_loss, l1_loss)

        global_epoch += 1

    for epoch in range(cfg.phase2_epochs):
        total_loss, data_loss, l1_loss = run_one_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            l1_coeff=cfg.l1_reg_coeff_phase2,
        )

        record("phase2", total_loss, data_loss, l1_loss)
        print_status("PHASE2", epoch, total_loss, data_loss, l1_loss)

        should_prune = (
            cfg.prune_every_epochs_phase2 > 0
            and cfg.pruning_fraction_phase2 > 0.0
            and (epoch + 1) % cfg.prune_every_epochs_phase2 == 0
            and model.count_active_edges() > cfg.pruning_min_active_edges
        )

        if should_prune:
            before = model.count_active_edges()

            pruned = prune_fraction_by_magnitude(
                model=model,
                fraction=cfg.pruning_fraction_phase2,
                min_active_edges=cfg.pruning_min_active_edges,
            )

            after = model.count_active_edges()

            if pruned > 0:
                print(
                    f"[PRUNE | global_epoch={global_epoch + 1}] "
                    f"pruned={pruned}, active_edges={before}->{after}"
                )

        global_epoch += 1

    for epoch in range(cfg.phase3_epochs):
        total_loss, data_loss, l1_loss = run_one_epoch(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            l1_coeff=0.0,
        )

        record("phase3", total_loss, data_loss, l1_loss)
        print_status("PHASE3", epoch, total_loss, data_loss, l1_loss)

        global_epoch += 1

    return model, history