import numpy as np
import torch

from config import ExperimentConfig
from neural_operators import build_neural_operator
from operator_datasets import build_operator_dataset
from trainer import evaluate, train_model
from visualization import save_prediction_figures, save_training_curves


def main() -> None:
    config = ExperimentConfig()

    torch.manual_seed(config.training.seed)
    np.random.seed(config.training.seed)

    device = torch.device(config.training.device)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.training.seed)
        torch.backends.cudnn.benchmark = True

    experiment_models_dir = config.paths.models_dir / config.experiment_name
    experiment_figures_dir = config.paths.figures_dir / config.experiment_name

    experiment_models_dir.mkdir(parents=True, exist_ok=True)
    experiment_figures_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, dataset_info = build_operator_dataset(config)

    model = build_neural_operator(config, device)

    best_model_path = experiment_models_dir / "best_model.pt"

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        best_model_path=best_model_path,
        dataset_info=dataset_info,
    )

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    loss_fn = torch.nn.MSELoss()

    test_mse, test_rel_l2 = evaluate(
        model=model,
        loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print(f"Best model saved to: {best_model_path}")
    print(f"Test MSE: {test_mse:.6e}")
    print(f"Test relative L2: {test_rel_l2:.6e}")

    save_training_curves(
        history=history,
        figures_dir=experiment_figures_dir,
    )

    save_prediction_figures(
        model=model,
        test_loader=test_loader,
        dataset_info=dataset_info,
        config=config,
        figures_dir=experiment_figures_dir,
        device=device,
    )

    print(f"Figures saved to: {experiment_figures_dir}")


if __name__ == "__main__":
    main()