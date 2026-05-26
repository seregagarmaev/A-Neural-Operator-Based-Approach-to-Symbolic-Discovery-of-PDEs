import torch

from hypernos.architectures import CNO, FNO

from config import ExperimentConfig


class FNNOperator(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_width: int,
        n_hidden_layers: int,
        activation: str,
    ) -> None:
        super().__init__()

        if activation == "gelu":
            activation_layer = torch.nn.GELU
        elif activation == "relu":
            activation_layer = torch.nn.ReLU
        elif activation == "tanh":
            activation_layer = torch.nn.Tanh
        else:
            raise ValueError(f"Unknown FNN activation: {activation}")

        layers = []

        layers.append(torch.nn.Linear(input_size, hidden_width))
        layers.append(activation_layer())

        for _ in range(n_hidden_layers - 1):
            layers.append(torch.nn.Linear(hidden_width, hidden_width))
            layers.append(activation_layer())

        layers.append(torch.nn.Linear(hidden_width, input_size))

        self.network = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]

        x = x[:, :, 0]
        y = self.network(x)
        y = y.reshape(batch_size, -1, 1)

        return y


def get_task_size(config: ExperimentConfig) -> int:
    if config.task.name == "hereditary_integral":
        return config.task.nt

    return config.task.nx


def build_hypernos_fno(
    config: ExperimentConfig,
    device: torch.device,
) -> FNO:
    model_config = config.model

    model = FNO(
        problem_dim=model_config.problem_dim,
        in_dim=model_config.in_dim,
        d_v=model_config.width,
        out_dim=model_config.out_dim,
        L=model_config.n_layers,
        modes=model_config.modes,
        fun_act=model_config.fun_act,
        weights_norm=model_config.weights_norm,
        arc=model_config.fno_arc,
        RNN=model_config.rnn,
        FFTnorm=model_config.fft_norm,
        padding=model_config.padding,
        device=device,
        example_output_normalizer=None,
        retrain_fno=config.training.seed,
    )

    return model.to(device)


def build_hypernos_cno(
    config: ExperimentConfig,
    device: torch.device,
) -> CNO:
    model_config = config.model

    size = get_task_size(config)

    model = CNO(
        problem_dim=model_config.problem_dim,
        in_dim=model_config.in_dim,
        out_dim=model_config.out_dim,
        size=size,
        N_layers=model_config.n_layers,
        N_res=model_config.n_res,
        N_res_neck=model_config.n_res_neck,
        channel_multiplier=model_config.channel_multiplier,
        kernel_size=model_config.kernel_size,
        use_bn=model_config.use_bn,
        device=device,
    )

    return model.to(device)


def build_fnn_operator(
    config: ExperimentConfig,
    device: torch.device,
) -> FNNOperator:
    model_config = config.model

    size = get_task_size(config)

    model = FNNOperator(
        input_size=size,
        hidden_width=model_config.hidden_width,
        n_hidden_layers=model_config.n_hidden_layers,
        activation=model_config.activation,
    )

    return model.to(device)


def build_neural_operator(
    config: ExperimentConfig,
    device: torch.device,
) -> torch.nn.Module:
    if config.model.name == "fno":
        return build_hypernos_fno(config, device)

    if config.model.name == "cno":
        return build_hypernos_cno(config, device)

    if config.model.name == "fnn":
        return build_fnn_operator(config, device)

    raise ValueError(f"Unknown neural operator model: {config.model.name}")