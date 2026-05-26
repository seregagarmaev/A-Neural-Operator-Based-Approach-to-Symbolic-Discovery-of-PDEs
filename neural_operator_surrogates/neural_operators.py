import torch

from hypernos.architectures import CNO, FNO

from config import ExperimentConfig


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

    size = config.task.nt if config.task.name == "hereditary_integral" else config.task.nx

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


def build_neural_operator(
    config: ExperimentConfig,
    device: torch.device,
) -> torch.nn.Module:
    if config.model.name == "fno":
        return build_hypernos_fno(config, device)

    if config.model.name == "cno":
        return build_hypernos_cno(config, device)

    raise ValueError(f"Unknown neural operator model: {config.model.name}")