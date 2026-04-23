from __future__ import annotations

import math

import numpy as np
import torch
from PIL import Image
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from diffusion.backbones.adm_unet import (
    ADMUNet,
    default_attention_resolutions,
    default_channel_mults,
)
from diffusion.data import build_diffusion_transform, resolve_diffusion_data_config
from diffusion.sampling import sample_images
from diffusion.scheduler import (
    get_diffusion_target,
    get_noise_schedule,
    predict_x0_from_model_output,
    q_sample,
)
from diffusion.training import train_diffusion_epoch


def test_resolve_diffusion_data_config_defaults() -> None:
    adm_config = resolve_diffusion_data_config(
        "mnist",
        diffusion_backbone="adm",
        image_size=None,
        channels=None,
    )
    legacy_config = resolve_diffusion_data_config(
        "mnist",
        diffusion_backbone="legacy",
        image_size=None,
        channels=None,
    )

    assert adm_config.image_size == 64
    assert adm_config.channels == 3
    assert legacy_config.image_size == 28
    assert legacy_config.channels == 1


def test_diffusion_transform_converts_mnist_to_rgb_64() -> None:
    image = Image.fromarray(np.random.randint(0, 255, size=(28, 28), dtype=np.uint8), mode="L")
    transform = build_diffusion_transform(
        "mnist",
        train=True,
        image_size=64,
        channels=3,
    )

    tensor = transform(image)

    assert tensor.shape == (3, 64, 64)
    assert tensor.dtype == torch.float32
    assert float(tensor.min()) >= -1.0 - 1e-5
    assert float(tensor.max()) <= 1.0 + 1e-5


def test_adm_unet_supports_conditional_and_unconditional_shapes() -> None:
    model = ADMUNet(
        in_channels=3,
        image_size=64,
        base_channels=8,
        time_dim=64,
        num_res_blocks=1,
        channel_mult=default_channel_mults(64),
        attention_resolutions=default_attention_resolutions(64, "mnist"),
        num_classes=10,
    )
    inputs = torch.randn(2, 3, 64, 64)
    timesteps = torch.tensor([0, 1], dtype=torch.long)
    labels = torch.tensor([3, 7], dtype=torch.long)

    model.train()
    conditional_output = model(inputs, timesteps, labels)

    model.eval()
    unconditional_output = model(inputs, timesteps, labels, force_uncond=True)

    assert conditional_output.shape == inputs.shape
    assert unconditional_output.shape == inputs.shape
    assert torch.isfinite(conditional_output).all()
    assert torch.isfinite(unconditional_output).all()


def test_adm_unet_supports_unconditional_forward_without_labels() -> None:
    model = ADMUNet(
        in_channels=3,
        image_size=64,
        base_channels=8,
        time_dim=64,
        num_res_blocks=1,
        channel_mult=default_channel_mults(64),
        attention_resolutions=default_attention_resolutions(64, "mnist"),
        num_classes=10,
    )
    inputs = torch.randn(2, 3, 64, 64)
    timesteps = torch.tensor([0, 1], dtype=torch.long)

    outputs = model(inputs, timesteps, labels=None)

    assert outputs.shape == inputs.shape
    assert torch.isfinite(outputs).all()


def test_sample_images_respects_requested_shape() -> None:
    class ZeroPredictor(torch.nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            labels: torch.Tensor | None = None,
            force_uncond: bool = False,
        ) -> torch.Tensor:
            del timesteps, labels, force_uncond
            return torch.zeros_like(x)

    schedule = get_noise_schedule(4, torch.device("cpu"))
    samples, intermediate_images, intermediate_steps = sample_images(
        ZeroPredictor(),
        schedule,
        torch.device("cpu"),
        num_samples=2,
        image_shape=(3, 64, 64),
        labels=torch.tensor([1, 2], dtype=torch.long),
        guidance_scale=2.0,
        return_intermediate=True,
        num_snapshots=3,
    )

    assert samples.shape == (2, 3, 64, 64)
    assert len(intermediate_images) == len(intermediate_steps)
    assert intermediate_steps[-1] == 0
    assert torch.isfinite(samples).all()


def test_ddim_sampling_respects_requested_shape() -> None:
    class ZeroPredictor(torch.nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            labels: torch.Tensor | None = None,
            force_uncond: bool = False,
        ) -> torch.Tensor:
            del timesteps, labels, force_uncond
            return torch.zeros_like(x)

    schedule = get_noise_schedule(8, torch.device("cpu"))
    samples = sample_images(
        ZeroPredictor(),
        schedule,
        torch.device("cpu"),
        num_samples=2,
        image_shape=(3, 64, 64),
        sampler_name="ddim",
        sampling_steps=4,
    )

    assert samples.shape == (2, 3, 64, 64)
    assert torch.isfinite(samples).all()


def test_cfg_sampling_changes_outputs() -> None:
    class GuidedPredictor(torch.nn.Module):
        def forward(
            self,
            x: torch.Tensor,
            timesteps: torch.Tensor,
            labels: torch.Tensor | None = None,
            force_uncond: bool = False,
        ) -> torch.Tensor:
            del timesteps
            if force_uncond or labels is None:
                return torch.zeros_like(x)
            return torch.ones_like(x)

    schedule = get_noise_schedule(8, torch.device("cpu"))
    labels = torch.tensor([1, 2], dtype=torch.long)

    torch.manual_seed(1234)
    unconditional_like = sample_images(
        GuidedPredictor(),
        schedule,
        torch.device("cpu"),
        num_samples=2,
        image_shape=(3, 64, 64),
        labels=labels,
        guidance_scale=0.0,
        sampler_name="ddim",
        sampling_steps=4,
    )
    torch.manual_seed(1234)
    guided = sample_images(
        GuidedPredictor(),
        schedule,
        torch.device("cpu"),
        num_samples=2,
        image_shape=(3, 64, 64),
        labels=labels,
        guidance_scale=3.0,
        sampler_name="ddim",
        sampling_steps=4,
    )

    assert unconditional_like.shape == guided.shape == (2, 3, 64, 64)
    assert torch.isfinite(guided).all()
    assert not torch.allclose(unconditional_like, guided)


def test_eps_and_v_targets_reconstruct_x0() -> None:
    schedule = get_noise_schedule(8, torch.device("cpu"))
    x0 = torch.randn(2, 3, 16, 16)
    noise = torch.randn_like(x0)
    timesteps = torch.tensor([1, 6], dtype=torch.long)
    xt = q_sample(x0, timesteps, noise, schedule)

    eps_target = get_diffusion_target(x0, noise, timesteps, schedule, "eps")
    v_target = get_diffusion_target(x0, noise, timesteps, schedule, "v")
    eps_reconstruction = predict_x0_from_model_output(xt, timesteps, eps_target, schedule, "eps")
    v_reconstruction = predict_x0_from_model_output(xt, timesteps, v_target, schedule, "v")

    assert torch.allclose(eps_reconstruction, x0, atol=1e-5, rtol=1e-5)
    assert torch.allclose(v_reconstruction, x0, atol=1e-5, rtol=1e-5)


def test_train_diffusion_epoch_smoke() -> None:
    device = torch.device("cpu")
    model = ADMUNet(
        in_channels=3,
        image_size=64,
        base_channels=8,
        time_dim=64,
        num_res_blocks=1,
        channel_mult=default_channel_mults(64),
        attention_resolutions=default_attention_resolutions(64, "mnist"),
        num_classes=10,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    schedule = get_noise_schedule(4, device)
    images = torch.randn(4, 3, 64, 64)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)

    loss = train_diffusion_epoch(
        model,
        loader,
        optimizer,
        schedule,
        device,
    )

    assert math.isfinite(loss)
    assert loss >= 0.0
