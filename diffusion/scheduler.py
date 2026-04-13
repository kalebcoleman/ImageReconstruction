from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    """Stores the fixed coefficients used by forward and reverse diffusion."""

    num_timesteps: int
    schedule_name: str
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_hat: torch.Tensor
    sqrt_alpha_hat: torch.Tensor
    sqrt_one_minus_alpha_hat: torch.Tensor
    sqrt_recip_alpha: torch.Tensor
    posterior_variance: torch.Tensor


def extract_timestep_values(
    values: torch.Tensor,
    timesteps: torch.Tensor,
    reference_tensor: torch.Tensor,
) -> torch.Tensor:
    """Gather schedule values for a batch of timesteps and reshape for images."""
    gathered = values.gather(0, timesteps)
    return gathered.view(-1, 1, 1, 1).to(reference_tensor.dtype)


def get_noise_schedule(
    T: int,
    device: torch.device,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    schedule_name: str = "linear",
) -> DiffusionSchedule:
    """Create the forward diffusion schedule.

    Conceptually, diffusion destroys an image a little bit at every step.
    The beta values control how much fresh Gaussian noise is added at each
    timestep, and alpha_hat tracks how much of the original image survives.
    """

    if schedule_name == "linear":
        betas = torch.linspace(beta_start, beta_end, T, device=device, dtype=torch.float32)
    elif schedule_name == "cosine":
        betas = _cosine_beta_schedule(T, device=device)
    else:  # pragma: no cover - guarded by argparse in train.py
        raise ValueError(f"Unsupported diffusion schedule: {schedule_name}")

    alphas = 1.0 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)
    alpha_hat_previous = torch.cat(
        [torch.ones(1, device=device, dtype=torch.float32), alpha_hat[:-1]],
        dim=0,
    )
    posterior_variance = betas * (1.0 - alpha_hat_previous) / (1.0 - alpha_hat)
    posterior_variance = posterior_variance.clamp(min=1e-20)

    return DiffusionSchedule(
        num_timesteps=T,
        schedule_name=schedule_name,
        betas=betas,
        alphas=alphas,
        alpha_hat=alpha_hat,
        sqrt_alpha_hat=torch.sqrt(alpha_hat),
        sqrt_one_minus_alpha_hat=torch.sqrt(1.0 - alpha_hat),
        sqrt_recip_alpha=torch.sqrt(1.0 / alphas),
        posterior_variance=posterior_variance,
    )


def _cosine_beta_schedule(
    num_timesteps: int,
    *,
    device: torch.device,
    offset: float = 0.008,
) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule used by improved DDPM baselines."""

    steps = torch.linspace(0, num_timesteps, num_timesteps + 1, device=device, dtype=torch.float32)
    phase = ((steps / num_timesteps) + offset) / (1.0 + offset)
    alpha_bar = torch.cos(phase * (math.pi / 2.0)).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(min=1e-8, max=0.999)


def q_sample(
    x0: torch.Tensor,
    t: torch.Tensor,
    noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Apply the forward diffusion process to clean data x0.

    If t is larger, the returned image is more corrupted. Training samples
    random timesteps so the model learns to undo noise at every stage.
    """

    sqrt_alpha_hat_t = extract_timestep_values(schedule.sqrt_alpha_hat, t, x0)
    sqrt_one_minus_alpha_hat_t = extract_timestep_values(
        schedule.sqrt_one_minus_alpha_hat,
        t,
        x0,
    )
    return sqrt_alpha_hat_t * x0 + sqrt_one_minus_alpha_hat_t * noise


def predict_x0_from_noise(
    xt: torch.Tensor,
    t: torch.Tensor,
    predicted_noise: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    """Recover an estimate of the clean image from x_t and predicted noise."""

    sqrt_alpha_hat_t = extract_timestep_values(schedule.sqrt_alpha_hat, t, xt)
    sqrt_one_minus_alpha_hat_t = extract_timestep_values(
        schedule.sqrt_one_minus_alpha_hat,
        t,
        xt,
    )
    return (xt - sqrt_one_minus_alpha_hat_t * predicted_noise) / sqrt_alpha_hat_t
