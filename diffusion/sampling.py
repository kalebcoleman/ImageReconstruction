from __future__ import annotations

import numpy as np
import torch

from diffusion.scheduler import DiffusionSchedule, extract_timestep_values


def _snapshot_steps(num_timesteps: int, num_snapshots: int) -> set[int]:
    if num_snapshots <= 0:
        return set()
    return set(np.linspace(num_timesteps - 1, 0, num=num_snapshots, dtype=int).tolist())


def _to_display_range(images: torch.Tensor) -> torch.Tensor:
    """Map diffusion samples from [-1, 1] into [0, 1] for saved artifacts."""

    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


@torch.no_grad()
def sample_images(
    model: torch.nn.Module,
    scheduler: DiffusionSchedule,
    device: torch.device,
    num_samples: int,
    return_intermediate: bool = False,
    num_snapshots: int = 8,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor], list[int]]:
    """Run the reverse diffusion process from pure noise back to images.

    Sampling starts from Gaussian noise because the forward process eventually
    turns every training image into noise. The model then walks backward from
    timestep T to 0, removing the predicted noise a little bit at a time.
    """

    model.eval()
    samples = torch.randn(num_samples, 1, 28, 28, device=device)
    snapshot_indices = _snapshot_steps(scheduler.num_timesteps, num_snapshots)
    intermediate_images: list[torch.Tensor] = []
    intermediate_steps: list[int] = []

    if return_intermediate:
        intermediate_images.append(samples.detach().cpu())
        intermediate_steps.append(scheduler.num_timesteps)

    for step in reversed(range(scheduler.num_timesteps)):
        timesteps = torch.full((num_samples,), step, device=device, dtype=torch.long)
        predicted_noise = model(samples, timesteps)

        beta_t = extract_timestep_values(scheduler.betas, timesteps, samples)
        sqrt_one_minus_alpha_hat_t = extract_timestep_values(
            scheduler.sqrt_one_minus_alpha_hat,
            timesteps,
            samples,
        )
        sqrt_recip_alpha_t = extract_timestep_values(scheduler.sqrt_recip_alpha, timesteps, samples)

        # Reverse DDPM update: remove predicted noise, then optionally inject a
        # smaller amount of fresh noise except on the final step.
        model_mean = sqrt_recip_alpha_t * (
            samples - (beta_t / sqrt_one_minus_alpha_hat_t) * predicted_noise
        )

        if step > 0:
            posterior_variance_t = extract_timestep_values(
                scheduler.posterior_variance,
                timesteps,
                samples,
            )
            samples = model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(samples)
        else:
            samples = model_mean

        if return_intermediate and step in snapshot_indices:
            intermediate_images.append(_to_display_range(samples.detach().cpu()))
            intermediate_steps.append(step)

    samples = _to_display_range(samples.detach().cpu())
    if not return_intermediate:
        return samples

    if not intermediate_steps or intermediate_steps[-1] != 0:
        intermediate_images.append(samples)
        intermediate_steps.append(0)

    return samples, intermediate_images, intermediate_steps
