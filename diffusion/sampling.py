from __future__ import annotations

import math

import numpy as np
import torch

from diffusion.runtime import autocast_context
from diffusion.scheduler import (
    DiffusionSchedule,
    extract_timestep_values,
    predict_noise_from_model_output,
    predict_x0_from_model_output,
)


def _snapshot_steps(num_timesteps: int, num_snapshots: int) -> set[int]:
    if num_snapshots <= 0:
        return set()
    return set(np.linspace(num_timesteps - 1, 0, num=num_snapshots, dtype=int).tolist())


def _to_display_range(images: torch.Tensor) -> torch.Tensor:
    """Map diffusion samples from [-1, 1] into [0, 1] for saved artifacts."""

    return ((images + 1.0) / 2.0).clamp(0.0, 1.0)


def _validate_sampling_inputs(
    num_samples: int,
    image_shape: tuple[int, int, int],
    labels: torch.Tensor | None,
    guidance_scale: float,
) -> None:
    channels, height, width = image_shape
    if channels < 1 or height < 1 or width < 1:
        raise ValueError(f"Invalid image_shape: {image_shape}")
    if labels is not None and labels.shape[0] != num_samples:
        raise ValueError(
            "Sampling labels must have the same batch size as num_samples. "
            f"Got labels={labels.shape[0]} and num_samples={num_samples}."
        )
    if guidance_scale < 0.0:
        raise ValueError("guidance_scale must be non-negative.")


def _guided_model_output(
    model: torch.nn.Module,
    samples: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    labels: torch.Tensor | None,
    guidance_scale: float,
    prediction_type: str,
    scheduler: DiffusionSchedule,
    amp_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    with autocast_context(amp_dtype, samples.device):
        model_output = model(samples, timesteps, labels)
        if labels is not None and guidance_scale != 1.0:
            unconditional_output = model(samples, timesteps, labels, force_uncond=True)
            model_output = unconditional_output + guidance_scale * (
                model_output - unconditional_output
            )

    if model_output.shape != samples.shape:
        raise ValueError(
            "Diffusion model output shape must match the sample tensor shape. "
            f"Expected {tuple(samples.shape)}, got {tuple(model_output.shape)}."
        )

    predicted_noise = predict_noise_from_model_output(
        samples,
        timesteps,
        model_output,
        scheduler,
        prediction_type,
    )
    predicted_x0 = predict_x0_from_model_output(
        samples,
        timesteps,
        model_output,
        scheduler,
        prediction_type,
    ).clamp(-1.0, 1.0)
    return predicted_noise, predicted_x0


def _snapshot_state(
    return_intermediate: bool,
    samples: torch.Tensor,
    intermediate_images: list[torch.Tensor],
    intermediate_steps: list[int],
    step: int,
    snapshot_indices: set[int],
) -> None:
    if return_intermediate and step in snapshot_indices:
        intermediate_images.append(_to_display_range(samples.detach().cpu()))
        intermediate_steps.append(step)


def _resolve_ddim_steps(
    schedule: DiffusionSchedule,
    sampling_steps: int | None,
) -> list[int]:
    requested_steps = sampling_steps or min(50, schedule.num_timesteps)
    if requested_steps < 1:
        raise ValueError("sampling_steps must be at least 1 for DDIM sampling.")

    raw_steps = np.linspace(
        schedule.num_timesteps - 1,
        0,
        num=min(requested_steps, schedule.num_timesteps),
        dtype=int,
    ).tolist()
    resolved_steps: list[int] = []
    for step in raw_steps:
        if not resolved_steps or step != resolved_steps[-1]:
            resolved_steps.append(step)
    return resolved_steps


@torch.no_grad()
def sample_ddpm(
    model: torch.nn.Module,
    scheduler: DiffusionSchedule,
    device: torch.device,
    *,
    num_samples: int,
    image_shape: tuple[int, int, int],
    initial_noise: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    prediction_type: str = "eps",
    amp_dtype: str = "none",
    return_intermediate: bool = False,
    num_snapshots: int = 8,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor], list[int]]:
    """Run ancestral DDPM sampling with optional classifier-free guidance."""

    _validate_sampling_inputs(num_samples, image_shape, labels, guidance_scale)
    model.eval()
    channels, height, width = image_shape
    samples = (
        initial_noise.to(device)
        if initial_noise is not None
        else torch.randn(num_samples, channels, height, width, device=device)
    )
    if samples.shape != (num_samples, channels, height, width):
        raise ValueError(
            "initial_noise must match the requested sample shape. "
            f"Expected {(num_samples, channels, height, width)}, got {tuple(samples.shape)}."
        )
    snapshot_indices = _snapshot_steps(scheduler.num_timesteps, num_snapshots)
    intermediate_images: list[torch.Tensor] = []
    intermediate_steps: list[int] = []

    if return_intermediate:
        intermediate_images.append(samples.detach().cpu())
        intermediate_steps.append(scheduler.num_timesteps)

    for step in reversed(range(scheduler.num_timesteps)):
        timesteps = torch.full((num_samples,), step, device=device, dtype=torch.long)
        predicted_noise, _ = _guided_model_output(
            model,
            samples,
            timesteps,
            labels=labels,
            guidance_scale=guidance_scale,
            prediction_type=prediction_type,
            scheduler=scheduler,
            amp_dtype=amp_dtype,
        )

        beta_t = extract_timestep_values(scheduler.betas, timesteps, samples)
        sqrt_one_minus_alpha_hat_t = extract_timestep_values(
            scheduler.sqrt_one_minus_alpha_hat,
            timesteps,
            samples,
        )
        sqrt_recip_alpha_t = extract_timestep_values(scheduler.sqrt_recip_alpha, timesteps, samples)
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

        _snapshot_state(
            return_intermediate,
            samples,
            intermediate_images,
            intermediate_steps,
            step,
            snapshot_indices,
        )

    samples = _to_display_range(samples.detach().cpu())
    if not return_intermediate:
        return samples

    if not intermediate_steps or intermediate_steps[-1] != 0:
        intermediate_images.append(samples)
        intermediate_steps.append(0)
    return samples, intermediate_images, intermediate_steps


@torch.no_grad()
def sample_ddim(
    model: torch.nn.Module,
    scheduler: DiffusionSchedule,
    device: torch.device,
    *,
    num_samples: int,
    image_shape: tuple[int, int, int],
    initial_noise: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    prediction_type: str = "eps",
    sampling_steps: int | None = None,
    ddim_eta: float = 0.0,
    amp_dtype: str = "none",
    return_intermediate: bool = False,
    num_snapshots: int = 8,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor], list[int]]:
    """Run DDIM sampling with arbitrary image shape and optional CFG."""

    _validate_sampling_inputs(num_samples, image_shape, labels, guidance_scale)
    if ddim_eta < 0.0:
        raise ValueError("ddim_eta must be non-negative.")

    model.eval()
    channels, height, width = image_shape
    samples = (
        initial_noise.to(device)
        if initial_noise is not None
        else torch.randn(num_samples, channels, height, width, device=device)
    )
    if samples.shape != (num_samples, channels, height, width):
        raise ValueError(
            "initial_noise must match the requested sample shape. "
            f"Expected {(num_samples, channels, height, width)}, got {tuple(samples.shape)}."
        )
    sampling_schedule = _resolve_ddim_steps(scheduler, sampling_steps)
    snapshot_indices = _snapshot_steps(len(sampling_schedule), num_snapshots)
    intermediate_images: list[torch.Tensor] = []
    intermediate_steps: list[int] = []

    if return_intermediate:
        intermediate_images.append(samples.detach().cpu())
        intermediate_steps.append(scheduler.num_timesteps)

    for step_index, step in enumerate(sampling_schedule):
        timesteps = torch.full((num_samples,), step, device=device, dtype=torch.long)
        predicted_noise, predicted_x0 = _guided_model_output(
            model,
            samples,
            timesteps,
            labels=labels,
            guidance_scale=guidance_scale,
            prediction_type=prediction_type,
            scheduler=scheduler,
            amp_dtype=amp_dtype,
        )

        alpha_hat_t = extract_timestep_values(scheduler.alpha_hat, timesteps, samples)
        prev_step = sampling_schedule[step_index + 1] if step_index + 1 < len(sampling_schedule) else -1
        if prev_step >= 0:
            prev_timesteps = torch.full((num_samples,), prev_step, device=device, dtype=torch.long)
            alpha_hat_prev = extract_timestep_values(scheduler.alpha_hat, prev_timesteps, samples)
        else:
            alpha_hat_prev = torch.ones_like(alpha_hat_t)

        sigma_t = ddim_eta * torch.sqrt(
            ((1.0 - alpha_hat_prev) / (1.0 - alpha_hat_t)).clamp(min=0.0)
            * (1.0 - (alpha_hat_t / alpha_hat_prev)).clamp(min=0.0)
        )
        direction = torch.sqrt((1.0 - alpha_hat_prev - sigma_t.square()).clamp(min=0.0)) * predicted_noise
        noise = torch.randn_like(samples) if step > 0 else torch.zeros_like(samples)
        samples = torch.sqrt(alpha_hat_prev) * predicted_x0 + direction + sigma_t * noise

        if return_intermediate and step_index in snapshot_indices:
            intermediate_images.append(_to_display_range(samples.detach().cpu()))
            intermediate_steps.append(step)

    samples = _to_display_range(samples.detach().cpu())
    if not return_intermediate:
        return samples

    if not intermediate_steps or intermediate_steps[-1] != 0:
        intermediate_images.append(samples)
        intermediate_steps.append(0)
    return samples, intermediate_images, intermediate_steps


@torch.no_grad()
def sample_images(
    model: torch.nn.Module,
    scheduler: DiffusionSchedule,
    device: torch.device,
    num_samples: int,
    image_shape: tuple[int, int, int],
    initial_noise: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    prediction_type: str = "eps",
    sampler_name: str = "ddpm",
    sampling_steps: int | None = None,
    ddim_eta: float = 0.0,
    amp_dtype: str = "none",
    return_intermediate: bool = False,
    num_snapshots: int = 8,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor], list[int]]:
    """Dispatch to DDPM or DDIM while preserving the existing sampling API."""

    normalized_sampler = sampler_name.lower()
    if normalized_sampler == "ddpm":
        if sampling_steps is not None and sampling_steps != scheduler.num_timesteps:
            raise ValueError(
                "DDPM sampling currently uses the full training schedule. "
                f"Expected sampling_steps={scheduler.num_timesteps} or omitted, got {sampling_steps}."
            )
        return sample_ddpm(
            model,
            scheduler,
            device,
            num_samples=num_samples,
            image_shape=image_shape,
            initial_noise=initial_noise,
            labels=labels,
            guidance_scale=guidance_scale,
            prediction_type=prediction_type,
            amp_dtype=amp_dtype,
            return_intermediate=return_intermediate,
            num_snapshots=num_snapshots,
        )
    if normalized_sampler == "ddim":
        return sample_ddim(
            model,
            scheduler,
            device,
            num_samples=num_samples,
            image_shape=image_shape,
            initial_noise=initial_noise,
            labels=labels,
            guidance_scale=guidance_scale,
            prediction_type=prediction_type,
            sampling_steps=sampling_steps,
            ddim_eta=ddim_eta,
            amp_dtype=amp_dtype,
            return_intermediate=return_intermediate,
            num_snapshots=num_snapshots,
        )
    raise ValueError(f"Unsupported sampler: {sampler_name}")
