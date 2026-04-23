from __future__ import annotations

from contextlib import nullcontext
import logging
from typing import Any

import torch


LOGGER = logging.getLogger(__name__)


def resolve_amp_dtype(
    amp_dtype: str,
    device: torch.device,
) -> torch.dtype | None:
    """Resolve a user-facing AMP mode into a concrete torch dtype."""

    normalized = amp_dtype.lower()
    if normalized == "none" or device.type != "cuda":
        return None

    if normalized == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if normalized == "bf16":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        LOGGER.warning("bf16 AMP was requested but is not supported on this GPU; falling back to fp16.")
        return torch.float16

    if normalized == "fp16":
        return torch.float16

    raise ValueError(f"Unsupported AMP dtype: {amp_dtype}")


def autocast_context(
    amp_dtype: str,
    device: torch.device,
) -> Any:
    """Return the right autocast context for the current device and precision mode."""

    resolved_dtype = resolve_amp_dtype(amp_dtype, device)
    if resolved_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=resolved_dtype)


def create_grad_scaler(
    amp_dtype: str,
    device: torch.device,
) -> torch.cuda.amp.GradScaler | None:
    """Create a gradient scaler when fp16 autocast is active on CUDA."""

    resolved_dtype = resolve_amp_dtype(amp_dtype, device)
    if device.type != "cuda" or resolved_dtype != torch.float16:
        return None
    return torch.cuda.amp.GradScaler(enabled=True)


def format_resolved_amp_dtype(
    amp_dtype: str,
    device: torch.device,
) -> str:
    """Human-readable AMP mode for logs and summaries."""

    resolved_dtype = resolve_amp_dtype(amp_dtype, device)
    if resolved_dtype is None:
        return "none"
    if resolved_dtype == torch.bfloat16:
        return "bf16"
    if resolved_dtype == torch.float16:
        return "fp16"
    return str(resolved_dtype)
