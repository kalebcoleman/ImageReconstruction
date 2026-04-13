from __future__ import annotations

import copy

import torch


def create_ema_model(model: torch.nn.Module) -> torch.nn.Module:
    """Clone a model for EMA tracking and freeze it for evaluation-only use."""

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema_model(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    decay: float,
) -> None:
    """Apply one EMA update after an optimizer step."""

    if not 0.0 <= decay < 1.0:
        raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")

    ema_state = ema_model.state_dict()
    model_state = model.state_dict()

    for key, ema_value in ema_state.items():
        model_value = model_state[key].detach()
        if torch.is_floating_point(ema_value):
            ema_value.lerp_(model_value, 1.0 - decay)
        else:
            ema_value.copy_(model_value)


def select_eval_model(
    model: torch.nn.Module,
    ema_model: torch.nn.Module | None,
) -> torch.nn.Module:
    """Prefer EMA weights for evaluation when available."""

    return ema_model if ema_model is not None else model
