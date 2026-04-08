from __future__ import annotations

import math

import torch
import torch.nn as nn


def _resolve_group_count(num_channels: int, preferred_groups: int = 8) -> int:
    """Choose a GroupNorm setting that always divides the channel count."""
    for group_count in range(preferred_groups, 0, -1):
        if num_channels % group_count == 0:
            return group_count
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    """Encode a timestep into a smooth vector the network can condition on.

    A timestep tells the model how far into the corruption process we are.
    Small timesteps are almost clean images, while large timesteps are heavily
    destroyed by Gaussian noise.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.embedding_dim // 2
        exponent = -math.log(10_000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * exponent
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embeddings = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.embedding_dim % 2 == 1:
            embeddings = torch.cat(
                [embeddings, torch.zeros_like(embeddings[:, :1])],
                dim=1,
            )
        return embeddings


class DiffusionResBlock(nn.Module):
    """A small residual block conditioned on the timestep embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_resolve_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_resolve_group_count(out_channels), out_channels)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.activation = nn.SiLU()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(x)
        hidden = self.norm1(hidden)
        hidden = self.activation(hidden)

        # The timestep embedding tells the block how much noise to expect.
        time_bias = self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        hidden = hidden + time_bias

        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        hidden = self.activation(hidden)
        return hidden + self.skip(x)


class DiffusionUNet(nn.Module):
    """A lightweight UNet-style network for 28x28 grayscale DDPM training.

    The network receives a noisy image x_t plus its timestep t and predicts the
    noise epsilon that was added. Predicting noise is stable because the target
    is always Gaussian, regardless of which digit or clothing item is present.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32, time_dim: int = 64) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        self.input_projection = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.down_block = DiffusionResBlock(base_channels, base_channels, time_dim)
        self.downsample = nn.Conv2d(
            base_channels,
            base_channels * 2,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.mid_block = DiffusionResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.bottleneck = DiffusionResBlock(base_channels * 2, base_channels * 2, time_dim)
        self.upsample = nn.ConvTranspose2d(
            base_channels * 2,
            base_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_block = DiffusionResBlock(base_channels * 2, base_channels, time_dim)
        self.output_norm = nn.GroupNorm(_resolve_group_count(base_channels), base_channels)
        self.output_activation = nn.SiLU()
        self.output_projection = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)
        time_embedding = self.time_mlp(time_embedding)

        x = self.input_projection(x)
        skip = self.down_block(x, time_embedding)
        x = self.downsample(skip)
        x = self.mid_block(x, time_embedding)
        x = self.bottleneck(x, time_embedding)
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.up_block(x, time_embedding)
        x = self.output_norm(x)
        x = self.output_activation(x)
        return self.output_projection(x)
