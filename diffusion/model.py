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
    """A residual block conditioned on the timestep embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_resolve_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_resolve_group_count(out_channels), out_channels)
        self.time_projection = nn.Linear(time_dim, out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(x)
        hidden = self.norm1(hidden)
        hidden = torch.relu(hidden)

        # The timestep embedding tells the block how much noise to expect.
        time_bias = self.time_projection(time_embedding).unsqueeze(-1).unsqueeze(-1)
        hidden = hidden + time_bias

        hidden = self.conv2(hidden)
        hidden = self.norm2(hidden)
        hidden = hidden + self.skip(x)
        return torch.relu(hidden)


class DiffusionUNet(nn.Module):
    """A lightweight 3-level DDPM-style UNet for 28x28 grayscale images.

    The network receives a noisy image x_t plus its timestep t and predicts the
    noise epsilon that was added. Predicting noise is stable because the target
    is always Gaussian, regardless of which digit or clothing item is present.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64, time_dim: int = 64) -> None:
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )

        level1_channels = base_channels
        level2_channels = base_channels * 2
        level3_channels = base_channels * 4

        self.input_projection = nn.Conv2d(in_channels, level1_channels, kernel_size=3, padding=1)

        # Downsample path: 28x28 -> 14x14 -> 7x7.
        self.enc1 = DiffusionResBlock(level1_channels, level1_channels, time_dim)
        self.downsample1 = nn.Conv2d(
            level1_channels,
            level2_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.enc2 = DiffusionResBlock(level2_channels, level2_channels, time_dim)
        self.downsample2 = nn.Conv2d(
            level2_channels,
            level3_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.enc3 = DiffusionResBlock(level3_channels, level3_channels, time_dim)

        # Bottleneck at 7x7.
        self.bottleneck1 = DiffusionResBlock(level3_channels, level3_channels, time_dim)
        self.bottleneck2 = DiffusionResBlock(level3_channels, level3_channels, time_dim)

        # Upsample path: 7x7 -> 14x14 -> 28x28.
        self.upsample1 = nn.ConvTranspose2d(
            level3_channels,
            level2_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.dec2 = DiffusionResBlock(level2_channels + level2_channels, level2_channels, time_dim)
        self.upsample2 = nn.ConvTranspose2d(
            level2_channels,
            level1_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.dec1 = DiffusionResBlock(level1_channels + level1_channels, level1_channels, time_dim)
        self.output_norm = nn.GroupNorm(_resolve_group_count(level1_channels), level1_channels)
        self.output_activation = nn.SiLU()
        self.output_projection = nn.Conv2d(level1_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        time_embedding = self.time_embedding(timesteps)
        time_embedding = self.time_mlp(time_embedding)

        x = self.input_projection(x)
        skip1 = self.enc1(x, time_embedding)

        x = self.downsample1(skip1)
        skip2 = self.enc2(x, time_embedding)

        x = self.downsample2(skip2)
        x = self.enc3(x, time_embedding)

        x = self.bottleneck1(x, time_embedding)
        x = self.bottleneck2(x, time_embedding)

        x = self.upsample1(x)
        x = torch.cat([x, skip2], dim=1)
        x = self.dec2(x, time_embedding)

        x = self.upsample2(x)
        x = torch.cat([x, skip1], dim=1)
        x = self.dec1(x, time_embedding)

        x = self.output_norm(x)
        x = self.output_activation(x)
        return self.output_projection(x)
