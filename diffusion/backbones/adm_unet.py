from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resolve_group_count(num_channels: int, preferred_groups: int = 32) -> int:
    """Choose a GroupNorm group count that always divides the channel width."""

    for group_count in range(min(preferred_groups, num_channels), 0, -1):
        if num_channels % group_count == 0:
            return group_count
    return 1


def _zero_module(module: nn.Module) -> nn.Module:
    """Initialize a module to zero so residual paths start near identity."""

    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


class SinusoidalTimeEmbedding(nn.Module):
    """Encode discrete timesteps into a continuous conditioning vector."""

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
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        if self.embedding_dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
        return embedding


class ADMResBlock(nn.Module):
    """Residual block with timestep/class conditioning via scale-shift GroupNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_dim: int,
        *,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True,
    ) -> None:
        super().__init__()
        self.use_scale_shift_norm = use_scale_shift_norm
        self.in_layers = nn.Sequential(
            nn.GroupNorm(_resolve_group_count(in_channels), in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        projection_width = out_channels * (2 if use_scale_shift_norm else 1)
        self.embedding_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_dim, projection_width),
        )
        self.out_norm = nn.GroupNorm(_resolve_group_count(out_channels), out_channels)
        self.out_activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.out_conv = _zero_module(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        hidden = self.in_layers(x)
        embedding_out = self.embedding_layers(embedding).type_as(hidden).unsqueeze(-1).unsqueeze(-1)

        if self.use_scale_shift_norm:
            scale, shift = embedding_out.chunk(2, dim=1)
            hidden = self.out_norm(hidden) * (1.0 + scale) + shift
        else:
            hidden = self.out_norm(hidden + embedding_out)

        hidden = self.out_activation(hidden)
        hidden = self.dropout(hidden)
        hidden = self.out_conv(hidden)
        return hidden + residual


class SelfAttention2d(nn.Module):
    """Low-resolution self-attention block used in ADM-style U-Nets."""

    def __init__(self, channels: int, *, preferred_heads: int = 4) -> None:
        super().__init__()
        self.channels = channels
        self.num_heads = _resolve_group_count(channels, preferred_heads)
        self.norm = nn.GroupNorm(_resolve_group_count(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.projection = _zero_module(nn.Conv2d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        hidden = self.norm(x)
        qkv = self.qkv(hidden)
        query, key, value = qkv.chunk(3, dim=1)
        head_dim = channels // self.num_heads
        tokens = height * width

        query = query.reshape(batch_size, self.num_heads, head_dim, tokens).permute(0, 1, 3, 2)
        key = key.reshape(batch_size, self.num_heads, head_dim, tokens)
        value = value.reshape(batch_size, self.num_heads, head_dim, tokens).permute(0, 1, 3, 2)

        scale = head_dim**-0.5
        weights = torch.matmul(query.float(), key.float()) * scale
        weights = torch.softmax(weights, dim=-1).to(query.dtype)
        attended = torch.matmul(weights, value.float()).to(query.dtype)
        attended = attended.permute(0, 1, 3, 2).reshape(batch_size, channels, height, width)
        return x + self.projection(attended)


class Downsample2d(nn.Module):
    """Strided convolution downsampling used between resolution stages."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class Upsample2d(nn.Module):
    """Nearest-neighbor upsampling followed by a 3x3 projection."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.projection(x)


class DownStage(nn.Module):
    """A resolution stage on the encoder side of the ADM U-Net."""

    def __init__(self, blocks: list[nn.Module], downsample: nn.Module | None) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.downsample = downsample

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for block in self.blocks:
            if isinstance(block, ADMResBlock):
                x = block(x, embedding)
            else:
                x = block(x)

        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, skip


class UpStage(nn.Module):
    """A resolution stage on the decoder side of the ADM U-Net."""

    def __init__(self, blocks: list[nn.Module], upsample: nn.Module | None) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.upsample = upsample

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([x, skip], dim=1)
        for block in self.blocks:
            if isinstance(block, ADMResBlock):
                x = block(x, embedding)
            else:
                x = block(x)

        if self.upsample is not None:
            x = self.upsample(x)
        return x


def default_channel_mults(image_size: int) -> tuple[int, ...]:
    """Choose a conservative channel schedule for a given image size."""

    preset_map: dict[int, tuple[int, ...]] = {
        28: (1, 2, 4),
        32: (1, 2, 2),
        64: (1, 2, 3, 4),
        128: (1, 1, 2, 3, 4),
        256: (1, 1, 2, 2, 4, 4),
    }
    if image_size in preset_map:
        return preset_map[image_size]
    raise ValueError(
        "No default channel multiplier preset is defined for image_size="
        f"{image_size}. Explicitly add a preset before using this size."
    )


def default_attention_resolutions(image_size: int, dataset_name: str) -> tuple[int, ...]:
    """Return low-resolution attention placements for the selected run."""

    resolutions = []
    if image_size >= 128 or (dataset_name == "imagenet" and image_size >= 64):
        resolutions.append(32)
    if image_size >= 64:
        resolutions.extend([16, 8])
    elif image_size >= 32:
        resolutions.append(8)
    return tuple(resolutions)


class ADMUNet(nn.Module):
    """ADM-style pixel-space U-Net for scalable image diffusion experiments."""

    def __init__(
        self,
        *,
        in_channels: int,
        image_size: int,
        base_channels: int,
        time_dim: int,
        num_res_blocks: int,
        channel_mult: tuple[int, ...],
        attention_resolutions: tuple[int, ...],
        num_classes: int | None,
        class_dropout_prob: float = 0.1,
        use_scale_shift_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size < 8:
            raise ValueError("ADMUNet image_size must be at least 8.")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be at least 1.")
        if base_channels < 1 or time_dim < 1:
            raise ValueError("base_channels and time_dim must both be positive.")
        downsample_factor = 2 ** (len(channel_mult) - 1)
        if image_size % downsample_factor != 0:
            raise ValueError(
                "image_size must be divisible by the total downsample factor. "
                f"Got image_size={image_size} and factor={downsample_factor}."
            )

        self.in_channels = in_channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.null_class_label = num_classes if num_classes is not None else None

        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )
        self.embedding_dim = time_dim * 4
        self.class_embedding = (
            nn.Embedding(num_classes + 1, self.embedding_dim)
            if num_classes is not None
            else None
        )

        stage_channels = [base_channels * multiplier for multiplier in channel_mult]
        self.input_projection = nn.Conv2d(in_channels, stage_channels[0], kernel_size=3, padding=1)

        self.down_stages = nn.ModuleList()
        current_channels = stage_channels[0]
        current_resolution = image_size
        for stage_index, stage_width in enumerate(stage_channels):
            blocks: list[nn.Module] = []
            for block_index in range(num_res_blocks):
                block_in_channels = current_channels if block_index > 0 else current_channels
                blocks.append(
                    ADMResBlock(
                        block_in_channels,
                        stage_width,
                        self.embedding_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                current_channels = stage_width
                if current_resolution in attention_resolutions:
                    blocks.append(SelfAttention2d(current_channels))

            next_width = None if stage_index == len(stage_channels) - 1 else stage_channels[stage_index + 1]
            downsample = (
                None
                if next_width is None
                else Downsample2d(current_channels, next_width)
            )
            self.down_stages.append(DownStage(blocks, downsample))
            if downsample is not None:
                current_channels = next_width
                current_resolution //= 2

        self.mid_block1 = ADMResBlock(
            current_channels,
            current_channels,
            self.embedding_dim,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
        )
        self.mid_attention = (
            SelfAttention2d(current_channels)
            if current_resolution in attention_resolutions
            else nn.Identity()
        )
        self.mid_block2 = ADMResBlock(
            current_channels,
            current_channels,
            self.embedding_dim,
            dropout=dropout,
            use_scale_shift_norm=use_scale_shift_norm,
        )

        self.up_stages = nn.ModuleList()
        for stage_index in reversed(range(len(stage_channels))):
            skip_channels = stage_channels[stage_index]
            blocks: list[nn.Module] = [
                ADMResBlock(
                    current_channels + skip_channels,
                    skip_channels,
                    self.embedding_dim,
                    dropout=dropout,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            ]
            current_channels = skip_channels
            if current_resolution in attention_resolutions:
                blocks.append(SelfAttention2d(current_channels))
            for _ in range(num_res_blocks - 1):
                blocks.append(
                    ADMResBlock(
                        current_channels,
                        current_channels,
                        self.embedding_dim,
                        dropout=dropout,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                )
                if current_resolution in attention_resolutions:
                    blocks.append(SelfAttention2d(current_channels))

            upsample = None
            if stage_index > 0:
                next_channels = stage_channels[stage_index - 1]
                upsample = Upsample2d(current_channels, next_channels)
                current_channels = next_channels
                current_resolution *= 2
            self.up_stages.append(UpStage(blocks, upsample))

        self.output_layers = nn.Sequential(
            nn.GroupNorm(_resolve_group_count(stage_channels[0]), stage_channels[0]),
            nn.SiLU(),
            _zero_module(nn.Conv2d(stage_channels[0], in_channels, kernel_size=3, padding=1)),
        )

    def _build_conditioning_embedding(
        self,
        timesteps: torch.Tensor,
        labels: torch.Tensor | None,
        *,
        force_uncond: bool,
    ) -> torch.Tensor:
        embedding = self.time_mlp(self.time_embedding(timesteps))

        if self.class_embedding is None:
            return embedding

        class_indices = torch.full(
            (timesteps.shape[0],),
            fill_value=self.null_class_label,
            device=timesteps.device,
            dtype=torch.long,
        )
        if labels is not None:
            labels = labels.to(device=timesteps.device, dtype=torch.long)
            if labels.shape[0] != timesteps.shape[0]:
                raise ValueError(
                    "Label batch size must match the image batch size. "
                    f"Got labels={labels.shape[0]} and batch={timesteps.shape[0]}."
                )
            class_indices = labels.clone()

        if force_uncond:
            class_indices.fill_(self.null_class_label)
        elif self.training and self.class_dropout_prob > 0.0:
            drop_mask = torch.rand(class_indices.shape, device=class_indices.device) < self.class_dropout_prob
            class_indices = torch.where(
                drop_mask,
                torch.full_like(class_indices, self.null_class_label),
                class_indices,
            )

        return embedding + self.class_embedding(class_indices)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor | None = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D image tensor, got shape {tuple(x.shape)}.")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels, got {x.shape[1]}."
            )
        if x.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                "Input image size does not match the configured model image_size. "
                f"Expected {(self.image_size, self.image_size)}, got {tuple(x.shape[-2:])}."
            )

        embedding = self._build_conditioning_embedding(
            timesteps,
            labels,
            force_uncond=force_uncond,
        )

        x = self.input_projection(x)
        skip_connections: list[torch.Tensor] = []
        for stage in self.down_stages:
            x, skip = stage(x, embedding)
            skip_connections.append(skip)

        x = self.mid_block1(x, embedding)
        x = self.mid_attention(x)
        x = self.mid_block2(x, embedding)

        for stage in self.up_stages:
            skip = skip_connections.pop()
            x = stage(x, skip, embedding)

        return self.output_layers(x)
