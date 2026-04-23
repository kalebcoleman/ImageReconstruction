from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


META_KEYS = {"inherits", "protocol"}
PATH_KEYS = {"data_dir", "output_dir", "config_path"}
TUPLE_LIST_KEYS = {
    "attention_resolutions",
    "eval_cfg_comparison_scales",
    "protocol_locked_fields",
    "protocol_allowed_overrides",
}
DEFAULT_ALLOWED_OVERRIDES = {
    "dataset",
    "batch_size",
    "num_workers",
    "epochs",
    "data_dir",
    "output_dir",
    "run_name",
    "eval_batch_size",
    "dataset_variant",
    "download",
}


@dataclass(frozen=True)
class LoadedRecipe:
    """Resolved config recipe plus protocol metadata."""

    path: Path
    values: dict[str, Any]
    sources: tuple[Path, ...]
    protocol: dict[str, Any]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_value(key: str, value: Any) -> Any:
    if key in PATH_KEYS and value is not None:
        return Path(value)
    if key in TUPLE_LIST_KEYS and value is not None:
        return tuple(value)
    return value


def _coerce_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {key: _coerce_value(key, value) for key, value in mapping.items()}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"Recipe at {path} must contain a YAML mapping.")
    return payload


def load_recipe(config_path: Path) -> LoadedRecipe:
    """Load a recipe YAML file, resolving inheritance and allowed overrides."""

    resolved_path = Path(config_path).expanduser().resolve()
    payload = _load_yaml_mapping(resolved_path)

    inherits = payload.get("inherits", [])
    if isinstance(inherits, str):
        inherits = [inherits]
    protocol = payload.get("protocol", {}) or {}
    if not isinstance(protocol, dict):
        raise ValueError(f"Recipe protocol section must be a mapping in {resolved_path}.")

    merged_values: dict[str, Any] = {}
    merged_protocol: dict[str, Any] = {}
    sources: list[Path] = []

    for parent_ref in inherits:
        parent_path = (resolved_path.parent / parent_ref).resolve()
        parent_recipe = load_recipe(parent_path)
        merged_values = _deep_merge(merged_values, parent_recipe.values)
        merged_protocol = _deep_merge(merged_protocol, parent_recipe.protocol)
        sources.extend(parent_recipe.sources)

    local_values = {key: value for key, value in payload.items() if key not in META_KEYS}
    local_values = _coerce_mapping(local_values)
    merged_protocol = _coerce_mapping(_deep_merge(merged_protocol, protocol))
    merged_values = _deep_merge(merged_values, local_values)

    allowed_overrides = set(
        merged_protocol.get("allowed_overrides")
        or DEFAULT_ALLOWED_OVERRIDES
    )
    if inherits:
        override_keys = {
            key
            for key, value in local_values.items()
            if merged_values.get(key) == value and key not in {"config_name", "config_path", "protocol_name"}
        }
        disallowed = override_keys - allowed_overrides
        if disallowed:
            raise ValueError(
                f"Recipe {resolved_path.name} overrides locked settings: {sorted(disallowed)}. "
                f"Allowed overrides are {sorted(allowed_overrides)}."
            )

    locked_fields = tuple(
        merged_protocol.get("locked_fields")
        or (
            "diffusion_backbone",
            "image_size",
            "diffusion_channels",
            "timesteps",
            "base_channels",
            "time_dim",
            "schedule",
            "ema_decay",
            "num_res_blocks",
            "prediction_type",
            "attention_resolutions",
            "class_dropout_prob",
            "sampler",
            "sampling_steps",
            "ddim_eta",
            "diffusion_preprocessing",
        )
    )
    merged_values.setdefault("config_name", resolved_path.stem)
    merged_values.setdefault("config_path", resolved_path)
    merged_values.setdefault("protocol_name", merged_protocol.get("name"))
    merged_values.setdefault("protocol_locked_fields", locked_fields)
    merged_values.setdefault("protocol_allowed_overrides", tuple(sorted(allowed_overrides)))
    merged_values.setdefault("dataset_variant", merged_protocol.get("dataset_variant"))
    merged_values.setdefault("diffusion_preprocessing", merged_protocol.get("diffusion_preprocessing", "default"))
    merged_values.setdefault("eval_batch_size", merged_protocol.get("eval_batch_size"))
    merged_values.setdefault("eval_num_generated_samples", merged_protocol.get("eval_num_generated_samples"))
    merged_values.setdefault("eval_cfg_comparison_scales", _coerce_value("eval_cfg_comparison_scales", merged_protocol.get("eval_cfg_comparison_scales")))

    sources.append(resolved_path)
    deduped_sources = tuple(dict.fromkeys(sources))
    return LoadedRecipe(
        path=resolved_path,
        values=merged_values,
        sources=deduped_sources,
        protocol=merged_protocol,
    )


def collect_explicit_cli_dests(
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> set[str]:
    """Return parser destination names explicitly provided on the CLI."""

    option_to_dest: dict[str, str] = {}
    for action in parser._actions:
        for option in action.option_strings:
            option_to_dest[option] = action.dest

    explicit_dests: set[str] = set()
    for token in argv:
        if not token.startswith("-"):
            continue
        option = token.split("=", 1)[0]
        destination = option_to_dest.get(option)
        if destination is not None:
            explicit_dests.add(destination)
    return explicit_dests


def apply_recipe_to_namespace(
    args: argparse.Namespace,
    *,
    parser: argparse.ArgumentParser,
    argv: list[str],
) -> argparse.Namespace:
    """Overlay a recipe onto parsed CLI args, preserving explicit CLI overrides."""

    config_path = getattr(args, "config", None)
    if config_path is None:
        return args

    recipe = load_recipe(Path(config_path))
    explicit_dests = collect_explicit_cli_dests(parser, argv)
    for key, value in recipe.values.items():
        if key in explicit_dests:
            continue
        setattr(args, key, value)

    setattr(args, "config", recipe.path)
    setattr(args, "config_sources", recipe.sources)
    setattr(args, "protocol_metadata", recipe.protocol)
    return args
