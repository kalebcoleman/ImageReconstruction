from __future__ import annotations

import json
import numbers
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _yaml_ready(value: Any) -> Any:
    """Recursively coerce values into plain YAML-safe Python primitives."""

    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        return str(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, np.ndarray):
        return [_yaml_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _yaml_ready(value.item())
    if isinstance(value, dict):
        return {str(key): _yaml_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_ready(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_yaml_ready(item) for item in sorted(value, key=lambda item: str(item))]
    return str(value)


def save_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Persist a payload as YAML with stable key ordering."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_yaml_ready(payload), sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


def flatten_mapping(payload: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping for CSV or compact markdown summaries."""

    flat: dict[str, Any] = {}
    for key, value in payload.items():
        nested_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten_mapping(value, nested_key))
        else:
            flat[nested_key] = value
    return flat


def save_markdown_summary(path: Path, title: str, payload: dict[str, Any]) -> None:
    """Write a concise human-readable summary for a manifest payload."""

    flat = flatten_mapping(payload)
    lines = [f"# {title}", ""]
    for key, value in flat.items():
        lines.append(f"- `{key}`: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_manifest_bundle(
    output_dir: Path,
    *,
    basename: str,
    title: str,
    payload: dict[str, Any],
) -> dict[str, str]:
    """Save matching JSON, YAML, and Markdown views of a manifest."""

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{basename}.json"
    yaml_path = output_dir / f"{basename}.yaml"
    markdown_path = output_dir / f"{basename}.md"

    json_path.write_text(json.dumps(_yaml_ready(payload), indent=2, sort_keys=True), encoding="utf-8")
    save_yaml(yaml_path, payload)
    save_markdown_summary(markdown_path, title, payload)

    return {
        "json": str(json_path.resolve()),
        "yaml": str(yaml_path.resolve()),
        "markdown": str(markdown_path.resolve()),
    }
