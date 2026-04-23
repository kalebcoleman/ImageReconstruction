from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from diffusion.reporting import save_manifest_bundle, save_yaml


def _torch_version_like(value: str) -> object:
    version_type = type(torch.__version__)
    try:
        return version_type(value)
    except Exception:  # pragma: no cover - defensive fallback if constructor changes.
        return value


def test_save_yaml_handles_torch_version_like_and_nested_types(tmp_path: Path) -> None:
    payload = {
        "torch_version": _torch_version_like("2.5.1+cu121"),
        "path_value": Path("/tmp/example"),
        "device": torch.device("cuda:0"),
        "numpy_scalar": np.int64(7),
        "numpy_array": np.asarray([[1, 2], [3, 4]], dtype=np.int64),
        "tuple_value": (1, 2, 3),
        "set_value": {"b", "a"},
        123: {
            "nested_path": Path("/tmp/nested"),
            "nested_device": torch.device("cpu"),
            "nested_float": np.float32(1.25),
        },
    }

    yaml_path = tmp_path / "payload.yaml"
    save_yaml(yaml_path, payload)

    loaded = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert loaded["torch_version"] == "2.5.1+cu121"
    assert loaded["path_value"] == "/tmp/example"
    assert loaded["device"] == "cuda:0"
    assert loaded["numpy_scalar"] == 7
    assert loaded["numpy_array"] == [[1, 2], [3, 4]]
    assert loaded["tuple_value"] == [1, 2, 3]
    assert loaded["set_value"] == ["a", "b"]
    assert loaded["123"]["nested_path"] == "/tmp/nested"
    assert loaded["123"]["nested_device"] == "cpu"
    assert loaded["123"]["nested_float"] == 1.25


def test_save_manifest_bundle_uses_same_yaml_safe_normalization(tmp_path: Path) -> None:
    payload = {
        "torch_version": _torch_version_like("2.5.1+cu121"),
        "metadata": {
            "output_dir": Path("/tmp/output"),
            "device": torch.device("cpu"),
            "shapes": {(64, 64), (32, 32)},
        },
    }

    paths = save_manifest_bundle(
        tmp_path,
        basename="manifest",
        title="Manifest",
        payload=payload,
    )

    json_payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))
    yaml_payload = yaml.safe_load(Path(paths["yaml"]).read_text(encoding="utf-8"))

    assert json_payload["torch_version"] == "2.5.1+cu121"
    assert yaml_payload["torch_version"] == "2.5.1+cu121"
    assert json_payload["metadata"]["output_dir"] == "/tmp/output"
    assert yaml_payload["metadata"]["output_dir"] == "/tmp/output"
    assert json_payload["metadata"]["device"] == "cpu"
    assert yaml_payload["metadata"]["device"] == "cpu"
