from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

from diffusion.parity_study import FINAL_STUDY_DATASETS
from diffusion.recipes import load_recipe


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "slurm" / "final_study"


def _run_dry(script_name: str, tmp_path: Path, task_id: int, **extra_env: str) -> str:
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "SLURM_ARRAY_TASK_ID": str(task_id),
            "REPO_DIR": str(REPO_ROOT),
            "STUDY_DIR": str(tmp_path / "study"),
            "DATA_DIR": str(tmp_path / "data"),
            "CONDA_ENV": "diffusion",
            "PYTHON_BIN": "python",
        }
    )
    env.update(extra_env)
    result = subprocess.run(
        ["bash", str(SLURM_DIR / script_name)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout + result.stderr


def _assert_command_has(output: str, *parts: str) -> None:
    for part in parts:
        pattern = r"(?<!\S)" + re.escape(part) + r"(?!\S)"
        assert re.search(pattern, output), output


def test_final_epoch_configs_match_recommended_full_study_epochs() -> None:
    assert load_recipe(REPO_ROOT / "configs/diffusion/mnist.yaml").values["epochs"] == 50
    assert load_recipe(REPO_ROOT / "configs/diffusion/fashion.yaml").values["epochs"] == 75
    assert load_recipe(REPO_ROOT / "configs/diffusion/cifar10.yaml").values["epochs"] == 150


def test_default_final_matrix_excludes_imagenet() -> None:
    assert FINAL_STUDY_DATASETS == ("mnist", "fashion", "cifar10")
    assert "imagenet" not in FINAL_STUDY_DATASETS


def test_smoke_array_maps_one_task_per_final_dataset(tmp_path: Path) -> None:
    outputs = [_run_dry("smoke_array.slurm", tmp_path, task_id) for task_id in range(3)]

    assert "Dataset: mnist" in outputs[0]
    assert "Dataset: fashion" in outputs[1]
    assert "Dataset: cifar10" in outputs[2]
    for output, dataset in zip(outputs, FINAL_STUDY_DATASETS, strict=True):
        _assert_command_has(output, "--smoke", "--phase", "train", "--datasets", dataset)
        assert "--allow-model-download" not in output


def test_train_arrays_map_task_id_to_seed_and_phase(tmp_path: Path) -> None:
    output = _run_dry("train_fashion_array.slurm", tmp_path, 1)

    assert "Dataset: fashion" in output
    assert "Seed: 2" in output
    _assert_command_has(output, "--phase", "train", "--datasets", "fashion", "--seeds", "2", "--skip-existing")


def test_eval_array_default_matrix_has_nine_tasks_and_maps_last_task(tmp_path: Path) -> None:
    script_text = (SLURM_DIR / "eval_all_array.slurm").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-8" in script_text

    output = _run_dry("eval_all_array.slurm", tmp_path, 8)

    assert "Dataset: cifar10" in output
    assert "Seed: 3" in output
    _assert_command_has(output, "--phase", "eval", "--datasets", "cifar10", "--seeds", "3", "--skip-existing")


def test_eval_array_passes_model_download_only_when_requested(tmp_path: Path) -> None:
    normal_output = _run_dry("eval_all_array.slurm", tmp_path, 0)
    download_output = _run_dry("eval_all_array.slurm", tmp_path, 0, ALLOW_MODEL_DOWNLOAD="1")

    assert "--allow-model-download" not in normal_output
    _assert_command_has(download_output, "--allow-model-download")
