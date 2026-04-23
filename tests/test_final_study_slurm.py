from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

from diffusion.parity_study import FINAL_STUDY_DATASETS
from diffusion.recipes import load_recipe


REPO_ROOT = Path(__file__).resolve().parents[1]
SLURM_DIR = REPO_ROOT / "slurm" / "final_study"


def _run_dry(
    script_name: str,
    tmp_path: Path,
    task_id: int,
    *,
    copied_to_spool: bool = False,
    **extra_env: str,
) -> str:
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
    script_path = SLURM_DIR / script_name
    if copied_to_spool:
        spool_dir = tmp_path / "var" / "spool" / "slurm" / "slurmd" / "job123"
        spool_dir.mkdir(parents=True, exist_ok=True)
        script_path = spool_dir / "slurm_script"
        shutil.copy2(SLURM_DIR / script_name, script_path)

    result = subprocess.run(
        ["bash", str(script_path)],
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


def test_final_study_slurm_scripts_source_common_from_repo_dir_when_copied_to_spool(tmp_path: Path) -> None:
    script_names = (
        "smoke_array.slurm",
        "train_mnist_array.slurm",
        "train_fashion_array.slurm",
        "train_cifar10_array.slurm",
        "eval_all_array.slurm",
        "finalize.slurm",
    )

    for script_name in script_names:
        output = _run_dry(script_name, tmp_path, 0, copied_to_spool=True)

        assert f"Repo: {REPO_ROOT}" in output
        assert "DRY_RUN=1; skipping conda activation." in output


def test_final_study_slurm_scripts_report_missing_common_path(tmp_path: Path) -> None:
    missing_repo = tmp_path / "missing" / "ImageReconstruction"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "SLURM_ARRAY_TASK_ID": "0",
            "REPO_DIR": str(missing_repo),
        }
    )

    result = subprocess.run(
        ["bash", str(SLURM_DIR / "train_mnist_array.slurm")],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    expected_common = missing_repo / "slurm" / "final_study" / "common.sh"
    assert result.returncode == 1
    assert f"Unable to find final study common.sh at: {expected_common}" in result.stderr
    assert "Set REPO_DIR to the ImageReconstruction checkout before submitting with sbatch." in result.stderr


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
