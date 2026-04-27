#!/usr/bin/env python3
"""Collect report-ready image assets into stable dataset folders."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

try:
    from PIL import Image, UnidentifiedImageError
except ImportError:  # pragma: no cover - exercised only when Pillow is missing.
    Image = None
    UnidentifiedImageError = OSError


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_SOURCE_ROOTS = (
    "outputs",
    "runs",
    "results",
    "experiments",
    "logs",
    "deliverables",
)
DATASET_DIRS = {
    "mnist": "mnist",
    "fashion_mnist": "fashion_mnist",
    "cifar10": "cifar10",
}


@dataclass(frozen=True)
class ImageCandidate:
    path: Path
    dataset: str
    source_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy generated image grids into results/{mnist,fashion_mnist,cifar10}."
    )
    parser.add_argument(
        "--source-root",
        action="append",
        dest="source_roots",
        help="Source folder to search. May be passed multiple times.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Destination folder for organized report assets.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the results folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned copies without writing files.",
    )
    parser.add_argument(
        "--cifar-scale",
        type=int,
        default=4,
        help="Integer nearest-neighbor scale for CIFAR presentation copy.",
    )
    return parser.parse_args()


def normalize_key(path: Path) -> str:
    return path.as_posix().lower().replace("-", "_")


def infer_dataset(path: Path) -> str | None:
    key = normalize_key(path)
    if any(token in key for token in ("fashion_mnist", "fashionmnist", "fashion")):
        return "fashion_mnist"
    if any(token in key for token in ("cifar10", "cifar_10", "cifar")):
        return "cifar10"
    if "mnist" in key:
        return "mnist"
    return None


def is_relevant_image(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return False
    key = normalize_key(path)
    return any(
        token in key
        for token in (
            "generated",
            "generation",
            "samples",
            "reconstruction",
            "recon",
            "latent",
            "loss",
            "comparison",
            "snapshot",
            "contact",
        )
    )


def discover_images(source_roots: list[Path], results_dir: Path) -> list[ImageCandidate]:
    candidates: list[ImageCandidate] = []
    resolved_results = results_dir.resolve()
    for root in source_roots:
        if not root.exists():
            print(f"missing source folder: {root}")
            continue
        if not root.is_dir():
            print(f"skipping non-directory source: {root}")
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or not is_relevant_image(path):
                continue
            try:
                if path.resolve().is_relative_to(resolved_results):
                    continue
            except OSError:
                pass
            dataset = infer_dataset(path)
            if dataset:
                candidates.append(ImageCandidate(path=path, dataset=dataset, source_root=root))
    return candidates


def safe_stem(value: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in value.lower())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "asset"


def duplicate_basenames(candidates: list[ImageCandidate]) -> set[tuple[str, str]]:
    counts: dict[tuple[str, str], int] = {}
    for candidate in candidates:
        key = (candidate.dataset, candidate.path.name)
        counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count > 1}


def target_name(candidate: ImageCandidate, duplicates: set[tuple[str, str]]) -> str:
    if (candidate.dataset, candidate.path.name) not in duplicates:
        return candidate.path.name
    try:
        relative = candidate.path.relative_to(candidate.source_root)
    except ValueError:
        relative = candidate.path
    prefix = safe_stem("__".join(relative.with_suffix("").parts[:-1]))
    return f"{prefix}__{candidate.path.name}"


def copy_file(source: Path, target: Path, *, overwrite: bool, dry_run: bool) -> bool:
    if target.exists() and not overwrite:
        print(f"skip existing: {target}")
        return False
    print(f"{'would copy' if dry_run else 'copy'}: {source} -> {target}")
    if dry_run:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def nearest_resample_filter() -> int:
    return getattr(getattr(Image, "Resampling", Image), "NEAREST")


def save_scaled_nearest(source: Path, target: Path, *, scale: int, overwrite: bool, dry_run: bool) -> bool:
    if scale < 1:
        raise ValueError("--cifar-scale must be at least 1")
    if target.exists() and not overwrite:
        print(f"skip existing: {target}")
        return False
    if Image is None:
        print(f"cannot scale CIFAR image because Pillow is not installed: {source}")
        return False
    print(f"{'would scale' if dry_run else 'scale'}: {source} -> {target} ({scale}x nearest)")
    if dry_run:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(source) as image:
            scaled = image.resize(
                (image.width * scale, image.height * scale),
                resample=nearest_resample_filter(),
            )
            scaled.save(target)
    except UnidentifiedImageError:
        print(f"cannot scale unreadable image, copying instead: {source}")
        shutil.copy2(source, target)
    return True


def cifar_generation_priority(candidate: ImageCandidate) -> tuple[int, str]:
    key = normalize_key(candidate.path)
    name = candidate.path.name.lower()
    penalties = 0
    if "nearest" in key or "_2x" in key or "_4x" in key or "presentation_crisp" in key:
        penalties += 100
    if "smoke" in key:
        penalties += 25
    if "image_reconstruction_final_study" in key or "final_study" in key:
        penalties -= 10
    if "/samples/" in key:
        penalties -= 2
    priorities = (
        ("generated_samples_native_grid", 0),
        ("generated_samples", 1),
        ("generation", 2),
        ("generated", 3),
        ("contact_100_native_1x_nopad", 4),
        ("contact_100_native_1x", 5),
        ("contact", 6),
        ("samples", 7),
    )
    for token, priority in priorities:
        if token in key or token in name:
            return (priority + penalties, candidate.path.as_posix())
    return (50 + penalties, candidate.path.as_posix())


def select_cifar_generation(candidates: list[ImageCandidate]) -> Path | None:
    cifar_candidates = [
        candidate
        for candidate in candidates
        if candidate.dataset == "cifar10"
        and any(token in normalize_key(candidate.path) for token in ("generated", "generation", "samples", "contact"))
    ]
    if not cifar_candidates:
        return None
    return min(cifar_candidates, key=cifar_generation_priority).path


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    results_dir = (repo_root / args.results_dir).resolve()
    source_roots = [repo_root / root for root in (args.source_roots or DEFAULT_SOURCE_ROOTS)]

    for dataset_dir in DATASET_DIRS.values():
        target_dir = results_dir / dataset_dir
        if args.dry_run:
            print(f"would ensure folder: {target_dir}")
        else:
            target_dir.mkdir(parents=True, exist_ok=True)

    candidates = discover_images(source_roots, results_dir)
    if not candidates:
        print("no matching report images found")
        return 0

    duplicates = duplicate_basenames(candidates)
    copied = 0
    for candidate in candidates:
        target_dir = results_dir / DATASET_DIRS[candidate.dataset]
        target_path = target_dir / target_name(candidate, duplicates)
        copied += int(copy_file(candidate.path, target_path, overwrite=args.overwrite, dry_run=args.dry_run))

    cifar_source = select_cifar_generation(candidates)
    if cifar_source is None:
        print("no CIFAR generation grid found for actual-size/scaled copies")
    else:
        cifar_dir = results_dir / "cifar10"
        copied += int(
            copy_file(
                cifar_source,
                cifar_dir / "generations_actual_size.png",
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        )
        copied += int(
            save_scaled_nearest(
                cifar_source,
                cifar_dir / "generations_scaled_nearest.png",
                scale=args.cifar_scale,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )
        )

    action = "would prepare" if args.dry_run else "prepared"
    print(f"{action} {copied} asset operation(s) from {len(candidates)} matching image(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
