#!/usr/bin/env python3
"""Collect existing report visuals into the GitHub Pages asset tree."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
SEARCH_DIRS = (
    "outputs",
    "output",
    "results",
    "runs",
    "experiments",
    "logs",
    "figures",
    "plots",
    "images",
    "docs/assets",
    "deliverables",
)
ASSET_DIRS = {
    "mnist": Path("docs/assets/mnist"),
    "fashion_mnist": Path("docs/assets/fashion_mnist"),
    "cifar10": Path("docs/assets/cifar10"),
    "combined": Path("docs/assets/combined"),
    "placeholders": Path("docs/assets/placeholders"),
}
PCA_PLACEHOLDERS = (
    "pca_mnist_reconstructions_placeholder.png",
    "pca_fashion_reconstructions_placeholder.png",
    "pca_metrics_placeholder.png",
    "pca_vs_ae_combined_grid_placeholder.png",
)
DATASET_KEYWORDS = {
    "mnist": ("mnist", "digit"),
    "fashion_mnist": ("fashion", "fmnist"),
    "cifar10": ("cifar", "cifar10"),
}
MODEL_KEYWORDS = {
    "ae": ("autoencoder", "ae", "reconstruction", "recon"),
    "dae": ("dae", "denoise", "denoising", "noisy", "noise"),
    "vae": ("vae", "variational", "latent", "interpolation", "interp"),
    "diffusion": ("diffusion", "ddpm", "sample", "generation", "generated", "reverse", "denoise_steps"),
}
LIKELY_PLOT_KEYWORDS = (
    "plot",
    "figure",
    "grid",
    "curve",
    "loss",
    "metric",
    "comparison",
    "ssim",
    "psnr",
    "mse",
    "latent",
)
GENERIC_REPORT_STEMS = {
    "generated_samples",
    "loss_curve",
    "reconstructions",
    "diffusion_snapshots",
}


@dataclass(frozen=True)
class Candidate:
    source: Path
    source_root: Path
    dataset: str | None
    tags: tuple[str, ...]
    likely_plot: bool


@dataclass
class Summary:
    found: list[Path] = field(default_factory=list)
    copied: list[tuple[Path, Path]] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    placeholders: list[Path] = field(default_factory=list)
    grids: list[Path] = field(default_factory=list)
    docs_images: list[tuple[str, Path, bool]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    uncategorized: list[Path] = field(default_factory=list)
    searched_missing_dirs: list[Path] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect report images for the GitHub Pages site.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files already copied into docs/assets.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing files.")
    parser.add_argument(
        "--source-dir",
        action="append",
        dest="source_dirs",
        help="Extra source directory to search. May be passed multiple times.",
    )
    return parser.parse_args()


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower())


def token_set(path: Path) -> set[str]:
    return set(normalize(path.as_posix()).split("_"))


def keyword_match(path: Path, keywords: tuple[str, ...]) -> bool:
    normalized = normalize(path.as_posix())
    tokens = token_set(path)
    for keyword in keywords:
        key = normalize(keyword)
        if key in tokens or key in normalized:
            return True
    return False


def classify_dataset(path: Path) -> str | None:
    normalized = normalize(path.as_posix())
    if "fashion" in normalized or "fmnist" in normalized:
        return "fashion_mnist"
    if "cifar10" in normalized or "cifar_10" in normalized or "cifar" in normalized:
        return "cifar10"
    if "mnist" in normalized or "digit" in normalized:
        return "mnist"
    return None


def classify_tags(path: Path) -> tuple[str, ...]:
    tags = []
    for tag, keywords in MODEL_KEYWORDS.items():
        if keyword_match(path, keywords):
            tags.append(tag)
    return tuple(tags)


def is_likely_plot(path: Path) -> bool:
    return keyword_match(path, LIKELY_PLOT_KEYWORDS)


def discover_candidates(source_dirs: list[Path], docs_assets: Path, summary: Summary) -> list[Candidate]:
    candidates: list[Candidate] = []
    for source_root in source_dirs:
        if not source_root.exists():
            summary.searched_missing_dirs.append(source_root)
            continue
        if not source_root.is_dir():
            continue
        for source in sorted(source_root.rglob("*")):
            if not source.is_file() or source.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            dataset = classify_dataset(source)
            tags = classify_tags(source)
            likely_plot = is_likely_plot(source)
            if dataset or likely_plot:
                summary.found.append(source)
                candidates.append(
                    Candidate(
                        source=source,
                        source_root=source_root,
                        dataset=dataset,
                        tags=tags,
                        likely_plot=likely_plot,
                    )
                )
            else:
                summary.uncategorized.append(source)
    return candidates


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9.]+", "_", value).strip("_")
    return cleaned or "asset"


def destination_for(candidate: Candidate) -> Path:
    group = candidate.dataset or "combined"
    filename = candidate.source.name
    if not is_in_tree(candidate.source, Path("docs/assets")):
        stem = candidate.source.stem
        if stem in GENERIC_REPORT_STEMS or stem.startswith("study_"):
            try:
                relative = candidate.source.relative_to(candidate.source_root)
            except ValueError:
                relative = candidate.source
            prefix = safe_name("_".join(relative.with_suffix("").parts[:-1]))
            if prefix:
                filename = f"{prefix}__{filename}"
    return ASSET_DIRS[group] / filename


def mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def is_in_tree(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def select_newest_by_destination(candidates: list[Candidate]) -> list[Candidate]:
    newest: dict[Path, Candidate] = {}
    for candidate in candidates:
        target = destination_for(candidate)
        existing = newest.get(target)
        if existing is None or mtime(candidate.source) > mtime(existing.source):
            newest[target] = candidate
    return [newest[target] for target in sorted(newest)]


def copy_asset(source: Path, target: Path, *, overwrite: bool, dry_run: bool, summary: Summary) -> bool:
    try:
        if source.resolve() == target.resolve():
            summary.skipped.append(target)
            return False
    except OSError:
        pass
    if target.exists() and not overwrite:
        summary.skipped.append(target)
        return False
    if dry_run:
        summary.copied.append((source, target))
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    summary.copied.append((source, target))
    return True


def load_font(size: int = 24) -> ImageFont.ImageFont:
    for name in ("Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_wrapped_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: str, width: int) -> None:
    y = xy[1]
    for paragraph in text.split("\n"):
        for line in textwrap.wrap(paragraph, width=width):
            draw.text((xy[0], y), line, font=font, fill=fill)
            bbox = draw.textbbox((xy[0], y), line, font=font)
            y += bbox[3] - bbox[1] + 6
        y += 8


def create_placeholder(path: Path, text: str, *, overwrite: bool, dry_run: bool, summary: Summary, size: tuple[int, int] = (900, 540)) -> None:
    if path.exists() and not overwrite:
        summary.skipped.append(path)
        return
    if dry_run:
        summary.placeholders.append(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, "#f7f3eb")
    draw = ImageDraw.Draw(image)
    title_font = load_font(30)
    body_font = load_font(22)
    draw.rectangle((18, 18, size[0] - 18, size[1] - 18), outline="#b13f32", width=4)
    draw.text((48, 52), "PCA Results Placeholder", font=title_font, fill="#17130f")
    draw_wrapped_text(draw, (48, 120), text, body_font, "#5f554d", width=54)
    image.save(path)
    summary.placeholders.append(path)


def create_pca_placeholders(*, overwrite: bool, dry_run: bool, summary: Summary) -> None:
    for filename in PCA_PLACEHOLDERS:
        create_placeholder(
            ASSET_DIRS["placeholders"] / filename,
            "PCA results placeholder -- add teammate visual here.",
            overwrite=overwrite,
            dry_run=dry_run,
            summary=summary,
        )


def open_image_or_placeholder(path: Path | None, label: str, *, tile_size: tuple[int, int]) -> Image.Image:
    if path and path.exists():
        try:
            with Image.open(path) as image:
                return image.convert("RGB")
        except (OSError, UnidentifiedImageError):
            pass
    image = Image.new("RGB", tile_size, "#fffdfa")
    draw = ImageDraw.Draw(image)
    font = load_font(22)
    draw.rectangle((0, 0, tile_size[0] - 1, tile_size[1] - 1), outline="#d8ccbd", width=3)
    draw_wrapped_text(draw, (24, 34), f"{label}\nMissing visual placeholder", font, "#6f665e", width=28)
    return image


def contain(image: Image.Image, size: tuple[int, int], *, resample: int = Image.Resampling.LANCZOS) -> Image.Image:
    result = Image.new("RGB", size, "#fffdfa")
    working = image.copy()
    working.thumbnail(size, resample=resample)
    x = (size[0] - working.width) // 2
    y = (size[1] - working.height) // 2
    result.paste(working, (x, y))
    return result


def make_grid(
    target: Path,
    tiles: list[tuple[str, Path | None]],
    *,
    overwrite: bool,
    dry_run: bool,
    summary: Summary,
    tile_size: tuple[int, int] = (420, 300),
) -> None:
    if target.exists() and not overwrite:
        summary.skipped.append(target)
        return
    if dry_run:
        summary.grids.append(target)
        return

    label_height = 48
    gap = 24
    columns = min(2, len(tiles))
    rows = (len(tiles) + columns - 1) // columns
    width = columns * tile_size[0] + (columns + 1) * gap
    height = rows * (tile_size[1] + label_height) + (rows + 1) * gap
    canvas = Image.new("RGB", (width, height), "#f7f3eb")
    draw = ImageDraw.Draw(canvas)
    label_font = load_font(22)

    for index, (label, path) in enumerate(tiles):
        row = index // columns
        column = index % columns
        x = gap + column * (tile_size[0] + gap)
        y = gap + row * (tile_size[1] + label_height + gap)
        draw.text((x, y), label, font=label_font, fill="#17130f")
        image = open_image_or_placeholder(path, label, tile_size=tile_size)
        rendered = contain(image, tile_size)
        canvas.paste(rendered, (x, y + label_height))

    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target)
    summary.grids.append(target)


def score_path(path: Path, keywords: tuple[str, ...]) -> tuple[int, str]:
    normalized = normalize(path.as_posix())
    score = 0
    for index, keyword in enumerate(keywords):
        if normalize(keyword) in normalized:
            score -= 20 - index
    if "smoke" in normalized:
        score += 30
    if "final_study" in normalized or "image_reconstruction_final_study" in normalized:
        score -= 10
    if "nearest" in normalized or "_4x" in normalized or "_2x" in normalized:
        score += 8
    return (score, path.as_posix())


def find_best(copied_targets: list[Path], dataset: str, keywords: tuple[str, ...]) -> Path | None:
    root = ASSET_DIRS[dataset]
    options = [path for path in copied_targets if root in path.parents and keyword_match(path, keywords)]
    if not options:
        return None
    return min(options, key=lambda path: score_path(path, keywords))


def find_best_combined(copied_targets: list[Path], keywords: tuple[str, ...]) -> Path | None:
    options = [path for path in copied_targets if keyword_match(path, keywords)]
    if not options:
        return None
    return min(options, key=lambda path: score_path(path, keywords))


def save_cifar_nearest_versions(copied_targets: list[Path], *, overwrite: bool, dry_run: bool, summary: Summary) -> tuple[Path | None, Path | None]:
    source = find_best(copied_targets, "cifar10", ("generated_samples", "generated", "sample", "contact"))
    actual = ASSET_DIRS["cifar10"] / "cifar_generations_actual_size.png"
    enlarged = ASSET_DIRS["cifar10"] / "cifar_generations_nearest_enlarged.png"
    if source is None:
        summary.missing.append("CIFAR diffusion generation image for actual-size and nearest-neighbor copies")
        return None, None

    copy_asset(source, actual, overwrite=overwrite, dry_run=dry_run, summary=summary)
    if enlarged.exists() and not overwrite:
        summary.skipped.append(enlarged)
        return actual, enlarged
    if dry_run:
        summary.copied.append((source, enlarged))
        return actual, enlarged
    try:
        with Image.open(source) as image:
            scale = 4
            resized = image.resize((image.width * scale, image.height * scale), resample=Image.Resampling.NEAREST)
            resized.save(enlarged)
    except (OSError, UnidentifiedImageError):
        shutil.copy2(source, enlarged)
    summary.copied.append((source, enlarged))
    return actual, enlarged


def create_combined_grids(copied_targets: list[Path], *, overwrite: bool, dry_run: bool, summary: Summary) -> None:
    cifar_actual, cifar_enlarged = save_cifar_nearest_versions(
        copied_targets,
        overwrite=overwrite,
        dry_run=dry_run,
        summary=summary,
    )

    mnist_ae = find_best(copied_targets, "mnist", ("ae", "recon", "reconstruction"))
    mnist_vae = find_best(copied_targets, "mnist", ("vae", "interpolation", "generated", "latent"))
    mnist_diffusion = find_best(copied_targets, "mnist", ("diffusion", "generated", "sample"))
    if mnist_ae is None:
        summary.missing.append("MNIST AE reconstruction visual")
    if mnist_vae is None:
        summary.missing.append("MNIST VAE/interpolation visual")
    if mnist_diffusion is None:
        summary.missing.append("MNIST diffusion generation visual")
    make_grid(
        ASSET_DIRS["combined"] / "mnist_ae_vae_diffusion_overview.png",
        [
            ("AE Reconstruction", mnist_ae),
            ("VAE / Latent Result", mnist_vae),
            ("Diffusion Generation", mnist_diffusion),
        ],
        overwrite=overwrite,
        dry_run=dry_run,
        summary=summary,
    )

    fashion_ae = find_best(copied_targets, "fashion_mnist", ("ae", "recon", "reconstruction"))
    fashion_vae = find_best(copied_targets, "fashion_mnist", ("vae", "interpolation", "generated", "latent"))
    fashion_diffusion = find_best(copied_targets, "fashion_mnist", ("diffusion", "generated", "sample"))
    if fashion_ae is None:
        summary.missing.append("Fashion-MNIST AE reconstruction visual")
    if fashion_vae is None:
        summary.missing.append("Fashion-MNIST VAE/interpolation visual")
    if fashion_diffusion is None:
        summary.missing.append("Fashion-MNIST diffusion generation visual")
    make_grid(
        ASSET_DIRS["combined"] / "fashion_ae_vae_diffusion_overview.png",
        [
            ("AE Reconstruction", fashion_ae),
            ("VAE / Latent Result", fashion_vae),
            ("Diffusion Generation", fashion_diffusion),
        ],
        overwrite=overwrite,
        dry_run=dry_run,
        summary=summary,
    )

    cifar_recon = find_best(copied_targets, "cifar10", ("recon", "reconstruction"))
    make_grid(
        ASSET_DIRS["combined"] / "cifar_diffusion_overview.png",
        [
            ("CIFAR Actual Size", cifar_actual),
            ("CIFAR Nearest Enlarged", cifar_enlarged),
            ("CIFAR Reconstruction", cifar_recon),
        ],
        overwrite=overwrite,
        dry_run=dry_run,
        summary=summary,
    )

    make_grid(
        ASSET_DIRS["combined"] / "extra_credit_diffusion_overview.png",
        [
            ("MNIST Diffusion", mnist_diffusion),
            ("Fashion-MNIST Diffusion", fashion_diffusion),
            ("CIFAR Actual Size", cifar_actual),
            ("CIFAR Nearest Enlarged", cifar_enlarged),
        ],
        overwrite=overwrite,
        dry_run=dry_run,
        summary=summary,
    )


DIFFUSION_GRID_SPECS = {
    "mnist": {
        "label": "MNIST",
        "asset_dir": "mnist",
        "sample_size": 28,
        "target_prefix": "mnist",
        "patterns": (
            "*contact_100_native_1x_nopad.png",
            "*contact_100_native_1x.png",
            "*generated_samples_native_grid.png",
        ),
    },
    "fashion_mnist": {
        "label": "Fashion-MNIST",
        "asset_dir": "fashion_mnist",
        "sample_size": 28,
        "target_prefix": "fashion",
        "patterns": (
            "*contact_100_native_1x_nopad.png",
            "*contact_100_native_1x.png",
            "*generated_samples_native_grid.png",
        ),
    },
    "cifar10": {
        "label": "CIFAR-10",
        "asset_dir": "cifar10",
        "sample_size": 32,
        "target_prefix": "cifar10",
        "patterns": (
            "*contact_100_native_1x_nopad.png",
            "*contact_100_native_1x.png",
            "*generated_samples_native_grid.png",
        ),
    },
}


def newest_existing(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: (mtime(path), path.as_posix()))


def find_native_diffusion_grid_source(dataset: str, copied_targets: list[Path]) -> Path | None:
    spec = DIFFUSION_GRID_SPECS[dataset]
    roots = [ASSET_DIRS[spec["asset_dir"]]]
    candidates: list[Path] = []
    for root in roots:
        for pattern in spec["patterns"]:
            candidates.extend(root.glob(pattern))
    candidates.extend(
        path
        for path in copied_targets
        if ASSET_DIRS[spec["asset_dir"]] in path.parents
        and any(path.match(pattern) for pattern in spec["patterns"])
    )
    return newest_existing(candidates)


def create_missing_image(
    path: Path,
    text: str,
    *,
    size: tuple[int, int],
    overwrite: bool,
    dry_run: bool,
    summary: Summary,
) -> None:
    if path.exists() and not overwrite:
        summary.skipped.append(path)
        return
    if dry_run:
        summary.grids.append(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, "#fffdfa")
    draw = ImageDraw.Draw(image)
    font = load_font(18 if size[0] <= 320 else 36)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline="#b13f32", width=max(2, size[0] // 160))
    draw_wrapped_text(draw, (max(8, size[0] // 18), max(8, size[1] // 6)), text, font, "#5f554d", width=34)
    image.save(path)
    summary.grids.append(path)


def create_diffusion_grid_versions(copied_targets: list[Path], *, overwrite: bool, dry_run: bool, summary: Summary) -> None:
    combined_dir = ASSET_DIRS["combined"]
    for dataset, spec in DIFFUSION_GRID_SPECS.items():
        source = find_native_diffusion_grid_source(dataset, copied_targets)
        one_x = combined_dir / f"{spec['target_prefix']}_diffusion_grid_1x.png"
        large = combined_dir / f"{spec['target_prefix']}_diffusion_grid_nearest_large.png"
        sample_size = int(spec["sample_size"])

        if source is None:
            summary.missing.append(
                f"{spec['label']} native diffusion sample grid ({sample_size}x{sample_size} samples)"
            )
            create_missing_image(
                one_x,
                f"{spec['label']} diffusion grid missing\nExpected native {sample_size}x{sample_size} samples.",
                size=(sample_size * 10, sample_size * 10),
                overwrite=overwrite,
                dry_run=dry_run,
                summary=summary,
            )
            create_missing_image(
                large,
                f"{spec['label']} diffusion grid missing\nRun collection after samples are available.",
                size=(sample_size * 40, sample_size * 40),
                overwrite=overwrite,
                dry_run=dry_run,
                summary=summary,
            )
            continue

        if one_x.exists() and not overwrite:
            summary.skipped.append(one_x)
        elif dry_run:
            summary.grids.append(one_x)
        else:
            one_x.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, one_x)
            summary.grids.append(one_x)

        if large.exists() and not overwrite:
            summary.skipped.append(large)
            continue
        if dry_run:
            summary.grids.append(large)
            continue

        with Image.open(source) as image:
            rgb = image.convert("RGB")
            resized = rgb.resize((rgb.width * 4, rgb.height * 4), resample=Image.Resampling.NEAREST)
            resized.save(large)
        summary.grids.append(large)


def extract_docs_image_sources(index_path: Path) -> list[str]:
    if not index_path.exists():
        return []
    text = index_path.read_text(encoding="utf-8")
    markdown = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    html = re.findall(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", text, flags=re.IGNORECASE)
    sources = []
    for source in markdown + html:
        clean = source.split("#", 1)[0].split("?", 1)[0].strip()
        if clean and not re.match(r"^[a-z]+://", clean):
            sources.append(clean)
    return sources


def record_docs_images(index_path: Path, summary: Summary) -> None:
    for source in extract_docs_image_sources(index_path):
        resolved = (index_path.parent / source).resolve()
        summary.docs_images.append((source, resolved, resolved.exists()))
        if not resolved.exists():
            summary.missing.append(f"docs/index.md image link is missing: {source}")


def warn_missing_expected_assets(summary: Summary) -> None:
    expected = (
        Path("docs/assets/combined/mnist_diffusion_grid_1x.png"),
        Path("docs/assets/combined/fashion_diffusion_grid_1x.png"),
        Path("docs/assets/combined/cifar10_diffusion_grid_1x.png"),
        Path("docs/assets/combined/mnist_diffusion_grid_nearest_large.png"),
        Path("docs/assets/combined/fashion_diffusion_grid_nearest_large.png"),
        Path("docs/assets/combined/cifar10_diffusion_grid_nearest_large.png"),
        Path("docs/assets/combined/vae_interpolation_better.png"),
    )
    for path in expected:
        if not path.exists():
            summary.missing.append(f"expected report asset missing: {path}")


def print_summary(summary: Summary) -> None:
    print("\nAsset collection summary")
    print(f"- images found: {len(summary.found)}")
    print(f"- images copied/planned: {len(summary.copied)}")
    print(f"- existing files skipped: {len(summary.skipped)}")
    print(f"- PCA placeholders created/planned: {len(summary.placeholders)}")
    print(f"- grids generated/planned: {len(summary.grids)}")
    print(f"- docs/index.md images referenced: {len(summary.docs_images)}")
    print(f"- missing source folders: {len(summary.searched_missing_dirs)}")
    print(f"- uncategorized images ignored: {len(summary.uncategorized)}")
    if summary.docs_images:
        print("- images used by docs/index.md:")
        for source, resolved, exists in summary.docs_images:
            status = "ok" if exists else "missing"
            print(f"  - {source} -> {resolved} [{status}]")
    if summary.copied:
        print("- images copied/planned:")
        for source, target in summary.copied:
            print(f"  - {source} -> {target}")
    if summary.missing:
        print("- missing expected visuals:")
        for item in dict.fromkeys(summary.missing):
            print(f"  - {item}")
    if summary.searched_missing_dirs:
        print("- source folders not found:")
        for path in summary.searched_missing_dirs:
            print(f"  - {path}")
    if summary.grids:
        print("- grids:")
        for path in summary.grids:
            print(f"  - {path}")


def main() -> int:
    args = parse_args()
    summary = Summary()

    for directory in ASSET_DIRS.values():
        if not args.dry_run:
            directory.mkdir(parents=True, exist_ok=True)
        else:
            print(f"would ensure folder: {directory}")

    create_pca_placeholders(overwrite=args.overwrite, dry_run=args.dry_run, summary=summary)

    source_dirs = [Path(path) for path in SEARCH_DIRS]
    if args.source_dirs:
        source_dirs.extend(Path(path) for path in args.source_dirs)

    candidates = select_newest_by_destination(discover_candidates(source_dirs, Path("docs/assets"), summary))
    copied_targets: list[Path] = []
    for candidate in candidates:
        target = destination_for(candidate)
        if is_in_tree(candidate.source, Path("docs/assets")):
            copied_targets.append(candidate.source)
            continue
        if candidate.dataset:
            copy_asset(candidate.source, target, overwrite=args.overwrite, dry_run=args.dry_run, summary=summary)
            copied_targets.append(target)
        elif candidate.likely_plot:
            copy_asset(candidate.source, target, overwrite=args.overwrite, dry_run=args.dry_run, summary=summary)
            copied_targets.append(target)

    create_combined_grids(copied_targets, overwrite=args.overwrite, dry_run=args.dry_run, summary=summary)
    create_diffusion_grid_versions(copied_targets, overwrite=args.overwrite, dry_run=args.dry_run, summary=summary)
    record_docs_images(Path("docs/index.md"), summary)
    warn_missing_expected_assets(summary)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
