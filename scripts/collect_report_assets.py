#!/usr/bin/env python3
"""Collect small, final report images into docs/assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError


DEFAULT_SOURCE_DIRS = ("outputs", "runs", "results", "figures", "plots", "images")
ASSET_ROOT = Path("docs/assets")
PCA_PLACEHOLDERS = {
    "pca_mnist_reconstructions_placeholder.png": "PCA MNIST reconstruction grid from teammate.",
    "pca_fashion_reconstructions_placeholder.png": "PCA Fashion-MNIST reconstruction grid from teammate.",
    "pca_metrics_placeholder.png": "PCA metrics plot: MSE, PSNR, and SSIM.",
    "pca_vs_ae_combined_grid_placeholder.png": "Final PCA vs autoencoder comparison grid.",
}
AUTOENCODER_TARGETS = {
    Path("mnist/ae_reconstruction_latent16.png"): (
        "*mnist*ae*recon*latent*16*.png",
        "*mnist_ae_latent_16.png",
    ),
    Path("mnist/ae_latent_space_latent16.png"): ("*mnist*ae*latent_space*16*.png",),
    Path("mnist/ae_loss_latent16.png"): ("*mnist*ae*loss*16*.png",),
    Path("mnist/dae_reconstruction_noise02_latent16.png"): (
        "*mnist*dae*noise*0.2*latent*16*.png",
        "*mnist*dae*recon*latent*16*.png",
    ),
    Path("mnist/dae_loss_noise02_latent16.png"): ("*mnist*dae*noise*0.2*loss*16*.png",),
    Path("mnist/vae_reconstruction_latent16.png"): ("*mnist*vae*latent*16*.png",),
    Path("mnist/vae_latent_space_latent16.png"): ("*mnist*vae*latent_space*16*.png",),
    Path("mnist/vae_generated_latent16.png"): ("*vae_generated_mnist_latent_16.png",),
    Path("mnist/metrics_by_model_latent.png"): ("*mnist*metrics*latent*model*.png",),
    Path("mnist/model_comparison_latent16.png"): ("*mnist*model_comparison*latent*16*.png",),
    Path("combined/vae_interpolation_latent16.png"): (
        "*vae_interpolation_better.png",
        "*vae_interpolation_mnist_latent_16.png",
    ),
    Path("fashion_mnist/ae_reconstruction_latent16.png"): ("*fashion*ae*recon*latent*16*.png",),
    Path("fashion_mnist/dae_reconstruction_noise02_latent16.png"): ("*fashion*dae*noise*0.2*latent*16*.png",),
    Path("fashion_mnist/vae_reconstruction_latent16.png"): ("*fashion*vae*latent*16*.png",),
}
DIFFUSION_TARGETS = {
    "mnist": {
        "prefix": "mnist",
        "sample_size": 28,
        "patterns": (
            "*mnist*/diffusion/*/samples/generated_samples_native_grid.png",
            "*mnist*contact*100*native*1x*nopad.png",
            "*mnist*contact*100*native*1x.png",
        ),
    },
    "cifar10": {
        "prefix": "cifar10",
        "sample_size": 32,
        "patterns": (
            "*cifar10*/diffusion/*/samples/generated_samples_native_grid.png",
            "*cifar*contact*100*native*1x*nopad.png",
            "*cifar*contact*100*native*1x.png",
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect final GitHub Pages image assets.")
    parser.add_argument("--source-dir", action="append", dest="source_dirs", help="Extra source directory to search.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing docs/assets files.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned writes without changing files.")
    return parser.parse_args()


def load_font(size: int) -> ImageFont.ImageFont:
    for name in ("Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, width: int) -> None:
    y = xy[1]
    for line in textwrap.wrap(text, width=width):
        draw.text((xy[0], y), line, font=font, fill="#5f554d")
        bbox = draw.textbbox((xy[0], y), line, font=font)
        y += bbox[3] - bbox[1] + 8


def create_placeholder(path: Path, title: str, body: str, *, size: tuple[int, int], overwrite: bool, dry_run: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    if dry_run:
        print(f"would create placeholder: {path}")
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, "#f7f3eb")
    draw = ImageDraw.Draw(image)
    draw.rectangle((18, 18, size[0] - 18, size[1] - 18), outline="#b13f32", width=4)
    draw.text((48, 52), title, font=load_font(30), fill="#17130f")
    draw_wrapped(draw, (48, 120), body, load_font(22), width=48)
    image.save(path)
    print(f"created placeholder: {path}")
    return True


def newest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path.is_file()]
    if not existing:
        return None
    return max(existing, key=lambda path: (path.stat().st_mtime, path.as_posix()))


def find_source(source_dirs: list[Path], patterns: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    for root in source_dirs:
        if not root.is_dir():
            continue
        for pattern in patterns:
            candidates.extend(root.rglob(pattern))
    return newest(candidates)


def copy_asset(source: Path, target: Path, *, overwrite: bool, dry_run: bool) -> bool:
    if target.exists() and not overwrite:
        return False
    if dry_run:
        print(f"would copy: {source} -> {target}")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"copied: {source} -> {target}")
    return True


def save_nearest_scaled(source: Path, target: Path, *, scale: int, overwrite: bool, dry_run: bool) -> bool:
    if target.exists() and not overwrite:
        return False
    if dry_run:
        print(f"would save nearest-neighbor scaled image: {source} -> {target}")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        resized = rgb.resize((rgb.width * scale, rgb.height * scale), resample=Image.Resampling.NEAREST)
        resized.save(target)
    print(f"saved nearest-neighbor scaled image: {target}")
    return True


def create_pca_placeholders(*, overwrite: bool, dry_run: bool) -> int:
    count = 0
    for filename, body in PCA_PLACEHOLDERS.items():
        count += int(
            create_placeholder(
                ASSET_ROOT / "placeholders" / filename,
                "PCA Results Placeholder",
                body,
                size=(900, 540),
                overwrite=overwrite,
                dry_run=dry_run,
            )
        )
    return count


def collect_autoencoder_assets(source_dirs: list[Path], *, overwrite: bool, dry_run: bool) -> int:
    count = 0
    for relative_target, patterns in AUTOENCODER_TARGETS.items():
        source = find_source(source_dirs, patterns)
        target = ASSET_ROOT / relative_target
        if source is None:
            if not target.exists():
                print(f"missing source for {target}: {patterns}")
            continue
        count += int(copy_asset(source, target, overwrite=overwrite, dry_run=dry_run))
    return count


def collect_diffusion_assets(source_dirs: list[Path], *, overwrite: bool, dry_run: bool) -> int:
    count = 0
    for dataset, spec in DIFFUSION_TARGETS.items():
        source = find_source(source_dirs, spec["patterns"])
        native = ASSET_ROOT / "diffusion" / f"{dataset}_samples_native.png"
        scaled = ASSET_ROOT / "diffusion" / f"{dataset}_samples_nearest_4x.png"
        sample_size = int(spec["sample_size"])

        if source is None:
            if not native.exists():
                count += int(
                    create_placeholder(
                        native,
                        f"{dataset.upper()} Diffusion Placeholder",
                        f"Native {sample_size}x{sample_size} generated sample grid missing.",
                        size=(sample_size * 10, sample_size * 10),
                        overwrite=overwrite,
                        dry_run=dry_run,
                    )
                )
            if not scaled.exists():
                count += int(
                    create_placeholder(
                        scaled,
                        f"{dataset.upper()} Diffusion Placeholder",
                        "Run diffusion training, then collect assets again.",
                        size=(sample_size * 40, sample_size * 40),
                        overwrite=overwrite,
                        dry_run=dry_run,
                    )
                )
            continue

        count += int(copy_asset(source, native, overwrite=overwrite, dry_run=dry_run))
        try:
            count += int(save_nearest_scaled(source, scaled, scale=4, overwrite=overwrite, dry_run=dry_run))
        except (OSError, UnidentifiedImageError):
            count += int(copy_asset(source, scaled, overwrite=overwrite, dry_run=dry_run))
    return count


def docs_image_sources(index_path: Path) -> list[str]:
    if not index_path.exists():
        return []
    text = index_path.read_text(encoding="utf-8")
    markdown = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    html = re.findall(r"<img\b[^>]*\bsrc=[\"']([^\"']+)[\"']", text, flags=re.IGNORECASE)
    return [source.split("#", 1)[0].split("?", 1)[0].strip() for source in markdown + html]


def report_docs_links() -> bool:
    index_path = Path("docs/index.md")
    ok = True
    for source in docs_image_sources(index_path):
        if re.match(r"^[a-z]+://", source):
            continue
        resolved = index_path.parent / source
        if not resolved.exists():
            ok = False
            print(f"missing docs image: {source}")
    return ok


def main() -> int:
    args = parse_args()
    source_dirs = [Path(path) for path in DEFAULT_SOURCE_DIRS]
    if args.source_dirs:
        source_dirs.extend(Path(path) for path in args.source_dirs)

    if not args.dry_run:
        for directory in ("mnist", "fashion_mnist", "diffusion", "combined", "placeholders"):
            (ASSET_ROOT / directory).mkdir(parents=True, exist_ok=True)

    changed = 0
    changed += create_pca_placeholders(overwrite=args.overwrite, dry_run=args.dry_run)
    changed += collect_autoencoder_assets(source_dirs, overwrite=args.overwrite, dry_run=args.dry_run)
    changed += collect_diffusion_assets(source_dirs, overwrite=args.overwrite, dry_run=args.dry_run)

    links_ok = report_docs_links()
    print(f"asset operations: {changed}")
    print(f"docs image links: {'ok' if links_ok else 'missing'}")
    return 0 if links_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
