from __future__ import annotations

from pathlib import Path
from typing import Iterable

import kagglehub

REAL_PATTERNS = {"real", "real_images", "reals", "human"}
AI_PATTERNS = {"ai", "fake", "generated", "synthetic", "ai_images"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _normalize(name: str) -> str:
    return name.lower().replace("-", "_").replace(" ", "_")


def _iter_dirs(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_dir():
            yield path


def _has_images(path: Path) -> bool:
    return any(p.suffix.lower() in IMAGE_EXTS for p in path.rglob("*"))


def find_class_dirs(dataset_root: Path) -> tuple[Path, Path]:
    real_candidates: list[Path] = []
    ai_candidates: list[Path] = []

    for folder in _iter_dirs(dataset_root):
        normalized = _normalize(folder.name)
        if normalized in REAL_PATTERNS and _has_images(folder):
            real_candidates.append(folder)
        if normalized in AI_PATTERNS and _has_images(folder):
            ai_candidates.append(folder)

    if not real_candidates or not ai_candidates:
        top_level = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])
        raise ValueError(
            "Could not auto-detect both classes. "
            f"Top-level folders found: {top_level}. "
            f"Expected real folder patterns: {sorted(REAL_PATTERNS)} and "
            f"AI folder patterns: {sorted(AI_PATTERNS)}."
        )

    real_dir = sorted(real_candidates, key=lambda p: len(p.parts))[0]
    ai_dir = sorted(ai_candidates, key=lambda p: len(p.parts))[0]
    return real_dir, ai_dir


def download_and_discover(dataset_ref: str) -> tuple[Path, Path, Path]:
    downloaded = Path(kagglehub.dataset_download(dataset_ref))
    real_dir, ai_dir = find_class_dirs(downloaded)
    return downloaded, real_dir, ai_dir
