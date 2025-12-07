"""Generate synthetic digits rendered using computer fonts."""

from __future__ import annotations

import random
import uuid
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image, ImageDraw, ImageFont

from expforge.datasets import DatasetRecord, ensure_dataset_dirs, save_manifest

DEFAULT_FONTS: List[Path] = [
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
]


def available_fonts(paths: Optional[Iterable[Path]]) -> List[Path]:
    candidates = list(paths) if paths else DEFAULT_FONTS
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError("No fonts found. Provide at least one .ttf or .otf file.")
    return existing


def render_digit(digit: int, font: ImageFont.FreeTypeFont, canvas_size: int = 28) -> Image.Image:
    image = Image.new("L", (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(image)
    text = str(digit)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (canvas_size - text_width) // 2
    y = (canvas_size - text_height) // 2
    draw.text((x, y), text, fill=255, font=font)
    return image


def build_font_manifest(
    count: int,
    output_dir: Path,
    fonts: Optional[Iterable[Path]] = None,
    seed: Optional[int] = None,
) -> List[DatasetRecord]:
    rng = random.Random(seed)
    fonts_to_use = available_fonts(fonts)
    images_dir = ensure_dataset_dirs(output_dir)

    records: List[DatasetRecord] = []
    for _ in range(count):
        digit = rng.randint(0, 9)
        font_path = rng.choice(fonts_to_use)
        font_size = rng.randint(18, 26)
        font = ImageFont.truetype(str(font_path), size=font_size)
        img = render_digit(digit, font)

        file_name = f"{uuid.uuid4().hex}.png"
        rel_path = Path("images") / file_name
        img.save(images_dir / file_name)

        records.append(
            DatasetRecord(
                path=str(rel_path),
                label=str(digit),
                split="train",
            )
        )

    return save_manifest(records, output_dir)
