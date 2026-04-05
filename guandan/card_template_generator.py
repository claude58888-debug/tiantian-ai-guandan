"""Card template image generator for Guandan recognition (M6).

Generates synthetic card corner template images using PIL/Pillow.
Each template consists of a rank label and suit symbol rendered onto
a small transparent-background PNG, suitable for OpenCV template
matching.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from guandan.models import Rank, Suit, JokerType


# ── Rendering constants ──────────────────────────────────────────────

# Template image dimensions (corner crop of a playing card).
TEMPLATE_WIDTH = 32
TEMPLATE_HEIGHT = 48

# Suit symbols and their display colours.
SUIT_SYMBOLS: Dict[Suit, str] = {
    Suit.HEARTS: '\u2665',
    Suit.DIAMONDS: '\u2666',
    Suit.CLUBS: '\u2663',
    Suit.SPADES: '\u2660',
}

SUIT_COLORS: Dict[Suit, Tuple[int, int, int]] = {
    Suit.HEARTS: (220, 20, 20),
    Suit.DIAMONDS: (220, 20, 20),
    Suit.CLUBS: (30, 30, 30),
    Suit.SPADES: (30, 30, 30),
}

RANK_LABELS: Dict[Rank, str] = {
    Rank.TWO: '2', Rank.THREE: '3', Rank.FOUR: '4',
    Rank.FIVE: '5', Rank.SIX: '6', Rank.SEVEN: '7',
    Rank.EIGHT: '8', Rank.NINE: '9', Rank.TEN: '10',
    Rank.JACK: 'J', Rank.QUEEN: 'Q', Rank.KING: 'K',
    Rank.ACE: 'A',
}


@dataclass(frozen=True)
class TemplateSpec:
    """Specification for a single card template."""
    filename: str
    rank_label: str
    suit_symbol: str
    color: Tuple[int, int, int]

    # Joker-specific fields
    is_joker: bool = False
    joker_label: str = ''


def _build_specs() -> List[TemplateSpec]:
    """Build the full list of 54 template specifications."""
    specs: List[TemplateSpec] = []
    for suit in Suit:
        for rank in Rank:
            label = RANK_LABELS[rank]
            symbol = SUIT_SYMBOLS[suit]
            color = SUIT_COLORS[suit]
            fname = f'{label}{suit.name[0]}'  # e.g. "3H", "10S"
            specs.append(TemplateSpec(
                filename=fname,
                rank_label=label,
                suit_symbol=symbol,
                color=color,
            ))
    # Jokers
    specs.append(TemplateSpec(
        filename='BJ',
        rank_label='',
        suit_symbol='',
        color=(30, 30, 30),
        is_joker=True,
        joker_label='BJ',
    ))
    specs.append(TemplateSpec(
        filename='RJ',
        rank_label='',
        suit_symbol='',
        color=(220, 20, 20),
        is_joker=True,
        joker_label='RJ',
    ))
    return specs


def _get_font(size: int) -> 'ImageFont.FreeTypeFont | ImageFont.ImageFont':
    """Try to load a TrueType font, falling back to the default bitmap font."""
    if not HAS_PIL:
        raise RuntimeError('Pillow is required for template generation')
    try:
        return ImageFont.truetype('DejaVuSans-Bold.ttf', size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', size)
        except (OSError, IOError):
            return ImageFont.load_default()


def render_template(spec: TemplateSpec,
                    width: int = TEMPLATE_WIDTH,
                    height: int = TEMPLATE_HEIGHT) -> 'Image.Image':
    """Render a single card template image with transparent background.

    Returns an RGBA PIL Image of *width* x *height* pixels.
    """
    if not HAS_PIL:
        raise RuntimeError('Pillow is required for template generation')

    img = Image.new('RGBA', (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)

    if spec.is_joker:
        font = _get_font(14)
        draw.text((2, height // 2 - 8), spec.joker_label, fill=(*spec.color, 255), font=font)
    else:
        rank_font = _get_font(16)
        suit_font = _get_font(14)
        draw.text((2, 2), spec.rank_label, fill=(*spec.color, 255), font=rank_font)
        draw.text((2, height // 2), spec.suit_symbol, fill=(*spec.color, 255), font=suit_font)

    return img


def generate_all_templates(
    output_dir: Path,
    width: int = TEMPLATE_WIDTH,
    height: int = TEMPLATE_HEIGHT,
) -> int:
    """Generate all 54 card template images and save to *output_dir*.

    Each template is saved as ``<filename>.png`` with a transparent
    background.  Returns the number of templates written.
    """
    if not HAS_PIL:
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_specs()
    count = 0
    for spec in specs:
        img = render_template(spec, width, height)
        img.save(output_dir / f'{spec.filename}.png')
        count += 1
    return count


def get_all_specs() -> List[TemplateSpec]:
    """Return the full list of 54 template specifications."""
    return _build_specs()
