"""Card template extractor from real game screenshots (M8).

Locates individual card corners from a hand of overlapping cards
and extracts rank/suit sub-images that can be used as templates
for :class:`CardRecognizer`.

Key insight: in the 天天爱掼蛋 game, cards overlap left-to-right
showing ~30-40 px of the left edge.  The rank text is large and
the suit symbol sits below it.  Vertical edge detection finds card
boundaries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

from guandan.models import Card, Rank, Suit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Typical card corner dimensions in the game (rank+suit area).
DEFAULT_CORNER_WIDTH = 40
DEFAULT_CORNER_HEIGHT = 60

# Minimum horizontal gap (px) between card left-edges.
MIN_CARD_GAP = 15

# Vertical strip used for edge detection to find card boundaries.
EDGE_KERNEL_WIDTH = 1

# Raised-card vertical offset (selected cards shift up ~15 px).
RAISED_CARD_OFFSET = 15


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CardCorner:
    """An extracted card corner crop with its position in the hand."""
    image: 'Image.Image'
    x: int = 0
    y: int = 0
    width: int = DEFAULT_CORNER_WIDTH
    height: int = DEFAULT_CORNER_HEIGHT

    def save(self, path: Path) -> None:
        """Save the corner image as PNG."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path)


@dataclass(frozen=True)
class ExtractedTemplate:
    """A labelled card template extracted from a real screenshot."""
    card: Card
    rank_image: 'Image.Image'
    suit_image: 'Image.Image'
    full_corner: 'Image.Image'


# ---------------------------------------------------------------------------
# Core extraction functions
# ---------------------------------------------------------------------------

def _find_card_edges(
    gray: 'np.ndarray',
    min_gap: int = MIN_CARD_GAP,
    threshold_ratio: float = 0.25,
) -> List[int]:
    """Find x-coordinates of card left-edges via vertical edge analysis.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image of the hand region.
    min_gap : int
        Minimum horizontal distance between detected edges.
    threshold_ratio : float
        Fraction of max column-sum to use as peak threshold.

    Returns
    -------
    list[int]
        Sorted list of x-coordinates where card edges were detected.
    """
    edges = cv2.Canny(gray, 80, 200)
    h, w = edges.shape[:2]

    # Build a vertical structuring element to emphasise card edges
    kernel_h = max(1, h // 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (EDGE_KERNEL_WIDTH, kernel_h))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    col_sum = np.sum(closed, axis=0).astype(float)
    max_val = col_sum.max()
    if max_val == 0:
        return []

    col_sum = col_sum / max_val
    threshold = threshold_ratio

    peaks: List[int] = []
    for x in range(w):
        if col_sum[x] > threshold:
            if not peaks or (x - peaks[-1]) >= min_gap:
                peaks.append(x)
    return peaks


def extract_card_corners(
    screenshot: 'Image.Image',
    hand_top_ratio: float = 0.60,
    hand_bottom_ratio: float = 1.0,
    corner_width: int = DEFAULT_CORNER_WIDTH,
    corner_height: int = DEFAULT_CORNER_HEIGHT,
    min_gap: int = MIN_CARD_GAP,
) -> List[CardCorner]:
    """Locate individual card corners from a screenshot's hand region.

    Parameters
    ----------
    screenshot : Image.Image
        Full game screenshot.
    hand_top_ratio / hand_bottom_ratio : float
        Vertical fraction defining the hand region.
    corner_width / corner_height : int
        Size of each corner crop.
    min_gap : int
        Minimum px between card edges.

    Returns
    -------
    list[CardCorner]
        Extracted corners sorted left-to-right.
    """
    if not HAS_CV2 or not HAS_PIL:
        return []

    w, h = screenshot.size
    top_px = int(h * hand_top_ratio)
    bottom_px = int(h * hand_bottom_ratio)
    hand = screenshot.crop((0, top_px, w, bottom_px))
    hand_w, hand_h = hand.size

    gray = cv2.cvtColor(
        np.array(hand.convert('RGB')), cv2.COLOR_RGB2GRAY,
    )
    peaks = _find_card_edges(gray, min_gap=min_gap)

    corners: List[CardCorner] = []
    cw = min(corner_width, hand_w)
    ch = min(corner_height, hand_h)

    for x in peaks:
        x_end = min(x + cw, hand_w)
        crop = hand.crop((x, 0, x_end, ch))
        corners.append(CardCorner(
            image=crop, x=x, y=top_px,
            width=x_end - x, height=ch,
        ))
    return corners


def extract_rank_region(
    card_corner: 'Image.Image',
) -> 'Image.Image':
    """Extract the rank text sub-image from a card corner.

    The rank occupies roughly the top 45 % of the corner image.
    """
    w, h = card_corner.size
    return card_corner.crop((0, 0, w, int(h * 0.45)))


def extract_suit_region(
    card_corner: 'Image.Image',
) -> 'Image.Image':
    """Extract the suit symbol sub-image from a card corner.

    The suit sits below the rank, roughly in the 40-75 % vertical band.
    """
    w, h = card_corner.size
    return card_corner.crop((0, int(h * 0.40), w, int(h * 0.75)))


def detect_raised_cards(
    screenshot: 'Image.Image',
    hand_top_ratio: float = 0.60,
    offset_px: int = RAISED_CARD_OFFSET,
) -> List[int]:
    """Detect which card positions are raised (selected).

    Raised cards shift upward by ~15 px.  We compare pixel intensity
    at two y-levels to spot the offset.

    Returns a list of x-coordinates of raised cards.
    """
    if not HAS_CV2 or not HAS_PIL:
        return []

    w, h = screenshot.size
    top_px = int(h * hand_top_ratio)
    # Look at a thin strip just above the normal hand top
    strip_top = max(0, top_px - offset_px - 5)
    strip_bottom = top_px
    strip = screenshot.crop((0, strip_top, w, strip_bottom))

    gray = cv2.cvtColor(
        np.array(strip.convert('RGB')), cv2.COLOR_RGB2GRAY,
    )
    # Card pixels are generally lighter than background
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    col_sum = np.sum(binary, axis=0).astype(float)
    max_val = col_sum.max() if col_sum.max() > 0 else 1.0
    col_sum = col_sum / max_val

    raised_xs: List[int] = []
    in_card = False
    card_start = 0
    for x in range(len(col_sum)):
        if col_sum[x] > 0.3 and not in_card:
            in_card = True
            card_start = x
        elif col_sum[x] <= 0.3 and in_card:
            in_card = False
            raised_xs.append((card_start + x) // 2)

    if in_card:
        raised_xs.append((card_start + len(col_sum)) // 2)

    return raised_xs


# ---------------------------------------------------------------------------
# Template set builder
# ---------------------------------------------------------------------------

def build_template_set(
    corners: List[CardCorner],
    labels: List[Card],
    output_dir: Optional[Path] = None,
) -> Dict[str, 'Image.Image']:
    """Pair extracted corners with card labels to build a template set.

    Parameters
    ----------
    corners : list[CardCorner]
        Extracted card corners (left-to-right order).
    labels : list[Card]
        Known card labels matching the corners (same order).
    output_dir : Path | None
        If provided, saves each template as ``<name>.png``.

    Returns
    -------
    dict[str, Image.Image]
        Mapping from card name to corner image.
    """
    templates: Dict[str, 'Image.Image'] = {}
    count = min(len(corners), len(labels))

    for i in range(count):
        card = labels[i]
        name = card.display().replace('\u2665', 'H').replace(
            '\u2666', 'D',
        ).replace('\u2663', 'C').replace('\u2660', 'S')
        templates[name] = corners[i].image

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            corners[i].image.save(output_dir / f'{name}.png')

    return templates


def classify_suit_by_colour(
    corner: 'Image.Image',
) -> Optional[str]:
    """Classify a card corner as 'red' or 'black' based on HSV colour.

    Returns ``'red'``, ``'black'``, or ``None`` if unclear.
    """
    if not HAS_CV2 or not HAS_PIL:
        return None

    bgr = cv2.cvtColor(
        np.array(corner.convert('RGB')), cv2.COLOR_RGB2BGR,
    )
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Red hue wraps around 0/180
    lower_red1 = np.array([0, 80, 80], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([160, 80, 80], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(
        hsv, lower_red2, upper_red2,
    )

    # Black/dark pixels
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([180, 50, 80], dtype=np.uint8)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)

    red_count = int(np.count_nonzero(mask_red))
    black_count = int(np.count_nonzero(mask_black))

    if red_count > black_count and red_count > 0:
        return 'red'
    if black_count > red_count and black_count > 0:
        return 'black'
    return None
