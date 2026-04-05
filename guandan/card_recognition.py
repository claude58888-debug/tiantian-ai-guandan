"""Card recognition engine using OpenCV template matching (M6).

Detects cards from game screenshots by matching against pre-generated
template images at multiple scales.  Includes non-maximum suppression
to eliminate duplicate detections.
"""
from __future__ import annotations

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

from guandan.models import Card, Rank, Suit, JokerType
from guandan.card_template_generator import (
    TEMPLATE_WIDTH,
    TEMPLATE_HEIGHT,
    generate_all_templates,
)


# ── Configuration ─────────────────────────────────────────────────────

DEFAULT_MATCH_THRESHOLD = 0.8
DEFAULT_SCALES: Tuple[float, ...] = (0.8, 1.0, 1.2)
NMS_OVERLAP_THRESHOLD = 0.4


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class RecognizedCard:
    """A card detected from a screenshot with location and confidence."""
    card: Card
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h

    def __repr__(self) -> str:
        return (
            f'RecognizedCard({self.card.display()}, '
            f'conf={self.confidence:.2f}, bbox={self.bbox})'
        )


@dataclass(frozen=True)
class CardTemplate:
    """Holds a loaded template image for a single card."""
    card: Card
    image: 'np.ndarray'  # grayscale OpenCV image
    name: str = ''


# ── Helpers ───────────────────────────────────────────────────────────

def _pil_to_cv2_gray(img: 'Image.Image') -> 'np.ndarray':
    """Convert a PIL Image to a grayscale OpenCV array."""
    return cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2GRAY)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute intersection-over-union of two (x, y, w, h) boxes."""
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0

    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter
    if union == 0:
        return 0.0
    return inter / union


def non_maximum_suppression(
    detections: List[RecognizedCard],
    overlap_threshold: float = NMS_OVERLAP_THRESHOLD,
) -> List[RecognizedCard]:
    """Remove overlapping detections, keeping those with highest confidence."""
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep: List[RecognizedCard] = []

    for det in sorted_dets:
        suppressed = False
        for kept in keep:
            if _iou(det.bbox, kept.bbox) > overlap_threshold:
                suppressed = True
                break
        if not suppressed:
            keep.append(det)
    return keep


def _parse_template_filename(name: str) -> Optional[Card]:
    """Parse a card from a template filename like '3H', '10S', 'BJ', 'RJ'."""
    name = name.upper()
    if name == 'BJ':
        return Card(joker=JokerType.BLACK)
    if name == 'RJ':
        return Card(joker=JokerType.RED)

    suit_map = {'H': Suit.HEARTS, 'D': Suit.DIAMONDS,
                'C': Suit.CLUBS, 'S': Suit.SPADES}
    rank_map = {'2': Rank.TWO, '3': Rank.THREE, '4': Rank.FOUR,
                '5': Rank.FIVE, '6': Rank.SIX, '7': Rank.SEVEN,
                '8': Rank.EIGHT, '9': Rank.NINE, '10': Rank.TEN,
                'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING,
                'A': Rank.ACE}

    if len(name) < 2:
        return None
    suit_char = name[-1]
    rank_str = name[:-1]
    if suit_char not in suit_map or rank_str not in rank_map:
        return None
    return Card(rank=rank_map[rank_str], suit=suit_map[suit_char])


# ── Core recogniser ──────────────────────────────────────────────────

class CardRecognizer:
    """OpenCV-based card recognition engine.

    Uses :func:`cv2.matchTemplate` with ``TM_CCOEFF_NORMED`` at
    multiple scales for resolution independence, followed by
    non-maximum suppression to deduplicate.
    """

    def __init__(
        self,
        threshold: float = DEFAULT_MATCH_THRESHOLD,
        scales: Sequence[float] = DEFAULT_SCALES,
        nms_overlap: float = NMS_OVERLAP_THRESHOLD,
    ) -> None:
        self.threshold = threshold
        self.scales = tuple(scales)
        self.nms_overlap = nms_overlap
        self._templates: List[CardTemplate] = []

    @property
    def is_loaded(self) -> bool:
        return len(self._templates) > 0

    @property
    def template_count(self) -> int:
        return len(self._templates)

    # ── Template loading ──────────────────────────────────────────

    def load_templates(self, template_dir: Path) -> int:
        """Load card template PNGs from *template_dir*.

        Each PNG filename (without extension) is parsed as a card
        identifier (e.g. ``3H.png``, ``10S.png``, ``BJ.png``).
        Returns the number of templates loaded.
        """
        if not HAS_CV2 or not HAS_PIL:
            return 0
        if not template_dir.is_dir():
            return 0

        self._templates.clear()
        for img_path in sorted(template_dir.glob('*.png')):
            card = _parse_template_filename(img_path.stem)
            if card is None:
                continue
            pil_img = Image.open(img_path)
            gray = _pil_to_cv2_gray(pil_img)
            self._templates.append(CardTemplate(
                card=card, image=gray, name=img_path.stem,
            ))
        return len(self._templates)

    def load_templates_from_pil(
        self,
        images: Dict[str, 'Image.Image'],
    ) -> int:
        """Load templates from a dict of ``{name: PIL.Image}`` pairs.

        Useful for testing without touching the filesystem.
        """
        if not HAS_CV2:
            return 0

        self._templates.clear()
        for name, pil_img in images.items():
            card = _parse_template_filename(name)
            if card is None:
                continue
            gray = _pil_to_cv2_gray(pil_img)
            self._templates.append(CardTemplate(
                card=card, image=gray, name=name,
            ))
        return len(self._templates)

    # ── Detection ─────────────────────────────────────────────────

    def find_cards_in_region(
        self,
        image: 'Image.Image',
        threshold: Optional[float] = None,
    ) -> List[RecognizedCard]:
        """Detect cards in a PIL image region using multi-scale template matching.

        Returns a deduplicated list of :class:`RecognizedCard` sorted by
        descending confidence.
        """
        if not HAS_CV2 or not self._templates:
            return []

        thresh = threshold if threshold is not None else self.threshold
        scene_gray = _pil_to_cv2_gray(image)
        scene_h, scene_w = scene_gray.shape[:2]
        raw_detections: List[RecognizedCard] = []

        for tmpl in self._templates:
            for scale in self.scales:
                th, tw = tmpl.image.shape[:2]
                new_w = max(1, int(tw * scale))
                new_h = max(1, int(th * scale))

                if new_w > scene_w or new_h > scene_h:
                    continue

                scaled_tmpl = cv2.resize(tmpl.image, (new_w, new_h))
                result = cv2.matchTemplate(
                    scene_gray, scaled_tmpl, cv2.TM_CCOEFF_NORMED,
                )

                locations = np.where(result >= thresh)
                for pt_y, pt_x in zip(*locations):
                    conf = float(result[pt_y, pt_x])
                    raw_detections.append(RecognizedCard(
                        card=tmpl.card,
                        confidence=conf,
                        bbox=(int(pt_x), int(pt_y), new_w, new_h),
                    ))

        return non_maximum_suppression(raw_detections, self.nms_overlap)

    def recognize_hand(
        self,
        screenshot: 'Image.Image',
    ) -> List[Card]:
        """Detect the player's hand from the bottom region of a screenshot.

        Crops the bottom ~20% of the image as the hand region and
        returns a sorted list of unique detected cards.
        """
        if not HAS_CV2:
            return []

        w, h = screenshot.size
        hand_region = screenshot.crop((0, int(h * 0.8), w, h))
        detections = self.find_cards_in_region(hand_region)
        seen: set[str] = set()
        cards: List[Card] = []
        for det in detections:
            key = det.card.display()
            if key not in seen:
                seen.add(key)
                cards.append(det.card)
        return cards

    def recognize_played_cards(
        self,
        screenshot: 'Image.Image',
    ) -> List[Card]:
        """Detect cards in the centre played area of a screenshot.

        Crops the middle ~30% vertically and ~40% horizontally.
        """
        if not HAS_CV2:
            return []

        w, h = screenshot.size
        left = int(w * 0.3)
        right = int(w * 0.7)
        top = int(h * 0.35)
        bottom = int(h * 0.65)
        center_region = screenshot.crop((left, top, right, bottom))
        detections = self.find_cards_in_region(center_region)
        seen: set[str] = set()
        cards: List[Card] = []
        for det in detections:
            key = det.card.display()
            if key not in seen:
                seen.add(key)
                cards.append(det.card)
        return cards

    def recognize_game_state(
        self,
        screenshot: 'Image.Image',
    ) -> Dict[str, object]:
        """Extract a high-level game state dict from a full screenshot.

        Returns a dict with keys ``hand``, ``played``, and
        ``opponent_counts`` (estimated from card-back regions — always
        ``{}`` in this implementation since card-back detection is not
        yet supported).
        """
        hand = self.recognize_hand(screenshot)
        played = self.recognize_played_cards(screenshot)
        return {
            'hand': hand,
            'played': played,
            'opponent_counts': {},
        }
