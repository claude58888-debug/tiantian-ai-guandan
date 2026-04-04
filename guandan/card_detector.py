"""Card detection from screenshots (Atom 2.2).

Uses template matching and color analysis to identify cards
from game screenshots. Provides both template-based and
hash-based recognition approaches.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from PIL import Image
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from guandan.models import Card, Rank, Suit, JokerType


@dataclass
class DetectedCard:
    """A card detected from a screenshot with position info."""
    card: Card
    confidence: float = 0.0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    center: Tuple[int, int] = (0, 0)

    def __repr__(self) -> str:
        return f'DetectedCard({self.card.display()}, conf={self.confidence:.2f})'


@dataclass
class CardTemplate:
    """Template image for a specific card."""
    card: Card
    image: Optional[object] = None  # PIL Image
    phash: str = ''

    @staticmethod
    def compute_phash(img: 'Image.Image', hash_size: int = 8) -> str:
        """Compute perceptual hash of an image."""
        resized = img.convert('L').resize((hash_size + 1, hash_size), Image.LANCZOS)
        pixels = list(resized.getdata())
        avg = sum(pixels) / len(pixels)
        bits = ''.join('1' if p > avg else '0' for p in pixels)
        return hashlib.md5(bits.encode()).hexdigest()[:16]


def hamming_distance(h1: str, h2: str) -> int:
    """Compute hamming distance between two hex hash strings."""
    b1 = bin(int(h1, 16))[2:].zfill(len(h1) * 4)
    b2 = bin(int(h2, 16))[2:].zfill(len(h2) * 4)
    return sum(c1 != c2 for c1, c2 in zip(b1, b2))


class TemplateStore:
    """Manages card template images for matching."""

    def __init__(self, template_dir: Optional[Path] = None):
        self.template_dir = template_dir or Path('templates')
        self.templates: Dict[str, CardTemplate] = {}

    def load_templates(self) -> int:
        """Load template images from directory. Returns count loaded."""
        if not HAS_DEPS or not self.template_dir.exists():
            return 0

        count = 0
        for img_path in self.template_dir.glob('*.png'):
            card = self._parse_filename(img_path.stem)
            if card is None:
                continue
            img = Image.open(img_path)
            phash = CardTemplate.compute_phash(img)
            key = card.display()
            self.templates[key] = CardTemplate(card=card, image=img, phash=phash)
            count += 1
        return count

    def _parse_filename(self, name: str) -> Optional[Card]:
        """Parse card from template filename like '3H', '10S', 'BJ', 'RJ'."""
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

    def find_best_match(self, img: 'Image.Image', threshold: float = 0.8) -> Optional[DetectedCard]:
        """Find the best matching template for a card image."""
        if not HAS_DEPS or not self.templates:
            return None

        query_hash = CardTemplate.compute_phash(img)
        best_match = None
        best_dist = float('inf')

        for key, tmpl in self.templates.items():
            dist = hamming_distance(query_hash, tmpl.phash)
            if dist < best_dist:
                best_dist = dist
                best_match = tmpl

        if best_match is None:
            return None

        max_dist = 64  # max possible hamming distance for 16-char hex
        confidence = 1.0 - (best_dist / max_dist)
        if confidence < threshold:
            return None

        return DetectedCard(card=best_match.card, confidence=confidence)


def segment_hand_cards(hand_img: 'Image.Image',
                       card_width: int = 40,
                       overlap: int = 15) -> List['Image.Image']:
    """Segment individual card images from a hand region.

    Cards in hand typically overlap. This segments them by
    stepping through the image at regular intervals.
    """
    if not HAS_DEPS:
        return []

    width, height = hand_img.size
    cards = []
    step = card_width - overlap
    x = 0
    while x + card_width <= width:
        card_img = hand_img.crop((x, 0, x + card_width, height))
        cards.append(card_img)
        x += step

    # Get last card if partial
    if x < width and width - x > overlap and width >= card_width:
        cards.append(hand_img.crop((width - card_width, 0, width, height)))

    return cards


def detect_card_color(img: 'Image.Image') -> Optional[str]:
    """Detect if a card region is red or black based on color analysis."""
    if not HAS_DEPS:
        return None

    arr = np.array(img.convert('RGB'))
    r_mean = arr[:, :, 0].mean()
    g_mean = arr[:, :, 1].mean()
    b_mean = arr[:, :, 2].mean()

    # Red cards have significantly higher red channel
    if r_mean > g_mean * 1.3 and r_mean > b_mean * 1.3:
        return 'red'
    return 'black'


class CardDetector:
    """Main card detection pipeline."""

    def __init__(self, template_dir: Optional[Path] = None):
        self.store = TemplateStore(template_dir)
        self._loaded = False

    def load(self) -> int:
        """Load templates. Returns count."""
        count = self.store.load_templates()
        self._loaded = count > 0
        return count

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def detect_hand(self, hand_img: 'Image.Image',
                    card_width: int = 40,
                    overlap: int = 15,
                    threshold: float = 0.7) -> List[DetectedCard]:
        """Detect all cards in a hand image."""
        segments = segment_hand_cards(hand_img, card_width, overlap)
        results = []
        for i, seg in enumerate(segments):
            match = self.store.find_best_match(seg, threshold)
            if match is not None:
                match.bbox = (i * (card_width - overlap), 0, card_width, hand_img.size[1])
                match.center = (match.bbox[0] + card_width // 2, hand_img.size[1] // 2)
                results.append(match)
        return results

    def detect_single(self, card_img: 'Image.Image',
                      threshold: float = 0.7) -> Optional[DetectedCard]:
        """Detect a single card from an image."""
        return self.store.find_best_match(card_img, threshold)
