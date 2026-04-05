"""Game-specific card recognizer tuned for 天天爱掼蛋 (M8).

Extends :class:`CardRecognizer` with:
- Real game card templates (extracted from screenshots) instead of
  synthetic ones.
- Optimised screen region detection for the actual game layout.
- Overlapping-card scanning (40 px vertical strips, left-to-right).
- HSV colour-based suit classification (red vs black).
- Raised-card state detection (selected cards shift up ~15 px).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
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

from guandan.calibration import CalibrationData, HSVRange
from guandan.card_extractor import (
    CardCorner,
    classify_suit_by_colour,
    extract_card_corners,
)
from guandan.card_recognition import (
    CardRecognizer,
    CardTemplate,
    RecognizedCard,
    non_maximum_suppression,
)
from guandan.game_screen_analyzer import ScreenRegions
from guandan.models import Card, Rank, Suit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Game-specific region defaults (measured from real screenshots)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GameLayout:
    """Pixel-ratio regions tuned for the actual 天天爱掼蛋 layout.

    All values are fractions of the full screenshot dimensions.
    """
    # Hand: bottom ~35-45 % of screen, cards spread 10-90 % width
    hand_top: float = 0.58
    hand_bottom: float = 1.0
    hand_left: float = 0.08
    hand_right: float = 0.92

    # Played: centre region
    played_top: float = 0.30
    played_bottom: float = 0.55
    played_left: float = 0.25
    played_right: float = 0.75

    # Counter: top-right
    counter_top: float = 0.0
    counter_bottom: float = 0.10
    counter_left: float = 0.65
    counter_right: float = 1.0

    # Buttons: centre
    button_top: float = 0.45
    button_bottom: float = 0.55
    button_left: float = 0.35
    button_right: float = 0.65

    def to_screen_regions(self) -> ScreenRegions:
        """Convert to a :class:`ScreenRegions` instance."""
        return ScreenRegions(
            hand_top=self.hand_top,
            hand_bottom=self.hand_bottom,
            hand_left=self.hand_left,
            hand_right=self.hand_right,
            played_top=self.played_top,
            played_bottom=self.played_bottom,
            played_left=self.played_left,
            played_right=self.played_right,
            counter_top=self.counter_top,
            counter_bottom=self.counter_bottom,
            counter_left=self.counter_left,
            counter_right=self.counter_right,
            button_top=self.button_top,
            button_bottom=self.button_bottom,
            button_left=self.button_left,
            button_right=self.button_right,
        )


DEFAULT_GAME_LAYOUT = GameLayout()

# Width of the vertical strip scanned per overlapping card (px).
STRIP_WIDTH = 40

# Raised card vertical offset (px).
RAISED_OFFSET_PX = 15


# ---------------------------------------------------------------------------
# GameSpecificRecognizer
# ---------------------------------------------------------------------------

class GameSpecificRecognizer(CardRecognizer):
    """Card recognizer tuned for the 天天爱掼蛋 game client.

    Differences from base :class:`CardRecognizer`:

    - Uses real game templates when available (loaded via
      :meth:`load_game_templates`).
    - :meth:`recognize_hand_strips` scans the hand region in 40 px
      vertical strips to handle overlapping cards.
    - Colour-based suit classification for ambiguous detections.
    - ``raised_threshold`` detects selected (raised) cards.
    """

    def __init__(
        self,
        threshold: float = 0.75,
        scales: Sequence[float] = (0.8, 0.9, 1.0, 1.1, 1.2),
        nms_overlap: float = 0.4,
        calibration: Optional[CalibrationData] = None,
        layout: Optional[GameLayout] = None,
    ) -> None:
        if calibration is not None:
            threshold = calibration.match_threshold
        super().__init__(
            threshold=threshold,
            scales=scales,
            nms_overlap=nms_overlap,
        )
        self._calibration = calibration
        self._layout = layout or DEFAULT_GAME_LAYOUT
        self._game_templates_loaded = False

    # -- properties --------------------------------------------------------

    @property
    def calibration(self) -> Optional[CalibrationData]:
        return self._calibration

    @property
    def layout(self) -> GameLayout:
        return self._layout

    @property
    def game_templates_loaded(self) -> bool:
        return self._game_templates_loaded

    # -- template loading --------------------------------------------------

    def load_game_templates(self, template_dir: Path) -> int:
        """Load real game card templates from a directory.

        Prefers a ``real/`` subdirectory if it exists, falling back
        to the parent directory.
        """
        real_dir = template_dir / 'real'
        target = real_dir if real_dir.is_dir() else template_dir
        count = self.load_templates(target)
        if count > 0:
            self._game_templates_loaded = True
        return count

    def load_game_templates_from_pil(
        self,
        images: Dict[str, 'Image.Image'],
    ) -> int:
        """Load real game templates from PIL images (for testing)."""
        count = self.load_templates_from_pil(images)
        if count > 0:
            self._game_templates_loaded = True
        return count

    # -- strip-based hand recognition --------------------------------------

    def recognize_hand_strips(
        self,
        screenshot: 'Image.Image',
        strip_width: int = STRIP_WIDTH,
    ) -> List[RecognizedCard]:
        """Scan the hand region in vertical strips for overlapping cards.

        Each strip is ``strip_width`` pixels wide.  Template matching
        is performed on each strip independently, then results are
        merged via NMS.

        Returns deduplicated :class:`RecognizedCard` list sorted by
        x-position.
        """
        if not HAS_CV2 or not HAS_PIL or not self.is_loaded:
            return []

        w, h = screenshot.size
        ly = self._layout
        hand_top = int(h * ly.hand_top)
        hand_bottom = int(h * ly.hand_bottom)
        hand_left = int(w * ly.hand_left)
        hand_right = int(w * ly.hand_right)

        hand = screenshot.crop((hand_left, hand_top, hand_right, hand_bottom))
        hand_w, hand_h = hand.size
        all_detections: List[RecognizedCard] = []

        for x_start in range(0, hand_w, strip_width):
            x_end = min(x_start + strip_width, hand_w)
            if x_end - x_start < 10:
                continue
            strip = hand.crop((x_start, 0, x_end, hand_h))
            detections = self.find_cards_in_region(strip)
            # Offset bboxes to hand-relative coords
            for det in detections:
                bx, by, bw, bh = det.bbox
                adjusted = RecognizedCard(
                    card=det.card,
                    confidence=det.confidence,
                    bbox=(bx + x_start + hand_left, by + hand_top, bw, bh),
                )
                all_detections.append(adjusted)

        return non_maximum_suppression(all_detections, self.nms_overlap)

    # -- colour-based suit detection ---------------------------------------

    def classify_suit_colour(
        self,
        card_image: 'Image.Image',
    ) -> Optional[str]:
        """Classify a card corner as 'red' or 'black' by colour."""
        return classify_suit_by_colour(card_image)

    # -- raised card detection ---------------------------------------------

    def detect_raised_cards(
        self,
        screenshot: 'Image.Image',
        detections: List[RecognizedCard],
        offset_px: int = RAISED_OFFSET_PX,
    ) -> List[RecognizedCard]:
        """Filter detections to only raised (selected) cards.

        A card is considered raised if its y-position is lower than
        the expected hand top minus the offset.
        """
        if not detections:
            return []

        _, h = screenshot.size
        hand_top = int(h * self._layout.hand_top)
        raised_y = hand_top - offset_px

        return [
            det for det in detections
            if det.bbox[1] < raised_y
        ]

    # -- full game state recognition ---------------------------------------

    def recognize_hand(
        self,
        screenshot: 'Image.Image',
    ) -> List[Card]:
        """Override: use strip-based scanning for overlapping cards."""
        detections = self.recognize_hand_strips(screenshot)
        seen: set[str] = set()
        cards: List[Card] = []
        for det in sorted(detections, key=lambda d: d.bbox[0]):
            key = det.card.display()
            if key not in seen:
                seen.add(key)
                cards.append(det.card)
        return cards

    def recognize_played_cards(
        self,
        screenshot: 'Image.Image',
    ) -> List[Card]:
        """Override: use game-specific played region."""
        if not HAS_CV2 or not HAS_PIL:
            return []

        w, h = screenshot.size
        ly = self._layout
        left = int(w * ly.played_left)
        right = int(w * ly.played_right)
        top = int(h * ly.played_top)
        bottom = int(h * ly.played_bottom)
        region = screenshot.crop((left, top, right, bottom))
        detections = self.find_cards_in_region(region)
        seen: set[str] = set()
        cards: List[Card] = []
        for det in detections:
            key = det.card.display()
            if key not in seen:
                seen.add(key)
                cards.append(det.card)
        return cards

    def get_screen_regions(self) -> ScreenRegions:
        """Return :class:`ScreenRegions` derived from calibration or layout."""
        if self._calibration and self._calibration.screen_regions:
            return self._calibration.to_screen_regions()
        return self._layout.to_screen_regions()
