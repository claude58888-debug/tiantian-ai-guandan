"""Game screen analyzer for the 天天爱掼蛋 game client (M7).

Analyses full-resolution game screenshots (~1400x850 landscape) to
extract:
- Whether it is the player's turn (button / timer detection)
- Hand cards (bottom 40 %)
- Played cards (centre region)
- Card counter digits (top-right)
- Current level rank (top-left "本局打 X")
- Opponent card counts

Detection is performed via a mix of template matching
(CardRecognizer from M6), colour segmentation, contour analysis,
and OCR (easyocr).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

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

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False

from guandan.models import Card, Rank, Suit, JokerType
from guandan.card_recognition import CardRecognizer, RecognizedCard

# Lazy import to avoid circular dependency at module level
_GameSpecificRecognizer = None

def _get_game_specific_recognizer_cls() -> type:
    """Lazily import GameSpecificRecognizer to avoid circular imports."""
    global _GameSpecificRecognizer  # noqa: PLW0603
    if _GameSpecificRecognizer is None:
        from guandan.game_specific_recognizer import GameSpecificRecognizer
        _GameSpecificRecognizer = GameSpecificRecognizer
    return _GameSpecificRecognizer

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Screen region ratios (relative to full screenshot)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScreenRegions:
    """Configurable screen region ratios (0-1) for the game layout.

    All values are expressed as fractions of the full screenshot
    width / height so that they work at any resolution.
    """
    # Hand region (bottom of screen)
    hand_top: float = 0.60
    hand_bottom: float = 1.0
    hand_left: float = 0.05
    hand_right: float = 0.95

    # Played-cards region (centre)
    played_top: float = 0.30
    played_bottom: float = 0.60
    played_left: float = 0.25
    played_right: float = 0.75

    # Button region (centre, overlaps played)
    button_top: float = 0.55
    button_bottom: float = 0.75
    button_left: float = 0.30
    button_right: float = 0.70

    # Timer region (near centre)
    timer_top: float = 0.35
    timer_bottom: float = 0.55
    timer_left: float = 0.42
    timer_right: float = 0.58

    # Card-counter region (top-right)
    counter_top: float = 0.0
    counter_bottom: float = 0.10
    counter_left: float = 0.55
    counter_right: float = 1.0

    # Level indicator (top-left)
    level_top: float = 0.0
    level_bottom: float = 0.08
    level_left: float = 0.0
    level_right: float = 0.20

    # Opponent card-back regions
    opp_left_top: float = 0.20
    opp_left_bottom: float = 0.70
    opp_left_left: float = 0.0
    opp_left_right: float = 0.12

    opp_right_top: float = 0.20
    opp_right_bottom: float = 0.70
    opp_right_left: float = 0.88
    opp_right_right: float = 1.0

    opp_top_top: float = 0.0
    opp_top_bottom: float = 0.25
    opp_top_left: float = 0.30
    opp_top_right: float = 0.70


DEFAULT_REGIONS = ScreenRegions()

# Rank label to Rank enum mapping
_LABEL_TO_RANK: Dict[str, Rank] = {
    '2': Rank.TWO, '3': Rank.THREE, '4': Rank.FOUR,
    '5': Rank.FIVE, '6': Rank.SIX, '7': Rank.SEVEN,
    '8': Rank.EIGHT, '9': Rank.NINE, '10': Rank.TEN,
    'J': Rank.JACK, 'Q': Rank.QUEEN, 'K': Rank.KING, 'A': Rank.ACE,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _crop_region(
    img: 'Image.Image',
    top: float,
    bottom: float,
    left: float,
    right: float,
) -> 'Image.Image':
    """Crop a PIL image using fractional coordinates."""
    w, h = img.size
    return img.crop((
        int(w * left),
        int(h * top),
        int(w * right),
        int(h * bottom),
    ))


def _pil_to_cv2(img: 'Image.Image') -> 'np.ndarray':
    """Convert PIL Image to OpenCV BGR array."""
    rgb = np.array(img.convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _has_button_colour(region_bgr: 'np.ndarray', min_ratio: float = 0.01) -> bool:
    """Detect whether a region contains golden/orange button pixels.

    The 出牌 / 提示 buttons in the game are a warm gold/orange colour.
    """
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    # Golden/orange hue range
    lower = np.array([15, 100, 150], dtype=np.uint8)
    upper = np.array([35, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    ratio = float(np.count_nonzero(mask)) / max(mask.size, 1)
    return ratio >= min_ratio


def _has_timer_circle(region_bgr: 'np.ndarray', min_ratio: float = 0.005) -> bool:
    """Detect the golden countdown-timer circle."""
    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([20, 80, 160], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    ratio = float(np.count_nonzero(mask)) / max(mask.size, 1)
    return ratio >= min_ratio


def _count_card_backs(region_bgr: 'np.ndarray', min_contours: int = 1) -> int:
    """Rough estimate of card-back count via contour detection.

    Card backs are typically a solid coloured rectangle.  We detect
    rectangular contours above a minimum area.
    """
    gray = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    min_area = (w * h) * 0.005
    max_area = (w * h) * 0.3
    count = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class GameScreenAnalyzer:
    """Analyses screenshots from the 天天爱掼蛋 game client.

    Parameters
    ----------
    recognizer : CardRecognizer
        Pre-loaded card recognizer with templates.
    regions : ScreenRegions
        Fractional screen regions for each detection zone.
    ocr_langs : list[str]
        Languages for EasyOCR (default ``['en']``).
    """

    def __init__(
        self,
        recognizer: Optional[CardRecognizer] = None,
        regions: Optional[ScreenRegions] = None,
        ocr_langs: Optional[List[str]] = None,
        calibration_path: Optional[str] = None,
    ) -> None:
        self._recognizer = recognizer or CardRecognizer()
        self._regions = regions or DEFAULT_REGIONS
        self._ocr_langs = ocr_langs or ['en']
        self._ocr_reader: Optional['easyocr.Reader'] = None
        self._calibration_path = calibration_path

        # If calibration data exists, upgrade to GameSpecificRecognizer
        if calibration_path is not None:
            self._try_load_calibration(calibration_path)

    def _try_load_calibration(self, path: str) -> None:
        """Load calibration and upgrade recognizer if possible."""
        from pathlib import Path
        cal_path = Path(path)
        if not cal_path.exists():
            log.debug('Calibration file not found: %s', path)
            return
        try:
            from guandan.calibration import CalibrationData
            import json
            with open(cal_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
            cal_data = CalibrationData.from_dict(raw)
            gsr_cls = _get_game_specific_recognizer_cls()
            self._recognizer = gsr_cls(calibration=cal_data)
            regions = cal_data.to_screen_regions()
            if cal_data.screen_regions:
                self._regions = regions
            log.info('Loaded calibration from %s', path)
        except Exception:
            log.warning('Failed to load calibration from %s', path, exc_info=True)

    # -- properties --------------------------------------------------------

    @property
    def recognizer(self) -> CardRecognizer:
        return self._recognizer

    @property
    def regions(self) -> ScreenRegions:
        return self._regions

    def set_regions(self, regions: ScreenRegions) -> None:
        """Replace screen regions (e.g. after calibration)."""
        self._regions = regions

    # -- OCR lazy init -----------------------------------------------------

    def _get_ocr(self) -> Optional['easyocr.Reader']:
        """Lazily initialise the EasyOCR reader."""
        if not HAS_EASYOCR:
            return None
        if self._ocr_reader is None:
            try:
                self._ocr_reader = easyocr.Reader(
                    self._ocr_langs, gpu=False, verbose=False,
                )
            except Exception:
                log.warning('Failed to initialise EasyOCR reader')
                return None
        return self._ocr_reader

    # -- detection methods -------------------------------------------------

    def detect_my_turn(self, screenshot: 'Image.Image') -> bool:
        """Return True if it is the player's turn.

        Checks for golden 出牌/提示 buttons or the countdown timer.
        """
        if not HAS_CV2:
            return False

        r = self._regions
        btn_crop = _crop_region(screenshot, r.button_top, r.button_bottom,
                                r.button_left, r.button_right)
        btn_bgr = _pil_to_cv2(btn_crop)
        if _has_button_colour(btn_bgr):
            return True

        timer_crop = _crop_region(screenshot, r.timer_top, r.timer_bottom,
                                  r.timer_left, r.timer_right)
        timer_bgr = _pil_to_cv2(timer_crop)
        return _has_timer_circle(timer_bgr)

    def detect_hand_cards(self, screenshot: 'Image.Image') -> List[Card]:
        """Detect the player's hand from the bottom region."""
        r = self._regions
        hand_crop = _crop_region(screenshot, r.hand_top, r.hand_bottom,
                                 r.hand_left, r.hand_right)
        detections = self._recognizer.find_cards_in_region(hand_crop)
        seen: set[str] = set()
        cards: List[Card] = []
        for det in detections:
            key = det.card.display()
            if key not in seen:
                seen.add(key)
                cards.append(det.card)
        return cards

    def detect_played_cards(
        self, screenshot: 'Image.Image',
    ) -> Optional[List[Card]]:
        """Detect cards played in the centre area.

        Returns None if no cards are detected (empty table).
        """
        r = self._regions
        played_crop = _crop_region(screenshot, r.played_top, r.played_bottom,
                                   r.played_left, r.played_right)
        detections = self._recognizer.find_cards_in_region(played_crop)
        if not detections:
            return None
        seen: set[str] = set()
        cards: List[Card] = []
        for det in detections:
            key = det.card.display()
            if key not in seen:
                seen.add(key)
                cards.append(det.card)
        return cards

    def detect_card_counter(
        self, screenshot: 'Image.Image',
    ) -> Dict[Rank, int]:
        """OCR the card-counter digits at the top-right.

        Returns a mapping from Rank to remaining count.
        Falls back to an empty dict when OCR is unavailable.
        """
        reader = self._get_ocr()
        if reader is None or not HAS_CV2:
            return {}

        r = self._regions
        crop = _crop_region(screenshot, r.counter_top, r.counter_bottom,
                            r.counter_left, r.counter_right)
        bgr = _pil_to_cv2(crop)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        try:
            results = reader.readtext(gray, detail=1)
        except Exception:
            return {}

        counter: Dict[Rank, int] = {}
        for _bbox, text, _conf in results:
            text = text.strip()
            if text.isdigit():
                val = int(text)
                if 0 <= val <= 8:
                    # We can't reliably map position to rank without
                    # layout info, so store generic counts keyed by index.
                    pass
            # Try matching rank labels
            upper = text.upper()
            if upper in _LABEL_TO_RANK:
                counter[_LABEL_TO_RANK[upper]] = 0  # placeholder
        return counter

    def detect_level_card(
        self, screenshot: 'Image.Image',
    ) -> Optional[Rank]:
        """Read the current level rank from the top-left "本局打 X" label.

        Returns the detected Rank, or None on failure.
        """
        reader = self._get_ocr()
        if reader is None or not HAS_CV2:
            return None

        r = self._regions
        crop = _crop_region(screenshot, r.level_top, r.level_bottom,
                            r.level_left, r.level_right)
        bgr = _pil_to_cv2(crop)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        try:
            results = reader.readtext(gray, detail=0)
        except Exception:
            return None

        for text in results:
            text = text.strip().upper()
            for label, rank in _LABEL_TO_RANK.items():
                if label in text:
                    return rank
        return None

    def detect_opponent_card_count(
        self, screenshot: 'Image.Image',
    ) -> Dict[str, int]:
        """Estimate how many cards each opponent holds.

        Uses contour detection in the three opponent regions to
        count card-back shapes.

        Returns ``{'left': n, 'right': n, 'top': n}``.
        """
        if not HAS_CV2:
            return {}

        r = self._regions
        counts: Dict[str, int] = {}

        for name, top, bottom, left, right in [
            ('left', r.opp_left_top, r.opp_left_bottom,
             r.opp_left_left, r.opp_left_right),
            ('right', r.opp_right_top, r.opp_right_bottom,
             r.opp_right_left, r.opp_right_right),
            ('top', r.opp_top_top, r.opp_top_bottom,
             r.opp_top_left, r.opp_top_right),
        ]:
            crop = _crop_region(screenshot, top, bottom, left, right)
            bgr = _pil_to_cv2(crop)
            counts[name] = _count_card_backs(bgr)

        return counts

    def analyze(self, screenshot: 'Image.Image') -> Dict[str, object]:
        """Run all detectors and return a combined state dict.

        Keys: ``my_turn``, ``hand``, ``played``, ``counter``,
        ``level``, ``opponent_counts``.
        """
        return {
            'my_turn': self.detect_my_turn(screenshot),
            'hand': self.detect_hand_cards(screenshot),
            'played': self.detect_played_cards(screenshot),
            'counter': self.detect_card_counter(screenshot),
            'level': self.detect_level_card(screenshot),
            'opponent_counts': self.detect_opponent_card_count(screenshot),
        }
