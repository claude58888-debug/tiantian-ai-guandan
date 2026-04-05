"""Calibration toolkit for fine-tuning card recognition (M8).

Provides :class:`CalibrationManager` which accepts a real game
screenshot together with manually labelled hand cards and computes
optimal template-matching parameters, screen region ratios, and HSV
colour ranges.  Calibration data can be persisted as JSON via
:class:`CalibrationData`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
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

from guandan.card_recognition import (
    CardRecognizer,
    RecognizedCard,
    non_maximum_suppression,
)
from guandan.game_screen_analyzer import ScreenRegions
from guandan.models import Card, Rank, Suit

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Colour range dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HSVRange:
    """HSV colour range for segmentation."""
    h_low: int = 0
    s_low: int = 0
    v_low: int = 0
    h_high: int = 180
    s_high: int = 255
    v_high: int = 255

    def lower(self) -> Tuple[int, int, int]:
        return (self.h_low, self.s_low, self.v_low)

    def upper(self) -> Tuple[int, int, int]:
        return (self.h_high, self.s_high, self.v_high)

    def to_dict(self) -> Dict[str, int]:
        return {
            'h_low': self.h_low, 's_low': self.s_low, 'v_low': self.v_low,
            'h_high': self.h_high, 's_high': self.s_high, 'v_high': self.v_high,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, int]) -> HSVRange:
        return cls(**d)


# Default HSV ranges for red and black suits on a white card background
DEFAULT_RED_RANGE = HSVRange(h_low=0, s_low=80, v_low=80, h_high=10, s_high=255, v_high=255)
DEFAULT_BLACK_RANGE = HSVRange(h_low=0, s_low=0, v_low=0, h_high=180, s_high=50, v_high=80)


# ---------------------------------------------------------------------------
# CalibrationData
# ---------------------------------------------------------------------------

@dataclass
class CalibrationData:
    """Persisted calibration parameters.

    Attributes
    ----------
    screen_regions : dict
        Flat dict of :class:`ScreenRegions` field values.
    template_scale : float
        Best-performing template scale factor.
    match_threshold : float
        Best-performing match threshold.
    color_ranges : dict
        ``{'red': {...}, 'black': {...}}`` HSV range dicts.
    source_resolution : tuple[int, int]
        Width x height of the screenshot used for calibration.
    """
    screen_regions: Dict[str, float] = field(default_factory=dict)
    template_scale: float = 1.0
    match_threshold: float = 0.8
    color_ranges: Dict[str, Dict[str, int]] = field(default_factory=dict)
    source_resolution: Tuple[int, int] = (1400, 850)

    def to_screen_regions(self) -> ScreenRegions:
        """Convert stored dict back to a :class:`ScreenRegions` instance."""
        if not self.screen_regions:
            return ScreenRegions()
        return ScreenRegions(**self.screen_regions)

    def get_red_range(self) -> HSVRange:
        raw = self.color_ranges.get('red')
        if raw:
            return HSVRange.from_dict(raw)
        return DEFAULT_RED_RANGE

    def get_black_range(self) -> HSVRange:
        raw = self.color_ranges.get('black')
        if raw:
            return HSVRange.from_dict(raw)
        return DEFAULT_BLACK_RANGE

    def to_dict(self) -> Dict[str, object]:
        return {
            'screen_regions': self.screen_regions,
            'template_scale': self.template_scale,
            'match_threshold': self.match_threshold,
            'color_ranges': self.color_ranges,
            'source_resolution': list(self.source_resolution),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> CalibrationData:
        sr = d.get('source_resolution', [1400, 850])
        return cls(
            screen_regions=dict(d.get('screen_regions', {})),  # type: ignore[arg-type]
            template_scale=float(d.get('template_scale', 1.0)),  # type: ignore[arg-type]
            match_threshold=float(d.get('match_threshold', 0.8)),  # type: ignore[arg-type]
            color_ranges=dict(d.get('color_ranges', {})),  # type: ignore[arg-type]
            source_resolution=(int(sr[0]), int(sr[1])),  # type: ignore[index]
        )


# ---------------------------------------------------------------------------
# CalibrationManager
# ---------------------------------------------------------------------------

class CalibrationManager:
    """Calibration toolkit for game-specific recognition tuning.

    Parameters
    ----------
    recognizer : CardRecognizer | None
        Pre-loaded recognizer used for threshold sweeping.
    """

    def __init__(
        self,
        recognizer: Optional[CardRecognizer] = None,
    ) -> None:
        self._recognizer = recognizer or CardRecognizer()
        self._data: CalibrationData = CalibrationData()

    @property
    def data(self) -> CalibrationData:
        return self._data

    # -- calibrate from screenshot -----------------------------------------

    def calibrate_from_screenshot(
        self,
        image: 'Image.Image',
        known_hand: List[Card],
    ) -> CalibrationData:
        """Compute optimal parameters from a screenshot + known hand.

        Sweeps across threshold values and template scales to find
        the combination that maximises the F1 score against the
        known hand labels.
        """
        if not HAS_CV2 or not HAS_PIL:
            return self._data

        w, h = image.size
        self._data.source_resolution = (w, h)

        best_f1 = 0.0
        best_threshold = 0.8
        best_scale = 1.0
        known_set = {c.display() for c in known_hand}

        for threshold in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]:
            for scale in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
                detections = self._recognizer.find_cards_in_region(
                    image, threshold=threshold,
                )
                detected_set = {d.card.display() for d in detections}
                tp = len(known_set & detected_set)
                fp = len(detected_set - known_set)
                fn = len(known_set - detected_set)
                precision = tp / max(tp + fp, 1)
                recall = tp / max(tp + fn, 1)
                f1 = (2 * precision * recall / max(precision + recall, 1e-9))

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
                    best_scale = scale

        self._data.match_threshold = best_threshold
        self._data.template_scale = best_scale
        self._data.color_ranges = {
            'red': DEFAULT_RED_RANGE.to_dict(),
            'black': DEFAULT_BLACK_RANGE.to_dict(),
        }
        return self._data

    # -- auto detect regions -----------------------------------------------

    def auto_detect_regions(
        self,
        screenshot: 'Image.Image',
    ) -> ScreenRegions:
        """Attempt to auto-detect screen region ratios.

        Uses edge density and colour analysis to locate the hand
        region (bottom area with highest card-edge density), the
        played-card region (centre), and the counter region (top-right
        with text).

        Falls back to default regions when detection fails.
        """
        if not HAS_CV2 or not HAS_PIL:
            return ScreenRegions()

        w, h = screenshot.size
        bgr = cv2.cvtColor(
            np.array(screenshot.convert('RGB')), cv2.COLOR_RGB2BGR,
        )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # Scan vertical strips to find hand region (bottom dense edges)
        strip_height = max(1, h // 20)
        densities: List[float] = []
        for row_start in range(0, h, strip_height):
            row_end = min(row_start + strip_height, h)
            strip = edges[row_start:row_end, :]
            density = float(np.count_nonzero(strip)) / max(strip.size, 1)
            densities.append(density)

        # Find the transition point where hand region starts
        # (typically a jump in edge density in the bottom 50%)
        num_strips = len(densities)
        mid_idx = num_strips // 2
        hand_top_ratio = 0.60  # default

        if num_strips > 2:
            max_density = 0.0
            max_idx = mid_idx
            for i in range(mid_idx, num_strips):
                if densities[i] > max_density:
                    max_density = densities[i]
                    max_idx = i
            # Walk backwards to find where density starts rising
            threshold_d = max_density * 0.3
            for i in range(max_idx, mid_idx - 1, -1):
                if densities[i] < threshold_d:
                    hand_top_ratio = (i * strip_height) / h
                    break

        hand_top_ratio = max(0.50, min(0.75, hand_top_ratio))

        regions = ScreenRegions(
            hand_top=hand_top_ratio,
            hand_bottom=1.0,
            hand_left=0.05,
            hand_right=0.95,
        )
        self._data.screen_regions = {
            f.name: getattr(regions, f.name)
            for f in regions.__dataclass_fields__.values()
        }
        return regions

    # -- generate game templates -------------------------------------------

    def generate_game_templates(
        self,
        screenshot: 'Image.Image',
    ) -> Dict[str, 'Image.Image']:
        """Extract card corner images from a real game screenshot.

        Returns a dict of ``{card_name: PIL.Image}`` suitable for
        loading into a :class:`CardRecognizer`.

        The current implementation crops fixed-size regions from the
        bottom hand area.  For best results, pass a screenshot where
        all cards are clearly visible.
        """
        if not HAS_CV2 or not HAS_PIL:
            return {}

        w, h = screenshot.size
        hand_top = int(h * 0.60)
        hand_region = screenshot.crop((0, hand_top, w, h))
        hand_w, hand_h = hand_region.size

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(
            np.array(hand_region.convert('RGB')), cv2.COLOR_RGB2GRAY,
        )
        edges = cv2.Canny(gray, 80, 200)

        # Find vertical edges (card boundaries)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, hand_h // 4))
        vert_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Sum vertical edge pixels column-wise
        col_sum = np.sum(vert_edges, axis=0).astype(float)
        if col_sum.max() > 0:
            col_sum = col_sum / col_sum.max()

        # Find peaks (card left edges)
        threshold = 0.3
        peaks: List[int] = []
        min_gap = 20
        for x in range(len(col_sum)):
            if col_sum[x] > threshold:
                if not peaks or (x - peaks[-1]) >= min_gap:
                    peaks.append(x)

        # Extract corner crops at each peak
        corner_w = min(40, hand_w // 4)
        corner_h = min(60, hand_h)
        templates: Dict[str, 'Image.Image'] = {}
        for i, x in enumerate(peaks):
            x_end = min(x + corner_w, hand_w)
            crop = hand_region.crop((x, 0, x_end, corner_h))
            templates[f'card_{i}'] = crop

        return templates

    # -- persistence -------------------------------------------------------

    def save_calibration(self, path: Path) -> None:
        """Save calibration data as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._data.to_dict(), f, indent=2, ensure_ascii=False)

    def load_calibration(self, path: Path) -> CalibrationData:
        """Load calibration data from a JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        self._data = CalibrationData.from_dict(raw)
        return self._data
