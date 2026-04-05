"""Automatic game mode detection for Tiantian Ai Guandan (天天爱掼蛋).

Analyses screenshots of the game lobby/room to determine which mode
the player has selected. Uses OCR and template matching on UI elements.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, List, Tuple

from guandan.mode_strategy import GameMode

log = logging.getLogger(__name__)

# Keywords that appear on-screen for each mode
_MODE_KEYWORDS: dict[GameMode, List[str]] = {
    GameMode.CLASSIC: ["经典", "经典模式", "传统"],
    GameMode.NO_SHUFFLE: ["不洗牌", "翻倍", "不洗牌翻倍"],
    GameMode.TEAM_2V2: ["2v2", "组队", "2V2", "二人"],
    GameMode.EIGHT_HEARTS: ["八红桃", "红桃", "8红桃"],
    GameMode.RANKED: ["排位", "竞技", "排位竞技", "排位赛"],
    GameMode.LEAGUE: ["联赛", "苏掼", "苏掼联赛", "比赛"],
}

# Priority order: more specific modes checked first
_DETECTION_ORDER: List[GameMode] = [
    GameMode.NO_SHUFFLE,
    GameMode.EIGHT_HEARTS,
    GameMode.LEAGUE,
    GameMode.RANKED,
    GameMode.TEAM_2V2,
    GameMode.CLASSIC,
]


@dataclass
class DetectionResult:
    """Result of mode detection."""
    mode: GameMode
    confidence: float  # 0.0-1.0
    matched_keywords: List[str]
    source: str  # "ocr", "template", "title_bar", "fallback"


def detect_mode_from_text(text: str) -> DetectionResult:
    """Detect game mode from OCR text extracted from the game screen.

    Args:
        text: Raw OCR text from the game lobby or room screen.

    Returns:
        DetectionResult with the detected mode and confidence.
    """
    if not text:
        return DetectionResult(
            mode=GameMode.CLASSIC,
            confidence=0.1,
            matched_keywords=[],
            source="fallback",
        )

    text_clean = text.replace(" ", "").lower()
    best_mode = GameMode.CLASSIC
    best_score = 0.0
    best_keywords: List[str] = []

    for mode in _DETECTION_ORDER:
        keywords = _MODE_KEYWORDS[mode]
        matched = [kw for kw in keywords if kw.lower() in text_clean]
        if matched:
            # Score based on number and specificity of matches
            score = len(matched) / len(keywords)
            # Bonus for longer keyword matches (more specific)
            avg_len = sum(len(kw) for kw in matched) / len(matched)
            score = min(1.0, score + avg_len * 0.05)
            if score > best_score:
                best_score = score
                best_mode = mode
                best_keywords = matched

    confidence = min(1.0, best_score) if best_keywords else 0.2
    source = "ocr" if best_keywords else "fallback"

    log.info(
        "Mode detected: %s (confidence=%.2f, keywords=%s)",
        best_mode.value, confidence, best_keywords,
    )

    return DetectionResult(
        mode=best_mode,
        confidence=confidence,
        matched_keywords=best_keywords,
        source=source,
    )


def detect_mode_from_window_title(title: str) -> DetectionResult:
    """Detect mode from the game window title text.

    Some modes change the window title or room name.
    """
    if not title:
        return DetectionResult(
            mode=GameMode.CLASSIC,
            confidence=0.1,
            matched_keywords=[],
            source="fallback",
        )

    for mode in _DETECTION_ORDER:
        keywords = _MODE_KEYWORDS[mode]
        matched = [kw for kw in keywords if kw in title]
        if matched:
            return DetectionResult(
                mode=mode,
                confidence=0.8,
                matched_keywords=matched,
                source="title_bar",
            )

    return DetectionResult(
        mode=GameMode.CLASSIC,
        confidence=0.3,
        matched_keywords=[],
        source="title_bar",
    )


def detect_mode_from_screenshot(image) -> DetectionResult:
    """Detect game mode from a screenshot image.

    Uses OCR to extract text from the lobby/room UI, then
    matches against known mode keywords.

    Args:
        image: numpy array (BGR) or PIL Image of the game screen.

    Returns:
        DetectionResult with detected mode.
    """
    try:
        import easyocr
    except ImportError:
        log.warning("easyocr not installed, falling back to CLASSIC mode")
        return DetectionResult(
            mode=GameMode.CLASSIC,
            confidence=0.1,
            matched_keywords=[],
            source="fallback",
        )

    try:
        reader = easyocr.Reader(["ch_sim", "en"], gpu=False)
        results = reader.readtext(image, detail=0)
        full_text = " ".join(results)
        return detect_mode_from_text(full_text)
    except Exception as e:
        log.error("OCR failed: %s", e)
        return DetectionResult(
            mode=GameMode.CLASSIC,
            confidence=0.1,
            matched_keywords=[],
            source="fallback",
        )


class ModeTracker:
    """Tracks the current game mode across frames.

    Uses a rolling window to stabilize mode detection and
    avoid flickering between modes.
    """

    def __init__(self, stability_threshold: int = 3):
        self._history: List[DetectionResult] = []
        self._stability_threshold = stability_threshold
        self._current_mode: Optional[GameMode] = None

    @property
    def current_mode(self) -> GameMode:
        """Return the current stable mode."""
        return self._current_mode or GameMode.CLASSIC

    def update(self, result: DetectionResult) -> GameMode:
        """Update with a new detection result.

        Returns the stable mode (may lag behind actual detection).
        """
        self._history.append(result)
        if len(self._history) > 10:
            self._history = self._history[-10:]

        # Check if recent results agree
        recent = self._history[-self._stability_threshold:]
        if len(recent) >= self._stability_threshold:
            modes = [r.mode for r in recent]
            if all(m == modes[0] for m in modes):
                if self._current_mode != modes[0]:
                    log.info(
                        "Mode changed: %s -> %s",
                        self._current_mode, modes[0].value,
                    )
                self._current_mode = modes[0]

        return self.current_mode

    def reset(self) -> None:
        """Reset detection history."""
        self._history.clear()
        self._current_mode = None
