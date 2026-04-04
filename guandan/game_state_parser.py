"""Game state parser for Guandan (Atom 2.3).

Parses visual game state from screenshots by combining screen capture
and card detection. Extracts:
- Current hand cards
- Last played combo on table
- Current level rank
- Turn indicator (whose turn)
- Card counts for other players
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from guandan.models import Card, Rank
from guandan.combos import Combo, classify_combo
from guandan.card_detector import DetectedCard

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class TurnPhase(Enum):
    """Current game phase as detected from screen."""
    UNKNOWN = auto()
    MY_TURN = auto()
    WAITING = auto()
    GAME_OVER = auto()
    BETWEEN_ROUNDS = auto()


@dataclass
class VisualGameState:
    """Parsed game state from a screenshot."""
    my_hand: List[Card] = field(default_factory=list)
    last_played: Optional[Combo] = None
    current_level: Rank = Rank.TWO
    turn_phase: TurnPhase = TurnPhase.UNKNOWN
    player_card_counts: Tuple[int, int, int, int] = (0, 0, 0, 0)
    confidence: float = 0.0
    raw_detections: List[DetectedCard] = field(default_factory=list)

    @property
    def is_my_turn(self) -> bool:
        return self.turn_phase == TurnPhase.MY_TURN

    @property
    def hand_size(self) -> int:
        return len(self.my_hand)

    def __repr__(self) -> str:
        hand_str = ', '.join(c.display() for c in self.my_hand)
        return (
            f'VisualGameState(hand=[{hand_str}], '
            f'level={self.current_level.name}, '
            f'phase={self.turn_phase.name}, '
            f'conf={self.confidence:.2f})'
        )


class GameStateParser:
    """Parses game state from screenshots.
    
    Combines GameCapture (screen_capture.py) and CardDetector
    (card_detector.py) to produce a VisualGameState.
    """

    def __init__(
        self,
        detector=None,
        capturer=None,
        current_level: Rank = Rank.TWO,
    ):
        self._detector = detector
        self._capturer = capturer
        self.current_level = current_level
        self._last_state: Optional[VisualGameState] = None

    def parse_hand_image(
        self, img: 'Image.Image'
    ) -> List[DetectedCard]:
        """Detect cards from a hand region image."""
        if self._detector is None:
            return []
        return self._detector.detect_cards(img)

    def parse_table_image(
        self, img: 'Image.Image'
    ) -> List[DetectedCard]:
        """Detect cards from the played-center region."""
        if self._detector is None:
            return []
        return self._detector.detect_cards(img)

    def detections_to_cards(
        self, detections: List[DetectedCard]
    ) -> List[Card]:
        """Convert detected cards to Card objects."""
        return [d.card for d in detections]

    def detections_to_combo(
        self, detections: List[DetectedCard]
    ) -> Optional[Combo]:
        """Try to classify detected table cards as a combo."""
        if not detections:
            return None
        cards = self.detections_to_cards(detections)
        return classify_combo(cards, self.current_level)

    def detect_turn_phase(
        self, img: 'Image.Image'
    ) -> TurnPhase:
        """Detect current turn phase from UI indicators.
        
        Looks for visual cues like:
        - Play/Pass buttons visible = MY_TURN
        - Waiting indicator = WAITING
        - Score screen = GAME_OVER
        
        Currently returns UNKNOWN as a placeholder.
        Real implementation requires template matching on UI elements.
        """
        # Placeholder - needs real UI button detection
        return TurnPhase.UNKNOWN

    def parse_screenshot(
        self, screenshot: 'Image.Image'
    ) -> VisualGameState:
        """Parse a full game screenshot into a VisualGameState.
        
        This is the main entry point. Takes a full game window
        screenshot and extracts all game state information.
        """
        state = VisualGameState(current_level=self.current_level)
        
        if not HAS_PIL:
            return state
        
        # Detect hand cards (bottom region)
        # In real use, we'd crop to the hand region first
        hand_detections = self.parse_hand_image(screenshot)
        state.my_hand = self.detections_to_cards(hand_detections)
        state.raw_detections = hand_detections
        
        # Detect played cards (center region)
        table_detections = self.parse_table_image(screenshot)
        state.last_played = self.detections_to_combo(table_detections)
        
        # Detect turn phase
        state.turn_phase = self.detect_turn_phase(screenshot)
        
        # Compute average confidence
        all_dets = hand_detections + table_detections
        if all_dets:
            state.confidence = sum(d.confidence for d in all_dets) / len(all_dets)
        
        self._last_state = state
        return state

    def parse_live(self) -> Optional[VisualGameState]:
        """Capture and parse from live game window.
        
        Uses the capturer to grab a screenshot, then parses it.
        Returns None if capture fails.
        """
        if self._capturer is None:
            return None
        
        screenshot = self._capturer.capture()
        if screenshot is None:
            return None
        
        return self.parse_screenshot(screenshot)

    @property
    def last_state(self) -> Optional[VisualGameState]:
        """Last parsed game state."""
        return self._last_state


def create_parser(
    current_level: Rank = Rank.TWO,
    template_dir: Optional[str] = None,
    window_title: Optional[str] = None,
) -> GameStateParser:
    """Factory function to create a configured GameStateParser.
    
    Attempts to initialize detector and capturer with sensible defaults.
    Falls back gracefully if dependencies are missing.
    """
    detector = None
    capturer = None
    
    try:
        from guandan.card_detector import CardDetector
        detector = CardDetector()
    except (ImportError, Exception):
        pass
    
    try:
        from guandan.screen_capture import GameCapture
        capturer = GameCapture(custom_title=window_title)
    except (ImportError, Exception):
        pass
    
    return GameStateParser(
        detector=detector,
        capturer=capturer,
        current_level=current_level,
    )
