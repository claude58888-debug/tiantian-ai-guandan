"""Real-time AI decision engine for Guandan (M7).

Integrates:
- GameScreenAnalyzer  (screen -> state)
- CardRecognizer      (template matching)
- SuggestionEngine    (strategy + card_counter -> top-N plays)
- CardCounter         (risk assessment)

to produce a single :class:`Decision` for the current game frame.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from guandan.card_counter import CardCounter
from guandan.card_recognition import CardRecognizer
from guandan.combos import Combo, ComboType, classify_combo
from guandan.game_screen_analyzer import GameScreenAnalyzer, ScreenRegions
from guandan.models import Card, Rank
from guandan.suggestion import (
    PlaySuggestion,
    SuggestionEngine,
    SuggestionRisk,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """A single AI decision for the current game state.

    Attributes
    ----------
    cards_to_play : tuple[Card, ...]
        Cards to play (empty tuple means PASS).
    combo_type : str
        Human-readable combo type name.
    confidence : float
        Confidence score 0.0-1.0.
    reasoning : str
        Short explanation.
    alternatives : tuple[PlaySuggestion, ...]
        Other top-N suggestions the engine considered.
    latency_ms : float
        Time taken to produce this decision (milliseconds).
    """

    cards_to_play: Tuple[Card, ...] = ()
    combo_type: str = 'pass'
    confidence: float = 0.0
    reasoning: str = ''
    alternatives: Tuple[PlaySuggestion, ...] = ()
    latency_ms: float = 0.0

    @property
    def is_pass(self) -> bool:
        return len(self.cards_to_play) == 0

    def display(self) -> str:
        cards_str = ' '.join(c.display() for c in self.cards_to_play) or 'PASS'
        return (
            f'[{self.combo_type}] {cards_str} '
            f'(conf={self.confidence:.2f}, {self.latency_ms:.0f}ms)\n'
            f'  {self.reasoning}'
        )


# Sentinel returned when it is not the player's turn.
NOT_MY_TURN = Decision(
    cards_to_play=(),
    combo_type='wait',
    confidence=1.0,
    reasoning='Not my turn',
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _generate_templates_in_memory():
    """Generate all 54 card templates as PIL images in memory."""
    from guandan.card_template_generator import get_all_specs, render_template
    specs = get_all_specs()
    templates = {}
    for spec in specs:
        img = render_template(spec)
        templates[spec.filename] = img
    return templates


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """Produces play decisions from raw screenshots.

    Parameters
    ----------
    recognizer : CardRecognizer | None
        Card recognizer with loaded templates.
    current_level : Rank
        Current game level rank (default TWO).
    regions : ScreenRegions | None
        Custom screen region ratios.
    top_n : int
        Number of suggestions to request from :class:`SuggestionEngine`.
    """

    def __init__(
        self,
        recognizer: Optional[CardRecognizer] = None,
        current_level: Rank = Rank.TWO,
        regions: Optional[ScreenRegions] = None,
        top_n: int = 3,
    ) -> None:
        self._recognizer = recognizer or CardRecognizer()

        # Auto-load synthetic templates if recognizer has none loaded
        if not self._recognizer.is_loaded:
            templates = _generate_templates_in_memory()
            self._recognizer.load_templates_from_pil(templates)
            log.info('Auto-loaded %d card templates', self._recognizer.template_count)

        self._analyzer = GameScreenAnalyzer(
            recognizer=self._recognizer,
            regions=regions,
        )
        self._counter = CardCounter(current_level=current_level)
        self._suggestion = SuggestionEngine(
            current_level=current_level,
            counter=self._counter,
            top_n=top_n,
        )
        self._current_level = current_level
        self._top_n = top_n

        # Tracking state across frames
        self._last_hand: List[Card] = []
        self._last_played: Optional[List[Card]] = None
        self._last_combo: Optional[Combo] = None

    # -- properties --------------------------------------------------------

    @property
    def analyzer(self) -> GameScreenAnalyzer:
        return self._analyzer

    @property
    def counter(self) -> CardCounter:
        return self._counter

    @property
    def current_level(self) -> Rank:
        return self._current_level

    def set_level(self, level: Rank) -> None:
        """Update the current game level rank."""
        self._current_level = level
        self._counter = CardCounter(current_level=level)
        self._suggestion = SuggestionEngine(
            current_level=level,
            counter=self._counter,
            top_n=self._top_n,
        )

    # -- main entry --------------------------------------------------------

    def decide(self, screenshot: 'Image.Image') -> Decision:
        """Analyse a screenshot and return a play decision.

        Returns :data:`NOT_MY_TURN` when buttons/timer are not detected.
        """
        t0 = time.monotonic()

        # 1. Check if it's our turn
        my_turn = self._analyzer.detect_my_turn(screenshot)
        if not my_turn:
            return NOT_MY_TURN

        # 2. Detect hand cards
        hand = self._analyzer.detect_hand_cards(screenshot)
        if not hand:
            return Decision(
                reasoning='Could not detect hand cards',
                latency_ms=_elapsed_ms(t0),
            )

        # 3. Detect played cards (to know if we lead or respond)
        played = self._analyzer.detect_played_cards(screenshot)

        # 4. Update tracking
        self._update_tracking(hand, played)

        # 5. Update counter with hand
        self._counter.set_hand(hand)

        # 6. Build last_play combo (None means we lead)
        last_combo = self._last_combo

        # 7. Get suggestions
        suggestions = self._suggestion.suggest(hand, last_combo)

        if not suggestions:
            return Decision(
                reasoning='No valid plays available',
                latency_ms=_elapsed_ms(t0),
            )

        best = suggestions[0]
        return Decision(
            cards_to_play=best.cards,
            combo_type=best.combo_type,
            confidence=best.confidence,
            reasoning=best.reason,
            alternatives=tuple(suggestions[1:]),
            latency_ms=_elapsed_ms(t0),
        )

    def decide_for_tribute(self, screenshot: 'Image.Image') -> Decision:
        """Specialised decision for the tribute phase.

        During tribute, the player must give a specific card to another
        player.  This returns a simple heuristic: give the lowest
        non-bomb single.
        """
        t0 = time.monotonic()
        hand = self._analyzer.detect_hand_cards(screenshot)
        if not hand:
            return Decision(
                reasoning='Could not detect hand for tribute',
                latency_ms=_elapsed_ms(t0),
            )
        # Pick lowest non-joker card
        non_jokers = [c for c in hand if not c.is_joker]
        target = non_jokers[0] if non_jokers else hand[0]
        return Decision(
            cards_to_play=(target,),
            combo_type='tribute',
            confidence=0.7,
            reasoning=f'Tribute lowest card: {target.display()}',
            latency_ms=_elapsed_ms(t0),
        )

    # -- tracking ----------------------------------------------------------

    def _update_tracking(
        self,
        hand: List[Card],
        played: Optional[List[Card]],
    ) -> None:
        """Track state changes across frames."""
        # Record played cards to counter
        if played and played != self._last_played:
            self._counter.record_play(played)
            combo = classify_combo(played, self._current_level)
            if combo and combo.combo_type != ComboType.PASS:
                self._last_combo = combo
            else:
                self._last_combo = None

        # Centre cleared -> new trick, we lead
        if not played and self._last_played:
            self._last_combo = None

        self._last_hand = hand
        self._last_played = played

    def reset(self) -> None:
        """Reset all tracking state for a new round."""
        self._counter.reset()
        self._last_hand = []
        self._last_played = None
        self._last_combo = None


def _elapsed_ms(t0: float) -> float:
    return (time.monotonic() - t0) * 1000.0
