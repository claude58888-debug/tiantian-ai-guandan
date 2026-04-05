"""Suggestion engine integrating strategy + card_counter (Atom 5.1).

Produces Top-3 play suggestions with explanations by combining:
- Strategy module: enumerate candidate plays
- CardCounter: risk assessment and bomb probability

Each suggestion contains the cards to play, a short reason,
a risk level tag, and a confidence score (0.0-1.0).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from guandan.card_counter import CardCounter, RiskLevel, RiskReport
from guandan.combos import Combo, ComboType, classify_combo
from guandan.models import Card, Rank
from guandan.strategy import (
    enumerate_plays,
    find_beating_plays,
)


class SuggestionRisk(Enum):
    """Risk level attached to a single suggestion."""
    SAFE = auto()
    MODERATE = auto()
    RISKY = auto()


@dataclass(frozen=True)
class PlaySuggestion:
    """A single play suggestion with explanation.

    Attributes:
        cards: The cards to play (empty list means PASS).
        combo_type: Human-readable combo type name.
        reason: Short explanation of why this play is suggested.
        risk: Risk level for this specific play.
        confidence: Confidence score 0.0-1.0.
    """
    cards: Tuple[Card, ...]
    combo_type: str
    reason: str
    risk: SuggestionRisk
    confidence: float

    @property
    def is_pass(self) -> bool:
        """Whether this suggestion is a PASS."""
        return len(self.cards) == 0

    def display_cards(self) -> str:
        """Format cards for display."""
        if self.is_pass:
            return "PASS"
        return " ".join(c.display() for c in self.cards)

    def display(self) -> str:
        """Full display string for this suggestion."""
        cards_str = self.display_cards()
        return (
            f"[{self.combo_type}] {cards_str} "
            f"(confidence={self.confidence:.2f}, risk={self.risk.name})\n"
            f"  reason: {self.reason}"
        )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

_COMBO_TYPE_NAMES: Dict[ComboType, str] = {
    ComboType.PASS: "pass",
    ComboType.SINGLE: "single",
    ComboType.PAIR: "pair",
    ComboType.TRIPLE: "triple",
    ComboType.FULL_HOUSE: "full_house",
    ComboType.STRAIGHT: "straight",
    ComboType.CONSECUTIVE_PAIRS: "consecutive_pairs",
    ComboType.PLATE: "plate",
    ComboType.BOMB_4: "bomb_4",
    ComboType.BOMB_5: "bomb_5",
    ComboType.BOMB_6: "bomb_6",
    ComboType.BOMB_7: "bomb_7",
    ComboType.BOMB_8: "bomb_8",
    ComboType.JOKER_BOMB: "joker_bomb",
}


def _combo_type_name(ct: ComboType) -> str:
    """Return a readable name for a combo type."""
    return _COMBO_TYPE_NAMES.get(ct, ct.name.lower())


def _base_score(combo: Combo, hand_size: int) -> float:
    """Compute a base desirability score for a combo play.

    Heuristics:
    - Playing more cards at once is generally better (reduces hand faster).
    - Lower rank_key plays are preferred for non-bombs (save strong cards).
    - Bombs get a penalty unless the hand is small.
    """
    size_bonus = combo.size / max(hand_size, 1) * 0.3

    # Rank penalty: higher rank_key -> lower score (save big cards)
    # Normalize rank_key roughly to 0-1 range (rank values 2-14, jokers ~100-200)
    if combo.rank_key >= 100:
        rank_penalty = 0.4  # joker plays are costly
    else:
        rank_penalty = (combo.rank_key - 2) / 14.0 * 0.3

    base = 0.6 + size_bonus - rank_penalty

    # Bombs: penalize unless hand is very small
    if combo.is_bomb:
        if hand_size <= 6:
            base += 0.1  # worth using bomb to finish
        else:
            base -= 0.25  # save bombs

    return max(0.05, min(1.0, base))


def _risk_for_play(
    combo: Combo,
    risk_report: RiskReport,
    hand_size: int,
) -> SuggestionRisk:
    """Determine the risk level of a specific play given game context."""
    # Using a bomb is risky if opponents might have bigger bombs
    if combo.is_bomb and risk_report.bomb_threat > 0.5:
        return SuggestionRisk.RISKY

    # Playing high cards when opponents have high bomb threat
    if combo.rank_key >= 13 and risk_report.bomb_threat > 0.6:
        return SuggestionRisk.MODERATE

    # Overall game risk is high
    if risk_report.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        if combo.rank_key >= 11:
            return SuggestionRisk.MODERATE

    return SuggestionRisk.SAFE


def _generate_reason(
    combo: Combo,
    is_lead: bool,
    hand_size: int,
    risk_report: RiskReport,
) -> str:
    """Generate a short human-readable reason for this suggestion."""
    ct_name = _combo_type_name(combo.combo_type)

    if combo.is_bomb:
        if hand_size <= combo.size:
            return "Use bomb to finish the round"
        if risk_report.bomb_threat < 0.3:
            return "Low opponent bomb threat, safe to use bomb"
        return "Bomb play, but opponents may have stronger bombs"

    if is_lead:
        if combo.rank_key <= 5:
            return f"Lead with low {ct_name} to probe opponents"
        if combo.rank_key >= 13:
            return f"Lead with strong {ct_name} to seize control"
        return f"Lead with mid-range {ct_name}"

    # Response
    if combo.rank_key <= 6:
        return f"Low-cost {ct_name} response, preserves strong cards"
    if combo.rank_key >= 13:
        return f"Strong {ct_name} to win the trick"
    return f"Beat with {ct_name}, balanced cost"


def _generate_pass_reason(risk_report: RiskReport, hand_size: int) -> str:
    """Generate a reason for passing."""
    if risk_report.bomb_threat > 0.5:
        return "Pass to avoid bomb risk, let partner take over"
    if hand_size > 15:
        return "Pass and conserve, many cards remaining"
    return "Pass this round, wait for better opportunity"


def _pass_confidence(risk_report: RiskReport, hand_size: int) -> float:
    """Confidence score for a PASS suggestion."""
    # Passing is more attractive when risk is high
    base = 0.30
    if risk_report.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
        base += 0.15
    if risk_report.bomb_threat > 0.6:
        base += 0.10
    return min(0.60, base)


def _pass_risk(risk_report: RiskReport) -> SuggestionRisk:
    """Risk level for passing."""
    if risk_report.risk_level == RiskLevel.CRITICAL:
        return SuggestionRisk.RISKY  # passing when critical is risky too
    return SuggestionRisk.SAFE


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class SuggestionEngine:
    """Integrates strategy and card counting to produce Top-3 suggestions.

    Usage:
        engine = SuggestionEngine(current_level=Rank.TWO)
        engine.update_counter(counter)  # or build internally
        suggestions = engine.suggest(hand_cards, last_play)
    """

    def __init__(
        self,
        current_level: Rank = Rank.TWO,
        counter: Optional[CardCounter] = None,
        top_n: int = 3,
    ) -> None:
        self.current_level = current_level
        self.counter = counter or CardCounter(current_level=current_level)
        self.top_n = top_n

    def update_counter(self, counter: CardCounter) -> None:
        """Replace the card counter with an updated one."""
        self.counter = counter

    def suggest(
        self,
        hand: List[Card],
        last_play: Optional[Combo] = None,
    ) -> List[PlaySuggestion]:
        """Produce up to top_n play suggestions.

        Args:
            hand: Current hand cards.
            last_play: The last combo played (None or PASS means we lead).

        Returns:
            List of PlaySuggestion sorted by confidence descending.
        """
        if not hand:
            return []

        is_lead = last_play is None or last_play.combo_type == ComboType.PASS
        risk_report = self.counter.assess_risk()

        if is_lead:
            candidates = self._lead_candidates(hand)
        else:
            candidates = self._response_candidates(hand, last_play)

        # Score and build suggestions
        scored: List[Tuple[float, PlaySuggestion]] = []
        for combo in candidates:
            score = _base_score(combo, len(hand))
            risk = _risk_for_play(combo, risk_report, len(hand))
            reason = _generate_reason(combo, is_lead, len(hand), risk_report)

            # Adjust score by risk
            if risk == SuggestionRisk.RISKY:
                score *= 0.75
            elif risk == SuggestionRisk.MODERATE:
                score *= 0.90

            confidence = max(0.05, min(0.99, score))
            suggestion = PlaySuggestion(
                cards=combo.cards,
                combo_type=_combo_type_name(combo.combo_type),
                reason=reason,
                risk=risk,
                confidence=round(confidence, 2),
            )
            scored.append((confidence, suggestion))

        # Deduplicate by cards tuple (keep highest score)
        seen_cards: dict[Tuple[Card, ...], float] = {}
        unique: List[Tuple[float, PlaySuggestion]] = []
        for score, sug in scored:
            key = sug.cards
            if key not in seen_cards or score > seen_cards[key]:
                seen_cards[key] = score
                unique = [(s, sg) for s, sg in unique if sg.cards != key]
                unique.append((score, sug))

        # Sort by confidence descending
        unique.sort(key=lambda x: x[0], reverse=True)

        results = [sug for _, sug in unique[: self.top_n]]

        # For response mode, always include a PASS option if not already capped
        if not is_lead and len(results) < self.top_n:
            pass_conf = _pass_confidence(risk_report, len(hand))
            pass_risk = _pass_risk(risk_report)
            pass_reason = _generate_pass_reason(risk_report, len(hand))
            results.append(
                PlaySuggestion(
                    cards=(),
                    combo_type="pass",
                    reason=pass_reason,
                    risk=pass_risk,
                    confidence=round(pass_conf, 2),
                )
            )

        return results[: self.top_n]

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _lead_candidates(self, hand: List[Card]) -> List[Combo]:
        """Generate candidate plays when leading."""
        plays = enumerate_plays(hand, self.current_level)
        if not plays:
            # Fallback: play lowest single
            combo = classify_combo([hand[0]], self.current_level)
            if combo:
                return [combo]
            return []

        # Prefer non-bomb plays first, keep a few bombs for variety
        non_bombs = [p for p in plays if not p.is_bomb]
        bombs = [p for p in plays if p.is_bomb]

        # Sort non-bombs: prefer lower rank (save big cards)
        non_bombs.sort(key=lambda c: (c.combo_type.value, c.rank_key))

        # Diversify: pick candidates across different combo types
        result: List[Combo] = []
        seen_types: set[ComboType] = set()
        for combo in non_bombs:
            if combo.combo_type not in seen_types:
                result.append(combo)
                seen_types.add(combo.combo_type)
            elif len(result) < self.top_n * 2:
                result.append(combo)

        # Add weakest bomb if available (as an option)
        if bombs:
            bombs.sort(key=lambda c: (c.combo_type.value, c.rank_key))
            result.append(bombs[0])

        return result

    def _response_candidates(
        self, hand: List[Card], last_play: Combo
    ) -> List[Combo]:
        """Generate candidate plays when responding to last_play."""
        beaters = find_beating_plays(hand, last_play, self.current_level)
        if not beaters:
            return []

        non_bombs = [b for b in beaters if not b.is_bomb]
        bombs = [b for b in beaters if b.is_bomb]

        # Sort: prefer cheapest beaters first
        non_bombs.sort(key=lambda c: c.rank_key)
        bombs.sort(key=lambda c: (c.combo_type.value, c.rank_key))

        result: List[Combo] = []
        result.extend(non_bombs[: self.top_n])
        if bombs and len(result) < self.top_n * 2:
            result.append(bombs[0])

        return result


def get_suggestions(
    hand: List[Card],
    last_play: Optional[Combo] = None,
    current_level: Rank = Rank.TWO,
    counter: Optional[CardCounter] = None,
    top_n: int = 3,
) -> List[PlaySuggestion]:
    """Convenience function to get play suggestions.

    Args:
        hand: Current hand cards.
        last_play: Last combo played (None means lead).
        current_level: Current game level rank.
        counter: Optional CardCounter with game state.
        top_n: Number of suggestions to return.

    Returns:
        List of PlaySuggestion sorted by confidence descending.
    """
    engine = SuggestionEngine(
        current_level=current_level,
        counter=counter,
        top_n=top_n,
    )
    return engine.suggest(hand, last_play)
