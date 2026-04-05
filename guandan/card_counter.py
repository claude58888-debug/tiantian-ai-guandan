"""Card counting and risk assessment module (Atom 4.1).

Tracks played cards across a Guandan round to provide:
- Counts of key cards remaining (Jokers, Aces, level cards)
- Bomb probability estimation
- Risk level assessment (low/medium/high)
- Opponent threat analysis

Designed per PRD: real-time stats on played Jokers, Aces, level cards,
bomb depletion likelihood, opponent big-card threats, and risk level.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple

from guandan.models import Card, Rank, Suit, JokerType


# Total cards in a 2-deck Guandan game
TOTAL_CARDS = 108
CARDS_PER_RANK_PER_SUIT = 2  # two decks
TOTAL_JOKERS = 4  # 2 red + 2 black
TOTAL_PER_RANK = 8  # 4 suits * 2 decks


class RiskLevel(Enum):
    """Current game risk assessment."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


@dataclass
class KeyCardStatus:
    """Status of a key card type."""
    name: str
    total: int
    played: int = 0
    in_hand: int = 0

    @property
    def remaining(self) -> int:
        """Cards remaining in opponents' hands."""
        return max(0, self.total - self.played - self.in_hand)

    @property
    def depletion_ratio(self) -> float:
        """Fraction of this card type already played/held (0.0-1.0)."""
        if self.total == 0:
            return 1.0
        return (self.played + self.in_hand) / self.total


@dataclass
class RiskReport:
    """Risk assessment report for the current game state."""
    risk_level: RiskLevel = RiskLevel.LOW
    bomb_threat: float = 0.0  # 0.0-1.0 probability opponents have bombs
    big_card_threat: float = 0.0  # opponent likely has big cards
    key_cards: Dict[str, KeyCardStatus] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    summary: str = ""


class CardCounter:
    """Tracks played cards and assesses risk in a Guandan round.

    Usage:
        counter = CardCounter(current_level=Rank.TWO)
        counter.set_hand([card1, card2, ...])
        counter.record_play([card3, card4])  # cards played on table
        report = counter.assess_risk()
    """

    def __init__(self, current_level: Rank = Rank.TWO) -> None:
        self.current_level = current_level
        self._hand: List[Card] = []
        self._played: List[Card] = []
        self._played_counter: Counter = Counter()
        self._hand_counter: Counter = Counter()
        self._round_history: List[List[Card]] = []

    def set_hand(self, cards: List[Card]) -> None:
        """Set current hand cards."""
        self._hand = list(cards)
        self._hand_counter = Counter(self._card_key(c) for c in cards)

    def record_play(self, cards: List[Card]) -> None:
        """Record cards played on the table."""
        self._played.extend(cards)
        for c in cards:
            self._played_counter[self._card_key(c)] += 1
        self._round_history.append(list(cards))

    def reset(self) -> None:
        """Reset for a new round."""
        self._hand.clear()
        self._played.clear()
        self._played_counter.clear()
        self._hand_counter.clear()
        self._round_history.clear()

    @property
    def total_played(self) -> int:
        return len(self._played)

    @property
    def play_history(self) -> List[List[Card]]:
        return list(self._round_history)

    def _card_key(self, card: Card) -> str:
        """Create a counting key for a card."""
        if card.joker:
            return f"JOKER_{card.joker.name}"
        return f"{card.rank.name}" if card.rank else "UNKNOWN"

    def get_key_card_statuses(self) -> Dict[str, KeyCardStatus]:
        """Get status of all key card types."""
        statuses: Dict[str, KeyCardStatus] = {}

        # Jokers (4 total: 2 red, 2 black)
        red_played = self._played_counter.get("JOKER_RED", 0)
        red_hand = self._hand_counter.get("JOKER_RED", 0)
        black_played = self._played_counter.get("JOKER_BLACK", 0)
        black_hand = self._hand_counter.get("JOKER_BLACK", 0)
        statuses["joker"] = KeyCardStatus(
            name="Jokers",
            total=TOTAL_JOKERS,
            played=red_played + black_played,
            in_hand=red_hand + black_hand,
        )

        # Aces
        ace_played = self._played_counter.get("ACE", 0)
        ace_hand = self._hand_counter.get("ACE", 0)
        statuses["ace"] = KeyCardStatus(
            name="Aces", total=TOTAL_PER_RANK,
            played=ace_played, in_hand=ace_hand,
        )

        # Level cards (current level rank)
        level_key = self.current_level.name
        level_played = self._played_counter.get(level_key, 0)
        level_hand = self._hand_counter.get(level_key, 0)
        statuses["level"] = KeyCardStatus(
            name=f"Level ({self.current_level.name})",
            total=TOTAL_PER_RANK,
            played=level_played, in_hand=level_hand,
        )

        # Kings
        king_played = self._played_counter.get("KING", 0)
        king_hand = self._hand_counter.get("KING", 0)
        statuses["king"] = KeyCardStatus(
            name="Kings", total=TOTAL_PER_RANK,
            played=king_played, in_hand=king_hand,
        )

        return statuses

    def estimate_bomb_probability(self) -> float:
        """Estimate probability that opponents hold a bomb.

        Heuristic based on:
        - How many cards of each rank have been seen
        - More unseen ranks = higher bomb chance
        """
        unseen_quads = 0
        for rank in Rank:
            key = rank.name
            seen = self._played_counter.get(key, 0) + self._hand_counter.get(key, 0)
            available = TOTAL_PER_RANK - seen
            # If 4+ cards of a rank are unseen, opponents could have a bomb
            if available >= 4:
                unseen_quads += 1

        # Also check joker bomb (need all 4 jokers)
        joker_seen = (
            self._played_counter.get("JOKER_RED", 0)
            + self._hand_counter.get("JOKER_RED", 0)
            + self._played_counter.get("JOKER_BLACK", 0)
            + self._hand_counter.get("JOKER_BLACK", 0)
        )
        joker_bomb_possible = joker_seen == 0  # all 4 unseen

        # Normalize: 13 ranks possible, scale to 0-1
        rank_factor = min(unseen_quads / 13.0, 1.0)
        joker_factor = 0.15 if joker_bomb_possible else 0.0

        return min(rank_factor * 0.85 + joker_factor, 1.0)

    def estimate_big_card_threat(self) -> float:
        """Estimate threat level from opponents' big cards.

        Considers remaining Aces, Kings, Jokers not in our hand.
        """
        statuses = self.get_key_card_statuses()
        total_big = 0
        remaining_big = 0
        for key in ["joker", "ace", "king"]:
            s = statuses[key]
            total_big += s.total
            remaining_big += s.remaining

        if total_big == 0:
            return 0.0
        return remaining_big / total_big

    def assess_risk(self) -> RiskReport:
        """Generate a comprehensive risk assessment."""
        key_cards = self.get_key_card_statuses()
        bomb_prob = self.estimate_bomb_probability()
        big_threat = self.estimate_big_card_threat()

        warnings: List[str] = []

        # Determine risk level
        if bomb_prob > 0.7 and big_threat > 0.6:
            risk = RiskLevel.CRITICAL
        elif bomb_prob > 0.5 or big_threat > 0.5:
            risk = RiskLevel.HIGH
        elif bomb_prob > 0.3 or big_threat > 0.3:
            risk = RiskLevel.MEDIUM
        else:
            risk = RiskLevel.LOW

        # Generate warnings
        joker_status = key_cards["joker"]
        if joker_status.remaining >= 3:
            warnings.append("Joker bomb still possible (3+ unseen)")
        elif joker_status.remaining >= 2:
            warnings.append(f"{joker_status.remaining} Jokers unaccounted for")

        level_status = key_cards["level"]
        if level_status.remaining >= 4:
            warnings.append(
                f"Level card bomb possible ({level_status.remaining} unseen)"
            )

        ace_status = key_cards["ace"]
        if ace_status.remaining >= 4:
            warnings.append(f"Opponents may hold Ace bomb ({ace_status.remaining} unseen)")

        if bomb_prob > 0.6:
            warnings.append(f"High bomb probability: {bomb_prob:.0%}")

        # Summary
        parts = [
            f"Risk: {risk.name}",
            f"Bomb threat: {bomb_prob:.0%}",
            f"Big card threat: {big_threat:.0%}",
            f"Cards played: {self.total_played}",
        ]
        summary = " | ".join(parts)

        return RiskReport(
            risk_level=risk,
            bomb_threat=bomb_prob,
            big_card_threat=big_threat,
            key_cards=key_cards,
            warnings=warnings,
            summary=summary,
        )

    def format_display(self) -> str:
        """Format counter state for UI display."""
        report = self.assess_risk()
        lines = [
            f"=== Card Counter ===",
            f"Risk Level: {report.risk_level.name}",
            f"Bomb Threat: {report.bomb_threat:.0%}",
            f"Big Card Threat: {report.big_card_threat:.0%}",
            "",
            "Key Cards:",
        ]
        for key, status in report.key_cards.items():
            lines.append(
                f"  {status.name}: {status.played} played, "
                f"{status.in_hand} in hand, {status.remaining} unseen"
            )

        if report.warnings:
            lines.append("")
            lines.append("Warnings:")
            for w in report.warnings:
                lines.append(f"  ! {w}")

        return "\n".join(lines)
