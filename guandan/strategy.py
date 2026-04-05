"""AI strategy module for Guandan (Atom 3.1).

Provides pluggable strategy classes for AI decision-making.
Strategies range from simple (random, greedy) to advanced.

V0.3 fix (H-2): SmartStrategy now accepts an optional PartnerContext
so that it adjusts play style when its teammate is winning/leading.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import List, Optional, Tuple

from guandan.models import Card, Rank
from guandan.combos import Combo, ComboType, classify_combo
from guandan.compare import can_beat


def find_all_singles(cards: List[Card], level: Rank) -> List[Combo]:
    """Find all valid single card plays."""
    result = []
    for c in cards:
        combo = classify_combo([c], level)
        if combo and combo.combo_type == ComboType.SINGLE:
            result.append(combo)
    return result


def find_all_pairs(cards: List[Card], level: Rank) -> List[Combo]:
    """Find all valid pair plays."""
    result = []
    seen = set()
    for i, c1 in enumerate(cards):
        for j, c2 in enumerate(cards):
            if j <= i:
                continue
            key = (min(id(c1), id(c2)), max(id(c1), id(c2)))
            if key in seen:
                continue
            combo = classify_combo([c1, c2], level)
            if combo and combo.combo_type == ComboType.PAIR:
                result.append(combo)
                seen.add(key)
    return result


def find_all_triples(cards: List[Card], level: Rank) -> List[Combo]:
    """Find all valid triple plays."""
    result = []
    for combo_cards in combinations(cards, 3):
        combo = classify_combo(list(combo_cards), level)
        if combo and combo.combo_type == ComboType.TRIPLE:
            result.append(combo)
    return result


def find_all_bombs(cards: List[Card], level: Rank) -> List[Combo]:
    """Find all bomb plays (4+ of a kind, joker bomb)."""
    result = []
    # Check 4-8 of a kind
    for size in range(4, min(len(cards) + 1, 9)):
        for combo_cards in combinations(cards, size):
            combo = classify_combo(list(combo_cards), level)
            if combo and combo.is_bomb:
                result.append(combo)
    return result


def enumerate_plays(cards: List[Card], level: Rank) -> List[Combo]:
    """Enumerate all possible plays from a hand."""
    plays = []
    plays.extend(find_all_singles(cards, level))
    plays.extend(find_all_pairs(cards, level))
    plays.extend(find_all_triples(cards, level))
    # Full houses: try all 3+2 combos
    for combo_cards in combinations(cards, 5):
        combo = classify_combo(list(combo_cards), level)
        if combo and combo.combo_type == ComboType.FULL_HOUSE:
            plays.append(combo)
    plays.extend(find_all_bombs(cards, level))
    return plays


def find_beating_plays(cards: List[Card], last_play: Combo, level: Rank) -> List[Combo]:
    """Find all plays that can beat the last play."""
    candidates = enumerate_plays(cards, level)
    return [c for c in candidates if can_beat(c, last_play)]


class Strategy(ABC):
    """Abstract base class for AI strategies."""

    @abstractmethod
    def choose_lead(self, cards: List[Card], level: Rank) -> Optional[List[Card]]:
        """Choose cards to lead (no previous play to beat)."""

    @abstractmethod
    def choose_response(
        self, cards: List[Card], last_play: Combo, level: Rank
    ) -> Optional[List[Card]]:
        """Choose cards to beat last_play, or None to pass."""

    def play(self, cards: List[Card], last_play: Optional[Combo], level: Rank) -> Optional[List[Card]]:
        """Main entry: decide what to play."""
        if last_play is None or last_play.combo_type == ComboType.PASS:
            return self.choose_lead(cards, level)
        return self.choose_response(cards, last_play, level)


class RandomStrategy(Strategy):
    """Pure random: pick any valid play."""

    def choose_lead(self, cards: List[Card], level: Rank) -> Optional[List[Card]]:
        if not cards:
            return None
        plays = enumerate_plays(cards, level)
        if not plays:
            return [cards[0]]
        choice = random.choice(plays)
        return list(choice.cards)

    def choose_response(
        self, cards: List[Card], last_play: Combo, level: Rank
    ) -> Optional[List[Card]]:
        beaters = find_beating_plays(cards, last_play, level)
        if not beaters:
            return None  # pass
        choice = random.choice(beaters)
        return list(choice.cards)


class GreedyStrategy(Strategy):
    """Greedy: play the smallest valid combo. Save bombs."""

    def choose_lead(self, cards: List[Card], level: Rank) -> Optional[List[Card]]:
        if not cards:
            return None
        # Prefer leading singles (lowest first), then pairs, then triples
        singles = find_all_singles(cards, level)
        if singles:
            singles.sort(key=lambda c: c.rank_key)
            return list(singles[0].cards)
        pairs = find_all_pairs(cards, level)
        if pairs:
            pairs.sort(key=lambda c: c.rank_key)
            return list(pairs[0].cards)
        triples = find_all_triples(cards, level)
        if triples:
            triples.sort(key=lambda c: c.rank_key)
            return list(triples[0].cards)
        # Fallback: play lowest single
        return [cards[0]]

    def choose_response(
        self, cards: List[Card], last_play: Combo, level: Rank
    ) -> Optional[List[Card]]:
        beaters = find_beating_plays(cards, last_play, level)
        if not beaters:
            return None
        # Play weakest beater; avoid bombs unless necessary
        non_bombs = [b for b in beaters if not b.is_bomb]
        if non_bombs:
            non_bombs.sort(key=lambda c: c.rank_key)
            return list(non_bombs[0].cards)
        # Only bombs left: play weakest bomb
        beaters.sort(key=lambda c: (c.combo_type.value, c.rank_key))
        return list(beaters[0].cards)


@dataclass(frozen=True)
class PartnerContext:
    """Information about the teammate's state.

    Attributes:
        partner_card_count: Number of cards the partner still holds.
            Use -1 when the count is unknown.
        partner_finished: True if the partner has already finished
            (played all their cards).
        partner_finish_order: 1-based finish position of the partner
            (0 if not finished yet).
    """
    partner_card_count: int = -1
    partner_finished: bool = False
    partner_finish_order: int = 0


class SmartStrategy(Strategy):
    """Smarter strategy: considers hand structure and teammate.

    When a PartnerContext is supplied, the strategy adjusts:
    - Partner about to finish (<=3 cards): play conservatively so
      the partner can win the trick and go out first.
    - Partner already finished: play aggressively to follow up and
      secure a good finish-order result for the team.
    - Partner far from finishing: play normally.
    """

    def __init__(
        self,
        aggression: float = 0.5,
        partner: Optional[PartnerContext] = None,
    ):
        self.aggression = max(0.0, min(1.0, aggression))
        self.partner = partner

    def set_partner(self, partner: PartnerContext) -> None:
        """Update partner context (e.g. after each trick)."""
        self.partner = partner

    # ------------------------------------------------------------------
    # Effective aggression accounting for partner state
    # ------------------------------------------------------------------
    def _effective_aggression(self) -> float:
        """Compute an aggression modifier based on partner state."""
        if self.partner is None:
            return self.aggression

        # Partner already finished -> play aggressively to follow quickly
        if self.partner.partner_finished:
            return min(1.0, self.aggression + 0.3)

        # Partner about to finish (few cards) -> play conservatively
        if 0 <= self.partner.partner_card_count <= 3:
            return max(0.0, self.aggression - 0.3)

        return self.aggression

    def _hand_strength(self, cards: List[Card], level: Rank) -> float:
        """Estimate hand strength 0-1."""
        if not cards:
            return 0.0
        bombs = find_all_bombs(cards, level)
        n_bombs = len(set(id(b) for b in bombs))
        card_count = len(cards)
        # Fewer cards + more bombs = stronger
        return min(1.0, n_bombs * 0.3 + max(0, (27 - card_count) / 27) * 0.7)

    def choose_lead(self, cards: List[Card], level: Rank) -> Optional[List[Card]]:
        if not cards:
            return None
        strength = self._hand_strength(cards, level)
        eff_aggro = self._effective_aggression()

        # When partner is about to finish, lead with smallest single
        # to avoid winning tricks the partner could use to go out.
        if (self.partner is not None
                and not self.partner.partner_finished
                and 0 <= self.partner.partner_card_count <= 3):
            singles = find_all_singles(cards, level)
            if singles:
                singles.sort(key=lambda c: c.rank_key)
                return list(singles[0].cards)
            return [cards[0]]

        # Strong hand or aggressive: lead with pairs/triples
        if strength > 0.6 or eff_aggro > 0.7:
            pairs = find_all_pairs(cards, level)
            if pairs:
                pairs.sort(key=lambda c: c.rank_key)
                return list(pairs[0].cards)
        # Default: lead smallest single
        singles = find_all_singles(cards, level)
        if singles:
            singles.sort(key=lambda c: c.rank_key)
            return list(singles[0].cards)
        return [cards[0]]

    def choose_response(
        self, cards: List[Card], last_play: Combo, level: Rank
    ) -> Optional[List[Card]]:
        beaters = find_beating_plays(cards, last_play, level)
        if not beaters:
            return None
        strength = self._hand_strength(cards, level)
        eff_aggro = self._effective_aggression()
        non_bombs = [b for b in beaters if not b.is_bomb]

        # When partner is about to finish, prefer to pass so
        # the partner can win the trick and clear their last cards.
        if (self.partner is not None
                and not self.partner.partner_finished
                and 0 <= self.partner.partner_card_count <= 3):
            return None  # pass to let partner finish

        # If hand is strong or high aggression, always respond
        if non_bombs:
            non_bombs.sort(key=lambda c: c.rank_key)
            return list(non_bombs[0].cards)
        # Use bombs only if hand is strong enough or few cards left
        if len(cards) <= 4 or strength > 0.5 or eff_aggro > 0.8:
            beaters.sort(key=lambda c: (c.combo_type.value, c.rank_key))
            return list(beaters[0].cards)
        return None  # pass, save bombs


# Strategy registry
STRATEGIES = {
    'random': RandomStrategy,
    'greedy': GreedyStrategy,
    'smart': SmartStrategy,
}


def get_strategy(name: str = 'greedy', **kwargs) -> Strategy:
    """Get a strategy by name."""
    cls = STRATEGIES.get(name)
    if cls is None:
        raise ValueError(f'Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}')
    return cls(**kwargs)
