"""Guandan combo recognition engine (Atom 1.2).

Recognizes all valid card combinations in Guandan:
- Single: 1 card
- Pair: 2 same-rank cards
- Triple: 3 same-rank cards
- Full House (San Dai Er): 3+2
- Bomb: 4 same-rank cards
- Straight: 5+ consecutive singles
- Consecutive Pairs (Lian Dui): 3+ consecutive pairs
- Plate (Ban): 2+ consecutive triples
- Rocket: both jokers (Red+Black from same or different decks)
- Bomb of 5/6/7/8: 5-8 same-rank cards (using wild or double deck)
- Joker Bomb: 4 jokers
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import List, Optional, Sequence, Tuple

from guandan.models import Card, Rank, Suit, JokerType


@unique
class ComboType(IntEnum):
    """Combo types ordered by base power."""
    PASS = 0
    SINGLE = 1
    PAIR = 2
    TRIPLE = 3
    FULL_HOUSE = 4      # 3+2
    STRAIGHT = 5        # 5+ consecutive
    CONSECUTIVE_PAIRS = 6  # 3+ consecutive pairs
    PLATE = 7           # 2+ consecutive triples
    BOMB_4 = 10         # 4 of a kind
    BOMB_5 = 11
    BOMB_6 = 12
    BOMB_7 = 13
    BOMB_8 = 14
    JOKER_BOMB = 20     # 4 jokers


@dataclass(frozen=True)
class Combo:
    """A recognized card combination."""
    combo_type: ComboType
    cards: Tuple[Card, ...]
    rank_key: int = 0   # primary rank for comparison

    @property
    def size(self) -> int:
        return len(self.cards)

    @property
    def is_bomb(self) -> bool:
        return self.combo_type >= ComboType.BOMB_4

    def __repr__(self) -> str:
        names = {ct: ct.name for ct in ComboType}
        cards_str = ', '.join(c.display() for c in self.cards)
        return f'Combo({names[self.combo_type]}, [{cards_str}], key={self.rank_key})'


def _rank_values(cards: Sequence[Card], level: Rank = Rank.TWO) -> List[int]:
    return sorted(c.rank_value(level) for c in cards)


def _rank_counter(cards: Sequence[Card], level: Rank = Rank.TWO) -> Counter:
    return Counter(c.rank_value(level) for c in cards)


def classify_combo(cards: Sequence[Card], level: Rank = Rank.TWO) -> Optional[Combo]:
    """Classify a set of cards into a Combo, or None if invalid."""
    n = len(cards)
    if n == 0:
        return Combo(ComboType.PASS, tuple(), 0)

    t_cards = tuple(cards)
    rc = _rank_counter(cards, level)
    rv = _rank_values(cards, level)
    counts = sorted(rc.values(), reverse=True)
    distinct = len(rc)
    joker_count = sum(1 for c in cards if c.is_joker)

    # Joker bomb: 4 jokers
    if joker_count == 4 and n == 4:
        return Combo(ComboType.JOKER_BOMB, t_cards, 200)

    # Single
    if n == 1:
        return Combo(ComboType.SINGLE, t_cards, rv[0])

    # Pair
    if n == 2 and distinct == 1 and not any(c.is_joker for c in cards):
        return Combo(ComboType.PAIR, t_cards, rv[0])

    # Rocket (pair of jokers, one red one black)
    if n == 2 and joker_count == 2:
        jtypes = {c.joker for c in cards}
        if len(jtypes) == 2:  # one RED, one BLACK
            return Combo(ComboType.BOMB_4, t_cards, 150)  # rocket beats normal bombs

    # Triple
    if n == 3 and distinct == 1:
        return Combo(ComboType.TRIPLE, t_cards, rv[0])

    # Bombs (4-8 of a kind)
    if distinct == 1 and n >= 4 and n <= 8:
        bomb_map = {4: ComboType.BOMB_4, 5: ComboType.BOMB_5,
                    6: ComboType.BOMB_6, 7: ComboType.BOMB_7,
                    8: ComboType.BOMB_8}
        return Combo(bomb_map[n], t_cards, rv[0])

    # Full house: 3+2
    if n == 5 and sorted(counts) == [2, 3]:
        triple_rank = [r for r, c in rc.items() if c == 3][0]
        return Combo(ComboType.FULL_HOUSE, t_cards, triple_rank)

    # Straight: 5+ consecutive singles
    if n >= 5 and distinct == n and all(c not in rv for c in [100, 101]):
        if rv[-1] - rv[0] == n - 1 and all(cnt == 1 for cnt in counts):
            return Combo(ComboType.STRAIGHT, t_cards, rv[-1])

    # Consecutive pairs: 3+ consecutive pairs (6+ cards)
    if n >= 6 and n % 2 == 0 and all(c == 2 for c in counts):
        ranks_sorted = sorted(rc.keys())
        if (len(ranks_sorted) >= 3 and
            ranks_sorted[-1] - ranks_sorted[0] == len(ranks_sorted) - 1 and
            all(r < 100 for r in ranks_sorted)):
            return Combo(ComboType.CONSECUTIVE_PAIRS, t_cards, ranks_sorted[-1])

    # Plate: 2+ consecutive triples (6+ cards)
    if n >= 6 and n % 3 == 0 and all(c == 3 for c in counts):
        ranks_sorted = sorted(rc.keys())
        if (len(ranks_sorted) >= 2 and
            ranks_sorted[-1] - ranks_sorted[0] == len(ranks_sorted) - 1 and
            all(r < 100 for r in ranks_sorted)):
            return Combo(ComboType.PLATE, t_cards, ranks_sorted[-1])

    return None  # Invalid combination


def is_valid_play(cards: Sequence[Card], level: Rank = Rank.TWO) -> bool:
    """Check if a set of cards forms a valid play."""
    return classify_combo(cards, level) is not None
