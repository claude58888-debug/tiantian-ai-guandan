"""Guandan card game - Card & Deck data model (Atom 1.1).

Guandan uses two standard 54-card decks (108 cards total).
Ranks: 2,3,4,5,6,7,8,9,10,J,Q,K,A + Red Joker, Black Joker
Suits: Hearts, Diamonds, Clubs, Spades
The current-level rank cards are wild cards.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum, unique
from typing import List, Optional, Sequence


@unique
class Suit(IntEnum):
    """Card suits ordered by Guandan convention."""
    DIAMONDS = 0
    CLUBS = 1
    HEARTS = 2
    SPADES = 3

    def symbol(self) -> str:
        return {0: '\u2666', 1: '\u2663', 2: '\u2665', 3: '\u2660'}[self.value]

    def __repr__(self) -> str:
        return f'Suit.{self.name}'


@unique
class Rank(IntEnum):
    """Card ranks. Numeric value used for base ordering (2 lowest, Ace highest).
    Jokers are special and always compare above Ace."""
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    def label(self) -> str:
        special = {11: 'J', 12: 'Q', 13: 'K', 14: 'A'}
        return special.get(self.value, str(self.value))

    def __repr__(self) -> str:
        return f'Rank.{self.name}'


@unique
class JokerType(IntEnum):
    BLACK = 0
    RED = 1


@dataclass(frozen=True, order=True)
class Card:
    """Immutable playing card.

    For normal cards: rank and suit are set, joker is None.
    For jokers: joker is set, rank and suit are None.
    deck_id distinguishes the two copies (0 or 1).
    """
    rank: Optional[Rank] = None
    suit: Optional[Suit] = None
    joker: Optional[JokerType] = None
    deck_id: int = 0

    def __post_init__(self):
        if self.joker is not None:
            if self.rank is not None or self.suit is not None:
                raise ValueError('Joker cards must not have rank or suit.')
        else:
            if self.rank is None or self.suit is None:
                raise ValueError('Normal cards must have both rank and suit.')

    @property
    def is_joker(self) -> bool:
        return self.joker is not None

    @property
    def is_red_joker(self) -> bool:
        return self.joker == JokerType.RED

    @property
    def is_black_joker(self) -> bool:
        return self.joker == JokerType.BLACK

    def display(self) -> str:
        if self.is_joker:
            return 'RJ' if self.is_red_joker else 'BJ'
        return f'{self.rank.label()}{self.suit.symbol()}'

    def rank_value(self, current_level: Rank = Rank.TWO) -> int:
        """Effective rank value considering current level for wild cards.

        In Guandan, the natural ordering is:
        2 < 3 < 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < BJ < RJ
        The current-level rank cards serve as wild and are highest normal rank.
        """
        if self.is_joker:
            return 100 + self.joker.value  # BJ=100, RJ=101
        return self.rank.value

    def __repr__(self) -> str:
        return f'Card({self.display()}, d{self.deck_id})'


def make_standard_deck(deck_id: int = 0) -> List[Card]:
    """Create one standard 54-card deck."""
    cards: List[Card] = []
    for suit in Suit:
        for rank in Rank:
            cards.append(Card(rank=rank, suit=suit, deck_id=deck_id))
    cards.append(Card(joker=JokerType.BLACK, deck_id=deck_id))
    cards.append(Card(joker=JokerType.RED, deck_id=deck_id))
    return cards


@dataclass
class Deck:
    """Double deck for Guandan (108 cards)."""
    cards: List[Card] = field(default_factory=list)

    def __post_init__(self):
        if not self.cards:
            self.cards = make_standard_deck(0) + make_standard_deck(1)

    def shuffle(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.cards)

    def deal(self, num_players: int = 4) -> List[List[Card]]:
        """Deal cards to players. In Guandan, 4 players get 27 cards each."""
        if num_players != 4:
            raise ValueError('Guandan requires exactly 4 players.')
        hands: List[List[Card]] = [[] for _ in range(num_players)]
        for i, card in enumerate(self.cards):
            hands[i % num_players].append(card)
        return hands

    def __len__(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        return f'Deck({len(self.cards)} cards)'


@dataclass
class Hand:
    """A player's hand of cards."""
    cards: List[Card] = field(default_factory=list)

    def add(self, card: Card) -> None:
        self.cards.append(card)

    def remove(self, card: Card) -> None:
        self.cards.remove(card)

    def remove_cards(self, cards: Sequence[Card]) -> None:
        for c in cards:
            self.remove(c)

    def sort(self, current_level: Rank = Rank.TWO) -> None:
        self.cards.sort(key=lambda c: (c.rank_value(current_level), c.suit if c.suit else -1))

    def count_by_rank(self, current_level: Rank = Rank.TWO) -> dict:
        """Group cards by effective rank value."""
        groups: dict = {}
        for c in self.cards:
            rv = c.rank_value(current_level)
            groups.setdefault(rv, []).append(c)
        return groups

    def __len__(self) -> int:
        return len(self.cards)

    def __repr__(self) -> str:
        return f'Hand([{", ".join(c.display() for c in self.cards)}])'
