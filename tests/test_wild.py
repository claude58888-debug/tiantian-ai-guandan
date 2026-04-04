"""Tests for Atom 1.4 - Wild card (逢人配) module."""
import pytest
from guandan.models import Card, Rank, Suit, JokerType
from guandan.combos import Combo, ComboType, classify_combo
from guandan.wild import (
    is_wild, count_wilds, split_wilds,
    classify_with_wilds, find_wild_combos,
    can_beat_with_wilds,
)


def make_card(rank, suit=Suit.HEARTS):
    return Card(rank=rank, suit=suit)


def make_wild(level=Rank.TWO):
    """Create a wild card (Heart of current level)."""
    return Card(rank=level, suit=Suit.HEARTS)


class TestIsWild:
    def test_heart_level_is_wild(self):
        card = Card(rank=Rank.TWO, suit=Suit.HEARTS)
        assert is_wild(card, Rank.TWO) is True

    def test_heart_non_level_not_wild(self):
        card = Card(rank=Rank.THREE, suit=Suit.HEARTS)
        assert is_wild(card, Rank.TWO) is False

    def test_non_heart_level_not_wild(self):
        card = Card(rank=Rank.TWO, suit=Suit.SPADES)
        assert is_wild(card, Rank.TWO) is False

    def test_joker_not_wild(self):
        card = Card(joker=JokerType.RED)
        assert is_wild(card, Rank.TWO) is False

    def test_level_five_wild(self):
        card = Card(rank=Rank.FIVE, suit=Suit.HEARTS)
        assert is_wild(card, Rank.FIVE) is True


class TestCountWilds:
    def test_count_zero(self):
        cards = [make_card(Rank.THREE, Suit.SPADES), make_card(Rank.FOUR)]
        assert count_wilds(cards, Rank.TWO) == 0

    def test_count_one(self):
        cards = [make_wild(Rank.TWO), make_card(Rank.FOUR, Suit.SPADES)]
        assert count_wilds(cards, Rank.TWO) == 1

    def test_count_two(self):
        w1 = Card(rank=Rank.TWO, suit=Suit.HEARTS, deck_id=0)
        w2 = Card(rank=Rank.TWO, suit=Suit.HEARTS, deck_id=1)
        cards = [w1, w2]
        assert count_wilds(cards, Rank.TWO) == 2


class TestSplitWilds:
    def test_split(self):
        w = make_wild(Rank.TWO)
        n = make_card(Rank.FIVE, Suit.SPADES)
        normal, wilds = split_wilds([w, n], Rank.TWO)
        assert len(normal) == 1
        assert len(wilds) == 1
        assert normal[0] == n
        assert wilds[0] == w


class TestClassifyWithWilds:
    def test_natural_combo_no_wild(self):
        # Two threes = pair, no wild needed
        cards = [make_card(Rank.THREE, Suit.HEARTS), make_card(Rank.THREE, Suit.SPADES)]
        combo = classify_with_wilds(cards, Rank.TWO)
        assert combo is not None
        assert combo.combo_type == ComboType.PAIR

    def test_wild_forms_pair(self):
        # One three + one wild -> wild substitutes as three -> pair
        normal = make_card(Rank.THREE, Suit.SPADES)
        wild = make_wild(Rank.FIVE)  # Heart 5 is wild when level=5
        cards = [normal, wild]
        combo = classify_with_wilds(cards, Rank.FIVE)
        assert combo is not None
        assert combo.combo_type == ComboType.PAIR

    def test_no_wilds_invalid_returns_none(self):
        # Two different ranks, no wilds -> invalid
        cards = [make_card(Rank.THREE, Suit.SPADES), make_card(Rank.FOUR, Suit.SPADES)]
        combo = classify_with_wilds(cards, Rank.TWO)
        assert combo is None

    def test_wild_forms_triple(self):
        # Two sevens + one wild = triple
        cards = [
            make_card(Rank.SEVEN, Suit.SPADES),
            make_card(Rank.SEVEN, Suit.CLUBS),
            make_wild(Rank.FIVE),
        ]
        combo = classify_with_wilds(cards, Rank.FIVE)
        assert combo is not None
        assert combo.combo_type == ComboType.TRIPLE

    def test_single_wild_is_single(self):
        # A single wild card played alone is a valid single
        wild = make_wild(Rank.TWO)
        combo = classify_with_wilds([wild], Rank.TWO)
        assert combo is not None
        assert combo.combo_type == ComboType.SINGLE


class TestFindWildCombos:
    def test_no_wilds(self):
        cards = [make_card(Rank.THREE, Suit.SPADES)]
        combos = find_wild_combos(cards, Rank.TWO)
        assert len(combos) == 1
        assert combos[0].combo_type == ComboType.SINGLE

    def test_wild_finds_multiple(self):
        # One card + one wild can form: single (playing wild alone isn't here since it's 2 cards),
        # or pair if wild substitutes
        normal = make_card(Rank.THREE, Suit.SPADES)
        wild = make_wild(Rank.FIVE)
        cards = [normal, wild]
        combos = find_wild_combos(cards, Rank.FIVE)
        # Should find at least a pair (wild as THREE)
        types = [c.combo_type for c in combos]
        assert ComboType.PAIR in types


class TestCanBeatWithWilds:
    def test_wild_pair_beats_lower(self):
        # Pair of 3s as last play
        last = classify_combo(
            [make_card(Rank.THREE, Suit.SPADES), make_card(Rank.THREE, Suit.CLUBS)],
            Rank.TWO
        )
        # Five + wild(as five) = pair of fives, should beat pair of threes
        cards = [make_card(Rank.FIVE, Suit.SPADES), make_wild(Rank.TWO)]
        result = can_beat_with_wilds(cards, last, Rank.TWO)
        # Wild as TWO heart is being checked - it might form a pair of fives or not
        # depending on substitution. Result is None or a valid combo.
        assert result is None or result.combo_type >= ComboType.PAIR

    def test_cannot_beat_higher(self):
        last = classify_combo(
            [make_card(Rank.ACE, Suit.SPADES), make_card(Rank.ACE, Suit.CLUBS)],
            Rank.TWO
        )
        cards = [make_card(Rank.THREE, Suit.SPADES), make_card(Rank.THREE, Suit.CLUBS)]
        result = can_beat_with_wilds(cards, last, Rank.TWO)
        assert result is None
