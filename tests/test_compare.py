"""Tests for Atom 1.3 - compare module (can_beat, compare_combos)."""
import pytest
from guandan.models import Card, Rank, Suit, JokerType
from guandan.combos import Combo, ComboType, classify_combo
from guandan.compare import can_beat, compare_combos


def c(rank, suit=Suit.HEARTS):
    return Card(rank=rank, suit=suit)


def make_combo(combo_type, rank_key, cards_tuple):
    return Combo(combo_type=combo_type, cards=cards_tuple, rank_key=rank_key)


class TestCanBeat:
    def test_anything_beats_pass(self):
        play = make_combo(ComboType.SINGLE, 3, (c(Rank.THREE),))
        prev = make_combo(ComboType.PASS, 0, ())
        assert can_beat(play, prev)

    def test_bomb_beats_non_bomb(self):
        bomb = make_combo(ComboType.BOMB_4, 10, (c(Rank.TEN), c(Rank.TEN, Suit.DIAMONDS), c(Rank.TEN, Suit.CLUBS), c(Rank.TEN, Suit.SPADES)))
        single = make_combo(ComboType.SINGLE, 14, (c(Rank.ACE),))
        assert can_beat(bomb, single)

    def test_non_bomb_cannot_beat_bomb(self):
        single = make_combo(ComboType.SINGLE, 14, (c(Rank.ACE),))
        bomb = make_combo(ComboType.BOMB_4, 5, (c(Rank.FIVE), c(Rank.FIVE, Suit.DIAMONDS), c(Rank.FIVE, Suit.CLUBS), c(Rank.FIVE, Suit.SPADES)))
        assert not can_beat(single, bomb)

    def test_higher_bomb_type_wins(self):
        b5 = make_combo(ComboType.BOMB_5, 10, tuple(c(Rank.TEN, s) for s in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES, Suit.HEARTS]))
        b4 = make_combo(ComboType.BOMB_4, 14, (c(Rank.ACE), c(Rank.ACE, Suit.DIAMONDS), c(Rank.ACE, Suit.CLUBS), c(Rank.ACE, Suit.SPADES)))
        assert can_beat(b5, b4)

    def test_lower_bomb_type_loses(self):
        b4 = make_combo(ComboType.BOMB_4, 14, (c(Rank.ACE), c(Rank.ACE, Suit.DIAMONDS), c(Rank.ACE, Suit.CLUBS), c(Rank.ACE, Suit.SPADES)))
        b5 = make_combo(ComboType.BOMB_5, 3, tuple(c(Rank.THREE, s) for s in [Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES, Suit.HEARTS]))
        assert not can_beat(b4, b5)

    def test_same_bomb_type_higher_rank_wins(self):
        high = make_combo(ComboType.BOMB_4, 10, (c(Rank.TEN), c(Rank.TEN, Suit.DIAMONDS), c(Rank.TEN, Suit.CLUBS), c(Rank.TEN, Suit.SPADES)))
        low = make_combo(ComboType.BOMB_4, 5, (c(Rank.FIVE), c(Rank.FIVE, Suit.DIAMONDS), c(Rank.FIVE, Suit.CLUBS), c(Rank.FIVE, Suit.SPADES)))
        assert can_beat(high, low)

    def test_same_bomb_type_lower_rank_loses(self):
        low = make_combo(ComboType.BOMB_4, 5, (c(Rank.FIVE), c(Rank.FIVE, Suit.DIAMONDS), c(Rank.FIVE, Suit.CLUBS), c(Rank.FIVE, Suit.SPADES)))
        high = make_combo(ComboType.BOMB_4, 10, (c(Rank.TEN), c(Rank.TEN, Suit.DIAMONDS), c(Rank.TEN, Suit.CLUBS), c(Rank.TEN, Suit.SPADES)))
        assert not can_beat(low, high)

    def test_same_type_higher_rank(self):
        high = make_combo(ComboType.SINGLE, 10, (c(Rank.TEN),))
        low = make_combo(ComboType.SINGLE, 5, (c(Rank.FIVE),))
        assert can_beat(high, low)

    def test_same_type_lower_rank(self):
        low = make_combo(ComboType.SINGLE, 5, (c(Rank.FIVE),))
        high = make_combo(ComboType.SINGLE, 10, (c(Rank.TEN),))
        assert not can_beat(low, high)

    def test_different_non_bomb_types(self):
        single = make_combo(ComboType.SINGLE, 14, (c(Rank.ACE),))
        pair = make_combo(ComboType.PAIR, 3, (c(Rank.THREE), c(Rank.THREE, Suit.DIAMONDS)))
        assert not can_beat(single, pair)

    def test_same_type_different_size(self):
        big = make_combo(ComboType.STRAIGHT, 10, tuple(c(Rank(r)) for r in range(5, 11)))
        small = make_combo(ComboType.STRAIGHT, 9, tuple(c(Rank(r)) for r in range(5, 10)))
        assert not can_beat(big, small)

    def test_joker_bomb_beats_bomb_8(self):
        jb = make_combo(ComboType.JOKER_BOMB, 99, (Card(joker=JokerType.RED), Card(joker=JokerType.BLACK), Card(joker=JokerType.RED), Card(joker=JokerType.BLACK)))
        b8 = make_combo(ComboType.BOMB_8, 14, tuple(c(Rank.ACE, s) for s in [Suit.HEARTS]*8))
        assert can_beat(jb, b8)


class TestCompareCombos:
    def test_a_beats_b(self):
        a = make_combo(ComboType.SINGLE, 10, (c(Rank.TEN),))
        b = make_combo(ComboType.SINGLE, 5, (c(Rank.FIVE),))
        assert compare_combos(a, b) == 1

    def test_b_beats_a(self):
        a = make_combo(ComboType.SINGLE, 5, (c(Rank.FIVE),))
        b = make_combo(ComboType.SINGLE, 10, (c(Rank.TEN),))
        assert compare_combos(a, b) == -1

    def test_incomparable(self):
        a = make_combo(ComboType.SINGLE, 10, (c(Rank.TEN),))
        b = make_combo(ComboType.PAIR, 5, (c(Rank.FIVE), c(Rank.FIVE, Suit.DIAMONDS)))
        assert compare_combos(a, b) == 0

    def test_equal_rank(self):
        a = make_combo(ComboType.SINGLE, 10, (c(Rank.TEN),))
        b = make_combo(ComboType.SINGLE, 10, (c(Rank.TEN, Suit.DIAMONDS),))
        result = compare_combos(a, b)
        assert result == 0
