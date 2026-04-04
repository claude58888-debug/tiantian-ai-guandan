"""Tests for Atom 1.2/1.3 - Combo recognition and comparison."""
import pytest
from guandan.models import Card, Rank, Suit, JokerType
from guandan.combos import classify_combo, ComboType, is_valid_play
from guandan.compare import can_beat


def c(rank, suit, d=0):
    return Card(rank=rank, suit=suit, deck_id=d)

def j(jtype, d=0):
    return Card(joker=jtype, deck_id=d)


class TestComboClassify:
    def test_single(self):
        combo = classify_combo([c(Rank.ACE, Suit.SPADES)])
        assert combo.combo_type == ComboType.SINGLE

    def test_pair(self):
        combo = classify_combo([c(Rank.FIVE, Suit.HEARTS), c(Rank.FIVE, Suit.CLUBS)])
        assert combo.combo_type == ComboType.PAIR

    def test_triple(self):
        cards = [c(Rank.SEVEN, Suit.HEARTS), c(Rank.SEVEN, Suit.CLUBS), c(Rank.SEVEN, Suit.DIAMONDS)]
        combo = classify_combo(cards)
        assert combo.combo_type == ComboType.TRIPLE

    def test_bomb_4(self):
        cards = [c(Rank.KING, s) for s in Suit]
        combo = classify_combo(cards)
        assert combo.combo_type == ComboType.BOMB_4

    def test_full_house(self):
        cards = [c(Rank.THREE, Suit.HEARTS), c(Rank.THREE, Suit.CLUBS), c(Rank.THREE, Suit.DIAMONDS),
                 c(Rank.FIVE, Suit.HEARTS), c(Rank.FIVE, Suit.CLUBS)]
        combo = classify_combo(cards)
        assert combo.combo_type == ComboType.FULL_HOUSE
        assert combo.rank_key == 3

    def test_straight(self):
        cards = [c(Rank.THREE, Suit.HEARTS), c(Rank.FOUR, Suit.CLUBS), c(Rank.FIVE, Suit.DIAMONDS),
                 c(Rank.SIX, Suit.SPADES), c(Rank.SEVEN, Suit.HEARTS)]
        combo = classify_combo(cards)
        assert combo.combo_type == ComboType.STRAIGHT

    def test_consecutive_pairs(self):
        cards = [c(Rank.THREE, Suit.HEARTS), c(Rank.THREE, Suit.CLUBS),
                 c(Rank.FOUR, Suit.HEARTS), c(Rank.FOUR, Suit.CLUBS),
                 c(Rank.FIVE, Suit.HEARTS), c(Rank.FIVE, Suit.CLUBS)]
        combo = classify_combo(cards)
        assert combo.combo_type == ComboType.CONSECUTIVE_PAIRS

    def test_plate(self):
        cards = [c(Rank.THREE, Suit.HEARTS), c(Rank.THREE, Suit.CLUBS), c(Rank.THREE, Suit.DIAMONDS),
                 c(Rank.FOUR, Suit.HEARTS), c(Rank.FOUR, Suit.CLUBS), c(Rank.FOUR, Suit.DIAMONDS)]
        combo = classify_combo(cards)
        assert combo.combo_type == ComboType.PLATE

    def test_joker_bomb(self):
        cards = [j(JokerType.BLACK, 0), j(JokerType.RED, 0), j(JokerType.BLACK, 1), j(JokerType.RED, 1)]
        combo = classify_combo(cards)
        assert combo.combo_type == ComboType.JOKER_BOMB

    def test_invalid(self):
        cards = [c(Rank.THREE, Suit.HEARTS), c(Rank.FIVE, Suit.CLUBS)]
        assert classify_combo(cards) is None

    def test_pass(self):
        combo = classify_combo([])
        assert combo.combo_type == ComboType.PASS


class TestCanBeat:
    def test_higher_single(self):
        a = classify_combo([c(Rank.KING, Suit.HEARTS)])
        b = classify_combo([c(Rank.QUEEN, Suit.HEARTS)])
        assert can_beat(a, b)
        assert not can_beat(b, a)

    def test_bomb_beats_single(self):
        bomb = classify_combo([c(Rank.THREE, s) for s in Suit])
        single = classify_combo([c(Rank.ACE, Suit.SPADES)])
        assert can_beat(bomb, single)

    def test_higher_bomb(self):
        b4 = classify_combo([c(Rank.THREE, s) for s in Suit])
        b5 = classify_combo([c(Rank.FIVE, Suit.HEARTS, 0), c(Rank.FIVE, Suit.CLUBS, 0),
                              c(Rank.FIVE, Suit.DIAMONDS, 0), c(Rank.FIVE, Suit.SPADES, 0),
                              c(Rank.FIVE, Suit.HEARTS, 1)])
        assert can_beat(b5, b4)

    def test_same_type_different_size_no_beat(self):
        s5 = classify_combo([c(Rank(r), Suit.HEARTS) for r in range(3, 8)])
        s6 = classify_combo([c(Rank(r), Suit.CLUBS) for r in range(3, 9)])
        assert not can_beat(s6, s5)  # different size straight

    def test_anything_beats_pass(self):
        p = classify_combo([])
        s = classify_combo([c(Rank.TWO, Suit.HEARTS)])
        assert can_beat(s, p)
