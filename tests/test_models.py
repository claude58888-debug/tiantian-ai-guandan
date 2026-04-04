"""Tests for Atom 1.1 - Card & Deck data models."""
import pytest
from guandan.models import Card, Deck, Hand, Rank, Suit, JokerType, make_standard_deck


class TestCard:
    def test_normal_card(self):
        c = Card(rank=Rank.ACE, suit=Suit.SPADES)
        assert c.rank == Rank.ACE
        assert c.suit == Suit.SPADES
        assert not c.is_joker

    def test_joker_card(self):
        c = Card(joker=JokerType.RED)
        assert c.is_joker
        assert c.is_red_joker
        assert not c.is_black_joker

    def test_invalid_joker_with_rank(self):
        with pytest.raises(ValueError):
            Card(rank=Rank.ACE, joker=JokerType.RED)

    def test_invalid_normal_missing_suit(self):
        with pytest.raises(ValueError):
            Card(rank=Rank.ACE)

    def test_display_normal(self):
        c = Card(rank=Rank.TEN, suit=Suit.HEARTS)
        assert '10' in c.display()

    def test_display_joker(self):
        assert Card(joker=JokerType.RED).display() == 'RJ'
        assert Card(joker=JokerType.BLACK).display() == 'BJ'

    def test_rank_value(self):
        c = Card(rank=Rank.THREE, suit=Suit.CLUBS)
        assert c.rank_value() == 3

    def test_joker_rank_value(self):
        bj = Card(joker=JokerType.BLACK)
        rj = Card(joker=JokerType.RED)
        assert rj.rank_value() > bj.rank_value()
        assert bj.rank_value() > 14  # higher than Ace

    def test_frozen(self):
        c = Card(rank=Rank.ACE, suit=Suit.SPADES)
        with pytest.raises(AttributeError):
            c.rank = Rank.TWO


class TestDeck:
    def test_standard_deck_size(self):
        cards = make_standard_deck(0)
        assert len(cards) == 54

    def test_double_deck_size(self):
        d = Deck()
        assert len(d) == 108

    def test_deal_four_players(self):
        d = Deck()
        d.shuffle(seed=0)
        hands = d.deal(4)
        assert len(hands) == 4
        assert all(len(h) == 27 for h in hands)

    def test_deal_wrong_players(self):
        d = Deck()
        with pytest.raises(ValueError):
            d.deal(3)

    def test_shuffle_deterministic(self):
        d1 = Deck()
        d1.shuffle(seed=42)
        d2 = Deck()
        d2.shuffle(seed=42)
        assert d1.cards == d2.cards


class TestHand:
    def test_add_remove(self):
        h = Hand()
        c = Card(rank=Rank.ACE, suit=Suit.SPADES)
        h.add(c)
        assert len(h) == 1
        h.remove(c)
        assert len(h) == 0

    def test_sort(self):
        h = Hand([Card(rank=Rank.KING, suit=Suit.HEARTS),
                  Card(rank=Rank.THREE, suit=Suit.CLUBS)])
        h.sort()
        assert h.cards[0].rank == Rank.THREE

    def test_count_by_rank(self):
        h = Hand([Card(rank=Rank.FIVE, suit=Suit.HEARTS),
                  Card(rank=Rank.FIVE, suit=Suit.CLUBS)])
        groups = h.count_by_rank()
        assert len(groups[5]) == 2
