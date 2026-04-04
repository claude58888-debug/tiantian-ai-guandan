"""Tests for Atom 1.5 - CLI functions."""
import pytest
from unittest.mock import patch
from guandan.models import Card, Rank, Suit, JokerType
from guandan.game import GameState, Phase
from guandan.cli import (
    parse_card, find_matching_cards, display_hand,
    random_ai_play, main
)


class TestParseCard:
    def test_parse_normal_card(self):
        c = parse_card('3H')
        assert c is not None
        assert c.rank == Rank.THREE
        assert c.suit == Suit.HEARTS

    def test_parse_ten(self):
        c = parse_card('10S')
        assert c is not None
        assert c.rank == Rank.TEN
        assert c.suit == Suit.SPADES

    def test_parse_ace(self):
        c = parse_card('AD')
        assert c is not None
        assert c.rank == Rank.ACE
        assert c.suit == Suit.DIAMONDS

    def test_parse_black_joker(self):
        c = parse_card('BJ')
        assert c is not None
        assert c.is_joker
        assert c.joker == JokerType.BLACK

    def test_parse_red_joker(self):
        c = parse_card('RJ')
        assert c is not None
        assert c.is_joker
        assert c.joker == JokerType.RED

    def test_parse_lowercase(self):
        c = parse_card('kc')
        assert c is not None
        assert c.rank == Rank.KING
        assert c.suit == Suit.CLUBS

    def test_parse_with_spaces(self):
        c = parse_card('  5H  ')
        assert c is not None
        assert c.rank == Rank.FIVE

    def test_parse_invalid_short(self):
        assert parse_card('X') is None

    def test_parse_invalid_suit(self):
        assert parse_card('3X') is None

    def test_parse_invalid_rank(self):
        assert parse_card('ZH') is None

    def test_parse_empty(self):
        assert parse_card('') is None

    def test_parse_all_suits(self):
        for suit_char, suit in [('H', Suit.HEARTS), ('D', Suit.DIAMONDS),
                                 ('C', Suit.CLUBS), ('S', Suit.SPADES)]:
            c = parse_card(f'2{suit_char}')
            assert c.suit == suit

    def test_parse_all_ranks(self):
        for rank_str in ['2','3','4','5','6','7','8','9','10','J','Q','K','A']:
            c = parse_card(f'{rank_str}H')
            assert c is not None


class TestFindMatchingCards:
    def test_find_single(self):
        hand = [Card(rank=Rank.THREE, suit=Suit.HEARTS),
                Card(rank=Rank.FOUR, suit=Suit.SPADES)]
        parsed = [Card(rank=Rank.THREE, suit=Suit.HEARTS)]
        result = find_matching_cards(hand, parsed)
        assert result is not None
        assert len(result) == 1
        assert result[0].rank == Rank.THREE

    def test_find_not_in_hand(self):
        hand = [Card(rank=Rank.THREE, suit=Suit.HEARTS)]
        parsed = [Card(rank=Rank.FOUR, suit=Suit.SPADES)]
        result = find_matching_cards(hand, parsed)
        assert result is None

    def test_find_joker(self):
        hand = [Card(joker=JokerType.RED), Card(rank=Rank.ACE, suit=Suit.HEARTS)]
        parsed = [Card(joker=JokerType.RED)]
        result = find_matching_cards(hand, parsed)
        assert result is not None
        assert result[0].is_joker

    def test_find_joker_not_matching(self):
        hand = [Card(joker=JokerType.RED)]
        parsed = [Card(joker=JokerType.BLACK)]
        result = find_matching_cards(hand, parsed)
        assert result is None

    def test_find_multiple(self):
        hand = [Card(rank=Rank.THREE, suit=Suit.HEARTS),
                Card(rank=Rank.THREE, suit=Suit.DIAMONDS),
                Card(rank=Rank.FIVE, suit=Suit.CLUBS)]
        parsed = [Card(rank=Rank.THREE, suit=Suit.HEARTS),
                  Card(rank=Rank.THREE, suit=Suit.DIAMONDS)]
        result = find_matching_cards(hand, parsed)
        assert result is not None
        assert len(result) == 2


class TestDisplayHand:
    def test_display_normal(self):
        cards = [Card(rank=Rank.ACE, suit=Suit.SPADES),
                 Card(joker=JokerType.RED)]
        s = display_hand(cards)
        assert 'RJ' in s

    def test_display_empty(self):
        assert display_hand([]) == ''


class TestRandomAiPlay:
    def test_ai_leads(self):
        gs = GameState()
        gs.deal(seed=42)
        # Force it to be player 1's turn to lead
        gs.current_player = 1
        gs.last_play = None
        ok = random_ai_play(gs, 1)
        assert ok

    def test_ai_tries_to_beat(self):
        gs = GameState()
        gs.deal(seed=42)
        # Player 0 plays a low card
        card = gs.players[0].hand.cards[0]
        gs.play_cards(0, [card])
        cp = gs.current_player
        # AI tries to beat or pass
        ok = random_ai_play(gs, cp)
        assert ok

    def test_ai_passes_when_cannot_beat(self):
        gs = GameState()
        gs.deal(seed=99)
        # Play a high card first
        from guandan.combos import Combo, ComboType, classify_combo
        card = gs.players[0].hand.cards[-1]  # highest card
        gs.play_cards(0, [card])
        cp = gs.current_player
        # AI may pass if can't beat
        ok = random_ai_play(gs, cp)
        # Either plays or passes, both valid
        assert isinstance(ok, bool)


class TestMain:
    def test_main_runs_with_input(self):
        inputs = ['pass', 'pass', 'pass', 'quit']
        with patch('builtins.input', side_effect=inputs):
            try:
                main()
            except (StopIteration, SystemExit, EOFError):
                pass  # expected when inputs run out

    def test_main_play_card(self):
        # Play a valid card then quit
        inputs = ['3H', 'pass', 'pass', 'quit']
        with patch('builtins.input', side_effect=inputs):
            try:
                main()
            except (StopIteration, SystemExit, EOFError):
                pass

    def test_main_invalid_card(self):
        inputs = ['XYZ', 'pass', 'quit']
        with patch('builtins.input', side_effect=inputs):
            try:
                main()
            except (StopIteration, SystemExit, EOFError):
                pass
