"""Tests for Atom 2.3 - Game state parser module."""
import pytest
from guandan.models import Card, Rank, Suit
from guandan.combos import ComboType
from guandan.card_detector import DetectedCard
from guandan.game_state_parser import (
    TurnPhase, VisualGameState, GameStateParser,
    create_parser,
)


def make_card(rank, suit=Suit.HEARTS):
    return Card(rank=rank, suit=suit)


def make_detection(rank, suit=Suit.HEARTS, conf=0.95):
    card = make_card(rank, suit)
    return DetectedCard(card=card, confidence=conf)


class TestTurnPhase:
    def test_enum_values(self):
        assert TurnPhase.UNKNOWN is not None
        assert TurnPhase.MY_TURN is not None
        assert TurnPhase.WAITING is not None
        assert TurnPhase.GAME_OVER is not None
        assert TurnPhase.BETWEEN_ROUNDS is not None


class TestVisualGameState:
    def test_default_state(self):
        state = VisualGameState()
        assert state.my_hand == []
        assert state.last_played is None
        assert state.current_level == Rank.TWO
        assert state.turn_phase == TurnPhase.UNKNOWN
        assert state.confidence == 0.0

    def test_is_my_turn(self):
        state = VisualGameState(turn_phase=TurnPhase.MY_TURN)
        assert state.is_my_turn is True

    def test_not_my_turn(self):
        state = VisualGameState(turn_phase=TurnPhase.WAITING)
        assert state.is_my_turn is False

    def test_hand_size(self):
        cards = [make_card(Rank.THREE), make_card(Rank.FIVE)]
        state = VisualGameState(my_hand=cards)
        assert state.hand_size == 2

    def test_repr(self):
        state = VisualGameState(current_level=Rank.FIVE)
        r = repr(state)
        assert 'FIVE' in r
        assert 'UNKNOWN' in r


class TestGameStateParser:
    def test_no_detector(self):
        parser = GameStateParser()
        assert parser.parse_hand_image(None) == []
        assert parser.parse_table_image(None) == []

    def test_detections_to_cards(self):
        parser = GameStateParser()
        dets = [make_detection(Rank.THREE), make_detection(Rank.FIVE)]
        cards = parser.detections_to_cards(dets)
        assert len(cards) == 2
        assert cards[0].rank == Rank.THREE

    def test_detections_to_combo_empty(self):
        parser = GameStateParser()
        assert parser.detections_to_combo([]) is None

    def test_detections_to_combo_single(self):
        parser = GameStateParser()
        dets = [make_detection(Rank.ACE)]
        combo = parser.detections_to_combo(dets)
        assert combo is not None
        assert combo.combo_type == ComboType.SINGLE

    def test_detections_to_combo_pair(self):
        parser = GameStateParser()
        dets = [
            make_detection(Rank.FIVE, Suit.HEARTS),
            make_detection(Rank.FIVE, Suit.SPADES),
        ]
        combo = parser.detections_to_combo(dets)
        assert combo is not None
        assert combo.combo_type == ComboType.PAIR

    def test_detect_turn_phase_default(self):
        parser = GameStateParser()
        assert parser.detect_turn_phase(None) == TurnPhase.UNKNOWN

    def test_parse_live_no_capturer(self):
        parser = GameStateParser()
        assert parser.parse_live() is None

    def test_last_state_initially_none(self):
        parser = GameStateParser()
        assert parser.last_state is None

    def test_current_level(self):
        parser = GameStateParser(current_level=Rank.SEVEN)
        assert parser.current_level == Rank.SEVEN


class TestCreateParser:
    def test_create_default(self):
        parser = create_parser()
        assert isinstance(parser, GameStateParser)
        assert parser.current_level == Rank.TWO

    def test_create_with_level(self):
        parser = create_parser(current_level=Rank.FIVE)
        assert parser.current_level == Rank.FIVE
