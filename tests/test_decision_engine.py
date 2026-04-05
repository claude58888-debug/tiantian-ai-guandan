"""Tests for guandan.decision_engine (M7)."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, List, Optional

import pytest

from guandan.card_counter import CardCounter
from guandan.card_recognition import CardRecognizer, RecognizedCard
from guandan.combos import Combo, ComboType, classify_combo
from guandan.decision_engine import (
    Decision,
    DecisionEngine,
    NOT_MY_TURN,
    _elapsed_ms,
)
from guandan.game_screen_analyzer import GameScreenAnalyzer, ScreenRegions
from guandan.models import Card, Rank, Suit, JokerType
from guandan.suggestion import PlaySuggestion, SuggestionRisk

from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_screenshot(w: int = 1400, h: int = 850) -> Image.Image:
    return Image.new('RGB', (w, h), 'black')


def _card(rank: Rank, suit: Suit = Suit.HEARTS) -> Card:
    return Card(rank=rank, suit=suit)


def _hand_5() -> List[Card]:
    return [
        _card(Rank.THREE, Suit.HEARTS),
        _card(Rank.FOUR, Suit.HEARTS),
        _card(Rank.FIVE, Suit.HEARTS),
        _card(Rank.SIX, Suit.HEARTS),
        _card(Rank.SEVEN, Suit.HEARTS),
    ]


# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------

class TestDecision:
    def test_empty_is_pass(self) -> None:
        d = Decision()
        assert d.is_pass is True
        assert d.combo_type == 'pass'

    def test_with_cards_not_pass(self) -> None:
        c = _card(Rank.ACE)
        d = Decision(cards_to_play=(c,), combo_type='single', confidence=0.9)
        assert d.is_pass is False

    def test_display_pass(self) -> None:
        d = Decision()
        assert 'PASS' in d.display()

    def test_display_with_cards(self) -> None:
        c = _card(Rank.KING)
        d = Decision(
            cards_to_play=(c,),
            combo_type='single',
            confidence=0.85,
            reasoning='Strong card',
            latency_ms=42.0,
        )
        text = d.display()
        assert 'single' in text
        assert '0.85' in text
        assert '42' in text
        assert 'Strong card' in text

    def test_frozen(self) -> None:
        d = Decision()
        with pytest.raises(AttributeError):
            d.confidence = 0.5  # type: ignore[misc]

    def test_alternatives_default_empty(self) -> None:
        d = Decision()
        assert d.alternatives == ()


# ---------------------------------------------------------------------------
# NOT_MY_TURN sentinel
# ---------------------------------------------------------------------------

class TestNotMyTurn:
    def test_is_pass(self) -> None:
        assert NOT_MY_TURN.is_pass is True

    def test_combo_type(self) -> None:
        assert NOT_MY_TURN.combo_type == 'wait'

    def test_confidence(self) -> None:
        assert NOT_MY_TURN.confidence == 1.0


# ---------------------------------------------------------------------------
# _elapsed_ms helper
# ---------------------------------------------------------------------------

class TestElapsedMs:
    def test_positive_elapsed(self) -> None:
        t0 = time.monotonic()
        time.sleep(0.01)
        ms = _elapsed_ms(t0)
        assert ms > 0

    def test_immediate(self) -> None:
        t0 = time.monotonic()
        ms = _elapsed_ms(t0)
        assert ms >= 0


# ---------------------------------------------------------------------------
# DecisionEngine construction
# ---------------------------------------------------------------------------

class TestEngineConstruction:
    def test_default(self) -> None:
        engine = DecisionEngine()
        assert engine.current_level == Rank.TWO
        assert engine.analyzer is not None
        assert engine.counter is not None

    def test_custom_level(self) -> None:
        engine = DecisionEngine(current_level=Rank.ACE)
        assert engine.current_level == Rank.ACE

    def test_custom_recognizer(self) -> None:
        rec = MagicMock(spec=CardRecognizer)
        engine = DecisionEngine(recognizer=rec)
        assert engine.analyzer.recognizer is rec

    def test_set_level(self) -> None:
        engine = DecisionEngine(current_level=Rank.TWO)
        engine.set_level(Rank.KING)
        assert engine.current_level == Rank.KING


# ---------------------------------------------------------------------------
# decide() — not my turn
# ---------------------------------------------------------------------------

class TestDecideNotMyTurn:
    def test_returns_not_my_turn(self) -> None:
        engine = DecisionEngine()
        with patch.object(engine.analyzer, 'detect_my_turn', return_value=False):
            result = engine.decide(_make_screenshot())
            assert result is NOT_MY_TURN


# ---------------------------------------------------------------------------
# decide() — my turn, no hand
# ---------------------------------------------------------------------------

class TestDecideNoHand:
    def test_no_hand_detected(self) -> None:
        engine = DecisionEngine()
        with patch.object(engine.analyzer, 'detect_my_turn', return_value=True):
            with patch.object(engine.analyzer, 'detect_hand_cards', return_value=[]):
                result = engine.decide(_make_screenshot())
                assert result.is_pass
                assert 'Could not detect' in result.reasoning


# ---------------------------------------------------------------------------
# decide() — my turn, with hand (lead)
# ---------------------------------------------------------------------------

class TestDecideLead:
    def test_lead_produces_decision(self) -> None:
        hand = _hand_5()
        engine = DecisionEngine()
        with patch.object(engine.analyzer, 'detect_my_turn', return_value=True):
            with patch.object(engine.analyzer, 'detect_hand_cards', return_value=hand):
                with patch.object(engine.analyzer, 'detect_played_cards', return_value=None):
                    result = engine.decide(_make_screenshot())
                    assert result.is_pass is False or result.reasoning != ''
                    assert result.latency_ms >= 0

    def test_lead_has_combo_type(self) -> None:
        hand = [_card(Rank.THREE), _card(Rank.THREE, Suit.DIAMONDS)]
        engine = DecisionEngine()
        with patch.object(engine.analyzer, 'detect_my_turn', return_value=True):
            with patch.object(engine.analyzer, 'detect_hand_cards', return_value=hand):
                with patch.object(engine.analyzer, 'detect_played_cards', return_value=None):
                    result = engine.decide(_make_screenshot())
                    assert result.combo_type != ''


# ---------------------------------------------------------------------------
# decide() — response mode
# ---------------------------------------------------------------------------

class TestDecideResponse:
    def test_response_to_played_cards(self) -> None:
        hand = _hand_5()
        played = [_card(Rank.THREE, Suit.CLUBS)]
        engine = DecisionEngine()

        # Simulate first frame with played cards to set _last_combo
        with patch.object(engine.analyzer, 'detect_my_turn', return_value=True):
            with patch.object(engine.analyzer, 'detect_hand_cards', return_value=hand):
                with patch.object(engine.analyzer, 'detect_played_cards', return_value=played):
                    result = engine.decide(_make_screenshot())
                    assert result.latency_ms >= 0


# ---------------------------------------------------------------------------
# decide_for_tribute
# ---------------------------------------------------------------------------

class TestDecideForTribute:
    def test_tribute_returns_lowest(self) -> None:
        hand = [
            _card(Rank.THREE, Suit.HEARTS),
            _card(Rank.ACE, Suit.SPADES),
        ]
        engine = DecisionEngine()
        with patch.object(engine.analyzer, 'detect_hand_cards', return_value=hand):
            result = engine.decide_for_tribute(_make_screenshot())
            assert len(result.cards_to_play) == 1
            assert result.cards_to_play[0].rank == Rank.THREE
            assert result.combo_type == 'tribute'

    def test_tribute_no_hand(self) -> None:
        engine = DecisionEngine()
        with patch.object(engine.analyzer, 'detect_hand_cards', return_value=[]):
            result = engine.decide_for_tribute(_make_screenshot())
            assert result.is_pass
            assert 'Could not detect' in result.reasoning

    def test_tribute_with_only_jokers(self) -> None:
        hand = [
            Card(joker=JokerType.BLACK),
            Card(joker=JokerType.RED),
        ]
        engine = DecisionEngine()
        with patch.object(engine.analyzer, 'detect_hand_cards', return_value=hand):
            result = engine.decide_for_tribute(_make_screenshot())
            assert len(result.cards_to_play) == 1


# ---------------------------------------------------------------------------
# Tracking state
# ---------------------------------------------------------------------------

class TestTracking:
    def test_reset_clears_state(self) -> None:
        engine = DecisionEngine()
        engine._last_hand = [_card(Rank.ACE)]
        engine._last_played = [_card(Rank.KING)]
        engine.reset()
        assert engine._last_hand == []
        assert engine._last_played is None
        assert engine._last_combo is None

    def test_update_tracking_records_play(self) -> None:
        engine = DecisionEngine()
        played = [_card(Rank.FIVE)]
        engine._update_tracking(_hand_5(), played)
        assert engine._last_played == played

    def test_update_tracking_clears_on_empty(self) -> None:
        engine = DecisionEngine()
        engine._last_played = [_card(Rank.KING)]
        engine._last_combo = classify_combo([_card(Rank.KING)], Rank.TWO)
        engine._update_tracking(_hand_5(), None)
        assert engine._last_combo is None

    def test_update_tracking_same_played_no_duplicate_record(self) -> None:
        engine = DecisionEngine()
        played = [_card(Rank.FIVE)]
        engine._update_tracking(_hand_5(), played)
        initial_count = engine.counter.total_played
        engine._update_tracking(_hand_5(), played)
        assert engine.counter.total_played == initial_count


# ---------------------------------------------------------------------------
# Integration: full decide pipeline
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_decide_pipeline_latency(self) -> None:
        hand = _hand_5()
        engine = DecisionEngine()
        with patch.object(engine.analyzer, 'detect_my_turn', return_value=True):
            with patch.object(engine.analyzer, 'detect_hand_cards', return_value=hand):
                with patch.object(engine.analyzer, 'detect_played_cards', return_value=None):
                    result = engine.decide(_make_screenshot())
                    assert result.latency_ms < 5000  # should be fast with mocks

    def test_no_valid_plays_message(self) -> None:
        hand = [_card(Rank.THREE)]
        engine = DecisionEngine()
        # Patch suggestion engine to return empty
        with patch.object(engine.analyzer, 'detect_my_turn', return_value=True):
            with patch.object(engine.analyzer, 'detect_hand_cards', return_value=hand):
                with patch.object(engine.analyzer, 'detect_played_cards', return_value=None):
                    with patch.object(engine._suggestion, 'suggest', return_value=[]):
                        result = engine.decide(_make_screenshot())
                        assert 'No valid plays' in result.reasoning
